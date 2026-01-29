#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::{
    helpers::{ParamsTrait, build_attn_matrix, build_eta_matrix},
    layer_norm::{layer_norm_forward_save_intermediates, layer_norm_l2_grad_save_intermediates},
};
use crate::ttt::cubecl_kernels::FusedTttConfig;

#[derive(CubeType, CubeLaunch)]
pub struct Inputs<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub xv: Tensor<Line<F>>,
    pub weight: Tensor<Line<F>>,
    pub bias: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Outputs<F: Float> {
    pub output: Tensor<Line<F>>,
    pub weight_out: Tensor<Line<F>>,
    pub bias_out: Tensor<Line<F>>,
}

/// Forward intermediates saved for backward pass.
/// These are computed during forward and passed to backward to avoid recomputation.
#[derive(CubeType, CubeLaunch)]
pub struct ForwardIntermediates<F: Float> {
    /// x_hat from fused LN (normalized Z1) [B, NH, CS, F]
    pub x_hat_fused: Tensor<Line<F>>,
    /// std from fused LN [B, NH, CS]
    pub std_fused: Tensor<Line<F>>,
    /// grad_output from fused LN (y - target) [B, NH, CS, F]
    pub grad_output_fused: Tensor<Line<F>>,
    /// grad_x_hat from fused LN (grad_output * ln_weight) [B, NH, CS, F]
    pub grad_x_hat_fused: Tensor<Line<F>>,
    /// grad_l_wrt_Z1 from fused LN [B, NH, CS, F]
    pub grad_l_wrt_Z1: Tensor<Line<F>>,
    /// x_hat from output LN (normalized Z1_bar) [B, NH, CS, F]
    pub x_hat_ln: Tensor<Line<F>>,
    /// std from output LN [B, NH, CS]
    pub std_ln: Tensor<Line<F>>,
}
/// Process one mini-batch stage of the TTT-Linear forward pass.
///
/// This is the inner loop body.
/// Weight and bias are kept in shared memory / registers and updated in place.
///
/// # Arguments
/// * `inputs` - Input tensors (xq, xk, xv indexed by stage_offset)
/// * `outputs` - Output tensor (indexed by stage_offset)
/// * `fwd_intermediates` - Output tensors for forward intermediates (for backward pass)
/// * `weight_smem` - Weight matrix in shared memory [F, F], updated in place
/// * `bias_rv` - Bias vector in registers [F], updated in place
/// * `ln_weight_rv` - Layer norm weight [F]
/// * `ln_bias_rv` - Layer norm bias [F]
/// * `stage_offset` - Offset into qkv/output for this mini-batch (in elements)
/// * `ttt_lr_eta_idx` - Base offset for ttt_lr_eta
/// * `epsilon` - Layer norm epsilon
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_forward_stage<P: ParamsTrait>(
    inputs: &Inputs<P::E>,
    outputs: &mut Outputs<P::E>,
    fwd_intermediates: &mut ForwardIntermediates<P::E>,
    weight_smem: &mut St<P::E, P::F, P::F>,
    bias_rv: &mut Rv<P::E, P::F>,
    ln_weight_rv: &Rv<P::E, P::F>,
    ln_bias_rv: &Rv<P::E, P::F>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    #[comptime] epsilon: f32,
) {
    // Scratch tiles - allocated per stage call
    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();
    let mut k_direct_smem = P::cs_f_tile();
    let mut v_direct_smem = P::cs_f_tile();
    let mut z1_smem = P::cs_f_tile();
    let mut temp_cs_f_smem = P::cs_f_tile();
    let mut eta_matrix_smem = P::cs_cs_tile();
    let mut attn_smem = P::cs_cs_tile();
    let mut reduce_buf = ReduceBuf::<P::E>::new();

    // Intermediate tiles for layer norm - to be saved for backward pass
    let mut x_hat_fused_smem = P::cs_f_tile();
    let mut std_fused_rv = P::cs_reg_big();
    let mut grad_output_fused_smem = P::cs_f_tile();
    let mut grad_x_hat_fused_smem = P::cs_f_tile();
    let mut x_hat_ln_smem = P::cs_f_tile();
    let mut std_ln_rv = P::cs_reg_big();

    // Load QKV for this stage
    cube::load_st_transpose(&inputs.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&inputs.xk, &mut k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&inputs.xk, &mut k_direct_smem, stage_offset, 0, 0);
    cube::load_st_direct(&inputs.xv, &mut v_direct_smem, stage_offset, 0, 0);

    sync_cube();

    // Step 1: z1 = xk @ W + b
    let mut z1_reg = P::cs_f_reg();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, &k_smem, weight_smem);

    sync_cube();

    // Add bias (need to broadcast from full bias_rv to the thread's portion)
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::f_reg();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = bias_rv.data[src_idx];
    }
    z1_reg.add_row(&bias_reg);

    cube::store_rt_to_st(&z1_reg, &mut z1_smem);

    sync_cube();

    // Step 2: reconstruction_target = xv - xk
    v_direct_smem.sub(&k_direct_smem);

    sync_cube();

    // Step 3: grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
    // This also saves intermediates (x_hat_fused, std_fused, grad_output_fused, grad_x_hat_fused)
    layer_norm_l2_grad_save_intermediates::<P::E, P::CS, P::F>(
        &mut z1_smem,
        &v_direct_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut temp_cs_f_smem,
        &mut reduce_buf,
        // Output intermediates
        &mut x_hat_fused_smem,
        &mut std_fused_rv,
        &mut grad_output_fused_smem,
        &mut grad_x_hat_fused_smem,
        epsilon,
    );

    sync_cube();

    // Store fused layer norm intermediates
    cube::store_st_direct(
        &x_hat_fused_smem,
        &mut fwd_intermediates.x_hat_fused,
        stage_offset,
        0,
        0,
    );
    cube::store_st_direct(
        &grad_output_fused_smem,
        &mut fwd_intermediates.grad_output_fused,
        stage_offset,
        0,
        0,
    );
    cube::store_st_direct(
        &grad_x_hat_fused_smem,
        &mut fwd_intermediates.grad_x_hat_fused,
        stage_offset,
        0,
        0,
    );
    cube::broadcast::store_rv_direct(
        &std_fused_rv,
        &mut fwd_intermediates.std_fused,
        ttt_lr_eta_idx,
    );
    // z1_smem now contains grad_l_wrt_Z1
    cube::store_st_direct(
        &z1_smem,
        &mut fwd_intermediates.grad_l_wrt_Z1,
        stage_offset,
        0,
        0,
    );

    sync_cube();

    // Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
    build_eta_matrix::<P>(
        &inputs.token_eta,
        &inputs.ttt_lr_eta,
        &mut eta_matrix_smem,
        ttt_lr_eta_idx,
        false,
    );

    // Step 5: attn_scores = xq @ xk^T, tril
    build_attn_matrix::<P>(&q_smem, &k_smem, &mut attn_smem, false);

    // Step 6: Rebuild eta for z1_bar computation (consumed by tril)
    build_eta_matrix::<P>(
        &inputs.token_eta,
        &inputs.ttt_lr_eta,
        &mut eta_matrix_smem,
        ttt_lr_eta_idx,
        false,
    );

    // eta @ grad
    let mut eta_grad_reg = P::cs_f_reg();
    eta_grad_reg.zero();
    cube::mma_AB(&mut eta_grad_reg, &eta_matrix_smem, &z1_smem);

    sync_cube();

    // eta * attn
    eta_matrix_smem.mul(&attn_smem);

    sync_cube();

    // (eta * attn) @ grad
    let mut eta_attn_grad_reg = P::cs_f_reg();
    eta_attn_grad_reg.zero();
    cube::mma_AB(&mut eta_attn_grad_reg, &eta_matrix_smem, &z1_smem);

    sync_cube();

    // z1_bar = xq @ W
    let mut z1_bar_reg = P::cs_f_reg();
    z1_bar_reg.zero();
    cube::mma_AtB(&mut z1_bar_reg, &q_smem, weight_smem);

    sync_cube();

    // z1_bar -= (eta * attn) @ grad
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // z1_bar += bias
    z1_bar_reg.add_row(&bias_reg);

    // z1_bar -= eta @ grad
    z1_bar_reg.sub(&eta_grad_reg);

    // Store z1_bar to shared memory for layer norm
    cube::store_rt_to_st(&z1_bar_reg, &mut temp_cs_f_smem);

    sync_cube();

    // Step 8: layer_norm + add xq
    // This also saves intermediates (x_hat_ln, std_ln)
    layer_norm_forward_save_intermediates::<P::E, P::CS, P::F>(
        &mut temp_cs_f_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut reduce_buf,
        &mut x_hat_ln_smem,
        &mut std_ln_rv,
        epsilon,
    );

    sync_cube();

    // Store output layer norm intermediates
    cube::store_st_direct(
        &x_hat_ln_smem,
        &mut fwd_intermediates.x_hat_ln,
        stage_offset,
        0,
        0,
    );
    cube::broadcast::store_rv_direct(&std_ln_rv, &mut fwd_intermediates.std_ln, ttt_lr_eta_idx);

    sync_cube();

    // Load xq into k_direct_smem
    cube::load_st_direct(&inputs.xq, &mut k_direct_smem, stage_offset, 0, 0);

    sync_cube();

    // Add: output = xq + layer_norm(z1_bar)
    temp_cs_f_smem.add(&k_direct_smem);

    sync_cube();

    // Store output for this stage
    cube::store_st_direct(&temp_cs_f_smem, &mut outputs.output, stage_offset, 0, 0);

    sync_cube();

    // === Steps 9-10: Weight and bias updates (in place) ===
    let last_token_eta_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_eta_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_eta_idx % comptime!(LINE_SIZE);
    let token_eta_line = inputs.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    // Load ttt_lr_eta and scale by token_eta[last]
    let mut last_eta_rv = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&inputs.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // Reload k transposed for weight update
    cube::load_st_transpose(&inputs.xk, &mut q_smem, stage_offset, 0, 0);

    sync_cube();

    // Scale columns of q_smem [F, CS] by last_eta: q_smem[f, k] *= last_eta[k]
    q_smem.mul_row(&last_eta_rv);

    sync_cube();

    // Compute weight_update = scaled_xk^T @ grad = q_smem @ z1_smem = [F, CS] @ [CS, F] = [F, F]
    let mut weight_update_reg = P::f_f_reg();
    weight_update_reg.zero();
    cube::mma_AB(&mut weight_update_reg, &q_smem, &z1_smem);

    sync_cube();

    // Update weight in place: weight -= weight_update
    let mut weight_reg = P::f_f_reg();
    cube::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    cube::store_rt_to_st(&weight_reg, weight_smem);

    sync_cube();

    // Bias update: bias -= last_eta^T @ grad
    temp_cs_f_smem.copy_from(&z1_smem);

    sync_cube();

    temp_cs_f_smem.mul_col(&last_eta_rv);

    sync_cube();

    let mut bias_update_rv = P::f_reg_big();
    cube::reduce_cols_plane::<P::E, P::CS, P::F, SumOp>(&temp_cs_f_smem, &mut bias_update_rv);

    // Update bias in place
    bias_rv.sub(&bias_update_rv);
}

/// Fused TTT-Linear forward pass kernel (single mini-batch).
///
/// Each CUBE handles one (batch, head) pair.
/// Computes the TTT-Linear forward pass with online weight updates.
///
/// Algorithm:
/// 1. z1 = xk @ W + b
/// 2. reconstruction_target = xv - xk
/// 3. grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
/// 4. eta_matrix = outer(token_eta, ttt_lr_eta).tril()
/// 5. attn_scores = xq @ xk^T, attn1 = attn_scores.tril()
/// 6. b1_bar = bias - eta_matrix @ grad_l_wrt_z1
/// 7. z1_bar = xq @ W - (eta_matrix * attn1) @ grad_l_wrt_z1 + b1_bar
/// 8. output = xq + layer_norm(z1_bar)
/// 9. weight_out = weight - (last_eta_col * xk).T @ grad
/// 10. bias_out = bias - sum_rows(last_eta_col * grad)
#[cube(launch)]
pub fn fused_ttt_forward_kernel<P: ParamsTrait>(
    inputs: &Inputs<P::E>,
    outputs: &mut Outputs<P::E>,
    fwd_intermediates: &mut ForwardIntermediates<P::E>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());

    // Compute base offsets
    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Initialize weight in shared memory
    let mut weight_smem = P::f_f_tile();
    cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_cube();

    // Initialize bias in register vector
    let mut bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::f_reg_big();
    let mut ln_bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process single stage
    fused_ttt_forward_stage::<P>(
        inputs,
        outputs,
        fwd_intermediates,
        &mut weight_smem,
        &mut bias_rv,
        &ln_weight_rv,
        &ln_bias_rv,
        base_qkv,
        ttt_lr_eta_idx,
        epsilon,
    );

    sync_cube();

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
}

/// Fused TTT-Linear forward pass kernel with multiple mini-batch stages.
///
/// Processes `num_stages` mini-batches in a single kernel launch, keeping
/// weight and bias in shared memory between stages to avoid global memory
/// round-trips.
///
/// Input tensors xq, xk, xv should have shape [batch, heads, num_stages * mini_batch_len, head_dim]
/// Output tensor should have the same shape.
#[cube(launch)]
pub fn fused_ttt_forward_kernel_multi<P: ParamsTrait>(
    inputs: &Inputs<P::E>,
    outputs: &mut Outputs<P::E>,
    fwd_intermediates: &mut ForwardIntermediates<P::E>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(config.mini_batch_len);
    let head_dim = comptime!(config.head_dim);

    // Stride to advance by one mini-batch in the sequence dimension (in scalars)
    let stage_stride = mini_batch_len * head_dim;

    // Compute base offsets
    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Initialize weight in shared memory
    let mut weight_smem = P::f_f_tile();
    cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_cube();

    // Initialize bias in register vector
    let mut bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params (shared across all stages)
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::f_reg_big();
    let mut ln_bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process all stages
    for stage in 0..num_stages {
        let stage_offset = base_qkv + (stage as usize) * stage_stride;
        let ttt_lr_offset = ttt_lr_eta_idx + (stage as usize) * mini_batch_len;

        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            fwd_intermediates,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            stage_offset,
            ttt_lr_offset,
            epsilon,
        );

        sync_cube();
    }

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
}
