#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{
    impl_reduction_ops,
    prelude::*,
    reduction_ops::{ReductionOp, SumOp},
    util::{index_2d, sync_planes},
};

impl_reduction_ops! {
    SumSq<F> {
        identity => Line::<F>::empty(LINE_SIZE).fill(F::new(0.0));
        combine(a, b) => a + b * b;
        finalize(line) => line[0] + line[1] + line[2] + line[3];
        plane_reduce(val) => plane_sum(val);
    }
}

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

// We use a trait here because
// inherent assoc types are unstable
#[cube]
pub trait ParamsTrait: Send + Sync + 'static {
    type E: Float;
    type CS: Dim;
    type F: Dim;

    type CS_Reg: Dim;
    type F_Reg: Dim;

    // CubeCL won't let us do default impls
    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F>;
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS>;
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS>;
    fn cs_vec() -> Sv<Self::E, Self::CS>;
    fn f_f_tile() -> St<Self::E, Self::F, Self::F>;
    fn f_vec() -> Sv<Self::E, Self::F>;

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg>;
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg>;
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg>;
    fn cs_reg() -> Rv<Self::E, Self::CS_Reg>;
    fn f_reg() -> Rv<Self::E, Self::F_Reg>;
}

pub struct Params<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> {
    _phantom: std::marker::PhantomData<(E, CS, F, CS_Reg, F_Reg)>,
}

#[cube]
impl<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> ParamsTrait
    for Params<E, CS, F, CS_Reg, F_Reg>
{
    type E = E;
    type CS = CS;
    type F = F;
    type CS_Reg = CS_Reg;
    type F_Reg = F_Reg;

    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F> {
        St::new()
    }
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS> {
        St::new()
    }
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS> {
        St::new()
    }
    fn cs_vec() -> Sv<Self::E, Self::CS> {
        Sv::new()
    }
    fn f_f_tile() -> St<Self::E, Self::F, Self::F> {
        St::new()
    }
    fn f_vec() -> Sv<Self::E, Self::F> {
        Sv::new()
    }

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg> {
        Rt::new()
    }
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg> {
        Rt::new()
    }
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg> {
        Rt::new()
    }

    fn cs_reg() -> Rv<Self::E, Self::CS_Reg> {
        Rv::new()
    }
    fn f_reg() -> Rv<Self::E, Self::F_Reg> {
        Rv::new()
    }
}

/// Computes layer norm forward pass only.
/// Uses shared memory tiles with plane-level cooperative operations.
///
/// Given input `x` of shape [R, C] (R rows, C columns),
/// and layer norm parameters weight/bias of shape [C]:
///
/// Forward:
///   mean[r] = sum(x[r, :]) / C
///   var[r] = sum((x[r, :] - mean[r])^2) / C
///   std[r] = sqrt(var[r] + epsilon)
///   norm[r, c] = (x[r, c] - mean[r]) / std[r]
///   out[r, c] = weight[c] * norm[r, c] + bias[c]
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain the normalized output
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_forward<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    plane::sum_st_rows(x, &mut mean);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    // Sync after modifying shared memory before reading again
    sync_planes();

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std = Rv::<F, R>::new();
    plane::reduce_st_rows::<F, R, C, SumSqOp>(x, &mut std);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Step 5: x /= std → x now contains norm
    x.div_col(&std);

    // Sync after modifying shared memory before reading again
    sync_planes();

    // Step 6: x = weight * x + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
}

/// Computes layer norm forward and L2 loss gradient backpropagated through layer norm.
/// Uses shared memory tiles with plane-level cooperative operations.
///
/// Given input `x` of shape [R, C] (R rows, C columns), target of same shape,
/// and layer norm parameters weight/bias of shape [C]:
///
/// Forward:
///   mean[r] = sum(x[r, :]) / C
///   var[r] = sum((x[r, :] - mean[r])^2) / C
///   std[r] = sqrt(var[r] + epsilon)
///   norm[r, c] = (x[r, c] - mean[r]) / std[r]
///   out[r, c] = weight[c] * norm[r, c] + bias[c]
///
/// L2 gradient backprop:
///   dl_dout = out - target
///   dl_dnorm = dl_dout * weight
///   dl_dx = (dl_dnorm * C - sum(dl_dnorm) - norm * sum(dl_dnorm * norm)) / (std * C)
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain dl_dx (the gradient)
/// * `target` - Target tile [R, C] for L2 loss
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `temp` - Scratch tile [R, C] for intermediate computations
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_l2_grad<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    target: &St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    temp: &mut St<F, R, C>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));
    let c_f = F::cast_from(C::VALUE as f32);

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    plane::sum_st_rows(x, &mut mean);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    // Sync after modifying x before reading for variance
    sync_planes();

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std = Rv::<F, R>::new();
    plane::reduce_st_rows::<F, R, C, SumSqOp>(x, &mut std);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Step 5: x /= std → x now contains norm
    x.div_col(&std);

    // Sync after modifying x before copying to temp
    sync_planes();

    // Step 6: temp = x (copy norm), then temp = weight * temp + bias
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);

    // Step 7: temp -= target → temp = dl_dout
    temp.sub(target);

    // Step 8: temp *= weight → temp = dl_dnorm
    temp.mul_row(ln_weight);

    // Sync after modifying temp before reading for reduction
    sync_planes();

    // Step 9: Compute reduction terms
    let mut sum_dl_dnorm = Rv::<F, R>::new();
    plane::sum_st_rows(temp, &mut sum_dl_dnorm);

    // temp = dl_dnorm currently, we need sum(dl_dnorm * norm)
    // Multiply temp by x (norm), sum, then rebuild dl_dnorm
    temp.mul(x);

    // Sync after modifying temp before reading for reduction
    sync_planes();

    let mut sum_dl_dnorm_norm = Rv::<F, R>::new();
    plane::sum_st_rows(temp, &mut sum_dl_dnorm_norm);

    // Recompute dl_dnorm in temp (x still has norm)
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);
    temp.sub(target);
    temp.mul_row(ln_weight);

    // Sync after modifying temp before continuing
    sync_planes();

    // Step 10: dl_dx = (dl_dnorm * C - sum_dl_dnorm - norm * sum_dl_dnorm_norm) / (std * C)

    // temp = dl_dnorm * C
    temp.mul_scalar(c_f);

    // temp -= sum_dl_dnorm
    temp.sub_col(&sum_dl_dnorm);

    // x = norm * sum_dl_dnorm_norm (col broadcast), then temp -= x
    x.mul_col(&sum_dl_dnorm_norm);

    // Sync after modifying x before reading
    sync_planes();

    temp.sub(x);

    // temp /= (std * C)
    std.mul_scalar(c_f);
    temp.div_col(&std);

    // Sync after modifying temp before final copy
    sync_planes();

    // Copy result to x
    x.copy_from(temp);
}

/// Extract the last row of a shared memory tile into a register vector.
/// Cooperative: all threads participate, result is broadcast to all.
#[cube]
#[must_use]
pub fn extract_last_row<F: Float, R: Dim, C: Dim>(st: &St<F, R, C>) -> Rv<F, C> {
    // The last row is at index R-1
    // We read it using plane::sum with a mask that selects only that row
    // Actually, simpler: each thread can just read the last row
    let mut result = Rv::<F, C>::new();
    let last_row = R::VALUE - 1;
    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll]
    for c_line in 0..C::LINES {
        let phys_col = plane::swizzle(last_row, c_line, mask);
        let s_idx = last_row * vec_stride + phys_col;
        result.data[c_line] = st.data[s_idx];
    }
    result
}

/// Process one mini-batch stage of the TTT-Linear forward pass.
///
/// This is the inner loop body.
/// Weight and bias are kept in shared memory / registers and updated in place.
///
/// # Arguments
/// * `inputs` - Input tensors (xq, xk, xv indexed by stage_offset)
/// * `outputs` - Output tensor (indexed by stage_offset)
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

    // Load QKV for this stage
    plane::load_st_transpose(&inputs.xq, &mut q_smem, stage_offset, 0, 0);
    plane::load_st_transpose(&inputs.xk, &mut k_smem, stage_offset, 0, 0);
    plane::load_st_direct(&inputs.xk, &mut k_direct_smem, stage_offset, 0, 0);
    plane::load_st_direct(&inputs.xv, &mut v_direct_smem, stage_offset, 0, 0);

    sync_planes();

    // Step 1: z1 = xk @ W + b
    let mut z1_reg = P::cs_f_reg();
    z1_reg.zero();
    plane::mma_AtB(&mut z1_reg, &k_smem, weight_smem);

    sync_planes();

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

    plane::store_rt_to_st(&z1_reg, &mut z1_smem);

    sync_planes();

    // Step 2: reconstruction_target = xv - xk
    v_direct_smem.sub(&k_direct_smem);

    sync_planes();

    // Step 3: grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
    layer_norm_l2_grad::<P::E, P::CS, P::F>(
        &mut z1_smem,
        &v_direct_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut temp_cs_f_smem,
        epsilon,
    );

    sync_planes();

    // Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
    let mut eta_reg = P::cs_cs_reg();
    eta_reg.zero();

    let tiles_per_row_eta = P::CS::VALUE / P::CS_Reg::VALUE;
    let num_cs_cs_tiles = tiles_per_row_eta * tiles_per_row_eta;
    let participates_in_cs_cs = (UNIT_POS as usize) < num_cs_cs_tiles;

    let tile_row_eta = (UNIT_POS as usize) / tiles_per_row_eta;
    let tile_col_eta = (UNIT_POS as usize) % tiles_per_row_eta;

    let mut token_eta_reg = P::cs_reg();
    let mut ttt_lr_eta_reg = P::cs_reg();
    if participates_in_cs_cs {
        let token_eta_offset = tile_row_eta * P::CS_Reg::VALUE;
        let ttt_lr_eta_offset = ttt_lr_eta_idx + tile_col_eta * P::CS_Reg::VALUE;
        plane::load_rv_direct(&inputs.token_eta, &mut token_eta_reg, token_eta_offset);
        plane::load_rv_direct(&inputs.ttt_lr_eta, &mut ttt_lr_eta_reg, ttt_lr_eta_offset);

        eta_reg.add_col(&token_eta_reg);
        eta_reg.mul_row(&ttt_lr_eta_reg);

        plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);
    }

    sync_planes();

    eta_matrix_smem.tril();

    sync_planes();

    // Step 5: attn_scores = xq @ xk^T, tril
    let mut attn_reg = P::cs_cs_reg();
    attn_reg.zero();
    plane::mma_AtB(&mut attn_reg, &q_smem, &k_smem);

    sync_planes();

    if participates_in_cs_cs {
        plane::store_rt_to_st(&attn_reg, &mut attn_smem);
    }

    sync_planes();

    attn_smem.tril();

    sync_planes();

    // Steps 6-7: z1_bar computation
    // Recompute eta (was overwritten by tril)
    if participates_in_cs_cs {
        eta_reg.zero();
        eta_reg.add_col(&token_eta_reg);
        eta_reg.mul_row(&ttt_lr_eta_reg);
        plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);
    }

    sync_planes();

    eta_matrix_smem.tril();

    sync_planes();

    // eta @ grad
    let mut eta_grad_reg = P::cs_f_reg();
    eta_grad_reg.zero();
    plane::mma_AB(&mut eta_grad_reg, &eta_matrix_smem, &z1_smem);

    sync_planes();

    // eta * attn
    eta_matrix_smem.mul(&attn_smem);

    sync_planes();

    // (eta * attn) @ grad
    let mut eta_attn_grad_reg = P::cs_f_reg();
    eta_attn_grad_reg.zero();
    plane::mma_AB(&mut eta_attn_grad_reg, &eta_matrix_smem, &z1_smem);

    sync_planes();

    // z1_bar = xq @ W
    let mut z1_bar_reg = P::cs_f_reg();
    z1_bar_reg.zero();
    plane::mma_AtB(&mut z1_bar_reg, &q_smem, weight_smem);

    sync_planes();

    // z1_bar -= (eta * attn) @ grad
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // z1_bar += bias
    z1_bar_reg.add_row(&bias_reg);

    // z1_bar -= eta @ grad
    z1_bar_reg.sub(&eta_grad_reg);

    // Store z1_bar to shared memory for layer norm
    plane::store_rt_to_st(&z1_bar_reg, &mut temp_cs_f_smem);

    sync_planes();

    // Step 8: layer_norm + add xq
    layer_norm_forward::<P::E, P::CS, P::F>(&mut temp_cs_f_smem, ln_weight_rv, ln_bias_rv, epsilon);

    sync_planes();

    // Load xq into k_direct_smem
    plane::load_st_direct(&inputs.xq, &mut k_direct_smem, stage_offset, 0, 0);

    sync_planes();

    // Add: output = xq + layer_norm(z1_bar)
    temp_cs_f_smem.add(&k_direct_smem);

    sync_planes();

    // Store output for this stage
    plane::store_st_direct(&temp_cs_f_smem, &mut outputs.output, stage_offset, 0, 0);

    sync_planes();

    // === Steps 9-10: Weight and bias updates (in place) ===
    let last_token_eta_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_eta_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_eta_idx % comptime!(LINE_SIZE);
    let token_eta_line = inputs.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    // Load ttt_lr_eta and scale by token_eta[last]
    let mut last_eta_rv = Rv::<P::E, P::CS>::new();
    plane::load_rv_direct(&inputs.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // Reload k transposed for weight update
    plane::load_st_transpose(&inputs.xk, &mut q_smem, stage_offset, 0, 0);

    sync_planes();

    // Scale columns of q_smem [F, CS] by last_eta: q_smem[f, k] *= last_eta[k]
    q_smem.mul_row(&last_eta_rv);

    sync_planes();

    // Compute weight_update = scaled_xk^T @ grad = q_smem @ z1_smem = [F, CS] @ [CS, F] = [F, F]
    let mut weight_update_reg = P::f_f_reg();
    weight_update_reg.zero();
    plane::mma_AB(&mut weight_update_reg, &q_smem, &z1_smem);

    sync_planes();

    // Update weight in place: weight -= weight_update
    let mut weight_reg = P::f_f_reg();
    plane::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    plane::store_rt_to_st(&weight_reg, weight_smem);

    sync_planes();

    // Bias update: bias -= last_eta^T @ grad
    temp_cs_f_smem.copy_from(&z1_smem);

    sync_planes();

    temp_cs_f_smem.mul_col(&last_eta_rv);

    sync_planes();

    let mut bias_update_rv = Rv::<P::E, P::F>::new();
    plane::reduce_st_cols::<P::E, P::CS, P::F, SumOp>(&temp_cs_f_smem, &mut bias_update_rv);

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
    plane::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_planes();

    // Initialize bias in register vector
    let mut bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = Rv::<P::E, P::F>::new();
    let mut ln_bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    plane::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process single stage
    fused_ttt_forward_stage::<P>(
        inputs,
        outputs,
        &mut weight_smem,
        &mut bias_rv,
        &ln_weight_rv,
        &ln_bias_rv,
        base_qkv,
        ttt_lr_eta_idx,
        epsilon,
    );

    sync_planes();

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    plane::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    plane::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
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
    plane::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_planes();

    // Initialize bias in register vector
    let mut bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params (shared across all stages)
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = Rv::<P::E, P::F>::new();
    let mut ln_bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    plane::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process all stages
    for stage in 0..num_stages {
        let stage_offset = base_qkv + (stage as usize) * stage_stride;
        let ttt_lr_offset = ttt_lr_eta_idx + (stage as usize) * mini_batch_len;

        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            stage_offset,
            ttt_lr_offset,
            epsilon,
        );

        sync_planes();
    }

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    plane::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    plane::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
}
