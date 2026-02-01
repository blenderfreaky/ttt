//! Fused TTT-Linear backward pass kernel (tiled implementation).
//!
//! # Mathematical Structure
//!
//! The forward pass computes:
//! ```text
//! z1 = XK @ W + b
//! grad_l = layer_norm_l2_grad(z1, XV - XK)
//! z1_bar = XQ @ W + b - (η + η·attn) @ grad_l
//! output = XQ + layer_norm(z1_bar)
//! W_out = W - last_η · XK^T @ grad_l
//! b_out = b - last_η @ grad_l
//! ```
//!
//! where `η[i,j] = token_eta[i] * ttt_lr_eta[j]` is lower triangular,
//! and `attn = tril(XQ @ XK^T)`.
//!
//! The backward pass computes gradients in reverse order:
//! - Stage 4: LN backward (output → z1_bar)
//! - Stage 3: Update backward (z1_bar → W, b, grad_l dependencies)
//! - Stage 2: LN+L2 second derivative (grad_l → Z1)
//! - Stage 1: MatMul backward (Z1 → inputs)
//!
//! # Memory Optimization
//!
//! This implementation reduces shared memory usage by:
//! 1. Loading forward intermediates on-demand instead of all at once
//! 2. Reusing CS×CS tiles across computations in stage 3
//! 3. Using a tile pool with explicit variable renaming when tiles are repurposed

use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::forward::ForwardIntermediates;
use crate::ttt::cubecl_kernels::{
    FusedTttConfig,
    linear_fused_tile::helpers::{ParamsTrait, build_attn_matrix, build_eta_matrix},
};

// =============================================================================
// Data structures
// =============================================================================

/// Saved tensors from forward pass needed for backward.
#[derive(CubeType, CubeLaunch)]
pub struct SavedTensors<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub xv: Tensor<Line<F>>,
    pub weight_init: Tensor<Line<F>>,
    pub bias_init: Tensor<Line<F>>,
    pub weight_last: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

/// Gradient outputs from backward pass.
#[derive(CubeType, CubeLaunch)]
pub struct GradOutputs<F: Float> {
    pub grad_xq: Tensor<Line<F>>,
    pub grad_xk: Tensor<Line<F>>,
    pub grad_xv: Tensor<Line<F>>,
    pub grad_weight: Tensor<Line<F>>,
    pub grad_bias: Tensor<Line<F>>,
    pub grad_ttt_lr_eta: Tensor<Line<F>>,
    /// Atomic tensor for accumulating ln_weight gradients across batches.
    /// Shape: [num_heads, head_dim] (unbatched, shared across batch dimension)
    pub grad_ln_weight: Tensor<Atomic<F>>,
    /// Atomic tensor for accumulating ln_bias gradients across batches.
    /// Shape: [num_heads, head_dim] (unbatched, shared across batch dimension)
    pub grad_ln_bias: Tensor<Atomic<F>>,
}

/// Atomically add a register value to an atomic tensor.
/// Each element in the register is added to the corresponding position in the tensor.
#[cube]
fn atomic_add_rv<F: Float, L: Dim>(
    rv: &Rv<F, L>,
    tensor: &mut Tensor<Atomic<F>>,
    base_offset: usize,
) {
    // Only one thread per cube does the atomic add (all threads have same data)
    if UNIT_POS == 0 {
        #[unroll]
        for line_idx in 0..L::LINES {
            let line = rv.data[line_idx];
            #[unroll]
            for elem_idx in 0..LINE_SIZE {
                let idx = base_offset + line_idx * LINE_SIZE + elem_idx;
                tensor[idx].fetch_add(line[elem_idx]);
            }
        }
    }
}

// =============================================================================
// Stage 4: Layer norm backward
// =============================================================================

/// Outputs from stage 4 (stored in tile pool).
/// Note: grad_W_z1bar is accumulated directly into grad_L_W_last to save an F×F tile.
#[derive(CubeType)]
struct Stage4Outputs<P: ParamsTrait> {
    // grad_z1_bar stored in dedicated tile passed by caller
    // grad_W_z1bar accumulated into grad_L_W_last via temp_f_f tile
    grad_b_z1bar: Rv<P::E, P::F>,
    grad_ln_weight: Rv<P::E, P::F>,
    grad_ln_bias: Rv<P::E, P::F>,
}

/// Stage 4: Compute gradients through output layer norm.
/// grad_z1_bar is written to the output tile passed by caller.
/// grad_W_z1bar is computed and stored in temp_f_f (accumulated later, after stage 3 reads grad_W_last).
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage4_ln<P: ParamsTrait>(
    grad_output: &St<P::E, P::CS, P::F>,
    x_hat_ln: &St<P::E, P::CS, P::F>,
    std_ln: &Rv<P::E, P::CS>,
    q_smem: &St<P::E, P::F, P::CS>,
    ln_weight: &Rv<P::E, P::F>,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
    grad_z1_bar_out: &mut St<P::E, P::CS, P::F>,
    temp_f_f: &mut St<P::E, P::F, P::F>,
) -> Stage4Outputs<P> {
    let f_f = P::E::cast_from(P::F::VALUE as f32);
    let f_inv = P::E::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_ln_weight = sum(grad_output * x_hat_ln)
    scratch1.copy_from(grad_output);
    scratch1.mul(x_hat_ln);

    sync_cube();

    let mut grad_ln_weight = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(scratch1, &mut grad_ln_weight, buf);

    // grad_ln_bias = sum(grad_output)
    let mut grad_ln_bias = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_output, &mut grad_ln_bias, buf);

    // grad_x_hat = grad_output * ln_weight
    scratch1.copy_from(grad_output);
    scratch1.mul_row(ln_weight);

    sync_cube();

    // sum(grad_x_hat) per row
    let mut sum_gxh = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut sum_gxh, buf);

    // sum(grad_x_hat * x_hat) per row
    scratch2.copy_from(scratch1);
    scratch2.mul(x_hat_ln);

    sync_cube();

    let mut sum_gxh_xh = P::cs_reg_big();
    cube::sum_rows(scratch2, &mut sum_gxh_xh, buf);

    // grad_z1_bar = (grad_x_hat * F - sum_gxh - x_hat * sum_gxh_xh) / (std * F)
    grad_z1_bar_out.copy_from(scratch1);
    grad_z1_bar_out.mul_scalar(f_f);
    grad_z1_bar_out.sub_col(&sum_gxh);

    sync_cube();

    scratch2.copy_from(x_hat_ln);
    scratch2.mul_col(&sum_gxh_xh);

    sync_cube();

    grad_z1_bar_out.sub(scratch2);
    grad_z1_bar_out.div_col(std_ln);
    grad_z1_bar_out.mul_scalar(f_inv);

    sync_cube();

    // grad_W_z1bar = XQ^T @ grad_z1_bar
    // Store into temp_f_f (accumulated into grad_W_last after stage 3, which reads it)
    let mut dW_reg = P::f_f_reg();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, q_smem, grad_z1_bar_out);

    sync_cube();

    cube::store_rt_to_st(&dW_reg, temp_f_f);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_z1_bar_out, &mut grad_b_z1bar, buf);

    Stage4Outputs::<P> {
        grad_b_z1bar,
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 3: Update backward (reuses CS×CS tiles)
// =============================================================================

/// Outputs from stage 3.
#[derive(CubeType)]
struct Stage3Outputs<P: ParamsTrait> {
    grad_ttt_lr_eta: Rv<P::E, P::CS>,
}

/// Stage 3: Compute gradients through the dual-form update equations.
/// Uses only 2 CS×CS tiles (reused) instead of 9.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage3_update<P: ParamsTrait>(
    grad_z1_bar: &St<P::E, P::CS, P::F>,
    grad_l: &St<P::E, P::CS, P::F>,
    grad_W_last: &St<P::E, P::F, P::F>,
    grad_b_last: &Rv<P::E, P::F>,
    q_smem: &St<P::E, P::F, P::CS>,
    k_smem: &St<P::E, P::F, P::CS>,
    xk_smem: &St<P::E, P::CS, P::F>,
    xq_smem: &St<P::E, P::CS, P::F>,
    weight_init: &St<P::E, P::F, P::F>,
    token_eta: &Tensor<Line<P::E>>,
    ttt_lr_eta: &Tensor<Line<P::E>>,
    last_eta: &Rv<P::E, P::CS>,
    last_token_eta: P::E,
    ttt_lr_eta_idx: usize,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
    cs_cs_a: &mut St<P::E, P::CS, P::CS>,
    cs_cs_b: &mut St<P::E, P::CS, P::CS>,
    grad_grad_l_out: &mut St<P::E, P::CS, P::F>,
    grad_xq_mini_out: &mut St<P::E, P::CS, P::F>,
    grad_xk_mini_out: &mut St<P::E, P::CS, P::F>,
    grad_xk_attn_out: &mut St<P::E, P::CS, P::F>,
) -> Stage3Outputs<P> {
    // --- Compute grad_grad_l from three sources ---

    // Build η^T (upper triangular) into cs_cs_a
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, cs_cs_a, ttt_lr_eta_idx, true);

    // Build attn^T (upper triangular) into cs_cs_b
    build_attn_matrix::<P>(q_smem, k_smem, cs_cs_b, true);

    // Term A: η^T @ grad_z1_bar
    let mut term_a_reg = P::cs_f_reg();
    term_a_reg.zero();
    cube::mma_AB(&mut term_a_reg, cs_cs_a, grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_a_reg, grad_grad_l_out);

    sync_cube();

    // Term B: (η·attn)^T @ grad_z1_bar = (η^T · attn^T) @ grad_z1_bar
    cs_cs_a.mul(cs_cs_b);

    sync_cube();

    let mut term_b_reg = P::cs_f_reg();
    term_b_reg.zero();
    cube::mma_AB(&mut term_b_reg, cs_cs_a, grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_b_reg, scratch1);

    sync_cube();

    // grad_grad_l = -(term_a + term_b)
    grad_grad_l_out.add(scratch1);
    grad_grad_l_out.neg();

    sync_cube();

    // Sources 1 & 2: -(last_η · XK) @ grad_W_last - last_η · grad_b_last
    scratch1.copy_from(xk_smem);
    scratch1.mul_col(last_eta);

    sync_cube();

    let mut src12_reg = P::cs_f_reg();
    src12_reg.zero();
    cube::mma_AB(&mut src12_reg, scratch1, grad_W_last);

    sync_cube();

    cube::store_rt_to_st(&src12_reg, scratch1);

    sync_cube();

    grad_grad_l_out.sub(scratch1);

    scratch1.set_row(grad_b_last);
    scratch1.mul_col(last_eta);

    sync_cube();

    grad_grad_l_out.sub(scratch1);

    sync_cube();

    // --- grad_xq_mini = grad_z1_bar @ W_init^T ---
    let mut grad_xq_reg = P::cs_f_reg();
    grad_xq_reg.zero();
    cube::mma_ABt(&mut grad_xq_reg, grad_z1_bar, weight_init);

    sync_cube();

    cube::store_rt_to_st(&grad_xq_reg, grad_xq_mini_out);

    sync_cube();

    // --- Gradient through attn = XQ @ XK^T ---
    // d_attn = -grad_z1_bar @ grad_l^T * η (element-wise, lower triangular)

    // Build η (lower triangular) into cs_cs_a (reuse)
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, cs_cs_a, ttt_lr_eta_idx, false);

    // d_attn_base = grad_z1_bar @ grad_l^T into cs_cs_b
    let mut d_attn_reg = P::cs_cs_reg();
    d_attn_reg.zero();
    cube::mma_ABt(&mut d_attn_reg, grad_z1_bar, grad_l);

    sync_cube();

    cube::store_rt_to_st(&d_attn_reg, cs_cs_b);

    sync_cube();

    cs_cs_b.mul(cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.tril();

    sync_cube();

    // d_xq (from attn) = d_attn @ XK
    let mut d_xq_attn_reg = P::cs_f_reg();
    d_xq_attn_reg.zero();
    cube::mma_AB(&mut d_xq_attn_reg, cs_cs_b, xk_smem);

    sync_cube();

    cube::store_rt_to_st(&d_xq_attn_reg, scratch1);

    sync_cube();

    grad_xq_mini_out.add(scratch1);

    sync_cube();

    // d_xk (from attn) = d_attn^T @ XQ
    // Compute d_attn^T directly: grad_l @ grad_z1_bar^T * η^T
    let mut d_attn_t_reg = P::cs_cs_reg();
    d_attn_t_reg.zero();
    cube::mma_ABt(&mut d_attn_t_reg, grad_l, grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&d_attn_t_reg, cs_cs_b);

    sync_cube();

    // Rebuild η^T into cs_cs_a
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, cs_cs_a, ttt_lr_eta_idx, true);

    cs_cs_b.mul(cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.triu();

    sync_cube();

    let mut d_xk_attn_reg = P::cs_f_reg();
    d_xk_attn_reg.zero();
    cube::mma_AB(&mut d_xk_attn_reg, cs_cs_b, xq_smem);

    sync_cube();

    cube::store_rt_to_st(&d_xk_attn_reg, grad_xk_attn_out);

    sync_cube();

    // --- grad_xk_mini from weight update term ---
    // grad_l_Last = grad_l @ grad_W_last^T
    let mut grad_l_last_reg = P::cs_f_reg();
    grad_l_last_reg.zero();
    cube::mma_ABt(&mut grad_l_last_reg, grad_l, grad_W_last);

    sync_cube();

    cube::store_rt_to_st(&grad_l_last_reg, scratch1);

    sync_cube();

    // grad_xk_mini = -grad_l_last * last_eta
    grad_xk_mini_out.copy_from(scratch1);
    grad_xk_mini_out.mul_col(last_eta);
    grad_xk_mini_out.neg();

    sync_cube();

    // --- grad_ttt_lr_eta from weight/bias update ---
    // grad_last_eta = sum(-(grad_l_last * XK) - (grad_b_last * grad_l))
    scratch2.copy_from(scratch1); // scratch1 = grad_l_last
    scratch2.mul(xk_smem);

    sync_cube();

    let mut grad_eta_term1 = P::cs_reg_big();
    cube::sum_rows(scratch2, &mut grad_eta_term1, buf);

    scratch2.set_row(grad_b_last);
    scratch2.mul(grad_l);

    sync_cube();

    let mut grad_eta_term2 = P::cs_reg_big();
    cube::sum_rows(scratch2, &mut grad_eta_term2, buf);

    grad_eta_term1.add(&grad_eta_term2);
    grad_eta_term1.neg();

    // Scale by last_token_eta
    grad_eta_term1.mul_scalar(last_token_eta);

    // --- grad_ttt_lr_eta from η terms in z1_bar ---
    // Reuse cs_cs_a/b for attn computation
    build_attn_matrix::<P>(q_smem, k_smem, cs_cs_a, false); // attn (lower tri)

    // d_eta_base = -grad_z1_bar @ grad_l^T (lower tri)
    let mut d_eta_base_reg = P::cs_cs_reg();
    d_eta_base_reg.zero();
    cube::mma_ABt(&mut d_eta_base_reg, grad_z1_bar, grad_l);

    sync_cube();

    cube::store_rt_to_st(&d_eta_base_reg, cs_cs_b);

    sync_cube();

    cs_cs_b.neg();
    cs_cs_b.tril();

    // d_eta = d_eta_base + d_eta_base * attn
    cs_cs_a.mul(cs_cs_b);

    sync_cube();

    cs_cs_b.add(cs_cs_a);

    sync_cube();

    // grad_ttt_lr_eta += sum_cols(d_eta * token_eta)
    // d_eta[i,j] contributes to grad_ttt_lr_eta[j] with weight token_eta[i]
    // Manual implementation: multiply each row by token_eta[row], then sum columns
    // Load token_eta into a register vector
    let mut token_eta_rv = P::cs_reg_big();
    cube::broadcast::load_rv_direct(token_eta, &mut token_eta_rv, 0);

    // Multiply cs_cs_b rows by token_eta (in-place)
    cs_cs_b.mul_col(&token_eta_rv);

    sync_cube();

    // Sum columns to get grad_ttt_lr_eta contribution
    let mut grad_ttt_lr_eta = P::cs_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::CS, SumOp>(cs_cs_b, &mut grad_ttt_lr_eta, buf);

    grad_ttt_lr_eta.add(&grad_eta_term1);

    Stage3Outputs::<P> {
        grad_ttt_lr_eta,
    }
}

// =============================================================================
// Stage 2: LN+L2 second derivative
// =============================================================================

/// Outputs from stage 2.
#[derive(CubeType)]
struct Stage2Outputs<P: ParamsTrait> {
    grad_ln_weight: Rv<P::E, P::F>,
    grad_ln_bias: Rv<P::E, P::F>,
}

/// Stage 2: Compute second derivative through fused LN+L2 gradient.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage2_ln_l2<P: ParamsTrait>(
    grad_grad_l: &St<P::E, P::CS, P::F>,
    x_hat_fused: &St<P::E, P::CS, P::F>,
    std_fused: &Rv<P::E, P::CS>,
    grad_output_fused: &St<P::E, P::CS, P::F>,
    grad_x_hat_fused: &St<P::E, P::CS, P::F>,
    grad_l: &St<P::E, P::CS, P::F>,
    ln_weight: &Rv<P::E, P::F>,
    sum_gxh_xh_precomputed: &Rv<P::E, P::CS>,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
    grad_Z1_out: &mut St<P::E, P::CS, P::F>,
    grad_target_out: &mut St<P::E, P::CS, P::F>,
) -> Stage2Outputs<P> {
    let f_inv = P::E::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_L_grad_x_hat = (1/std) * grad_grad_l
    //                   + (1/F) * sum(-grad_grad_l / std)
    //                   + (1/F) * x_hat * sum(-grad_grad_l / std * x_hat)

    // Compute -grad_grad_l / std
    scratch1.copy_from(grad_grad_l);
    scratch1.neg();
    scratch1.div_col(std_fused);

    sync_cube();

    let mut sum1 = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut sum1, buf);

    scratch1.mul(x_hat_fused);

    sync_cube();

    let mut sum2 = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut sum2, buf);

    // Build grad_L_grad_x_hat in scratch2
    scratch2.copy_from(grad_grad_l);
    scratch2.div_col(std_fused);

    let mut s1 = sum1;
    s1.mul_scalar(f_inv);
    scratch2.add_col(&s1);

    sync_cube();

    scratch1.copy_from(x_hat_fused);
    scratch1.mul_col(&sum2);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    scratch2.add(scratch1);  // scratch2 = grad_L_grad_x_hat

    sync_cube();

    // grad_L_y = ln_weight * grad_L_grad_x_hat
    // Store in grad_target_out temporarily
    grad_target_out.copy_from(scratch2);
    grad_target_out.mul_row(ln_weight);

    sync_cube();

    // grad_ln_weight = sum(grad_output_fused * grad_L_grad_x_hat + grad_L_y * x_hat)
    scratch1.copy_from(grad_output_fused);
    scratch1.mul(scratch2);

    // Need another temp - use grad_Z1_out temporarily
    grad_Z1_out.copy_from(grad_target_out);
    grad_Z1_out.mul(x_hat_fused);

    sync_cube();

    scratch1.add(grad_Z1_out);

    sync_cube();

    let mut grad_ln_weight = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(scratch1, &mut grad_ln_weight, buf);

    // grad_ln_bias = sum(grad_L_y)
    let mut grad_ln_bias = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_target_out, &mut grad_ln_bias, buf);

    // grad_L_x_hat = grad_L_y * ln_weight
    //              + (1/F) * grad_x_hat * sum2
    //              + (1/F) * sum(grad_x_hat * x_hat) * (-grad_grad_l / std)
    // Store in grad_Z1_out
    grad_Z1_out.copy_from(grad_target_out);
    grad_Z1_out.mul_row(ln_weight);

    scratch1.copy_from(grad_x_hat_fused);
    scratch1.mul_col(&sum2);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    grad_Z1_out.add(scratch1);

    sync_cube();

    scratch1.copy_from(grad_grad_l);
    scratch1.neg();
    scratch1.div_col(std_fused);
    scratch1.mul_col(sum_gxh_xh_precomputed);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    grad_Z1_out.add(scratch1);  // grad_Z1_out now = grad_L_x_hat

    sync_cube();

    // grad_L_std = -grad_L_x_hat * (x_hat / std) - grad_grad_l * (grad_l / std)
    scratch1.copy_from(grad_Z1_out);
    scratch1.mul(x_hat_fused);
    scratch1.div_col(std_fused);
    scratch1.neg();

    // scratch2 still has grad_L_grad_x_hat, reuse it
    scratch2.copy_from(grad_grad_l);
    scratch2.mul(grad_l);
    scratch2.div_col(std_fused);

    sync_cube();

    scratch1.sub(scratch2);

    sync_cube();

    let mut sum_grad_L_std = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut sum_grad_L_std, buf);

    // Compute final grad_Z1:
    // grad_Z1 = grad_L_x_hat / std - (1/F) * sum(grad_L_x_hat) / std + (1/F) * sum(grad_L_std) * x_hat
    // grad_Z1_out currently holds grad_L_x_hat

    let mut sum_grad_L_x_hat = P::cs_reg_big();
    cube::sum_rows(grad_Z1_out, &mut sum_grad_L_x_hat, buf);

    grad_Z1_out.div_col(std_fused);

    let mut term2 = sum_grad_L_x_hat;
    term2.div(std_fused);
    term2.mul_scalar(f_inv);
    grad_Z1_out.sub_col(&term2);

    sync_cube();

    scratch1.copy_from(x_hat_fused);
    let mut term3 = sum_grad_L_std;
    term3.mul_scalar(f_inv);
    scratch1.mul_col(&term3);

    sync_cube();

    grad_Z1_out.add(scratch1);

    sync_cube();

    // grad_target = -ln_weight * grad_L_grad_x_hat
    // We lost grad_L_grad_x_hat... need to recompute or save it
    // Recompute: grad_L_grad_x_hat = grad_grad_l/std + (1/F)*sum1 + (1/F)*x_hat*sum2
    scratch2.copy_from(grad_grad_l);
    scratch2.div_col(std_fused);
    scratch2.add_col(&s1);

    scratch1.copy_from(x_hat_fused);
    scratch1.mul_col(&sum2);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    scratch2.add(scratch1);

    sync_cube();

    grad_target_out.copy_from(scratch2);
    grad_target_out.mul_row(ln_weight);
    grad_target_out.neg();

    sync_cube();

    Stage2Outputs::<P> {
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 1: Final assembly
// =============================================================================

/// Stage 1: Final gradient assembly and output storage.
/// Note: grad_W_z1bar is already accumulated into grad_W_last in stage 4.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage1_assemble<P: ParamsTrait>(
    grad_output: &St<P::E, P::CS, P::F>,
    // Stage 4 outputs (grad_W_z1bar accumulated earlier via temp_f_f)
    _grad_z1_bar: &St<P::E, P::CS, P::F>,
    grad_b_z1bar: &Rv<P::E, P::F>,
    grad_ln_weight_s4: &Rv<P::E, P::F>,
    grad_ln_bias_s4: &Rv<P::E, P::F>,
    // Stage 3 outputs
    grad_xq_mini: &St<P::E, P::CS, P::F>,
    grad_xk_mini: &St<P::E, P::CS, P::F>,
    grad_xk_attn: &St<P::E, P::CS, P::F>,
    grad_ttt_lr_eta: &Rv<P::E, P::CS>,
    // Stage 2 outputs
    grad_Z1: &St<P::E, P::CS, P::F>,
    grad_target: &St<P::E, P::CS, P::F>,
    grad_ln_weight_s2: &Rv<P::E, P::F>,
    grad_ln_bias_s2: &Rv<P::E, P::F>,
    // Inputs
    k_smem: &St<P::E, P::F, P::CS>,
    weight_init: &St<P::E, P::F, P::F>,
    // Temp F×F tile (reused from stage 4)
    temp_f_f: &mut St<P::E, P::F, P::F>,
    // Accumulated gradients (in/out)
    grad_W_last: &mut St<P::E, P::F, P::F>,
    grad_b_last: &mut Rv<P::E, P::F>,
    grad_ln_weight_acc: &mut Rv<P::E, P::F>,
    grad_ln_bias_acc: &mut Rv<P::E, P::F>,
    // Output storage
    grads: &mut GradOutputs<P::E>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
) {
    // grad_XQ = grad_output + grad_xq_mini
    scratch1.copy_from(grad_output);
    scratch1.add(grad_xq_mini);

    sync_cube();

    // Store grad_XQ
    cube::store_st_direct(scratch1, &mut grads.grad_xq, stage_offset, 0, 0);

    // grad_XV = grad_target
    cube::store_st_direct(grad_target, &mut grads.grad_xv, stage_offset, 0, 0);

    // grad_XK = -grad_target + grad_xk_mini + grad_xk_attn + grad_Z1 @ W_init^T
    let mut grad_xk_reg = P::cs_f_reg();
    grad_xk_reg.zero();
    cube::mma_ABt(&mut grad_xk_reg, grad_Z1, weight_init);

    sync_cube();

    cube::store_rt_to_st(&grad_xk_reg, scratch1);

    sync_cube();

    scratch1.sub(grad_target);
    scratch1.add(grad_xk_mini);
    scratch1.add(grad_xk_attn);

    sync_cube();

    cube::store_st_direct(scratch1, &mut grads.grad_xk, stage_offset, 0, 0);

    // Accumulate weight gradients
    // grad_W = grad_W_z1bar (already accumulated in stage 4) + XK^T @ grad_Z1
    let mut dW_reg = P::f_f_reg();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, k_smem, grad_Z1);

    sync_cube();

    // Reuse temp_f_f tile (was used for grad_W_z1bar in stage 4)
    cube::store_rt_to_st(&dW_reg, temp_f_f);

    sync_cube();

    grad_W_last.add(temp_f_f);
    // Note: grad_W_z1bar already accumulated into grad_W_last in stage 4

    sync_cube();

    // grad_b = grad_b_z1bar + sum(grad_Z1)
    let mut grad_b_z1 = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_Z1, &mut grad_b_z1, buf);

    grad_b_last.add(&grad_b_z1);
    grad_b_last.add(grad_b_z1bar);

    // Accumulate LN gradients
    grad_ln_weight_acc.add(grad_ln_weight_s4);
    grad_ln_weight_acc.add(grad_ln_weight_s2);
    grad_ln_bias_acc.add(grad_ln_bias_s4);
    grad_ln_bias_acc.add(grad_ln_bias_s2);

    // Store grad_ttt_lr_eta
    cube::broadcast::store_rv_direct(grad_ttt_lr_eta, &mut grads.grad_ttt_lr_eta, ttt_lr_eta_idx);
}

// =============================================================================
// Main backward stage function
// =============================================================================

/// Memory layout (for 16×32 tiles):
/// - Q/K transposed: q_smem, k_smem (2 F×CS = 8KB)
/// - Scratch: scratch1, scratch2 (2 CS×F = 8KB)
/// - CS×CS pool: cs_cs_a, cs_cs_b (2 CS×CS = 8KB)
/// - CS×F pool: 10 tiles for stage outputs (10 CS×F = 40KB)
/// - grad_l_smem (CS×F = 4KB) - included in pool as pool_cs_f_a
/// - temp_f_f: shared F×F tile for grad_W_z1bar (stage 4) and dW_tile (stage 1)
/// - weight_init_smem: F×F tile (persistent)
/// Total with F×F optimization: saves one F×F tile by reusing temp_f_f
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_backward_stage<P: ParamsTrait>(
    saved: &SavedTensors<P::E>,
    fwd: &ForwardIntermediates<P::E>,
    grad_L_XQW: &Tensor<Line<P::E>>,
    grad_L_W_last: &mut St<P::E, P::F, P::F>,
    grad_L_b_last: &mut Rv<P::E, P::F>,
    grad_L_ln_weight_acc: &mut Rv<P::E, P::F>,
    grad_L_ln_bias_acc: &mut Rv<P::E, P::F>,
    grads: &mut GradOutputs<P::E>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    #[comptime] _epsilon: f32,
) {
    let mut buf = ReduceBuf::<P::E>::new();

    // =========================================================================
    // Tile pool allocation (total 64KB for 16×32)
    // =========================================================================

    // Scratch tiles - reused within stages (8KB)
    let mut scratch1 = P::cs_f_tile();
    let mut scratch2 = P::cs_f_tile();

    // CS×CS tiles - reused across stage 3 computations (8KB)
    let mut cs_cs_a = P::cs_cs_tile();
    let mut cs_cs_b = P::cs_cs_tile();

    // CS×F tile pool for stage outputs (40KB = 10 tiles)
    let mut tile_grad_z1_bar = P::cs_f_tile();
    let mut tile_b = P::cs_f_tile();
    let mut tile_c = P::cs_f_tile();
    let mut tile_grad_xk_mini = P::cs_f_tile();
    let mut tile_e = P::cs_f_tile();
    let mut tile_f = P::cs_f_tile();
    let mut tile_grad_xk_attn = P::cs_f_tile();
    let mut tile_h = P::cs_f_tile();
    let mut tile_grad_Z1 = P::cs_f_tile();
    let mut tile_grad_target = P::cs_f_tile();

    // Shared F×F tile: used for grad_W_z1bar in stage 4, then reused for dW_tile in stage 1
    let mut temp_f_f = P::f_f_tile();

    // =========================================================================
    // Load persistent data
    // =========================================================================

    // Q/K transposed (needed by stages 4, 3)
    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();
    cube::load_st_transpose(&saved.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&saved.xk, &mut k_smem, stage_offset, 0, 0);

    // Weight init (needed by stages 3, 1)
    let mut weight_init_smem = P::f_f_tile();
    let base_weight_init = index_2d(&saved.weight_init, CUBE_POS_X as usize, CUBE_POS_Y as usize);
    cube::load_st_direct(&saved.weight_init, &mut weight_init_smem, base_weight_init, 0, 0);

    // LN weight
    let base_ln = index_2d(&saved.ln_weight, CUBE_POS_Y as usize, 0);
    let mut ln_weight_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);

    // Last eta computation
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta);

    // grad_l (needed by stages 3, 2)
    let mut grad_l_smem = P::cs_f_tile();
    cube::load_st_direct(&fwd.grad_l_wrt_Z1, &mut grad_l_smem, stage_offset, 0, 0);

    sync_cube();

    // =========================================================================
    // Stage 4: LN backward
    // =========================================================================

    // Load x_hat_ln and grad_output on-demand
    cube::load_st_direct(&fwd.x_hat_ln, &mut tile_b, stage_offset, 0, 0);
    cube::load_st_direct(grad_L_XQW, &mut tile_c, stage_offset, 0, 0);

    // Rename tiles to reflect their content
    let mut x_hat_ln = tile_b;
    let mut grad_output_s4 = tile_c;

    // Load std_ln
    let mut std_ln = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&fwd.std_ln, &mut std_ln, ttt_lr_eta_idx);

    sync_cube();

    let stage4_out = backward_stage4_ln::<P>(
        &grad_output_s4,
        &x_hat_ln,
        &std_ln,
        &q_smem,
        &ln_weight_rv,
        &mut buf,
        &mut scratch1,
        &mut scratch2,
        &mut tile_grad_z1_bar,
        &mut temp_f_f,
    );

    // =========================================================================
    // Stage 3: Update backward
    // =========================================================================

    // Repurpose tiles: x_hat_ln -> xk_smem, grad_output_s4 -> xq_smem
    let mut xk_smem = x_hat_ln;
    let mut xq_smem = grad_output_s4;
    cube::load_st_direct(&saved.xk, &mut xk_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, &mut xq_smem, stage_offset, 0, 0);

    sync_cube();

    let stage3_out = backward_stage3_update::<P>(
        &tile_grad_z1_bar,
        &grad_l_smem,
        grad_L_W_last,
        grad_L_b_last,
        &q_smem,
        &k_smem,
        &xk_smem,
        &xq_smem,
        &weight_init_smem,
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &last_eta_rv,
        last_token_eta,
        ttt_lr_eta_idx,
        &mut buf,
        &mut scratch1,
        &mut scratch2,
        &mut cs_cs_a,
        &mut cs_cs_b,
        &mut tile_e,           // grad_grad_l output
        &mut tile_f,           // grad_xq_mini output (temporary)
        &mut tile_grad_xk_mini,
        &mut tile_grad_xk_attn,
    );

    // Rename outputs from stage 3
    let grad_grad_l = tile_e;
    let mut grad_xq_mini = xq_smem;  // Repurpose xq_smem tile
    grad_xq_mini.copy_from(&tile_f);

    sync_cube();

    // Now that stage 3 is done reading grad_L_W_last, accumulate grad_W_z1bar from temp_f_f
    grad_L_W_last.add(&temp_f_f);

    sync_cube();

    // =========================================================================
    // Stage 2: LN+L2 second derivative
    // =========================================================================

    // Repurpose tiles for stage 2 inputs
    let mut x_hat_fused = xk_smem;  // Repurpose xk_smem
    let mut grad_x_hat_fused = tile_f;  // Repurpose tile_f

    // Load std_fused
    let mut std_fused = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&fwd.std_fused, &mut std_fused, ttt_lr_eta_idx);

    // Load x_hat_fused and grad_x_hat_fused
    cube::load_st_direct(&fwd.x_hat_fused, &mut x_hat_fused, stage_offset, 0, 0);
    cube::load_st_direct(&fwd.grad_x_hat_fused, &mut grad_x_hat_fused, stage_offset, 0, 0);

    sync_cube();

    // Compute sum(grad_x_hat * x_hat) using scratch1 as temp
    scratch1.copy_from(&grad_x_hat_fused);
    scratch1.mul(&x_hat_fused);

    sync_cube();

    let mut sum_gxh_xh = P::cs_reg_big();
    cube::sum_rows(&scratch1, &mut sum_gxh_xh, &mut buf);

    // Load grad_output_fused
    let mut grad_output_fused = tile_h;
    cube::load_st_direct(&fwd.grad_output_fused, &mut grad_output_fused, stage_offset, 0, 0);

    sync_cube();

    let stage2_out = backward_stage2_ln_l2::<P>(
        &grad_grad_l,
        &x_hat_fused,
        &std_fused,
        &grad_output_fused,
        &grad_x_hat_fused,
        &grad_l_smem,
        &ln_weight_rv,
        &sum_gxh_xh,
        &mut buf,
        &mut scratch1,
        &mut scratch2,
        &mut tile_grad_Z1,
        &mut tile_grad_target,
    );

    // =========================================================================
    // Stage 1: Final assembly
    // =========================================================================

    // Repurpose grad_grad_l tile for grad_output reload
    let mut grad_output_s1 = grad_grad_l;
    cube::load_st_direct(grad_L_XQW, &mut grad_output_s1, stage_offset, 0, 0);

    // Repurpose x_hat_fused for scratch in stage 1
    let mut scratch_s1 = x_hat_fused;

    sync_cube();

    backward_stage1_assemble::<P>(
        &grad_output_s1,
        // Stage 4 outputs (grad_W_z1bar already accumulated via temp_f_f)
        &tile_grad_z1_bar,
        &stage4_out.grad_b_z1bar,
        &stage4_out.grad_ln_weight,
        &stage4_out.grad_ln_bias,
        // Stage 3 outputs
        &grad_xq_mini,
        &tile_grad_xk_mini,
        &tile_grad_xk_attn,
        &stage3_out.grad_ttt_lr_eta,
        // Stage 2 outputs
        &tile_grad_Z1,
        &tile_grad_target,
        &stage2_out.grad_ln_weight,
        &stage2_out.grad_ln_bias,
        // Inputs
        &k_smem,
        &weight_init_smem,
        // Temp F×F tile (reused from stage 4)
        &mut temp_f_f,
        // Accumulators
        grad_L_W_last,
        grad_L_b_last,
        grad_L_ln_weight_acc,
        grad_L_ln_bias_acc,
        // Outputs
        grads,
        stage_offset,
        ttt_lr_eta_idx,
        &mut buf,
        &mut scratch_s1,
    );
}

// =============================================================================
// Kernel entry points
// =============================================================================

/// Fused TTT-Linear backward pass kernel (single mini-batch).
#[cube(launch)]
pub fn fused_ttt_backward_kernel<P: ParamsTrait>(
    saved: &SavedTensors<P::E>,
    fwd: &ForwardIntermediates<P::E>,
    grad_output: &Tensor<Line<P::E>>,
    grads: &mut GradOutputs<P::E>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    let mut grad_L_W_last = P::f_f_tile();
    grad_L_W_last.fill(P::E::new(0.0));

    let mut grad_L_b_last = P::f_reg_big();
    grad_L_b_last.fill(P::E::new(0.0));

    let mut grad_L_ln_weight_acc = P::f_reg_big();
    grad_L_ln_weight_acc.fill(P::E::new(0.0));

    let mut grad_L_ln_bias_acc = P::f_reg_big();
    grad_L_ln_bias_acc.fill(P::E::new(0.0));

    sync_cube();

    fused_ttt_backward_stage::<P>(
        saved,
        fwd,
        grad_output,
        &mut grad_L_W_last,
        &mut grad_L_b_last,
        &mut grad_L_ln_weight_acc,
        &mut grad_L_ln_bias_acc,
        grads,
        base_qkv,
        ttt_lr_eta_idx,
        epsilon,
    );

    sync_cube();

    // Store accumulated weight/bias gradients
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);
    cube::store_st_direct(&grad_L_W_last, &mut grads.grad_weight, grad_weight_base, 0, 0);
    cube::broadcast::store_rv_direct(&grad_L_b_last, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let base_ln = head_idx * P::F::VALUE;
    atomic_add_rv::<P::E, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    atomic_add_rv::<P::E, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
}

/// Fused TTT-Linear backward pass kernel (multi-stage).
#[cube(launch)]
pub fn fused_ttt_backward_kernel_multi<P: ParamsTrait>(
    saved: &SavedTensors<P::E>,
    fwd: &ForwardIntermediates<P::E>,
    grad_output: &Tensor<Line<P::E>>,
    grads: &mut GradOutputs<P::E>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(P::CS::VALUE);
    let num_stages = num_stages as usize;

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let base_ttt_lr_eta = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    let mut grad_L_W_last = P::f_f_tile();
    grad_L_W_last.fill(P::E::new(0.0));

    let mut grad_L_b_last = P::f_reg_big();
    grad_L_b_last.fill(P::E::new(0.0));

    let mut grad_L_ln_weight_acc = P::f_reg_big();
    grad_L_ln_weight_acc.fill(P::E::new(0.0));

    let mut grad_L_ln_bias_acc = P::f_reg_big();
    grad_L_ln_bias_acc.fill(P::E::new(0.0));

    sync_cube();

    // Process stages in reverse order (backward through time)
    #[unroll]
    for stage in 0..num_stages {
        let stage_idx = num_stages - 1 - stage;
        let stage_offset = base_qkv + stage_idx * mini_batch_len;
        let ttt_lr_eta_idx = base_ttt_lr_eta + stage_idx;

        fused_ttt_backward_stage::<P>(
            saved,
            fwd,
            grad_output,
            &mut grad_L_W_last,
            &mut grad_L_b_last,
            &mut grad_L_ln_weight_acc,
            &mut grad_L_ln_bias_acc,
            grads,
            stage_offset,
            ttt_lr_eta_idx,
            epsilon,
        );

        sync_cube();
    }

    // Store accumulated weight/bias gradients
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);
    cube::store_st_direct(&grad_L_W_last, &mut grads.grad_weight, grad_weight_base, 0, 0);
    cube::broadcast::store_rv_direct(&grad_L_b_last, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let base_ln = head_idx * P::F::VALUE;
    atomic_add_rv::<P::E, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    atomic_add_rv::<P::E, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
}
