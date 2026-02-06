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

use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::{
    helpers::{
        ParamsTrait, RvbCsA, RvbCsV, RvbFA, RvbFV, StCsCs, StCsF, StFCs, StFF, build_attn_matrix,
        build_eta_attn_fused, build_eta_matrix,
    },
    layer_norm::{compute_grad_x_from_grad_x_hat, normalize_to_x_hat},
};
use crate::FusedTttConfig;

// =============================================================================
// Data structures
// =============================================================================

/// Saved tensors from forward pass needed for backward.
#[derive(CubeType, CubeLaunch)]
pub struct SavedTensors<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub weight_init: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
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
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_ln_weight: Tensor<Atomic<f32>>,
    /// Atomic tensor for accumulating ln_bias gradients across batches.
    /// Shape: [num_heads, head_dim] (unbatched, shared across batch dimension)
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_ln_bias: Tensor<Atomic<f32>>,
}

/// Additional inputs needed to recompute forward intermediates during backward.
/// These are inputs that weren't already in SavedTensors.
#[derive(CubeType, CubeLaunch)]
pub struct RecomputationInputs<F: Float> {
    pub xv: Tensor<Line<F>>,
    pub bias: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

/// Atomically add a register value to an f32 atomic tensor.
/// Values are cast to f32 before the atomic add since HIP/ROCm doesn't support bf16 atomics.
/// Each element in the register is added to the corresponding position in the tensor.
#[cube]
fn atomic_add_rv<F: Float, L: Dim>(
    rv: &Rv<F, L>,
    tensor: &mut Tensor<Atomic<f32>>,
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
                // Cast to f32 for atomic add (HIP doesn't support bf16 atomics)
                let val_f32: f32 = f32::cast_from(line[elem_idx]);
                tensor[idx].fetch_add(val_f32);
            }
        }
    }
}

// =============================================================================
// Stage 4: Layer norm backward
// =============================================================================

#[derive(CubeType)]
struct Stage4Outputs<P: ParamsTrait> {
    grad_b_z1bar: RvbFA<P>,
    grad_ln_weight: RvbFA<P>,
    grad_ln_bias: RvbFA<P>,
}

/// Stage 4: Compute gradients through output layer norm.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage4_ln<P: ParamsTrait>(
    grad_output: &StCsF<P>,
    x_hat_ln: &StCsF<P>,
    std_ln: &RvbCsV<P>,
    q_smem: &StFCs<P>,
    ln_weight: &RvbFV<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    scratch1: &mut StCsF<P>,
    scratch2: &mut StCsF<P>,
    grad_z1_bar_out: &mut StCsF<P>,
    temp_f_f: &mut StFF<P>,
) -> Stage4Outputs<P> {
    let f_f = P::EVal::cast_from(P::F::VALUE as f32);
    let f_inv = P::EVal::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_ln_weight = sum(grad_output * x_hat_ln)
    scratch1.copy_from(grad_output);
    scratch1.mul(x_hat_ln);

    sync_cube();

    let mut grad_ln_weight = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(scratch1, &mut grad_ln_weight, buf);

    // grad_ln_bias = sum(grad_output)
    let mut grad_ln_bias = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(grad_output, &mut grad_ln_bias, buf);

    // grad_x_hat = grad_output * ln_weight
    scratch1.copy_from(grad_output);
    scratch1.mul_row(ln_weight);

    sync_cube();

    // sum(grad_x_hat) per row
    let mut sum_gxh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch1, &mut sum_gxh_acc, buf);
    let sum_gxh = sum_gxh_acc.cast::<P::EVal>();

    // sum(grad_x_hat * x_hat) per row
    scratch2.copy_from(scratch1);
    scratch2.mul(x_hat_ln);

    sync_cube();

    let mut sum_gxh_xh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut sum_gxh_xh_acc, buf);
    let sum_gxh_xh = sum_gxh_xh_acc.cast::<P::EVal>();

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
    let mut dW_reg = P::rt_ff();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, q_smem, grad_z1_bar_out);

    sync_cube();

    cube::store_rt_to_st(&dW_reg, temp_f_f);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        grad_z1_bar_out,
        &mut grad_b_z1bar,
        buf,
    );

    Stage4Outputs::<P> {
        grad_b_z1bar,
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 3: Update backward
// Split into two parts to allow weight_init reload between them.
// =============================================================================

/// Outputs from stage 3 part 1.
#[derive(CubeType)]
struct Stage3Part1Outputs<P: ParamsTrait> {
    /// Intermediate for grad_ttt_lr_eta computation (from weight/bias update term)
    grad_eta_term1: RvbCsA<P>,
}

/// Outputs from stage 3 (final).
#[derive(CubeType)]
struct Stage3Outputs<P: ParamsTrait> {
    grad_ttt_lr_eta: RvbCsA<P>,
}

/// Stage 3 Part 1: Compute gradients that depend on grad_W_last.
/// This includes grad_grad_l, grad_xk_attn, grad_xk_mini, and part of grad_ttt_lr_eta.
/// After this returns, it's safe to accumulate grad_W_z1bar into grad_L_W_last.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage3_part1<P: ParamsTrait>(
    grad_z1_bar: &StCsF<P>,
    grad_l: &StCsF<P>,
    grad_W_last: &StFF<P>,
    grad_b_last: &RvbFA<P>,
    q_smem: &StFCs<P>,
    k_smem: &StFCs<P>,
    xk_smem: &StCsF<P>,
    xq_smem: &StCsF<P>,
    token_eta: &Tensor<Line<P::EVal>>,
    ttt_lr_eta: &Tensor<Line<P::EVal>>,
    last_eta: &RvbCsV<P>,
    last_token_eta: P::EVal,
    ttt_lr_eta_idx: usize,
    buf: &mut ReduceBuf<P::EAcc>,
    scratch1: &mut StCsF<P>,
    scratch2: &mut StCsF<P>,
    cs_cs_a: &mut StCsCs<P>,
    cs_cs_b: &mut StCsCs<P>,
    grad_grad_l_out: &mut StCsF<P>,
    grad_xk_combined_out: &mut StCsF<P>,
) -> Stage3Part1Outputs<P> {
    // --- Compute grad_grad_l from three sources ---

    // Build η^T (upper triangular) into cs_cs_a
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, cs_cs_a, ttt_lr_eta_idx, true);

    // Build attn^T (upper triangular) into cs_cs_b
    build_attn_matrix::<P>(q_smem, k_smem, cs_cs_b, true);

    // Term A: η^T @ grad_z1_bar
    let mut term_a_reg = P::rt_cs_f();
    term_a_reg.zero();
    cube::mma_AB(&mut term_a_reg, cs_cs_a, grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_a_reg, grad_grad_l_out);

    sync_cube();

    // Term B: (η·attn)^T @ grad_z1_bar = (η^T · attn^T) @ grad_z1_bar
    cs_cs_a.mul(cs_cs_b);

    sync_cube();

    let mut term_b_reg = P::rt_cs_f();
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

    let mut src12_reg = P::rt_cs_f();
    src12_reg.zero();
    cube::mma_AB(&mut src12_reg, scratch1, grad_W_last);

    sync_cube();

    cube::store_rt_to_st(&src12_reg, scratch1);

    sync_cube();

    grad_grad_l_out.sub(scratch1);

    let grad_b_last_val = grad_b_last.cast::<P::EVal>();
    scratch1.set_row(&grad_b_last_val);
    scratch1.mul_col(last_eta);

    sync_cube();

    grad_grad_l_out.sub(scratch1);

    sync_cube();

    // --- d_xk (from attn) = d_attn^T @ XQ ---
    // Compute d_attn^T directly: grad_l @ grad_z1_bar^T * η^T
    let mut d_attn_t_reg = P::rt_cs_cs();
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

    let mut d_xk_attn_reg = P::rt_cs_f();
    d_xk_attn_reg.zero();
    cube::mma_AB(&mut d_xk_attn_reg, cs_cs_b, xq_smem);

    sync_cube();

    // Store first d_xk_attn contribution to combined output
    cube::store_rt_to_st(&d_xk_attn_reg, grad_xk_combined_out);

    sync_cube();

    // --- grad_xk_mini from weight update term ---
    // grad_l_Last = grad_l @ grad_W_last^T
    let mut grad_l_last_reg = P::rt_cs_f();
    grad_l_last_reg.zero();
    cube::mma_ABt(&mut grad_l_last_reg, grad_l, grad_W_last);

    sync_cube();

    cube::store_rt_to_st(&grad_l_last_reg, scratch1);

    sync_cube();

    // grad_xk_mini = -grad_l_last * last_eta
    // Compute in scratch1 and add to combined output
    scratch1.mul_col(last_eta);
    scratch1.neg();

    sync_cube();

    // Add grad_xk_mini to the combined output (second contribution)
    grad_xk_combined_out.add(scratch1);

    sync_cube();

    // --- grad_ttt_lr_eta from weight/bias update (first part) ---
    // grad_last_eta = sum(-(grad_l_last * XK) - (grad_b_last * grad_l))
    scratch2.copy_from(scratch1); // scratch1 = grad_l_last
    scratch2.mul(xk_smem);

    sync_cube();

    let mut grad_eta_term1 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term1, buf);

    scratch2.set_row(&grad_b_last_val);
    scratch2.mul(grad_l);

    sync_cube();

    let mut grad_eta_term2 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term2, buf);

    grad_eta_term1.add(&grad_eta_term2);
    grad_eta_term1.neg();

    // Scale by last_token_eta
    grad_eta_term1.mul_scalar(P::EAcc::cast_from(last_token_eta));

    Stage3Part1Outputs::<P> { grad_eta_term1 }
}

/// Stage 3 Part 2: Compute gradients that depend on weight_init.
/// This includes grad_xq_mini (with d_xq_attn) and the remaining grad_ttt_lr_eta computation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage3_part2<P: ParamsTrait>(
    grad_z1_bar: &StCsF<P>,
    grad_l: &StCsF<P>,
    q_smem: &StFCs<P>,
    k_smem: &StFCs<P>,
    xk_smem: &StCsF<P>,
    weight_init: &StFF<P>,
    token_eta: &Tensor<Line<P::EVal>>,
    ttt_lr_eta: &Tensor<Line<P::EVal>>,
    ttt_lr_eta_idx: usize,
    grad_eta_term1: &RvbCsA<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    scratch1: &mut StCsF<P>,
    cs_cs_a: &mut StCsCs<P>,
    cs_cs_b: &mut StCsCs<P>,
    grad_xq_mini_out: &mut StCsF<P>,
) -> Stage3Outputs<P> {
    // --- grad_xq_mini = grad_z1_bar @ W_init^T ---
    let mut grad_xq_reg = P::rt_cs_f();
    grad_xq_reg.zero();
    cube::mma_ABt(&mut grad_xq_reg, grad_z1_bar, weight_init);

    sync_cube();

    cube::store_rt_to_st(&grad_xq_reg, grad_xq_mini_out);

    sync_cube();

    // --- Gradient through attn = XQ @ XK^T ---
    // d_attn = -grad_z1_bar @ grad_l^T * η (element-wise, lower triangular)

    // Build η (lower triangular) into cs_cs_a
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, cs_cs_a, ttt_lr_eta_idx, false);

    // d_attn_base = grad_z1_bar @ grad_l^T into cs_cs_b
    let mut d_attn_reg = P::rt_cs_cs();
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
    let mut d_xq_attn_reg = P::rt_cs_f();
    d_xq_attn_reg.zero();
    cube::mma_AB(&mut d_xq_attn_reg, cs_cs_b, xk_smem);

    sync_cube();

    cube::store_rt_to_st(&d_xq_attn_reg, scratch1);

    sync_cube();

    grad_xq_mini_out.add(scratch1);

    sync_cube();

    // --- grad_ttt_lr_eta from η terms in z1_bar ---
    build_attn_matrix::<P>(q_smem, k_smem, cs_cs_a, false);

    // d_eta_base = -grad_z1_bar @ grad_l^T (lower tri)
    let mut d_eta_base_reg = P::rt_cs_cs();
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
    let mut token_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(token_eta, &mut token_eta_rv, 0);

    cs_cs_b.mul_col(&token_eta_rv);

    sync_cube();

    // Sum columns to get grad_ttt_lr_eta contribution
    let mut grad_ttt_lr_eta = P::rvb_cs_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::CS, SumOp>(cs_cs_b, &mut grad_ttt_lr_eta, buf);

    grad_ttt_lr_eta.add(grad_eta_term1);

    Stage3Outputs::<P> { grad_ttt_lr_eta }
}

// =============================================================================
// Stage 2: LN+L2 second derivative
// =============================================================================

/// Outputs from stage 2.
#[derive(CubeType)]
struct Stage2Outputs<P: ParamsTrait> {
    grad_ln_weight: RvbFA<P>,
    grad_ln_bias: RvbFA<P>,
}

/// Stage 2: Compute second derivative through fused LN+L2 gradient.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage2_ln_l2<P: ParamsTrait>(
    grad_grad_l: &StCsF<P>,
    x_hat_fused: &StCsF<P>,
    std_fused: &RvbCsV<P>,
    grad_output_fused: &StCsF<P>,
    grad_x_hat_fused: &StCsF<P>,
    grad_l: &StCsF<P>,
    ln_weight: &RvbFV<P>,
    sum_gxh_xh_precomputed: &RvbCsV<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    scratch1: &mut StCsF<P>,
    scratch2: &mut StCsF<P>,
    grad_Z1_out: &mut StCsF<P>,
    grad_target_out: &mut StCsF<P>,
) -> Stage2Outputs<P> {
    let f_inv = P::EVal::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_L_grad_x_hat = (1/std) * grad_grad_l
    //                   + (1/F) * sum(-grad_grad_l / std)
    //                   + (1/F) * x_hat * sum(-grad_grad_l / std * x_hat)

    // Compute -grad_grad_l / std
    scratch1.copy_from(grad_grad_l);
    scratch1.neg();
    scratch1.div_col(std_fused);

    sync_cube();

    let mut sum1_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch1, &mut sum1_acc, buf);
    let sum1 = sum1_acc.cast::<P::EVal>();

    scratch1.mul(x_hat_fused);

    sync_cube();

    let mut sum2_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch1, &mut sum2_acc, buf);
    let sum2 = sum2_acc.cast::<P::EVal>();

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

    scratch2.add(scratch1); // scratch2 = grad_L_grad_x_hat

    sync_cube();

    // grad_L_y = ln_weight * grad_L_grad_x_hat
    // Store in grad_target_out temporarily
    grad_target_out.copy_from(scratch2);
    grad_target_out.mul_row(ln_weight);

    sync_cube();

    // grad_ln_weight = sum(grad_output_fused * grad_L_grad_x_hat + grad_L_y * x_hat)
    scratch1.copy_from(grad_output_fused);
    scratch1.mul(scratch2);

    grad_Z1_out.copy_from(grad_target_out);
    grad_Z1_out.mul(x_hat_fused);

    sync_cube();

    scratch1.add(grad_Z1_out);

    sync_cube();

    let mut grad_ln_weight = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(scratch1, &mut grad_ln_weight, buf);

    // grad_ln_bias = sum(grad_L_y)
    let mut grad_ln_bias = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        grad_target_out,
        &mut grad_ln_bias,
        buf,
    );

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

    grad_Z1_out.add(scratch1);

    sync_cube();

    // grad_L_std = -grad_L_x_hat * (x_hat / std) - grad_grad_l * (grad_l / std)
    scratch1.copy_from(grad_Z1_out);
    scratch1.mul(x_hat_fused);
    scratch1.div_col(std_fused);
    scratch1.neg();

    scratch2.copy_from(grad_grad_l);
    scratch2.mul(grad_l);
    scratch2.div_col(std_fused);

    sync_cube();

    scratch1.sub(scratch2);

    sync_cube();

    let mut sum_grad_L_std_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch1, &mut sum_grad_L_std_acc, buf);
    let sum_grad_L_std = sum_grad_L_std_acc.cast::<P::EVal>();

    // Compute final grad_Z1:
    // grad_Z1 = grad_L_x_hat / std - (1/F) * sum(grad_L_x_hat) / std + (1/F) * sum(grad_L_std) * x_hat
    // grad_Z1_out currently holds grad_L_x_hat

    let mut sum_grad_L_x_hat_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(grad_Z1_out, &mut sum_grad_L_x_hat_acc, buf);
    let sum_grad_L_x_hat = sum_grad_L_x_hat_acc.cast::<P::EVal>();

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
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage1_assemble<P: ParamsTrait>(
    grad_output: &StCsF<P>,
    grad_b_z1bar: &RvbFA<P>,
    grad_ln_weight_s4: &RvbFA<P>,
    grad_ln_bias_s4: &RvbFA<P>,
    // Stage 3 outputs
    grad_xq_mini: &StCsF<P>,
    grad_xk_combined: &StCsF<P>,
    grad_ttt_lr_eta: &RvbCsA<P>,
    // Stage 2 outputs
    grad_Z1: &StCsF<P>,
    grad_target: &StCsF<P>,
    grad_ln_weight_s2: &RvbFA<P>,
    grad_ln_bias_s2: &RvbFA<P>,
    // Inputs
    k_smem: &StFCs<P>,
    // Temp F×F tile: contains weight_init on entry, overwritten with dW_tile
    temp_f_f: &mut StFF<P>,
    // Accumulated gradients (in/out)
    grad_W_last: &mut StFF<P>,
    grad_b_last: &mut RvbFA<P>,
    grad_ln_weight_acc: &mut RvbFA<P>,
    grad_ln_bias_acc: &mut RvbFA<P>,
    // Output storage
    grads: &mut GradOutputs<P::EVal>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    buf: &mut ReduceBuf<P::EAcc>,
    scratch1: &mut StCsF<P>,
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
    // Note: temp_f_f contains weight_init at this point
    let mut grad_xk_reg = P::rt_cs_f();
    grad_xk_reg.zero();
    cube::mma_ABt(&mut grad_xk_reg, grad_Z1, temp_f_f);

    sync_cube();

    cube::store_rt_to_st(&grad_xk_reg, scratch1);

    sync_cube();

    scratch1.sub(grad_target);
    scratch1.add(grad_xk_combined);

    sync_cube();

    cube::store_st_direct(scratch1, &mut grads.grad_xk, stage_offset, 0, 0);

    // Accumulate weight gradients: grad_W = grad_W_z1bar + XK^T @ grad_Z1
    let mut dW_reg = P::rt_ff();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, k_smem, grad_Z1);

    sync_cube();

    cube::store_rt_to_st(&dW_reg, temp_f_f);

    sync_cube();

    grad_W_last.add(temp_f_f);

    sync_cube();

    // grad_b = grad_b_z1bar + sum(grad_Z1)
    let mut grad_b_z1 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(grad_Z1, &mut grad_b_z1, buf);

    grad_b_last.add(&grad_b_z1);
    grad_b_last.add(grad_b_z1bar);

    // Accumulate LN gradients
    grad_ln_weight_acc.add(grad_ln_weight_s4);
    grad_ln_weight_acc.add(grad_ln_weight_s2);
    grad_ln_bias_acc.add(grad_ln_bias_s4);
    grad_ln_bias_acc.add(grad_ln_bias_s2);

    // Store grad_ttt_lr_eta
    let grad_ttt_lr_eta_val = grad_ttt_lr_eta.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(
        &grad_ttt_lr_eta_val,
        &mut grads.grad_ttt_lr_eta,
        ttt_lr_eta_idx,
    );
}

// =============================================================================
// Main backward stage function
// =============================================================================

/// Memory layout:
/// For CS=mini_batch_size, F=head_dim:
/// - 11 CS×F tiles: scratch1, scratch2, tile_grad_z1_bar, tile_b, tile_c,
///                  tile_grad_xk_combined, tile_e, tile_f, tile_grad_Z1,
///                  tile_grad_target, grad_l_smem
/// - 2 CS×CS tiles: cs_cs_a, cs_cs_b
/// - 2 F×CS tiles: q_smem, k_smem
/// - 2 F×F tiles: temp_f_f, grad_L_W_last
///
/// Example sizes:
/// - 16×32:  11*1KB + 2*0.5KB + 2*1KB + 2*2KB = 18KB
/// - 16×64:  11*2KB + 2*0.5KB + 2*2KB + 2*8KB = 43KB (fits 64KB LDS)
/// - 32×64: 11*4KB + 2*2KB + 2*4KB + 2*8KB = 72KB (exceeds 64KB)
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_backward_stage<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_L_XQW: &Tensor<Line<P::EVal>>,
    weight_stage: &StFF<P>,
    bias_stage: &RvbFV<P>,
    grad_L_W_last: &mut StFF<P>,
    grad_L_b_last: &mut RvbFA<P>,
    grad_L_ln_weight_acc: &mut RvbFA<P>,
    grad_L_ln_bias_acc: &mut RvbFA<P>,
    grads: &mut GradOutputs<P::EVal>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    #[comptime] epsilon: f32,
) {
    let mut buf = ReduceBuf::<P::EAcc>::new();

    let mut scratch1 = P::st_cs_f();
    let mut scratch2 = P::st_cs_f();
    let mut cs_cs_a = P::st_cs_cs();
    let mut cs_cs_b = P::st_cs_cs();
    let mut tile_grad_z1_bar = P::st_cs_f();
    let mut tile_b = P::st_cs_f();
    let mut tile_c = P::st_cs_f();
    let mut tile_grad_xk_combined = P::st_cs_f();
    let mut tile_e = P::st_cs_f();
    let mut tile_f = P::st_cs_f();
    let mut tile_grad_Z1 = P::st_cs_f();
    let mut tile_grad_target = P::st_cs_f();
    let mut temp_f_f = P::st_ff();

    // =========================================================================
    // Load persistent data
    // =========================================================================

    let mut q_smem = P::st_f_cs();
    let mut k_smem = P::st_f_cs();
    cube::load_st_transpose(&saved.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&saved.xk, &mut k_smem, stage_offset, 0, 0);

    // LN weight and bias
    let base_ln = index_2d(&saved.ln_weight, CUBE_POS_Y as usize, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&recomp.ln_bias, &mut ln_bias_rv, base_ln);

    // Last eta computation
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta);

    // =========================================================================
    // Recompute grad_l (forward intermediates)
    // =========================================================================
    // z1 = xk @ W_stage + b_stage
    let mut xk_smem = P::st_cs_f();
    cube::load_st_direct(&saved.xk, &mut xk_smem, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W_stage
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, &k_smem, weight_stage);

    sync_cube();

    // Add bias_stage
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_stage.data[src_idx]);
    }
    z1_reg.add_row(&bias_reg);

    // Store z1 to grad_l_smem (will be overwritten with grad_l)
    let mut grad_l_smem = P::st_cs_f();
    cube::store_rt_to_st(&z1_reg, &mut grad_l_smem);

    sync_cube();

    // Compute target = xv - xk
    let mut v_direct_smem = P::st_cs_f();
    cube::load_st_direct(&recomp.xv, &mut v_direct_smem, stage_offset, 0, 0);

    sync_cube();

    v_direct_smem.sub(&xk_smem);

    sync_cube();

    // Recompute layer_norm_l2_grad: normalizes z1 in place to get grad_l
    // This also gives us x_hat_fused and std_fused as intermediates
    // First normalize z1 -> x_hat, get std
    let std_fused = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut grad_l_smem,
        &mut buf,
        epsilon,
    );

    // grad_l_smem now contains x_hat_fused - save it
    // We need x_hat_fused for stage 2, so copy it
    let mut x_hat_fused_smem = P::st_cs_f();
    x_hat_fused_smem.copy_from(&grad_l_smem);

    sync_cube();

    // Compute y = weight * x_hat + bias, then grad_output = y - target
    scratch1.copy_from(&grad_l_smem);
    scratch1.mul_row(&ln_weight_rv);
    scratch1.add_row(&ln_bias_rv);
    scratch1.sub(&v_direct_smem);

    sync_cube();

    // scratch1 = grad_output_fused (y - target)
    // Save for stage 2
    let mut grad_output_fused_smem = P::st_cs_f();
    grad_output_fused_smem.copy_from(&scratch1);

    // grad_x_hat = grad_output * ln_weight
    scratch1.mul_row(&ln_weight_rv);

    sync_cube();

    // scratch1 = grad_x_hat_fused - save for stage 2
    let mut grad_x_hat_fused_smem = P::st_cs_f();
    grad_x_hat_fused_smem.copy_from(&scratch1);

    sync_cube();

    // Compute grad_x from grad_x_hat -> overwrites grad_l_smem (which has x_hat) with grad_l
    compute_grad_x_from_grad_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        &scratch1,
        &mut grad_l_smem,
        &std_fused,
        &mut scratch2,
        &mut buf,
    );

    sync_cube();

    // grad_l_smem now contains the recomputed grad_l

    // =========================================================================
    // Recompute z1_bar and its layer norm intermediates
    // =========================================================================

    // z1_bar = xq @ W_stage + b_stage - eta @ grad_l - (eta * attn) @ grad_l

    // Step 1: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
    build_eta_matrix::<P>(
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
        false,
    );

    // Step 2: eta @ grad_l
    let mut eta_grad_reg = P::rt_cs_f();
    eta_grad_reg.zero();
    cube::mma_AB(&mut eta_grad_reg, &cs_cs_a, &grad_l_smem);

    sync_cube();

    // Step 3: Build (eta * attn) fused
    build_eta_attn_fused::<P>(
        &q_smem,
        &k_smem,
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
    );

    // (eta * attn) @ grad_l
    let mut eta_attn_grad_reg = P::rt_cs_f();
    eta_attn_grad_reg.zero();
    cube::mma_AB(&mut eta_attn_grad_reg, &cs_cs_a, &grad_l_smem);

    sync_cube();

    // z1_bar = xq @ W_stage
    let mut z1_bar_reg = P::rt_cs_f();
    z1_bar_reg.zero();
    cube::mma_AtB(&mut z1_bar_reg, &q_smem, weight_stage);

    sync_cube();

    // z1_bar += bias_stage
    z1_bar_reg.add_row(&bias_reg);
    // z1_bar -= eta @ grad_l
    z1_bar_reg.sub(&eta_grad_reg);
    // z1_bar -= (eta * attn) @ grad_l
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // Store z1_bar into scratch1 for layer norm
    cube::store_rt_to_st(&z1_bar_reg, &mut scratch1);

    sync_cube();

    // Normalize z1_bar to get x_hat_ln and std_ln
    let std_ln = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut scratch1,
        &mut buf,
        epsilon,
    );

    // scratch1 now contains x_hat_ln

    // =========================================================================
    // Stage 4: LN backward (using recomputed x_hat_ln, std_ln)
    // =========================================================================

    // scratch1 holds x_hat_ln; use tile_b as scratch for stage4
    // Load upstream gradient
    cube::load_st_direct(grad_L_XQW, &mut tile_c, stage_offset, 0, 0);

    sync_cube();

    let stage4_out = backward_stage4_ln::<P>(
        &tile_c,       // grad_output
        &scratch1,     // x_hat_ln
        &std_ln,
        &q_smem,
        &ln_weight_rv,
        &mut buf,
        &mut tile_b,   // scratch (was unused)
        &mut scratch2,
        &mut tile_grad_z1_bar,
        &mut temp_f_f,
    );

    // =========================================================================
    // Stage 3 Part 1: Update backward (grad_W_last-dependent parts)
    // =========================================================================

    // Reload xk, xq (reuse scratch1 -> xk, tile_c -> xq; these tiles are dead now)
    // scratch1 held x_hat_ln (consumed by stage4), tile_c held grad_output (consumed by stage4)
    cube::load_st_direct(&saved.xk, &mut scratch1, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, &mut tile_c, stage_offset, 0, 0);

    sync_cube();

    let stage3_part1_out = backward_stage3_part1::<P>(
        &tile_grad_z1_bar,
        &grad_l_smem,
        grad_L_W_last,
        grad_L_b_last,
        &q_smem,
        &k_smem,
        &scratch1,  // xk_smem
        &tile_c,    // xq_smem
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &last_eta_rv,
        last_token_eta,
        ttt_lr_eta_idx,
        &mut buf,
        &mut tile_b,   // scratch (tile_b is free)
        &mut scratch2,
        &mut cs_cs_a,
        &mut cs_cs_b,
        &mut tile_e,                // grad_grad_l output
        &mut tile_grad_xk_combined, // grad_xk_mini + grad_xk_attn combined
    );

    // tile_e now holds grad_grad_l

    sync_cube();

    grad_L_W_last.add(&temp_f_f);

    sync_cube();

    // Load weight_init into temp_f_f for stage 3 part 2
    let base_weight_init = index_2d(&saved.weight_init, CUBE_POS_X as usize, CUBE_POS_Y as usize);
    cube::load_st_direct(&saved.weight_init, &mut temp_f_f, base_weight_init, 0, 0);

    sync_cube();

    // =========================================================================
    // Stage 3 Part 2: Update backward (weight_init-dependent parts)
    // =========================================================================

    let stage3_out = backward_stage3_part2::<P>(
        &tile_grad_z1_bar,
        &grad_l_smem,
        &q_smem,
        &k_smem,
        &scratch1,  // xk_smem
        &temp_f_f,  // weight_init loaded into temp_f_f
        &saved.token_eta,
        &saved.ttt_lr_eta,
        ttt_lr_eta_idx,
        &stage3_part1_out.grad_eta_term1,
        &mut buf,
        &mut tile_b,   // scratch (tile_b is free)
        &mut cs_cs_a,
        &mut cs_cs_b,
        &mut tile_f,   // grad_xq_mini output
    );

    // Copy grad_xq_mini into tile_c (which held xq_smem, now dead)
    tile_c.copy_from(&tile_f);

    sync_cube();

    // =========================================================================
    // Stage 2: LN+L2 second derivative (using recomputed intermediates)
    // =========================================================================

    // x_hat_fused_smem, std_fused, grad_output_fused_smem, grad_x_hat_fused_smem
    // are already computed above from recomputation

    // Compute sum(grad_x_hat * x_hat) using scratch1 as temp (scratch1 was xk, now dead)
    scratch1.copy_from(&grad_x_hat_fused_smem);
    scratch1.mul(&x_hat_fused_smem);

    sync_cube();

    let mut sum_gxh_xh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(&scratch1, &mut sum_gxh_xh_acc, &mut buf);
    let sum_gxh_xh = sum_gxh_xh_acc.cast::<P::EVal>();

    let stage2_out = backward_stage2_ln_l2::<P>(
        &tile_e,              // grad_grad_l
        &x_hat_fused_smem,
        &std_fused,
        &grad_output_fused_smem,
        &grad_x_hat_fused_smem,
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

    // Reuse tile_e (was grad_grad_l, consumed by stage2) for grad_output reload
    cube::load_st_direct(grad_L_XQW, &mut tile_e, stage_offset, 0, 0);

    sync_cube();

    backward_stage1_assemble::<P>(
        &tile_e,   // grad_output
        &stage4_out.grad_b_z1bar,
        &stage4_out.grad_ln_weight,
        &stage4_out.grad_ln_bias,
        // Stage 3 outputs
        &tile_c,   // grad_xq_mini (copied from tile_f)
        &tile_grad_xk_combined,
        &stage3_out.grad_ttt_lr_eta,
        // Stage 2 outputs
        &tile_grad_Z1,
        &tile_grad_target,
        &stage2_out.grad_ln_weight,
        &stage2_out.grad_ln_bias,
        // Inputs
        &k_smem,
        // contains weight_init, will be overwritten with dW_tile
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
        &mut scratch1, // scratch for stage1
    );
}

// =============================================================================
// Kernel entry points
// =============================================================================

/// Fused TTT-Linear backward pass kernel (single mini-batch).
#[cube(launch)]
pub fn fused_ttt_backward_kernel<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_output: &Tensor<Line<P::EVal>>,
    grads: &mut GradOutputs<P::EVal>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    // For single-stage, weight_stage = weight_init, bias_stage = bias_init
    let base_weight = index_2d(&saved.weight_init, batch_idx, head_idx);
    let base_bias = index_2d(&recomp.bias, batch_idx, head_idx);

    let mut weight_stage = P::st_ff();
    cube::load_st_direct(&saved.weight_init, &mut weight_stage, base_weight, 0, 0);

    sync_cube();

    let mut bias_stage = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&recomp.bias, &mut bias_stage, base_bias);

    let mut grad_L_W_last = P::st_ff();
    grad_L_W_last.fill(P::EVal::new(0.0));

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    sync_cube();

    fused_ttt_backward_stage::<P>(
        saved,
        recomp,
        grad_output,
        &weight_stage,
        &bias_stage,
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
    cube::store_st_direct(
        &grad_L_W_last,
        &mut grads.grad_weight,
        grad_weight_base,
        0,
        0,
    );
    let grad_L_b_last_val = grad_L_b_last.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(&grad_L_b_last_val, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let base_ln = head_idx * P::F::VALUE;
    atomic_add_rv::<_, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    atomic_add_rv::<_, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
}

/// Simulate one forward stage to update weight and bias in place.
/// This recomputes the weight/bias evolution without producing output.
///
/// weight_out = weight - last_eta * XK^T @ grad_l
/// bias_out = bias - last_eta @ grad_l
#[cube]
#[allow(clippy::too_many_arguments)]
fn forward_simulate_weight_update<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    weight_smem: &mut StFF<P>,
    bias_rv: &mut RvbFV<P>,
    ln_weight_rv: &RvbFV<P>,
    ln_bias_rv: &RvbFV<P>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    #[comptime] epsilon: f32,
) {
    let mut buf = ReduceBuf::<P::EAcc>::new();
    let mut k_smem = P::st_f_cs();
    let mut q_smem = P::st_f_cs();
    let mut xk_smem = P::st_cs_f();
    let mut v_direct_smem = P::st_cs_f();
    let mut z1_smem = P::st_cs_f();
    let mut temp_smem = P::st_cs_f();

    // Load XK transposed and direct
    cube::load_st_transpose(&saved.xk, &mut k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xk, &mut xk_smem, stage_offset, 0, 0);
    cube::load_st_direct(&recomp.xv, &mut v_direct_smem, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W + b
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, &k_smem, weight_smem);

    sync_cube();

    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_rv.data[src_idx]);
    }
    z1_reg.add_row(&bias_reg);

    cube::store_rt_to_st(&z1_reg, &mut z1_smem);

    sync_cube();

    // target = xv - xk
    v_direct_smem.sub(&xk_smem);

    sync_cube();

    // grad_l = layer_norm_l2_grad(z1, target)
    layer_norm_l2_grad::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut z1_smem,
        &v_direct_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut temp_smem,
        &mut buf,
        epsilon,
    );

    sync_cube();

    // z1_smem now contains grad_l

    // Weight update: W -= last_eta * XK^T @ grad_l
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // Reload k transposed (reuse q_smem slot)
    cube::load_st_transpose(&saved.xk, &mut q_smem, stage_offset, 0, 0);

    sync_cube();

    // Scale by last_eta
    q_smem.mul_row(&last_eta_rv);

    sync_cube();

    // weight_update = scaled_xk^T @ grad_l
    let mut weight_update_reg = P::rt_ff();
    weight_update_reg.zero();
    cube::mma_AB(&mut weight_update_reg, &q_smem, &z1_smem);

    sync_cube();

    // W -= weight_update
    let mut weight_reg = P::rt_ff();
    cube::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    cube::store_rt_to_st(&weight_reg, weight_smem);

    sync_cube();

    // Bias update: b -= last_eta @ grad_l
    temp_smem.copy_from(&z1_smem);

    sync_cube();

    temp_smem.mul_col(&last_eta_rv);

    sync_cube();

    let mut bias_update_rv = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &temp_smem,
        &mut bias_update_rv,
        &mut buf,
    );

    let bias_update_val = bias_update_rv.cast::<P::EVal>();
    bias_rv.sub(&bias_update_val);
}

use super::layer_norm::layer_norm_l2_grad;

/// Fused TTT-Linear backward pass kernel (multi-stage).
///
/// For multi-stage backward, we process stages in reverse order.
/// For each backward stage k, we need the weight/bias state at stage k,
/// which requires simulating the forward weight evolution from weight_init
/// through stages 0..k-1.
#[cube(launch)]
pub fn fused_ttt_backward_kernel_multi<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_output: &Tensor<Line<P::EVal>>,
    grads: &mut GradOutputs<P::EVal>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(P::CS::VALUE);
    let head_dim = comptime!(P::F::VALUE);
    let stage_stride = mini_batch_len * head_dim;
    let num_stages = num_stages as usize;

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let base_ttt_lr_eta = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);
    let base_weight = index_2d(&saved.weight_init, batch_idx, head_idx);
    let base_bias = index_2d(&recomp.bias, batch_idx, head_idx);
    let base_ln = index_2d(&saved.ln_weight, head_idx, 0);

    // Load LN params once
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&recomp.ln_bias, &mut ln_bias_rv, base_ln);

    let mut grad_L_W_last = P::st_ff();
    grad_L_W_last.fill(P::EVal::new(0.0));

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    sync_cube();

    // Process stages in reverse order (backward through time)
    for stage in 0..num_stages {
        let stage_idx = num_stages - 1 - stage;
        let stage_offset = base_qkv + stage_idx * stage_stride;
        let ttt_lr_eta_idx = base_ttt_lr_eta + stage_idx * mini_batch_len;

        // Compute weight_stage by simulating forward from weight_init through stages 0..stage_idx-1
        let mut weight_stage = P::st_ff();
        cube::load_st_direct(&saved.weight_init, &mut weight_stage, base_weight, 0, 0);

        sync_cube();

        let mut bias_stage = P::rvb_f_v();
        cube::broadcast::load_rv_direct(&recomp.bias, &mut bias_stage, base_bias);

        // Simulate forward weight updates for stages 0..stage_idx-1
        for fwd_stage in 0..stage_idx {
            let fwd_offset = base_qkv + fwd_stage * stage_stride;
            let fwd_ttt_lr_eta_idx = base_ttt_lr_eta + fwd_stage * mini_batch_len;

            forward_simulate_weight_update::<P>(
                saved,
                recomp,
                &mut weight_stage,
                &mut bias_stage,
                &ln_weight_rv,
                &ln_bias_rv,
                fwd_offset,
                fwd_ttt_lr_eta_idx,
                epsilon,
            );

            sync_cube();
        }

        fused_ttt_backward_stage::<P>(
            saved,
            recomp,
            grad_output,
            &weight_stage,
            &bias_stage,
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
    cube::store_st_direct(
        &grad_L_W_last,
        &mut grads.grad_weight,
        grad_weight_base,
        0,
        0,
    );
    let grad_L_b_last_val = grad_L_b_last.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(&grad_L_b_last_val, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let ln_base = head_idx * P::F::VALUE;
    atomic_add_rv::<_, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, ln_base);
    atomic_add_rv::<_, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, ln_base);
}
