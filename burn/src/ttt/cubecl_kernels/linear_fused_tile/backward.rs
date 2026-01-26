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
    pub grad_ln_weight: Tensor<Line<F>>,
    pub grad_ln_bias: Tensor<Line<F>>,
}
// =============================================================================
// Stage 4: Layer norm backward (output → z1_bar)
// =============================================================================

/// Outputs from stage 4 (LN backward on output).
#[derive(CubeType)]
struct Stage4Outputs<P: ParamsTrait> {
    grad_z1_bar: St<P::E, P::CS, P::F>,
    grad_W_z1bar: St<P::E, P::F, P::F>,
    grad_b_z1bar: Rv<P::E, P::F>,
    grad_ln_weight: Rv<P::E, P::F>,
    grad_ln_bias: Rv<P::E, P::F>,
}

/// Stage 4: Compute gradients through output layer norm.
///
/// Given upstream gradient `grad_output`, computes:
/// - grad_z1_bar: gradient w.r.t. z1_bar (input to output LN)
/// - grad_W_z1bar: gradient w.r.t. W from z1_bar = XQ @ W + ...
/// - grad_b_z1bar: gradient w.r.t. b from z1_bar
/// - grad_ln_weight/bias: gradients w.r.t. LN parameters
#[cube]
fn backward_stage4_ln<P: ParamsTrait>(
    grad_output: &St<P::E, P::CS, P::F>,
    x_hat_ln: &St<P::E, P::CS, P::F>,
    std_ln: &Rv<P::E, P::CS>,
    q_smem: &St<P::E, P::F, P::CS>,
    ln_weight: &Rv<P::E, P::F>,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
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
    let mut grad_z1_bar = P::cs_f_tile();
    grad_z1_bar.copy_from(scratch1);
    grad_z1_bar.mul_scalar(f_f);
    grad_z1_bar.sub_col(&sum_gxh);

    sync_cube();

    scratch2.copy_from(x_hat_ln);
    scratch2.mul_col(&sum_gxh_xh);

    sync_cube();

    grad_z1_bar.sub(scratch2);
    grad_z1_bar.div_col(std_ln);
    grad_z1_bar.mul_scalar(f_inv);

    sync_cube();

    // grad_W_z1bar = XQ^T @ grad_z1_bar
    let mut dW_reg = P::f_f_reg();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, q_smem, &grad_z1_bar);

    sync_cube();

    let mut grad_W_z1bar = P::f_f_tile();
    cube::store_rt_to_st(&dW_reg, &mut grad_W_z1bar);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(&grad_z1_bar, &mut grad_b_z1bar, buf);

    Stage4Outputs::<P> {
        grad_z1_bar,
        grad_W_z1bar,
        grad_b_z1bar,
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 3: Update backward (dual form gradients)
// =============================================================================

/// Outputs from stage 3 (update backward).
#[derive(CubeType)]
struct Stage3Outputs<P: ParamsTrait> {
    grad_grad_l: St<P::E, P::CS, P::F>,
    grad_xq_mini: St<P::E, P::CS, P::F>,
    grad_xk_mini: St<P::E, P::CS, P::F>,
    grad_xk_attn: St<P::E, P::CS, P::F>,
    grad_ttt_lr_eta: Rv<P::E, P::CS>,
}

/// Stage 3: Compute gradients through the dual-form update equations.
///
/// This handles the complex gradients through:
/// - z1_bar = XQ @ W + b - (η + η·attn) @ grad_l
/// - W_out = W - last_η · XK^T @ grad_l
/// - b_out = b - last_η @ grad_l
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
) -> Stage3Outputs<P> {
    // --- Compute grad_grad_l from three sources ---
    // Source 1 & 2: Weight/bias updates
    // Source 3: z1_bar term

    // Build η^T (upper triangular) for source 3
    let mut eta_upper = P::cs_cs_tile();
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, &mut eta_upper, ttt_lr_eta_idx, true);

    // Build attn^T (upper triangular)
    let mut attn_upper = P::cs_cs_tile();
    build_attn_matrix::<P>(q_smem, k_smem, &mut attn_upper, true);

    // Term A: η^T @ grad_z1_bar
    let mut term_a_reg = P::cs_f_reg();
    term_a_reg.zero();
    cube::mma_AB(&mut term_a_reg, &eta_upper, grad_z1_bar);

    sync_cube();

    let mut grad_grad_l = P::cs_f_tile();
    cube::store_rt_to_st(&term_a_reg, &mut grad_grad_l);

    sync_cube();

    // Term B: (η·attn)^T @ grad_z1_bar = (η^T · attn^T) @ grad_z1_bar
    eta_upper.mul(&attn_upper);

    sync_cube();

    let mut term_b_reg = P::cs_f_reg();
    term_b_reg.zero();
    cube::mma_AB(&mut term_b_reg, &eta_upper, grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_b_reg, scratch1);

    sync_cube();

    // grad_grad_l = -(term_a + term_b)
    grad_grad_l.add(scratch1);
    grad_grad_l.neg();

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

    grad_grad_l.sub(scratch1);

    scratch1.set_row(grad_b_last);
    scratch1.mul_col(last_eta);

    sync_cube();

    grad_grad_l.sub(scratch1);

    sync_cube();

    // --- grad_xq_mini = grad_z1_bar @ W_init^T ---
    let mut grad_xq_reg = P::cs_f_reg();
    grad_xq_reg.zero();
    cube::mma_ABt(&mut grad_xq_reg, grad_z1_bar, weight_init);

    sync_cube();

    let mut grad_xq_mini = P::cs_f_tile();
    cube::store_rt_to_st(&grad_xq_reg, &mut grad_xq_mini);

    sync_cube();

    // --- Gradient through attn = XQ @ XK^T ---
    // d_attn = -grad_z1_bar @ grad_l^T * η (element-wise, lower triangular)

    // Build η (lower triangular)
    let mut eta_lower = P::cs_cs_tile();
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, &mut eta_lower, ttt_lr_eta_idx, false);

    // d_attn_base = grad_z1_bar @ grad_l^T
    let mut d_attn_reg = P::cs_cs_reg();
    d_attn_reg.zero();
    cube::mma_ABt(&mut d_attn_reg, grad_z1_bar, grad_l);

    sync_cube();

    let mut d_attn = P::cs_cs_tile();
    cube::store_rt_to_st(&d_attn_reg, &mut d_attn);

    sync_cube();

    d_attn.mul(&eta_lower);
    d_attn.neg();
    d_attn.tril();

    sync_cube();

    // d_xq (from attn) = d_attn @ XK
    let mut d_xq_attn_reg = P::cs_f_reg();
    d_xq_attn_reg.zero();
    cube::mma_AB(&mut d_xq_attn_reg, &d_attn, xk_smem);

    sync_cube();

    cube::store_rt_to_st(&d_xq_attn_reg, scratch1);

    sync_cube();

    grad_xq_mini.add(scratch1);

    sync_cube();

    // d_xk (from attn) = d_attn^T @ XQ
    // Build d_attn^T via swapped computation
    let mut d_attn_t_reg = P::cs_cs_reg();
    d_attn_t_reg.zero();
    cube::mma_ABt(&mut d_attn_t_reg, grad_l, grad_z1_bar);

    sync_cube();

    let mut d_attn_t = P::cs_cs_tile();
    cube::store_rt_to_st(&d_attn_t_reg, &mut d_attn_t);

    sync_cube();

    // Rebuild η^T for d_attn^T
    let mut eta_upper2 = P::cs_cs_tile();
    build_eta_matrix::<P>(token_eta, ttt_lr_eta, &mut eta_upper2, ttt_lr_eta_idx, true);

    d_attn_t.mul(&eta_upper2);
    d_attn_t.neg();
    d_attn_t.triu();

    sync_cube();

    let mut d_xk_attn_reg = P::cs_f_reg();
    d_xk_attn_reg.zero();
    cube::mma_AB(&mut d_xk_attn_reg, &d_attn_t, xq_smem);

    sync_cube();

    let mut grad_xk_attn = P::cs_f_tile();
    cube::store_rt_to_st(&d_xk_attn_reg, &mut grad_xk_attn);

    sync_cube();

    // --- grad_xk_mini from weight update term ---
    // grad_l_Last = grad_l @ grad_W_last^T
    let mut grad_l_last_reg = P::cs_f_reg();
    grad_l_last_reg.zero();
    cube::mma_ABt(&mut grad_l_last_reg, grad_l, grad_W_last);

    sync_cube();

    let mut grad_l_last = P::cs_f_tile();
    cube::store_rt_to_st(&grad_l_last_reg, &mut grad_l_last);

    sync_cube();

    // grad_xk_mini = -grad_l_last * last_eta
    let mut grad_xk_mini = P::cs_f_tile();
    grad_xk_mini.copy_from(&grad_l_last);
    grad_xk_mini.mul_col(last_eta);
    grad_xk_mini.neg();

    sync_cube();

    // --- grad_ttt_lr_eta from weight/bias update ---
    // grad_last_eta = sum(-(grad_l_last * XK) - (grad_b_last * grad_l))
    scratch1.copy_from(&grad_l_last);
    scratch1.mul(xk_smem);

    sync_cube();

    scratch2.copy_from(grad_l);
    scratch2.mul_row(grad_b_last);

    sync_cube();

    scratch1.add(scratch2);
    scratch1.neg();

    sync_cube();

    let mut grad_last_eta = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut grad_last_eta, buf);

    // --- Additional gradient through η from z1_bar ---
    // d_eta = -(grad_z1_bar @ grad_l^T) * (1 + attn), with tril mask

    // Build attn (lower triangular)
    let mut attn_lower = P::cs_cs_tile();
    build_attn_matrix::<P>(q_smem, k_smem, &mut attn_lower, false);

    // d_eta_base = grad_z1_bar @ grad_l^T
    let mut d_eta_base_reg = P::cs_cs_reg();
    d_eta_base_reg.zero();
    cube::mma_ABt(&mut d_eta_base_reg, grad_z1_bar, grad_l);

    sync_cube();

    let mut d_eta_base = P::cs_cs_tile();
    cube::store_rt_to_st(&d_eta_base_reg, &mut d_eta_base);

    sync_cube();

    // d_eta = -d_eta_base - d_eta_base * attn
    let mut d_eta = P::cs_cs_tile();
    d_eta.copy_from(&d_eta_base);
    d_eta.neg();

    sync_cube();

    d_eta_base.mul(&attn_lower);
    d_eta_base.neg();

    sync_cube();

    d_eta.add(&d_eta_base);

    sync_cube();

    d_eta.tril();

    sync_cube();

    // d_ttt_lr_eta[j] = sum_i(d_eta[i,j] * token_eta[i])
    let mut token_eta_full = P::cs_reg_big();
    cube::broadcast::load_rv_direct(token_eta, &mut token_eta_full, 0);

    sync_cube();

    d_eta.mul_col(&token_eta_full);

    sync_cube();

    let mut grad_ttt_lr_eta_z1bar = P::cs_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::CS, SumOp>(&d_eta, &mut grad_ttt_lr_eta_z1bar, buf);

    // Combine: grad_ttt_lr_eta = grad_last_eta * last_token_eta + grad_z1bar_contribution
    grad_last_eta.mul_scalar(last_token_eta);
    grad_last_eta.add(&grad_ttt_lr_eta_z1bar);

    Stage3Outputs::<P> {
        grad_grad_l,
        grad_xq_mini,
        grad_xk_mini,
        grad_xk_attn,
        grad_ttt_lr_eta: grad_last_eta,
    }
}

// =============================================================================
// Stage 2: LN+L2 second derivative
// =============================================================================

/// Outputs from stage 2 (LN+L2 backward).
#[derive(CubeType)]
struct Stage2Outputs<P: ParamsTrait> {
    grad_Z1: St<P::E, P::CS, P::F>,
    grad_target: St<P::E, P::CS, P::F>,
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

    // Build grad_L_grad_x_hat
    let mut grad_L_grad_x_hat = P::cs_f_tile();
    grad_L_grad_x_hat.copy_from(grad_grad_l);
    grad_L_grad_x_hat.div_col(std_fused);

    let mut s1 = sum1;
    s1.mul_scalar(f_inv);
    grad_L_grad_x_hat.add_col(&s1);

    sync_cube();

    scratch1.copy_from(x_hat_fused);
    scratch1.mul_col(&sum2);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    grad_L_grad_x_hat.add(scratch1);

    sync_cube();

    // grad_L_y = ln_weight * grad_L_grad_x_hat
    let mut grad_L_y = P::cs_f_tile();
    grad_L_y.copy_from(&grad_L_grad_x_hat);
    grad_L_y.mul_row(ln_weight);

    sync_cube();

    // grad_ln_weight = sum(grad_output_fused * grad_L_grad_x_hat + grad_L_y * x_hat)
    scratch1.copy_from(grad_output_fused);
    scratch1.mul(&grad_L_grad_x_hat);

    scratch2.copy_from(&grad_L_y);
    scratch2.mul(x_hat_fused);

    sync_cube();

    scratch1.add(scratch2);

    sync_cube();

    let mut grad_ln_weight = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(scratch1, &mut grad_ln_weight, buf);

    // grad_ln_bias = sum(grad_L_y)
    let mut grad_ln_bias = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(&grad_L_y, &mut grad_ln_bias, buf);

    // grad_L_x_hat = grad_L_y * ln_weight
    //              + (1/F) * grad_x_hat * sum2
    //              + (1/F) * sum(grad_x_hat * x_hat) * (-grad_grad_l / std)
    let mut grad_L_x_hat = P::cs_f_tile();
    grad_L_x_hat.copy_from(&grad_L_y);
    grad_L_x_hat.mul_row(ln_weight);

    scratch1.copy_from(grad_x_hat_fused);
    scratch1.mul_col(&sum2);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    grad_L_x_hat.add(scratch1);

    sync_cube();

    scratch1.copy_from(grad_grad_l);
    scratch1.neg();
    scratch1.div_col(std_fused);
    scratch1.mul_col(sum_gxh_xh_precomputed);
    scratch1.mul_scalar(f_inv);

    sync_cube();

    grad_L_x_hat.add(scratch1);

    sync_cube();

    // grad_L_std = -grad_L_x_hat * (x_hat / std) - grad_grad_l * (grad_l / std)
    scratch1.copy_from(&grad_L_x_hat);
    scratch1.mul(x_hat_fused);
    scratch1.div_col(std_fused);
    scratch1.neg();

    scratch2.copy_from(grad_grad_l);
    scratch2.mul(grad_l);
    scratch2.div_col(std_fused);

    sync_cube();

    scratch1.sub(scratch2);

    sync_cube();

    let mut sum_grad_L_std = P::cs_reg_big();
    cube::sum_rows(scratch1, &mut sum_grad_L_std, buf);

    // grad_Z1 = grad_L_x_hat / std - (1/F) * sum(grad_L_x_hat) / std + (1/F) * sum(grad_L_std) * x_hat
    let mut grad_Z1 = P::cs_f_tile();
    grad_Z1.copy_from(&grad_L_x_hat);
    grad_Z1.div_col(std_fused);

    let mut sum_grad_L_x_hat = P::cs_reg_big();
    cube::sum_rows(&grad_L_x_hat, &mut sum_grad_L_x_hat, buf);

    let mut term2 = sum_grad_L_x_hat;
    term2.div(std_fused);
    term2.mul_scalar(f_inv);
    grad_Z1.sub_col(&term2);

    sync_cube();

    scratch1.copy_from(x_hat_fused);
    let mut term3 = sum_grad_L_std;
    term3.mul_scalar(f_inv);
    scratch1.mul_col(&term3);

    sync_cube();

    grad_Z1.add(scratch1);

    sync_cube();

    // grad_target = -ln_weight * grad_L_grad_x_hat
    let mut grad_target = P::cs_f_tile();
    grad_target.copy_from(&grad_L_grad_x_hat);
    grad_target.mul_row(ln_weight);
    grad_target.neg();

    sync_cube();

    Stage2Outputs::<P> {
        grad_Z1,
        grad_target,
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 1: MatMul backward (final assembly)
// =============================================================================

/// Stage 1: Final gradient assembly and output storage.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage1_assemble<P: ParamsTrait>(
    // From upstream
    grad_output: &St<P::E, P::CS, P::F>,
    stage4: &Stage4Outputs<P>,
    stage3: &Stage3Outputs<P>,
    stage2: &Stage2Outputs<P>,
    // Inputs
    k_smem: &St<P::E, P::F, P::CS>,
    weight_init: &St<P::E, P::F, P::F>,
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
) {
    // grad_XQ = grad_output + grad_xq_mini (which already includes attn contribution)
    let mut grad_XQ = P::cs_f_tile();
    grad_XQ.copy_from(grad_output);
    grad_XQ.add(&stage3.grad_xq_mini);

    sync_cube();

    // grad_XV = grad_target (from stage 2)
    // (reconstruction_target = XV - XK, so dL/dXV = dL/d_target)

    // grad_XK = -grad_target + grad_xk_mini + grad_xk_attn + grad_Z1 @ W_init^T
    let mut grad_XK_matmul_reg = P::cs_f_reg();
    grad_XK_matmul_reg.zero();
    cube::mma_ABt(&mut grad_XK_matmul_reg, &stage2.grad_Z1, weight_init);

    sync_cube();

    let mut grad_XK_matmul = P::cs_f_tile();
    cube::store_rt_to_st(&grad_XK_matmul_reg, &mut grad_XK_matmul);

    sync_cube();

    let mut grad_XK = P::cs_f_tile();
    grad_XK.copy_from(&stage2.grad_target);
    grad_XK.neg();
    grad_XK.add(&stage3.grad_xk_mini);
    grad_XK.add(&stage3.grad_xk_attn);
    grad_XK.add(&grad_XK_matmul);

    sync_cube();

    // grad_W_init = grad_W_last + XK^T @ grad_Z1 + grad_W_z1bar
    let mut dW_init_reg = P::f_f_reg();
    dW_init_reg.zero();
    cube::mma_AB(&mut dW_init_reg, k_smem, &stage2.grad_Z1);

    sync_cube();

    let mut grad_W_init = P::f_f_tile();
    cube::store_rt_to_st(&dW_init_reg, &mut grad_W_init);

    sync_cube();

    grad_W_init.add(grad_W_last);
    grad_W_init.add(&stage4.grad_W_z1bar);

    sync_cube();

    // grad_b_init = grad_b_last + sum(grad_Z1) + grad_b_z1bar
    let mut grad_b_Z1 = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(&stage2.grad_Z1, &mut grad_b_Z1, buf);

    let mut grad_b_init = P::f_reg_big();
    grad_b_init.set(grad_b_last);
    grad_b_init.add(&grad_b_Z1);
    grad_b_init.add(&stage4.grad_b_z1bar);

    // Accumulate layer norm gradients
    grad_ln_weight_acc.add(&stage4.grad_ln_weight);
    grad_ln_weight_acc.add(&stage2.grad_ln_weight);

    grad_ln_bias_acc.add(&stage4.grad_ln_bias);
    grad_ln_bias_acc.add(&stage2.grad_ln_bias);

    // Store output gradients
    cube::store_st_direct(&grad_XQ, &mut grads.grad_xq, stage_offset, 0, 0);
    cube::store_st_direct(&stage2.grad_target, &mut grads.grad_xv, stage_offset, 0, 0);
    cube::store_st_direct(&grad_XK, &mut grads.grad_xk, stage_offset, 0, 0);

    let grad_weight_base = index_2d(&grads.grad_weight, CUBE_POS_X as usize, CUBE_POS_Y as usize);
    let grad_bias_base = index_2d(&grads.grad_bias, CUBE_POS_X as usize, CUBE_POS_Y as usize);

    cube::store_st_direct(&grad_W_init, &mut grads.grad_weight, grad_weight_base, 0, 0);
    cube::broadcast::store_rv_direct(&grad_b_init, &mut grads.grad_bias, grad_bias_base);
    cube::broadcast::store_rv_direct(
        &stage3.grad_ttt_lr_eta,
        &mut grads.grad_ttt_lr_eta,
        ttt_lr_eta_idx,
    );

    // Update accumulators for next iteration
    grad_W_last.copy_from(&grad_W_init);
    grad_b_last.set(&grad_b_init);

    sync_cube();
}

// =============================================================================
// Main backward stage function
// =============================================================================

/// Process one mini-batch stage of the TTT-Linear backward pass.
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
    #[comptime] epsilon: f32,
) {
    let mut buf = ReduceBuf::<P::E>::new();

    // Scratch tiles (reused across stages)
    let mut scratch1 = P::cs_f_tile();
    let mut scratch2 = P::cs_f_tile();

    // Load inputs into shared memory
    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();
    let mut xk_smem = P::cs_f_tile();
    let mut xq_smem = P::cs_f_tile();
    let mut grad_output_smem = P::cs_f_tile();
    let mut weight_init_smem = P::f_f_tile();

    cube::load_st_transpose(&saved.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&saved.xk, &mut k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xk, &mut xk_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, &mut xq_smem, stage_offset, 0, 0);
    cube::load_st_direct(grad_L_XQW, &mut grad_output_smem, stage_offset, 0, 0);

    let base_weight_init = index_2d(&saved.weight_init, CUBE_POS_X as usize, CUBE_POS_Y as usize);
    let base_ln = index_2d(&saved.ln_weight, CUBE_POS_Y as usize, 0);

    cube::load_st_direct(
        &saved.weight_init,
        &mut weight_init_smem,
        base_weight_init,
        0,
        0,
    );

    let mut ln_weight_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);

    // Load forward intermediates
    let mut x_hat_fused = P::cs_f_tile();
    let mut grad_output_fused = P::cs_f_tile();
    let mut grad_x_hat_fused = P::cs_f_tile();
    let mut grad_l_smem = P::cs_f_tile();
    let mut x_hat_ln = P::cs_f_tile();

    cube::load_st_direct(&fwd.x_hat_fused, &mut x_hat_fused, stage_offset, 0, 0);
    cube::load_st_direct(
        &fwd.grad_output_fused,
        &mut grad_output_fused,
        stage_offset,
        0,
        0,
    );
    cube::load_st_direct(
        &fwd.grad_x_hat_fused,
        &mut grad_x_hat_fused,
        stage_offset,
        0,
        0,
    );
    cube::load_st_direct(&fwd.grad_l_wrt_Z1, &mut grad_l_smem, stage_offset, 0, 0);
    cube::load_st_direct(&fwd.x_hat_ln, &mut x_hat_ln, stage_offset, 0, 0);

    let mut std_fused = P::cs_reg_big();
    let mut std_ln = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&fwd.std_fused, &mut std_fused, ttt_lr_eta_idx);
    cube::broadcast::load_rv_direct(&fwd.std_ln, &mut std_ln, ttt_lr_eta_idx);

    // Load last_eta = token_eta[-1] * ttt_lr_eta[:]
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta);

    sync_cube();

    // Precompute sum(grad_x_hat * x_hat) for stage 2
    scratch1.copy_from(&grad_x_hat_fused);
    scratch1.mul(&x_hat_fused);

    sync_cube();

    let mut sum_gxh_xh = P::cs_reg_big();
    cube::sum_rows(&scratch1, &mut sum_gxh_xh, &mut buf);

    // === Stage 4: LN backward ===
    let stage4_out = backward_stage4_ln::<P>(
        &grad_output_smem,
        &x_hat_ln,
        &std_ln,
        &q_smem,
        &ln_weight_rv,
        &mut buf,
        &mut scratch1,
        &mut scratch2,
    );

    // === Stage 3: Update backward ===
    let stage3_out = backward_stage3_update::<P>(
        &stage4_out.grad_z1_bar,
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
    );

    // === Stage 2: LN+L2 second derivative ===
    let stage2_out = backward_stage2_ln_l2::<P>(
        &stage3_out.grad_grad_l,
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
    );

    // === Stage 1: Final assembly ===
    backward_stage1_assemble::<P>(
        &grad_output_smem,
        &stage4_out,
        &stage3_out,
        &stage2_out,
        &k_smem,
        &weight_init_smem,
        grad_L_W_last,
        grad_L_b_last,
        grad_L_ln_weight_acc,
        grad_L_ln_bias_acc,
        grads,
        stage_offset,
        ttt_lr_eta_idx,
        &mut buf,
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

    let base_ln = index_2d(&grads.grad_ln_weight, batch_idx, head_idx);
    cube::broadcast::store_rv_direct(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    cube::broadcast::store_rv_direct(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
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

    for stage in 0..num_stages {
        let rev_stage = num_stages - 1 - stage;
        let stage_offset = base_qkv + rev_stage * mini_batch_len * P::F::VALUE;
        let ttt_lr_eta_idx = base_ttt_lr_eta + rev_stage * mini_batch_len;

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

    let base_ln = index_2d(&grads.grad_ln_weight, batch_idx, head_idx);
    cube::broadcast::store_rv_direct(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    cube::broadcast::store_rv_direct(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
}
