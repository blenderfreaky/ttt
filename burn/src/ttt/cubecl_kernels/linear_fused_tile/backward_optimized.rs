//! Optimized fused TTT-Linear backward pass kernel (reduced shared memory).
//!
//! This version reduces shared memory usage by:
//! 1. Loading forward intermediates on-demand instead of all at once
//! 2. Reusing CS×CS tiles across computations in stage 3
//! 3. Using a tile pool instead of allocating per-stage tiles

use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::forward::ForwardIntermediates;
use super::backward::{SavedTensors, GradOutputs};
use crate::ttt::cubecl_kernels::{
    FusedTttConfig,
    linear_fused_tile::helpers::{ParamsTrait, build_attn_matrix, build_eta_matrix},
};

// =============================================================================
// Stage 4: Layer norm backward (optimized)
// =============================================================================

/// Outputs from stage 4 (stored in tile pool).
#[derive(CubeType)]
struct Stage4OutputsCompact<P: ParamsTrait> {
    // grad_z1_bar stored in pool_cs_f_a
    grad_W_z1bar: St<P::E, P::F, P::F>,
    grad_b_z1bar: Rv<P::E, P::F>,
    grad_ln_weight: Rv<P::E, P::F>,
    grad_ln_bias: Rv<P::E, P::F>,
}

/// Stage 4: Compute gradients through output layer norm.
/// grad_z1_bar is written to pool_cs_f_a (passed by caller).
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage4_ln_optimized<P: ParamsTrait>(
    grad_output: &St<P::E, P::CS, P::F>,
    x_hat_ln: &St<P::E, P::CS, P::F>,
    std_ln: &Rv<P::E, P::CS>,
    q_smem: &St<P::E, P::F, P::CS>,
    ln_weight: &Rv<P::E, P::F>,
    buf: &mut ReduceBuf<P::E>,
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
    grad_z1_bar_out: &mut St<P::E, P::CS, P::F>,  // Output tile (from pool)
) -> Stage4OutputsCompact<P> {
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
    let mut dW_reg = P::f_f_reg();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, q_smem, grad_z1_bar_out);

    sync_cube();

    let mut grad_W_z1bar = P::f_f_tile();
    cube::store_rt_to_st(&dW_reg, &mut grad_W_z1bar);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_z1_bar_out, &mut grad_b_z1bar, buf);

    Stage4OutputsCompact::<P> {
        grad_W_z1bar,
        grad_b_z1bar,
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 3: Update backward (optimized - reuses CS×CS tiles)
// =============================================================================

/// Outputs from stage 3.
/// grad_grad_l is written to pool_cs_f_b
/// grad_xq_mini is written to pool_cs_f_c
/// grad_xk_mini is written to pool_cs_f_d
/// grad_xk_attn is written to scratch1
#[derive(CubeType)]
struct Stage3OutputsCompact<P: ParamsTrait> {
    grad_ttt_lr_eta: Rv<P::E, P::CS>,
}

/// Stage 3: Compute gradients through the dual-form update equations.
/// Uses only 2 CS×CS tiles (reused) instead of 9.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage3_update_optimized<P: ParamsTrait>(
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
    // Scratch tiles (2 CS×F)
    scratch1: &mut St<P::E, P::CS, P::F>,
    scratch2: &mut St<P::E, P::CS, P::F>,
    // Reusable CS×CS tiles (only 2 needed!)
    cs_cs_a: &mut St<P::E, P::CS, P::CS>,
    cs_cs_b: &mut St<P::E, P::CS, P::CS>,
    // Output tiles (from pool)
    grad_grad_l_out: &mut St<P::E, P::CS, P::F>,
    grad_xq_mini_out: &mut St<P::E, P::CS, P::F>,
    grad_xk_mini_out: &mut St<P::E, P::CS, P::F>,
    grad_xk_attn_out: &mut St<P::E, P::CS, P::F>,
) -> Stage3OutputsCompact<P> {
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
    scratch2.copy_from(scratch1);  // scratch1 = grad_l_last
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
    build_attn_matrix::<P>(q_smem, k_smem, cs_cs_a, false);  // attn (lower tri)

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

    Stage3OutputsCompact::<P> {
        grad_ttt_lr_eta,
    }
}

// =============================================================================
// Stage 2: LN+L2 second derivative (optimized)
// =============================================================================

/// Outputs from stage 2.
/// grad_Z1 written to pool tile, grad_target written to another pool tile
#[derive(CubeType)]
struct Stage2OutputsCompact<P: ParamsTrait> {
    grad_ln_weight: Rv<P::E, P::F>,
    grad_ln_bias: Rv<P::E, P::F>,
}

/// Stage 2: Compute second derivative through fused LN+L2 gradient.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage2_ln_l2_optimized<P: ParamsTrait>(
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
    // Output tiles
    grad_Z1_out: &mut St<P::E, P::CS, P::F>,
    grad_target_out: &mut St<P::E, P::CS, P::F>,
) -> Stage2OutputsCompact<P> {
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

    Stage2OutputsCompact::<P> {
        grad_ln_weight,
        grad_ln_bias,
    }
}

// =============================================================================
// Stage 1: Final assembly (optimized)
// =============================================================================

/// Stage 1: Final gradient assembly and output storage.
#[cube]
#[allow(clippy::too_many_arguments)]
fn backward_stage1_assemble_optimized<P: ParamsTrait>(
    // From upstream
    grad_output: &St<P::E, P::CS, P::F>,
    // Stage 4 outputs
    grad_z1_bar: &St<P::E, P::CS, P::F>,
    grad_W_z1bar: &St<P::E, P::F, P::F>,
    grad_b_z1bar: &Rv<P::E, P::F>,
    grad_ln_weight_s4: &Rv<P::E, P::F>,
    grad_ln_bias_s4: &Rv<P::E, P::F>,
    // Stage 3 outputs (in pool tiles)
    grad_xq_mini: &St<P::E, P::CS, P::F>,
    grad_xk_mini: &St<P::E, P::CS, P::F>,
    grad_xk_attn: &St<P::E, P::CS, P::F>,
    grad_ttt_lr_eta: &Rv<P::E, P::CS>,
    // Stage 2 outputs (in pool tiles)
    grad_Z1: &St<P::E, P::CS, P::F>,
    grad_target: &St<P::E, P::CS, P::F>,
    grad_ln_weight_s2: &Rv<P::E, P::F>,
    grad_ln_bias_s2: &Rv<P::E, P::F>,
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
    // Scratch tiles
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
    let mut grad_XK_matmul_reg = P::cs_f_reg();
    grad_XK_matmul_reg.zero();
    cube::mma_ABt(&mut grad_XK_matmul_reg, grad_Z1, weight_init);

    sync_cube();

    cube::store_rt_to_st(&grad_XK_matmul_reg, scratch1);

    sync_cube();

    scratch1.sub(grad_target);
    scratch1.add(grad_xk_mini);
    scratch1.add(grad_xk_attn);

    sync_cube();

    cube::store_st_direct(scratch1, &mut grads.grad_xk, stage_offset, 0, 0);

    // grad_W_init = grad_W_last + XK^T @ grad_Z1 + grad_W_z1bar
    let mut dW_init_reg = P::f_f_reg();
    dW_init_reg.zero();
    cube::mma_AB(&mut dW_init_reg, k_smem, grad_Z1);

    sync_cube();

    let mut grad_W_init = P::f_f_tile();
    cube::store_rt_to_st(&dW_init_reg, &mut grad_W_init);

    sync_cube();

    grad_W_init.add(grad_W_last);
    grad_W_init.add(grad_W_z1bar);

    sync_cube();

    // grad_b_init = grad_b_last + sum(grad_Z1) + grad_b_z1bar
    let mut grad_b_Z1 = P::f_reg_big();
    cube::reduce_cols::<P::E, P::CS, P::F, SumOp>(grad_Z1, &mut grad_b_Z1, buf);

    let mut grad_b_init = P::f_reg_big();
    grad_b_init.set(grad_b_last);
    grad_b_init.add(&grad_b_Z1);
    grad_b_init.add(grad_b_z1bar);

    // Accumulate layer norm gradients
    grad_ln_weight_acc.add(grad_ln_weight_s4);
    grad_ln_weight_acc.add(grad_ln_weight_s2);

    grad_ln_bias_acc.add(grad_ln_bias_s4);
    grad_ln_bias_acc.add(grad_ln_bias_s2);

    // Store weight/bias gradients
    let grad_weight_base = index_2d(&grads.grad_weight, CUBE_POS_X as usize, CUBE_POS_Y as usize);
    let grad_bias_base = index_2d(&grads.grad_bias, CUBE_POS_X as usize, CUBE_POS_Y as usize);

    cube::store_st_direct(&grad_W_init, &mut grads.grad_weight, grad_weight_base, 0, 0);
    cube::broadcast::store_rv_direct(&grad_b_init, &mut grads.grad_bias, grad_bias_base);
    cube::broadcast::store_rv_direct(
        grad_ttt_lr_eta,
        &mut grads.grad_ttt_lr_eta,
        ttt_lr_eta_idx,
    );

    // Update accumulators for next iteration
    grad_W_last.copy_from(&grad_W_init);
    grad_b_last.set(&grad_b_init);

    sync_cube();
}

// =============================================================================
// Main backward stage function (optimized)
// =============================================================================

/// Process one mini-batch stage of the TTT-Linear backward pass.
///
/// Tile allocation (for 32×32):
/// - Persistent: weight_init (F×F = 4KB)
/// - Q/K transposed: q_smem, k_smem (2 F×CS = 8KB)
/// - Scratch: scratch1, scratch2 (2 CS×F = 8KB)
/// - CS×CS pool: cs_cs_a, cs_cs_b (2 CS×CS = 8KB)
/// - CS×F pool: 10 tiles for stage outputs (10 CS×F = 40KB)
/// - grad_l_smem (CS×F = 4KB) - included in pool as pool_cs_f_a
/// Total: 64KB (exactly at LDS limit)
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_backward_stage_optimized<P: ParamsTrait>(
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
    // Tile pool allocation (total 64KB for 32×32)
    // =========================================================================

    // Scratch tiles - reused within stages (8KB)
    let mut scratch1 = P::cs_f_tile();
    let mut scratch2 = P::cs_f_tile();

    // CS×CS tiles - reused across stage 3 computations (8KB)
    let mut cs_cs_a = P::cs_cs_tile();
    let mut cs_cs_b = P::cs_cs_tile();

    // CS×F tile pool for stage outputs (40KB = 10 tiles)
    // These hold intermediate results passed between stages
    let mut pool_cs_f_a = P::cs_f_tile();  // grad_z1_bar (s4→s3→s1)
    let mut pool_cs_f_b = P::cs_f_tile();  // x_hat_fused (s2 input), then s2 scratch
    let mut pool_cs_f_c = P::cs_f_tile();  // xq_smem (s3 input), then grad_xq_mini (s3→s1)
    let mut pool_cs_f_d = P::cs_f_tile();  // grad_xk_mini (s3→s1)
    let mut pool_cs_f_e = P::cs_f_tile();  // grad_grad_l (s3→s2), then grad_output (s1 input)
    let mut pool_cs_f_f = P::cs_f_tile();  // grad_x_hat_fused (s2 input), then s2 scratch
    let mut pool_cs_f_g = P::cs_f_tile();  // grad_xk_attn (s3→s1)
    let mut pool_cs_f_h = P::cs_f_tile();  // grad_output_fused (s2 input)
    let mut pool_cs_f_i = P::cs_f_tile();  // grad_Z1 (s2→s1)
    let mut pool_cs_f_j = P::cs_f_tile();  // grad_target (s2→s1)

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
    // Load x_hat_ln and grad_output on-demand
    // =========================================================================

    // Load x_hat_ln into pool_cs_f_b (temporary)
    cube::load_st_direct(&fwd.x_hat_ln, &mut pool_cs_f_b, stage_offset, 0, 0);

    // Load grad_output into pool_cs_f_c (temporary)
    cube::load_st_direct(grad_L_XQW, &mut pool_cs_f_c, stage_offset, 0, 0);

    // Load std_ln
    let mut std_ln = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&fwd.std_ln, &mut std_ln, ttt_lr_eta_idx);

    sync_cube();

    let stage4_out = backward_stage4_ln_optimized::<P>(
        &pool_cs_f_c,      // grad_output
        &pool_cs_f_b,      // x_hat_ln
        &std_ln,
        &q_smem,
        &ln_weight_rv,
        &mut buf,
        &mut scratch1,
        &mut scratch2,
        &mut pool_cs_f_a,  // grad_z1_bar output
    );

    // pool_cs_f_b (x_hat_ln) and pool_cs_f_c (grad_output) can now be reused
    // But we need grad_output again in stage 1, so keep it or reload

    // =========================================================================
    // Stage 3: Update backward
    // Load xk_smem and xq_smem into pool tiles
    // =========================================================================

    // Load xk into pool_cs_f_b, xq into pool_cs_f_c
    cube::load_st_direct(&saved.xk, &mut pool_cs_f_b, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, &mut pool_cs_f_c, stage_offset, 0, 0);

    sync_cube();

    // Stage 3 outputs go to separate tiles (no reuse within call):
    // - grad_grad_l → pool_cs_f_e
    // - grad_xq_mini → pool_cs_f_f (temporary, will be overwritten)
    // - grad_xk_mini → pool_cs_f_d
    // - grad_xk_attn → pool_cs_f_g
    let stage3_out = backward_stage3_update_optimized::<P>(
        &pool_cs_f_a,       // grad_z1_bar (from stage 4)
        &grad_l_smem,
        grad_L_W_last,
        grad_L_b_last,
        &q_smem,
        &k_smem,
        &pool_cs_f_b,       // xk_smem
        &pool_cs_f_c,       // xq_smem
        &weight_init_smem,
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &last_eta_rv,
        last_token_eta,
        ttt_lr_eta_idx,
        &mut buf,
        &mut scratch1,      // internal scratch
        &mut scratch2,      // internal scratch
        &mut cs_cs_a,
        &mut cs_cs_b,
        &mut pool_cs_f_e,   // grad_grad_l output
        &mut pool_cs_f_f,   // grad_xq_mini output (will copy to pool_cs_f_c later)
        &mut pool_cs_f_d,   // grad_xk_mini output
        &mut pool_cs_f_g,   // grad_xk_attn output
    );

    // Copy grad_xq_mini from pool_cs_f_f to pool_cs_f_c (reuse xq tile)
    pool_cs_f_c.copy_from(&pool_cs_f_f);

    sync_cube();

    // After stage 3:
    // pool_cs_f_a = grad_z1_bar (s4→s1)
    // pool_cs_f_c = grad_xq_mini (s3→s1)
    // pool_cs_f_d = grad_xk_mini (s3→s1)
    // pool_cs_f_e = grad_grad_l (s3→s2)
    // pool_cs_f_g = grad_xk_attn (s3→s1)
    // pool_cs_f_b, pool_cs_f_f are now free

    // =========================================================================
    // Stage 2: LN+L2 second derivative
    // Load x_hat_fused, grad_output_fused, grad_x_hat_fused on-demand
    // =========================================================================

    // Load fused intermediates into free pool tiles (pool_cs_f_b, pool_cs_f_f are free)

    // Load std_fused
    let mut std_fused = P::cs_reg_big();
    cube::broadcast::load_rv_direct(&fwd.std_fused, &mut std_fused, ttt_lr_eta_idx);

    // Load x_hat_fused into pool_cs_f_b, grad_x_hat_fused into pool_cs_f_f
    cube::load_st_direct(&fwd.x_hat_fused, &mut pool_cs_f_b, stage_offset, 0, 0);
    cube::load_st_direct(&fwd.grad_x_hat_fused, &mut pool_cs_f_f, stage_offset, 0, 0);

    sync_cube();

    // Compute sum(grad_x_hat * x_hat) using scratch1 as temp
    scratch1.copy_from(&pool_cs_f_f);
    scratch1.mul(&pool_cs_f_b);

    sync_cube();

    let mut sum_gxh_xh = P::cs_reg_big();
    cube::sum_rows(&scratch1, &mut sum_gxh_xh, &mut buf);

    // Load grad_output_fused into pool_cs_f_h (dedicated tile for stage 2 input)
    cube::load_st_direct(&fwd.grad_output_fused, &mut pool_cs_f_h, stage_offset, 0, 0);

    sync_cube();

    // Call stage 2 with loaded data
    // Inputs: pool_cs_f_e (grad_grad_l), pool_cs_f_b (x_hat), pool_cs_f_f (grad_x_hat), pool_cs_f_h (grad_output)
    // Outputs go to dedicated tiles (pool_cs_f_i, pool_cs_f_j) to avoid borrow conflicts
    let stage2_out = backward_stage2_ln_l2_optimized::<P>(
        &pool_cs_f_e,       // grad_grad_l
        &pool_cs_f_b,       // x_hat_fused
        &std_fused,
        &pool_cs_f_h,       // grad_output_fused
        &pool_cs_f_f,       // grad_x_hat_fused
        &grad_l_smem,
        &ln_weight_rv,
        &sum_gxh_xh,
        &mut buf,
        &mut scratch1,      // internal scratch
        &mut scratch2,      // internal scratch
        &mut pool_cs_f_i,   // grad_Z1 output
        &mut pool_cs_f_j,   // grad_target output
    );

    // After stage 2:
    // pool_cs_f_i = grad_Z1 (s2→s1)
    // pool_cs_f_j = grad_target (s2→s1)
    // pool_cs_f_b, pool_cs_f_f, pool_cs_f_e, pool_cs_f_h are now free

    // =========================================================================
    // Stage 1: Final assembly
    // Reload grad_output for final assembly
    // =========================================================================

    // Load grad_output into pool_cs_f_e (free after stage 2)
    cube::load_st_direct(grad_L_XQW, &mut pool_cs_f_e, stage_offset, 0, 0);

    sync_cube();

    backward_stage1_assemble_optimized::<P>(
        &pool_cs_f_e,       // grad_output (reloaded)
        // Stage 4 outputs
        &pool_cs_f_a,       // grad_z1_bar
        &stage4_out.grad_W_z1bar,
        &stage4_out.grad_b_z1bar,
        &stage4_out.grad_ln_weight,
        &stage4_out.grad_ln_bias,
        // Stage 3 outputs
        &pool_cs_f_c,       // grad_xq_mini
        &pool_cs_f_d,       // grad_xk_mini
        &pool_cs_f_g,       // grad_xk_attn (from stage 3)
        &stage3_out.grad_ttt_lr_eta,
        // Stage 2 outputs
        &pool_cs_f_i,       // grad_Z1 (from stage 2)
        &pool_cs_f_j,       // grad_target (from stage 2)
        &stage2_out.grad_ln_weight,
        &stage2_out.grad_ln_bias,
        // Inputs
        &k_smem,
        &weight_init_smem,
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
        &mut pool_cs_f_b,   // scratch for stage 1 (free tile)
    );
}

// =============================================================================
// Kernel entry points (optimized versions)
// =============================================================================

/// Fused TTT-Linear backward pass kernel (single mini-batch, optimized).
#[cube(launch)]
pub fn fused_ttt_backward_kernel_optimized<P: ParamsTrait>(
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

    fused_ttt_backward_stage_optimized::<P>(
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

/// Fused TTT-Linear backward pass kernel (multi-stage, optimized).
#[cube(launch)]
pub fn fused_ttt_backward_kernel_multi_optimized<P: ParamsTrait>(
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

        fused_ttt_backward_stage_optimized::<P>(
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
