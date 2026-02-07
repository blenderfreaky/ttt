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
        ParamsTrait, RvbFA, RvbFV, StCsF, StFCs, StFF, build_attn_matrix, build_eta_attn_fused,
        build_eta_matrix,
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
    /// Atomic tensor for accumulating token_eta gradients across batches/heads.
    /// Shape: [seq_len] (unbatched, shared across batch and head dimensions)
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_token_eta: Tensor<Atomic<f32>>,
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
// Main backward stage function
// =============================================================================

/// Memory layout (optimized for shared memory reduction):
/// For CS=mini_batch_size, F=head_dim:
///
/// Internal tiles:
/// - 5 CS×F tiles: tile_grad_z1_bar, tile_grad_xk_combined, tile_e, grad_l_smem (+1 q_smem as F×CS)
/// - 2 CS×CS tiles: cs_cs_a, cs_cs_b
/// - 1 F×CS tile: q_smem
///
/// External tiles (from caller):
/// - 4 CS×F tiles: scratch1, scratch2, tile_b, tile_c
/// - 1 F×CS tile: k_smem
/// - 1 F×F tile: weight_stage
/// - 1 ReduceBuf
///
/// Weight gradient (grad_L_W_last) uses global memory instead of a 2nd F×F tile.
/// Stage 2 uses dual-purpose tiles instead of separate grad_Z1/grad_target tiles.
/// Fused LN intermediates (x_hat, grad_output, grad_x_hat) are recomputed before
/// stage 2 instead of saved, eliminating 3 CS×F tiles.
///
/// Example sizes (at f32):
/// - 16×32:  5*1KB + 2*0.5KB + 1*1KB + 4*1KB + 1*1KB + 1*4KB + 1KB = 18KB
/// - 16×64:  5*4KB + 2*1KB + 1*4KB + 4*4KB + 1*4KB + 1*16KB + 1KB = 59KB (fits 64KB LDS)
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_backward_stage<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_L_XQW: &Tensor<Line<P::EVal>>,
    // Weight at this stage. Reused as scratch F×F tile after recomputation.
    weight_stage: &mut StFF<P>,
    bias_stage: &RvbFV<P>,
    // Weight gradient accumulated via global memory (grad_L_W_last eliminated from smem)
    grad_L_b_last: &mut RvbFA<P>,
    grad_L_ln_weight_acc: &mut RvbFA<P>,
    grad_L_ln_bias_acc: &mut RvbFA<P>,
    grads: &mut GradOutputs<P::EVal>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    token_eta_base: usize,
    // Tensor + base offset for reloading the stage weight after it's been overwritten.
    // For single-stage: saved.weight_init with batch/head offset.
    // For multi-stage: weight_stage_buf with batch/head offset.
    weight_z1bar_tensor: &Tensor<Line<P::EVal>>,
    weight_z1bar_base: usize,
    // Global memory offset for weight gradient accumulation
    grad_weight_base: usize,
    // External tiles (allocated by caller, shared with forward_simulate in multi-stage)
    scratch1: &mut StCsF<P>,
    scratch2: &mut StCsF<P>,
    k_smem: &mut StFCs<P>,
    tile_b: &mut StCsF<P>,
    tile_c: &mut StCsF<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    #[comptime] epsilon: f32,
) {
    let mut cs_cs_a = P::st_cs_cs();
    let mut cs_cs_b = P::st_cs_cs();
    let mut tile_grad_z1_bar = P::st_cs_f();
    let mut tile_grad_xk_combined = P::st_cs_f();
    let mut tile_e = P::st_cs_f();
    // REMOVED: tile_grad_Z1, tile_grad_target (folded into stage2 dual-purpose tiles)
    // REMOVED: xk_smem (use tile_b temporarily)
    // REMOVED: x_hat_fused_smem, grad_output_fused_smem, grad_x_hat_fused_smem (recomputed before stage 2)

    // =========================================================================
    // Load persistent data
    // =========================================================================

    let mut q_smem = P::st_f_cs();
    // k_smem passed from caller (shared with forward_simulate)
    cube::load_st_transpose(&saved.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);

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

    // Bias as register tile row (needed for z1 and z1_bar computations)
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_stage.data[src_idx]);
    }

    // =========================================================================
    // Recompute grad_l (without saving fused intermediates)
    // =========================================================================

    // Load xk direct into tile_b (instead of separate xk_smem allocation)
    cube::load_st_direct(&saved.xk, tile_b, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W_stage
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, k_smem, weight_stage);

    sync_cube();

    // z1 += bias_stage
    z1_reg.add_row(&bias_reg);

    // Store z1 to grad_l_smem (will be overwritten with grad_l)
    let mut grad_l_smem = P::st_cs_f();
    cube::store_rt_to_st(&z1_reg, &mut grad_l_smem);

    sync_cube();

    // Compute target = xv - xk (using tile_b for xk, tile_e as temp for xv)
    cube::load_st_direct(&recomp.xv, &mut tile_e, stage_offset, 0, 0);

    sync_cube();

    tile_e.sub(tile_b); // tile_e = target = xv - xk

    sync_cube();

    // Normalize z1 -> x_hat (in place in grad_l_smem), get std
    let _std_fused_initial = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut grad_l_smem,
        buf,
        epsilon,
    );
    // grad_l_smem now has x_hat. DON'T save x_hat — will recompute before stage 2.

    // Compute y = ln_weight * x_hat + ln_bias, then grad_output = y - target
    scratch1.copy_from(&grad_l_smem);
    scratch1.mul_row(&ln_weight_rv);
    scratch1.add_row(&ln_bias_rv);
    scratch1.sub(&tile_e);

    sync_cube();

    // DON'T save grad_output — will recompute before stage 2.
    // grad_x_hat = grad_output * ln_weight
    scratch1.mul_row(&ln_weight_rv);

    sync_cube();

    // DON'T save grad_x_hat — will recompute before stage 2.
    // Compute grad_x from grad_x_hat -> overwrites grad_l_smem (x_hat -> grad_l)
    compute_grad_x_from_grad_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        scratch1,
        &mut grad_l_smem,
        &_std_fused_initial,
        scratch2,
        buf,
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
        k_smem,
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
    cube::store_rt_to_st(&z1_bar_reg, scratch1);

    sync_cube();

    // Normalize z1_bar to get x_hat_ln and std_ln
    let std_ln = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        scratch1,
        buf,
        epsilon,
    );

    // scratch1 now contains x_hat_ln


    // =========================================================================
    // Stage 4: LN backward (inlined from backward_stage4_ln)
    // =========================================================================

    // Load upstream gradient
    cube::load_st_direct(grad_L_XQW, tile_c, stage_offset, 0, 0);

    sync_cube();

    let f_f = P::EVal::cast_from(P::F::VALUE as f32);
    let f_inv = P::EVal::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_ln_weight_s4 = sum(grad_output * x_hat_ln)
    tile_b.copy_from(tile_c);
    tile_b.mul(scratch1);

    sync_cube();

    let mut grad_ln_weight_s4 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(tile_b, &mut grad_ln_weight_s4, buf);

    // grad_ln_bias_s4 = sum(grad_output)
    let mut grad_ln_bias_s4 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(tile_c, &mut grad_ln_bias_s4, buf);

    // grad_x_hat = grad_output * ln_weight
    tile_b.copy_from(tile_c);
    tile_b.mul_row(&ln_weight_rv);

    sync_cube();

    // sum(grad_x_hat) per row
    let mut sum_gxh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_gxh_acc, buf);
    let sum_gxh = sum_gxh_acc.cast::<P::EVal>();

    // sum(grad_x_hat * x_hat) per row
    scratch2.copy_from(tile_b);
    scratch2.mul(scratch1);

    sync_cube();

    let mut sum_gxh_xh_s4_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut sum_gxh_xh_s4_acc, buf);
    let sum_gxh_xh_s4 = sum_gxh_xh_s4_acc.cast::<P::EVal>();

    // grad_z1_bar = (grad_x_hat * F - sum_gxh - x_hat * sum_gxh_xh) / (std * F)
    tile_grad_z1_bar.copy_from(tile_b);
    tile_grad_z1_bar.mul_scalar(f_f);
    tile_grad_z1_bar.sub_col(&sum_gxh);

    sync_cube();

    scratch2.copy_from(scratch1);
    scratch2.mul_col(&sum_gxh_xh_s4);

    sync_cube();

    tile_grad_z1_bar.sub(scratch2);
    tile_grad_z1_bar.div_col(&std_ln);
    tile_grad_z1_bar.mul_scalar(f_inv);

    sync_cube();

    // grad_W_z1bar = XQ^T @ grad_z1_bar
    let mut dW_reg = P::rt_ff();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, &q_smem, &tile_grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&dW_reg, weight_stage);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &tile_grad_z1_bar,
        &mut grad_b_z1bar,
        buf,
    );

    // =========================================================================
    // Global memory accumulation: save dW_z1bar, load accumulated grad_W
    // =========================================================================

    // Save dW_z1bar from weight_stage to register before overwriting
    let mut dw_z1bar_rt = P::rt_ff();
    cube::load_rt_from_st(weight_stage, &mut dw_z1bar_rt);

    sync_cube();

    // Load current accumulated weight gradient from global memory
    cube::load_st_direct(&grads.grad_weight, weight_stage, grad_weight_base, 0, 0);

    sync_cube();

    // weight_stage now serves as grad_W_last for stage 3 part 1 (read-only)

    // =========================================================================
    // Stage 3 Part 1: Update backward (inlined from backward_stage3_part1)
    // =========================================================================

    // Reload xk, xq (scratch1 and tile_c are dead after stage 4)
    cube::load_st_direct(&saved.xk, scratch1, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, tile_c, stage_offset, 0, 0);

    sync_cube();

    // Rename: scratch1 now holds XK data, tile_c now holds XQ data
    let xk_smem = scratch1;
    let xq_smem = tile_c;

    // --- Compute grad_grad_l from three sources ---

    // Build η^T (upper triangular) into cs_cs_a
    build_eta_matrix::<P>(&saved.token_eta, &saved.ttt_lr_eta, &mut cs_cs_a, ttt_lr_eta_idx, true);

    // Build attn^T (upper triangular) into cs_cs_b
    build_attn_matrix::<P>(&q_smem, k_smem, &mut cs_cs_b, true);

    // Term A: η^T @ grad_z1_bar
    let mut term_a_reg = P::rt_cs_f();
    term_a_reg.zero();
    cube::mma_AB(&mut term_a_reg, &cs_cs_a, &tile_grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_a_reg, &mut tile_e);

    sync_cube();

    // Term B: (η·attn)^T @ grad_z1_bar = (η^T · attn^T) @ grad_z1_bar
    cs_cs_a.mul(&cs_cs_b);

    sync_cube();

    let mut term_b_reg = P::rt_cs_f();
    term_b_reg.zero();
    cube::mma_AB(&mut term_b_reg, &cs_cs_a, &tile_grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&term_b_reg, tile_b);

    sync_cube();

    // grad_grad_l = -(term_a + term_b) → tile_e
    tile_e.add(tile_b);
    tile_e.neg();

    sync_cube();

    // Sources 1 & 2: -(last_η · XK) @ grad_W_last - last_η · grad_b_last
    tile_b.copy_from(xk_smem);
    tile_b.mul_col(&last_eta_rv);

    sync_cube();

    let mut src12_reg = P::rt_cs_f();
    src12_reg.zero();
    cube::mma_AB(&mut src12_reg, tile_b, weight_stage);

    sync_cube();

    cube::store_rt_to_st(&src12_reg, tile_b);

    sync_cube();

    tile_e.sub(tile_b);

    let grad_b_last_val = grad_L_b_last.cast::<P::EVal>();
    tile_b.set_row(&grad_b_last_val);
    tile_b.mul_col(&last_eta_rv);

    sync_cube();

    tile_e.sub(tile_b);

    sync_cube();

    // --- d_xk (from attn) = d_attn^T @ XQ ---
    // Compute d_attn^T directly: grad_l @ grad_z1_bar^T * η^T
    let mut d_attn_t_reg = P::rt_cs_cs();
    d_attn_t_reg.zero();
    cube::mma_ABt(&mut d_attn_t_reg, &grad_l_smem, &tile_grad_z1_bar);

    sync_cube();

    cube::store_rt_to_st(&d_attn_t_reg, &mut cs_cs_b);

    sync_cube();

    // Rebuild η^T into cs_cs_a
    build_eta_matrix::<P>(&saved.token_eta, &saved.ttt_lr_eta, &mut cs_cs_a, ttt_lr_eta_idx, true);

    cs_cs_b.mul(&cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.triu();

    sync_cube();

    let mut d_xk_attn_reg = P::rt_cs_f();
    d_xk_attn_reg.zero();
    cube::mma_AB(&mut d_xk_attn_reg, &cs_cs_b, xq_smem);

    sync_cube();

    // Store first d_xk_attn contribution to combined output
    cube::store_rt_to_st(&d_xk_attn_reg, &mut tile_grad_xk_combined);

    sync_cube();

    // --- grad_xk_mini from weight update term ---
    // grad_l_Last = grad_l @ grad_W_last^T
    let mut grad_l_last_reg = P::rt_cs_f();
    grad_l_last_reg.zero();
    cube::mma_ABt(&mut grad_l_last_reg, &grad_l_smem, weight_stage);

    sync_cube();

    cube::store_rt_to_st(&grad_l_last_reg, tile_b);

    sync_cube();

    // --- grad_ttt_lr_eta from weight/bias update (first part) ---
    // Must compute BEFORE modifying tile_b for grad_xk_mini (tile_b = grad_l_last here)
    // grad_last_eta = sum(-(grad_l_last * XK) - (grad_b_last * grad_l))
    scratch2.copy_from(tile_b); // tile_b = grad_l_last (unmodified)
    scratch2.mul(xk_smem);

    sync_cube();

    let mut grad_eta_term1 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term1, buf);

    scratch2.set_row(&grad_b_last_val);
    scratch2.mul(&grad_l_smem);

    sync_cube();

    let mut grad_eta_term2 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term2, buf);

    grad_eta_term1.add(&grad_eta_term2);
    grad_eta_term1.neg();

    // d_last_eta = -(sum_rows(grad_l_last * XK) + sum_rows(grad_b_last * grad_l))
    // Save before scaling by last_token_eta (needed for grad_token_eta[CS-1])
    let mut d_last_eta = P::rvb_cs_a();
    d_last_eta.copy_from(&grad_eta_term1);

    // Scale by last_token_eta for grad_ttt_lr_eta contribution
    grad_eta_term1.mul_scalar(P::EAcc::cast_from(last_token_eta));

    // Now modify tile_b for grad_xk_mini (safe: eta terms already computed)
    // grad_xk_mini = -grad_l_last * last_eta
    tile_b.mul_col(&last_eta_rv);
    tile_b.neg();

    sync_cube();

    // Add grad_xk_mini to the combined output (second contribution)
    tile_grad_xk_combined.add(tile_b);

    sync_cube();

    // =========================================================================
    // Accumulate dW_z1bar into the global weight gradient
    // =========================================================================

    // tile_e now holds grad_grad_l

    sync_cube();

    // weight_stage still has the old accumulated value (read-only in stage3_part1)
    let mut acc_rt = P::rt_ff();
    cube::load_rt_from_st(weight_stage, &mut acc_rt);
    acc_rt.add(&dw_z1bar_rt);
    cube::store_rt_to_st(&acc_rt, weight_stage);

    sync_cube();

    cube::store_st_direct(weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // =========================================================================
    // Reload W_stage for stage 3 part 2 and subsequent recomputation
    // =========================================================================

    cube::load_st_direct(weight_z1bar_tensor, weight_stage, weight_z1bar_base, 0, 0);

    sync_cube();

    // =========================================================================
    // Stage 3 Part 2: Update backward (inlined from backward_stage3_part2)
    // =========================================================================

    // --- grad_xq_mini = grad_z1_bar @ W_init^T ---
    let mut grad_xq_reg = P::rt_cs_f();
    grad_xq_reg.zero();
    cube::mma_ABt(&mut grad_xq_reg, &tile_grad_z1_bar, weight_stage);

    sync_cube();

    cube::store_rt_to_st(&grad_xq_reg, xq_smem);

    sync_cube();

    // --- Gradient through attn = XQ @ XK^T ---
    // d_attn = -grad_z1_bar @ grad_l^T * η (element-wise, lower triangular)

    // Build η (lower triangular) into cs_cs_a
    build_eta_matrix::<P>(&saved.token_eta, &saved.ttt_lr_eta, &mut cs_cs_a, ttt_lr_eta_idx, false);

    // d_attn_base = grad_z1_bar @ grad_l^T into cs_cs_b
    let mut d_attn_reg = P::rt_cs_cs();
    d_attn_reg.zero();
    cube::mma_ABt(&mut d_attn_reg, &tile_grad_z1_bar, &grad_l_smem);

    sync_cube();

    cube::store_rt_to_st(&d_attn_reg, &mut cs_cs_b);

    sync_cube();

    cs_cs_b.mul(&cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.tril();

    sync_cube();

    // d_xq (from attn) = d_attn @ XK
    let mut d_xq_attn_reg = P::rt_cs_f();
    d_xq_attn_reg.zero();
    cube::mma_AB(&mut d_xq_attn_reg, &cs_cs_b, xk_smem);

    sync_cube();

    cube::store_rt_to_st(&d_xq_attn_reg, tile_b);

    sync_cube();

    xq_smem.add(tile_b);

    sync_cube();

    // --- grad_ttt_lr_eta from η terms in z1_bar ---
    build_attn_matrix::<P>(&q_smem, k_smem, &mut cs_cs_a, false);

    // d_eta_base = -grad_z1_bar @ grad_l^T (lower tri)
    let mut d_eta_base_reg = P::rt_cs_cs();
    d_eta_base_reg.zero();
    cube::mma_ABt(&mut d_eta_base_reg, &tile_grad_z1_bar, &grad_l_smem);

    sync_cube();

    cube::store_rt_to_st(&d_eta_base_reg, &mut cs_cs_b);

    sync_cube();

    cs_cs_b.neg();
    cs_cs_b.tril();

    // d_eta = d_eta_base + d_eta_base * attn
    cs_cs_a.mul(&cs_cs_b);

    sync_cube();

    cs_cs_b.add(&cs_cs_a);

    sync_cube();

    // --- grad_token_eta from η terms in z1_bar + weight/bias update ---
    // cs_cs_b = d_eta, cs_cs_a is dead (was d_eta_base * attn, consumed)
    // grad_token_eta[i] = Σ_j(d_eta[i,j] * ttt_lr_eta[j])
    // Plus: grad_token_eta[CS-1] += Σ_j(d_last_eta[j] * ttt_lr_eta[j])
    cs_cs_a.copy_from(&cs_cs_b);

    sync_cube();

    // Load ttt_lr_eta for row-wise multiplication
    let mut ttt_lr_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut ttt_lr_eta_rv, ttt_lr_eta_idx);

    // Multiply each column j by ttt_lr_eta[j]
    cs_cs_a.mul_row(&ttt_lr_eta_rv);

    sync_cube();

    // Sum rows: grad_token_eta[i] = Σ_j(d_eta[i,j] * ttt_lr_eta[j])
    let mut grad_token_eta = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::CS>(&cs_cs_a, &mut grad_token_eta, buf);

    // Add weight/bias update contribution to last element:
    // grad_token_eta[CS-1] += dot(d_last_eta, ttt_lr_eta)
    let ttt_lr_eta_acc = ttt_lr_eta_rv.cast::<P::EAcc>();
    let mut d_last_eta_scaled = P::rvb_cs_a();
    d_last_eta_scaled.copy_from(&d_last_eta);
    d_last_eta_scaled.mul(&ttt_lr_eta_acc);
    // Sum all elements of d_last_eta_scaled to get the scalar dot product
    let mut dot_sum = P::EAcc::new(0.0);
    #[unroll]
    for line_idx in 0..P::CS::LINES {
        let line = d_last_eta_scaled.data[line_idx];
        #[unroll]
        for elem_idx in 0..LINE_SIZE {
            dot_sum += line[elem_idx];
        }
    }
    // Add to last element of grad_token_eta
    let gte_last_line = comptime!((P::CS::VALUE - 1) / LINE_SIZE);
    let gte_last_elem = comptime!((P::CS::VALUE - 1) % LINE_SIZE);
    let mut gte_line = grad_token_eta.data[gte_last_line];
    gte_line[gte_last_elem] += dot_sum;
    grad_token_eta.data[gte_last_line] = gte_line;

    // --- grad_ttt_lr_eta from η terms in z1_bar ---
    // grad_ttt_lr_eta += sum_cols(d_eta * token_eta)
    // d_eta[i,j] contributes to grad_ttt_lr_eta[j] with weight token_eta[i]
    // cs_cs_b still holds d_eta (unmodified)
    let mut token_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.token_eta, &mut token_eta_rv, 0);

    cs_cs_b.mul_col(&token_eta_rv);

    sync_cube();

    // Sum columns to get grad_ttt_lr_eta contribution
    let mut grad_ttt_lr_eta = P::rvb_cs_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::CS, SumOp>(&cs_cs_b, &mut grad_ttt_lr_eta, buf);

    grad_ttt_lr_eta.add(&grad_eta_term1);

    // --- End stage 3 ---

    // Rename: xq_smem now holds grad_xq_mini output
    let grad_xq_mini = xq_smem;
    // Rename back: xk_smem is no longer needed, restore scratch1 identity
    let scratch1 = xk_smem;

    // =========================================================================
    // Stage 2: Recompute fused LN intermediates, then backward
    // =========================================================================
    // weight_stage still has W_stage (read-only in stage3_part2). Reuse for z1 computation.

    // Store grad_xk_combined to global memory (frees tile for use in stage 2).
    // Will be reloaded before stage 1.
    cube::store_st_direct(&tile_grad_xk_combined, &mut grads.grad_xk, stage_offset, 0, 0);

    // Load xk direct → tile_b, xv → scratch1
    cube::load_st_direct(&saved.xk, tile_b, stage_offset, 0, 0);
    cube::load_st_direct(&recomp.xv, scratch1, stage_offset, 0, 0);

    sync_cube();

    // target = xv - xk
    scratch1.sub(tile_b);

    sync_cube();

    // z1 = xk @ W_stage + bias (k_smem has XK^T, weight_stage has W_stage)
    let mut z1_recomp_reg = P::rt_cs_f();
    z1_recomp_reg.zero();
    cube::mma_AtB(&mut z1_recomp_reg, k_smem, weight_stage);

    sync_cube();

    z1_recomp_reg.add_row(&bias_reg);

    // Store z1 to tile_grad_z1_bar (will become x_hat_fused)
    cube::store_rt_to_st(&z1_recomp_reg, &mut tile_grad_z1_bar);

    sync_cube();

    // Normalize z1 → x_hat, std_fused
    let std_fused = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut tile_grad_z1_bar,
        buf,
        epsilon,
    );
    // tile_grad_z1_bar now has x_hat_fused

    // Compute grad_output = ln_weight * x_hat + ln_bias - target
    scratch2.copy_from(&tile_grad_z1_bar);
    scratch2.mul_row(&ln_weight_rv);
    scratch2.add_row(&ln_bias_rv);
    scratch2.sub(scratch1); // scratch2 = grad_output (scratch1 had target)

    sync_cube();

    // Move grad_output to scratch1
    scratch1.copy_from(scratch2);

    sync_cube();

    // Compute grad_x_hat = grad_output * ln_weight → scratch2
    scratch2.mul_row(&ln_weight_rv);

    sync_cube();

    // Compute sum_gxh_xh = sum_rows(grad_x_hat * x_hat) using tile_b as temp
    tile_b.copy_from(scratch2);
    tile_b.mul(&tile_grad_z1_bar);

    sync_cube();

    let mut sum_gxh_xh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_gxh_xh_acc, buf);
    let sum_gxh_xh = sum_gxh_xh_acc.cast::<P::EVal>();

    // Rename: scratch1/scratch2 take on their stage 2 dual-purpose identities
    let grad_out_then_Z1 = scratch1;
    let grad_xhat_then_target = scratch2;

    // =========================================================================
    // Stage 2: LN+L2 second derivative (inlined from backward_stage2_ln_l2)
    // =========================================================================
    // tile_e = grad_grad_l (read-only)
    // tile_grad_z1_bar = x_hat_fused (read-only)
    // grad_out_then_Z1: grad_output on entry, grad_Z1 on exit
    // grad_xhat_then_target: grad_x_hat on entry, grad_target on exit
    // grad_l_smem: grad_l on entry, scratch after phase 0
    // tile_b = scratch, tile_grad_xk_combined = scratch

    // === Phase 0: Precompute grad_l-dependent term (frees grad_l tile for scratch) ===
    // term7_partial = sum_rows(grad_grad_l * grad_l / std)
    tile_b.copy_from(&tile_e);
    tile_b.mul(&grad_l_smem);
    tile_b.div_col(&std_fused);

    sync_cube();

    let mut term7_partial_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut term7_partial_acc, buf);
    let term7_partial = term7_partial_acc.cast::<P::EVal>();
    // grad_l tile is now consumed; grad_l_smem becomes general scratch

    // === Phase 1: Compute sum1, sum2 ===
    tile_b.copy_from(&tile_e);
    tile_b.neg();
    tile_b.div_col(&std_fused);

    sync_cube();

    let mut sum1_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum1_acc, buf);
    let sum1 = sum1_acc.cast::<P::EVal>();

    tile_b.mul(&tile_grad_z1_bar);

    sync_cube();

    let mut sum2_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum2_acc, buf);
    let sum2 = sum2_acc.cast::<P::EVal>();

    // === Phase 2: Compute grad_L_gxh in tile_grad_xk_combined (preserved for phase 7) ===
    // grad_L_gxh = grad_grad_l/std + (1/F)*sum1 + (1/F)*x_hat*sum2
    tile_grad_xk_combined.copy_from(&tile_e);
    tile_grad_xk_combined.div_col(&std_fused);

    let mut s1 = sum1;
    s1.mul_scalar(f_inv);
    tile_grad_xk_combined.add_col(&s1);

    sync_cube();

    grad_l_smem.copy_from(&tile_grad_z1_bar);
    grad_l_smem.mul_col(&sum2);
    grad_l_smem.mul_scalar(f_inv);

    sync_cube();

    tile_grad_xk_combined.add(&grad_l_smem); // tile_grad_xk_combined = grad_L_gxh

    sync_cube();

    // === Phase 3: Consume grad_output, compute grad_ln_weight + grad_ln_bias ===
    // grad_L_y = ln_weight * grad_L_gxh → store in grad_l_smem
    grad_l_smem.copy_from(&tile_grad_xk_combined);
    grad_l_smem.mul_row(&ln_weight_rv); // grad_l_smem = grad_L_y

    sync_cube();

    // grad_ln_weight = reduce_cols(grad_output * grad_L_gxh + grad_L_y * x_hat)
    // Part 1: grad_out_then_Z1 *= grad_L_gxh (CONSUMES grad_output)
    grad_out_then_Z1.mul(&tile_grad_xk_combined);

    // Part 2: tile_b = grad_L_y * x_hat
    tile_b.copy_from(&grad_l_smem);
    tile_b.mul(&tile_grad_z1_bar);

    sync_cube();

    grad_out_then_Z1.add(tile_b);

    sync_cube();

    let mut grad_ln_weight_s2 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        grad_out_then_Z1,
        &mut grad_ln_weight_s2,
        buf,
    );

    // grad_ln_bias = reduce_cols(grad_L_y)
    let mut grad_ln_bias_s2 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &grad_l_smem,
        &mut grad_ln_bias_s2,
        buf,
    );

    // === Phase 4: Compute grad_L_x_hat in grad_out_then_Z1 (becomes grad_Z1) ===
    // grad_L_x_hat = grad_L_y*ln_weight + (1/F)*grad_x_hat*sum2 + (1/F)*sum_gxh_xh*(-grad_grad_l/std)

    // Term 1: grad_out_then_Z1 = grad_L_y * ln_weight
    grad_out_then_Z1.copy_from(&grad_l_smem);
    grad_out_then_Z1.mul_row(&ln_weight_rv);

    // Term 2: tile_b = (1/F)*grad_x_hat*sum2 (grad_xhat_then_target still holds grad_x_hat data)
    tile_b.copy_from(grad_xhat_then_target);
    tile_b.mul_col(&sum2);
    tile_b.mul_scalar(f_inv);

    sync_cube();

    grad_out_then_Z1.add(tile_b);

    sync_cube();

    // Term 3: tile_b = (1/F)*sum_gxh_xh*(-grad_grad_l/std)
    tile_b.copy_from(&tile_e);
    tile_b.neg();
    tile_b.div_col(&std_fused);
    tile_b.mul_col(&sum_gxh_xh);
    tile_b.mul_scalar(f_inv);

    sync_cube();

    grad_out_then_Z1.add(tile_b); // grad_out_then_Z1 = grad_L_x_hat

    sync_cube();

    // === Phase 5: Compute grad_L_std ===
    // sum_grad_L_std = sum_rows(-grad_L_x_hat*x_hat/std) - term7_partial
    tile_b.copy_from(grad_out_then_Z1);
    tile_b.mul(&tile_grad_z1_bar);
    tile_b.div_col(&std_fused);
    tile_b.neg();

    sync_cube();

    let mut sum_grad_L_std_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_grad_L_std_acc, buf);
    let mut sum_grad_L_std = sum_grad_L_std_acc.cast::<P::EVal>();
    sum_grad_L_std.sub(&term7_partial);

    // === Phase 6: Compute final grad_Z1 in grad_out_then_Z1 ===
    // grad_out_then_Z1 currently holds grad_L_x_hat
    let mut sum_grad_L_x_hat_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(
        grad_out_then_Z1,
        &mut sum_grad_L_x_hat_acc,
        buf,
    );
    let sum_grad_L_x_hat = sum_grad_L_x_hat_acc.cast::<P::EVal>();

    grad_out_then_Z1.div_col(&std_fused);

    let mut term2 = sum_grad_L_x_hat;
    term2.div(&std_fused);
    term2.mul_scalar(f_inv);
    grad_out_then_Z1.sub_col(&term2);

    sync_cube();

    tile_b.copy_from(&tile_grad_z1_bar);
    let mut term3 = sum_grad_L_std;
    term3.mul_scalar(f_inv);
    tile_b.mul_col(&term3);

    sync_cube();

    grad_out_then_Z1.add(tile_b); // grad_out_then_Z1 = grad_Z1

    sync_cube();

    // === Phase 7: Compute grad_target in grad_xhat_then_target ===
    // grad_target = -ln_weight * grad_L_gxh (tile_grad_xk_combined still holds grad_L_gxh)
    grad_xhat_then_target.copy_from(&tile_grad_xk_combined);
    grad_xhat_then_target.mul_row(&ln_weight_rv);
    grad_xhat_then_target.neg(); // grad_xhat_then_target = grad_target

    sync_cube();

    // --- End stage 2 ---

    // Rename: dual-purpose tiles to their final identities
    let grad_Z1 = grad_out_then_Z1;
    let grad_target = grad_xhat_then_target;

    // =========================================================================
    // Stage 1: Final assembly (inlined from backward_stage1_assemble)
    // =========================================================================

    // Reload upstream gradient → tile_e (was grad_grad_l, consumed by stage 2)
    cube::load_st_direct(grad_L_XQW, &mut tile_e, stage_offset, 0, 0);

    // Reload grad_xk_combined from global (stored before stage 2 recomputation)
    cube::load_st_direct(&grads.grad_xk, &mut tile_grad_xk_combined, stage_offset, 0, 0);

    sync_cube();

    // grad_XQ = grad_output + grad_xq_mini
    grad_l_smem.copy_from(&tile_e);
    grad_l_smem.add(grad_xq_mini);

    sync_cube();

    // Store grad_XQ
    cube::store_st_direct(&grad_l_smem, &mut grads.grad_xq, stage_offset, 0, 0);

    // grad_XV = grad_target
    cube::store_st_direct(grad_target, &mut grads.grad_xv, stage_offset, 0, 0);

    // grad_XK = -grad_target + grad_xk_combined + grad_Z1 @ W_stage^T
    // weight_stage still has W_stage
    let mut grad_xk_reg = P::rt_cs_f();
    grad_xk_reg.zero();
    cube::mma_ABt(&mut grad_xk_reg, grad_Z1, weight_stage);

    sync_cube();

    cube::store_rt_to_st(&grad_xk_reg, &mut grad_l_smem);

    sync_cube();

    grad_l_smem.sub(grad_target);
    grad_l_smem.add(&tile_grad_xk_combined);

    sync_cube();

    cube::store_st_direct(&grad_l_smem, &mut grads.grad_xk, stage_offset, 0, 0);

    // Accumulate weight gradients via global memory: dW = XK^T @ grad_Z1
    let mut dW_s1_reg = P::rt_ff();
    dW_s1_reg.zero();
    cube::mma_AB(&mut dW_s1_reg, k_smem, grad_Z1);

    sync_cube();

    // Load current global accumulator into weight_stage
    cube::load_st_direct(&grads.grad_weight, weight_stage, grad_weight_base, 0, 0);

    sync_cube();

    // Add dW to accumulator via register tiles
    let mut acc_rt_s1 = P::rt_ff();
    cube::load_rt_from_st(weight_stage, &mut acc_rt_s1);
    acc_rt_s1.add(&dW_s1_reg);
    cube::store_rt_to_st(&acc_rt_s1, weight_stage);

    sync_cube();

    // Store updated accumulator back to global
    cube::store_st_direct(weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // grad_b = grad_b_z1bar + sum(grad_Z1)
    let mut grad_b_z1 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(grad_Z1, &mut grad_b_z1, buf);

    grad_L_b_last.add(&grad_b_z1);
    grad_L_b_last.add(&grad_b_z1bar);

    // Accumulate LN gradients
    grad_L_ln_weight_acc.add(&grad_ln_weight_s4);
    grad_L_ln_weight_acc.add(&grad_ln_weight_s2);
    grad_L_ln_bias_acc.add(&grad_ln_bias_s4);
    grad_L_ln_bias_acc.add(&grad_ln_bias_s2);

    // Store grad_ttt_lr_eta
    let grad_ttt_lr_eta_val = grad_ttt_lr_eta.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(
        &grad_ttt_lr_eta_val,
        &mut grads.grad_ttt_lr_eta,
        ttt_lr_eta_idx,
    );

    // Atomically add grad_token_eta (shared across batch/head dimensions)
    atomic_add_rv::<_, P::CS>(&grad_token_eta, &mut grads.grad_token_eta, token_eta_base);
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

    // Weight gradient accumulator uses global memory (no StFF allocation for grad_L_W_last)
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);

    let mut weight_stage = P::st_ff();
    // Zero the weight gradient in global memory using weight_stage as a shuttle
    weight_stage.fill(P::EVal::new(0.0));

    sync_cube();

    cube::store_st_direct(&weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // Now load the actual weight_init
    cube::load_st_direct(&saved.weight_init, &mut weight_stage, base_weight, 0, 0);

    sync_cube();

    let mut bias_stage = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&recomp.bias, &mut bias_stage, base_bias);

    // REMOVED: grad_L_W_last StFF allocation (now uses global memory)

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    // Shared scratch tiles
    let mut scratch1 = P::st_cs_f();
    let mut scratch2 = P::st_cs_f();
    let mut ext_k_smem = P::st_f_cs();
    let mut tile_b = P::st_cs_f();
    let mut tile_c = P::st_cs_f();
    let mut ext_buf = ReduceBuf::<P::EAcc>::new();

    sync_cube();

    let base_weight_init = index_2d(&saved.weight_init, batch_idx, head_idx);

    fused_ttt_backward_stage::<P>(
        saved,
        recomp,
        grad_output,
        &mut weight_stage,
        &bias_stage,
        &mut grad_L_b_last,
        &mut grad_L_ln_weight_acc,
        &mut grad_L_ln_bias_acc,
        grads,
        base_qkv,
        ttt_lr_eta_idx,
        0, // token_eta_base: single-stage, offset = 0
        &saved.weight_init,
        base_weight_init,
        grad_weight_base,
        &mut scratch1,
        &mut scratch2,
        &mut ext_k_smem,
        &mut tile_b,
        &mut tile_c,
        &mut ext_buf,
        epsilon,
    );

    sync_cube();

    // Weight gradient already accumulated to global memory by backward_stage.
    // Store accumulated bias gradient.
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
///
/// Uses external scratch tiles (shared with backward stage) to avoid
/// exceeding shared memory limits in the multi-stage kernel.
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
    // External tiles (shared with backward stage to avoid duplicate smem allocs)
    k_smem: &mut StFCs<P>,
    scratch_a: &mut StCsF<P>,
    scratch_b: &mut StCsF<P>,
    scratch_c: &mut StCsF<P>,
    scratch_d: &mut StCsF<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    #[comptime] epsilon: f32,
) {
    // scratch_a = xk_smem, scratch_b = v_direct_smem, scratch_c = z1_smem, scratch_d = temp_smem

    // Load XK transposed and direct
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xk, scratch_a, stage_offset, 0, 0);
    cube::load_st_direct(&recomp.xv, scratch_b, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W + b
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, k_smem, weight_smem);

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

    cube::store_rt_to_st(&z1_reg, scratch_c);

    sync_cube();

    // target = xv - xk
    scratch_b.sub(scratch_a);

    sync_cube();

    // grad_l = layer_norm_l2_grad(z1, target)
    layer_norm_l2_grad::<P::EVal, P::EAcc, P::CS, P::F>(
        scratch_c,
        scratch_b,
        ln_weight_rv,
        ln_bias_rv,
        scratch_d,
        buf,
        epsilon,
    );

    sync_cube();

    // scratch_c now contains grad_l

    // Weight update: W -= last_eta * XK^T @ grad_l
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // Reload k transposed (reuse k_smem)
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);

    sync_cube();

    // Scale by last_eta
    k_smem.mul_row(&last_eta_rv);

    sync_cube();

    // weight_update = scaled_xk^T @ grad_l
    let mut weight_update_reg = P::rt_ff();
    weight_update_reg.zero();
    cube::mma_AB(&mut weight_update_reg, k_smem, scratch_c);

    sync_cube();

    // W -= weight_update
    let mut weight_reg = P::rt_ff();
    cube::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    cube::store_rt_to_st(&weight_reg, weight_smem);

    sync_cube();

    // Bias update: b -= last_eta @ grad_l
    scratch_d.copy_from(scratch_c);

    sync_cube();

    scratch_d.mul_col(&last_eta_rv);

    sync_cube();

    let mut bias_update_rv = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        scratch_d,
        &mut bias_update_rv,
        buf,
    );

    let bias_update_val = bias_update_rv.cast::<P::EVal>();
    bias_rv.sub(&bias_update_val);
}

use super::layer_norm::layer_norm_l2_grad;

/// Fused TTT-Linear backward pass kernel (multi-stage).
///
/// For multi-stage backward, we process stages in reverse order.
/// Per-stage weight/bias checkpoints are loaded from tensors saved during forward,
/// eliminating the O(N²) forward re-simulation.
#[cube(launch)]
pub fn fused_ttt_backward_kernel_multi<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    weight_checkpoints: &Tensor<Line<P::EVal>>,
    bias_checkpoints: &Tensor<Line<P::EVal>>,
    // Per-(batch,head) scratch buffer for storing reconstructed W[stage_idx]
    // before backward_stage overwrites it. Shape: [batch, heads, head_dim, head_dim].
    weight_stage_buf: &mut Tensor<Line<P::EVal>>,
    grad_output: &Tensor<Line<P::EVal>>,
    grads: &mut GradOutputs<P::EVal>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = saved.xq.shape(1);
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(P::CS::VALUE);
    let head_dim = comptime!(P::F::VALUE);
    let checkpoint_interval = comptime!(config.checkpoint_interval);
    let stage_stride = mini_batch_len * head_dim;
    let num_stages = num_stages as usize;

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let base_ttt_lr_eta = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    // Checkpoint layout matches forward: ceil(num_stages / checkpoint_interval) per batch/head
    let num_checkpoints = (num_stages + checkpoint_interval - 1) / checkpoint_interval;
    let ckpt_bh = (batch_idx * num_heads + head_idx) * num_checkpoints;

    // Per-(batch,head) offset into weight_stage_buf [batch, heads, F, F]
    let bh_buf_offset = index_2d(weight_stage_buf, batch_idx, head_idx);

    // Weight gradient accumulator uses global memory (no StFF allocation for grad_L_W_last)
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);

    // REMOVED: grad_L_W_last StFF allocation (now uses global memory)

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    // Shared scratch tiles (reused between forward_simulate and backward_stage)
    let mut scratch1 = P::st_cs_f();
    let mut scratch2 = P::st_cs_f();
    let mut ext_k_smem = P::st_f_cs();
    let mut tile_b = P::st_cs_f();
    let mut tile_c = P::st_cs_f();
    let mut ext_buf = ReduceBuf::<P::EAcc>::new();
    let mut weight_stage = P::st_ff();

    sync_cube();

    // Zero the weight gradient in global memory using weight_stage as a shuttle
    weight_stage.fill(P::EVal::new(0.0));

    sync_cube();

    cube::store_st_direct(&weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // Load LN params once (needed for forward simulation)
    let base_ln = index_2d(&saved.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&recomp.ln_bias, &mut ln_bias_rv, base_ln);

    // Process stages in reverse order (backward through time).
    for stage in 0..num_stages {
        let stage_idx = num_stages - 1 - stage;
        let stage_offset = base_qkv + stage_idx * stage_stride;
        let ttt_lr_eta_idx = base_ttt_lr_eta + stage_idx * mini_batch_len;

        // === Reconstruct W[stage_idx] from nearest earlier checkpoint ===
        let ckpt_stage = (stage_idx / checkpoint_interval) * checkpoint_interval;
        let ckpt_idx = ckpt_stage / checkpoint_interval;

        let ckpt_weight_offset = (ckpt_bh + ckpt_idx) * head_dim * head_dim;
        cube::load_st_direct(weight_checkpoints, &mut weight_stage, ckpt_weight_offset, 0, 0);

        sync_cube();

        let mut bias_stage = P::rvb_f_v();
        let ckpt_bias_offset = (ckpt_bh + ckpt_idx) * head_dim;
        cube::broadcast::load_rv_direct(bias_checkpoints, &mut bias_stage, ckpt_bias_offset);

        // Forward-simulate from checkpoint to stage_idx (0 iterations if checkpoint_interval=1)
        for fwd in ckpt_stage..stage_idx {
            let fwd_offset = base_qkv + fwd * stage_stride;
            let fwd_ttt_lr = base_ttt_lr_eta + fwd * mini_batch_len;

            forward_simulate_weight_update::<P>(
                saved,
                recomp,
                &mut weight_stage,
                &mut bias_stage,
                &ln_weight_rv,
                &ln_bias_rv,
                fwd_offset,
                fwd_ttt_lr,
                &mut ext_k_smem,
                &mut scratch1,
                &mut scratch2,
                &mut tile_b,
                &mut tile_c,
                &mut ext_buf,
                epsilon,
            );

            sync_cube();
        }

        // Store reconstructed W[stage_idx] to global buffer before backward_stage
        // overwrites weight_stage (reused as temp_f_f). The backward stage will
        // reload from this buffer when it needs W[stage_idx] again.
        cube::store_st_direct(&weight_stage, weight_stage_buf, bh_buf_offset, 0, 0);

        // Each stage writes its token_eta gradient to its own offset within [seq_len]
        let token_eta_base = stage_idx * mini_batch_len;
        fused_ttt_backward_stage::<P>(
            saved,
            recomp,
            grad_output,
            &mut weight_stage,
            &bias_stage,
            &mut grad_L_b_last,
            &mut grad_L_ln_weight_acc,
            &mut grad_L_ln_bias_acc,
            grads,
            stage_offset,
            ttt_lr_eta_idx,
            token_eta_base,
            weight_stage_buf,
            bh_buf_offset,
            grad_weight_base,
            &mut scratch1,
            &mut scratch2,
            &mut ext_k_smem,
            &mut tile_b,
            &mut tile_c,
            &mut ext_buf,
            epsilon,
        );

        sync_cube();
    }

    // Weight gradient already accumulated to global memory by backward_stage.
    // Store accumulated bias gradient.
    let grad_L_b_last_val = grad_L_b_last.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(&grad_L_b_last_val, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let ln_base = head_idx * P::F::VALUE;
    atomic_add_rv::<_, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, ln_base);
    atomic_add_rv::<_, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, ln_base);
}
