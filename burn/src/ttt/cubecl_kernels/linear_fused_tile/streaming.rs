//! Streaming TTT-Linear kernel for persistent GPU execution.
//!
//! This kernel runs persistently on the GPU, receiving mini-batches incrementally
//! via async memory transfers. Weight and bias are kept in shared memory between
//! stages to avoid global memory round-trips.
//!
//! ## Control Protocol
//!
//! Communication between host and kernel uses a control array with indices per cube:
//! - `control[base + CTRL_READY]`: Host sets to 1 when input is available
//! - `control[base + CTRL_DONE]`: Kernel sets to 1 when output is ready
//! - `control[base + CTRL_SHUTDOWN]`: Host sets to 1 to signal kernel exit

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{prelude::*, util::index_2d};

use super::{
    forward::{ForwardIntermediates, Inputs, Outputs, fused_ttt_forward_stage},
    helpers::ParamsTrait,
};
use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Configuration for streaming kernel with debug flag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamingKernelConfig {
    pub fused: FusedTttConfig,
    pub debug: bool,
}

impl StreamingKernelConfig {
    pub fn new(fused: FusedTttConfig, debug: bool) -> Self {
        Self { fused, debug }
    }
}

/// Control array indices (per cube):
/// [READY, DONE, SHUTDOWN]
pub const CTRL_READY: usize = 0;
pub const CTRL_DONE: usize = 1;
pub const CTRL_SHUTDOWN: usize = 2;
pub const CTRL_ARRAY_SIZE: usize = 3;

/// Streaming input/output buffers for a single mini-batch.
///
/// These buffers are sized for one mini-batch `[batch, heads, mini_batch_len, head_dim]`
/// and are reused across stages.
#[derive(CubeType, CubeLaunch)]
pub struct StreamingBuffers<F: Float> {
    /// Query input [batch, heads, mini_batch_len, head_dim]
    pub xq: Tensor<Line<F>>,
    /// Key input [batch, heads, mini_batch_len, head_dim]
    pub xk: Tensor<Line<F>>,
    /// Value input [batch, heads, mini_batch_len, head_dim]
    pub xv: Tensor<Line<F>>,
    /// TTT learning rate eta [batch, heads, mini_batch_len]
    pub ttt_lr_eta: Tensor<Line<F>>,
    /// Output [batch, heads, mini_batch_len, head_dim]
    pub output: Tensor<Line<F>>,
    /// Control array for host-kernel communication [batch * heads * CTRL_ARRAY_SIZE]
    pub control: Tensor<Atomic<u32>>,
}

/// Streaming kernel that processes mini-batches incrementally.
///
/// The kernel runs in a persistent loop:
/// 1. Wait for CTRL_READY flag from host
/// 2. Process the mini-batch using `fused_ttt_forward_stage`
/// 3. Set CTRL_DONE flag for host
/// 4. Repeat until CTRL_SHUTDOWN is received
///
/// Weight and bias are kept in shared memory between stages.
/// Forward intermediates are written to global memory with stage-based offsets.
#[cube(launch)]
pub fn fused_ttt_streaming_kernel<P: ParamsTrait>(
    // Input/output streaming buffers
    inputs: &Inputs<P::E>,
    outputs: &mut Outputs<P::E>,
    // Control array for synchronization [status, stage_idx, shutdown, reserved] per cube
    control: &mut Array<u32>,
    // Forward intermediates storage
    fwd_intermediates: &mut ForwardIntermediates<P::E>,
    #[comptime] config: StreamingKernelConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = CUBE_COUNT_Y as usize;
    let epsilon = comptime!(config.fused.epsilon());
    let mini_batch_len = comptime!(config.fused.mini_batch_len);
    let head_dim = comptime!(config.fused.head_dim);
    let debug = comptime!(config.debug);

    // Control array index for this cube
    let ctrl_base = (batch_idx * num_heads + head_idx) * (CTRL_ARRAY_SIZE as usize);

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!(
                "STREAM: kernel start batch=%u head=%u\n",
                batch_idx,
                head_idx
            );
        }
    }

    // Stride to advance by one mini-batch in the sequence dimension (in scalars)
    let _stage_stride = mini_batch_len * head_dim;

    // Compute base offsets for this (batch, head) pair
    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
    let base_eta = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Initialize weight in shared memory from inputs.weight
    let mut weight_smem = P::f_f_tile();
    cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_cube();

    // Initialize bias in register vector from inputs.bias
    let mut bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params (constant across all stages)
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::f_reg_big();
    let mut ln_bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("STREAM: init done, entering main loop x=%u\n", batch_idx);
        }
    }

    // Main processing loop
    loop {
        // Only thread 0 polls for control flags, then broadcasts via sync_cube
        if UNIT_POS == 0 {
            loop {
                // Check for shutdown first
                if control[ctrl_base + CTRL_SHUTDOWN] != 0 {
                    // Signal shutdown to other threads via READY = MAX
                    control[ctrl_base + CTRL_READY] = u32::MAX;
                    if comptime!(debug) {
                        debug_print!("STREAM: shutdown received cube=%u\n", ctrl_base);
                    }
                    break;
                }
                // Check for ready
                if control[ctrl_base + CTRL_READY] != 0 {
                    break;
                }
                gpu_sleep(50u32);
            }
        }

        sync_cube();

        // Check if we should shutdown (thread 0 set READY to MAX)
        if control[ctrl_base + CTRL_READY] == u32::MAX {
            break;
        }

        // Process the mini-batch
        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            fwd_intermediates,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            base_qkv,
            base_eta,
            epsilon,
        );

        sync_cube();

        // Signal completion - only thread 0 writes
        if UNIT_POS == 0 {
            control[ctrl_base + CTRL_READY] = 0;
            control[ctrl_base + CTRL_DONE] = 1;
        }

        sync_cube();
    }

    // Shutdown: store final weight and bias to global memory
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("STREAM: kernel exit x=%u\n", batch_idx);
        }
    }
}
