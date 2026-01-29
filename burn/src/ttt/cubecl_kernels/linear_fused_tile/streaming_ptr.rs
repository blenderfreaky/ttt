//! Streaming TTT-Linear kernel with pointer indirection for zero-copy input.
//!
//! This kernel uses a pointer table to access input tensors directly without
//! any memory copies. The host writes tensor addresses to the pointer table,
//! and the kernel dereferences them directly.
//!
//! ## Pointer Table Layout
//! - Slot 0: xq address
//! - Slot 1: xk address
//! - Slot 2: xv address
//! - Slot 3: ttt_lr_eta address
//! - Slot 4: output address (for writing)
//!
//! ## Buffer Indices (kernel parameters)
//! - buffer_0: ptr_table (u64 addresses)
//! - buffer_1: control (atomic u32)
//! - buffer_2: weight
//! - buffer_3: bias
//! - buffer_4: token_eta
//! - buffer_5: ln_weight
//! - buffer_6: ln_bias
//! - buffer_7: weight_out
//! - buffer_8: bias_out
//! - buffer_9: xq_buf (Array parameter)
//! - buffer_10: xk_buf (Array parameter)
//! - buffer_11: xv_buf (Array parameter)
//! - buffer_12: eta_buf (Array parameter)

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::prelude::*;

use super::helpers::ParamsTrait;
use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Pointer table slot indices
pub const PTR_XQ: usize = 0;
pub const PTR_XK: usize = 1;
pub const PTR_XV: usize = 2;
pub const PTR_TTT_LR_ETA: usize = 3;
pub const PTR_OUTPUT: usize = 4;
pub const PTR_TABLE_SIZE: usize = 5;

/// Control flags
pub const CTRL_STATUS: usize = 0;
pub const CTRL_ARRAY_SIZE: usize = 1;

pub const STATUS_IDLE: u32 = 0;
pub const STATUS_READY: u32 = 1;
pub const STATUS_DONE: u32 = 2;
pub const STATUS_SHUTDOWN: u32 = 3;

/// Buffer indices for injected HIP code
pub const BUF_PTR_TABLE: usize = 0;
pub const BUF_CONTROL: usize = 1;
pub const BUF_XQ: usize = 9;
pub const BUF_XK: usize = 10;
pub const BUF_XV: usize = 11;
pub const BUF_ETA: usize = 12;

/// Inject HIP code to load from pointer table into Array parameters.
///
/// Uses buffer indices for the Array parameters (buffer_9 through buffer_12).
#[cube]
#[allow(unused_variables)]
fn load_from_pointers<P: ParamsTrait>(
    xq_buf: &mut Array<Line<P::E>>,
    xk_buf: &mut Array<Line<P::E>>,
    xv_buf: &mut Array<Line<P::E>>,
    eta_buf: &mut Array<Line<P::E>>,
    #[comptime] qkv_count: usize,
    #[comptime] eta_count: usize,
) {
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: format!(
                r#"*/
// Load xq from pointer table slot 0 into buffer_9
{{
    const uint64 xq_addr = ((const uint64*)buffer_{buf_ptr})[{ptr_xq}];
    const float_4* xq_src = (const float_4*)(xq_addr);
    const uint32 xq_offset = (blockIdx.x * gridDim.y + blockIdx.y) * {qkv}u;
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xq}[i] = xq_src[xq_offset + i]; }}
}}
// Load xk from pointer table slot 1 into buffer_10
{{
    const uint64 xk_addr = ((const uint64*)buffer_{buf_ptr})[{ptr_xk}];
    const float_4* xk_src = (const float_4*)(xk_addr);
    const uint32 xk_offset = (blockIdx.x * gridDim.y + blockIdx.y) * {qkv}u;
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xk}[i] = xk_src[xk_offset + i]; }}
}}
// Load xv from pointer table slot 2 into buffer_11
{{
    const uint64 xv_addr = ((const uint64*)buffer_{buf_ptr})[{ptr_xv}];
    const float_4* xv_src = (const float_4*)(xv_addr);
    const uint32 xv_offset = (blockIdx.x * gridDim.y + blockIdx.y) * {qkv}u;
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xv}[i] = xv_src[xv_offset + i]; }}
}}
// Load ttt_lr_eta from pointer table slot 3 into buffer_12
{{
    const uint64 eta_addr = ((const uint64*)buffer_{buf_ptr})[{ptr_eta}];
    const float_4* eta_src = (const float_4*)(eta_addr);
    const uint32 eta_offset = (blockIdx.x * gridDim.y + blockIdx.y) * {eta}u;
    for (uint32 i = 0; i < {eta}u; i++) {{ buffer_{buf_eta}[i] = eta_src[eta_offset + i]; }}
}}
/*"#,
                buf_ptr = BUF_PTR_TABLE,
                buf_xq = BUF_XQ,
                buf_xk = BUF_XK,
                buf_xv = BUF_XV,
                buf_eta = BUF_ETA,
                ptr_xq = PTR_XQ,
                ptr_xk = PTR_XK,
                ptr_xv = PTR_XV,
                ptr_eta = PTR_TTT_LR_ETA,
                qkv = qkv_count,
                eta = eta_count,
            ),
        });
    });
}

/// Inject HIP code to store from buffer_9 (xq_buf) to output pointer.
#[cube]
#[allow(unused_variables)]
fn store_to_output<P: ParamsTrait>(
    xq_buf: &Array<Line<P::E>>,
    #[comptime] count: usize,
) {
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: format!(
                r#"*/
// Store from buffer_9 to output via pointer table slot 4
{{
    const uint64 out_addr = ((const uint64*)buffer_{buf_ptr})[{ptr_out}];
    float_4* out_dst = (float_4*)(out_addr);
    const uint32 out_offset = (blockIdx.x * gridDim.y + blockIdx.y) * {count}u;
    for (uint32 i = 0; i < {count}u; i++) {{ out_dst[out_offset + i] = buffer_{buf_xq}[i]; }}
}}
/*"#,
                buf_ptr = BUF_PTR_TABLE,
                buf_xq = BUF_XQ,
                ptr_out = PTR_OUTPUT,
                count = count,
            ),
        });
    });
}

/// Streaming kernel with pointer indirection.
///
/// Parameter order (determines buffer indices):
/// - buffer_0: ptr_table (u64 addresses)
/// - buffer_1: control (atomic u32)
/// - buffer_2: weight
/// - buffer_3: bias
/// - buffer_4: token_eta
/// - buffer_5: ln_weight
/// - buffer_6: ln_bias
/// - buffer_7: weight_out
/// - buffer_8: bias_out
/// - buffer_9: xq_buf (Array for loading xq)
/// - buffer_10: xk_buf (Array for loading xk)
/// - buffer_11: xv_buf (Array for loading xv)
/// - buffer_12: eta_buf (Array for loading eta)
#[cube(launch)]
pub fn fused_ttt_streaming_ptr_kernel<P: ParamsTrait>(
    // Pointer table: addresses of input tensors [PTR_TABLE_SIZE]
    ptr_table: &Tensor<u64>, // buffer_0
    // Control array [batch * heads] - mutable for atomic ops
    control: &mut Tensor<Atomic<u32>>, // buffer_1
    // Constant/state tensors
    weight: &Tensor<Line<P::E>>,    // buffer_2
    bias: &Tensor<Line<P::E>>,      // buffer_3
    token_eta: &Tensor<Line<P::E>>, // buffer_4
    ln_weight: &Tensor<Line<P::E>>, // buffer_5
    ln_bias: &Tensor<Line<P::E>>,   // buffer_6
    // Output state tensors
    weight_out: &mut Tensor<Line<P::E>>, // buffer_7
    bias_out: &mut Tensor<Line<P::E>>,   // buffer_8
    // Array buffers for pointer-based loading (get predictable buffer_N names)
    xq_buf: &mut Array<Line<P::E>>,  // buffer_9
    xk_buf: &mut Array<Line<P::E>>,  // buffer_10
    xv_buf: &mut Array<Line<P::E>>,  // buffer_11
    eta_buf: &mut Array<Line<P::E>>, // buffer_12
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = CUBE_COUNT_Y as usize;
    let _epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(config.mini_batch_len);
    let head_dim = comptime!(config.head_dim);
    let _threads = comptime!(config.threads);

    // Control index for this cube
    let ctrl_idx = batch_idx * num_heads + head_idx;

    // Compute sizes for pointer loads
    let _qkv_size = mini_batch_len * head_dim / 4; // In float_4 units
    let _eta_size = mini_batch_len / 4; // In float_4 units (ttt_lr_eta is smaller)

    // Initialize weight in shared memory
    let mut weight_smem = P::f_f_tile();
    let weight_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / 4;
    cube::load_st_direct(weight, &mut weight_smem, weight_offset, 0, 0);

    sync_cube();

    // Initialize bias in registers
    let mut bias_rv = P::f_reg_big();
    let bias_offset = (batch_idx * num_heads + head_idx) * head_dim / 4;
    cube::broadcast::load_rv_direct(bias, &mut bias_rv, bias_offset);

    // Load layer norm params
    let ln_offset = head_idx * head_dim / 4;
    let mut ln_weight_rv = P::f_reg_big();
    let mut ln_bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(ln_weight, &mut ln_weight_rv, ln_offset);
    cube::broadcast::load_rv_direct(ln_bias, &mut ln_bias_rv, ln_offset);

    // Main processing loop
    loop {
        // Poll for status change (only thread 0)
        let mut status: u32 = 0u32;
        if UNIT_POS == 0 {
            loop {
                status = Atomic::load(&control[ctrl_idx]);
                if status != 0u32 {
                    break;
                }
                // Small sleep to reduce memory bus contention
                gpu_sleep(10u32);
            }
        }

        // Broadcast status to all threads
        sync_cube();
        status = Atomic::load(&control[ctrl_idx]);

        if status == 3u32 {
            // SHUTDOWN
            break;
        }

        // Load input data from pointers via injected HIP code
        load_from_pointers::<P>(
            xq_buf,
            xk_buf,
            xv_buf,
            eta_buf,
            comptime!(config.mini_batch_len * config.head_dim / 4),
            comptime!(config.mini_batch_len / 4),
        );

        sync_cube();

        // TODO: Process the mini-batch using Array buffers
        // This requires refactoring fused_ttt_forward_stage to work with Arrays
        // instead of Tensor slices

        // For now, copy xq to output to verify pointer indirection works
        store_to_output::<P>(xq_buf, comptime!(config.mini_batch_len * config.head_dim / 4));

        sync_cube();

        // Mark as done
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], 2u32); // DONE
        }

        sync_cube();
    }

    // Shutdown: store final weight and bias
    let weight_out_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / 4;
    let bias_out_offset = (batch_idx * num_heads + head_idx) * head_dim / 4;

    cube::store_st_direct(&weight_smem, weight_out, weight_out_offset, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, bias_out, bias_out_offset);
}
