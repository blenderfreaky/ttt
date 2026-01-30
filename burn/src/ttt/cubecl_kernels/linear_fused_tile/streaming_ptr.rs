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
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp};
use thundercube::util::transpose_4;

use super::helpers::ParamsTrait;
use super::forward::{Inputs, Outputs, ForwardIntermediates, fused_ttt_forward_stage};
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

/// Buffer indices for injected HIP code (must match kernel parameter order)
pub const BUF_PTR_TABLE: usize = 0;
pub const BUF_CONTROL: usize = 1;
pub const BUF_XQ: usize = 2;
pub const BUF_XK: usize = 3;
pub const BUF_XV: usize = 4;
pub const BUF_ETA: usize = 5;

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

/// Cooperatively copy from Array to Tensor.
/// All threads participate in the copy.
#[cube]
fn copy_array_to_tensor<F: Float>(
    src: &Array<Line<F>>,
    dst: &mut Tensor<Line<F>>,
    dst_offset: usize,
    #[comptime] count: usize,
) {
    for i in range_stepped(UNIT_POS as usize, count, CUBE_DIM as usize) {
        dst[dst_offset + i] = src[i];
    }
}

/// Cooperatively copy from Tensor to Array.
/// All threads participate in the copy.
#[cube]
fn copy_tensor_to_array<F: Float>(
    src: &Tensor<Line<F>>,
    src_offset: usize,
    dst: &mut Array<Line<F>>,
    #[comptime] count: usize,
) {
    for i in range_stepped(UNIT_POS as usize, count, CUBE_DIM as usize) {
        dst[i] = src[src_offset + i];
    }
}

/// Load from a Slice into shared memory tile (direct, no transpose).
/// Assumes row-major contiguous layout with known dimensions.
#[cube]
pub fn load_st_from_slice<F: Float, R: Dim, C: Dim>(
    src: Slice<Line<F>>,
    dst: &mut St<F, R, C>,
    #[comptime] src_cols: usize, // Number of columns in source (head_dim)
) {
    let vec_stride = C::LINES;
    let total_vecs = R::VALUE * vec_stride;
    let mask = vec_stride - 1;

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    // Source has src_cols columns, so LINE stride is src_cols / LINE_SIZE
    let src_line_stride = comptime!(src_cols / LINE_SIZE);

    for i in range_stepped(tid, total_vecs, num_threads) {
        let r = i / vec_stride;
        let c_vec = i % vec_stride;

        // Source index: row * line_stride + col_vec
        let src_idx = r * src_line_stride + c_vec;
        let val = src[src_idx];

        let phys_c = cube::swizzle(r, c_vec, mask);
        let s_idx = r * vec_stride + phys_c;
        dst.data[s_idx] = val;
    }
}

/// Load from a Slice into shared memory tile with transpose.
/// Assumes row-major contiguous layout with known dimensions.
#[cube]
pub fn load_st_from_slice_transpose<F: Float, R: Dim, C: Dim>(
    src: Slice<Line<F>>,
    dst: &mut St<F, R, C>,
    #[comptime] src_cols: usize, // Number of columns in source (head_dim)
) {

    let s_stride = C::LINES;
    let patches_h = R::LINES;
    let patches_w = C::LINES;
    let total_patches = patches_h * patches_w;

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;
    let mask = s_stride - 1;

    // Source has src_cols columns, so LINE stride is src_cols / LINE_SIZE
    let src_line_stride = comptime!(src_cols / LINE_SIZE);

    for i in range_stepped(tid, total_patches, num_threads) {
        let patch_row = i / patches_w;
        let patch_col = i % patches_w;

        let base_row = patch_row * LINE_SIZE;
        let base_col_vec = patch_col;

        // Load 4 consecutive rows
        let r0 = src[(base_row + 0) * src_line_stride + base_col_vec];
        let r1 = src[(base_row + 1) * src_line_stride + base_col_vec];
        let r2 = src[(base_row + 2) * src_line_stride + base_col_vec];
        let r3 = src[(base_row + 3) * src_line_stride + base_col_vec];

        // Transpose
        let (c0, c1, c2, c3) = transpose_4(r0, r1, r2, r3);

        // Store transposed
        let s_base_row = patch_col * LINE_SIZE;
        let s_col_vec = patch_row;

        let s0 = cube::swizzle(s_base_row + 0, s_col_vec, mask);
        let s1 = cube::swizzle(s_base_row + 1, s_col_vec, mask);
        let s2 = cube::swizzle(s_base_row + 2, s_col_vec, mask);
        let s3 = cube::swizzle(s_base_row + 3, s_col_vec, mask);

        dst.data[(s_base_row + 0) * s_stride + s0] = c0;
        dst.data[(s_base_row + 1) * s_stride + s1] = c1;
        dst.data[(s_base_row + 2) * s_stride + s2] = c2;
        dst.data[(s_base_row + 3) * s_stride + s3] = c3;
    }
}

/// Streaming kernel with pointer indirection.
///
/// The kernel receives data via pointer table (zero-copy from host tensors),
/// copies to scratch Tensors, then calls the standard TTT forward stage.
///
/// Arrays (buffer_9-12) receive data via injected HIP code that dereferences
/// addresses from ptr_table. After sync, we copy Array -> scratch Tensor
/// so the standard forward stage can operate on Tensors.
#[cube(launch)]
pub fn fused_ttt_streaming_ptr_kernel<P: ParamsTrait>(
    // Pointer table: addresses of input tensors [PTR_TABLE_SIZE]
    ptr_table: &Tensor<u64>,
    // Control array [batch * heads] - mutable for atomic ops
    control: &mut Tensor<Atomic<u32>>,
    // Array buffers for pointer-based loading (get predictable buffer_N names)
    xq_buf: &mut Array<Line<P::E>>,
    xk_buf: &mut Array<Line<P::E>>,
    xv_buf: &mut Array<Line<P::E>>,
    eta_buf: &mut Array<Line<P::E>>,
    // Inputs struct (scratch tensors for xq/xk/xv/ttt_lr_eta, constants for others)
    inputs: &mut Inputs<P::E>,
    // Outputs struct (output tensor, weight_out, bias_out)
    outputs: &mut Outputs<P::E>,
    // Forward intermediates (for backward pass compatibility)
    fwd_intermediates: &mut ForwardIntermediates<P::E>,
    #[comptime] config: FusedTttConfig,
    #[comptime] debug: bool,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = CUBE_COUNT_Y as usize;
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(config.mini_batch_len);
    let head_dim = comptime!(config.head_dim);

    // Control index for this cube
    let ctrl_idx = batch_idx * num_heads + head_idx;

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: kernel start ctrl_idx=%u\n", ctrl_idx);
        }
    }

    // Sizes in Lines (float_4 units)
    let qkv_lines = comptime!(mini_batch_len * head_dim / LINE_SIZE);
    let eta_lines = comptime!(mini_batch_len / LINE_SIZE);

    // Initialize weight in shared memory from inputs.weight
    let mut weight_smem = P::f_f_tile();
    let weight_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / LINE_SIZE;
    cube::load_st_direct(&inputs.weight, &mut weight_smem, weight_offset, 0, 0);

    sync_cube();

    // Initialize bias in registers from inputs.bias
    let mut bias_rv = P::f_reg_big();
    let bias_offset = (batch_idx * num_heads + head_idx) * head_dim / LINE_SIZE;
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, bias_offset);

    // Load layer norm params
    let ln_offset = head_idx * head_dim / LINE_SIZE;
    let mut ln_weight_rv = P::f_reg_big();
    let mut ln_bias_rv = P::f_reg_big();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, ln_offset);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, ln_offset);

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: init done, entering main loop ctrl=%u\n", ctrl_idx);
        }
    }

    // Main processing loop
    loop {
        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: loop iteration ctrl=%u\n", ctrl_idx);
            }
        }

        // Poll for status change (only thread 0)
        // Wait for READY (1) or SHUTDOWN (3), skip IDLE (0) and DONE (2)
        let mut status: u32 = 0u32;
        if UNIT_POS == 0 {
            loop {
                status = Atomic::load(&control[ctrl_idx]);
                if comptime!(debug) {
                    if batch_idx == 0 && head_idx == 0 {
                        debug_print!("PTR_STREAM: poll status=%u\n", status);
                    }
                }
                // Only break on READY (1) or SHUTDOWN (3)
                if status == 1u32 || status == 3u32 {
                    break;
                }
                // Small sleep to reduce memory bus contention
                gpu_sleep(10u32);
            }
        }

        // Broadcast status to all threads
        sync_cube();
        status = Atomic::load(&control[ctrl_idx]);

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: after sync status=%u\n", status);
            }
        }

        if status == 3u32 {
            // SHUTDOWN
            if comptime!(debug) {
                if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                    debug_print!("PTR_STREAM: shutdown received ctrl=%u\n", ctrl_idx);
                }
            }
            break;
        }

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: loading from pointers ctrl=%u\n", ctrl_idx);
            }
        }

        // Load input data from pointers via injected HIP code
        load_from_pointers::<P>(
            xq_buf,
            xk_buf,
            xv_buf,
            eta_buf,
            qkv_lines,
            eta_lines,
        );

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: copying to scratch tensors ctrl=%u\n", ctrl_idx);
            }
        }

        // Copy from Arrays to scratch Tensors in Inputs struct.
        // This indirection allows reusing fused_ttt_forward_stage which expects Tensors.
        // The compiler should optimize this to direct register/memory access.
        copy_array_to_tensor(xq_buf, &mut inputs.xq, 0, qkv_lines);
        copy_array_to_tensor(xk_buf, &mut inputs.xk, 0, qkv_lines);
        copy_array_to_tensor(xv_buf, &mut inputs.xv, 0, qkv_lines);
        copy_array_to_tensor(eta_buf, &mut inputs.ttt_lr_eta, 0, eta_lines);

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: running forward stage ctrl=%u\n", ctrl_idx);
            }
        }

        // Run TTT forward stage
        // stage_offset=0, ttt_lr_eta_idx=0 since scratch tensors contain exactly one mini-batch
        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            fwd_intermediates,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            0,  // stage_offset
            0,  // ttt_lr_eta_idx
            epsilon,
        );

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: storing output ctrl=%u\n", ctrl_idx);
            }
        }

        // Copy output back to pointer destination
        copy_tensor_to_array(&outputs.output, 0, xq_buf, qkv_lines);
        store_to_output::<P>(xq_buf, qkv_lines);

        sync_cube();

        // Mark as done
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], 2u32); // DONE
        }

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: marked DONE ctrl=%u\n", ctrl_idx);
            }
        }

        sync_cube();
    }

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: storing final weight/bias ctrl=%u\n", ctrl_idx);
        }
    }

    // Shutdown: store final weight and bias to outputs
    let weight_out_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / LINE_SIZE;
    let bias_out_offset = (batch_idx * num_heads + head_idx) * head_dim / LINE_SIZE;

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, weight_out_offset, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, bias_out_offset);

    // Ensure all stores are complete before kernel exits
    sync_cube();

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: kernel exit ctrl=%u\n", ctrl_idx);
        }
    }
}
