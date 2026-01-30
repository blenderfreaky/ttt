//! Fused TTT-Linear kernel (tiled implementation using thundercube).
//!
//! This module contains the tiled fused TTT-Linear implementation that uses
//! shared memory tiles from the thundercube library for optimized memory access.

mod api;
mod backward;
mod forward;
mod helpers;
mod launch;
pub mod layer_norm;
#[cfg(feature = "rocm")]
pub mod streaming;
#[cfg(feature = "rocm")]
pub mod streaming_host;
#[cfg(feature = "rocm")]
pub mod streaming_ptr;
#[cfg(feature = "rocm")]
pub mod streaming_ptr_host;
#[cfg(feature = "rocm")]
pub mod streaming_wrapper;
#[cfg(feature = "rocm")]
pub mod streaming_ptr_wrapper;
mod wrapper;

pub use api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};
pub use forward::{
    ForwardIntermediates, Inputs, Outputs, fused_ttt_forward_kernel,
    fused_ttt_forward_kernel_multi, fused_ttt_forward_stage,
};
pub use launch::{
    TttTileKernel, TttTileMultiKernel, forward, forward_multi, launch_tile_forward,
    launch_tile_forward_multi,
};
pub use layer_norm::{
    SumSqOp, layer_norm_backward, layer_norm_forward, layer_norm_forward_save_intermediates,
    layer_norm_forward_with_intermediates, layer_norm_l2_grad, layer_norm_l2_grad_backward,
    layer_norm_l2_grad_stream_intermediates,
};
#[cfg(feature = "rocm")]
pub use streaming::{
    CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY, CTRL_SHUTDOWN, fused_ttt_streaming_kernel,
};
#[cfg(feature = "rocm")]
pub use streaming_host::{StreamingConfig, TttStreamingState, get_or_create_streaming_state};
#[cfg(feature = "rocm")]
pub use streaming_wrapper::{StreamingKernelConfig, TttStreamingKernel};
#[cfg(feature = "rocm")]
pub use streaming_ptr::{
    CTRL_ARRAY_SIZE as PTR_CTRL_ARRAY_SIZE, PTR_TABLE_SIZE,
    STATUS_DONE, STATUS_IDLE, STATUS_READY, STATUS_SHUTDOWN,
    fused_ttt_streaming_ptr_kernel,
};
#[cfg(feature = "rocm")]
pub use streaming_ptr_host::{PtrStreamingConfig, PtrStreamingTensors, TttPtrStreamingState};
#[cfg(feature = "rocm")]
pub use streaming_ptr_wrapper::{PtrStreamingKernelConfig, TttPtrStreamingKernel};
