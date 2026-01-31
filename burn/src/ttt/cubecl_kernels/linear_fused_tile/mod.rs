//! Fused TTT-Linear kernel (tiled implementation using thundercube).
//!
//! This module contains the tiled fused TTT-Linear implementation that uses
//! shared memory tiles from the thundercube library for optimized memory access.

#[cfg(feature = "rocm")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "rocm")]
use burn_backend::StreamId;

/// Global mutex for streaming kernel tests.
///
/// Streaming tests cannot run concurrently because `get_resource()` triggers
/// cross-stream synchronization, which blocks forever if a persistent kernel
/// is running on another stream. This mutex ensures only one streaming test
/// runs at a time.
#[cfg(all(test, feature = "rocm"))]
pub static STREAMING_TEST_MUTEX: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

/// Shared counter for generating unique kernel stream IDs across all persistent kernels.
/// Starts at 1 to avoid collision with the default stream (0).
/// CubeCL maps stream IDs to physical streams via `stream_id % max_streams`.
#[cfg(feature = "rocm")]
static PERSISTENT_KERNEL_STREAM_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Get a unique stream ID for a persistent kernel.
/// All persistent kernel types share this counter to avoid stream ID collisions.
#[cfg(feature = "rocm")]
pub fn next_persistent_kernel_stream_id() -> StreamId {
    StreamId {
        value: PERSISTENT_KERNEL_STREAM_COUNTER.fetch_add(1, Ordering::Relaxed),
    }
}

mod api;
mod backward;
mod backward_optimized;
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
    layer_norm_forward_stream_intermediates, layer_norm_forward_with_intermediates,
    layer_norm_l2_grad, layer_norm_l2_grad_backward, layer_norm_l2_grad_stream_intermediates,
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
