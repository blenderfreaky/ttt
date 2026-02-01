//! Fused TTT-Linear kernel (tiled implementation using thundercube).
//!
//! This module contains the tiled fused TTT-Linear implementation that uses
//! shared memory tiles from the thundercube library for optimized memory access.

#[cfg(feature = "rocm")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "rocm")]
use burn_backend::StreamId;

// =============================================================================
// Tile configuration macros (shared across launch.rs and streaming modules)
// =============================================================================

/// Supported tile configurations.
/// Format: (mini_batch_len, head_dim, threads, CS_dim, F_dim, CS_Reg_dim, F_Reg_dim)
///
/// threads = max((CS/CSR)², (CS/CSR)×(F/FR), (F/FR)²)
/// Dispatch matches on (mini_batch_len, head_dim, threads) to select config.
#[cfg(not(feature = "tile-tuning"))]
macro_rules! supported_tile_configs {
    ($callback:ident!($($args:tt)*)) => {
        $callback!($($args)*
            // Base configs
            // (mini_batch, head_dim, threads, CS, F, CSR, FR)
            //
            // Note: Larger tiles (16x128, 32x64, 64x64) exceed GPU shared memory limits
            // and have been removed. Shared memory usage is approximately:
            //   2*(F*CS) + 4*(CS*F) + (CS*CS) + (F*F) elements
            // Most GPUs have 48-64KB shared memory limit per block.
            ( 8,  32,  64, D8,  D32,  D4, D4),    // max(2², 2×8, 8²) = 64, ~10KB smem
            ( 8,  64,  64, D8,  D64,  D4, D8),    // max(2², 2×8, 8²) = 64, ~22KB smem
            (16,  32,  16, D16, D32,  D4, D8),    // max(4², 4×4, 4²) = 16, ~14KB smem
            (16,  64, 256, D16, D64,  D4, D4),    // max(4², 4×16, 16²) = 256, ~30KB smem
            (32,  32,  64, D32, D32,  D4, D8),    // max(8², 8×4, 4²) = 64, ~26KB smem
        )
    };
}

#[cfg(feature = "tile-tuning")]
macro_rules! supported_tile_configs {
    ($callback:ident!($($args:tt)*)) => {
        $callback!($($args)*
            // Base configs
            // (mini_batch, head_dim, threads, CS, F, CSR, FR)
            //
            // Note: Larger tiles (16x128, 32x64, 64x64) exceed GPU shared memory limits
            // and have been removed. Shared memory usage is approximately:
            //   2*(F*CS) + 4*(CS*F) + (CS*CS) + (F*F) elements
            // Most GPUs have 48-64KB shared memory limit per block.
            ( 8,  32,  64, D8,  D32,  D4, D4),    // max(2², 2×8, 8²) = 64, ~10KB smem
            ( 8,  64,  64, D8,  D64,  D4, D8),    // max(2², 2×8, 8²) = 64, ~22KB smem
            (16,  32,  16, D16, D32,  D4, D8),    // max(4², 4×4, 4²) = 16, ~14KB smem
            (16,  64, 256, D16, D64,  D4, D4),    // max(4², 4×16, 16²) = 256, ~30KB smem
            (32,  32,  64, D32, D32,  D4, D8),    // max(8², 8×4, 4²) = 64, ~26KB smem

            // Tuning configs - alternative thread counts for each tile size

            // 8x32 - test 4
            ( 8,  32,   4, D8,  D32,  D4, D16),   // max(2², 2×2, 2²) = 4

            // 8x64 - test 4
            ( 8,  64,   4, D8,  D64,  D4, D32),   // max(2², 2×2, 2²) = 4

            // 16x32 - test 4 and 64
            (16,  32,   4, D16, D32,  D8, D16),   // max(2², 2×2, 2²) = 4
            (16,  32,  64, D16, D32,  D4, D4),    // max(4², 4×8, 8²) = 64

            // 16x64 - test 16 and 64
            (16,  64,  16, D16, D64,  D4, D16),   // max(4², 4×4, 4²) = 16
            (16,  64,  64, D16, D64,  D4, D8),    // max(4², 4×8, 8²) = 64

            // 32x32 - test 4 and 16
            (32,  32,   4, D32, D32, D16, D16),   // max(2², 2×2, 2²) = 4
            (32,  32,  16, D32, D32,  D8,  D8),   // max(4², 4×4, 4²) = 16
        )
    };
}

// Note: macro_rules! macros are automatically available to submodules
// defined after the macro in the same module

/// Unified tile dispatch macro.
/// Args: kernel_ident, client, cube_count, mini_batch_len, head_dim, threads, ...rest
/// The client is passed to CubeDim::new for inferring max supported dimensions.
macro_rules! tile_dispatch {
    (
        $kernel:ident,
        $client:expr, $cube_count:expr,
        $mini_batch_len:expr, $head_dim:expr, $threads:expr
        $(, $($rest:tt)+)?
    ) => {
        supported_tile_configs!(tile_dispatch_inner!(
            $kernel, $client, $cube_count,
            $mini_batch_len, $head_dim, $threads;
            [$($($rest)+)?];
        ))
    };
}

macro_rules! tile_dispatch_inner {
    (
        $kernel:ident, $client:expr, $cube_count:expr,
        $mini_batch_len:expr, $head_dim:expr, $threads:expr;
        $rest:tt;
        $(($s:literal, $h:literal, $t:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty)),* $(,)?
    ) => {
        match ($mini_batch_len, $head_dim, $threads) {
            $(($s, $h, $t) => tile_dispatch_arm!($kernel, $client, $cube_count, $t, $CS, $F, $CSR, $FR, $rest),)*
            _ => panic!("Unsupported tile config: {}×{}@{}", $mini_batch_len, $head_dim, $threads)
        }
    };
}

macro_rules! tile_dispatch_arm {
    ($kernel:ident, $client:expr, $cube_count:expr, $t:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty, [$($rest:tt)*]) => {{
        type P<E> = Params<E, $CS, $F, $CSR, $FR>;
        $kernel::launch::<P<_>, _>(
            $client, $cube_count, CubeDim::new($client, $t), $($rest)*
        ).unwrap()
    }};
}

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
    FwdIntermediates, TttGradInputs, TttSavedTensors, TttTileKernel, TttTileMultiKernel, backward,
    backward_multi, forward, forward_multi, launch_tile_backward, launch_tile_backward_multi,
    launch_tile_forward, launch_tile_forward_multi,
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
