//! Streaming TTT kernels for persistent GPU execution.
//!
//! This module contains two streaming implementations:
//! - `d2d`: Device-to-device memory copy based streaming
//! - `ptr`: Raw GPU pointer based streaming (zero-copy)

pub mod d2d;
pub mod ptr;

// Re-export commonly used items
pub use d2d::{
    CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY, CTRL_SHUTDOWN, FusedTileStreamingState,
    StreamingConfig, StreamingKernelConfig, TttStreamingKernel, TttStreamingState,
    fused_ttt_streaming_kernel, get_or_create_streaming_state,
};
pub use ptr::{
    CTRL_ARRAY_SIZE as PTR_CTRL_ARRAY_SIZE, PTR_TABLE_SIZE, STATUS_DONE, STATUS_IDLE,
    STATUS_READY, STATUS_SHUTDOWN, FusedTilePtrStreamingState, PtrStreamingConfig,
    PtrStreamingKernelConfig, PtrStreamingTensors, TttPtrStreamingKernel, TttPtrStreamingState,
    fused_ttt_streaming_ptr_kernel,
};
