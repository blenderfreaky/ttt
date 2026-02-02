//! Streaming TTT kernels for persistent GPU execution.
//!
//! This module contains two streaming implementations:
//! - `d2d`: Device-to-device memory copy based streaming
//! - `ptr`: Raw GPU pointer based streaming (zero-copy)

use std::fmt::Debug;

use ttt_kernels::tensor_bundle;

pub mod d2d;
pub mod ptr;

tensor_bundle! {
    /// Outputs for streaming TTT kernels.
    /// Contains 3 main outputs + 7 forward intermediates = 10 tensors.
    pub struct TttTileOutputs {
        output, weight_out, bias_out,
        x_hat_fused, std_fused, grad_output_fused, grad_x_hat_fused, grad_l_wrt_Z1, x_hat_ln, std_ln
    }
}

// Re-export commonly used items
pub use d2d::{
    CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY, CTRL_SHUTDOWN, FusedTileStreamingState,
    StreamingConfig, StreamingKernelConfig, TttStreamingKernel, TttStreamingState,
    fused_ttt_streaming_kernel, get_or_create_streaming_state,
};
pub use ptr::{
    CTRL_ARRAY_SIZE as PTR_CTRL_ARRAY_SIZE, FusedTilePtrStreamingState, PTR_TABLE_SIZE,
    PtrStreamingConfig, PtrStreamingKernelConfig, PtrStreamingTensors, STATUS_DONE, STATUS_IDLE,
    STATUS_READY, STATUS_SHUTDOWN, TttPtrStreamingKernel, TttPtrStreamingState,
    fused_ttt_streaming_ptr_kernel,
};
