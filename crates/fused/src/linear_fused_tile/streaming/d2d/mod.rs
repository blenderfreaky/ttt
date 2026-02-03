//! Device-to-device streaming TTT kernel.
//!
//! Uses device-to-device memory copies for communication between host and persistent kernel.

mod host;
mod kernel;
mod wrapper;

#[cfg(test)]
mod tests;

pub use host::{
    D2dStreamingBufferTensors, D2dStreamingConfig, TttD2dStreamingState,
    get_or_create_d2d_streaming_state, remove_d2d_streaming_state,
    remove_d2d_streaming_state_by_id, shutdown_d2d_streaming_state,
};
pub use kernel::{
    CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY, CTRL_SHUTDOWN, D2dStreamingBuffers,
    fused_ttt_d2d_streaming_kernel,
};
pub use wrapper::{
    D2dStreamingKernelConfig, FusedTileD2dStreamingState, StreamHandle, TttD2dStreamingKernel,
    fused_ttt_d2d_streaming_forward, next_stream_id,
};
