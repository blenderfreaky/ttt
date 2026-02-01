//! Device-to-device streaming TTT kernel.
//!
//! Uses device-to-device memory copies for communication between host and persistent kernel.

mod host;
mod kernel;
mod wrapper;

pub use host::{
    StreamingConfig, StreamingBufferTensors, TttStreamingState, get_or_create_streaming_state,
    remove_streaming_state, remove_streaming_state_by_id, shutdown_streaming_state,
};
pub use kernel::{
    CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY, CTRL_SHUTDOWN, StreamingBuffers,
    fused_ttt_streaming_kernel,
};
pub use wrapper::{
    FusedTileStreamingState, StreamHandle, StreamingBackward, StreamingKernelConfig,
    TttStreamingKernel, fused_ttt_streaming_forward, next_stream_id,
};
