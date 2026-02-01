//! FusedKernel implementation for the streaming TTT-Linear kernel.
//!
//! This implements the FusedKernel trait for `TttStreamingKernel`, which uses
//! a persistent GPU kernel with a global registry for state management.

use std::{
    fmt::Debug,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use burn::{
    module::Ignored,
    tensor::{Tensor, TensorPrimitive},
};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};
use tracing::trace;
use ttt_core::{TTTConfig, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearState};
use ttt_kernels::kernel::{CanBackwardNoOut, FusedKernel};

use super::{
    launch::TttTileOutputs,
    streaming_host::{
        StreamingConfig, get_or_create_streaming_state, remove_streaming_state_by_id,
    },
};
use crate::{Fused, FusedTttBackend, StreamingKernel, ttt::TttInputs};

/// Inner handle that cleans up the streaming state on drop.
#[derive(Debug)]
struct StreamHandleInner(u64);

impl Drop for StreamHandleInner {
    fn drop(&mut self) {
        remove_streaming_state_by_id(self.0);
    }
}

/// Handle that cleans up the streaming state when the last clone is dropped.
#[derive(Debug, Clone)]
pub struct StreamHandle(Arc<StreamHandleInner>);

impl StreamHandle {
    pub fn new(stream_id: u64) -> Self {
        Self(Arc::new(StreamHandleInner(stream_id)))
    }

    pub fn id(&self) -> u64 {
        self.0.0
    }
}

/// State for FusedTileStreaming that wraps TTTLinearState and adds stream_id.
#[derive(burn::module::Module, Debug)]
pub struct FusedTileStreamingState<B: FusedTttBackend> {
    /// The underlying linear state (weight and bias)
    pub inner: TTTLinearState<B>,
    /// Handle that cleans up on drop (not a module parameter)
    pub stream_handle: Ignored<StreamHandle>,
}

impl<B: FusedTttBackend> FusedTileStreamingState<B> {
    pub fn stream_id(&self) -> u64 {
        self.stream_handle.0.id()
    }
}

impl<B: FusedTttBackend> AsRef<TTTLinearState<B>> for FusedTileStreamingState<B> {
    fn as_ref(&self) -> &TTTLinearState<B> {
        &self.inner
    }
}

/// Configuration for the streaming kernel.
/// Extends FusedTttConfig with a stream_id for registry lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamingKernelConfig {
    /// Unique stream identifier for registry lookup
    pub stream_id: u64,
    /// Mini-batch sequence length (CS)
    pub mini_batch_len: usize,
    /// Head dimension (F)
    pub head_dim: usize,
    /// Layer norm epsilon, stored as scaled integer
    pub epsilon_scaled: u32,
    /// Number of threads per cube
    pub threads: usize,
}

impl StreamingKernelConfig {
    pub fn new(
        stream_id: u64,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            stream_id,
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / 1e-9) as u32,
            threads,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * 1e-9
    }
}

/// Marker type for the streaming TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttStreamingKernel;

impl FusedKernel<9, 10> for TttStreamingKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttTileOutputs<T>;
    type Backward = StreamingBackward;
    type Config = StreamingKernelConfig;

    fn forward_launch<R: CubeRuntime + 'static, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: StreamingKernelConfig,
    ) -> TttTileOutputs<CubeTensor<R>> {
        let [batch_size, num_heads, _seq_len, head_dim] = inputs.xq.shape.dims();

        let streaming_config = StreamingConfig::new(
            config.stream_id,
            batch_size,
            num_heads,
            config.mini_batch_len,
            head_dim,
            config.epsilon(),
            config.threads,
        );

        let client = inputs.xq.client.clone();
        let device = inputs.xq.device.clone();

        // Get or create the streaming state from the global registry
        let state = get_or_create_streaming_state::<R, F>(
            streaming_config,
            client.clone(),
            device.clone(),
            inputs.weight.clone(),
            inputs.bias.clone(),
            inputs.token_eta.clone(),
            inputs.ln_weight.clone(),
            inputs.ln_bias.clone(),
        );

        trace!("streaming forward_d2d start");
        // Use D2D copy to feed inputs to the streaming kernel (no CPU round-trip)
        let output = state.forward_d2d(&inputs.xq, &inputs.xk, &inputs.xv, &inputs.ttt_lr_eta);

        trace!("streaming forward_d2d complete, cloning output");
        // Clone output tensor since we're returning ownership
        let output = output.clone();

        // Return outputs - use result buffers which can be read without blocking
        let result = TttTileOutputs {
            output,
            weight_out: state.tensors.result_weight.clone(),
            bias_out: state.tensors.result_bias.clone(),
            // Forward intermediates from the streaming state
            x_hat_fused: state.tensors.x_hat_fused.clone(),
            std_fused: state.tensors.std_fused.clone(),
            grad_output_fused: state.tensors.grad_output_fused.clone(),
            grad_x_hat_fused: state.tensors.grad_x_hat_fused.clone(),
            grad_l_wrt_Z1: state.tensors.grad_l_wrt_Z1.clone(),
            x_hat_ln: state.tensors.x_hat_ln.clone(),
            std_ln: state.tensors.std_ln.clone(),
        };
        trace!(
            "streaming forward complete, output handle stream: {:?}",
            result.output.handle.stream
        );
        result
    }
}

/// Marker for streaming backward - uses no saved outputs (recomputes if needed)
pub struct StreamingBackward;

impl CanBackwardNoOut<9, 10> for TttStreamingKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        _inputs: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttTileOutputs<CubeTensor<R>>,
        _config: StreamingKernelConfig,
    ) -> TttInputs<CubeTensor<R>> {
        // TODO: Implement streaming backward
        // For now, panic - streaming is forward-only
        panic!("Streaming kernel backward not yet implemented")
    }
}

impl<K, const N: usize, const M: usize> ttt_kernels::kernel::BackwardImpl<K, N, M>
    for StreamingBackward
where
    K: CanBackwardNoOut<N, M>,
{
    fn should_save_outputs() -> bool {
        false
    }

    fn call<R: CubeRuntime, F: FloatElement>(
        inputs: K::Inputs<CubeTensor<R>>,
        _outputs: Option<K::Outputs<CubeTensor<R>>>,
        grad_outputs: K::Outputs<CubeTensor<R>>,
        config: K::Config,
    ) -> K::Inputs<CubeTensor<R>> {
        K::backward_no_out::<R, F>(inputs, grad_outputs, config)
    }
}

// ============================================================================
// High-level API for streaming kernel
// ============================================================================

/// Global counter for generating unique stream IDs.
static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique stream ID for a new streaming session.
pub fn next_stream_id() -> u64 {
    STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// High-level API for the streaming TTT-Linear forward pass.
///
/// This function takes burn Tensors, converts them to CubeTensors,
/// calls the streaming kernel, and returns burn Tensors.
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_streaming_forward<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    stream_id: u64,
    mini_batch_len: usize,
    head_dim: usize,
    epsilon: f32,
    threads: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    use ttt_kernels::FusedKernelBackend;

    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let config = StreamingKernelConfig::new(stream_id, mini_batch_len, head_dim, epsilon, threads);

    let outputs = <B as FusedKernelBackend<TttStreamingKernel, 9, 10>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight_out)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias_out)),
    )
}

// ============================================================================
// TTTInnerModel implementation for streaming kernel
// ============================================================================

/// TTTInnerModel implementation for the streaming fused kernel.
///
/// The streaming kernel maintains a persistent GPU kernel that processes
/// mini-batches incrementally, keeping weight/bias in shared memory between calls.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, StreamingKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = FusedTileStreamingState<B>;

    fn name() -> &'static str {
        "FusedStreamingTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused::new(TTTLinear::new(general_config, config, device))
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        FusedTileStreamingState {
            inner: self.inner.init_state(batch_size),
            stream_handle: Ignored(StreamHandle::new(next_stream_id())),
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = inner.get_config();
        let threads = inner_config
            .threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));

        let (output, weight_updated, bias_updated) = fused_ttt_streaming_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.inner.weight.clone(),
            state.inner.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            state.stream_id(),
            inner_config.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

        state.inner.weight = weight_updated;
        state.inner.bias = bias_updated;

        output
    }
}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;
    use crate::FusedTileStreaming;
    use ttt_core::{GpuBackend, test_utils::{TestDims, test_fwd}};

    // TODO: streaming kernel has issues, needs investigation
    #[test]
    #[ignore]
    fn test_streaming_vs_ttt_linear() {
        let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX.lock().unwrap();

        let dims = TestDims::new(2, 2, 32, 8).with_iterations(2);
        test_fwd::<
            GpuBackend,
            FusedTileStreaming<GpuBackend>,
            FusedTileStreamingState<GpuBackend>,
            _,
        >(dims, |m| m.into(), 1e-3, 1e-3, "Streaming");
    }
}
