//! FusedKernel implementation for the pointer-based streaming TTT-Linear kernel.
//!
//! This implements the FusedKernel trait for `TttPtrStreamingKernel`, which uses
//! a persistent GPU kernel with true zero-copy input via pointer tables.

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
use burn_cubecl::{CubeRuntime, FloatElement, kernel::into_contiguous, tensor::CubeTensor};
use tracing::trace;

use super::{
    launch::TttTileOutputs,
    streaming_ptr_host::{
        PtrStreamingConfig, get_or_create_ptr_streaming_state, remove_ptr_streaming_state_by_id,
    },
};
use crate::{Fused, FusedTttBackend, PtrStreamingKernel, ttt::TttInputs};
use ttt_core::{TTTConfig, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearState};
use ttt_kernels::kernel::{CanBackwardNoOut, FusedKernel};

/// Inner handle that cleans up the streaming state on drop.
#[derive(Debug)]
struct PtrStreamHandleInner(u64);

impl Drop for PtrStreamHandleInner {
    fn drop(&mut self) {
        remove_ptr_streaming_state_by_id(self.0);
    }
}

/// Handle that cleans up the streaming state when the last clone is dropped.
#[derive(Debug, Clone)]
pub struct PtrStreamHandle(Arc<PtrStreamHandleInner>);

impl PtrStreamHandle {
    pub fn new(stream_id: u64) -> Self {
        Self(Arc::new(PtrStreamHandleInner(stream_id)))
    }

    pub fn id(&self) -> u64 {
        self.0.0
    }
}

/// State for FusedTilePtrStreaming that wraps TTTLinearState and adds stream_id.
#[derive(burn::module::Module, Debug)]
pub struct FusedTilePtrStreamingState<B: FusedTttBackend> {
    /// The underlying linear state (weight and bias)
    pub inner: TTTLinearState<B>,
    /// Handle that cleans up on drop (not a module parameter)
    pub stream_handle: Ignored<PtrStreamHandle>,
}

impl<B: FusedTttBackend> FusedTilePtrStreamingState<B> {
    pub fn stream_id(&self) -> u64 {
        self.stream_handle.0.id()
    }
}

impl<B: FusedTttBackend> AsRef<TTTLinearState<B>> for FusedTilePtrStreamingState<B> {
    fn as_ref(&self) -> &TTTLinearState<B> {
        &self.inner
    }
}

/// Configuration for the ptr streaming kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrStreamingKernelConfig {
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

impl PtrStreamingKernelConfig {
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

/// Marker type for the pointer-based streaming TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttPtrStreamingKernel;

impl FusedKernel<9, 10> for TttPtrStreamingKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttTileOutputs<T>;
    type Backward = PtrStreamingBackward;
    type Config = PtrStreamingKernelConfig;

    fn forward_launch<R: CubeRuntime + 'static, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: PtrStreamingKernelConfig,
    ) -> TttTileOutputs<CubeTensor<R>> {
        let [batch_size, num_heads, _seq_len, head_dim] = inputs.xq.shape.dims();

        let ptr_config = PtrStreamingConfig::new(
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
        let state = get_or_create_ptr_streaming_state::<R, F>(
            config.stream_id,
            ptr_config,
            client.clone(),
            device.clone(),
            inputs.weight.clone(),
            inputs.bias.clone(),
            inputs.token_eta.clone(),
            inputs.ln_weight.clone(),
            inputs.ln_bias.clone(),
        );

        trace!("ptr streaming forward start");

        // Make input tensors contiguous if they have non-contiguous strides.
        // Sliced tensors with contiguous strides but an offset are handled by
        // feed_mini_batch which adds handle.offset_start to the GPU addresses.
        let xq = into_contiguous(inputs.xq);
        let xk = into_contiguous(inputs.xk);
        let xv = into_contiguous(inputs.xv);
        let ttt_lr_eta = into_contiguous(inputs.ttt_lr_eta);

        // Sync the default stream to ensure contiguous copies are complete
        // before the persistent kernel (on a different stream) reads from them.
        use thundercube::util::wait_for_sync;
        if let Err(e) = wait_for_sync(&client) {
            trace!("forward_launch sync warning: {:?}", e);
        }

        // Use pointer-based forward
        let output = state.forward_tensor(&xq, &xk, &xv, &ttt_lr_eta);

        trace!("ptr streaming forward complete");
        // Make a true copy of the output - the kernel reuses its buffer for each mini-batch
        // so we need to copy the data before it gets overwritten by the next call.
        // Note: into_contiguous skips copy if already contiguous, so we force a copy
        // using mul_scalar by 1.0 which allocates a new output buffer.
        use burn_cubecl::ops::numeric::mul_scalar;
        use cubecl::prelude::InputScalar;
        let dtype = output.dtype;
        let output = mul_scalar(output.clone(), InputScalar::new(1.0f32, dtype));

        TttTileOutputs {
            output,
            weight_out: state.tensors.weight.clone(),
            bias_out: state.tensors.bias.clone(),
            x_hat_fused: state.tensors.x_hat_fused.clone(),
            std_fused: state.tensors.std_fused.clone(),
            grad_output_fused: state.tensors.grad_output_fused.clone(),
            grad_x_hat_fused: state.tensors.grad_x_hat_fused.clone(),
            grad_l_wrt_Z1: state.tensors.grad_l_wrt_Z1.clone(),
            x_hat_ln: state.tensors.x_hat_ln.clone(),
            std_ln: state.tensors.std_ln.clone(),
        }
    }
}

/// Marker for ptr streaming backward - uses no saved outputs
pub struct PtrStreamingBackward;

impl CanBackwardNoOut<9, 10> for TttPtrStreamingKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        _inputs: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttTileOutputs<CubeTensor<R>>,
        _config: PtrStreamingKernelConfig,
    ) -> TttInputs<CubeTensor<R>> {
        panic!("Ptr streaming kernel backward not yet implemented")
    }
}

impl<K, const N: usize, const M: usize> ttt_kernels::kernel::BackwardImpl<K, N, M>
    for PtrStreamingBackward
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
// High-level API
// ============================================================================

/// Global counter for generating unique stream IDs for ptr streaming.
static PTR_STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(1_000_000);

/// Generate a unique stream ID for a new ptr streaming session.
pub fn next_ptr_stream_id() -> u64 {
    PTR_STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// High-level API for the ptr streaming TTT-Linear forward pass.
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_ptr_streaming_forward<B: FusedTttBackend>(
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

    let config =
        PtrStreamingKernelConfig::new(stream_id, mini_batch_len, head_dim, epsilon, threads);

    let outputs = <B as FusedKernelBackend<TttPtrStreamingKernel, 9, 10>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight_out)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias_out)),
    )
}

// ============================================================================
// TTTInnerModel implementation for ptr streaming kernel
// ============================================================================

/// TTTInnerModel implementation for the ptr streaming fused kernel.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, PtrStreamingKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = FusedTilePtrStreamingState<B>;

    fn name() -> &'static str {
        "FusedPtrStreamingTTTLinear"
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
        FusedTilePtrStreamingState {
            inner: self.inner.init_state(batch_size),
            stream_handle: Ignored(PtrStreamHandle::new(next_ptr_stream_id())),
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

        let (output, weight_updated, bias_updated) = fused_ttt_ptr_streaming_forward(
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
