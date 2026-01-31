//! FusedKernel implementation for the streaming TTT-Linear kernel.
//!
//! This implements the FusedKernel trait for `TttStreamingKernel`, which uses
//! a persistent GPU kernel with a global registry for state management.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

use burn::module::Ignored;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};
use tracing::trace;

use super::{
    launch::TttTileOutputs,
    streaming_host::{StreamingConfig, get_or_create_streaming_state, remove_streaming_state_by_id},
};
use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{
        Fused, FusedTttBackend,
        kernel::{FusedKernel, CanBackwardNoOut},
        ttt::TttInputs,
    },
    layer::{TTTInnerModel, TTTInputsInner},
    linear::{TTTLinear, TTTLinearState},
};

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
        let output = state.forward_d2d(
            &inputs.xq,
            &inputs.xk,
            &inputs.xv,
            &inputs.ttt_lr_eta,
        );

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
        trace!("streaming forward complete, output handle stream: {:?}", result.output.handle.stream);
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

impl<K, const N: usize, const M: usize> crate::ttt::cubecl_kernels::kernel::BackwardImpl<K, N, M>
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
    use crate::ttt::cubecl_kernels::kernel::FusedKernelBackend;

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
/// Uses quadruple Fused wrapper: `Fused<B, Fused<B, Fused<B, Fused<B, TTTLinear<B>>>>>`.
///
/// The streaming kernel maintains a persistent GPU kernel that processes
/// mini-batches incrementally, keeping weight/bias in shared memory between calls.
impl<B: FusedTttBackend> TTTInnerModel<B>
    for Fused<B, Fused<B, Fused<B, Fused<B, TTTLinear<B>>>>>
{
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
        Fused {
            inner: Fused {
                inner: Fused {
                    inner: Fused {
                        inner: TTTLinear::new(general_config, config, device),
                        _backend: PhantomData,
                    },
                    _backend: PhantomData,
                },
                _backend: PhantomData,
            },
            _backend: PhantomData,
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.inner.inner.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        FusedTileStreamingState {
            inner: self.inner.inner.inner.inner.init_state(batch_size),
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

        let inner = &self.inner.inner.inner.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = inner.get_config();
        let threads = inner_config.threads
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
    use std::sync::Arc;

    use burn::{
        module::{Ignored, Param},
        tensor::{Tensor, TensorData},
    };
    use burn_backend::Backend;

    use super::*;
    use crate::ttt::{
        CpuBackend, GpuBackend,
        cubecl_kernels::Fused,
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
        util::MultiHeadLayerNorm,
    };

    fn assert_data_close(a: &[f32], b: &[f32], rtol: f32, atol: f32, name: &str) {
        assert_eq!(a.len(), b.len(), "{name}: Data sizes don't match");

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        let mut max_av = 0.0f32;
        let mut max_bv = 0.0f32;

        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (av - bv).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
                max_av = av;
                max_bv = bv;
            }
        }

        let tolerance = atol + rtol * max_bv.abs();
        assert!(
            max_diff <= tolerance,
            "{name}: Max mismatch at index {max_idx}: {max_av} vs {max_bv} (diff: {max_diff}, tolerance: {tolerance})",
        );
    }

    #[test]
    fn test_streaming_vs_ttt_linear() {
        // Initialize tracing for tests (ignore if already initialized)
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();

        // Test streaming kernel against CPU reference
        // Use 8x32 tiles (same as tiled kernel tests)
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 32usize;
        let seq_len = 8usize; // single mini-batch
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f64;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon,
            threads: Some(8), // 8×32 tile config requires 8 threads
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // token_eta is constant across iterations
        let token_eta_cpu: Tensor<CpuBackend, 1> =
            Tensor::arange(1..(seq_len as i64 + 1), &cpu_device)
                .float()
                .recip();
        let token_eta_data: Vec<f32> = token_eta_cpu.to_data().to_vec().unwrap();

        // Create CPU reference implementation
        let ttt_linear_cpu: TTTLinear<CpuBackend> =
            TTTLinear::new(&config, &linear_config, &cpu_device);
        let mut state_cpu = ttt_linear_cpu.init_state(batch_size);

        // Get weight/bias/ln params for GPU
        let weight_init_data: Vec<f32> =
            ttt_linear_cpu.weight_init.val().to_data().to_vec().unwrap();
        let bias_init_data: Vec<f32> = ttt_linear_cpu.bias_init.val().to_data().to_vec().unwrap();
        let ln_weight_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .weight
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        // Create GPU tensors
        let gpu_device: <GpuBackend as Backend>::Device = Default::default();

        // Create GPU model with same weights as CPU reference
        let ttt_linear_gpu: TTTLinear<GpuBackend> = TTTLinear {
            weight_init: Param::from_tensor(Tensor::from_data(
                TensorData::new(weight_init_data, [num_heads, head_dim, head_dim]),
                &gpu_device,
            )),
            bias_init: Param::from_tensor(Tensor::from_data(
                TensorData::new(bias_init_data, [num_heads, head_dim]),
                &gpu_device,
            )),
            layer_norm: MultiHeadLayerNorm {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(ln_weight_data, [num_heads, head_dim]),
                    &gpu_device,
                )),
                bias: Param::from_tensor(Tensor::from_data(
                    TensorData::new(ln_bias_data, [num_heads, head_dim]),
                    &gpu_device,
                )),
                epsilon,
            },
            config: Ignored(config),
        };

        // Create quadruple-Fused wrapper for streaming kernel
        let inner_fused: Fused<GpuBackend, TTTLinear<GpuBackend>> = ttt_linear_gpu.into();
        let inner_fused2: Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>> =
            inner_fused.into();
        let inner_fused3: Fused<
            GpuBackend,
            Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>>,
        > = inner_fused2.into();
        let fused_streaming: Fused<
            GpuBackend,
            Fused<GpuBackend, Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>>>,
        > = inner_fused3.into();

        let mut streaming_state = fused_streaming.init_state(batch_size);

        use cubecl::prelude::ComputeClient;
        let client = ComputeClient::<cubecl::hip::HipRuntime>::load(&gpu_device);

        // Run multiple iterations to verify persistent kernel works across forward calls
        let num_iterations = 2;
        for iter in 0..num_iterations {
            trace!("[TEST] === Iteration {} ===", iter + 1);

            // Generate fresh random inputs for each iteration
            let xq_cpu: Tensor<CpuBackend, 4> = Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &cpu_device,
            );
            let xk_cpu: Tensor<CpuBackend, 4> = Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &cpu_device,
            );
            let xv_cpu: Tensor<CpuBackend, 4> = Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &cpu_device,
            );
            let ttt_lr_eta_cpu: Tensor<CpuBackend, 3> = Tensor::random(
                [batch_size, num_heads, seq_len],
                burn::tensor::Distribution::Uniform(0.01, 0.05),
                &cpu_device,
            );

            // Run CPU reference (state carries over between iterations)
            let inputs_cpu = TTTInputsInner {
                qkv: Qkv {
                    xq: xq_cpu.clone(),
                    xk: xk_cpu.clone(),
                    xv: xv_cpu.clone(),
                },
                token_eta: token_eta_cpu.clone(),
                ttt_lr_eta: ttt_lr_eta_cpu.clone(),
                start_idx: iter * seq_len,
            };
            let output_ref = ttt_linear_cpu.forward(&mut state_cpu, inputs_cpu);
            let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();

            // Create GPU input tensors
            let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
            let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
            let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
            let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

            let xq_gpu: Tensor<GpuBackend, 4> = Tensor::from_data(
                TensorData::new(xq_data, [batch_size, num_heads, seq_len, head_dim]),
                &gpu_device,
            );
            let xk_gpu: Tensor<GpuBackend, 4> = Tensor::from_data(
                TensorData::new(xk_data, [batch_size, num_heads, seq_len, head_dim]),
                &gpu_device,
            );
            let xv_gpu: Tensor<GpuBackend, 4> = Tensor::from_data(
                TensorData::new(xv_data, [batch_size, num_heads, seq_len, head_dim]),
                &gpu_device,
            );
            let token_eta_gpu: Tensor<GpuBackend, 1> =
                Tensor::from_data(TensorData::new(token_eta_data.clone(), [seq_len]), &gpu_device);
            let ttt_lr_eta_gpu: Tensor<GpuBackend, 3> = Tensor::from_data(
                TensorData::new(ttt_lr_eta_data, [batch_size, num_heads, seq_len]),
                &gpu_device,
            );

            let inputs_gpu = TTTInputsInner {
                qkv: Qkv {
                    xq: xq_gpu,
                    xk: xk_gpu,
                    xv: xv_gpu,
                },
                token_eta: token_eta_gpu,
                ttt_lr_eta: ttt_lr_eta_gpu,
                start_idx: iter * seq_len,
            };

            // Run GPU forward (state carries over between iterations)
            trace!("[TEST] calling forward (iter {})...", iter + 1);
            let output_streaming = fused_streaming.forward(&mut streaming_state, inputs_gpu);
            trace!("[TEST] forward returned, output shape: {:?}", output_streaming.shape());

            // Sync and compare
            thundercube::util::wait_for_sync(&client).expect("sync failed");
            let output_streaming_data: Vec<f32> = output_streaming.to_data().to_vec().unwrap();

            // Compute max diff and correlation
            let mut max_diff = 0.0f32;
            let mut sum_product = 0.0f32;
            let mut sum_sq_a = 0.0f32;
            let mut sum_sq_b = 0.0f32;
            for (&a, &b) in output_streaming_data.iter().zip(output_ref_data.iter()) {
                max_diff = max_diff.max((a - b).abs());
                sum_product += a * b;
                sum_sq_a += a * a;
                sum_sq_b += b * b;
            }
            let correlation = sum_product / (sum_sq_a.sqrt() * sum_sq_b.sqrt());
            trace!("Iteration {} - Output max diff: {}, correlation: {}", iter + 1, max_diff, correlation);

            // Compare outputs
            assert_data_close(
                &output_streaming_data,
                &output_ref_data,
                0.5,
                0.2,
                &format!("Iteration {}: output", iter + 1),
            );
        }

        trace!("D2D streaming vs CPU ref test passed with {} iterations!", num_iterations);
    }
}
