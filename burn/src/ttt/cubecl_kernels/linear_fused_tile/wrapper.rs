//! TTTInnerModel implementation for the tiled TTT-Linear kernel.

use std::{marker::PhantomData, sync::Arc};

use burn::tensor::Tensor;

use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{Fused, FusedTttBackend},
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};

use super::api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};

/// TTTInnerModel implementation for the tiled fused kernel.
/// Uses double Fused wrapper: `Fused<B, Fused<B, TTTLinear<B>>>`.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, Fused<B, TTTLinear<B>>> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused {
            inner: Fused {
                inner: TTTLinear::new(general_config, config, device),
                _backend: PhantomData,
            },
            _backend: PhantomData,
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.inner.init_state(batch_size)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let inner = &self.inner.inner;

        let qkv = inputs.qkv;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            epsilon,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

/// TTTInnerModel implementation for the multi-stage tiled fused kernel.
/// Uses triple Fused wrapper: `Fused<B, Fused<B, Fused<B, TTTLinear<B>>>>`.
///
/// This implementation overrides `forward()` to process all mini-batches in a
/// single kernel launch, rather than launching the kernel once per mini-batch.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, Fused<B, Fused<B, TTTLinear<B>>>> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileMultiTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused {
            inner: Fused {
                inner: Fused {
                    inner: TTTLinear::new(general_config, config, device),
                    _backend: PhantomData,
                },
                _backend: PhantomData,
            },
            _backend: PhantomData,
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.inner.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.inner.inner.init_state(batch_size)
    }

    /// Override forward to use multi-stage kernel for full sequence processing.
    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let inner = &self.inner.inner.inner;
        let config = inner.get_config();
        let mini_batch_size = config.mini_batch_size;

        let [_batch_size, _num_heads, seq_len, _head_dim] = inputs.qkv.xv.shape().dims();
        let num_full_batches = seq_len / mini_batch_size;
        let remainder = seq_len % mini_batch_size;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        if num_full_batches > 0 {
            // Process full mini-batches with multi-stage kernel
            let full_seq_len = num_full_batches * mini_batch_size;
            let full_qkv = inputs.qkv.slice_seq(0..full_seq_len);
            let [batch_size, num_heads, _] = inputs.ttt_lr_eta.shape().dims();
            let full_ttt_lr_eta =
                inputs
                    .ttt_lr_eta
                    .clone()
                    .slice([0..batch_size, 0..num_heads, 0..full_seq_len]);

            // token_eta is [mini_batch_size] and constant across stages - pass directly
            let (output, weight_updated, bias_updated) = fused_ttt_tile_forward_multi::<B>(
                full_qkv.xq,
                full_qkv.xk,
                full_qkv.xv,
                state.weight.clone(),
                state.bias.clone(),
                inputs.token_eta.clone(),
                full_ttt_lr_eta,
                ln_weight.clone(),
                ln_bias.clone(),
                epsilon,
                mini_batch_size,
            );

            state.weight = weight_updated;
            state.bias = bias_updated;

            if remainder == 0 {
                output
            } else {
                // Process remainder with single-stage kernel
                let remainder_inputs = inputs.slice_seq(full_seq_len..seq_len);
                let remainder_output = self.forward_mini_batch(state, remainder_inputs);

                // Concatenate outputs
                Tensor::cat(vec![output, remainder_output], 2)
            }
        } else {
            // Sequence shorter than mini_batch_size, use single-stage kernel
            self.forward_mini_batch(state, inputs)
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let inner = &self.inner.inner.inner;

        let qkv = inputs.qkv;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            epsilon,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttt::{
        CpuBackend, GpuAutodiffBackend, GpuBackend,
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
        util::MultiHeadLayerNorm,
    };
    use burn::{
        backend::Autodiff,
        module::{Ignored, Param},
        tensor::TensorData,
    };
    use burn_backend::Backend;

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
    fn test_fused_tile_vs_ttt_linear() {
        // Use 8x64 tiles (CS=8, F=32)
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 32usize;
        let seq_len = 8usize;
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f64;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon,
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // Create random input tensors on CPU
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
        let token_eta_cpu: Tensor<CpuBackend, 1> =
            Tensor::arange(1..(seq_len as i64 + 1), &cpu_device)
                .float()
                .recip();
        let ttt_lr_eta_cpu: Tensor<CpuBackend, 3> = Tensor::random(
            [batch_size, num_heads, seq_len],
            burn::tensor::Distribution::Uniform(0.01, 0.05),
            &cpu_device,
        );

        // Get data as vectors for transfer
        let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

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

        // Run CPU reference
        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu.clone(),
                xk: xk_cpu.clone(),
                xv: xv_cpu.clone(),
            },
            token_eta: token_eta_cpu.clone(),
            ttt_lr_eta: ttt_lr_eta_cpu.clone(),
            start_idx: 0,
        };

        let output_ref = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, inputs_cpu);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();
        let weight_ref_data: Vec<f32> = state_cpu.weight.to_data().to_vec().unwrap();
        let bias_ref_data: Vec<f32> = state_cpu.bias.to_data().to_vec().unwrap();

        // Create GPU tensors
        let gpu_device: <GpuBackend as Backend>::Device = Default::default();

        let ttt_linear: TTTLinear<GpuBackend> = TTTLinear {
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
        let inner_fused: Fused<GpuBackend, TTTLinear<GpuBackend>> = ttt_linear.into();
        let fused_tile: Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>> =
            inner_fused.into();

        let mut fused_state = fused_tile.init_state(batch_size);

        // Create GPU input tensors
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
            Tensor::from_data(TensorData::new(token_eta_data, [seq_len]), &gpu_device);
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
            start_idx: 0,
        };

        // Run tiled fused kernel
        let output_fused = fused_tile.forward_mini_batch(&mut fused_state, inputs_gpu);
        let output_fused_data: Vec<f32> = output_fused.to_data().to_vec().unwrap();
        let weight_fused_data: Vec<f32> = fused_state.weight.to_data().to_vec().unwrap();
        let bias_fused_data: Vec<f32> = fused_state.bias.to_data().to_vec().unwrap();

        // Compare outputs
        assert_data_close(
            &output_fused_data,
            &output_ref_data,
            1e-2, // rtol - slightly looser for GPU
            1e-3, // atol
            "FusedTileTTTLinear vs TTTLinear output",
        );

        // Compare weight updates
        assert_data_close(
            &weight_fused_data,
            &weight_ref_data,
            1e-2,
            1e-3,
            "FusedTileTTTLinear vs TTTLinear weight update",
        );

        // Compare bias updates
        assert_data_close(
            &bias_fused_data,
            &bias_ref_data,
            1e-2,
            1e-3,
            "FusedTileTTTLinear vs TTTLinear bias update",
        );
    }

    #[test]
    fn test_fused_tile_multi_vs_ttt_linear() {
        // Test multi-stage kernel: 4 mini-batches of 8 tokens each = 64 total
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 32usize;
        let mini_batch_size = 8usize;
        let num_stages = 4usize;
        let seq_len = mini_batch_size * num_stages;
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f64;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size,
            base_lr: 1.0,
            epsilon,
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // Create random input tensors on CPU
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
        // token_eta is [mini_batch_size] repeated for CPU (slice_seq needs full length)
        // GPU kernel only reads first mini_batch_size elements
        let token_eta_base: Tensor<CpuBackend, 1> =
            Tensor::arange(1..(mini_batch_size as i64 + 1), &cpu_device)
                .float()
                .recip();
        let token_eta_cpu = token_eta_base.clone().repeat_dim(0, num_stages);
        let ttt_lr_eta_cpu: Tensor<CpuBackend, 3> = Tensor::random(
            [batch_size, num_heads, seq_len],
            burn::tensor::Distribution::Uniform(0.01, 0.05),
            &cpu_device,
        );

        // Get data as vectors for transfer
        let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
        // GPU gets only [mini_batch_size] token_eta (kernel reuses for all stages)
        let token_eta_data: Vec<f32> = token_eta_base.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

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

        // Run CPU reference - process all mini-batches sequentially
        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu.clone(),
                xk: xk_cpu.clone(),
                xv: xv_cpu.clone(),
            },
            token_eta: token_eta_cpu.clone(),
            ttt_lr_eta: ttt_lr_eta_cpu.clone(),
            start_idx: 0,
        };

        // Use forward() which processes all mini-batches
        let output_ref = ttt_linear_cpu.forward(&mut state_cpu, inputs_cpu);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();
        let weight_ref_data: Vec<f32> = state_cpu.weight.to_data().to_vec().unwrap();
        let bias_ref_data: Vec<f32> = state_cpu.bias.to_data().to_vec().unwrap();

        // Create GPU tensors
        let gpu_device: <GpuBackend as Backend>::Device = Default::default();

        let ttt_linear: TTTLinear<GpuBackend> = TTTLinear {
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

        // Create triple-Fused wrapper for multi-stage kernel
        let inner_fused: Fused<GpuBackend, TTTLinear<GpuBackend>> = ttt_linear.into();
        let inner_fused2: Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>> =
            inner_fused.into();
        let fused_tile_multi: Fused<
            GpuBackend,
            Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>>,
        > = inner_fused2.into();

        let mut fused_state = fused_tile_multi.init_state(batch_size);

        // Create GPU input tensors
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
        let token_eta_gpu: Tensor<GpuBackend, 1> = Tensor::from_data(
            TensorData::new(token_eta_data, [mini_batch_size]),
            &gpu_device,
        );
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
            start_idx: 0,
        };

        // Run multi-stage tiled fused kernel via forward()
        let output_fused = fused_tile_multi.forward(&mut fused_state, inputs_gpu);
        let output_fused_data: Vec<f32> = output_fused.to_data().to_vec().unwrap();
        let weight_fused_data: Vec<f32> = fused_state.weight.to_data().to_vec().unwrap();
        let bias_fused_data: Vec<f32> = fused_state.bias.to_data().to_vec().unwrap();

        // Compare outputs
        assert_data_close(
            &output_fused_data,
            &output_ref_data,
            1e-2,
            1e-3,
            "FusedTileMultiTTTLinear vs TTTLinear output",
        );

        // Compare weight updates
        assert_data_close(
            &weight_fused_data,
            &weight_ref_data,
            1e-2,
            1e-3,
            "FusedTileMultiTTTLinear vs TTTLinear weight update",
        );

        // Compare bias updates
        assert_data_close(
            &bias_fused_data,
            &bias_ref_data,
            1e-2,
            1e-3,
            "FusedTileMultiTTTLinear vs TTTLinear bias update",
        );
    }

    #[test]
    fn test_fused_tile_backward_gradients_vs_reference() {
        // Test backward gradients of tiled kernel against CPU autodiff reference
        // Use 8x32 tiles (CS=8, F=32) - smaller tiles to fit in shared memory
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 32usize;
        let seq_len = 8usize;
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f64;

        // Seed RNG for deterministic results
        let cpu_device: <CpuBackend as Backend>::Device = Default::default();
        CpuBackend::seed(&cpu_device, 42);

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon,
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        // Create CPU reference with autodiff
        let ttt_linear_cpu: TTTLinear<Autodiff<CpuBackend>> =
            TTTLinear::new(&config, &linear_config, &cpu_device);
        let mut state_cpu = ttt_linear_cpu.init_state(batch_size);

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
        let ln_weight_data_copy = ln_weight_data.clone();
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        // Create input tensors with gradients
        let xq_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let xk_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let xv_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let token_eta_cpu: Tensor<Autodiff<CpuBackend>, 1> =
            Tensor::arange(1..(seq_len as i64 + 1), &cpu_device)
                .float()
                .recip();
        let ttt_lr_eta_cpu: Tensor<Autodiff<CpuBackend>, 3> = Tensor::random(
            [batch_size, num_heads, seq_len],
            burn::tensor::Distribution::Uniform(0.01, 0.05),
            &cpu_device,
        )
        .require_grad();

        // Extract data for GPU
        let xq_data: Vec<f32> = xq_cpu.clone().inner().to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.clone().inner().to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.clone().inner().to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.clone().inner().to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.clone().inner().to_data().to_vec().unwrap();

        // Run CPU forward and backward
        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu.clone(),
                xk: xk_cpu.clone(),
                xv: xv_cpu.clone(),
            },
            token_eta: token_eta_cpu,
            ttt_lr_eta: ttt_lr_eta_cpu.clone(),
            start_idx: 0,
        };

        let output_cpu = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, inputs_cpu);
        let loss_cpu = output_cpu.sum();
        let grads_cpu = loss_cpu.backward();

        let grad_xq_cpu: Vec<f32> = xq_cpu.grad(&grads_cpu).unwrap().to_data().to_vec().unwrap();
        let grad_xk_cpu: Vec<f32> = xk_cpu.grad(&grads_cpu).unwrap().to_data().to_vec().unwrap();
        let grad_xv_cpu: Vec<f32> = xv_cpu.grad(&grads_cpu).unwrap().to_data().to_vec().unwrap();
        let grad_ttt_lr_eta_cpu: Vec<f32> = ttt_lr_eta_cpu
            .grad(&grads_cpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();

        // Create GPU tensors for tiled kernel with autodiff
        let gpu_device: <GpuAutodiffBackend as Backend>::Device = Default::default();

        let ttt_linear: TTTLinear<GpuAutodiffBackend> = TTTLinear {
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

        // Create double-Fused wrapper for tiled kernel
        let inner_fused: Fused<GpuAutodiffBackend, TTTLinear<GpuAutodiffBackend>> =
            ttt_linear.into();
        let fused_tile: Fused<
            GpuAutodiffBackend,
            Fused<GpuAutodiffBackend, TTTLinear<GpuAutodiffBackend>>,
        > = inner_fused.into();

        let mut fused_state = fused_tile.init_state(batch_size);

        // Create GPU input tensors with gradients
        let xq_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xq_data, [batch_size, num_heads, seq_len, head_dim]),
            &gpu_device,
        )
        .require_grad();
        let xk_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xk_data, [batch_size, num_heads, seq_len, head_dim]),
            &gpu_device,
        )
        .require_grad();
        let xv_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xv_data, [batch_size, num_heads, seq_len, head_dim]),
            &gpu_device,
        )
        .require_grad();
        let token_eta_gpu: Tensor<GpuAutodiffBackend, 1> =
            Tensor::from_data(TensorData::new(token_eta_data, [seq_len]), &gpu_device);
        let ttt_lr_eta_gpu: Tensor<GpuAutodiffBackend, 3> = Tensor::from_data(
            TensorData::new(ttt_lr_eta_data, [batch_size, num_heads, seq_len]),
            &gpu_device,
        )
        .require_grad();

        let inputs_gpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_gpu.clone(),
                xk: xk_gpu.clone(),
                xv: xv_gpu.clone(),
            },
            token_eta: token_eta_gpu,
            ttt_lr_eta: ttt_lr_eta_gpu.clone(),
            start_idx: 0,
        };

        // Run tiled forward and backward
        let output_gpu = fused_tile.forward_mini_batch(&mut fused_state, inputs_gpu);
        let loss_gpu = output_gpu.sum();
        let grads_gpu = loss_gpu.backward();

        let grad_xq_gpu: Vec<f32> = xq_gpu.grad(&grads_gpu).unwrap().to_data().to_vec().unwrap();
        let grad_xk_gpu: Vec<f32> = xk_gpu.grad(&grads_gpu).unwrap().to_data().to_vec().unwrap();
        let grad_xv_gpu: Vec<f32> = xv_gpu.grad(&grads_gpu).unwrap().to_data().to_vec().unwrap();
        let grad_ttt_lr_eta_gpu: Vec<f32> = ttt_lr_eta_gpu
            .grad(&grads_gpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();

        // Debug: print ln_weight and verify it matches GPU
        println!("\n=== Inputs Analysis ===");
        println!("ln_weight[0..5]: {:?}", &ln_weight_data_copy[0..5]);

        // Debug: print tolerance analysis and ratios for all gradients
        println!("\n=== Gradient Analysis ===");

        // Analyze xq
        let xq_ratios: Vec<f32> = grad_xq_cpu
            .iter()
            .zip(grad_xq_gpu.iter())
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, g)| g / c)
            .collect();
        let xq_mean_ratio: f32 = xq_ratios.iter().sum::<f32>() / xq_ratios.len() as f32;
        println!("grad_xq: mean ratio GPU/CPU = {:.4}", xq_mean_ratio);
        println!("grad_xq_cpu[0..5]: {:?}", &grad_xq_cpu[0..5]);
        println!("grad_xq_gpu[0..5]: {:?}", &grad_xq_gpu[0..5]);

        // Analyze xk
        let xk_ratios: Vec<f32> = grad_xk_cpu
            .iter()
            .zip(grad_xk_gpu.iter())
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, g)| g / c)
            .collect();
        let xk_mean_ratio: f32 = xk_ratios.iter().sum::<f32>() / xk_ratios.len() as f32;
        println!("grad_xk: mean ratio GPU/CPU = {:.4}", xk_mean_ratio);
        println!("grad_xk_cpu[0..5]: {:?}", &grad_xk_cpu[0..5]);
        println!("grad_xk_gpu[0..5]: {:?}", &grad_xk_gpu[0..5]);

        // Analyze xv
        let xv_ratios: Vec<f32> = grad_xv_cpu
            .iter()
            .zip(grad_xv_gpu.iter())
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, g)| g / c)
            .collect();
        let xv_mean_ratio: f32 = xv_ratios.iter().sum::<f32>() / xv_ratios.len() as f32;
        println!("grad_xv: mean ratio GPU/CPU = {:.4}", xv_mean_ratio);
        println!("grad_xv_cpu[0..5]: {:?}", &grad_xv_cpu[0..5]);
        println!("grad_xv_gpu[0..5]: {:?}", &grad_xv_gpu[0..5]);

        // Analyze ttt_lr_eta
        let eta_ratios: Vec<f32> = grad_ttt_lr_eta_cpu
            .iter()
            .zip(grad_ttt_lr_eta_gpu.iter())
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, g)| g / c)
            .collect();
        let eta_mean_ratio: f32 = if !eta_ratios.is_empty() {
            eta_ratios.iter().sum::<f32>() / eta_ratios.len() as f32
        } else {
            0.0
        };
        println!(
            "grad_ttt_lr_eta: mean ratio GPU/CPU = {:.4}",
            eta_mean_ratio
        );
        println!(
            "grad_eta_cpu[0..5]: {:?}",
            &grad_ttt_lr_eta_cpu[0..5.min(grad_ttt_lr_eta_cpu.len())]
        );
        println!(
            "grad_eta_gpu[0..5]: {:?}",
            &grad_ttt_lr_eta_gpu[0..5.min(grad_ttt_lr_eta_gpu.len())]
        );

        println!("=== End Analysis ===\n");

        for tol_mult in [1.0, 2.0, 5.0, 10.0] {
            let rtol = 2e-2 * tol_mult;
            let atol = 1e-3 * tol_mult;
            let mut xk_mismatches = 0;
            for i in 0..grad_xk_cpu.len() {
                let tol = atol + rtol * grad_xk_cpu[i].abs();
                if (grad_xk_cpu[i] - grad_xk_gpu[i]).abs() > tol {
                    xk_mismatches += 1;
                }
            }
            println!(
                "At {}x tolerance: xk mismatches={}/{}",
                tol_mult,
                xk_mismatches,
                grad_xk_cpu.len()
            );
        }

        // Compare gradients
        assert_data_close(
            &grad_xq_gpu,
            &grad_xq_cpu,
            2e-2,
            1e-3,
            "grad_xq: tiled fused vs reference",
        );
        assert_data_close(
            &grad_xk_gpu,
            &grad_xk_cpu,
            2e-2,
            1e-3,
            "grad_xk: tiled fused vs reference",
        );
        assert_data_close(
            &grad_xv_gpu,
            &grad_xv_cpu,
            2e-2,
            1e-3,
            "grad_xv: tiled fused vs reference",
        );
        assert_data_close(
            &grad_ttt_lr_eta_gpu,
            &grad_ttt_lr_eta_cpu,
            2e-2,
            1e-3,
            "grad_ttt_lr_eta: tiled fused vs reference",
        );
    }
}
