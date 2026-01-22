//! TTTInnerModel implementation for the tiled TTT-Linear kernel.

use std::{marker::PhantomData, sync::Arc};

use burn::tensor::Tensor;

use crate::ttt::{
    TTTConfig,
    cubecl_kernels::Fused,
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};
use crate::ttt::cubecl_kernels::kernel::FusedKernelBackend;

use super::api::fused_ttt_tile_forward;
use super::launch::TttTileKernel;

/// Trait alias for backends that support the tiled TTT kernel.
pub trait FusedTttTileBackend: FusedKernelBackend<TttTileKernel, 9, 3> {}

impl<B> FusedTttTileBackend for B where B: FusedKernelBackend<TttTileKernel, 9, 3> {}

/// TTTInnerModel implementation for the tiled fused kernel.
/// Uses double Fused wrapper: `Fused<B, Fused<B, TTTLinear<B>>>`.
impl<B: FusedTttTileBackend> TTTInnerModel<B> for Fused<B, Fused<B, TTTLinear<B>>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttt::{
        CpuBackend, GpuBackend,
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
        util::MultiHeadLayerNorm,
    };
    use burn::{
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
        // Use 16x64 tiles (CS=16, F=64)
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 64usize;
        let seq_len = 16usize;
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
}
