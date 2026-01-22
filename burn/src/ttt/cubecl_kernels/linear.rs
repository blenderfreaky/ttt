use std::{marker::PhantomData, sync::Arc};

use burn::tensor::Tensor;

use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{
        Fused,
        backend::{FusedTttBackend, api::fused_ttt_forward},
    },
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};

impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused {
            inner: TTTLinear::new(general_config, config, device),
            _backend: PhantomData,
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let inner = &self.inner;

        let qkv = inputs.qkv;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let (output, weight_updated, bias_updated) = fused_ttt_forward(
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
        module::{Ignored, Param},
        tensor::TensorData,
    };
    use burn_backend::Backend;
    use std::sync::Arc;

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
    fn test_fused_api_vs_reference() {
        let batch_size = 2usize;
        let num_heads = 4usize;
        let head_dim = 16usize;
        let seq_len = 8usize;
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f32;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon: f64::from(epsilon),
            ..crate::ttt::TTTConfig::new()
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // random input tensors
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

        let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

        let ttt_linear_cpu: TTTLinear<CpuBackend> =
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
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu,
                xk: xk_cpu,
                xv: xv_cpu,
            },
            token_eta: token_eta_cpu,
            ttt_lr_eta: ttt_lr_eta_cpu,
            start_idx: 0,
        };

        let output_ref = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, inputs_cpu);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();

        let gpu_device: <GpuBackend as Backend>::Device = Default::default();

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

        let weight_expanded: Vec<f32> = (0..batch_size)
            .flat_map(|_| weight_init_data.iter().copied())
            .collect();
        let weight_gpu: Tensor<GpuBackend, 4> = Tensor::from_data(
            TensorData::new(weight_expanded, [batch_size, num_heads, head_dim, head_dim]),
            &gpu_device,
        );

        let bias_expanded: Vec<f32> = (0..batch_size)
            .flat_map(|_| bias_init_data.iter().copied())
            .collect();
        let bias_gpu: Tensor<GpuBackend, 3> = Tensor::from_data(
            TensorData::new(bias_expanded, [batch_size, num_heads, head_dim]),
            &gpu_device,
        );

        let ln_weight_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
            TensorData::new(ln_weight_data, [num_heads, head_dim]),
            &gpu_device,
        );
        let ln_bias_gpu: Tensor<GpuBackend, 2> = Tensor::from_data(
            TensorData::new(ln_bias_data, [num_heads, head_dim]),
            &gpu_device,
        );

        let (output_gpu, _, _) = fused_ttt_forward(
            xq_gpu,
            xk_gpu,
            xv_gpu,
            weight_gpu,
            bias_gpu,
            token_eta_gpu,
            ttt_lr_eta_gpu,
            ln_weight_gpu,
            ln_bias_gpu,
            epsilon,
        );

        let output_gpu_data: Vec<f32> = output_gpu.to_data().to_vec().unwrap();

        assert_data_close(
            &output_gpu_data,
            &output_ref_data,
            1e-3, // rtol
            1e-4, // atol
            "fused_ttt_forward API vs TTTLinear output",
        );
    }

    #[test]
    fn test_fused_ttt_linear_vs_reference() {
        let batch_size = 2usize;
        let num_heads = 4usize;
        let head_dim = 16usize;
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
            ..crate::ttt::TTTConfig::new()
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // random input tensors
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

        let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

        let ttt_linear_cpu: TTTLinear<CpuBackend> =
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
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu,
                xk: xk_cpu,
                xv: xv_cpu,
            },
            token_eta: token_eta_cpu,
            ttt_lr_eta: ttt_lr_eta_cpu,
            start_idx: 0,
        };

        let output_ref = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, inputs_cpu);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();

        let gpu_device: <GpuBackend as Backend>::Device = Default::default();

        let fused_linear: Fused<GpuBackend, TTTLinear<GpuBackend>> = TTTLinear {
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
        }
        .into();

        let mut fused_state = fused_linear.init_state(batch_size);

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

        let output_fused = fused_linear.forward_mini_batch(&mut fused_state, inputs_gpu);
        let output_fused_data: Vec<f32> = output_fused.to_data().to_vec().unwrap();

        assert_data_close(
            &output_fused_data,
            &output_ref_data,
            1e-3, // rtol
            1e-4, // atol
            "FusedTTTLinear vs TTTLinear output",
        );
    }

    #[test]
    fn test_fused_backward_gradients_vs_reference() {
        use burn::backend::Autodiff;

        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 8usize;
        let seq_len = 4usize;
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
            ..crate::ttt::TTTConfig::new()
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

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
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

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

        let xq_data: Vec<f32> = xq_cpu.clone().inner().to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.clone().inner().to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.clone().inner().to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.clone().inner().to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.clone().inner().to_data().to_vec().unwrap();

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

        let gpu_device: <GpuAutodiffBackend as Backend>::Device = Default::default();

        let fused_linear: Fused<_, TTTLinear<_>> = TTTLinear {
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
        }
        .into();

        let mut fused_state = fused_linear.init_state(batch_size);

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

        let output_gpu = fused_linear.forward_mini_batch(&mut fused_state, inputs_gpu);

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
        println!("grad_xk_cpu[0..5]: {:?}", &grad_xk_cpu[0..5]);
        println!("grad_xk_gpu[0..5]: {:?}", &grad_xk_gpu[0..5]);

        assert_data_close(
            &grad_xq_gpu,
            &grad_xq_cpu,
            2e-2, // rtol
            1e-3, // atol
            "grad_xq: fused vs reference",
        );
        assert_data_close(
            &grad_xk_gpu,
            &grad_xk_cpu,
            2e-2,
            1e-3,
            "grad_xk: fused vs reference",
        );
        assert_data_close(
            &grad_xv_gpu,
            &grad_xv_cpu,
            2e-2,
            1e-3,
            "grad_xv: fused vs reference",
        );
        assert_data_close(
            &grad_ttt_lr_eta_gpu,
            &grad_ttt_lr_eta_cpu,
            2e-2,
            1e-3,
            "grad_ttt_lr_eta: fused vs reference",
        );
    }
}
