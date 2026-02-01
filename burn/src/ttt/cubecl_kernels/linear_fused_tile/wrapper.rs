//! TTTInnerModel implementation for the tiled TTT-Linear kernel.

use std::{marker::PhantomData, ops::Range, sync::Arc};

use burn::tensor::Tensor;

use super::api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};
use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{Fused, FusedTttBackend, FusedTttConfig},
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};

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
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = self.inner.inner.get_config();
        let threads = inner_config.threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config = FusedTttConfig::new(
            inner_config.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

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
            config,
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

        let [_batch_size, _num_heads, seq_len, head_dim] = inputs.qkv.xv.shape().dims();
        let num_full_batches = seq_len / mini_batch_size;
        let remainder = seq_len % mini_batch_size;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = config.threads
            .unwrap_or_else(|| super::api::default_threads(mini_batch_size, head_dim));
        let config = FusedTttConfig::new(mini_batch_size, head_dim, epsilon, threads);

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

            // token_eta is constant across stages - slice to [mini_batch_size] if needed
            let token_eta = inputs.token_eta.clone().slice([0..mini_batch_size]);
            let (output, weight_updated, bias_updated) = fused_ttt_tile_forward_multi::<B>(
                full_qkv.xq,
                full_qkv.xk,
                full_qkv.xv,
                state.weight.clone(),
                state.bias.clone(),
                token_eta,
                full_ttt_lr_eta,
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );

            state.weight = weight_updated;
            state.bias = bias_updated;

            if remainder == 0 {
                output
            } else {
                let remainder_output =
                    self.forward_mini_batch(state, &inputs, full_seq_len..seq_len);

                Tensor::cat(vec![output, remainder_output], 2)
            }
        } else {
            // Sequence shorter than mini_batch_size, use single-stage kernel
            self.forward_mini_batch(state, &inputs, 0..seq_len)
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        if range.len() != self.inner.inner.inner.config.mini_batch_size {
            panic!("Sequence length must be equal to mini_batch_size");
        }

        let inputs = inputs.slice_seq(range);

        let inner = &self.inner.inner.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = inner.config.threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config = FusedTttConfig::new(
            inner.config.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

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
            config,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Autodiff,
        tensor::TensorData,
    };
    use burn_backend::Backend;

    use super::*;
    use crate::ttt::{
        CpuBackend, GpuAutodiffBackend, GpuBackend,
        cubecl_kernels::test_utils::{
            TestDims, assert_data_close, create_model_from_params, default_test_config,
            extract_model_params, generate_test_inputs, inputs_from_data, inputs_to_data,
            run_forward,
        },
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
    };

    #[test]
    fn test_fused_tile_vs_ttt_linear() {
        // Use 8x32 tiles
        let dims = TestDims::new(2, 2, 32, 8);
        let config = default_test_config(dims);
        let cpu_device: <CpuBackend as Backend>::Device = Default::default();

        // Create CPU reference model and extract params
        let ref_model: TTTLinear<CpuBackend> = TTTLinear::new(
            &config,
            &Arc::new(TTTLinearConfig::new()),
            &cpu_device,
        );
        let params = extract_model_params(&ref_model);

        // Generate inputs and extract data
        let inputs = generate_test_inputs(dims, dims.seq_len, &cpu_device);
        let input_data = inputs_to_data(&inputs);

        // Run reference
        let (output_ref, state_ref) = run_forward(&ref_model, &inputs, dims.batch_size, dims.seq_len);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();
        let weight_ref_data: Vec<f32> = state_ref.weight.to_data().to_vec().unwrap();
        let bias_ref_data: Vec<f32> = state_ref.bias.to_data().to_vec().unwrap();

        // Create GPU fused model with same params
        let gpu_device: <GpuBackend as Backend>::Device = Default::default();
        let fused_model: TTTLinear<GpuBackend> =
            create_model_from_params(&params, &config, dims, &gpu_device);
        let fused_tile: Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>> =
            Fused::from(Fused::from(fused_model));

        // Recreate inputs on GPU
        let inputs_gpu = inputs_from_data(&input_data, dims, &gpu_device);
        let (output_fused, state_fused) =
            run_forward(&fused_tile, &inputs_gpu, dims.batch_size, dims.seq_len);
        let output_fused_data: Vec<f32> = output_fused.to_data().to_vec().unwrap();
        let weight_fused_data: Vec<f32> = state_fused.weight.to_data().to_vec().unwrap();
        let bias_fused_data: Vec<f32> = state_fused.bias.to_data().to_vec().unwrap();

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
        // Test multi-stage kernel: 4 mini-batches of 8 tokens each = 32 total
        let mini_batch_size = 8usize;
        let num_stages = 4usize;
        let seq_len = mini_batch_size * num_stages;
        let dims = TestDims::new(2, 2, 32, seq_len);

        // Config uses mini_batch_size, not full seq_len
        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads: dims.num_heads,
            hidden_size: dims.hidden_size(),
            token_size: dims.hidden_size(),
            mini_batch_size,
            base_lr: 1.0,
            epsilon: 1e-6,
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });

        let cpu_device: <CpuBackend as Backend>::Device = Default::default();

        // Create CPU reference model and extract params
        let ref_model: TTTLinear<CpuBackend> = TTTLinear::new(
            &config,
            &Arc::new(TTTLinearConfig::new()),
            &cpu_device,
        );
        let params = extract_model_params(&ref_model);

        // Generate inputs with repeating token_eta pattern for multi-stage
        let inputs = generate_test_inputs(dims, mini_batch_size, &cpu_device);
        let input_data = inputs_to_data(&inputs);

        // Run CPU reference using forward() which processes all mini-batches
        let mut state_cpu = ref_model.init_state(dims.batch_size);
        let output_ref = ref_model.forward(&mut state_cpu, inputs);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();
        let weight_ref_data: Vec<f32> = state_cpu.weight.to_data().to_vec().unwrap();
        let bias_ref_data: Vec<f32> = state_cpu.bias.to_data().to_vec().unwrap();

        // Create GPU multi-stage fused model with same params
        let gpu_device: <GpuBackend as Backend>::Device = Default::default();
        let fused_model: TTTLinear<GpuBackend> =
            create_model_from_params(&params, &config, dims, &gpu_device);
        let inner_fused: Fused<GpuBackend, TTTLinear<GpuBackend>> = fused_model.into();
        let inner_fused2: Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>> =
            inner_fused.into();
        let fused_tile_multi: Fused<
            GpuBackend,
            Fused<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>>,
        > = inner_fused2.into();

        let mut fused_state = fused_tile_multi.init_state(dims.batch_size);

        // Recreate inputs on GPU
        let inputs_gpu = inputs_from_data(&input_data, dims, &gpu_device);

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
        // Use 8x32 tiles
        let dims = TestDims::new(2, 2, 32, 8);
        let config = default_test_config(dims);

        // Seed RNG for deterministic results
        let cpu_device: <CpuBackend as Backend>::Device = Default::default();
        CpuBackend::seed(&cpu_device, 42);

        // Create CPU reference with autodiff
        let ttt_linear_cpu: TTTLinear<Autodiff<CpuBackend>> =
            TTTLinear::new(&config, &Arc::new(TTTLinearConfig::new()), &cpu_device);

        // Extract params for GPU model
        let params = extract_model_params(&ttt_linear_cpu);
        let ln_weight_data_copy = params.ln_weight.clone();

        // Init state, detach from weight_init graph, then require_grad to get batched gradients
        let state_cpu_init = ttt_linear_cpu.init_state(dims.batch_size);
        let state_weight_cpu: Tensor<Autodiff<CpuBackend>, 4> =
            Tensor::from_inner(state_cpu_init.weight.inner()).require_grad();
        let state_bias_cpu: Tensor<Autodiff<CpuBackend>, 3> =
            Tensor::from_inner(state_cpu_init.bias.inner()).require_grad();
        let mut state_cpu = crate::ttt::linear::TTTLinearState {
            weight: state_weight_cpu.clone(),
            bias: state_bias_cpu.clone(),
        };

        // Create input tensors with gradients (autodiff requires manual setup)
        let shape = [dims.batch_size, dims.num_heads, dims.seq_len, dims.head_dim];
        let xq_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            shape,
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let xk_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            shape,
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let xv_cpu: Tensor<Autodiff<CpuBackend>, 4> = Tensor::random(
            shape,
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        )
        .require_grad();
        let token_eta_cpu: Tensor<Autodiff<CpuBackend>, 1> =
            Tensor::arange(1..(dims.seq_len as i64 + 1), &cpu_device)
                .float()
                .recip();
        let ttt_lr_eta_cpu: Tensor<Autodiff<CpuBackend>, 3> = Tensor::random(
            [dims.batch_size, dims.num_heads, dims.seq_len],
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

        let output_cpu = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, &inputs_cpu, 0..dims.seq_len);
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

        // Get batched state gradients from CPU (not weight_init which would be summed)
        let grad_weight_cpu: Vec<f32> = state_weight_cpu
            .grad(&grads_cpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_bias_cpu: Vec<f32> = state_bias_cpu
            .grad(&grads_cpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_ln_weight_cpu: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .weight
            .grad(&grads_cpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_ln_bias_cpu: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .grad(&grads_cpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();

        // Create GPU model with same params
        let gpu_device: <GpuAutodiffBackend as Backend>::Device = Default::default();
        let ttt_linear: TTTLinear<GpuAutodiffBackend> =
            create_model_from_params(&params, &config, dims, &gpu_device);

        // Create double-Fused wrapper for tiled kernel
        let fused_tile: Fused<
            GpuAutodiffBackend,
            Fused<GpuAutodiffBackend, TTTLinear<GpuAutodiffBackend>>,
        > = Fused::from(Fused::from(ttt_linear));

        // Init state, detach from weight_init graph, then require_grad to get batched gradients
        let fused_state_init = fused_tile.init_state(dims.batch_size);
        let state_weight_gpu: Tensor<GpuAutodiffBackend, 4> =
            Tensor::from_inner(fused_state_init.weight.inner()).require_grad();
        let state_bias_gpu: Tensor<GpuAutodiffBackend, 3> =
            Tensor::from_inner(fused_state_init.bias.inner()).require_grad();
        let mut fused_state = crate::ttt::linear::TTTLinearState {
            weight: state_weight_gpu.clone(),
            bias: state_bias_gpu.clone(),
        };

        // Create GPU input tensors with gradients (autodiff requires manual setup)
        let xq_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xq_data, shape),
            &gpu_device,
        )
        .require_grad();
        let xk_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xk_data, shape),
            &gpu_device,
        )
        .require_grad();
        let xv_gpu: Tensor<GpuAutodiffBackend, 4> = Tensor::from_data(
            TensorData::new(xv_data, shape),
            &gpu_device,
        )
        .require_grad();
        let token_eta_gpu: Tensor<GpuAutodiffBackend, 1> =
            Tensor::from_data(TensorData::new(token_eta_data, [dims.seq_len]), &gpu_device);
        let ttt_lr_eta_gpu: Tensor<GpuAutodiffBackend, 3> = Tensor::from_data(
            TensorData::new(ttt_lr_eta_data, [dims.batch_size, dims.num_heads, dims.seq_len]),
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
        let output_gpu = fused_tile.forward_mini_batch(&mut fused_state, &inputs_gpu, 0..dims.seq_len);
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

        // Get batched state gradients from GPU (not weight_init which would be summed)
        let grad_weight_gpu: Vec<f32> = state_weight_gpu
            .grad(&grads_gpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_bias_gpu: Vec<f32> = state_bias_gpu
            .grad(&grads_gpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_ln_weight_gpu: Vec<f32> = fused_tile
            .inner
            .inner
            .layer_norm
            .weight
            .grad(&grads_gpu)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let grad_ln_bias_gpu: Vec<f32> = fused_tile
            .inner
            .inner
            .layer_norm
            .bias
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

        // Assert parameter gradient shapes match between CPU and GPU
        // Verify batched shapes: [batch_size, num_heads, head_dim, head_dim] and [batch_size, num_heads, head_dim]
        let expected_weight_len = dims.batch_size * dims.num_heads * dims.head_dim * dims.head_dim;
        let expected_bias_len = dims.batch_size * dims.num_heads * dims.head_dim;
        assert_eq!(grad_weight_cpu.len(), expected_weight_len,
            "grad_weight_cpu should be batched: expected {}, got {}", expected_weight_len, grad_weight_cpu.len());
        assert_eq!(grad_weight_gpu.len(), expected_weight_len,
            "grad_weight_gpu should be batched: expected {}, got {}", expected_weight_len, grad_weight_gpu.len());
        assert_eq!(grad_bias_cpu.len(), expected_bias_len,
            "grad_bias_cpu should be batched: expected {}, got {}", expected_bias_len, grad_bias_cpu.len());
        assert_eq!(grad_bias_gpu.len(), expected_bias_len,
            "grad_bias_gpu should be batched: expected {}, got {}", expected_bias_len, grad_bias_gpu.len());
        assert_eq!(grad_ln_weight_cpu.len(), grad_ln_weight_gpu.len(),
            "grad_ln_weight shape mismatch: cpu={}, gpu={}", grad_ln_weight_cpu.len(), grad_ln_weight_gpu.len());
        assert_eq!(grad_ln_bias_cpu.len(), grad_ln_bias_gpu.len(),
            "grad_ln_bias shape mismatch: cpu={}, gpu={}", grad_ln_bias_cpu.len(), grad_ln_bias_gpu.len());

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

        // Validate parameter gradients
        assert_data_close(
            &grad_weight_gpu,
            &grad_weight_cpu,
            2e-2,
            1e-3,
            "grad_weight: tiled fused vs reference",
        );
        assert_data_close(
            &grad_bias_gpu,
            &grad_bias_cpu,
            2e-2,
            1e-3,
            "grad_bias: tiled fused vs reference",
        );
        assert_data_close(
            &grad_ln_weight_gpu,
            &grad_ln_weight_cpu,
            2e-2,
            1e-3,
            "grad_ln_weight: tiled fused vs reference",
        );
        assert_data_close(
            &grad_ln_bias_gpu,
            &grad_ln_bias_cpu,
            2e-2,
            1e-3,
            "grad_ln_bias: tiled fused vs reference",
        );
    }
}
