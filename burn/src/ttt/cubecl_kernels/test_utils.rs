//! Shared test utilities for cubecl_kernels.

use std::sync::Arc;

use burn::tensor::Tensor;
use burn_backend::Backend;

use crate::ttt::{
    TTTConfig, TEST_VOCAB_SIZE,
    layer::{Qkv, TTTInnerModel, TTTInputsInner},
    linear::{TTTLinear, TTTLinearConfig},
};

/// Dimensions for TTT tests.
#[derive(Debug, Clone, Copy)]
pub struct TestDims {
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub mini_batch_size: usize,
}

impl TestDims {
    /// Create test dimensions where seq_len == mini_batch_size (single mini-batch).
    pub fn new(batch_size: usize, num_heads: usize, head_dim: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            num_heads,
            head_dim,
            seq_len,
            mini_batch_size: seq_len,
        }
    }

    /// Create test dimensions with explicit mini_batch_size (for multi-stage tests).
    pub fn multi_stage(batch_size: usize, num_heads: usize, head_dim: usize, mini_batch_size: usize, num_stages: usize) -> Self {
        Self {
            batch_size,
            num_heads,
            head_dim,
            seq_len: mini_batch_size * num_stages,
            mini_batch_size,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Default test dimensions.
pub fn default_test_dims() -> TestDims {
    TestDims::new(2, 4, 16, 8)
}

/// Small test dimensions for gradient tests.
pub fn small_test_dims() -> TestDims {
    TestDims::new(2, 2, 8, 4)
}

/// Create a default TTTConfig for testing.
pub fn default_test_config(dims: TestDims) -> Arc<TTTConfig> {
    Arc::new(TTTConfig {
        num_heads: dims.num_heads,
        hidden_size: dims.hidden_size(),
        token_size: dims.hidden_size(),
        mini_batch_size: dims.mini_batch_size,
        base_lr: 1.0,
        epsilon: 1e-6,
        ..TTTConfig::new(TEST_VOCAB_SIZE)
    })
}

/// Create a TTTLinear model for testing.
pub fn create_test_model<B: Backend>(
    config: &Arc<TTTConfig>,
    device: &B::Device,
) -> TTTLinear<B> {
    let linear_config = Arc::new(TTTLinearConfig::new());
    TTTLinear::new(config, &linear_config, device)
}

/// Generate random input tensors for TTT tests.
///
/// Uses `dims.mini_batch_size` to generate token_eta as `[1/1, 1/2, ..., 1/mini_batch_size]`
/// repeated to fill `seq_len`. This ensures consistent behavior between single-stage and
/// multi-stage kernels.
pub fn generate_test_inputs<B: Backend>(
    dims: TestDims,
    device: &B::Device,
) -> TTTInputsInner<B> {
    let shape = [dims.batch_size, dims.num_heads, dims.seq_len, dims.head_dim];

    let xq = Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let xk = Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let xv = Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 0.1), device);

    // Generate token_eta as repeating pattern of [1/1, 1/2, ..., 1/mini_batch_size]
    let token_eta_base = Tensor::arange(1..(dims.mini_batch_size as i64 + 1), device)
        .float()
        .recip();
    let num_repeats = dims.seq_len / dims.mini_batch_size;
    let token_eta = if num_repeats > 1 {
        token_eta_base.repeat_dim(0, num_repeats)
    } else {
        token_eta_base
    };

    let ttt_lr_eta = Tensor::random(
        [dims.batch_size, dims.num_heads, dims.seq_len],
        burn::tensor::Distribution::Uniform(0.01, 0.05),
        device,
    );

    TTTInputsInner {
        qkv: Qkv { xq, xk, xv },
        token_eta,
        ttt_lr_eta,
        start_idx: 0,
    }
}

/// Run forward pass on a TTTInnerModel, returning output and updated state.
pub fn run_forward<B: Backend, M: TTTInnerModel<B>>(
    model: &M,
    inputs: &TTTInputsInner<B>,
    batch_size: usize,
    seq_len: usize,
) -> (Tensor<B, 4>, M::State) {
    let mut state = model.init_state(batch_size);
    let output = model.forward_mini_batch(&mut state, inputs, 0..seq_len);
    (output, state)
}

/// Assert two f32 slices are close within relative and absolute tolerance.
///
/// Uses the formula: |a - b| <= atol + rtol * |b|
pub fn assert_data_close(a: &[f32], b: &[f32], rtol: f32, atol: f32, name: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{name}: Data sizes don't match: {} vs {}",
        a.len(),
        b.len()
    );

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

/// Model parameters extracted as raw data for cross-backend transfer.
#[derive(Debug, Clone)]
pub struct ModelParamsData {
    pub weight_init: Vec<f32>,
    pub bias_init: Vec<f32>,
    pub ln_weight: Vec<f32>,
    pub ln_bias: Vec<f32>,
    pub epsilon: f64,
}

/// Extract model parameters as raw data from a TTTLinear.
pub fn extract_model_params<B: Backend>(model: &TTTLinear<B>) -> ModelParamsData {
    ModelParamsData {
        weight_init: model.weight_init.val().to_data().to_vec().unwrap(),
        bias_init: model.bias_init.val().to_data().to_vec().unwrap(),
        ln_weight: model.layer_norm.weight.val().to_data().to_vec().unwrap(),
        ln_bias: model.layer_norm.bias.val().to_data().to_vec().unwrap(),
        epsilon: model.layer_norm.epsilon,
    }
}

/// Create a TTTLinear from extracted params on a different backend.
pub fn create_model_from_params<B: Backend>(
    params: &ModelParamsData,
    config: &Arc<TTTConfig>,
    dims: TestDims,
    device: &B::Device,
) -> TTTLinear<B> {
    use burn::module::{Ignored, Param};
    use burn::tensor::TensorData;
    use crate::ttt::util::MultiHeadLayerNorm;

    TTTLinear {
        weight_init: Param::from_tensor(Tensor::from_data(
            TensorData::new(params.weight_init.clone(), [dims.num_heads, dims.head_dim, dims.head_dim]),
            device,
        )),
        bias_init: Param::from_tensor(Tensor::from_data(
            TensorData::new(params.bias_init.clone(), [dims.num_heads, dims.head_dim]),
            device,
        )),
        layer_norm: MultiHeadLayerNorm {
            weight: Param::from_tensor(Tensor::from_data(
                TensorData::new(params.ln_weight.clone(), [dims.num_heads, dims.head_dim]),
                device,
            )),
            bias: Param::from_tensor(Tensor::from_data(
                TensorData::new(params.ln_bias.clone(), [dims.num_heads, dims.head_dim]),
                device,
            )),
            epsilon: params.epsilon,
        },
        config: Ignored(config.clone()),
    }
}

/// Input data for cross-backend transfer.
#[derive(Debug, Clone)]
pub struct InputsData {
    pub xq: Vec<f32>,
    pub xk: Vec<f32>,
    pub xv: Vec<f32>,
    pub token_eta: Vec<f32>,
    pub ttt_lr_eta: Vec<f32>,
}

/// Extract inputs as raw data from a TTTInputsInner.
pub fn inputs_to_data<B: Backend>(inputs: &TTTInputsInner<B>) -> InputsData {
    InputsData {
        xq: inputs.qkv.xq.to_data().to_vec().unwrap(),
        xk: inputs.qkv.xk.to_data().to_vec().unwrap(),
        xv: inputs.qkv.xv.to_data().to_vec().unwrap(),
        token_eta: inputs.token_eta.to_data().to_vec().unwrap(),
        ttt_lr_eta: inputs.ttt_lr_eta.to_data().to_vec().unwrap(),
    }
}

/// Create inputs from raw data on a backend.
pub fn inputs_from_data<B: Backend>(
    data: &InputsData,
    dims: TestDims,
    device: &B::Device,
) -> TTTInputsInner<B> {
    use burn::tensor::TensorData;

    let shape = [dims.batch_size, dims.num_heads, dims.seq_len, dims.head_dim];

    TTTInputsInner {
        qkv: Qkv {
            xq: Tensor::from_data(TensorData::new(data.xq.clone(), shape), device),
            xk: Tensor::from_data(TensorData::new(data.xk.clone(), shape), device),
            xv: Tensor::from_data(TensorData::new(data.xv.clone(), shape), device),
        },
        token_eta: Tensor::from_data(
            TensorData::new(data.token_eta.clone(), [dims.seq_len]),
            device,
        ),
        ttt_lr_eta: Tensor::from_data(
            TensorData::new(data.ttt_lr_eta.clone(), [dims.batch_size, dims.num_heads, dims.seq_len]),
            device,
        ),
        start_idx: 0,
    }
}
