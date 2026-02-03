//! Shared test utilities for cubecl_kernels.

use std::sync::Arc;

use burn::tensor::Tensor;
use burn_backend::Backend;

use crate::{
    Qkv, TEST_VOCAB_SIZE, TTTConfig, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearConfig,
    TTTLinearState,
};

/// Dimensions for TTT tests.
#[derive(Debug, Clone, Copy)]
pub struct TestDims {
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub mini_batch_size: usize,
    /// Number of iterations to run with fresh data (default: 1).
    pub iterations: usize,
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
            iterations: 1,
        }
    }

    /// Create test dimensions with explicit mini_batch_size (for multi-stage tests).
    #[must_use]
    pub fn multi_stage(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        mini_batch_size: usize,
        num_stages: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            head_dim,
            seq_len: mini_batch_size * num_stages,
            mini_batch_size,
            iterations: 1,
        }
    }

    /// Set the number of iterations to run.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
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
pub fn create_test_model<B: Backend>(config: &Arc<TTTConfig>, device: &B::Device) -> TTTLinear<B> {
    let linear_config = Arc::new(TTTLinearConfig::new());
    TTTLinear::new(config, &linear_config, device)
}

/// Generate random input tensors for TTT tests.
///
/// Uses `dims.mini_batch_size` to generate token_eta as `[1/1, 1/2, ..., 1/mini_batch_size]`
/// repeated to fill `seq_len`. This ensures consistent behavior between single-stage and
/// multi-stage kernels.
pub fn generate_test_inputs<B: Backend>(dims: TestDims, device: &B::Device) -> TTTInputsInner<B> {
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

/// Input data for cloning inputs (since forward consumes them).
#[derive(Debug, Clone)]
pub struct InputsData {
    pub xq: Vec<f32>,
    pub xk: Vec<f32>,
    pub xv: Vec<f32>,
    pub token_eta: Vec<f32>,
    pub ttt_lr_eta: Vec<f32>,
}

/// Extract inputs as raw data from a TTTInputsInner.
/// Converts to f32 regardless of backend element type.
pub fn inputs_to_data<B: Backend>(inputs: &TTTInputsInner<B>) -> InputsData {
    InputsData {
        xq: inputs.qkv.xq.to_data().convert::<f32>().to_vec().unwrap(),
        xk: inputs.qkv.xk.to_data().convert::<f32>().to_vec().unwrap(),
        xv: inputs.qkv.xv.to_data().convert::<f32>().to_vec().unwrap(),
        token_eta: inputs
            .token_eta
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap(),
        ttt_lr_eta: inputs
            .ttt_lr_eta
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap(),
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
            TensorData::new(
                data.ttt_lr_eta.clone(),
                [dims.batch_size, dims.num_heads, dims.seq_len],
            ),
            device,
        ),
        start_idx: 0,
    }
}

use burn::tensor::{TensorData, backend::AutodiffBackend};

impl<B: Backend> AsRef<TTTLinearState<B>> for TTTLinearState<B> {
    fn as_ref(&self) -> &TTTLinearState<B> {
        self
    }
}

/// Test backward using `forward_mini_batch` for both.
pub fn test_backward_fmb<B, T, Transform>(
    dims: TestDims,
    transform: Transform,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: AutodiffBackend<Device: Default>,
    T: TTTInnerModel<B, State = TTTLinearState<B>>,
    Transform: FnOnce(TTTLinear<B>) -> T,
{
    let seq_len = dims.seq_len;
    test_backward(
        dims,
        transform,
        move |m, s, i| m.forward_mini_batch(s, &i, 0..seq_len),
        move |m, s, i| m.forward_mini_batch(s, &i, 0..seq_len),
        rtol,
        atol,
        name,
    )
}

/// Test backward using `forward` for both.
pub fn test_backward_fwd<B, T, Transform>(
    dims: TestDims,
    transform: Transform,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: AutodiffBackend<Device: Default>,
    T: TTTInnerModel<B, State = TTTLinearState<B>>,
    Transform: FnOnce(TTTLinear<B>) -> T,
{
    test_backward(
        dims,
        transform,
        |m, s, i| m.forward(s, i),
        |m, s, i| m.forward(s, i),
        rtol,
        atol,
        name,
    )
}

/// Test backward gradients with custom run functions.
/// Compares all gradients: xq, xk, xv, ttt_lr_eta, state weight, state bias.
/// Runs `dims.iterations` iterations with fresh data each time.
pub fn test_backward<B, T, Transform, RunRef, RunTested>(
    dims: TestDims,
    transform: Transform,
    mut run_ref: RunRef,
    mut run_tested: RunTested,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: AutodiffBackend<Device: Default>,
    T: TTTInnerModel<B, State = TTTLinearState<B>>,
    Transform: FnOnce(TTTLinear<B>) -> T,
    RunRef: FnMut(&TTTLinear<B>, &mut TTTLinearState<B>, TTTInputsInner<B>) -> Tensor<B, 4>,
    RunTested: FnMut(&T, &mut TTTLinearState<B>, TTTInputsInner<B>) -> Tensor<B, 4>,
{
    let config = default_test_config(dims);
    let device: B::Device = Default::default();

    // Create ref model and clone for tested
    let ref_model: TTTLinear<B> = create_test_model(&config, &device);
    let tested_model = transform(ref_model.clone());

    let shape = [dims.batch_size, dims.num_heads, dims.seq_len, dims.head_dim];

    for iter in 0..dims.iterations {
        // Generate fresh input data
        let inputs: TTTInputsInner<B> = generate_test_inputs(dims, &device);
        let input_data = inputs_to_data(&inputs);

        // Create ref inputs with require_grad
        let xq_ref: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xq.clone(), shape), &device)
                .require_grad();
        let xk_ref: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xk.clone(), shape), &device)
                .require_grad();
        let xv_ref: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xv.clone(), shape), &device)
                .require_grad();
        let ttt_lr_eta_ref: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(
                input_data.ttt_lr_eta.clone(),
                [dims.batch_size, dims.num_heads, dims.seq_len],
            ),
            &device,
        )
        .require_grad();
        let token_eta: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(input_data.token_eta.clone(), [dims.seq_len]),
            &device,
        );

        // Init ref state with require_grad (detach from weight_init graph first)
        let state_ref_init = ref_model.init_state(dims.batch_size);
        let weight_ref: Tensor<B, 4> =
            Tensor::from_inner(state_ref_init.weight.inner()).require_grad();
        let bias_ref: Tensor<B, 3> = Tensor::from_inner(state_ref_init.bias.inner()).require_grad();
        let mut state_ref = TTTLinearState {
            weight: weight_ref.clone(),
            bias: bias_ref.clone(),
        };

        let inputs_ref = TTTInputsInner {
            qkv: Qkv {
                xq: xq_ref.clone(),
                xk: xk_ref.clone(),
                xv: xv_ref.clone(),
            },
            token_eta: token_eta.clone(),
            ttt_lr_eta: ttt_lr_eta_ref.clone(),
            start_idx: 0,
        };

        // Run ref forward + backward
        let output_ref = run_ref(&ref_model, &mut state_ref, inputs_ref);
        let grads_ref = output_ref.sum().backward();

        // Create tested inputs with require_grad
        let xq_tested: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xq.clone(), shape), &device)
                .require_grad();
        let xk_tested: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xk.clone(), shape), &device)
                .require_grad();
        let xv_tested: Tensor<B, 4> =
            Tensor::from_data(TensorData::new(input_data.xv.clone(), shape), &device)
                .require_grad();
        let ttt_lr_eta_tested: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(
                input_data.ttt_lr_eta.clone(),
                [dims.batch_size, dims.num_heads, dims.seq_len],
            ),
            &device,
        )
        .require_grad();

        // Init tested state with require_grad
        let state_tested_init = tested_model.init_state(dims.batch_size);
        let weight_tested: Tensor<B, 4> =
            Tensor::from_inner(state_tested_init.weight.inner()).require_grad();
        let bias_tested: Tensor<B, 3> =
            Tensor::from_inner(state_tested_init.bias.inner()).require_grad();
        let mut state_tested = TTTLinearState {
            weight: weight_tested.clone(),
            bias: bias_tested.clone(),
        };

        let inputs_tested = TTTInputsInner {
            qkv: Qkv {
                xq: xq_tested.clone(),
                xk: xk_tested.clone(),
                xv: xv_tested.clone(),
            },
            token_eta,
            ttt_lr_eta: ttt_lr_eta_tested.clone(),
            start_idx: 0,
        };

        // Run tested forward + backward
        let output_tested = run_tested(&tested_model, &mut state_tested, inputs_tested);
        let grads_tested = output_tested.sum().backward();

        // Compare input gradients
        let iter_name = if dims.iterations > 1 {
            format!("{name} iter {}", iter + 1)
        } else {
            name.to_string()
        };
        assert_data_close(
            &xq_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &xq_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_xq"),
        );
        assert_data_close(
            &xk_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &xk_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_xk"),
        );
        assert_data_close(
            &xv_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &xv_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_xv"),
        );
        assert_data_close(
            &ttt_lr_eta_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &ttt_lr_eta_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_ttt_lr_eta"),
        );

        // Compare state gradients
        assert_data_close(
            &weight_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &weight_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_weight"),
        );
        assert_data_close(
            &bias_tested
                .grad(&grads_tested)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &bias_ref
                .grad(&grads_ref)
                .unwrap()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} grad_bias"),
        );
    }
}

/// Test using `forward_mini_batch` for both ref and tested.
/// Each iteration generates fresh inputs and processes the full seq_len range.
pub fn test_fmb<B, T, S, Transform>(
    dims: TestDims,
    transform: Transform,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: Backend<Device: Default>,
    S: AsRef<TTTLinearState<B>>,
    T: TTTInnerModel<B, State = S>,
    Transform: FnOnce(TTTLinear<B>) -> T,
{
    let seq_len = dims.seq_len;
    test_vs_ttt_linear(
        dims,
        transform,
        move |m, s, i| m.forward_mini_batch(s, &i, 0..seq_len),
        move |m, s, i| m.forward_mini_batch(s, &i, 0..seq_len),
        rtol,
        atol,
        name,
    )
}

/// Test using `forward` for both ref and tested.
pub fn test_fwd<B, T, S, Transform>(
    dims: TestDims,
    transform: Transform,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: Backend<Device: Default>,
    S: AsRef<TTTLinearState<B>>,
    T: TTTInnerModel<B, State = S>,
    Transform: FnOnce(TTTLinear<B>) -> T,
{
    test_vs_ttt_linear(
        dims,
        transform,
        |m, s, i| m.forward(s, i),
        |m, s, i| m.forward(s, i),
        rtol,
        atol,
        name,
    )
}

/// Test a TTTInnerModel implementation against TTTLinear reference.
///
/// Both models run on the same backend. The `transform` converts TTTLinear to the tested model.
/// `run_ref` and `run_tested` control how to execute the forward pass for each.
/// Runs `dims.iterations` iterations with fresh data each time.
pub fn test_vs_ttt_linear<B, T, S, Transform, RunRef, RunTested>(
    dims: TestDims,
    transform: Transform,
    mut run_ref: RunRef,
    mut run_tested: RunTested,
    rtol: f32,
    atol: f32,
    name: &str,
) where
    B: Backend<Device: Default>,
    S: AsRef<TTTLinearState<B>>,
    T: TTTInnerModel<B, State = S>,
    Transform: FnOnce(TTTLinear<B>) -> T,
    RunRef: FnMut(&TTTLinear<B>, &mut TTTLinearState<B>, TTTInputsInner<B>) -> Tensor<B, 4>,
    RunTested: FnMut(&T, &mut S, TTTInputsInner<B>) -> Tensor<B, 4>,
{
    let config = default_test_config(dims);
    let device: B::Device = Default::default();

    let ref_model: TTTLinear<B> = create_test_model(&config, &device);
    let tested_model = transform(ref_model.clone());

    let mut state_ref = ref_model.init_state(dims.batch_size);
    let mut state_tested = tested_model.init_state(dims.batch_size);

    for iter in 0..dims.iterations {
        let inputs_ref = generate_test_inputs(dims, &device);
        let input_data = inputs_to_data(&inputs_ref);
        let inputs_tested = inputs_from_data(&input_data, dims, &device);

        // Run reference
        let output_ref = run_ref(&ref_model, &mut state_ref, inputs_ref);

        // Run tested
        let output_tested = run_tested(&tested_model, &mut state_tested, inputs_tested);

        // Compare
        let iter_name = if dims.iterations > 1 {
            format!("{name} iter {}", iter + 1)
        } else {
            name.to_string()
        };
        assert_data_close(
            &output_tested.to_data().convert::<f32>().to_vec().unwrap(),
            &output_ref.to_data().convert::<f32>().to_vec().unwrap(),
            rtol,
            atol,
            &format!("{iter_name} output"),
        );
        let tested_state = state_tested.as_ref();
        assert_data_close(
            &tested_state
                .weight
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &state_ref
                .weight
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            rtol,
            atol,
            &format!("{iter_name} weight"),
        );
        assert_data_close(
            &tested_state
                .bias
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap(),
            &state_ref.bias.to_data().convert::<f32>().to_vec().unwrap(),
            rtol,
            atol,
            &format!("{iter_name} bias"),
        );
    }
}
