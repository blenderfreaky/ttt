//! Launch functions for the tiled TTT-Linear forward kernel.

use std::fmt::Debug;

use burn_backend::Element;
use burn_cubecl::{
    CubeRuntime, FloatElement,
    kernel::{cast, into_contiguous},
    ops::numeric::zeros_client,
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use thundercube::prelude::{D4, D8, D16, D32, D64, LINE_SIZE};
use ttt_kernels::{TensorBundle, kernel::FusedKernel, tensor_bundle, util::empty_like};

use super::{
    backward::{
        GradOutputsLaunch, RecomputationInputsLaunch, SavedTensorsLaunch,
        fused_ttt_backward_kernel, fused_ttt_backward_kernel_multi,
    },
    forward::{InputsLaunch, OutputsLaunch, fused_ttt_forward_kernel, fused_ttt_forward_kernel_multi},
    helpers::Params,
};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttOutputs},
};

tensor_bundle! {
    /// Saved state for backward pass.
    /// Contains all inputs needed for activation recomputation.
    /// No forward intermediates are saved - they are recomputed during backward.
    pub struct TttSavedState {
        // Saved inputs for recomputation
        xq, xk, xv,           // All three projections (xv for target = xv - xk)
        weight_init, bias,    // Initial weight and bias for recomputation
        token_eta, ttt_lr_eta,
        ln_weight, ln_bias    // Layer norm params for recomputation
    }
}

// TODO: 64 subtiles (full cubes?)

/// Marker type for the tiled TTT kernel (single mini-batch).
#[derive(Debug, Clone, Copy)]
pub struct TttTileKernel;

/// Marker type for the multi-stage tiled TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttTileMultiKernel;

/// Launch the tiled TTT forward kernel with automatic tile size dispatch.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8×32, 8×64, 16×32, 16×64, 16×128, 32×32, 32×64
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_forward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = xq.shape[2];
    let head_dim = xq.shape[3];

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let inputs_launch = InputsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        xv.as_tensor_arg(vectorization),
        weight.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let outputs_launch = OutputsLaunch::<F, R>::new(
        output.as_tensor_arg(vectorization),
        weight_out.as_tensor_arg(vectorization),
        bias_out.as_tensor_arg(vectorization),
    );

    tile_dispatch!(
        fused_ttt_forward_kernel,
        client,
        cube_count,
        seq_len,
        head_dim,
        config.threads,
        inputs_launch,
        outputs_launch,
        config
    );
}

/// Launch the tiled TTT backward kernel.
///
/// Supports same tile configurations as forward.
/// Recomputes forward intermediates from saved inputs instead of loading them.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward (for backward computation)
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    // Additional inputs for recomputation
    xv: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Upstream gradient
    grad_output: TensorHandleRef<R>,
    // Output gradients
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = xq.shape[2];
    let head_dim = xq.shape[3];

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let saved_launch = SavedTensorsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
    );

    let recompute_launch = RecomputationInputsLaunch::<F, R>::new(
        xv.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let grad_output_arg = grad_output.as_tensor_arg(vectorization);

    let grads_launch = GradOutputsLaunch::<F, R>::new(
        grad_xq.as_tensor_arg(vectorization),
        grad_xk.as_tensor_arg(vectorization),
        grad_xv.as_tensor_arg(vectorization),
        grad_weight.as_tensor_arg(vectorization),
        grad_bias.as_tensor_arg(vectorization),
        grad_ttt_lr_eta.as_tensor_arg(vectorization),
        // Atomic tensors use scalar (vectorization=1), not Line<F>
        grad_ln_weight.as_tensor_arg(1),
        grad_ln_bias.as_tensor_arg(1),
    );

    tile_dispatch!(
        fused_ttt_backward_kernel,
        client,
        cube_count,
        seq_len,
        head_dim,
        config.threads,
        saved_launch,
        recompute_launch,
        grad_output_arg,
        grads_launch,
        config
    );
}

/// Forward pass using the tiled kernel.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8×32, 8×64, 16×32, 16×64, 16×128, 32×32, 32×64
///
/// Returns the outputs. Forward intermediates are recomputed during backward.
pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    launch_tile_forward::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        output.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        config,
    );

    TttOutputs {
        output,
        weight: weight_out,
        bias: bias_out,
    }
}

/// Saved tensors needed for backward pass (for recomputation).
#[derive(Debug, Clone)]
pub struct TttSavedTensors<T> {
    pub xq: T,
    pub xk: T,
    pub xv: T,
    pub weight_init: T,
    pub bias: T,
    pub token_eta: T,
    pub ttt_lr_eta: T,
    pub ln_weight: T,
    pub ln_bias: T,
}

/// Gradient inputs for backward pass.
#[derive(Debug, Clone)]
pub struct TttGradInputs<T> {
    pub grad_xq: T,
    pub grad_xk: T,
    pub grad_xv: T,
    pub grad_weight: T,
    pub grad_bias: T,
    pub grad_ttt_lr_eta: T,
    pub grad_ln_weight: T,
    pub grad_ln_bias: T,
}

/// Backward pass using the tiled kernel.
///
/// Recomputes forward intermediates from saved inputs.
/// Takes saved tensors and upstream gradients, returns gradients w.r.t. all inputs.
pub fn backward<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    grad_output: CubeTensor<R>,
    epsilon: f32,
    threads: usize,
) -> TttGradInputs<CubeTensor<R>> {
    let shape = saved.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();

    // Allocate output gradient tensors (xk, xv have same shape as xq)
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched =
        empty_like::<R, F>(&saved.weight_init, [batch_size, num_heads, head_dim]);

    // LN gradients are unbatched (atomic accumulation across batches)
    // Must be zero-initialized since kernel uses atomic adds
    // NOTE: These use f32 because HIP/ROCm doesn't support bf16 atomics
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, threads);

    launch_tile_backward::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.bias.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight_batched.as_handle_ref(),
        grad_bias_batched.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        config,
    );

    // These are in f32 (see above), so we need to cast them back
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttGradInputs {
        grad_xq,
        grad_xk,
        grad_xv,
        grad_weight: grad_weight_batched,
        grad_bias: grad_bias_batched,
        grad_ttt_lr_eta,
        grad_ln_weight,
        grad_ln_bias,
    }
}

/// Launch the multi-stage tiled TTT backward kernel.
///
/// Processes `num_stages` mini-batches in reverse order (backward through time).
/// Recomputes forward intermediates from saved inputs.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    // Additional inputs for recomputation
    xv: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Upstream gradient
    grad_output: TensorHandleRef<R>,
    // Output gradients
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttConfig,
    num_stages: usize,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let mini_batch_len = config.mini_batch_len;
    let head_dim = config.head_dim;

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let saved_launch = SavedTensorsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
    );

    let recompute_launch = RecomputationInputsLaunch::<F, R>::new(
        xv.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let grad_output_arg = grad_output.as_tensor_arg(vectorization);

    let grads_launch = GradOutputsLaunch::<F, R>::new(
        grad_xq.as_tensor_arg(vectorization),
        grad_xk.as_tensor_arg(vectorization),
        grad_xv.as_tensor_arg(vectorization),
        grad_weight.as_tensor_arg(vectorization),
        grad_bias.as_tensor_arg(vectorization),
        grad_ttt_lr_eta.as_tensor_arg(vectorization),
        // Atomic tensors use scalar (vectorization=1), not Line<F>
        grad_ln_weight.as_tensor_arg(1),
        grad_ln_bias.as_tensor_arg(1),
    );

    tile_dispatch!(
        fused_ttt_backward_kernel_multi,
        client,
        cube_count,
        mini_batch_len,
        head_dim,
        config.threads,
        saved_launch,
        recompute_launch,
        grad_output_arg,
        grads_launch,
        ScalarArg::new(num_stages as u32),
        config
    );
}

/// Backward pass using the multi-stage tiled kernel.
///
/// Processes the full sequence in reverse order by dividing it into
/// mini-batches and processing them backward through time.
/// Recomputes forward intermediates from saved inputs.
pub fn backward_multi<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    grad_output: CubeTensor<R>,
    mini_batch_len: usize,
    epsilon: f32,
    threads: usize,
) -> TttGradInputs<CubeTensor<R>> {
    let shape = saved.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();

    assert_eq!(
        seq_len % mini_batch_len,
        0,
        "seq_len ({}) must be divisible by mini_batch_len ({})",
        seq_len,
        mini_batch_len
    );
    let num_stages = seq_len / mini_batch_len;

    // Allocate output gradient tensors (xk, xv have same shape as xq)
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched =
        empty_like::<R, F>(&saved.weight_init, [batch_size, num_heads, head_dim]);

    // LN gradients are unbatched (atomic accumulation across batches)
    // Must be zero-initialized since kernel uses atomic adds
    // NOTE: These use f32 because HIP/ROCm doesn't support bf16 atomics
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    let config = FusedTttConfig::new(mini_batch_len, head_dim, epsilon, threads);

    launch_tile_backward_multi::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.bias.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight_batched.as_handle_ref(),
        grad_bias_batched.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        config,
        num_stages,
    );

    // These are in f32 (see above), so we need to cast them back
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttGradInputs {
        grad_xq,
        grad_xk,
        grad_xv,
        grad_weight: grad_weight_batched,
        grad_bias: grad_bias_batched,
        grad_ttt_lr_eta,
        grad_ln_weight,
        grad_ln_bias,
    }
}

/// Launch the multi-stage tiled TTT forward kernel.
///
/// Processes `num_stages` mini-batches in a single kernel launch.
/// Input seq_len should be `mini_batch_len * num_stages`.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_forward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
    config: FusedTttConfig,
    num_stages: usize,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let mini_batch_len = config.mini_batch_len;
    let head_dim = config.head_dim;

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let inputs_launch = InputsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        xv.as_tensor_arg(vectorization),
        weight.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let outputs_launch = OutputsLaunch::<F, R>::new(
        output.as_tensor_arg(vectorization),
        weight_out.as_tensor_arg(vectorization),
        bias_out.as_tensor_arg(vectorization),
    );

    tile_dispatch!(
        fused_ttt_forward_kernel_multi,
        client,
        cube_count,
        mini_batch_len,
        head_dim,
        config.threads,
        inputs_launch,
        outputs_launch,
        ScalarArg::new(num_stages as u32),
        config
    );
}

/// Forward pass using the multi-stage tiled kernel.
///
/// Processes the full sequence in a single kernel launch by dividing it into
/// mini-batches and processing them in a loop inside the kernel.
///
/// # Arguments
/// * `inputs` - Input tensors with seq_len = mini_batch_len * num_stages
/// * `mini_batch_len` - Size of each mini-batch (must be a supported tile size)
///
/// Returns the outputs. Forward intermediates are recomputed during backward.
pub fn forward_multi<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [_batch_size, _num_heads, seq_len, _head_dim] = shape.dims();
    let mini_batch_len = config.mini_batch_len;

    assert_eq!(
        seq_len % mini_batch_len,
        0,
        "seq_len ({}) must be divisible by mini_batch_len ({})",
        seq_len,
        mini_batch_len
    );
    let num_stages = seq_len / mini_batch_len;

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    launch_tile_forward_multi::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        output.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        config,
        num_stages,
    );

    TttOutputs {
        output,
        weight: weight_out,
        bias: bias_out,
    }
}

impl FusedKernel for TttTileKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttSavedState<T>;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttSavedState<CubeTensor<R>>) {
        // Clone inputs needed for saved state before moving into forward
        let saved = TttSavedState {
            xq: inputs.xq.clone(),
            xk: inputs.xk.clone(),
            xv: inputs.xv.clone(),
            weight_init: inputs.weight.clone(),
            bias: inputs.bias.clone(),
            token_eta: inputs.token_eta.clone(),
            ttt_lr_eta: inputs.ttt_lr_eta.clone(),
            ln_weight: inputs.ln_weight.clone(),
            ln_bias: inputs.ln_bias.clone(),
        };

        let outputs = forward::<R, F>(inputs, config);

        (outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: TttSavedState<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        let token_eta_shape = saved.token_eta.shape.clone();
        let epsilon = config.epsilon();
        let threads = config.threads;

        let saved_tensors = TttSavedTensors {
            xq: saved.xq,
            xk: saved.xk,
            xv: saved.xv,
            weight_init: saved.weight_init,
            bias: saved.bias,
            token_eta: saved.token_eta,
            ttt_lr_eta: saved.ttt_lr_eta,
            ln_weight: saved.ln_weight,
            ln_bias: saved.ln_bias,
        };

        let grad_inputs =
            backward::<R, F>(saved_tensors, grad_outputs.output, epsilon, threads);

        // TODO: token_eta gradient is currently zeros (not computed by backward kernel)
        let grad_token_eta = empty_like::<R, F>(&grad_inputs.grad_xq, token_eta_shape);

        TttInputs {
            xq: grad_inputs.grad_xq,
            xk: grad_inputs.grad_xk,
            xv: grad_inputs.grad_xv,
            weight: grad_inputs.grad_weight,
            bias: grad_inputs.grad_bias,
            token_eta: grad_token_eta,
            ttt_lr_eta: grad_inputs.grad_ttt_lr_eta,
            ln_weight: grad_inputs.grad_ln_weight,
            ln_bias: grad_inputs.grad_ln_bias,
        }
    }
}

impl FusedKernel for TttTileMultiKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttSavedState<T>;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttSavedState<CubeTensor<R>>) {
        let mini_batch_len = config.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel"
        );

        // Clone inputs needed for saved state before moving into forward
        let saved = TttSavedState {
            xq: inputs.xq.clone(),
            xk: inputs.xk.clone(),
            xv: inputs.xv.clone(),
            weight_init: inputs.weight.clone(),
            bias: inputs.bias.clone(),
            token_eta: inputs.token_eta.clone(),
            ttt_lr_eta: inputs.ttt_lr_eta.clone(),
            ln_weight: inputs.ln_weight.clone(),
            ln_bias: inputs.ln_bias.clone(),
        };

        let outputs = forward_multi::<R, F>(inputs, config);

        (outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: TttSavedState<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        let mini_batch_len = config.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel backward"
        );

        let token_eta_shape = saved.token_eta.shape.clone();
        let epsilon = config.epsilon();
        let threads = config.threads;

        let saved_tensors = TttSavedTensors {
            xq: saved.xq,
            xk: saved.xk,
            xv: saved.xv,
            weight_init: saved.weight_init,
            bias: saved.bias,
            token_eta: saved.token_eta,
            ttt_lr_eta: saved.ttt_lr_eta,
            ln_weight: saved.ln_weight,
            ln_bias: saved.ln_bias,
        };

        let grad_inputs = backward_multi::<R, F>(
            saved_tensors,
            grad_outputs.output,
            mini_batch_len,
            epsilon,
            threads,
        );

        // TODO: token_eta gradient is currently zeros (not computed by backward kernel)
        let grad_token_eta = empty_like::<R, F>(&grad_inputs.grad_xq, token_eta_shape);

        TttInputs {
            xq: grad_inputs.grad_xq,
            xk: grad_inputs.grad_xk,
            xv: grad_inputs.grad_xv,
            weight: grad_inputs.grad_weight,
            bias: grad_inputs.grad_bias,
            token_eta: grad_token_eta,
            ttt_lr_eta: grad_inputs.grad_ttt_lr_eta,
            ln_weight: grad_inputs.grad_ln_weight,
            ln_bias: grad_inputs.grad_ln_bias,
        }
    }
}
