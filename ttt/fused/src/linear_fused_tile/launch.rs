//! Launch functions for the tiled TTT-Linear forward kernel.

use std::fmt::Debug;

use burn_cubecl::{
    CubeRuntime, FloatElement, kernel::into_contiguous, ops::numeric::zeros_client,
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use thundercube::prelude::{D4, D8, D16, D32, D64, LINE_SIZE};

use super::{
    backward::{
        GradOutputsLaunch, SavedTensorsLaunch, fused_ttt_backward_kernel,
        fused_ttt_backward_kernel_multi,
    },
    forward::{
        ForwardIntermediatesLaunch, InputsLaunch, OutputsLaunch, fused_ttt_forward_kernel,
        fused_ttt_forward_kernel_multi,
    },
};
use crate::{FusedTttConfig, ttt::{TttInputs, TttOutputs}};
use super::helpers::Params;
use ttt_kernels::{
    tensor_bundle, TensorBundle,
    kernel::{CanBackwardWithOut, FusedKernel, UseWithOut},
    util::empty_like,
};

tensor_bundle! {
    /// Extended outputs including forward intermediates for backward.
    /// Contains 3 outputs + 7 forward intermediates = 10 tensors.
    pub struct TttTileOutputs[10] {
        output, weight_out, bias_out,
        x_hat_fused, std_fused, grad_output_fused, grad_x_hat_fused, grad_l_wrt_Z1, x_hat_ln, std_ln
    }
}

/// Forward intermediates needed for backward pass.
#[derive(Debug, Clone)]
pub struct FwdIntermediates<T> {
    pub x_hat_fused: T,
    pub std_fused: T,
    pub grad_output_fused: T,
    pub grad_x_hat_fused: T,
    pub grad_l_wrt_Z1: T,
    pub x_hat_ln: T,
    pub std_ln: T,
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
    // Forward intermediates
    x_hat_fused: TensorHandleRef<R>,
    std_fused: TensorHandleRef<R>,
    grad_output_fused: TensorHandleRef<R>,
    grad_x_hat_fused: TensorHandleRef<R>,
    grad_l_wrt_Z1: TensorHandleRef<R>,
    x_hat_ln: TensorHandleRef<R>,
    std_ln: TensorHandleRef<R>,
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

    let fwd_intermediates_launch = ForwardIntermediatesLaunch::<F, R>::new(
        x_hat_fused.as_tensor_arg(vectorization),
        std_fused.as_tensor_arg(vectorization),
        grad_output_fused.as_tensor_arg(vectorization),
        grad_x_hat_fused.as_tensor_arg(vectorization),
        grad_l_wrt_Z1.as_tensor_arg(vectorization),
        x_hat_ln.as_tensor_arg(vectorization),
        std_ln.as_tensor_arg(vectorization),
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
        fwd_intermediates_launch,
        config
    );
}

/// Launch the tiled TTT backward kernel.
///
/// Supports same tile configurations as forward.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    bias_init: TensorHandleRef<R>,
    weight_last: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Forward intermediates
    x_hat_fused: TensorHandleRef<R>,
    std_fused: TensorHandleRef<R>,
    grad_output_fused: TensorHandleRef<R>,
    grad_x_hat_fused: TensorHandleRef<R>,
    grad_l_wrt_Z1: TensorHandleRef<R>,
    x_hat_ln: TensorHandleRef<R>,
    std_ln: TensorHandleRef<R>,
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
        xv.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        bias_init.as_tensor_arg(vectorization),
        weight_last.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let fwd_launch = ForwardIntermediatesLaunch::<F, R>::new(
        x_hat_fused.as_tensor_arg(vectorization),
        std_fused.as_tensor_arg(vectorization),
        grad_output_fused.as_tensor_arg(vectorization),
        grad_x_hat_fused.as_tensor_arg(vectorization),
        grad_l_wrt_Z1.as_tensor_arg(vectorization),
        x_hat_ln.as_tensor_arg(vectorization),
        std_ln.as_tensor_arg(vectorization),
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
        fwd_launch,
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
/// Returns both the outputs and forward intermediates needed for backward.
pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> (TttOutputs<CubeTensor<R>>, FwdIntermediates<CubeTensor<R>>) {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [batch_size, num_heads, seq_len, _head_dim] = shape.dims();

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    // Allocate forward intermediate tensors
    let x_hat_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let std_fused = empty_like::<R, F>(&inputs.ttt_lr_eta, [batch_size, num_heads, seq_len]);
    let grad_output_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let grad_x_hat_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let grad_l_wrt_Z1 = empty_like::<R, F>(&inputs.xq, shape.clone());
    let x_hat_ln = empty_like::<R, F>(&inputs.xq, shape.clone());
    let std_ln = empty_like::<R, F>(&inputs.ttt_lr_eta, [batch_size, num_heads, seq_len]);

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
        x_hat_fused.as_handle_ref(),
        std_fused.as_handle_ref(),
        grad_output_fused.as_handle_ref(),
        grad_x_hat_fused.as_handle_ref(),
        grad_l_wrt_Z1.as_handle_ref(),
        x_hat_ln.as_handle_ref(),
        std_ln.as_handle_ref(),
        config,
    );

    (
        TttOutputs {
            output,
            weight: weight_out,
            bias: bias_out,
        },
        FwdIntermediates {
            x_hat_fused,
            std_fused,
            grad_output_fused,
            grad_x_hat_fused,
            grad_l_wrt_Z1,
            x_hat_ln,
            std_ln,
        },
    )
}

/// Saved tensors needed for backward pass.
#[derive(Debug, Clone)]
pub struct TttSavedTensors<T> {
    pub xq: T,
    pub xk: T,
    pub xv: T,
    pub weight_init: T,
    pub bias_init: T,
    pub weight_last: T,
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
/// Takes saved tensors from forward, forward intermediates, and upstream gradients,
/// returns gradients w.r.t. all inputs.
pub fn backward<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    fwd: FwdIntermediates<CubeTensor<R>>,
    grad_output: CubeTensor<R>,
    epsilon: f32,
    threads: usize,
) -> TttGradInputs<CubeTensor<R>> {
    let shape = saved.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();

    // Allocate output gradient tensors
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xk, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xv, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched = empty_like::<R, F>(&saved.bias_init, [batch_size, num_heads, head_dim]);

    // LN gradients are unbatched (atomic accumulation across batches)
    // Must be zero-initialized since kernel uses atomic adds
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        F::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_bias.client.clone(),
        saved.ln_bias.device.clone(),
        [num_heads, head_dim].into(),
        F::dtype(),
    );

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, threads);

    launch_tile_backward::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.bias_init.as_handle_ref(),
        saved.weight_last.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        fwd.x_hat_fused.as_handle_ref(),
        fwd.std_fused.as_handle_ref(),
        fwd.grad_output_fused.as_handle_ref(),
        fwd.grad_x_hat_fused.as_handle_ref(),
        fwd.grad_l_wrt_Z1.as_handle_ref(),
        fwd.x_hat_ln.as_handle_ref(),
        fwd.std_ln.as_handle_ref(),
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
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    bias_init: TensorHandleRef<R>,
    weight_last: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Forward intermediates
    x_hat_fused: TensorHandleRef<R>,
    std_fused: TensorHandleRef<R>,
    grad_output_fused: TensorHandleRef<R>,
    grad_x_hat_fused: TensorHandleRef<R>,
    grad_l_wrt_Z1: TensorHandleRef<R>,
    x_hat_ln: TensorHandleRef<R>,
    std_ln: TensorHandleRef<R>,
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
        xv.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        bias_init.as_tensor_arg(vectorization),
        weight_last.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let fwd_launch = ForwardIntermediatesLaunch::<F, R>::new(
        x_hat_fused.as_tensor_arg(vectorization),
        std_fused.as_tensor_arg(vectorization),
        grad_output_fused.as_tensor_arg(vectorization),
        grad_x_hat_fused.as_tensor_arg(vectorization),
        grad_l_wrt_Z1.as_tensor_arg(vectorization),
        x_hat_ln.as_tensor_arg(vectorization),
        std_ln.as_tensor_arg(vectorization),
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
        fwd_launch,
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
pub fn backward_multi<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    fwd: FwdIntermediates<CubeTensor<R>>,
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

    // Allocate output gradient tensors
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xk, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xv, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched = empty_like::<R, F>(&saved.bias_init, [batch_size, num_heads, head_dim]);

    // LN gradients are unbatched (atomic accumulation across batches)
    // Must be zero-initialized since kernel uses atomic adds
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        F::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_bias.client.clone(),
        saved.ln_bias.device.clone(),
        [num_heads, head_dim].into(),
        F::dtype(),
    );

    let config = FusedTttConfig::new(mini_batch_len, head_dim, epsilon, threads);

    launch_tile_backward_multi::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.bias_init.as_handle_ref(),
        saved.weight_last.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        fwd.x_hat_fused.as_handle_ref(),
        fwd.std_fused.as_handle_ref(),
        fwd.grad_output_fused.as_handle_ref(),
        fwd.grad_x_hat_fused.as_handle_ref(),
        fwd.grad_l_wrt_Z1.as_handle_ref(),
        fwd.x_hat_ln.as_handle_ref(),
        fwd.std_ln.as_handle_ref(),
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
    // Forward intermediates
    x_hat_fused: TensorHandleRef<R>,
    std_fused: TensorHandleRef<R>,
    grad_output_fused: TensorHandleRef<R>,
    grad_x_hat_fused: TensorHandleRef<R>,
    grad_l_wrt_Z1: TensorHandleRef<R>,
    x_hat_ln: TensorHandleRef<R>,
    std_ln: TensorHandleRef<R>,
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

    let fwd_intermediates_launch = ForwardIntermediatesLaunch::<F, R>::new(
        x_hat_fused.as_tensor_arg(vectorization),
        std_fused.as_tensor_arg(vectorization),
        grad_output_fused.as_tensor_arg(vectorization),
        grad_x_hat_fused.as_tensor_arg(vectorization),
        grad_l_wrt_Z1.as_tensor_arg(vectorization),
        x_hat_ln.as_tensor_arg(vectorization),
        std_ln.as_tensor_arg(vectorization),
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
        fwd_intermediates_launch,
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
/// Returns both the outputs and forward intermediates needed for backward.
pub fn forward_multi<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> (TttOutputs<CubeTensor<R>>, FwdIntermediates<CubeTensor<R>>) {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [batch_size, num_heads, seq_len, _head_dim] = shape.dims();
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

    // Allocate forward intermediate tensors
    let x_hat_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let std_fused = empty_like::<R, F>(&inputs.ttt_lr_eta, [batch_size, num_heads, seq_len]);
    let grad_output_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let grad_x_hat_fused = empty_like::<R, F>(&inputs.xq, shape.clone());
    let grad_l_wrt_Z1 = empty_like::<R, F>(&inputs.xq, shape.clone());
    let x_hat_ln = empty_like::<R, F>(&inputs.xq, shape.clone());
    let std_ln = empty_like::<R, F>(&inputs.ttt_lr_eta, [batch_size, num_heads, seq_len]);

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
        x_hat_fused.as_handle_ref(),
        std_fused.as_handle_ref(),
        grad_output_fused.as_handle_ref(),
        grad_x_hat_fused.as_handle_ref(),
        grad_l_wrt_Z1.as_handle_ref(),
        x_hat_ln.as_handle_ref(),
        std_ln.as_handle_ref(),
        config,
        num_stages,
    );

    (
        TttOutputs {
            output,
            weight: weight_out,
            bias: bias_out,
        },
        FwdIntermediates {
            x_hat_fused,
            std_fused,
            grad_output_fused,
            grad_x_hat_fused,
            grad_l_wrt_Z1,
            x_hat_ln,
            std_ln,
        },
    )
}

impl FusedKernel<9, 10> for TttTileKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttTileOutputs<T>;
    type Backward = UseWithOut;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttTileOutputs<CubeTensor<R>> {
        let (outputs, fwd) = forward::<R, F>(inputs, config);
        TttTileOutputs {
            output: outputs.output,
            weight_out: outputs.weight,
            bias_out: outputs.bias,
            x_hat_fused: fwd.x_hat_fused,
            std_fused: fwd.std_fused,
            grad_output_fused: fwd.grad_output_fused,
            grad_x_hat_fused: fwd.grad_x_hat_fused,
            grad_l_wrt_Z1: fwd.grad_l_wrt_Z1,
            x_hat_ln: fwd.x_hat_ln,
            std_ln: fwd.std_ln,
        }
    }
}

impl CanBackwardWithOut<9, 10> for TttTileKernel {
    fn backward_with_out<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        outputs: TttTileOutputs<CubeTensor<R>>,
        grad_outputs: TttTileOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        let token_eta_shape = inputs.token_eta.shape.clone();
        let epsilon = config.epsilon();
        let threads = config.threads;

        let saved = TttSavedTensors {
            xq: inputs.xq,
            xk: inputs.xk,
            xv: inputs.xv,
            weight_init: inputs.weight,
            bias_init: inputs.bias,
            weight_last: outputs.weight_out, // Final weight from forward pass
            token_eta: inputs.token_eta,
            ttt_lr_eta: inputs.ttt_lr_eta,
            ln_weight: inputs.ln_weight,
            ln_bias: inputs.ln_bias,
        };

        let fwd = FwdIntermediates {
            x_hat_fused: outputs.x_hat_fused,
            std_fused: outputs.std_fused,
            grad_output_fused: outputs.grad_output_fused,
            grad_x_hat_fused: outputs.grad_x_hat_fused,
            grad_l_wrt_Z1: outputs.grad_l_wrt_Z1,
            x_hat_ln: outputs.x_hat_ln,
            std_ln: outputs.std_ln,
        };

        let grad_inputs = backward::<R, F>(saved, fwd, grad_outputs.output, epsilon, threads);

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

impl FusedKernel<9, 10> for TttTileMultiKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttTileOutputs<T>;
    type Backward = UseWithOut;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttTileOutputs<CubeTensor<R>> {
        let mini_batch_len = config.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel"
        );
        let (outputs, fwd) = forward_multi::<R, F>(inputs, config);
        TttTileOutputs {
            output: outputs.output,
            weight_out: outputs.weight,
            bias_out: outputs.bias,
            x_hat_fused: fwd.x_hat_fused,
            std_fused: fwd.std_fused,
            grad_output_fused: fwd.grad_output_fused,
            grad_x_hat_fused: fwd.grad_x_hat_fused,
            grad_l_wrt_Z1: fwd.grad_l_wrt_Z1,
            x_hat_ln: fwd.x_hat_ln,
            std_ln: fwd.std_ln,
        }
    }
}

impl CanBackwardWithOut<9, 10> for TttTileMultiKernel {
    fn backward_with_out<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        outputs: TttTileOutputs<CubeTensor<R>>,
        grad_outputs: TttTileOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        let mini_batch_len = config.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel backward"
        );

        // Capture scalars before moving into saved
        let token_eta_shape = inputs.token_eta.shape.clone();
        let epsilon = config.epsilon();
        let threads = config.threads;

        // Construct saved tensors from inputs and outputs
        let saved = TttSavedTensors {
            xq: inputs.xq,
            xk: inputs.xk,
            xv: inputs.xv,
            weight_init: inputs.weight,
            bias_init: inputs.bias,
            weight_last: outputs.weight_out, // Final weight from forward pass
            token_eta: inputs.token_eta,
            ttt_lr_eta: inputs.ttt_lr_eta,
            ln_weight: inputs.ln_weight,
            ln_bias: inputs.ln_bias,
        };

        let fwd = FwdIntermediates {
            x_hat_fused: outputs.x_hat_fused,
            std_fused: outputs.std_fused,
            grad_output_fused: outputs.grad_output_fused,
            grad_x_hat_fused: outputs.grad_x_hat_fused,
            grad_l_wrt_Z1: outputs.grad_l_wrt_Z1,
            x_hat_ln: outputs.x_hat_ln,
            std_ln: outputs.std_ln,
        };

        let grad_inputs = backward_multi::<R, F>(
            saved,
            fwd,
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
