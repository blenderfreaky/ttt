use std::fmt::Debug;

use burn_backend::Shape;
use burn_cubecl::{
    CubeRuntime, FloatElement,
    kernel::{
        into_contiguous,
        reduce::{KernelReduceStrategy, reduce_dim},
    },
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use cubek::reduce::components::instructions::ReduceOperationConfig;

use super::{backward::launch_fused_ttt_backward, forward::launch_fused_ttt_forward};
use crate::ttt::cubecl_kernels::{
    FusedTttConfig,
    bundle::TensorBundle,
    kernel::{CanBackwardNoOut, FusedKernel, UseNoOut},
    ttt::{TttInputs, TttKernel, TttOutputs},
};

/// Create an empty tensor with the same client/device as the template.
pub fn empty_like<R: CubeRuntime, F: FloatElement>(
    template: &CubeTensor<R>,
    shape: impl Into<Shape>,
) -> CubeTensor<R> {
    empty_device::<R, F>(
        template.client.clone(),
        template.device.clone(),
        shape.into(),
    )
}

/// Reduce sum over batch dimension (dim 0) and reshape to remove that dimension.
fn reduce_sum_batch<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    output_shape: impl Into<Shape>,
) -> CubeTensor<R> {
    let reduced = reduce_dim::<R>(
        tensor,
        None,
        0,
        KernelReduceStrategy::Unspecified,
        ReduceOperationConfig::Sum,
    )
    .expect("reduce_dim failed");

    let output_shape = output_shape.into();
    let strides = output_shape
        .dims
        .iter()
        .rev()
        .scan(1, |acc, &d| {
            let s = *acc;
            *acc *= d;
            Some(s)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    CubeTensor::new(
        reduced.client,
        reduced.handle,
        output_shape,
        reduced.device,
        strides,
        reduced.dtype,
    )
}

pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    epsilon: f32,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [_batch_size, _num_heads, seq_len, head_dim] = shape.dims();

    let output = empty_like::<R, F>(&inputs.xq, shape);

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, 0); // threads unused for non-tile kernel

    launch_fused_ttt_forward::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        output.as_handle_ref(),
        config,
    );

    TttOutputs {
        output,
        weight: inputs.weight,
        bias: inputs.bias,
    }
}

pub fn backward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    grad_outputs: TttOutputs<CubeTensor<R>>,
    epsilon: f32,
) -> TttInputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);
    let grad_output = into_contiguous(grad_outputs.output);

    let [batch_size, num_heads, seq_len, head_dim] = inputs.xq.shape.dims();
    let t = &inputs.xq; // template for empty_like

    let grad_xq = empty_like::<R, F>(t, inputs.xq.shape.clone());
    let grad_xk = empty_like::<R, F>(t, inputs.xk.shape.clone());
    let grad_xv = empty_like::<R, F>(t, inputs.xv.shape.clone());
    let grad_weight = empty_like::<R, F>(t, inputs.weight.shape.clone());
    let grad_bias = empty_like::<R, F>(t, inputs.bias.shape.clone());
    // TODO: token_eta gradient is currently zeros
    let grad_token_eta = empty_like::<R, F>(t, inputs.token_eta.shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(t, inputs.ttt_lr_eta.shape.clone());
    let grad_ln_weight_per_batch = empty_like::<R, F>(t, [batch_size, num_heads, head_dim]);
    let grad_ln_bias_per_batch = empty_like::<R, F>(t, [batch_size, num_heads, head_dim]);

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, 0); // threads unused for non-tile kernel

    launch_fused_ttt_backward::<R, F>(
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
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight.as_handle_ref(),
        grad_bias.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_ln_weight_per_batch.as_handle_ref(),
        grad_ln_bias_per_batch.as_handle_ref(),
        config,
    );

    let grad_ln_weight = reduce_sum_batch(grad_ln_weight_per_batch, [num_heads, head_dim]);
    let grad_ln_bias = reduce_sum_batch(grad_ln_bias_per_batch, [num_heads, head_dim]);

    TttInputs {
        xq: grad_xq,
        xk: grad_xk,
        xv: grad_xv,
        weight: grad_weight,
        bias: grad_bias,
        token_eta: grad_token_eta,
        ttt_lr_eta: grad_ttt_lr_eta,
        ln_weight: grad_ln_weight,
        ln_bias: grad_ln_bias,
    }
}

impl FusedKernel<9, 3> for TttKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type Backward = UseNoOut;
    type Config = f32;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        epsilon: f32,
    ) -> TttOutputs<CubeTensor<R>> {
        forward::<R, F>(inputs, epsilon)
    }
}

impl CanBackwardNoOut<9, 3> for TttKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        epsilon: f32,
    ) -> TttInputs<CubeTensor<R>> {
        backward::<R, F>(inputs, grad_outputs, epsilon)
    }
}
