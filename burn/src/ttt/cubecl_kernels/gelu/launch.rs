use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeRuntime, FloatElement};
use std::fmt::Debug;

use crate::ttt::cubecl_kernels::gelu_tanh::{
    launch_gelu_bwd_forward, launch_gelu_tanh, launch_gelu_tanh_backward,
    launch_gelu_tanh_backward_backward,
};
use crate::ttt::cubecl_kernels::kernel::{CanBackwardNoOut, FusedKernel, UseNoOut};

use super::types::{
    GeluBwdKernel, GeluInput, GeluOutput, GeluTanhBackwardBackwardKernel, GeluTanhBackwardKernel,
    GeluTanhKernel,
};

fn empty_like<R: CubeRuntime, F: FloatElement>(template: &CubeTensor<R>) -> CubeTensor<R> {
    empty_device::<R, F>(
        template.client.clone(),
        template.device.clone(),
        template.shape.clone(),
    )
}

// GELU tanh forward kernel
impl FusedKernel<1, 1> for GeluTanhKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type Backward = UseNoOut;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
    ) -> GeluOutput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_tanh::<R, F>(&input.client, input.as_handle_ref(), output.as_handle_ref());

        GeluOutput { output }
    }
}

impl CanBackwardNoOut<1, 1> for GeluTanhKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

// GELU backward derivative forward kernel (computes gelu'(x))
impl FusedKernel<1, 1> for GeluBwdKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type Backward = UseNoOut;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
    ) -> GeluOutput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_bwd_forward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
        );

        GeluOutput { output }
    }
}

impl CanBackwardNoOut<1, 1> for GeluBwdKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

// GELU tanh backward kernel (for second-order gradients)
impl FusedKernel<1, 1> for GeluTanhBackwardKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type Backward = UseNoOut;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
    ) -> GeluOutput<CubeTensor<R>> {
        // This is gelu_bwd(x) - same as GeluBwdKernel
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_bwd_forward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
        );

        GeluOutput { output }
    }
}

impl CanBackwardNoOut<1, 1> for GeluTanhBackwardKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

// GELU tanh backward-backward kernel (third-order not supported)
impl FusedKernel<1, 1> for GeluTanhBackwardBackwardKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type Backward = UseNoOut;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
    ) -> GeluOutput<CubeTensor<R>> {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            // For forward of backward_backward, we use ones as grad_output
            // This shouldn't be called directly - it's for the autodiff system
            output.as_handle_ref(),
            output.as_handle_ref(),
        );

        GeluOutput { output }
    }
}

impl CanBackwardNoOut<1, 1> for GeluTanhBackwardBackwardKernel {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        _inputs: GeluInput<CubeTensor<R>>,
        _grad_outputs: GeluOutput<CubeTensor<R>>,
    ) -> GeluInput<CubeTensor<R>> {
        panic!("Third-order gradients through GELU are not supported")
    }
}
