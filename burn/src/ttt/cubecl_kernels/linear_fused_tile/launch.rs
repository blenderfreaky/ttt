//! Launch functions for the tiled TTT-Linear forward kernel.

use burn_backend::Shape;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeRuntime, FloatElement};
use cubecl::prelude::*;
use std::fmt::Debug;
use thundercube::prelude::{D4, D16, D64, LINE_SIZE};

use crate::ttt::cubecl_kernels::bundle::TensorBundle;
use crate::ttt::cubecl_kernels::kernel::FusedKernel;
use crate::ttt::cubecl_kernels::ttt::{TttInputs, TttOutputs};
use crate::ttt::cubecl_kernels::FusedTttConfig;

use super::forward::{fused_ttt_forward_kernel, InputsLaunch, OutputsLaunch, Params};

/// Marker type for the tiled TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttTileKernel;

/// Create an empty tensor with the same client/device as the template.
fn empty_like<R: CubeRuntime, F: FloatElement>(
    template: &CubeTensor<R>,
    shape: impl Into<Shape>,
) -> CubeTensor<R> {
    empty_device::<R, F>(
        template.client.clone(),
        template.device.clone(),
        shape.into(),
    )
}

/// Launch the tiled TTT forward kernel for 8x8 tiles.
pub fn launch_tile_forward_64x64<R: Runtime, F: Float + CubeElement>(
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

    // 32 threads required for plane reductions (PLANE_DIM = 32)
    // With 16 sub-tiles, threads 16-31 are idle for tile ops but participate in reductions
    let cube_dim = CubeDim::new_1d(32);

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Params: E=F, CS=D16, F=D64, CS_Reg=D4, F_Reg=D16
    // All tile types have 16 sub-tiles (threads 0-15 active, 16-31 idle for tile ops):
    // - cs_cs (16×16 with 4×4): (16/4)² = 16 ✓
    // - cs_f (16×64 with 4×16): (16/4)×(64/16) = 4×4 = 16 ✓
    // - f_f (64×64 with 16×16): (64/16)² = 16 ✓
    type P<F> = Params<F, D16, D64, D4, D16>;

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    // Use the TensorHandleRef's as_tensor_arg which correctly uses elem_size from the handle
    // (This matches how burn's as_tensor_arg works internally)
    unsafe {
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
        fused_ttt_forward_kernel::launch::<P<F>, R>(
            client,
            cube_count,
            cube_dim,
            inputs_launch,
            outputs_launch,
            config,
        )
        .unwrap();
    }
}

/// Forward pass using the tiled kernel.
pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [_batch_size, _num_heads, seq_len, head_dim] = shape.dims();

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    let config = FusedTttConfig::new(seq_len, head_dim, inputs.epsilon);

    // Currently only support 16x64 tiles (CS=16, F=64)
    assert_eq!(seq_len, 16, "Tile kernel currently only supports seq_len=16");
    assert_eq!(head_dim, 64, "Tile kernel currently only supports head_dim=64");

    launch_tile_forward_64x64::<R, F>(
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

impl FusedKernel<9, 3> for TttTileKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
    ) -> TttOutputs<CubeTensor<R>> {
        forward::<R, F>(inputs)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        _inputs: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttOutputs<CubeTensor<R>>,
    ) -> TttInputs<CubeTensor<R>> {
        // Backward not yet implemented for tile kernel
        unimplemented!("Backward pass not yet implemented for tile kernel")
    }
}
