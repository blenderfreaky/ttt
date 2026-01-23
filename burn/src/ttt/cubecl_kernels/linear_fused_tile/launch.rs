//! Launch functions for the tiled TTT-Linear forward kernel.

use burn_backend::Shape;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeRuntime, FloatElement};
use cubecl::prelude::*;
use std::fmt::Debug;
use thundercube::prelude::{D4, D8, D16, D32, D64, D128, LINE_SIZE};

use crate::ttt::cubecl_kernels::FusedTttConfig;
use crate::ttt::cubecl_kernels::bundle::TensorBundle;
use crate::ttt::cubecl_kernels::kernel::FusedKernel;
use crate::ttt::cubecl_kernels::ttt::{TttInputs, TttOutputs};

use super::forward::{
    InputsLaunch, OutputsLaunch, Params, fused_ttt_forward_kernel, fused_ttt_forward_kernel_multi,
};

/// Supported tile configurations.
/// Format: (mini_batch_len, head_dim, CS_dim, F_dim, CS_Reg_dim, F_Reg_dim)
///
/// Sub-tile counts (must be ≤ 32):
/// - 8×32:   (8/4)×(32/8)   = 2×4 = 8 sub-tiles
/// - 8×64:   (8/4)×(64/16)  = 2×4 = 8 sub-tiles
/// - 16×32:  (16/4)×(32/8)  = 4×4 = 16 sub-tiles
/// - 16×64:  (16/4)×(64/16) = 4×4 = 16 sub-tiles
/// - 16×128: (16/4)×(128/32)= 4×4 = 16 sub-tiles
/// - 32×32:  (32/8)×(32/8)  = 4×4 = 16 sub-tiles
/// - 32×64:  (32/4)×(64/16) = 8×4 = 32 sub-tiles
macro_rules! supported_tile_configs {
    ($callback:ident!($($args:tt)*)) => {
        $callback!($($args)*
            ( 8,  32, D8,  D32,  D4, D8),
            ( 8,  64, D8,  D64,  D4, D16),
            (16,  32, D16, D32,  D4, D8),
            (16,  64, D16, D64,  D4, D16),
            (16, 128, D16, D128, D4, D32),
            (32,  32, D32, D32,  D8, D8),
            (32,  64, D32, D64,  D4, D16),
        )
    };
}

/// Dispatch macro for single-stage kernel.
macro_rules! impl_tile_dispatch {
    (
        $client:expr, $cube_count:expr, $cube_dim:expr,
        $inputs:expr, $outputs:expr, $config:expr,
        $mini_batch_len:expr, $head_dim:expr;
        $(($s:literal, $h:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty)),* $(,)?
    ) => {
        match ($mini_batch_len, $head_dim) {
            $(
                ($s, $h) => {
                    type P<E> = Params<E, $CS, $F, $CSR, $FR>;
                    fused_ttt_forward_kernel::launch::<P<_>, _>(
                        $client, $cube_count, $cube_dim, $inputs, $outputs, $config,
                    ).unwrap()
                }
            )*
            _ => {
                let supported = [$((stringify!($s), stringify!($h))),*];
                let supported_str: Vec<_> = supported.iter()
                    .map(|(s, h)| format!("{}×{}", s, h))
                    .collect();
                panic!(
                    "Unsupported tile size: mini_batch_len={}, head_dim={}. Supported: {}",
                    $mini_batch_len, $head_dim, supported_str.join(", ")
                )
            }
        }
    };
}

/// Dispatch macro for multi-stage kernel.
macro_rules! impl_tile_dispatch_multi {
    (
        $client:expr, $cube_count:expr, $cube_dim:expr,
        $inputs:expr, $outputs:expr, $num_stages:expr, $config:expr,
        $mini_batch_len:expr, $head_dim:expr;
        $(($s:literal, $h:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty)),* $(,)?
    ) => {
        match ($mini_batch_len, $head_dim) {
            $(
                ($s, $h) => {
                    type P<E> = Params<E, $CS, $F, $CSR, $FR>;
                    fused_ttt_forward_kernel_multi::launch::<P<_>, _>(
                        $client, $cube_count, $cube_dim, $inputs, $outputs, $num_stages, $config,
                    ).unwrap()
                }
            )*
            _ => {
                let supported = [$((stringify!($s), stringify!($h))),*];
                let supported_str: Vec<_> = supported.iter()
                    .map(|(s, h)| format!("{}×{}", s, h))
                    .collect();
                panic!(
                    "Unsupported tile size: mini_batch_len={}, head_dim={}. Supported: {}",
                    $mini_batch_len, $head_dim, supported_str.join(", ")
                )
            }
        }
    };
}

macro_rules! dispatch_tile_kernel {
    ($client:expr, $cube_count:expr, $cube_dim:expr, $inputs:expr, $outputs:expr, $config:expr, $mini_batch_len:expr, $head_dim:expr) => {
        supported_tile_configs!(impl_tile_dispatch!(
            $client, $cube_count, $cube_dim, $inputs, $outputs, $config, $mini_batch_len, $head_dim;
        ))
    };
}

macro_rules! dispatch_tile_kernel_multi {
    ($client:expr, $cube_count:expr, $cube_dim:expr, $inputs:expr, $outputs:expr, $num_stages:expr, $config:expr, $mini_batch_len:expr, $head_dim:expr) => {
        supported_tile_configs!(impl_tile_dispatch_multi!(
            $client, $cube_count, $cube_dim, $inputs, $outputs, $num_stages, $config, $mini_batch_len, $head_dim;
        ))
    };
}

/// Marker type for the tiled TTT kernel (single mini-batch).
#[derive(Debug, Clone, Copy)]
pub struct TttTileKernel;

/// Marker type for the multi-stage tiled TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttTileMultiKernel;

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

/// Launch the tiled TTT forward kernel with automatic tile size dispatch.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8×32, 8×64, 16×32, 16×64, 16×128, 32×32, 32×64
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

    // 32 threads required for plane reductions (PLANE_DIM = 32)
    let cube_dim = CubeDim::new_1d(32);

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

    dispatch_tile_kernel!(
        client,
        cube_count,
        cube_dim,
        inputs_launch,
        outputs_launch,
        config,
        seq_len,
        head_dim
    );
}

/// Forward pass using the tiled kernel.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8×32, 8×64, 16×32, 16×64, 16×128, 32×32, 32×64
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

/// Launch the multi-stage tiled TTT forward kernel.
///
/// Processes `num_stages` mini-batches in a single kernel launch.
/// Input seq_len should be `mini_batch_len * num_stages`.
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

    // 32 threads required for plane reductions (PLANE_DIM = 32)
    let cube_dim = CubeDim::new_1d(32);

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

    dispatch_tile_kernel_multi!(
        client,
        cube_count,
        cube_dim,
        inputs_launch,
        outputs_launch,
        ScalarArg::new(num_stages as u32),
        config,
        mini_batch_len,
        head_dim
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
pub fn forward_multi<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    mini_batch_len: usize,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [_batch_size, _num_heads, seq_len, head_dim] = shape.dims();

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

    let config = FusedTttConfig::new(mini_batch_len, head_dim, inputs.epsilon);

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

impl FusedKernel<9, 3> for TttTileMultiKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
    ) -> TttOutputs<CubeTensor<R>> {
        let mini_batch_len = inputs.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel"
        );
        forward_multi::<R, F>(inputs, mini_batch_len)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        _inputs: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttOutputs<CubeTensor<R>>,
    ) -> TttInputs<CubeTensor<R>> {
        unimplemented!("Backward pass not yet implemented for multi-stage tile kernel")
    }
}
