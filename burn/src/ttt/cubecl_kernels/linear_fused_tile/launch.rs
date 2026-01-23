//! Launch functions for the tiled TTT-Linear forward kernel.

use burn_backend::Shape;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeRuntime, FloatElement};
use cubecl::prelude::*;
use std::fmt::Debug;
use thundercube::prelude::{D4, D8, D16, D32, D64, D128, LINE_SIZE};

use crate::ttt::cubecl_kernels::bundle::TensorBundle;
use crate::ttt::cubecl_kernels::kernel::FusedKernel;
use crate::ttt::cubecl_kernels::ttt::{TttInputs, TttOutputs};
use crate::ttt::cubecl_kernels::FusedTttConfig;

use super::forward::{fused_ttt_forward_kernel, InputsLaunch, OutputsLaunch, Params};

/// Declarative macro to define supported tile configurations and generate dispatch code.
///
/// Each entry is: (seq_len, head_dim, CS_dim, F_dim, CS_Reg_dim, F_Reg_dim)
/// The macro generates the match arms automatically.
///
/// Tile constraints:
/// - Number of sub-tiles = (CS/CS_Reg) × (F/F_Reg) must be ≤ 32 (plane size)
macro_rules! impl_tile_dispatch {
    // Entry point: defines configs and generates the dispatch function body
    (
        $client:expr, $cube_count:expr, $cube_dim:expr,
        $inputs:expr, $outputs:expr, $config:expr,
        $seq_len:expr, $head_dim:expr;
        // List of (seq_len, head_dim, CS, F, CS_Reg, F_Reg)
        $(($s:literal, $h:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty)),* $(,)?
    ) => {
        match ($seq_len, $head_dim) {
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
                    "Unsupported tile size: seq_len={}, head_dim={}. Supported: {}",
                    $seq_len, $head_dim, supported_str.join(", ")
                )
            }
        }
    };
}

/// Supported tile configurations.
/// Format: (seq_len, head_dim, CS_dim, F_dim, CS_Reg_dim, F_Reg_dim)
///
/// Sub-tile counts (must be ≤ 32):
/// - 8×32:   (8/4)×(32/8)   = 2×4 = 8 sub-tiles
/// - 8×64:   (8/4)×(64/16)  = 2×4 = 8 sub-tiles
/// - 16×32:  (16/4)×(32/8)  = 4×4 = 16 sub-tiles
/// - 16×64:  (16/4)×(64/16) = 4×4 = 16 sub-tiles
/// - 16×128: (16/4)×(128/32)= 4×4 = 16 sub-tiles
/// - 32×32:  (32/8)×(32/8)  = 4×4 = 16 sub-tiles
/// - 32×64:  (32/4)×(64/16) = 8×4 = 32 sub-tiles
macro_rules! dispatch_tile_kernel {
    ($client:expr, $cube_count:expr, $cube_dim:expr, $inputs:expr, $outputs:expr, $config:expr, $seq_len:expr, $head_dim:expr) => {
        impl_tile_dispatch!(
            $client, $cube_count, $cube_dim, $inputs, $outputs, $config, $seq_len, $head_dim;
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

/// Launch the tiled TTT forward kernel for 16x64 tiles specifically.
/// This is kept for backward compatibility.
#[deprecated(since = "0.2.0", note = "Use launch_tile_forward instead")]
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
    launch_tile_forward::<R, F>(
        client, xq, xk, xv, weight, bias, token_eta, ttt_lr_eta,
        ln_weight, ln_bias, output, weight_out, bias_out, config,
    )
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
