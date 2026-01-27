//! High-level API for the tiled TTT-Linear forward kernel.

use burn::tensor::{Tensor, TensorPrimitive};

use super::launch::{TttTileKernel, TttTileMultiKernel};
use crate::ttt::cubecl_kernels::{
    FusedTttBackend, FusedTttConfig, kernel::FusedKernelBackend, ttt::TttInputs,
};

/// Get the default thread count for a given (mini_batch_len, head_dim) configuration.
/// Returns the number of subtiles for the matching tile config.
pub fn default_threads(mini_batch_len: usize, head_dim: usize) -> usize {
    match (mini_batch_len, head_dim) {
        (8, 32) => 8,
        (8, 64) => 8,
        (16, 32) => 16,
        (16, 128) => 16,
        (64, 64) => 64,
        _ => panic!(
            "No default thread count for tile config: mini_batch_len={}, head_dim={}",
            mini_batch_len, head_dim
        ),
    }
}

/// Perform fused TTT-Linear forward pass using the tiled kernel.
///
/// This kernel uses shared memory tiles for optimized memory access patterns.
/// Currently only supports seq_len=8 and head_dim=8.
pub fn fused_ttt_tile_forward<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    config: FusedTttConfig,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let outputs = <B as FusedKernelBackend<TttTileKernel, 9, 10>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight_out)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias_out)),
    )
}

/// Perform fused TTT-Linear forward pass using the multi-stage tiled kernel.
///
/// Processes the full sequence in a single kernel launch by iterating over
/// mini-batches internally, keeping weight/bias in shared memory between stages.
///
/// # Arguments
/// * `mini_batch_len` - Size of each mini-batch (must be a supported tile size: 8, 16, 32)
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_tile_forward_multi<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    config: FusedTttConfig,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let outputs = <B as FusedKernelBackend<TttTileMultiKernel, 9, 10>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight_out)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias_out)),
    )
}
