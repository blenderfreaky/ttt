use std::fmt::Debug;

use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};

use crate::bundle::TensorBundle;

// =============================================================================
// Main kernel trait
// =============================================================================

/// Trait for defining fused CubeCL kernels.
///
/// `N` is the number of inputs, `M` is the number of outputs, `S` is the number
/// of tensors in the saved state for backward.
///
/// The kernel defines what state needs to be saved for backward via `SavedState`.
/// This can include a subset of inputs and/or forward intermediates.
pub trait FusedKernel<const N: usize, const M: usize, const S: usize>:
    'static + Send + Debug + Clone
{
    type Inputs<T: Debug + Clone + Send>: TensorBundle<T, N>;
    type Outputs<T: Debug + Clone + Send>: TensorBundle<T, M>;
    /// State saved from forward pass for backward. Only includes what's actually needed.
    type SavedState<T: Debug + Clone + Send>: TensorBundle<T, S>;
    type Config: Debug + Clone + Send;

    /// Run the forward pass, returning outputs and the state needed for backward.
    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> (Self::Outputs<CubeTensor<R>>, Self::SavedState<CubeTensor<R>>);

    /// Run the backward pass using saved state and upstream gradients.
    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: Self::SavedState<CubeTensor<R>>,
        grad_outputs: Self::Outputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> Self::Inputs<CubeTensor<R>>;
}

// =============================================================================
// Backend trait
// =============================================================================

/// Backend trait for a specific kernel.
/// Allows different backends to implement it.
pub trait FusedKernelBackend<K, const N: usize, const M: usize, const S: usize>: Backend
where
    K: FusedKernel<N, M, S>,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (K::Outputs<FloatTensor<Self>>, K::SavedState<FloatTensor<Self>>);

    fn backward(
        saved: K::SavedState<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>>;
}
