use std::fmt::Debug;

use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};

use crate::bundle::TensorBundle;

// =============================================================================
// Main kernel trait
// =============================================================================

/// Trait for defining fused CubeCL kernels.
///
/// The tensor counts are encoded in the `Array` associated types of each bundle,
/// avoiding const generics on the trait itself.
pub trait FusedKernel: 'static + Send + Debug + Clone {
    type Inputs<T: Debug + Clone + Send>: TensorBundle<T>;
    type Outputs<T: Debug + Clone + Send>: TensorBundle<T>;
    /// State saved from forward pass for backward. Only includes what's actually needed.
    type SavedState<T: Debug + Clone + Send>: TensorBundle<T>;
    type Config: Debug + Clone + Send;

    /// Run the forward pass, returning outputs and the state needed for backward.
    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> (
        Self::Outputs<CubeTensor<R>>,
        Self::SavedState<CubeTensor<R>>,
    );

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
pub trait FusedKernelBackend<K: FusedKernel>: Backend {
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (
        K::Outputs<FloatTensor<Self>>,
        K::SavedState<FloatTensor<Self>>,
    );

    fn backward(
        saved: K::SavedState<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>>;
}
