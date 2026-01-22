use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeRuntime, FloatElement};
use std::fmt::Debug;

use super::bundle::TensorBundle;

/// Trait for defining fused CubeCL kernels.
///
/// `N` is the number of inputs, `M` is the number of outputs.
pub trait FusedKernel<const N: usize, const M: usize>: 'static + Send + Debug + Clone {
    type Inputs<T: Debug + Clone + Send>: TensorBundle<T, N>;
    type Outputs<T: Debug + Clone + Send>: TensorBundle<T, M>;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
    ) -> Self::Outputs<CubeTensor<R>>;

    /// Returns gradients with same structure as inputs.
    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        grad_outputs: Self::Outputs<CubeTensor<R>>,
    ) -> Self::Inputs<CubeTensor<R>>;
}

/// Backend trait for a specific kernel.
/// Allows different backends to implement it.
pub trait FusedKernelBackend<K, const N: usize, const M: usize>: Backend
where
    K: FusedKernel<N, M>,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>) -> K::Outputs<FloatTensor<Self>>;
    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
    ) -> K::Inputs<FloatTensor<Self>>;
}
