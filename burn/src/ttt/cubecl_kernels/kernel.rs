use std::fmt::Debug;

use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};

use super::bundle::TensorBundle;

// =============================================================================
// Backward capability traits
// =============================================================================

/// Capability trait for kernels that can compute backward without saved outputs.
/// These kernels recompute intermediate values from inputs during backward.
pub trait CanBackwardNoOut<const N: usize, const M: usize>: FusedKernel<N, M> {
    fn backward_no_out<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        grad_outputs: Self::Outputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> Self::Inputs<CubeTensor<R>>;
}

/// Capability trait for kernels that require saved outputs for backward.
/// These kernels use the outputs from forward (e.g., weight_last) during backward.
pub trait CanBackwardWithOut<const N: usize, const M: usize>: FusedKernel<N, M> {
    fn backward_with_out<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        outputs: Self::Outputs<CubeTensor<R>>,
        grad_outputs: Self::Outputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> Self::Inputs<CubeTensor<R>>;
}

// =============================================================================
// Backward dispatcher trait and marker types
// =============================================================================

/// Trait for dispatching backward calls based on whether outputs are needed.
/// This is implemented by marker types that enforce the appropriate capability.
pub trait BackwardImpl<K: FusedKernel<N, M>, const N: usize, const M: usize> {
    /// Whether the forward pass should save outputs for backward.
    fn should_save_outputs() -> bool;

    /// Call the appropriate backward implementation.
    /// If `should_save_outputs()` is false, `outputs` will be None.
    fn call<R: CubeRuntime, F: FloatElement>(
        inputs: K::Inputs<CubeTensor<R>>,
        outputs: Option<K::Outputs<CubeTensor<R>>>,
        grad_outputs: K::Outputs<CubeTensor<R>>,
        config: K::Config,
    ) -> K::Inputs<CubeTensor<R>>;
}

/// Marker type indicating backward doesn't need saved outputs.
/// Using this type requires the kernel to implement `CanBackwardNoOut`.
pub struct UseNoOut;

impl<K, const N: usize, const M: usize> BackwardImpl<K, N, M> for UseNoOut
where
    K: CanBackwardNoOut<N, M>,
{
    fn should_save_outputs() -> bool {
        false
    }

    fn call<R: CubeRuntime, F: FloatElement>(
        inputs: K::Inputs<CubeTensor<R>>,
        _outputs: Option<K::Outputs<CubeTensor<R>>>,
        grad_outputs: K::Outputs<CubeTensor<R>>,
        config: K::Config,
    ) -> K::Inputs<CubeTensor<R>> {
        K::backward_no_out::<R, F>(inputs, grad_outputs, config)
    }
}

/// Marker type indicating backward requires saved outputs.
/// Using this type requires the kernel to implement `CanBackwardWithOut`.
pub struct UseWithOut;

impl<K, const N: usize, const M: usize> BackwardImpl<K, N, M> for UseWithOut
where
    K: CanBackwardWithOut<N, M>,
{
    fn should_save_outputs() -> bool {
        true
    }

    fn call<R: CubeRuntime, F: FloatElement>(
        inputs: K::Inputs<CubeTensor<R>>,
        outputs: Option<K::Outputs<CubeTensor<R>>>,
        grad_outputs: K::Outputs<CubeTensor<R>>,
        config: K::Config,
    ) -> K::Inputs<CubeTensor<R>> {
        let outputs = outputs.expect(
            "UseWithOut requires saved outputs, but none were provided. \
             This indicates a bug in the autodiff implementation.",
        );
        K::backward_with_out::<R, F>(inputs, outputs, grad_outputs, config)
    }
}

// =============================================================================
// Main kernel trait
// =============================================================================

/// Trait for defining fused CubeCL kernels.
///
/// `N` is the number of inputs, `M` is the number of outputs.
///
/// The `Backward` associated type determines how backward is computed:
/// - `UseNoOut` - backward recomputes intermediates from inputs (must impl `CanBackwardNoOut`)
/// - `UseWithOut` - backward uses saved outputs (must impl `CanBackwardWithOut`)
pub trait FusedKernel<const N: usize, const M: usize>: 'static + Send + Debug + Clone {
    type Inputs<T: Debug + Clone + Send>: TensorBundle<T, N>;
    type Outputs<T: Debug + Clone + Send>: TensorBundle<T, M>;
    type Backward: BackwardImpl<Self, N, M>;
    type Config: Debug + Clone + Send;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> Self::Outputs<CubeTensor<R>>;
}

// =============================================================================
// Backend trait
// =============================================================================

/// Backend trait for a specific kernel.
/// Allows different backends to implement it.
pub trait FusedKernelBackend<K, const N: usize, const M: usize>: Backend
where
    K: FusedKernel<N, M>,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>, config: K::Config) -> K::Outputs<FloatTensor<Self>>;
    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        outputs: Option<K::Outputs<FloatTensor<Self>>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>>;
}
