use burn::tensor::ops::FloatTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use crate::kernel::{FusedKernel, FusedKernelBackend};

impl<K, const N: usize, const M: usize, const S: usize, R, F, I, BT> FusedKernelBackend<K, N, M, S>
    for CubeBackend<R, F, I, BT>
where
    K: FusedKernel<N, M, S>,
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (K::Outputs<FloatTensor<Self>>, K::SavedState<FloatTensor<Self>>) {
        K::forward_launch::<R, F>(inputs, config)
    }

    fn backward(
        saved: K::SavedState<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>> {
        K::backward_launch::<R, F>(saved, grad_outputs, config)
    }
}
