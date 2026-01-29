use burn::tensor::ops::FloatTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use crate::ttt::cubecl_kernels::kernel::{BackwardImpl, FusedKernel, FusedKernelBackend};

impl<K, const N: usize, const M: usize, R, F, I, BT> FusedKernelBackend<K, N, M>
    for CubeBackend<R, F, I, BT>
where
    K: FusedKernel<N, M>,
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Outputs<FloatTensor<Self>> {
        K::forward_launch::<R, F>(inputs, config)
    }

    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        outputs: Option<K::Outputs<FloatTensor<Self>>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>> {
        K::Backward::call::<R, F>(inputs, outputs, grad_outputs, config)
    }
}
