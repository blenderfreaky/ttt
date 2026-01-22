use burn::tensor::ops::FloatTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

use crate::ttt::cubecl_kernels::kernel::{FusedKernel, FusedKernelBackend};

impl<K, const N: usize, const M: usize, R, F, I, BT> FusedKernelBackend<K, N, M>
    for CubeBackend<R, F, I, BT>
where
    K: FusedKernel<N, M>,
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>) -> K::Outputs<FloatTensor<Self>> {
        K::forward_launch::<R, F>(inputs)
    }

    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
    ) -> K::Inputs<FloatTensor<Self>> {
        K::backward_launch::<R, F>(inputs, grad_outputs)
    }
}
