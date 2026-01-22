use burn::backend::ir::{InitOperationIr, OperationIr};
use burn::tensor::ops::FloatTensor;
use burn_backend::TensorMetadata;
use burn_fusion::{
    Fusion, FusionBackend, NoOp, client::GlobalFusionClient, stream::OperationStreams,
};

use crate::ttt::cubecl_kernels::kernel::{FusedKernel, FusedKernelBackend};

fn fusion_in<B: FusionBackend>(tensor: FloatTensor<Fusion<B>>) -> FloatTensor<B> {
    tensor.client.clone().resolve_tensor_float::<B>(tensor)
}

fn fusion_out<B: FusionBackend>(
    tensor: FloatTensor<B>,
    client: &GlobalFusionClient<B::FusionRuntime>,
) -> FloatTensor<Fusion<B>> {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let handle = B::float_tensor_handle(tensor);
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

    let mut new = client.register(
        OperationStreams::default(),
        OperationIr::Init(desc),
        NoOp::<B>::new(),
    );

    assert_eq!(new.len(), 1);
    new.pop().unwrap()
}

/// Helper trait to get the client from the first tensor in a bundle.
/// Generic over `B` to allow the compiler to infer the backend type.
pub trait HasClient<B: FusionBackend> {
    fn client(&self) -> &GlobalFusionClient<B::FusionRuntime>;
}

impl<K, const N: usize, const M: usize, B> FusedKernelBackend<K, N, M> for Fusion<B>
where
    K: FusedKernel<N, M>,
    B: FusedKernelBackend<K, N, M> + FusionBackend,
    K::Inputs<FloatTensor<Self>>: HasClient<B>,
    K::Outputs<FloatTensor<Self>>: HasClient<B>,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>) -> K::Outputs<FloatTensor<Self>> {
        let client = inputs.client().clone();
        let inner_inputs = inputs.map(|t| fusion_in::<B>(t));
        let outputs = B::forward(inner_inputs);
        outputs.map(|t| fusion_out::<B>(t, &client))
    }

    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
    ) -> K::Inputs<FloatTensor<Self>> {
        let client = inputs.client().clone();
        let inner_inputs = inputs.map(|t| fusion_in::<B>(t));
        let inner_grad_outputs = grad_outputs.map(|t| fusion_in::<B>(t));
        let grads = B::backward(inner_inputs, inner_grad_outputs);
        grads.map(|t| fusion_out::<B>(t, &client))
    }
}
