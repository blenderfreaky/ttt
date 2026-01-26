use burn::{
    backend::ir::{InitOperationIr, OperationIr},
    tensor::ops::FloatTensor,
};
use burn_backend::TensorMetadata;
use burn_fusion::{
    Fusion, FusionBackend, NoOp, client::GlobalFusionClient, stream::OperationStreams,
};

use crate::ttt::cubecl_kernels::{
    TensorBundle,
    kernel::{FusedKernel, FusedKernelBackend},
};

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
    // Mapped<U> -> K::Inputs/Outputs<U> conversions (trivially satisfied since Mapped<U> = Self<U>)
    <K::Inputs<FloatTensor<Self>> as TensorBundle<FloatTensor<Self>, N>>::Mapped<FloatTensor<B>>:
        Into<K::Inputs<FloatTensor<B>>>,
    <K::Outputs<FloatTensor<B>> as TensorBundle<FloatTensor<B>, M>>::Mapped<FloatTensor<Self>>:
        Into<K::Outputs<FloatTensor<Self>>>,
    <K::Outputs<FloatTensor<Self>> as TensorBundle<FloatTensor<Self>, M>>::Mapped<FloatTensor<B>>:
        Into<K::Outputs<FloatTensor<B>>>,
    <K::Inputs<FloatTensor<B>> as TensorBundle<FloatTensor<B>, N>>::Mapped<FloatTensor<Self>>:
        Into<K::Inputs<FloatTensor<Self>>>,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>) -> K::Outputs<FloatTensor<Self>> {
        let client = inputs.client().clone();
        let inner_inputs = inputs.map(|t| fusion_in::<B>(t)).into();
        let outputs = B::forward(inner_inputs);
        outputs.map(|t| fusion_out::<B>(t, &client)).into()
    }

    fn backward(
        inputs: K::Inputs<FloatTensor<Self>>,
        outputs: Option<K::Outputs<FloatTensor<Self>>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
    ) -> K::Inputs<FloatTensor<Self>> {
        let client = inputs.client().clone();
        let inner_inputs = inputs.map(|t| fusion_in::<B>(t)).into();
        let inner_outputs = outputs.map(|o| o.map(|t| fusion_in::<B>(t)).into());
        let inner_grad_outputs = grad_outputs.map(|t| fusion_in::<B>(t)).into();
        let grads = B::backward(inner_inputs, inner_outputs, inner_grad_outputs);
        grads.map(|t| fusion_out::<B>(t, &client)).into()
    }
}
