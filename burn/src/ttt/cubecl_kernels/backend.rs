use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_fusion::FusionBackend;

pub trait FusedTttBackend: Backend {
    /// Returns (output, updated_weight, updated_bias)
    fn fused_ttt_forward(
        xq: FloatTensor<Self>,
        xk: FloatTensor<Self>,
        xv: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        token_eta: FloatTensor<Self>,
        ttt_lr_eta: FloatTensor<Self>,
        ln_weight: FloatTensor<Self>,
        ln_bias: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// Returns (grad_xq, grad_xk, grad_xv, grad_weight, grad_bias, grad_ttt_lr_eta, grad_ln_weight, grad_ln_bias)
    fn fused_ttt_backward(
        xq: FloatTensor<Self>,
        xk: FloatTensor<Self>,
        xv: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        token_eta: FloatTensor<Self>,
        ttt_lr_eta: FloatTensor<Self>,
        ln_weight: FloatTensor<Self>,
        ln_bias: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        epsilon: f32,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

pub mod cube_impl {
    use super::{FloatTensor, FusedTttBackend};
    use burn_backend::Shape;
    use burn_cubecl::kernel::into_contiguous;
    use burn_cubecl::kernel::reduce::{KernelReduceStrategy, reduce_dim};
    use burn_cubecl::ops::numeric::empty_device;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
    use cubek::reduce::components::instructions::ReduceOperationConfig;

    use crate::ttt::cubecl_kernels::FusedTttConfig;
    use crate::ttt::cubecl_kernels::linear_backward::launch_fused_ttt_backward;
    use crate::ttt::cubecl_kernels::linear_forward::launch_fused_ttt_forward;

    pub fn fused_ttt_forward_launch<R: CubeRuntime, F: FloatElement>(
        xq: CubeTensor<R>,
        xk: CubeTensor<R>,
        xv: CubeTensor<R>,
        weight: CubeTensor<R>,
        bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ttt_lr_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
        epsilon: f32,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        let xq = into_contiguous(xq);
        let xk = into_contiguous(xk);
        let xv = into_contiguous(xv);
        let weight = into_contiguous(weight);
        let bias = into_contiguous(bias);
        let token_eta = into_contiguous(token_eta);
        let ttt_lr_eta = into_contiguous(ttt_lr_eta);
        let ln_weight = into_contiguous(ln_weight);
        let ln_bias = into_contiguous(ln_bias);

        let shape = xq.shape.clone();
        let [_batch_size, _num_heads, seq_len, head_dim] = shape.dims();

        let output = empty_device::<R, F>(xq.client.clone(), xq.device.clone(), shape);

        let config = FusedTttConfig::new(seq_len, head_dim, epsilon);

        launch_fused_ttt_forward::<R, F>(
            &xq.client,
            xq.as_handle_ref(),
            xk.as_handle_ref(),
            xv.as_handle_ref(),
            token_eta.as_handle_ref(),
            ttt_lr_eta.as_handle_ref(),
            ln_weight.as_handle_ref(),
            ln_bias.as_handle_ref(),
            weight.as_handle_ref(),
            bias.as_handle_ref(),
            output.as_handle_ref(),
            config,
        );

        (output, weight, bias)
    }

    pub fn fused_ttt_backward_launch<R: CubeRuntime, F: FloatElement>(
        xq: CubeTensor<R>,
        xk: CubeTensor<R>,
        xv: CubeTensor<R>,
        weight: CubeTensor<R>,
        bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ttt_lr_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
        grad_output: CubeTensor<R>,
        epsilon: f32,
    ) -> (
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    ) {
        let xq = into_contiguous(xq);
        let xk = into_contiguous(xk);
        let xv = into_contiguous(xv);
        let weight = into_contiguous(weight);
        let bias = into_contiguous(bias);
        let token_eta = into_contiguous(token_eta);
        let ttt_lr_eta = into_contiguous(ttt_lr_eta);
        let ln_weight = into_contiguous(ln_weight);
        let ln_bias = into_contiguous(ln_bias);
        let grad_output = into_contiguous(grad_output);

        let [batch_size, num_heads, seq_len, head_dim] = xq.shape.dims();

        let grad_xq = empty_device::<R, F>(xq.client.clone(), xq.device.clone(), xq.shape.clone());
        let grad_xk = empty_device::<R, F>(xq.client.clone(), xq.device.clone(), xk.shape.clone());
        let grad_xv = empty_device::<R, F>(xq.client.clone(), xq.device.clone(), xv.shape.clone());
        let grad_weight =
            empty_device::<R, F>(xq.client.clone(), xq.device.clone(), weight.shape.clone());
        let grad_bias =
            empty_device::<R, F>(xq.client.clone(), xq.device.clone(), bias.shape.clone());
        let grad_ttt_lr_eta = empty_device::<R, F>(
            xq.client.clone(),
            xq.device.clone(),
            ttt_lr_eta.shape.clone(),
        );

        let grad_ln_weight_per_batch = empty_device::<R, F>(
            xq.client.clone(),
            xq.device.clone(),
            Shape::new([batch_size, num_heads, head_dim]),
        );
        let grad_ln_bias_per_batch = empty_device::<R, F>(
            xq.client.clone(),
            xq.device.clone(),
            Shape::new([batch_size, num_heads, head_dim]),
        );

        let config = FusedTttConfig::new(seq_len, head_dim, epsilon);

        launch_fused_ttt_backward::<R, F>(
            &xq.client,
            xq.as_handle_ref(),
            xk.as_handle_ref(),
            xv.as_handle_ref(),
            weight.as_handle_ref(),
            bias.as_handle_ref(),
            token_eta.as_handle_ref(),
            ttt_lr_eta.as_handle_ref(),
            ln_weight.as_handle_ref(),
            ln_bias.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_xq.as_handle_ref(),
            grad_xk.as_handle_ref(),
            grad_xv.as_handle_ref(),
            grad_weight.as_handle_ref(),
            grad_bias.as_handle_ref(),
            grad_ttt_lr_eta.as_handle_ref(),
            grad_ln_weight_per_batch.as_handle_ref(),
            grad_ln_bias_per_batch.as_handle_ref(),
            config,
        );

        let grad_ln_weight = reduce_dim::<R>(
            grad_ln_weight_per_batch,
            None,
            0,
            KernelReduceStrategy::Unspecified,
            ReduceOperationConfig::Sum,
        )
        .expect("reduce_dim failed for grad_ln_weight");
        let grad_ln_weight = CubeTensor::new(
            grad_ln_weight.client,
            grad_ln_weight.handle,
            Shape::new([num_heads, head_dim]),
            grad_ln_weight.device,
            vec![head_dim, 1],
            grad_ln_weight.dtype,
        );

        let grad_ln_bias = reduce_dim::<R>(
            grad_ln_bias_per_batch,
            None,
            0,
            KernelReduceStrategy::Unspecified,
            ReduceOperationConfig::Sum,
        )
        .expect("reduce_dim failed for grad_ln_bias");
        let grad_ln_bias = CubeTensor::new(
            grad_ln_bias.client,
            grad_ln_bias.handle,
            Shape::new([num_heads, head_dim]),
            grad_ln_bias.device,
            vec![head_dim, 1],
            grad_ln_bias.dtype,
        );

        (
            grad_xq,
            grad_xk,
            grad_xv,
            grad_weight,
            grad_bias,
            grad_ttt_lr_eta,
            grad_ln_weight,
            grad_ln_bias,
        )
    }

    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusedTttBackend
        for CubeBackend<R, F, I, BT>
    {
        fn fused_ttt_forward(
            xq: FloatTensor<Self>,
            xk: FloatTensor<Self>,
            xv: FloatTensor<Self>,
            weight: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            token_eta: FloatTensor<Self>,
            ttt_lr_eta: FloatTensor<Self>,
            ln_weight: FloatTensor<Self>,
            ln_bias: FloatTensor<Self>,
            epsilon: f32,
        ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
            fused_ttt_forward_launch::<R, F>(
                xq, xk, xv, weight, bias, token_eta, ttt_lr_eta, ln_weight, ln_bias, epsilon,
            )
        }

        fn fused_ttt_backward(
            xq: FloatTensor<Self>,
            xk: FloatTensor<Self>,
            xv: FloatTensor<Self>,
            weight: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            token_eta: FloatTensor<Self>,
            ttt_lr_eta: FloatTensor<Self>,
            ln_weight: FloatTensor<Self>,
            ln_bias: FloatTensor<Self>,
            grad_output: FloatTensor<Self>,
            epsilon: f32,
        ) -> (
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
        ) {
            fused_ttt_backward_launch::<R, F>(
                xq,
                xk,
                xv,
                weight,
                bias,
                token_eta,
                ttt_lr_eta,
                ln_weight,
                ln_bias,
                grad_output,
                epsilon,
            )
        }
    }
}

pub mod autodiff_impl {
    use super::{Backend, FloatTensor, FusedTttBackend};
    use burn::backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    };

    #[derive(Debug, Clone)]
    struct FusedTttState<B: Backend> {
        xq: B::FloatTensorPrimitive,
        xk: B::FloatTensorPrimitive,
        xv: B::FloatTensorPrimitive,
        weight: B::FloatTensorPrimitive,
        bias: B::FloatTensorPrimitive,
        token_eta: B::FloatTensorPrimitive,
        ttt_lr_eta: B::FloatTensorPrimitive,
        ln_weight: B::FloatTensorPrimitive,
        ln_bias: B::FloatTensorPrimitive,
        epsilon: f32,
    }

    #[derive(Debug)]
    struct FusedTttBackwardOp;

    impl<B: FusedTttBackend> Backward<B, 8> for FusedTttBackwardOp
    where
        B::FloatTensorPrimitive: Clone,
    {
        type State = FusedTttState<B>;

        fn backward(
            self,
            ops: Ops<Self::State, 8>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let grad_output = grads.consume::<B>(&ops.node);
            let state = ops.state;

            let (
                grad_xq,
                grad_xk,
                grad_xv,
                grad_weight,
                grad_bias,
                grad_ttt_lr_eta,
                grad_ln_weight,
                grad_ln_bias,
            ) = B::fused_ttt_backward(
                state.xq,
                state.xk,
                state.xv,
                state.weight,
                state.bias,
                state.token_eta,
                state.ttt_lr_eta,
                state.ln_weight,
                state.ln_bias,
                grad_output,
                state.epsilon,
            );

            // Register gradients for all tracked parents
            // Parents: [xq, xk, xv, weight, bias, ttt_lr_eta, ln_weight, ln_bias]
            if let Some(node) = ops.parents.first().and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_xq);
            }
            if let Some(node) = ops.parents.get(1).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_xk);
            }
            if let Some(node) = ops.parents.get(2).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_xv);
            }
            if let Some(node) = ops.parents.get(3).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_weight);
            }
            if let Some(node) = ops.parents.get(4).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_bias);
            }
            if let Some(node) = ops.parents.get(5).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_ttt_lr_eta);
            }
            if let Some(node) = ops.parents.get(6).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_ln_weight);
            }
            if let Some(node) = ops.parents.get(7).and_then(|n| n.as_ref()) {
                grads.register::<B>(node.id, grad_ln_bias);
            }
        }
    }

    impl<B: FusedTttBackend, C: CheckpointStrategy> FusedTttBackend for Autodiff<B, C>
    where
        B::FloatTensorPrimitive: Clone,
    {
        fn fused_ttt_forward(
            xq: FloatTensor<Self>,
            xk: FloatTensor<Self>,
            xv: FloatTensor<Self>,
            weight: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            token_eta: FloatTensor<Self>,
            ttt_lr_eta: FloatTensor<Self>,
            ln_weight: FloatTensor<Self>,
            ln_bias: FloatTensor<Self>,
            epsilon: f32,
        ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
            use burn::prelude::ElementConversion;
            let zero: B::FloatElem = 0.0_f32.elem();
            let weight_orig = B::float_add_scalar(weight.primitive.clone(), zero);
            let bias_orig = B::float_add_scalar(bias.primitive.clone(), zero);

            let (output_inner, _weight_updated, _bias_updated) = B::fused_ttt_forward(
                xq.primitive.clone(),
                xk.primitive.clone(),
                xv.primitive.clone(),
                weight.primitive.clone(),
                bias.primitive.clone(),
                token_eta.primitive.clone(),
                ttt_lr_eta.primitive.clone(),
                ln_weight.primitive.clone(),
                ln_bias.primitive.clone(),
                epsilon,
            );

            match FusedTttBackwardOp
                .prepare::<C>([
                    xq.node.clone(),
                    xk.node.clone(),
                    xv.node.clone(),
                    weight.node.clone(),
                    bias.node.clone(),
                    ttt_lr_eta.node.clone(),
                    ln_weight.node.clone(),
                    ln_bias.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    let state = FusedTttState {
                        xq: xq.primitive,
                        xk: xk.primitive,
                        xv: xv.primitive,
                        weight: weight_orig, // Use the saved original, not the modified one
                        bias: bias_orig,     // Use the saved original, not the modified one
                        token_eta: token_eta.primitive,
                        ttt_lr_eta: ttt_lr_eta.primitive,
                        ln_weight: ln_weight.primitive,
                        ln_bias: ln_bias.primitive,
                        epsilon,
                    };

                    let output = prep.finish(state, output_inner);

                    (output, weight, bias)
                }
                OpsKind::UnTracked(prep) => {
                    let output = prep.finish(output_inner);
                    (output, weight, bias)
                }
            }
        }

        fn fused_ttt_backward(
            _xq: FloatTensor<Self>,
            _xk: FloatTensor<Self>,
            _xv: FloatTensor<Self>,
            _weight: FloatTensor<Self>,
            _bias: FloatTensor<Self>,
            _token_eta: FloatTensor<Self>,
            _ttt_lr_eta: FloatTensor<Self>,
            _ln_weight: FloatTensor<Self>,
            _ln_bias: FloatTensor<Self>,
            _grad_output: FloatTensor<Self>,
            _epsilon: f32,
        ) -> (
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
        ) {
            // This method is never called during normal backpropagation.
            // This would only be needed for second-order gradients, which we don't support.
            panic!("Second-order gradients through fused TTT backward are not supported")
        }
    }
}

pub mod api {
    use super::FusedTttBackend;
    use burn::tensor::{Tensor, TensorPrimitive};

    pub fn fused_ttt_forward<B: FusedTttBackend>(
        xq: Tensor<B, 4>,
        xk: Tensor<B, 4>,
        xv: Tensor<B, 4>,
        weight: Tensor<B, 4>,
        bias: Tensor<B, 3>,
        token_eta: Tensor<B, 1>,
        ttt_lr_eta: Tensor<B, 3>,
        ln_weight: Tensor<B, 2>,
        ln_bias: Tensor<B, 2>,
        epsilon: f32,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
        let (output, weight_out, bias_out) = B::fused_ttt_forward(
            xq.into_primitive().tensor(),
            xk.into_primitive().tensor(),
            xv.into_primitive().tensor(),
            weight.into_primitive().tensor(),
            bias.into_primitive().tensor(),
            token_eta.into_primitive().tensor(),
            ttt_lr_eta.into_primitive().tensor(),
            ln_weight.into_primitive().tensor(),
            ln_bias.into_primitive().tensor(),
            epsilon,
        );

        (
            Tensor::from_primitive(TensorPrimitive::Float(output)),
            Tensor::from_primitive(TensorPrimitive::Float(weight_out)),
            Tensor::from_primitive(TensorPrimitive::Float(bias_out)),
        )
    }
}

mod fusion_helpers {
    use burn::backend::ir::{InitOperationIr, OperationIr};
    use burn_backend::{TensorMetadata, tensor::FloatTensor};
    use burn_fusion::{
        Fusion, FusionBackend, NoOp, client::GlobalFusionClient, stream::OperationStreams,
    };

    /// Unwraps a Fusion tensor, forces execution of the graph up to this point,
    /// and returns the inner backend tensor.
    pub fn fusion_in<B: FusionBackend>(tensor: FloatTensor<Fusion<B>>) -> FloatTensor<B> {
        tensor.client.clone().resolve_tensor_float::<B>(tensor)
    }

    /// Wraps an inner backend tensor back into a Fusion tensor as a new leaf in the graph.
    pub fn fusion_out<B: FusionBackend>(
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

        assert_eq!(
            new.len(),
            1,
            "Fusion client returned {} tensors when exactly one was expected",
            new.len()
        );

        new.pop().unwrap()
    }
}

impl<B: FusedTttBackend + FusionBackend> FusedTttBackend for burn_fusion::Fusion<B> {
    fn fused_ttt_forward(
        xq: FloatTensor<Self>,
        xk: FloatTensor<Self>,
        xv: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        token_eta: FloatTensor<Self>,
        ttt_lr_eta: FloatTensor<Self>,
        ln_weight: FloatTensor<Self>,
        ln_bias: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        use fusion_helpers::{fusion_in, fusion_out};

        // We assume all tensors share the same client.
        // Checking may be good, but I'm lazy.
        // I'm pretty sure it's always the same client.
        let client = xq.client.clone();

        let (out, weight_out, bias_out) = B::fused_ttt_forward(
            fusion_in::<B>(xq),
            fusion_in::<B>(xk),
            fusion_in::<B>(xv),
            fusion_in::<B>(weight),
            fusion_in::<B>(bias),
            fusion_in::<B>(token_eta),
            fusion_in::<B>(ttt_lr_eta),
            fusion_in::<B>(ln_weight),
            fusion_in::<B>(ln_bias),
            epsilon,
        );

        (
            fusion_out::<B>(out, &client),
            fusion_out::<B>(weight_out, &client),
            fusion_out::<B>(bias_out, &client),
        )
    }

    fn fused_ttt_backward(
        xq: FloatTensor<Self>,
        xk: FloatTensor<Self>,
        xv: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        token_eta: FloatTensor<Self>,
        ttt_lr_eta: FloatTensor<Self>,
        ln_weight: FloatTensor<Self>,
        ln_bias: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        epsilon: f32,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        use fusion_helpers::{fusion_in, fusion_out};

        let client = xq.client.clone();

        let (
            grad_xq,
            grad_xk,
            grad_xv,
            grad_weight,
            grad_bias,
            grad_ttt_lr_eta,
            grad_ln_weight,
            grad_ln_bias,
        ) = B::fused_ttt_backward(
            fusion_in::<B>(xq),
            fusion_in::<B>(xk),
            fusion_in::<B>(xv),
            fusion_in::<B>(weight),
            fusion_in::<B>(bias),
            fusion_in::<B>(token_eta),
            fusion_in::<B>(ttt_lr_eta),
            fusion_in::<B>(ln_weight),
            fusion_in::<B>(ln_bias),
            fusion_in::<B>(grad_output),
            epsilon,
        );

        (
            fusion_out::<B>(grad_xq, &client),
            fusion_out::<B>(grad_xk, &client),
            fusion_out::<B>(grad_xv, &client),
            fusion_out::<B>(grad_weight, &client),
            fusion_out::<B>(grad_bias, &client),
            fusion_out::<B>(grad_ttt_lr_eta, &client),
            fusion_out::<B>(grad_ln_weight, &client),
            fusion_out::<B>(grad_ln_bias, &client),
        )
    }
}
