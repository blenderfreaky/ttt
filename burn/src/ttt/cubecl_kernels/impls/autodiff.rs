use std::fmt::Debug;

use burn::{
    backend::autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{FloatDType, ops::FloatTensor},
};
use burn_backend::{Shape, TensorMetadata};

use crate::ttt::cubecl_kernels::{
    bundle::TensorBundle,
    kernel::{BackwardImpl, FusedKernel, FusedKernelBackend},
};

/// Backward op for a specific output index.
/// Each output is tracked separately and runs the backward kernel independently.
/// Gradients to inputs accumulate via burn's gradient system.
#[derive(Debug)]
struct OutputBackwardOp<K, const N: usize, const M: usize> {
    output_idx: usize,
    _marker: std::marker::PhantomData<K>,
}

impl<K, const N: usize, const M: usize, B> Backward<B, N> for OutputBackwardOp<K, N, M>
where
    K: FusedKernel<N, M>,
    B: FusedKernelBackend<K, N, M>,
    B::FloatTensorPrimitive: Send,
{
    type State = (
        K::Inputs<B::FloatTensorPrimitive>,
        Option<K::Outputs<B::FloatTensorPrimitive>>, // Saved outputs (if needed)
        [(Vec<usize>, FloatDType); M],               // Output shapes for creating zeros
        [Option<NodeId>; N],                         // Input node IDs for gradient registration
    );

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<B>(&ops.node);
        let (saved_inputs, saved_outputs, output_shapes, input_node_ids) = ops.state;

        // Get device from one of the saved inputs
        let inputs_arr = saved_inputs.into_array();
        let device = B::float_device(&inputs_arr[0]);

        // Build grad_outputs with this gradient at the appropriate index, zeros elsewhere
        let grad_outputs_arr: [B::FloatTensorPrimitive; M] = std::array::from_fn(|i| {
            if i == self.output_idx {
                grad_output.clone()
            } else {
                B::float_zeros(
                    Shape::from(output_shapes[i].0.clone()),
                    &device,
                    output_shapes[i].1,
                )
            }
        });
        let grad_outputs = K::Outputs::from_array(grad_outputs_arr);

        // Reconstruct saved_inputs from the array
        let saved_inputs = K::Inputs::from_array(inputs_arr);

        let grad_inputs = B::backward(saved_inputs, saved_outputs, grad_outputs);

        // Register gradients for all tracked parents (accumulates if called multiple times)
        for (grad, node_id) in grad_inputs
            .into_array()
            .into_iter()
            .zip(input_node_ids.iter())
        {
            if let Some(id) = node_id {
                grads.register::<B>(*id, grad);
            }
        }
    }
}

impl<K, const N: usize, const M: usize, B, C> FusedKernelBackend<K, N, M> for Autodiff<B, C>
where
    K: FusedKernel<N, M>,
    B: FusedKernelBackend<K, N, M>,
    C: CheckpointStrategy,
    B::FloatTensorPrimitive: Clone,
{
    fn forward(inputs: K::Inputs<FloatTensor<Self>>) -> K::Outputs<FloatTensor<Self>> {
        let input_arr = inputs.into_array();
        let nodes_array: [_; N] = input_arr.each_ref().map(|t| t.node.clone());
        let input_node_ids: [Option<NodeId>; N] = nodes_array.each_ref().map(|n| Some(n.id));
        let primitives: [_; N] = input_arr.map(|t| t.primitive.clone());

        let inner_inputs = K::Inputs::from_array(primitives.clone());
        let outputs = B::forward(inner_inputs);
        let output_primitives: [_; M] = outputs.into_array();

        // Save output shapes for creating zeros in backward
        let output_shapes: [_; M] = std::array::from_fn(|i| {
            (
                output_primitives[i].shape().dims.clone(),
                output_primitives[i].dtype().into(),
            )
        });

        // Save inputs for backward (shared by all outputs)
        let saved_inputs = K::Inputs::from_array(primitives);

        // Optionally save outputs for backward (based on kernel's backward requirements)
        let saved_outputs = if K::Backward::should_save_outputs() {
            Some(K::Outputs::from_array(output_primitives.clone()))
        } else {
            None
        };

        // Track each output separately
        let tracked_outputs: [FloatTensor<Self>; M] = std::array::from_fn(|idx| {
            let backward_op = OutputBackwardOp::<K, N, M> {
                output_idx: idx,
                _marker: std::marker::PhantomData,
            };

            match backward_op
                .prepare::<C>(nodes_array.clone())
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => prep.finish(
                    (
                        saved_inputs.clone(),
                        saved_outputs.clone(),
                        output_shapes.clone(),
                        input_node_ids,
                    ),
                    output_primitives[idx].clone(),
                ),
                OpsKind::UnTracked(prep) => prep.finish(output_primitives[idx].clone()),
            }
        });

        K::Outputs::from_array(tracked_outputs)
    }

    fn backward(
        _inputs: K::Inputs<FloatTensor<Self>>,
        _outputs: Option<K::Outputs<FloatTensor<Self>>>,
        _grad_outputs: K::Outputs<FloatTensor<Self>>,
    ) -> K::Inputs<FloatTensor<Self>> {
        panic!("Second-order gradients not supported")
    }
}
