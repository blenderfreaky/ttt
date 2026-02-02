use std::{
    fmt::Debug,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

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

use crate::{
    TensorBundle,
    kernel::{FusedKernel, FusedKernelBackend},
};

/// No-op backward used to wrap primitives into untracked AutodiffTensors.
/// With 0 parents, `prepare([])` always yields `UnTracked`, allowing us to
/// convert inner primitives to autodiff tensors without memory allocation.
#[derive(Debug)]
struct NoOpBackward;

impl<B: burn_backend::Backend> Backward<B, 0> for NoOpBackward {
    type State = ();

    fn backward(self, _ops: Ops<(), 0>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do - this is never called since we're always untracked
    }
}

/// Shared state for backward ops across all outputs of a fused kernel.
/// This allows us to run the backward kernel only once, collecting all
/// output gradients before computing input gradients.
struct SharedBackwardState<B: burn_backend::Backend, const N: usize, const M: usize, const S: usize>
{
    saved_state: [B::FloatTensorPrimitive; S],
    output_shapes: [(Vec<usize>, FloatDType); M],
    input_node_ids: [Option<NodeId>; N],

    /// Bitmask of which outputs are tracked (will receive backward calls)
    outputs_tracked: AtomicUsize,
    /// Bitmask of which outputs have had backward called
    backwards_called: AtomicUsize,
    /// Collected grad_outputs from each output's backward call
    grad_outputs: Mutex<[Option<B::FloatTensorPrimitive>; M]>,
}

impl<B: burn_backend::Backend, const N: usize, const M: usize, const S: usize> Debug
    for SharedBackwardState<B, N, M, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBackwardState")
            .field("outputs_tracked", &self.outputs_tracked)
            .field("backwards_called", &self.backwards_called)
            .finish_non_exhaustive()
    }
}

/// Backward op that shares state across all outputs.
/// Only runs the backward kernel once when all tracked outputs have contributed gradients.
#[derive(Debug)]
struct SharedOutputBackwardOp<K, const N: usize, const M: usize, const S: usize> {
    output_idx: usize,
    _marker: std::marker::PhantomData<K>,
}

impl<K, const N: usize, const M: usize, const S: usize, B> Backward<B, N>
    for SharedOutputBackwardOp<K, N, M, S>
where
    K: FusedKernel,
    B: FusedKernelBackend<K>,
    B::FloatTensorPrimitive: Send + Clone,
    K::SavedState<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; S]>,
    K::Outputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; M]>,
    K::Inputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; N]>,
{
    type State = (Arc<SharedBackwardState<B, N, M, S>>, K::Config);

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<B>(&ops.node);
        let (shared, config) = ops.state;

        // Store this output's gradient
        {
            let mut grad_outputs = shared.grad_outputs.lock().unwrap();
            grad_outputs[self.output_idx] = Some(grad_output);
        }

        // Mark this output as processed and check if we're the last one
        let old_called = shared
            .backwards_called
            .fetch_or(1 << self.output_idx, Ordering::AcqRel);
        let new_called = old_called | (1 << self.output_idx);
        let tracked = shared.outputs_tracked.load(Ordering::Acquire);

        if new_called == tracked {
            // All tracked outputs have contributed their gradients.
            // Run backward kernel once with the combined gradients.
            let grad_outputs_lock = shared.grad_outputs.lock().unwrap();

            // Get device from saved state
            let device = B::float_device(&shared.saved_state[0]);

            let grad_outputs_arr: [B::FloatTensorPrimitive; M] = std::array::from_fn(|i| {
                grad_outputs_lock[i].clone().unwrap_or_else(|| {
                    B::float_zeros(
                        Shape::from(shared.output_shapes[i].0.clone()),
                        &device,
                        shared.output_shapes[i].1,
                    )
                })
            });
            drop(grad_outputs_lock);

            let grad_outputs_bundle = K::Outputs::from_array(grad_outputs_arr);
            let saved_state = K::SavedState::from_array(shared.saved_state.clone());

            let grad_inputs = B::backward(saved_state, grad_outputs_bundle, config);

            // Register gradients for all tracked parents
            for (grad, node_id) in grad_inputs
                .into_array()
                .into_iter()
                .zip(shared.input_node_ids.iter())
            {
                if let Some(id) = node_id {
                    grads.register::<B>(*id, grad);
                }
            }
        }
        // Else: still waiting for other outputs to contribute their gradients.
        // The last output to call backward will register all input gradients.
    }
}

impl<K, B, C, const N: usize, const M: usize, const S: usize> FusedKernelBackend<K>
    for Autodiff<B, C>
where
    K: FusedKernel,
    B: FusedKernelBackend<K>,
    C: CheckpointStrategy,
    B::FloatTensorPrimitive: Clone,
    // Extract N, M, S from the Array associated types
    K::Inputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; N]>,
    K::Outputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; M]>,
    K::SavedState<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; S]>,
    K::Inputs<FloatTensor<Self>>:
        TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; N]>,
    K::Outputs<FloatTensor<Self>>:
        TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; M]>,
    K::SavedState<FloatTensor<Self>>:
        TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; S]>,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (
        K::Outputs<FloatTensor<Self>>,
        K::SavedState<FloatTensor<Self>>,
    ) {
        let input_arr = inputs.into_array();
        let nodes_array: [_; N] = input_arr.each_ref().map(|t| t.node.clone());
        let input_node_ids: [Option<NodeId>; N] = nodes_array.each_ref().map(|n| Some(n.id));
        let primitives: [_; N] = input_arr.map(|t| t.primitive.clone());

        let inner_inputs = K::Inputs::from_array(primitives);
        let (outputs, saved_state) = B::forward(inner_inputs, config.clone());
        let output_primitives: [_; M] = outputs.into_array();
        let saved_primitives: [_; S] = saved_state.into_array();

        // Save output shapes for creating zeros in backward
        let output_shapes: [_; M] = std::array::from_fn(|i| {
            (
                output_primitives[i].shape().dims.clone(),
                output_primitives[i].dtype().into(),
            )
        });

        // Create shared state for all output backward ops
        let shared_state = Arc::new(SharedBackwardState::<B, N, M, S> {
            saved_state: saved_primitives.clone(),
            output_shapes,
            input_node_ids,
            outputs_tracked: AtomicUsize::new(0),
            backwards_called: AtomicUsize::new(0),
            grad_outputs: Mutex::new(std::array::from_fn(|_| None)),
        });

        // Track each output, sharing the backward state
        let tracked_outputs: [FloatTensor<Self>; M] = std::array::from_fn(|idx| {
            let backward_op = SharedOutputBackwardOp::<K, N, M, S> {
                output_idx: idx,
                _marker: std::marker::PhantomData,
            };

            match backward_op
                .prepare::<C>(nodes_array.clone())
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    // Mark this output as tracked (will receive backward call)
                    shared_state
                        .outputs_tracked
                        .fetch_or(1 << idx, Ordering::Release);
                    prep.finish(
                        (shared_state.clone(), config.clone()),
                        output_primitives[idx].clone(),
                    )
                }
                OpsKind::UnTracked(prep) => prep.finish(output_primitives[idx].clone()),
            }
        });

        // Wrap saved state primitives as untracked AutodiffTensors.
        let wrapped_saved: [FloatTensor<Self>; S] = std::array::from_fn(|i| {
            match NoOpBackward.prepare::<C>([]).compute_bound().stateful() {
                OpsKind::UnTracked(prep) => prep.finish(saved_primitives[i].clone()),
                OpsKind::Tracked(_) => unreachable!("0 parents always yields UnTracked"),
            }
        });

        (
            K::Outputs::from_array(tracked_outputs),
            K::SavedState::from_array(wrapped_saved),
        )
    }

    fn backward(
        _saved: K::SavedState<FloatTensor<Self>>,
        _grad_outputs: K::Outputs<FloatTensor<Self>>,
        _config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>> {
        panic!("Second-order gradients not supported")
    }
}
