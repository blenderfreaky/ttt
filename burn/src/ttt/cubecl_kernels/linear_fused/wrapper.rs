use std::{marker::PhantomData, ops::Range, sync::Arc};

use burn::tensor::Tensor;

use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{
        Fused,
        backend::{FusedTttBackend, api::fused_ttt_forward},
    },
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};

impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused {
            inner: TTTLinear::new(general_config, config, device),
            _backend: PhantomData,
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let (output, weight_updated, bias_updated) = fused_ttt_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            epsilon,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttt::{
        GpuAutodiffBackend, GpuBackend,
        cubecl_kernels::test_utils::{TestDims, test_backward_fmb, test_fmb},
        linear::TTTLinear,
    };

    #[test]
    fn test_fused_ttt_linear_vs_reference() {
        let dims = TestDims::new(2, 4, 16, 8);
        test_fmb::<GpuBackend, Fused<GpuBackend, TTTLinear<GpuBackend>>, _>(
            dims,
            |m| m.into(),
            1e-3,
            1e-4,
            "Fused",
        );
    }

    // TODO: investigate
    #[test]
    #[ignore]
    fn test_fused_backward_gradients_vs_reference() {
        let dims = TestDims::new(2, 2, 8, 4);
        test_backward_fmb::<
            GpuAutodiffBackend,
            Fused<GpuAutodiffBackend, TTTLinear<GpuAutodiffBackend>>,
            _,
        >(dims, |m| m.into(), 2e-2, 1e-3, "Fused");
    }
}
