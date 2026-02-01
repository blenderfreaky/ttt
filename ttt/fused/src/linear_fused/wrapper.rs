use std::{ops::Range, sync::Arc};

use burn::tensor::Tensor;
use ttt_core::{TTTConfig, TTTInnerModel, TTTInputsInner, TTTLinear};

use crate::{Fused, FusedTttBackend, LinearKernel, fused_ttt_forward};

impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, LinearKernel> {
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
        Fused::new(TTTLinear::new(general_config, config, device))
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
    use ttt_core::{GpuAutodiffBackend, GpuBackend, TTTLinearState};
    use ttt_kernels::test_utils::{TestDims, test_backward_fmb, test_fmb};

    use crate::FusedLinear;

    #[test]
    fn test_fused_ttt_linear_vs_reference() {
        let dims = TestDims::new(2, 4, 16, 8);
        test_fmb::<GpuBackend, FusedLinear<GpuBackend>, TTTLinearState<GpuBackend>, _>(
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
        test_backward_fmb::<GpuAutodiffBackend, FusedLinear<GpuAutodiffBackend>, _>(
            dims,
            |m| m.into(),
            2e-2,
            1e-3,
            "Fused",
        );
    }
}
