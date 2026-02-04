use std::{ops::Range, sync::Arc};

use burn::tensor::Tensor;
use ttt_core::{TTTInnerModel, TTTInputsInner, TTTLinear, config::ModelConfig};

use crate::{Fused, FusedTttBackend, NaiveKernel, fused_ttt_forward};

impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, NaiveKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTTTLinear"
    }

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Fused::new(TTTLinear::new(config, inner_config, device))
    }

    fn get_config(&self) -> &ModelConfig {
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
