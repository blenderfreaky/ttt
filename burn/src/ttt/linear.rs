use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::Tensor,
};

use super::{
    layer::{TTTConfig, TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

#[derive(Module, Debug)]
pub struct TTTLinear<B: Backend> {
    /// [num_heads, value_size, value_size]
    weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, value_size]
    bias_init: Param<Tensor<B, 2>>,
    layer_norm: MultiHeadLayerNorm<B>,
    config: Ignored<Arc<TTTConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTLinearState<B: Backend> {
    /// [batch_size, num_heads, value_size, value_size]
    weight: Tensor<B, 4>,
    /// [batch_size, num_heads, value_size]
    bias: Tensor<B, 3>,
    /// [batch_size, num_heads, value_size, value_size]
    weight_grad: Tensor<B, 4>,
    /// [batch_size, num_heads, value_size]
    bias_grad: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTLinearConfig {
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl<B: Backend> TTTInnerModel<B> for TTTLinear<B> {
    type Config = TTTLinearConfig;
    type State = TTTLinearState<B>;

    fn new(global_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Self {
            weight_init: config.initializer.init(
                [
                    global_config.num_heads,
                    global_config.value_size,
                    global_config.value_size,
                ],
                device,
            ),
            bias_init: config
                .initializer
                .init([global_config.num_heads, global_config.value_size], device),
            layer_norm: MultiHeadLayerNormConfig::new(
                global_config.num_heads,
                global_config.value_size,
            )
            .with_initializer(config.initializer.clone())
            .with_epsilon(global_config.epsilon)
            .init(device),
            config: Ignored(global_config.clone()),
        }
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        let weight = self.weight_init.val().unsqueeze().repeat_dim(0, batch_size);
        let bias = self.bias_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTLinearState {
            weight_grad: weight.zeros_like(),
            bias_grad: bias.zeros_like(),
            weight,
            bias,
        }
    }

    // x + LayerNorm(Linear(x))
    //
    // TODO:
    fn forward(&self, state: &mut TTTLinearState<B>, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let qkv = inputs.qkv;

        let [_batch_size, num_heads, seq_len, value_size] = qkv.xv.shape().dims();
        debug_assert_eq!(value_size, self.config.value_size);
        debug_assert_eq!(num_heads, self.config.num_heads);

        // No prefill for now
        debug_assert_eq!(seq_len, 1);

        let z = qkv.xq.clone().matmul(state.weight.clone().unsqueeze())
            + state.bias.clone().unsqueeze();

        // Since we're using a residual connection, our target is the residual
        let target = qkv.xv - qkv.xk.clone();

        let (_ln_out, dl_dz) = self.layer_norm.forward_and_grad(z, target);

        let step = inputs.lr.unsqueeze_dim(3) * dl_dz;

        let weight_grad = qkv.xk.swap_dims(2, 3).matmul(step.clone());
        let bias_grad = step.squeeze(2);

        // It seems we only accumulate gradients, and don't update the weights per-step, even outside of prefill
        state.weight_grad.inplace(|x| x.add(weight_grad.clone()));
        state.bias_grad.inplace(|x| x.add(bias_grad.clone()));

        let weight_new = state.weight.clone() - inputs.token_idx.clone().unsqueeze() * weight_grad;
        let bias_new =
            state.bias.clone() - (inputs.token_idx.clone().unsqueeze() * bias_grad).squeeze(2);

        if (inputs.start_idx + 1) % self.config.mini_batch_size == 0 {
            state.weight = weight_new.clone();
            state.bias = bias_new.clone();
            // TODO: This is rather hacky
            state.weight_grad = state.weight_grad.zeros_like();
            state.bias_grad = state.bias_grad.zeros_like();
        }

        // Recalculate after the backprop step

        // TODO: Reference implementation does layernorm outside, I think we should do in here rather

        qkv.xq.matmul(weight_new.unsqueeze()) + bias_new.unsqueeze()
    }
}
