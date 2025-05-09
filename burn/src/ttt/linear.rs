use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::Tensor,
};

use super::{
    layer::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
    TTTConfig,
};

#[derive(Module, Debug)]
pub struct TTTLinear<B: Backend> {
    /// [num_heads, head_dim, head_dim]
    weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, head_dim]
    bias_init: Param<Tensor<B, 2>>,
    layer_norm: MultiHeadLayerNorm<B>,
    config: Ignored<Arc<TTTConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTLinearState<B: Backend> {
    /// [batch_size, num_heads, head_dim, head_dim]
    weight: Tensor<B, 4>,
    /// [batch_size, num_heads, head_dim]
    bias: Tensor<B, 3>,
    /// [batch_size, num_heads, head_dim, head_dim]
    weight_grad: Tensor<B, 4>,
    /// [batch_size, num_heads, head_dim]
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
        let len = global_config.hidden_size;
        Self {
            weight_init: config.initializer.init_with(
                [
                    global_config.num_heads,
                    global_config.head_dim(),
                    global_config.head_dim(),
                ],
                Some(len),
                Some(len),
                device,
            ),
            bias_init: config.initializer.init_with(
                [global_config.num_heads, global_config.head_dim()],
                Some(len),
                Some(len),
                device,
            ),
            layer_norm: MultiHeadLayerNormConfig::new(
                global_config.num_heads,
                global_config.head_dim(),
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
    fn forward_one(
        &self,
        state: &mut TTTLinearState<B>,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let qkv = inputs.qkv;

        let [_batch_size, num_heads, seq_len, head_dim] = qkv.xv.shape().dims();
        debug_assert_eq!(head_dim, self.config.head_dim());
        debug_assert_eq!(num_heads, self.config.num_heads);

        assert_eq!(seq_len, 1);

        let z = qkv.xq.clone().matmul(state.weight.clone()) + state.bias.clone().unsqueeze_dim(2);

        // Since we're using a residual connection, our target is the difference
        let target = qkv.xv - qkv.xk.clone();

        dbg!(z.shape());
        dbg!(target.shape());
        let (_ln_out, dl_dz) = self.layer_norm.forward_and_grad(z, target);

        let step = inputs.lr.unsqueeze_dim(3) * dl_dz;

        dbg!(step.shape());
        dbg!(qkv.xk.shape());

        let weight_grad = qkv.xk.swap_dims(2, 3).matmul(step.clone());
        let bias_grad = step.squeeze(2);

        // It seems we only accumulate gradients, and don't update the weights per-step, even outside of prefill
        state.weight_grad.inplace(|x| x.add(weight_grad.clone()));
        state.bias_grad.inplace(|x| x.add(bias_grad.clone()));

        let weight_new = state.weight.clone() - inputs.token_idx.clone().unsqueeze() * weight_grad;
        let bias_new =
            state.bias.clone() - (inputs.token_idx.clone().unsqueeze() * bias_grad);

        dbg!(state.weight.shape());
        dbg!(state.bias.shape());
        dbg!(weight_new.shape());
        dbg!(bias_new.shape());

        if (inputs.start_idx + 1) % self.config.mini_batch_size == 0 {
            state.weight = weight_new.clone();
            state.bias = bias_new.clone();
            // TODO: This is rather hacky
            state.weight_grad = state.weight_grad.zeros_like();
            state.bias_grad = state.bias_grad.zeros_like();
        }

        // Recalculate after the backprop step

        // TODO: Reference implementation does layernorm outside, I think we should do in here rather

        qkv.xq.matmul(weight_new) + bias_new.unsqueeze_dim(2)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let qkv = inputs.qkv;

        let [batch_size, num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        debug_assert_eq!(seq_len, self.config.mini_batch_size);

        let z = qkv.xk.clone().matmul(state.weight.clone().unsqueeze())
            + state.bias.clone().unsqueeze();

        let l2_target = qkv.xv.clone() - qkv.xk.clone();

        let dl_dz = todo!();

        todo!();
    }
}
