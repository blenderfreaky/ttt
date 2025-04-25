use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::Tensor,
};

use super::layer::{TTTConfig, TTTInnerModel, TTTInputsInner};

#[derive(Module, Debug)]
pub struct TTTLinear<B: Backend> {
    /// [num_heads, value_size, value_size]
    weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, value_size]
    bias_init: Param<Tensor<B, 2>>,
    /// [num_heads, value_size]
    norm_weight: Param<Tensor<B, 2>>,
    /// [num_heads, value_size]
    norm_bias: Param<Tensor<B, 2>>,
    config: Ignored<Arc<TTTConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTLinearState<B: Backend> {
    /// [batch_size, num_heads, value_size, value_size]
    weight: Tensor<B, 4>,
    /// [batch_size, num_heads, value_size]
    bias: Tensor<B, 3>,
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
            // layer: LinearConfig::new(global_config.value_size, global_config.value_size)
            //     .with_bias(true)
            //     .init(device),
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
            // layer_norm: LayerNormConfig::new(global_config.num_heads * global_config.value_size)
            //     .with_epsilon(global_config.epsilon)
            //     .init(device),
            norm_weight: config
                .initializer
                .init([global_config.num_heads, global_config.value_size], device),
            norm_bias: config
                .initializer
                .init([global_config.num_heads, global_config.value_size], device),
            config: Ignored(global_config.clone()),
        }
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        TTTLinearState {
            weight: self.weight_init.val().unsqueeze().repeat_dim(0, batch_size),
            bias: self.bias_init.val().unsqueeze().repeat_dim(0, batch_size),
        }
    }

    // x + LayerNorm(Linear(x))
    //
    // TODO: Pass current weights (don't use weight_init)
    fn forward(&self, state: &mut TTTLinearState<B>, inputs: TTTInputsInner<B>) -> Tensor<B, 3> {
        let qkv = inputs.qkv;

        let [batch_times_heads, seq_len, value_size] = inputs.qkv.xv.shape().dims();
        debug_assert_eq!(value_size, self.config.value_size);
        debug_assert_eq!(batch_times_heads % self.config.num_heads, 0);

        let batch_size = batch_times_heads / self.config.num_heads;

        let z = qkv.xq.matmul(state.weight.unsqueeze()) + state.bias.unsqueeze();

        let (var, mean) = z.var_mean_bias(2);
        let std = (var + self.config.epsilon).sqrt();

        let norm = (z - mean) / std;

        let out = self.norm_weight.val().unsqueeze_dims(&[0, 1])
            * norm.reshape([batch_size, self.config.num_heads, seq_len, value_size])
            + self.norm_bias.val().unsqueeze_dims(&[0, 1]);

        // Since we're using a residual connection, our target is the residual
        let target = qkv.xv - qkv.xk;

        let dl_dout =
            out - target.reshape([batch_size, self.config.num_heads, seq_len, value_size]);

        let dl_dnorm = (dl_dout * self.norm_weight.val().unsqueeze_dims(&[0, 1])).reshape([
            batch_times_heads,
            seq_len,
            value_size,
        ]);

        let dl_dz_term1 = dl_dnorm * (value_size as f32);
        let dl_dz_term2 = dl_dnorm.sum_dim(2).unsqueeze_dim(2);
        let dl_dz_term3 = norm * (dl_dnorm * norm).sum_dim(2).unsqueeze_dim(2);

        let dl_dz = (dl_dz_term1 - dl_dz_term2 - dl_dz_term3) / (std * (value_size as f32));

        let step = inputs.lr * dl_dz;

        let weight_grad = qkv.xk.swap_dims(2, 3).matmul(step);
        let bias_grad = step;

        let weight_new = state.weight - inputs.token_idx.unsqueeze() * weight_grad;
        let bias_new = state.bias - inputs.token_idx.unsqueeze() * bias_grad;

        // Recalculate after the backprop step
        let z_new = qkv.xq.matmul(weight_new.unsqueeze()) + bias_new.unsqueeze();

        // No layernorm after in reference!?

        z_new
    }
}
