use burn::{
    config::Config,
    module::{Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{module::conv1d, ops::ConvOptions, Tensor},
};

/// Applies causal convolution with no initial states
// Returns (batch_size, dim, seq_len)
pub fn causal_conv1d_fn<B: Backend>(
    // [batch_size, dim, seq_len]
    x: Tensor<B, 3>,
    // [dim, 1, width]
    weight: Tensor<B, 3>,
    // [dim,]
    bias: Option<Tensor<B, 1>>,
    // // (batch, dim, width - 1)
    // initial_states: Option<Tensor<B, 3>>,
    // // (batch, dim, width - 1)
    // final_states_out: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [_batch_size, dim, seq_len] = x.shape().dims();
    let [channels_out, channels_in_per_group, width] = weight.shape().dims();
    debug_assert_eq!(channels_out, dim);
    debug_assert_eq!(channels_in_per_group, 1);

    // if let Some(initial_states) = initial_states {
    //     x = Tensor::cat(vec![initial_states, x], 2);

    let out = conv1d(
        x,
        weight.unsqueeze_dim(1),
        bias,
        // Why is groups = dim? This seems inefficient. (Taken from the original implementation)
        ConvOptions::new([1], [width - 1], [1], dim),
    );

    out.slice([0..dim, 0..seq_len])
}

#[derive(Config, Debug)]
pub struct MultiHeadLayerNormConfig {
    num_heads: usize,
    value_size: usize,
    #[config(default = 1e-6)]
    epsilon: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct MultiHeadLayerNorm<B: Backend> {
    /// [num_heads, value_size]
    norm_weight: Param<Tensor<B, 2>>,
    /// [num_heads, value_size]
    norm_bias: Param<Tensor<B, 2>>,
    epsilon: f64,
}

impl MultiHeadLayerNormConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MultiHeadLayerNorm<B> {
        let norm_weight = self
            .initializer
            .init([self.num_heads, self.value_size], device);
        let norm_bias = self
            .initializer
            .init([self.num_heads, self.value_size], device);
        MultiHeadLayerNorm {
            norm_weight,
            norm_bias,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> MultiHeadLayerNorm<B> {
    /// # Parameters
    /// - `x`: Input tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (var, mean) = x.clone().var_mean_bias(2);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        self.norm_weight.val().unsqueeze_dims(&[0, 1]) * norm.clone()
            + self.norm_bias.val().unsqueeze_dims(&[0, 1])
    }

    /// # Parameters
    /// - `x`: Input tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    /// - `target`: Target tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    pub fn forward_and_grad(
        &self,
        x: Tensor<B, 4>,
        target: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch_size, _num_heads, _seq_len, value_size] = x.shape().dims();

        let (var, mean) = x.clone().var_mean_bias(2);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        let out = self.norm_weight.val().unsqueeze_dims(&[0, 1]) * norm.clone()
            + self.norm_bias.val().unsqueeze_dims(&[0, 1]);

        let dl_dout = out.clone() - target;

        let dl_dnorm = dl_dout * self.norm_weight.val().unsqueeze_dims(&[0, 1]);

        let dl_dx_term1 = dl_dnorm.clone() * (value_size as f32);
        let dl_dx_term2 = dl_dnorm.clone().sum_dim(3).unsqueeze_dim(3);
        let dl_dx_term3 = norm.clone() * (dl_dnorm * norm.clone()).sum_dim(3).unsqueeze_dim(3);

        let dl_dx = (dl_dx_term1 - dl_dx_term2 - dl_dx_term3) / (std * (value_size as f32));

        (out, dl_dx)
    }
}
