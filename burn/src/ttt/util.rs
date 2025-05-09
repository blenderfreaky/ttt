use burn::{
    backend::Wgpu,
    config::Config,
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{activation::silu, module::conv1d, ops::ConvOptions, Tensor},
};

/// Applies causal convolution with no initial states
// Returns [batch_size, dim, seq_len]
pub fn causal_conv1d_fn<B: Backend>(
    // [batch_size, dim, seq_len]
    x: Tensor<B, 3>,
    //// [dim, 1, width]
    // [channels_out, channels_in/dim, kernel_size]
    weight: Tensor<B, 3>,
    //// [dim,]
    // [channels_out]
    bias: Option<Tensor<B, 1>>,
    // // (batch, dim, width - 1)
    // initial_states: Option<Tensor<B, 3>>,
    // // (batch, dim, width - 1)
    // final_states_out: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [_batch_size, dim, seq_len] = x.shape().dims();
    let [_channels_out, _channels_in_per_group, kernel_size] = weight.shape().dims();
    // debug_assert_eq!(channels_out, dim);
    // debug_assert_eq!(channels_in_per_group, 1);

    // if let Some(initial_states) = initial_states {
    //     x = Tensor::cat(vec![initial_states, x], 2);

    let out = conv1d(
        x,
        weight,
        bias,
        // Why is groups = dim? This seems inefficient. (Taken from the original implementation)
        ConvOptions::new([1], [kernel_size - 1], [1], dim),
    );

    out.slice([0..dim, 0..dim, 0..seq_len])
}

#[derive(Config, Debug)]
pub struct CausalConvConfig {
    // Used for cache indexing in src
    // layer_idx: usize,
    hidden_size: usize,
    kernel_size: usize,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct CausalConv<B: Backend> {
    weight: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
}

impl CausalConvConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> CausalConv<B> {
        let weight = self.initializer.init_with(
            [self.hidden_size, 1, self.kernel_size],
            Some(self.kernel_size),
            None,
            device,
        );
        let bias =
            self.initializer
                .init_with([self.hidden_size], Some(self.kernel_size), None, device);
        CausalConv { weight, bias }
    }
}

impl<B: Backend> CausalConv<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [channels_out, channels_in_per_group, kernel_size] = self.weight.shape().dims();
        // debug_assert_eq!(channels_out, kernel_size);
        debug_assert_eq!(channels_in_per_group, 1);

        causal_conv1d_fn(x, self.weight.val(), Some(self.bias.val()))
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadLayerNormConfig {
    num_heads: usize,
    head_dim: usize,
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
    weight: Param<Tensor<B, 2>>,
    /// [num_heads, value_size]
    bias: Param<Tensor<B, 2>>,
    epsilon: f64,
}

impl MultiHeadLayerNormConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MultiHeadLayerNorm<B> {
        let len = self.num_heads * self.head_dim;
        let weight = self.initializer.init_with(
            [self.num_heads, self.head_dim],
            Some(len),
            Some(len),
            device,
        );
        let bias = self.initializer.init_with(
            [self.num_heads, self.head_dim],
            Some(len),
            Some(len),
            device,
        );
        MultiHeadLayerNorm {
            weight,
            bias,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> MultiHeadLayerNorm<B> {
    fn weight_and_bias(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let w = self.weight.val();
        let b = self.bias.val();
        let [num_heads, head_dim] = w.shape().dims();
        debug_assert_eq!([num_heads, head_dim], b.shape().dims());
        (
            w.reshape([1, num_heads, 1, head_dim]),
            b.reshape([1, num_heads, 1, head_dim]),
        )
    }

    /// # Parameters
    /// - `x`: Input tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (var, mean) = x.clone().var_mean_bias(2);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        let (weight, bias) = self.weight_and_bias();

        weight * norm.clone() + bias
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

        let (var, mean) = x.clone().var_mean_bias(3);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        let (weight, bias) = self.weight_and_bias();

        let out = weight.clone() * norm.clone() + bias;

        let dl_dout = out.clone() - target;

        let dl_dnorm = dl_dout * weight;

        let dl_dx_term1 = dl_dnorm.clone() * (value_size as f32);
        let dl_dx_term2 = dl_dnorm.clone().sum_dim(3);
        let dl_dx_term3 = norm.clone() * (dl_dnorm * norm.clone()).sum_dim(3);

        let dl_dx = (dl_dx_term1 - dl_dx_term2 - dl_dx_term3) / (std * (value_size as f32));

        (out, dl_dx)
    }
}

// burn-rs's SwiGlu is not quite the same,
// theirs does
//  silu(linear(input)) * linear(input)
// this one does wraps the entire expression with another linear layer,
// and merges the other two linear layers into one projection that gets split

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    up_gate_proj: Linear<B>,
    down_proj: Linear<B>,
    intermediate_size: usize,
}

#[derive(Config, Debug)]
pub struct SwiGluMlpConfig {
    hidden_size: usize,
    intermediate_size: usize,
}

impl SwiGluMlpConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> SwiGluMlp<B> {
        SwiGluMlp {
            up_gate_proj: LinearConfig::new(self.hidden_size, 2 * self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            intermediate_size: self.intermediate_size,
        }
    }
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [gate, up] = self
            .up_gate_proj
            .forward(x)
            .split(self.intermediate_size, 2)
            .try_into()
            .unwrap();

        self.down_proj.forward(silu(gate) * up)
    }
}
