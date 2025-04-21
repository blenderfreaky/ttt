use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig, RotaryEncoding,
        RotaryEncodingConfig,
    },
    prelude::*,
    tensor::{
        activation::{gelu, sigmoid},
        module::conv1d,
        ops::ConvOptions,
        Tensor,
    },
};

/// Configuration for the TTT layer.
#[derive(Config, Debug)]
struct TTTConfig {
    /// The size of token vectors.
    token_size: usize,
    // /// The size of key and query vectors.
    // /// In source it seems to be token_size/2
    // key_size: usize,
    /// The size of value vectors.
    /// In source it seems to be token_size
    value_size: usize,
    /// The number of TTT heads.
    num_heads: usize,
    /// The kernel size for the convolutional layers.
    conv_kernel_size: usize,
    /// The mini batch size.
    mini_batch_size: usize,
    // TODO: Make positional encoding generic/exchangable for different types of positional encodings
    /// The maximum sequence length (only used for rotary encoding).
    max_sequence_len: usize,
    /// The theta value for the rotary encoding.
    rope_theta: f32,
    /// The base learning rate for the TTT module.
    base_lr: f32,
    epsilon: f64,
}

#[derive(Module, Debug)]
struct TTT<B: Backend> {
    qkvg_proj: Linear<B>,
    o_proj: Linear<B>,
    q_conv: Conv1d<B>,
    k_conv: Conv1d<B>,
    learning_rate: Linear<B>,
    learnable_token_idx: Tensor<B, 1>,
    post_norm: LayerNorm<B>,
    config: Ignored<Arc<TTTConfig>>,
    // TODO: Make positional encoding generic/exchangable for different types of positional encodings
    rot_enc: RotaryEncoding<B>,
}

pub trait TTTInnerModel<B: Backend> {
    type Config: Config;

    fn new(general_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device)
        -> Self;

    fn forward(&self, inputs: TTTInputsInner<B>) -> Tensor<B, 3>;
}

impl TTTConfig {
    pub fn init<B: Backend>(self: &Arc<Self>, device: &B::Device) -> TTT<B> {
        let linear = |in_size, out_size, bias| {
            LinearConfig::new(in_size, out_size)
                .with_bias(bias)
                .init(device)
        };

        let conv = |size| {
            Conv1dConfig::new(size, size, self.conv_kernel_size)
                .with_groups(size)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(
                    self.conv_kernel_size - 1,
                ))
                .with_bias(true)
                .init(device)
        };

        TTT {
            qkvg_proj: linear(self.token_size, self.value_size * 3 + self.num_heads, false),
            o_proj: linear(self.value_size, self.token_size, false),
            q_conv: conv(self.value_size),
            k_conv: conv(self.value_size),
            // One output per head
            learning_rate: linear(self.token_size, self.num_heads, true),
            // We store token_idx + bias as one, their impl splits it
            learnable_token_idx: Tensor::arange(1..(self.mini_batch_size as i64), device)
                .float()
                .recip(),
            post_norm: LayerNormConfig::new(self.token_size)
                .with_epsilon(1e-6)
                .init(device),
            rot_enc: RotaryEncodingConfig::new(self.max_sequence_len, self.token_size)
                .with_theta(self.rope_theta)
                .init(device),
            config: Ignored(self.clone()),
        }
    }
}

struct QKV<B: Backend> {
    /// [batch_size*num_heads, seq_len, head_dim]
    xq: Tensor<B, 3>,
    /// [batch_size*num_heads, seq_len, head_dim]
    xk: Tensor<B, 3>,
    /// [batch_size*num_heads, seq_len, head_dim]
    xv: Tensor<B, 3>,
}

struct TTTInputsInner<B: Backend> {
    /// The key, query and value vectors
    qkv: QKV<B>,
    /// Offset token index for each token
    /// `[seq_len]`
    token_idx: Tensor<B, 1>,
    /// Learning rate for each head of each token
    /// `[batch_size, num_heads, seq_len]`
    lr: Tensor<B, 3>,
}

// TODO: Better name
struct TTTInputs<B: Backend> {
    inner: TTTInputsInner<B>,
    /// [batch_size, seq_len, num_heads]
    gate: Tensor<B, 3>,
}

impl<B: Backend> TTT<B> {
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    fn get_qkvg(&self, x: Tensor<B, 3>, start_idx: usize) -> (QKV<B>, Tensor<B, 3>) {
        let [batch_size, seq_len, _] = x.shape().dims();

        let proj = self.qkvg_proj.forward(x);
        let [xqk, gate, xv] = proj.split(self.config.value_size, 2).try_into().unwrap();

        let (xq, xk) = self.conv_qk(xqk);

        // [B, seq_len, num_heads*dim] -> [B, num_heads, seq_len, dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([0, 0, self.config.num_heads as i32, -1])
                .permute([0, 2, 1, 3])
        });

        // // TODO: The source uses position_ids%mini_batch_size
        // //       We just use start_idx for now
        // // let (xq, xk) = self.apply_rotary_emb(xq, xk, position_ids);
        let [xq, xk] = [xq, xk].map(|x| self.rot_enc.apply(x, start_idx));

        // [B, num_heads, seq_len, dim] -> [B*num_heads, seq_len, dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([
                (batch_size * self.config.num_heads) as i32,
                seq_len as i32,
                -1,
            ])
        });

        (QKV { xq, xk, xv }, gate)
    }

    /// Gets the learning rate for each head of each token
    ///
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    ///
    /// Returns a tensor of shape `[batch_size, num_heads, seq_len]`
    fn get_lr(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [B, seq_len, token_dim] -> [B, seq_len, num_heads]
        let lr = self.learning_rate.forward(x);
        // [B, seq_len, num_heads] -> [B, num_heads, seq_len]
        let lr = sigmoid(lr.permute([0, 2, 1])).add_scalar(self.config.base_lr);
        lr
    }

    fn conv_qk(&self, xqk: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let conv_q_weight = self.q_conv.weight.val();
        let conv_k_weight = self.k_conv.weight.val();

        let conv_q_bias = self.q_conv.bias.as_ref().map(|x| x.val());
        let conv_k_bias = self.k_conv.bias.as_ref().map(|x| x.val());

        // Since causal_conv has groups=dim, this means that conv(q concat k) == conv(q) concat conv(k)
        // So we could merge these here into one convolution call

        let xq = causal_conv1d_fn(xqk.clone(), conv_q_weight, conv_q_bias);
        let xk = causal_conv1d_fn(xqk, conv_k_weight, conv_k_bias);

        (xq, xk)
    }

    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, dim]`.
    /// - `start_idx`: TODO
    fn get_inner_loop_inputs(&self, x: Tensor<B, 3>, start_idx: usize) -> TTTInputs<B> {
        // In source, this is token_idx + learnable_token_idx_bias, we merge the two right now.
        let token_idx = self.learnable_token_idx.clone().clamp_min(0.);

        let (qkv, gate) = self.get_qkvg(x.clone(), start_idx);
        let lr = self.get_lr(x);

        TTTInputs {
            inner: TTTInputsInner { qkv, token_idx, lr },
            gate,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        inner: impl TTTInnerModel<B>,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.shape().dims();

        let inputs = self.get_inner_loop_inputs(x, start_idx);

        debug_assert_eq!(
            batch_size * self.config.num_heads,
            inputs.inner.qkv.xq.shape().dims[0]
        );

        let out = inner.forward(inputs.inner);
        // TODO: Gate and norm

        let out = out
            .reshape([
                batch_size,
                self.config.num_heads,
                seq_len,
                self.config.value_size,
            ])
            .permute([0, 2, 1, 3])
            .reshape([
                batch_size,
                seq_len,
                self.config.num_heads * self.config.value_size,
            ]);

        // [..] -> [unchanged]
        let out = gelu(inputs.gate) * self.post_norm.forward(out);
        // [.., num_heads*value_size] -> [.., num_heads*token_size] (equal if using reference impls config)
        let out = self.o_proj.forward(out);

        return out;
    }
}

/// Applies causal convolution with no initial states
// Returns (batch_size, dim, seq_len)
fn causal_conv1d_fn<B: Backend>(
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
    let out = out.slice([0..dim, 0..seq_len]);

    out
}

#[derive(Module, Debug)]
struct TTTLinear<B: Backend> {
    // layer: Linear<B>,
    weight_init: Param<Tensor<B, 2>>,
    bias_init: Param<Tensor<B, 1>>,
    // layer_norm: LayerNorm<B>,
    norm_weight: Param<Tensor<B, 4>>,
    norm_bias: Param<Tensor<B, 4>>,
    config: Ignored<Arc<TTTConfig>>,
}

#[derive(Config, Debug)]
struct TTTLinearConfig {
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl<B: Backend> TTTInnerModel<B> for TTTLinear<B> {
    type Config = TTTLinearConfig;

    fn new(global_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Self {
            // layer: LinearConfig::new(global_config.value_size, global_config.value_size)
            //     .with_bias(true)
            //     .init(device),
            weight_init: config.initializer.init(
                [
                    1,
                    global_config.num_heads * global_config.value_size,
                    global_config.num_heads * global_config.value_size,
                ],
                device,
            ),
            bias_init: config.initializer.init(
                [1, global_config.num_heads * global_config.value_size],
                device,
            ),
            // layer_norm: LayerNormConfig::new(global_config.num_heads * global_config.value_size)
            //     .with_epsilon(global_config.epsilon)
            //     .init(device),
            norm_weight: config.initializer.init(
                [1, global_config.num_heads, 1, global_config.value_size],
                device,
            ),
            norm_bias: config.initializer.init(
                [1, global_config.num_heads, 1, global_config.value_size],
                device,
            ),
            config: Ignored(global_config.clone()),
        }
    }

    // x + LayerNorm(Linear(x))
    //
    // TODO: Pass current weights (don't use weight_init)
    fn forward(&self, inputs: TTTInputsInner<B>) -> Tensor<B, 3> {
        let qkv = inputs.qkv;

        let [batch_times_heads, seq_len, value_size] = inputs.qkv.xv.shape().dims();
        debug_assert_eq!(value_size, self.config.value_size);
        debug_assert_eq!(batch_times_heads % self.config.num_heads, 0);

        let batch_size = batch_times_heads / self.config.num_heads;

        let z =
            qkv.xq.matmul(self.weight_init.val().unsqueeze()) + self.bias_init.val().unsqueeze();

        let (var, mean) = z.var_mean_bias(2);
        let std = (var + self.config.epsilon).sqrt();

        let norm = (z - mean) / std;
        let out = self.norm_weight.val()
            * norm.reshape([batch_size, self.config.num_heads, seq_len, value_size])
            + self.norm_bias.val();

        // Since we're using a residual connection, our target is the residual
        let target = qkv.xv - qkv.xk;

        let dl_dout =
            out - target.reshape([batch_size, self.config.num_heads, seq_len, value_size]);

        let dl_dnorm =
            (dl_dout * self.norm_weight.val()).reshape([batch_times_heads, seq_len, value_size]);

        let dl_dz_term1 = dl_dnorm * (value_size as f32);
        let dl_dz_term2 = dl_dnorm.sum_dim(2).unsqueeze_dim(2);
        let dl_dz_term3 = norm * (dl_dnorm * norm).sum_dim(2).unsqueeze_dim(2);

        let dl_dz = (dl_dz_term1 - dl_dz_term2 - dl_dz_term3) / (std * (value_size as f32));

        let step = inputs.lr * dl_dz;

        let weight_grad = qkv.xk.swap_dims(2, 3).matmul(step);
        let bias_grad = step;

        let weight_new = self.weight_init.val() - inputs.token_idx.unsqueeze() * weight_grad;
        let bias_new = self.bias_init.val() - inputs.token_idx.unsqueeze() * bias_grad;

        // Recalculate after the backprop step
        let z_new = weight_new.matmul(qkv.xq) + bias_new;

        // No layernorm after in reference!?

        z_new
    }
}
