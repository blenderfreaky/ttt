use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        LayerNorm, LayerNormConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    prelude::Backend,
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
    /// The size of key and query vectors.
    /// In source it seems to be token_size/2
    key_size: usize,
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
    config: Ignored<TTTConfig>,
    // TODO: Make positional encoding generic/exchangable for different types of positional encodings
    rot_enc: RotaryEncoding<B>,
}

pub trait TTTVariation<B: Backend> {
    fn process(&self, inputs: TTTInputs<B>) -> Tensor<B, 3>;
}

impl TTTConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TTT<B> {
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
            qkvg_proj: linear(
                self.token_size,
                self.key_size * 2 + self.value_size + self.num_heads,
                false,
            ),
            o_proj: linear(self.token_size, self.token_size, false),
            q_conv: conv(self.key_size),
            k_conv: conv(self.key_size),
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
            config: Ignored(self),
        }
    }
}

struct QKVG<B: Backend> {
    /// [batch_size*num_heads, seq_len, head_dim]
    xq: Tensor<B, 3>,
    /// [batch_size*num_heads, seq_len, head_dim]
    xk: Tensor<B, 3>,
    /// [batch_size*num_heads, seq_len, head_dim]
    xv: Tensor<B, 3>,
    /// [batch_size, seq_len, num_heads]
    gate: Tensor<B, 3>,
}

struct TTTInputs<B: Backend> {
    /// The key, query and value vectors as well as the gate values
    qkvg: QKVG<B>,
    /// Learning rate for each head of each token
    /// `[batch_size, num_heads, seq_len]`
    lr: Tensor<B, 3>,
    /// Offset token index for each token
    /// `[seq_len]`
    token_idx: Tensor<B, 1>,
}

impl<B: Backend> TTT<B> {
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    fn get_qkvg(&self, x: Tensor<B, 3>, start_idx: usize) -> QKVG<B> {
        let [batch_size, seq_len, _] = x.shape().dims();

        let qkv = self.qkvg_proj.forward(x);
        let [xq, xk, xv, gate] = qkv
            .split_with_sizes(
                vec![
                    self.config.key_size,
                    self.config.key_size,
                    self.config.value_size,
                    self.config.num_heads,
                ],
                2,
            )
            .try_into()
            .unwrap();

        let (xq, xk) = self.conv_qk(xq, xk);

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

        QKVG { xq, xk, xv, gate }
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

    fn conv_qk(&self, xq: Tensor<B, 3>, xk: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let conv_q_weight = self.q_conv.weight.val();
        let conv_k_weight = self.k_conv.weight.val();

        let conv_q_bias = self.q_conv.bias.as_ref().map(|x| x.val());
        let conv_k_bias = self.k_conv.bias.as_ref().map(|x| x.val());

        // Since causal_conv has groups=dim, this means that conv(q concat k) == conv(q) concat conv(k)
        // So we could merge these here into one convolution call

        let xq = causal_conv1d_fn(xq, conv_q_weight, conv_q_bias);
        let xk = causal_conv1d_fn(xk, conv_k_weight, conv_k_bias);

        (xq, xk)
    }

    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, dim]`.
    /// - `start_idx`: TODO
    fn get_inner_loop_inputs(&self, x: Tensor<B, 3>, start_idx: usize) -> TTTInputs<B> {
        // In source, this is token_idx + learnable_token_idx_bias, we merge the two right now.
        let token_idx = self.learnable_token_idx.clone().clamp_min(0.);

        let qkvg = self.get_qkvg(x.clone(), start_idx);
        let lr = self.get_lr(x);

        TTTInputs {
            qkvg,
            lr,
            token_idx,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, inner: !, start_idx: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.shape().dims();

        let inputs = self.get_inner_loop_inputs(x, start_idx);

        debug_assert_eq!(
            batch_size * self.config.num_heads,
            inputs.qkvg.xq.shape().dims[0]
        );

        let out = inner(inputs);
        // TODO: Gate and norm

        let out = out
            .reshape([
                batch_size,
                self.config.num_heads,
                seq_len,
                self.config.token_size,
            ])
            .permute([0, 2, 1, 3]);

        // [..] -> [unchanged]
        let out = gelu(inputs.qkvg.gate) * self.post_norm.forward(out);
        // [..] -> [unchanged]
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
