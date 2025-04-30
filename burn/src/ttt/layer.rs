use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        LayerNorm, LayerNormConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    prelude::*,
    tensor::{
        activation::{gelu, sigmoid},
        Tensor,
    },
};

use super::{util::causal_conv1d_fn, TTTConfig};

pub trait TTTInnerModel<B: Backend> {
    type Config: Config;
    type State: Module<B>;

    fn new(general_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device)
        -> Self;

    fn init_state(&self, batch_size: usize) -> Self::State;

    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4>;
}

#[derive(Module, Debug)]
pub struct TTT<B: Backend> {
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

impl TTTConfig {
    pub fn init_ttt_seq<B: Backend>(self: &Arc<Self>, device: &B::Device) -> TTT<B> {
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
            qkvg_proj: linear(self.token_size, self.value_size * 3, false),
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

pub struct Qkv<B: Backend> {
    /// [batch_size, num_heads, seq_len, value_size]
    pub xq: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, value_size]
    pub xk: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, value_size]
    pub xv: Tensor<B, 4>,
}

pub struct TTTInputsInner<B: Backend> {
    /// The key, query and value vectors
    pub qkv: Qkv<B>,
    /// Offset token index for each token
    /// `[seq_len]`
    pub token_idx: Tensor<B, 1>,
    /// Learning rate for each head of each token
    /// `[batch_size, num_heads, seq_len]`
    pub lr: Tensor<B, 3>,
    /// Index of the first token in the sequence
    pub start_idx: usize,
}

// TODO: Better name
pub struct TTTInputs<B: Backend> {
    pub inner: TTTInputsInner<B>,
    /// [batch_size, seq_len, num_heads]
    pub gate: Tensor<B, 3>,
}

impl<B: Backend> TTT<B> {
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    fn get_qkvg(&self, x: Tensor<B, 3>, start_idx: usize) -> (Qkv<B>, Tensor<B, 3>) {
        let [_batch_size, _seq_len, _token_dim] = x.shape().dims();

        let proj = self.qkvg_proj.forward(x);
        dbg!(proj.shape());
        let [xqk, gate, xv] = proj.split(self.config.value_size, 2).try_into().unwrap();

        let (xq, xk) = self.conv_qk(xqk);

        // [B, seq_len, num_heads*dim] -> [B, num_heads, seq_len, dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([0, 0, self.config.num_heads as i32, -1])
                .permute([0, 2, 1, 3])
        });

        dbg!(xq.shape());
        dbg!(xk.shape());
        dbg!(xv.shape());

        // // TODO: The source uses position_ids%mini_batch_size
        // //       We just use start_idx for now
        // // let (xq, xk) = self.apply_rotary_emb(xq, xk, position_ids);
        let [xq, xk] = [xq, xk].map(|x| self.rot_enc.apply(x, start_idx));

        (Qkv { xq, xk, xv }, gate)
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
        (sigmoid(lr.permute([0, 2, 1])) + self.config.base_lr) / (self.config.value_size as f32)
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
            inner: TTTInputsInner {
                qkv,
                token_idx,
                lr,
                start_idx,
            },
            gate,
        }
    }

    pub fn forward<Inner: TTTInnerModel<B>>(
        &self,
        // [batch_size, seq_len, token_size]
        x: Tensor<B, 3>,
        inner: &Inner,
        state: &mut Inner::State,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _token_size] = x.shape().dims();

        let inputs = self.get_inner_loop_inputs(x, start_idx);

        debug_assert_eq!(
            batch_size * self.config.num_heads,
            inputs.inner.qkv.xq.shape().dims[0]
        );

        let out = inner.forward(state, inputs.inner);
        // TODO: Gate and norm

        let out = out
            // .reshape([
            //     batch_size,
            //     self.config.num_heads,
            //     seq_len,
            //     self.config.value_size,
            // ])
            .permute([0, 2, 1, 3])
            .reshape([
                batch_size,
                seq_len,
                self.config.num_heads * self.config.value_size,
            ]);

        // [..] -> [unchanged]
        let out = gelu(inputs.gate) * self.post_norm.forward(out);
        // [.., num_heads*value_size] -> [.., num_heads*token_size] (equal if using reference impls config)
        self.o_proj.forward(out)
    }
}
