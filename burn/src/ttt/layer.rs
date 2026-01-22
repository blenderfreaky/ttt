use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, ModuleDisplay},
    nn::{
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    prelude::*,
    tensor::{
        Tensor,
        activation::{gelu, sigmoid},
    },
};
use burn_backend::Distribution;

use super::{
    PositionEncodingType, TTTConfig,
    util::{RotaryEmbedding, RotaryEmbeddingConfig, causal_conv1d_fn},
};

/// Permute Q/K dimensions to match JAX/EasyLM rotary embedding implementation.
/// This reorders the head dimension to match how EasyLM computes rotary embeddings.
/// Reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
fn permute_qk<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
    // [B, H, L, D] -> [B, H, L, D//2, 2] -> transpose(3,4) -> [B, H, L, 2, D//2] -> [B, H, L, D]
    x.reshape([batch_size, num_heads, seq_len, head_dim / 2, 2])
        .permute([0, 1, 2, 4, 3])
        .reshape([batch_size, num_heads, seq_len, head_dim])
}

/// Undo the Q/K permutation after rotary embedding.
fn undo_permute_qk<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
    // [B, H, L, D] -> [B, H, L, 2, D//2] -> transpose(3,4) -> [B, H, L, D//2, 2] -> [B, H, L, D]
    x.reshape([batch_size, num_heads, seq_len, 2, head_dim / 2])
        .permute([0, 1, 2, 4, 3])
        .reshape([batch_size, num_heads, seq_len, head_dim])
}

pub trait TTTInnerModel<B: Backend>: Module<B> + ModuleDisplay {
    type Config: Config + Default;
    type State: Module<B> + ModuleDisplay;

    fn name() -> &'static str;

    fn new(general_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device)
    -> Self;

    fn init_state(&self, batch_size: usize) -> Self::State;

    fn get_config(&self) -> &Arc<TTTConfig>;

    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let mut output = inputs.qkv.xv.zeros_like();

        let [batch_size, num_heads, seq_len, head_dim] = inputs.qkv.xv.shape().dims();

        let mini_batch_size = self.get_config().mini_batch_size;
        let num_mini_batch = seq_len / mini_batch_size;

        for i in 0..num_mini_batch {
            let start_idx = i * mini_batch_size;
            let inputs = inputs.slice_seq(start_idx..start_idx + mini_batch_size);
            let z = self.forward_mini_batch(state, inputs);
            output = output.slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    start_idx..start_idx + mini_batch_size,
                    0..head_dim,
                ],
                z,
            );
        }

        let last_mini_batch_end = num_mini_batch * mini_batch_size;

        // Process any remaining tokens in a single batch (should work for any seq_len < mini_batch_size)
        if last_mini_batch_end < seq_len {
            let remainder_inputs = inputs.slice_seq(last_mini_batch_end..seq_len);
            let z = self.forward_mini_batch(state, remainder_inputs);
            output = output.slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    last_mini_batch_end..seq_len,
                    0..head_dim,
                ],
                z,
            );
        }

        output
    }

    fn forward_one(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4>;

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4>;
}

#[derive(Module, Debug)]
pub struct TTT<B: Backend> {
    pub q_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub g_proj: Option<Linear<B>>, // Only present if use_gate is true
    pub o_proj: Linear<B>,
    pub q_conv: Conv1d<B>,
    pub k_conv: Conv1d<B>,
    // Per-head learning rate weights
    pub learnable_ttt_lr_weight: Tensor<B, 3>, // [num_heads, token_size, 1]
    pub learnable_ttt_lr_bias: Tensor<B, 2>,   // [num_heads, 1]
    // Base token_idx: 1/k for k=1..mini_batch_size (position-based scale factor)
    pub token_idx: Tensor<B, 1>,
    // Learnable offset to token_idx
    pub learnable_token_idx: Tensor<B, 1>,
    pub post_norm: LayerNorm<B>,
    pub config: Ignored<Arc<TTTConfig>>,
    pub rot_enc: Option<RotaryEmbedding<B>>,
}

impl TTTConfig {
    pub fn init_ttt_seq<B: Backend>(self: &Arc<Self>, device: &B::Device) -> TTT<B> {
        let linear = |in_size, out_size, bias| {
            LinearConfig::new(in_size, out_size)
                .with_bias(bias)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
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

        let learnable_ttt_lr_weight = Tensor::random(
            [self.num_heads, self.token_size, 1],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );

        let learnable_ttt_lr_bias = Tensor::zeros([self.num_heads, 1], device);

        // Base token_idx: 1/k for k=1,2,...,mini_batch_size (position-based scale factor)
        let token_idx = Tensor::arange(1..(self.mini_batch_size as i64 + 1), device)
            .float()
            .recip();

        // Learnable offset to token_idx, initialized to zeros
        let learnable_token_idx = Tensor::zeros([self.mini_batch_size], device);

        TTT {
            q_proj: linear(self.token_size, self.hidden_size, false),
            v_proj: linear(self.token_size, self.hidden_size, false),
            g_proj: if self.use_gate {
                Some(linear(self.token_size, self.hidden_size, false))
            } else {
                None
            },
            o_proj: linear(self.hidden_size, self.token_size, false),
            q_conv: conv(self.hidden_size),
            k_conv: conv(self.hidden_size),
            // Per-head learning rate weights
            learnable_ttt_lr_weight,
            learnable_ttt_lr_bias,
            token_idx,
            learnable_token_idx,
            // Post-norm operates on hidden_size dimension (output of inner model)
            post_norm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            rot_enc: match self.pos_encoding {
                PositionEncodingType::RoPE => Some(
                    RotaryEmbeddingConfig::new(self.head_dim())
                        .with_base(f64::from(self.rope_theta))
                        .init(device),
                ),
                _ => None,
            },
            config: Ignored(self.clone()),
        }
    }
}

#[derive(Clone)]
pub struct Qkv<B: Backend> {
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xq: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xk: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xv: Tensor<B, 4>,
}

impl<B: Backend> Qkv<B> {
    #[must_use]
    pub fn slice_seq(&self, range: Range<usize>) -> Self {
        let [batch_size, num_heads, _, head_dim] = self.xq.shape().dims();
        Self {
            xq: self
                .xq
                .clone()
                .slice([0..batch_size, 0..num_heads, range.clone(), 0..head_dim]),
            xk: self
                .xk
                .clone()
                .slice([0..batch_size, 0..num_heads, range.clone(), 0..head_dim]),
            xv: self
                .xv
                .clone()
                .slice([0..batch_size, 0..num_heads, range, 0..head_dim]),
        }
    }

    pub fn random(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            xq: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
            xk: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
            xv: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
        }
    }
}

#[derive(Clone)]
pub struct TTTInputsInner<B: Backend> {
    /// The key, query and value vectors
    pub qkv: Qkv<B>,
    /// Position-based token scale factor (1/k + learnable offset), clamped >= 0
    /// `[seq_len]`
    pub token_eta: Tensor<B, 1>,
    /// Per-head learning rate (base_lr * sigmoid(learned) / head_dim)
    /// `[batch_size, num_heads, seq_len]`
    pub ttt_lr_eta: Tensor<B, 3>,
    /// Index of the first token in the sequence
    pub start_idx: usize,
}

impl<B: Backend> TTTInputsInner<B> {
    #[must_use]
    pub fn slice_seq(&self, range: Range<usize>) -> Self {
        let [batch_size, num_heads, _] = self.ttt_lr_eta.shape().dims();
        Self {
            qkv: self.qkv.slice_seq(range.clone()),
            token_eta: self.token_eta.clone().slice([range.clone()]),
            start_idx: range.start,
            ttt_lr_eta: self
                .ttt_lr_eta
                .clone()
                .slice([0..batch_size, 0..num_heads, range.clone()]),
        }
    }

    pub fn random(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            qkv: Qkv::random(batch_size, num_heads, seq_len, head_dim, device),
            token_eta: Tensor::random(
                [seq_len],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            ) + Tensor::arange(0..(seq_len as i64), device).float().recip(),
            start_idx: 0,
            ttt_lr_eta: Tensor::random(
                [batch_size, num_heads, seq_len],
                burn::tensor::Distribution::Default,
                device,
            ),
        }
    }
}

impl<B: Backend> TTT<B> {
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    /// - `start_idx`: Starting position index for the sequence
    fn get_qkv(&self, x: Tensor<B, 3>, start_idx: usize) -> Qkv<B> {
        let [batch_size, seq_len, _token_dim] = x.shape().dims();

        let xqk = self.q_proj.forward(x.clone()); // Shared base for Q and K
        let xv = self.v_proj.forward(x);

        let (xq, xk) = self.conv_qk(xqk);

        // [B, seq_len, num_heads*dim] -> [B, num_heads, seq_len, dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim(),
            ])
            .permute([0, 2, 1, 3])
        });

        // Apply rotary position encoding with position_ids % mini_batch_size
        // Use permute_qk/undo_permute_qk to match JAX/EasyLM format (see doc comment on permute_qk)
        let (xq, xk) = match &self.rot_enc {
            Some(rot_enc) => {
                let xq = permute_qk(xq);
                let xk = permute_qk(xk);
                let (xq, xk) = rot_enc.apply(
                    xq,
                    xk,
                    start_idx % self.config.mini_batch_size,
                    self.config.mini_batch_size,
                );
                (undo_permute_qk(xq), undo_permute_qk(xk))
            }
            None => (xq, xk),
        };

        Qkv { xq, xk, xv }
    }

    /// Gets the learning rate for each head of each token using per-head weights
    ///
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    ///
    /// Returns a tensor of shape `[batch_size, num_heads, seq_len]`
    fn get_lr(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let lr_weight = self.learnable_ttt_lr_weight.clone().squeeze_dim::<2>(2); // [num_heads, token_size]

        let lr_weight_t = lr_weight.transpose(); // [token_size, num_heads]

        let lr_weight_t = lr_weight_t.unsqueeze_dim::<3>(0); // [1, token_size, num_heads]

        let lr = x.matmul(lr_weight_t); // [B, seq_len, num_heads]

        let lr_bias_expanded = self
            .learnable_ttt_lr_bias
            .clone()
            .squeeze_dim::<1>(1) // [num_heads]
            .unsqueeze_dim::<2>(0) // [1, num_heads]
            .unsqueeze_dim::<3>(0); // [1, 1, num_heads]

        let lr = lr + lr_bias_expanded; // [B, seq_len, num_heads]

        // [B, seq_len, num_heads] -> [B, num_heads, seq_len]
        let lr_sigmoid = sigmoid(lr.permute([0, 2, 1]));

        (self.config.base_lr * lr_sigmoid) / (self.config.head_dim() as f32)
    }

    fn conv_qk(&self, xqk: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let conv_q_weight = self.q_conv.weight.val();
        let conv_k_weight = self.k_conv.weight.val();

        let conv_q_bias = self.q_conv.bias.as_ref().map(burn::module::Param::val);
        let conv_k_bias = self.k_conv.bias.as_ref().map(burn::module::Param::val);

        // Since causal_conv has groups=dim, this means that conv(q concat k) == conv(q) concat conv(k)
        // So we could merge these here into one convolution call

        // Transpose from [batch, seq_len, hidden_size] to [batch, hidden_size, seq_len] for causal_conv1d_fn
        let xqk_transposed = xqk.permute([0, 2, 1]);

        let xq_transposed = causal_conv1d_fn(xqk_transposed.clone(), conv_q_weight, conv_q_bias);
        let xk_transposed = causal_conv1d_fn(xqk_transposed, conv_k_weight, conv_k_bias);

        // Transpose back to [batch, seq_len, hidden_size]
        let xq = xq_transposed.permute([0, 2, 1]);
        let xk = xk_transposed.permute([0, 2, 1]);

        (xq, xk)
    }

    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, dim]`.
    /// - `start_idx`: Starting index for positional encoding
    fn get_inner_loop_inputs(&self, x: Tensor<B, 3>, start_idx: usize) -> TTTInputsInner<B> {
        let [_batch_size, seq_len, _] = x.shape().dims();

        let token_eta = (self.token_idx.clone() + self.learnable_token_idx.clone())
            .repeat_dim(0, seq_len.div_ceil(self.config.mini_batch_size))
            .slice(s![0..seq_len])
            .clamp_min(0.);

        let qkv = self.get_qkv(x.clone(), start_idx);
        let ttt_lr_eta = self.get_lr(x);

        TTTInputsInner {
            qkv,
            token_eta,
            ttt_lr_eta,
            start_idx,
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

        let inputs = self.get_inner_loop_inputs(x.clone(), start_idx);

        debug_assert_eq!(batch_size, inputs.qkv.xq.shape().dims[0]);

        let out = inner.forward(state, inputs);

        let out = out
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.config.hidden_size]);

        let out = self.post_norm.forward(out);

        let out = if let Some(g_proj) = &self.g_proj {
            let gate = g_proj.forward(x);
            gelu(gate) * out
        } else {
            out
        };

        self.o_proj.forward(out)
    }
}
