use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d, RotaryEncoding,
        RotaryEncodingConfig,
    },
    prelude::Backend,
    tensor::{
        activation::{gelu, sigmoid},
        module::conv1d,
        ops::ConvOptions,
        Dim, NamedDim, Tensor,
    },
};

NamedDim!(Batch);
NamedDim!(SeqLength);
NamedDim!(NumHeads);
NamedDim!(DHead);
NamedDim!(DToken);
NamedDim!(DKey);
NamedDim!(DValue);

/// Configuration for the TTT layer.
#[derive(Config, Debug)]
struct TTTConfig {
    // TODO: INFER!!!
    token_dim: usize,
    width: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    conv_kernel_size: usize,
    mini_batch_size: usize,
    max_sequence_len: usize,
    base_lr: f32,
}

/// Configuration for the TTT layers internal model.
#[derive(Module, Clone, Debug)]
struct TTTTestTimeConfig {
    mini_batch_size: usize,
    width: usize,
    num_heads: usize,
    head_dim: usize,
    base_lr: f32,
}

#[derive(Module, Debug)]
struct TTT<B: Backend> {
    qkv_proj: Linear<B>,
    o_proj: Linear<B>,
    q_conv: Conv1d<B>,
    k_conv: Conv1d<B>,
    learning_rate: Linear<B>,
    learnable_token_idx: Tensor<B, 1>,
    post_norm: LayerNorm<B>,
    config: TTTTestTimeConfig,
    // config: Ignored<TTTConfig>,
    rot_enc: RotaryEncoding<B>,
}

pub trait TTTVariation<B: Backend> {
    fn process(&self, inputs: TTTInputs<B>) -> Tensor<B, 3>;
}

impl TTTConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TTT<B> {
        let projected_size = self.num_heads * self.head_dim;

        let linear = |in_size, out_size, bias| {
            LinearConfig::new(in_size, out_size)
                .with_bias(bias)
                .init(device)
        };

        let mk_conv = || {
            Conv1dConfig::new(self.hidden_size, self.hidden_size, self.conv_kernel_size)
                .with_groups(self.hidden_size)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(
                    self.conv_kernel_size - 1,
                ))
                .with_bias(true)
                .init(device)
        };

        TTT {
            // TODO: I'm not sure if dim(q/k) == dim(v)
            qkv_proj: linear(self.width, projected_size * 3, false),
            o_proj: linear(projected_size, self.width, false),
            q_conv: mk_conv(),
            k_conv: mk_conv(),
            // One output per head
            learning_rate: linear(self.hidden_size, self.num_heads, true),
            // We store token_idx + bias as one, their impl splits it
            learnable_token_idx: Tensor::arange(1..(self.mini_batch_size as i64), device)
                .float()
                .recip(),
            post_norm: LayerNormConfig::new(self.width)
                .with_epsilon(1e-6)
                .init(device),
            rot_enc: RotaryEncodingConfig::new(self.max_sequence_len, self.token_dim).init(device),
            config: TTTTestTimeConfig {
                mini_batch_size: self.mini_batch_size,
                width: self.width,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
                base_lr: self.base_lr,
            },
        }
    }
}

struct QKV<B: Backend> {
    xq: Tensor<B, 3>,
    xk: Tensor<B, 3>,
    xv: Tensor<B, 3>,
}

struct TTTInputs<B: Backend> {
    // This is NOT unmodified QKV projections, this is post conv and positional encoding
    // q, k, v: [batch_size*num_heads, seq_len, head_dim]
    qkv: QKV<B>,
    // []
    gate: Tensor<B, 3>,
    lr: Tensor<B, 3>,
    token_idx: Tensor<B, 1>,
}

impl<B: Backend> TTT<B> {
    fn get_qkv_projections(&self, x: Tensor<B, 3>) -> QKV<B> {
        let qkv = self.qkv_proj.forward(x);
        let [xq, xk, xv] = qkv.split(self.config.width, 3).try_into().unwrap();
        QKV { xq, xk, xv }
    }

    fn get_lr(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [B, seq_len, token_dim] -> [B, seq_len, num_heads]
        let lr = self.learning_rate.forward(x);
        // [B, seq_len, num_heads] -> [B, num_heads, seq_len]
        let lr = sigmoid(lr.permute([0, 2, 1])).add_scalar(self.config.base_lr);
        lr
    }

    // fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
    //     x.reshape([0, 0, self.config.num_heads, self.config.head_dim])
    //         .permute([0, 2, 1, 3])
    // }

    fn conv_qk(&self, xq: Tensor<B, 3>, xk: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // let conv_q = self.conv_q.forward(xq);
        // let conv_k = self.conv_k.forward(xk);
        let conv_q_weight = self.q_conv.weight.val();
        let conv_k_weight = self.k_conv.weight.val();

        let conv_q_bias = self.q_conv.bias.as_ref().map(|x| x.val());
        let conv_k_bias = self.k_conv.bias.as_ref().map(|x| x.val());

        // Since causal_conv has groups=dim, this means that conv(q concat k) == conv(q) concat conv(k)
        // So we could merge these here

        let xq = causal_conv1d_fn(xq, conv_q_weight, conv_q_bias);
        let xk = causal_conv1d_fn(xk, conv_k_weight, conv_k_bias);

        (xq, xk)
    }

    fn get_inner_loop_inputs(&self, x: Tensor<B, 3>, start_idx: usize) -> TTTInputs<B> {
        let [batch_size, seq_len, dim] = x.shape().dims();

        // let n_mini_batch = batch_size / self.config.mini_batch_size;
        // let x = batch.reshape([
        //     batch_size,
        //     n_mini_batch,
        //     self.config.mini_batch_size,
        //     self.config.width,
        // ]);

        // In source, this is token_idx + learnable_token_idx_bias, we merge the two right now.
        let token_idx = self.learnable_token_idx.clone().clamp_min(0.);

        let QKV { xq, xk, xv } = self.get_qkv_projections(x.clone());
        let gate = self.o_proj.forward(x.clone());
        let lr = self.get_lr(x);

        // let [xq, xk, xv] = [xq, xk, xv].map(|x| self.split_heads(x));
        // // TODO: The source uses position_ids%mini_batch_size
        // //       We just use start_idx for now
        // // let (xq, xk) = self.apply_rotary_emb(xq, xk, position_ids);
        // let [xq, xk] = [xq, xk].map(|x| self.rot_enc.apply(x, start_idx));
        // let [xq, xk, xv] = [xq, xk, xv].map(|x| self.split_mini_batches(x));

        let (xq, xk) = self.conv_qk(xq, xk);

        // [B, seq_len, num_heads*head_dim] -> [B, num_heads, seq_len, head_dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([0, 0, self.config.num_heads, self.config.head_dim])
                .permute([0, 2, 1, 3])
        });

        let [xq, xk] = [xq, xk].map(|x| self.rot_enc.apply(x, start_idx));

        // [B, num_heads, seq_len, head_dim] -> [B*num_heads, seq_len, head_dim]
        let [xq, xk, xv] =
            [xq, xk, xv].map(|x| x.reshape([-1, seq_len as i32, self.config.head_dim as i32]));

        TTTInputs {
            qkv: QKV { xq, xk, xv },
            gate,
            lr,
            token_idx,
        }
    }

    // fn split_mini_batches(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
    //     // let batch_size = x.shape()[0];

    //     x.reshape([
    //         0,
    //         -1,
    //         self.config.mini_batch_size as i32,
    //         self.config.num_heads as i32,
    //         self.config.head_dim as i32,
    //     ])
    //     .permute([0, 3, 1, 2, 4])
    // }

    fn forward(&self, x: Tensor<B, 3>, inner: todo!(), start_idx: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = x.shape().dims();

        let inputs = self.get_inner_loop_inputs(x, start_idx);

        debug_assert_eq!(
            batch_size * self.config.num_heads,
            inputs.qkv.xq.shape().dims[0]
        );

        let out = inner(inputs);
        // TODO: Gate and norm

        let out = out
            .reshape([
                batch_size,
                self.config.num_heads,
                seq_len,
                self.config.head_dim,
            ])
            .permute([0, 2, 1, 3]);

        let out = gelu(inputs.gate) * self.post_norm.forward(out);

        let out = self.o_proj.forward(out);

        return out;
    }
}

/// Applies causal convolution with no initial states
// Returns (batch_size, dim, seq_len)
fn causal_conv1d_fn<B: Backend>(
    // (batch_size, dim, seq_len)
    x: Tensor<B, 3>,
    // (1, dim, width)
    weight: Tensor<B, 3>,
    // (dim,)
    bias: Option<Tensor<B, 1>>,
    // // (batch, dim, width - 1)
    // initial_states: Option<Tensor<B, 3>>,
    // // (batch, dim, width - 1)
    // final_states_out: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [_batch_size, dim, seq_len] = x.shape().dims();
    let [_dim, width] = weight.shape().dims();

    // if let Some(initial_states) = initial_states {
    //     x = Tensor::cat(vec![initial_states, x], 2);

    let out = conv1d(
        x,
        weight.unsqueeze_dim(1),
        bias,
        // Why is groups = dim? This seems inefficient.
        ConvOptions::new([1], [width - 1], [1], dim),
    );
    let out = out.slice([0..dim, 0..seq_len]);

    out
}
