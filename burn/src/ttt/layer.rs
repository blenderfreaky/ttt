use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    prelude::Backend,
    tensor::{activation::sigmoid, Tensor},
};

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
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_conv: Conv1d<B>,
    k_conv: Conv1d<B>,
    learning_rate: Linear<B>,
    learnable_token_idx: Tensor<B, 1>,
    config: TTTTestTimeConfig,
    rot_enc: RotaryEncoding<B>,
}

impl TTTConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TTT<B> {
        let projected_size = self.num_heads * self.head_dim;
        let mk_proj_layer = || {
            LinearConfig::new(self.width, projected_size)
                .with_bias(false)
                .init(device)
        };
        TTT {
            q_proj: mk_proj_layer(),
            k_proj: mk_proj_layer(),
            v_proj: mk_proj_layer(),
            o_proj: mk_proj_layer(),
            q_conv: Conv1dConfig::new(self.hidden_size, self.hidden_size, self.conv_kernel_size)
                .init(device),
            k_conv: Conv1dConfig::new(self.hidden_size, self.hidden_size, self.conv_kernel_size)
                .init(device),
            // One output per head
            learning_rate: LinearConfig::new(self.hidden_size, self.num_heads)
                .with_bias(true)
                .init(device),
            learnable_token_idx: Tensor::arange(1..(self.mini_batch_size as i64), device)
                .float()
                .recip(),
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

impl<B: Backend> TTT<B> {
    fn get_qkv_projections(&self, batch: &Tensor<B, 3>) -> [Tensor<B, 3>; 3] {
        let q = self.q_proj.forward(batch.clone());
        let k = self.k_proj.forward(batch.clone());
        let v = self.v_proj.forward(batch.clone());
        [q, k, v]
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        x.reshape([0, 0, self.config.num_heads, self.config.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn ttt_inputs(&self, batch: &Tensor<B, 3>, start_idx: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = batch.shape().dims();
        let n_mini_batch = batch_size / self.config.mini_batch_size;
        let x = batch.reshape([
            batch_size,
            n_mini_batch,
            self.config.mini_batch_size,
            self.config.width,
        ]);

        let [xq, xk, xv] = self.get_qkv_projections(&batch);

        let [xq, xk, xv] = [xq, xk, xv].map(|x| self.split_heads(x));

        // TODO: The source uses position_ids%mini_batch_size
        //       We just use start_idx for now
        // let (xq, xk) = self.apply_rotary_emb(xq, xk, position_ids);
        let [xq, xk] = [xq, xk].map(|x| self.rot_enc.apply(x, start_idx));

        let [xq, xk, xv] = [xq, xk, xv].map(|x| self.split_mini_batches(x));

        ()
    }

    // fn apply_rotary_emb(&self, xq: Tensor<B, 4>, xk: Tensor<B, 4>, position_ids: &Tensor<B, 2>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    //     // Create pairs of two
    //     // let xq = xq.reshape([0, 0, 0, -1, 2]);
    //     // let xk = xk.reshape([0, 0, 0, -1, 2]);

    //     let xq = self.rot_enc.apply(xq, todo!());
    //     let xk = self.rot_enc.apply(xk, todo!());

    //     (xq, xk)
    // }

    fn split_mini_batches(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        // let batch_size = x.shape()[0];

        x.reshape([
            0,
            -1,
            self.config.mini_batch_size as i32,
            self.config.num_heads as i32,
            self.config.head_dim as i32,
        ])
        .permute([0, 3, 1, 2, 4])
    }

    // fn get_qkv_ttt_lr(&self, x: Tensor<B, 3>) {
    //     let [batch_size, seq_len, dim] = x.shape();
    // }

    fn get_lr(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let lr = self.learning_rate.forward(x);

        let lr = sigmoid(lr.permute([0, 2, 1])).add_scalar(self.config.base_lr);

        lr
    }
}
