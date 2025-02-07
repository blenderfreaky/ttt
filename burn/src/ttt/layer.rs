use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::Tensor,
};

/// Configuration for the TTT layer.
#[derive(Config, Debug)]
struct TTTConfig {
    width: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    conv_kernel_size: usize,
    mini_batch_size: usize,
}

/// Configuration for the TTT layers internal model.
#[derive(Module, Clone, Debug)]
struct TTTTestTimeConfig {
    mini_batch_size: usize,
    width: usize,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Module, Debug)]
struct TTT<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_conv: Conv1d<B>,
    k_conv: Conv1d<B>,
    config: TTTTestTimeConfig,
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
            config: TTTTestTimeConfig {
                mini_batch_size: self.mini_batch_size,
                width: self.width,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
            },
        }
    }
}

impl<B: Backend> TTT<B> {
    fn get_qkv_projections(
        &self,
        batch: &Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let q = self.q_proj.forward(batch.clone());
        let k = self.k_proj.forward(batch.clone());
        let v = self.v_proj.forward(batch.clone());
        (q, k, v)
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = x.shape().dims();
        x.reshape([
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])
    }

    fn ttt_inputs(&self, batch: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = batch.shape().dims();
        let n_mini_batch = batch_size / self.config.mini_batch_size;
        let x = batch.reshape([
            batch_size,
            n_mini_batch,
            self.config.mini_batch_size,
            self.config.width,
        ]);

        let (xq, xk, xv) = self.get_qkv_projections(&batch);

        let (xq, xk, xv) = (
            self.split_heads(xq),
            self.split_heads(xk),
            self.split_heads(xv),
        );

        

        ()
    }
}
