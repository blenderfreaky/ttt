use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Tensor, s},
};

use super::{
    TTTConfig,
    layer::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

#[derive(Module, Debug)]
pub struct TTTLinear<B: Backend> {
    /// [num_heads, head_dim, head_dim]
    pub weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, head_dim]
    pub bias_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<Arc<TTTConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTLinearState<B: Backend> {
    /// [batch_size, num_heads, head_dim, head_dim]
    pub weight: Tensor<B, 4>,
    /// [batch_size, num_heads, head_dim]
    pub bias: Tensor<B, 3>,
    /// [batch_size, num_heads, head_dim, head_dim]
    pub weight_grad: Tensor<B, 4>,
    /// [batch_size, num_heads, head_dim]
    pub bias_grad: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTLinearConfig {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
}

impl Default for TTTLinearConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TTTInnerModel<B> for TTTLinear<B> {
    type Config = TTTLinearConfig;
    type State = TTTLinearState<B>;

    fn name() -> &'static str {
        "TTTLinear"
    }

    fn new(global_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        let len = global_config.hidden_size;
        Self {
            weight_init: config.initializer.init_with(
                [
                    global_config.num_heads,
                    global_config.head_dim(),
                    global_config.head_dim(),
                ],
                Some(len),
                Some(len),
                device,
            ),
            bias_init: config.initializer.init_with(
                [global_config.num_heads, global_config.head_dim()],
                Some(len),
                Some(len),
                device,
            ),
            layer_norm: MultiHeadLayerNormConfig::new(
                global_config.num_heads,
                global_config.head_dim(),
            )
            .with_initializer(config.initializer.clone())
            .with_epsilon(global_config.epsilon)
            .init(device),
            config: Ignored(global_config.clone()),
        }
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        &self.config.0
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        let weight = self.weight_init.val().unsqueeze().repeat_dim(0, batch_size);
        let bias = self.bias_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTLinearState {
            weight_grad: weight.zeros_like(),
            bias_grad: bias.zeros_like(),
            weight,
            bias,
        }
    }

    fn forward_one(
        &self,
        state: &mut TTTLinearState<B>,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        // Single token is just mini-batch with size 1
        // We're not optimizing for inference, so this isn't our focus here
        // TODO: I think this is not actually correct,
        //       due to ignoring token_eta and so on. Reference impl implements
        //       a variation of primal here.
        //       I don't think we really need to implement this, but keep it in mind.
        debug_assert_eq!(inputs.qkv.xq.shape().dims::<4>()[2], 1);
        self.forward_mini_batch(state, inputs)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        // For reference:
        // qkv: [batch_size, num_heads, seq_len, head_dim]
        let qkv = inputs.qkv;

        let x1 = qkv.xk.clone();

        // Z1 = X1 @ W1_init + b1_init
        let z1 = x1.clone().matmul(state.weight.clone()) + state.bias.clone().unsqueeze_dim(2);

        let reconstruction_target = qkv.xv - qkv.xk;

        // Compute gradients using fused layer norm + L2 loss backward
        let (_ln_out, grad_l_wrt_z1) = self
            .layer_norm
            .forward_and_l2_grad(z1, reconstruction_target);

        // token_eta: [seq_len] - position-based scale factor (1/k)
        // ttt_lr_eta: [B, H, seq_len] - per-head learned learning rate

        let token_eta = inputs.token_eta.unsqueeze_dims::<4>(&[0, 0, -1]); // [1, 1, seq_len, 1]

        let ttt_lr_eta = inputs.ttt_lr_eta.unsqueeze_dim::<4>(2); // [B, H, 1, seq_len]

        // Combined eta: token_eta * ttt_lr_eta
        // token_eta: [1, 1, seq_len, 1], ttt_lr_eta: [B, H, 1, seq_len]
        // Result: [B, H, seq_len, seq_len] where eta[i,j] = token_eta[i] * ttt_lr_eta[j]
        let eta_combined = token_eta * ttt_lr_eta;

        let eta_batch = eta_combined.tril(0); // [B, H, seq_len, seq_len]

        // Attn1 = tril(XQ_mini_batch @ X1.transpose(-2, -1))
        let attn1 = qkv.xq.clone().matmul(x1.clone().transpose()).tril(0);

        // b1_bar = b1_init - eta_batch @ grad_l_wrt_Z1
        // state.bias is [B, H, D], unsqueeze to [B, H, 1, D] for broadcasting
        // eta_batch @ grad: [B, H, K, K] @ [B, H, K, D] -> [B, H, K, D]
        let b1_bar =
            state.bias.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z1.clone());

        // Z1_bar = XQ_batch @ W1_init - (eta_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
        let z1_bar = qkv.xq.clone().matmul(state.weight.clone())
            - (eta_batch.clone() * attn1).matmul(grad_l_wrt_z1.clone())
            + b1_bar;

        // Update weights with last row of eta matrix
        // Extract last row: eta[:, :, -1, :] -> [B, H, 1, K]
        let last_eta_row = eta_batch.slice(s![.., .., -1.., ..]);

        // W1_last = W1_init - (last_eta_row * X1).transpose(-1, -2) @ grad_l_wrt_Z1
        // last_eta_row: [B, H, 1, K], X1: [B, H, K, D]
        // Need to transpose last_eta_row to [B, H, K, 1] for proper broadcasting
        let last_eta_col = last_eta_row.transpose(); // [B, H, K, 1]

        // (last_eta_col * X1): [B, H, K, 1] * [B, H, K, D] -> [B, H, K, D]
        // .transpose(): [B, H, D, K]
        // @ grad: [B, H, D, K] @ [B, H, K, D] -> [B, H, D, D]
        state.weight = state.weight.clone()
            - (last_eta_col.clone() * x1)
                .transpose()
                .matmul(grad_l_wrt_z1.clone());

        // b1_last = b1_init - sum(last_eta_col * grad_l_wrt_Z1, dim=-2)
        // last_eta_col * grad: [B, H, K, 1] * [B, H, K, D] -> [B, H, K, D]
        // .sum_dim(2): [B, H, 1, D] -> squeeze -> [B, H, D]
        state.bias = state.bias.clone()
            - (last_eta_col * grad_l_wrt_z1)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        // Clear accumulated gradients after mini-batch
        state.weight_grad = state.weight_grad.zeros_like();
        state.bias_grad = state.bias_grad.zeros_like();

        // Apply layer norm to final output
        let z1_bar_normalized = self.layer_norm.forward(z1_bar);

        // XQW_mini_batch = XQ_mini_batch + Z1_bar
        qkv.xq + z1_bar_normalized
    }
}
