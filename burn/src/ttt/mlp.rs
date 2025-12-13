use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Tensor, activation::gelu, s},
};

use super::{
    TTTConfig,
    layer::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

/// GELU backward (derivative of tanh approximation)
/// Reference: https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py#L509
fn gelu_bwd<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // Constants from the tanh approximation of GELU
    let sqrt_2_over_pi = 0.797_884_6_f32;
    let coeff = 0.044_715_f32;

    // tanh_out = tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2))
    let x_sq = x.clone().powf_scalar(2.0);
    let inner = x.clone() * (x_sq.clone() * coeff + 1.0) * sqrt_2_over_pi;
    let tanh_out = inner.tanh();

    // ff = 0.5 * x * ((1 - tanh_out^2) * (sqrt(2/pi) + 3 * 0.044715 * sqrt(2/pi) * x^2)) + 0.5 * (1 + tanh_out)
    let tanh_sq = tanh_out.clone().powf_scalar(2.0);
    let one_minus_tanh_sq = tanh_sq.neg() + 1.0;
    let coeff2 = 3.0 * coeff * sqrt_2_over_pi; // 0.1070322243
    let inner2 = one_minus_tanh_sq * (x_sq * coeff2 + sqrt_2_over_pi);
    let term1 = x * inner2 * 0.5;
    let term2 = (tanh_out + 1.0) * 0.5;

    term1 + term2
}

#[derive(Module, Debug)]
pub struct TTTMLP<B: Backend> {
    /// First layer weight: [num_heads, head_dim, 4*head_dim]
    pub w1_init: Param<Tensor<B, 3>>,
    /// First layer bias: [num_heads, 4*head_dim]
    pub b1_init: Param<Tensor<B, 2>>,
    /// Second layer weight: [num_heads, 4*head_dim, head_dim]
    pub w2_init: Param<Tensor<B, 3>>,
    /// Second layer bias: [num_heads, head_dim]
    pub b2_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<Arc<TTTConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTMLPState<B: Backend> {
    /// First layer weight: [batch_size, num_heads, head_dim, 4*head_dim]
    pub w1: Tensor<B, 4>,
    /// First layer bias: [batch_size, num_heads, 4*head_dim]
    pub b1: Tensor<B, 3>,
    /// Second layer weight: [batch_size, num_heads, 4*head_dim, head_dim]
    pub w2: Tensor<B, 4>,
    /// Second layer bias: [batch_size, num_heads, head_dim]
    pub b2: Tensor<B, 3>,
    /// Gradient accumulators (for primal form, unused in dual form)
    pub w1_grad: Tensor<B, 4>,
    pub b1_grad: Tensor<B, 3>,
    pub w2_grad: Tensor<B, 4>,
    pub b2_grad: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTMLPConfig {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
}

impl Default for TTTMLPConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TTTInnerModel<B> for TTTMLP<B> {
    type Config = TTTMLPConfig;
    type State = TTTMLPState<B>;

    fn name() -> &'static str {
        "TTTMLP"
    }

    fn new(global_config: &Arc<TTTConfig>, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        let len = global_config.hidden_size;
        let head_dim = global_config.head_dim();
        let mlp_dim = 4 * head_dim;

        Self {
            w1_init: config.initializer.init_with(
                [global_config.num_heads, head_dim, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            b1_init: config.initializer.init_with(
                [global_config.num_heads, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            w2_init: config.initializer.init_with(
                [global_config.num_heads, mlp_dim, head_dim],
                Some(len),
                Some(len),
                device,
            ),
            b2_init: config.initializer.init_with(
                [global_config.num_heads, head_dim],
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
        let w1 = self.w1_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b1 = self.b1_init.val().unsqueeze().repeat_dim(0, batch_size);
        let w2 = self.w2_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b2 = self.b2_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTMLPState {
            w1_grad: w1.zeros_like(),
            b1_grad: b1.zeros_like(),
            w2_grad: w2.zeros_like(),
            b2_grad: b2.zeros_like(),
            w1,
            b1,
            w2,
            b2,
        }
    }

    fn forward_one(&self, state: &mut TTTMLPState<B>, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        // Single token is just mini-batch with size 1
        debug_assert_eq!(inputs.qkv.xq.shape().dims::<4>()[2], 1);
        self.forward_mini_batch(state, inputs)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: TTTInputsInner<B>,
    ) -> Tensor<B, 4> {
        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, _head_dim] = qkv.xq.shape().dims();

        // X1 = XK
        let x1 = qkv.xk.clone();

        // Forward through first layer
        // Z1 = X1 @ W1 + b1
        // X1: [B, H, K, D], W1: [B, H, D, 4D]
        let z1 = x1.clone().matmul(state.w1.clone()) + state.b1.clone().unsqueeze_dim(2);

        // X2 = GELU(Z1)
        let x2 = gelu(z1.clone());

        // Forward through second layer
        // Z2 = X2 @ W2 + b2
        // X2: [B, H, K, 4D], W2: [B, H, 4D, D]
        let z2 = x2.clone().matmul(state.w2.clone()) + state.b2.clone().unsqueeze_dim(2);

        // Reconstruction target
        let reconstruction_target = qkv.xv - qkv.xk;

        // Compute gradient of loss with respect to Z2
        let (_ln_out, grad_l_wrt_z2) = self
            .layer_norm
            .forward_and_l2_grad(z2, reconstruction_target);

        // Backpropagate through second layer and GELU
        // grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2^T * gelu_bwd(Z1)
        let grad_l_wrt_z1 =
            grad_l_wrt_z2.clone().matmul(state.w2.clone().transpose()) * gelu_bwd(z1);

        // Compute eta matrices
        let token_eta = inputs.token_eta.unsqueeze_dims::<4>(&[0, 0, -1]); // [1, 1, K, 1]
        let ttt_lr_eta = inputs.ttt_lr_eta.unsqueeze_dim::<4>(2); // [B, H, 1, K]
        let eta_combined = token_eta * ttt_lr_eta;
        let eta_batch = eta_combined.tril(0); // [B, H, K, K]

        // --- First layer updates (dual form) ---

        // Attn1 = tril(XQ @ X1^T)
        let attn1 = qkv.xq.clone().matmul(x1.clone().transpose()).tril(0);

        // b1_bar = b1 - eta_batch @ grad_l_wrt_Z1
        let b1_bar =
            state.b1.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z1.clone());

        // Z1_bar = XQ @ W1 - (eta_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
        let z1_bar = qkv.xq.clone().matmul(state.w1.clone())
            - (eta_batch.clone() * attn1).matmul(grad_l_wrt_z1.clone())
            + b1_bar;

        // X2_bar = GELU(Z1_bar)
        let x2_bar = gelu(z1_bar);

        // --- Second layer updates (dual form) ---

        // Attn2 = tril(X2_bar @ X2^T)
        let attn2 = x2_bar.clone().matmul(x2.clone().transpose()).tril(0);

        // b2_bar = b2 - eta_batch @ grad_l_wrt_Z2
        let b2_bar =
            state.b2.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z2.clone());

        // Z2_bar = X2_bar @ W2 - (eta_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar
        let z2_bar = x2_bar.matmul(state.w2.clone())
            - (eta_batch.clone() * attn2).matmul(grad_l_wrt_z2.clone())
            + b2_bar;

        // --- Update state with last row of eta ---

        let last_eta_row = eta_batch.slice(s![.., .., seq_len - 1..seq_len, ..]);
        let last_eta_col = last_eta_row.transpose(); // [B, H, K, 1]

        // Update W1: W1_last = W1 - (last_eta_col * X1)^T @ grad_l_wrt_Z1
        state.w1 = state.w1.clone()
            - (last_eta_col.clone() * x1)
                .transpose()
                .matmul(grad_l_wrt_z1.clone());

        // Update b1: b1_last = b1 - sum(last_eta_col * grad_l_wrt_Z1, dim=-2)
        state.b1 = state.b1.clone()
            - (last_eta_col.clone() * grad_l_wrt_z1)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        // Update W2: W2_last = W2 - (last_eta_col * X2)^T @ grad_l_wrt_Z2
        state.w2 = state.w2.clone()
            - (last_eta_col.clone() * x2)
                .transpose()
                .matmul(grad_l_wrt_z2.clone());

        // Update b2: b2_last = b2 - sum(last_eta_col * grad_l_wrt_Z2, dim=-2)
        state.b2 = state.b2.clone()
            - (last_eta_col * grad_l_wrt_z2)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        // Clear accumulated gradients after mini-batch
        state.w1_grad = state.w1_grad.zeros_like();
        state.b1_grad = state.b1_grad.zeros_like();
        state.w2_grad = state.w2_grad.zeros_like();
        state.b2_grad = state.b2_grad.zeros_like();

        // Apply layer norm to final output
        let z2_bar_normalized = self.layer_norm.forward(z2_bar);

        // Output: XQ + Z2_bar
        qkv.xq + z2_bar_normalized
    }
}
