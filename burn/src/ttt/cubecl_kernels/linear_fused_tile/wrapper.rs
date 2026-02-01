//! TTTInnerModel implementation for the tiled TTT-Linear kernel.

use std::{ops::Range, sync::Arc};

use burn::tensor::Tensor;

use super::api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};
use crate::ttt::{
    TTTConfig,
    cubecl_kernels::{Fused, FusedTttBackend, FusedTttConfig, TileKernel, TileMultiKernel},
    layer::{TTTInnerModel, TTTInputsInner},
    linear::TTTLinear,
};

/// TTTInnerModel implementation for the tiled fused kernel.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, TileKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused::new(TTTLinear::new(general_config, config, device))
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = self.inner.get_config();
        let threads = inner_config.threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config = FusedTttConfig::new(
            inner_config.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            config,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

/// TTTInnerModel implementation for the multi-stage tiled fused kernel.
///
/// This implementation overrides `forward()` to process all mini-batches in a
/// single kernel launch, rather than launching the kernel once per mini-batch.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, TileMultiKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileMultiTTTLinear"
    }

    fn new(
        general_config: &Arc<TTTConfig>,
        config: &Arc<Self::Config>,
        device: &B::Device,
    ) -> Self {
        Fused::new(TTTLinear::new(general_config, config, device))
    }

    fn get_config(&self) -> &Arc<TTTConfig> {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    /// Override forward to use multi-stage kernel for full sequence processing.
    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let inner = &self.inner;
        let config = inner.get_config();
        let mini_batch_size = config.mini_batch_size;

        let [_batch_size, _num_heads, seq_len, head_dim] = inputs.qkv.xv.shape().dims();
        let num_full_batches = seq_len / mini_batch_size;
        let remainder = seq_len % mini_batch_size;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = config.threads
            .unwrap_or_else(|| super::api::default_threads(mini_batch_size, head_dim));
        let config = FusedTttConfig::new(mini_batch_size, head_dim, epsilon, threads);

        if num_full_batches > 0 {
            // Process full mini-batches with multi-stage kernel
            let full_seq_len = num_full_batches * mini_batch_size;
            let full_qkv = inputs.qkv.slice_seq(0..full_seq_len);
            let [batch_size, num_heads, _] = inputs.ttt_lr_eta.shape().dims();
            let full_ttt_lr_eta =
                inputs
                    .ttt_lr_eta
                    .clone()
                    .slice([0..batch_size, 0..num_heads, 0..full_seq_len]);

            // token_eta is constant across stages - slice to [mini_batch_size] if needed
            let token_eta = inputs.token_eta.clone().slice([0..mini_batch_size]);
            let (output, weight_updated, bias_updated) = fused_ttt_tile_forward_multi::<B>(
                full_qkv.xq,
                full_qkv.xk,
                full_qkv.xv,
                state.weight.clone(),
                state.bias.clone(),
                token_eta,
                full_ttt_lr_eta,
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );

            state.weight = weight_updated;
            state.bias = bias_updated;

            if remainder == 0 {
                output
            } else {
                let remainder_output =
                    self.forward_mini_batch(state, &inputs, full_seq_len..seq_len);

                Tensor::cat(vec![output, remainder_output], 2)
            }
        } else {
            // Sequence shorter than mini_batch_size, use single-stage kernel
            self.forward_mini_batch(state, &inputs, 0..seq_len)
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        if range.len() != self.inner.config.mini_batch_size {
            panic!("Sequence length must be equal to mini_batch_size");
        }

        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = inner.config.threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config = FusedTttConfig::new(
            inner.config.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            config,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::ttt::{
        GpuAutodiffBackend, GpuBackend,
        cubecl_kernels::{FusedTile, FusedTileMulti, test_utils::{TestDims, test_fmb, test_fwd, test_backward_fmb}},
        linear::TTTLinearState,
    };

    #[test]
    fn test_fused_tile_vs_ttt_linear() {
        let dims = TestDims::new(2, 2, 32, 8);
        test_fmb::<GpuBackend, FusedTile<GpuBackend>, TTTLinearState<GpuBackend>, _>(
            dims,
            |m| m.into(),
            1e-2,
            1e-3,
            "FusedTile",
        );
    }

    #[test]
    fn test_fused_tile_multi_vs_ttt_linear() {
        let dims = TestDims::multi_stage(2, 2, 32, 8, 4);
        test_fwd::<GpuBackend, FusedTileMulti<GpuBackend>, TTTLinearState<GpuBackend>, _>(
            dims,
            |m| m.into(),
            1e-2,
            1e-3,
            "FusedTileMulti",
        );
    }

    #[test]
    fn test_fused_tile_backward_gradients_vs_reference() {
        let dims = TestDims::new(2, 2, 32, 8);
        test_backward_fmb::<GpuAutodiffBackend, FusedTile<GpuAutodiffBackend>, _>(
            dims,
            |m| m.into(),
            2e-2,
            1e-3,
            "FusedTile",
        );
    }
}
