use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::Tensor,
};

use super::{
    layer::{TTTInnerModel, TTT},
    util::{CausalConv, CausalConvConfig, SwiGluMlp, SwiGluMlpConfig},
    TTTConfig,
};

// We can't write the trait bound due to a limitation of the Module derive macro
// but impls enforce it
//   Inner: TTTInnerModel<B>
#[derive(Module, Debug)]
pub struct TTTBlockWithSeq<B: Backend, Inner> {
    block: TTTBlock<B>,
    inner: Inner,
}

impl<B: Backend, Inner: TTTInnerModel<B>> TTTBlockWithSeq<B, Inner> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        residual: Option<Tensor<B, 3>>,
        state: &mut Inner::State,
        start_idx: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.block
            .forward(input, residual, state, &self.inner, start_idx)
    }

    pub fn init_state(&self, batch_size: usize) -> Inner::State {
        self.inner.init_state(batch_size)
    }
}

#[derive(Module, Debug)]
pub struct TTTBlock<B: Backend> {
    layer_idx: usize,
    conv: Option<(CausalConv<B>, RmsNorm<B>)>,
    // conv_norm: Option<RmsNorm<B>>,
    config: Ignored<Arc<TTTConfig>>,
    seq_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,
    ttt: TTT<B>,
    swi_glu_mlp: SwiGluMlp<B>,
}

#[derive(Config, Debug)]
pub struct TTTBlockConfig {
    ttt_config: Arc<TTTConfig>,
    layer_idx: usize,
}

impl TTTBlockConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TTTBlock<B> {
        TTTBlock {
            layer_idx: self.layer_idx,
            conv: if self.ttt_config.conv_before_ttt {
                Some((
                    CausalConvConfig::new(
                        self.ttt_config.hidden_size,
                        self.ttt_config.conv_kernel_size,
                    )
                    .init(device),
                    RmsNormConfig::new(self.ttt_config.hidden_size).init(device),
                ))
            } else {
                None
            },
            seq_norm: RmsNormConfig::new(self.ttt_config.hidden_size).init(device),
            ffn_norm: RmsNormConfig::new(self.ttt_config.hidden_size).init(device),
            ttt: self.ttt_config.init_ttt_seq(device),
            swi_glu_mlp: SwiGluMlpConfig::new(
                self.ttt_config.hidden_size,
                self.ttt_config.swi_glu_mlp_intermediate_size,
            )
            .init(device),
            config: Ignored(self.ttt_config),
        }
    }

    pub fn init_with_inner<B: Backend, Inner: TTTInnerModel<B>>(
        self,
        inner: Inner,
        device: &B::Device,
    ) -> TTTBlockWithSeq<B, Inner> {
        let block = self.init(device);
        TTTBlockWithSeq { block, inner }
    }
}

// fn add_opt<B: Backend>(x: Tensor<B, 3>, opt: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
//     match opt {
//         Some(a) => x + a,
//         None => x,
//     }
// }

impl<B: Backend> TTTBlock<B> {
    pub fn forward<Inner: TTTInnerModel<B>>(
        &self,
        input: Tensor<B, 3>,
        residual: Option<Tensor<B, 3>>,
        state: &mut Inner::State,
        inner: &Inner,
        start_idx: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // This is not optimal for perf but I'm lazy rn
        let mut residual = residual.unwrap_or_else(|| input.zeros_like());
        let mut hidden_state = input;

        if let Some((conv, conv_norm)) = &self.conv {
            // TODO: Fused add norm
            // TODO: Residual in f32

            // let residual = add_opt(hidden_state, residual);

            residual = hidden_state + residual;
            hidden_state = conv_norm.forward(residual.clone());
            hidden_state = conv.forward(hidden_state);
        };

        residual = hidden_state + residual;
        hidden_state = self.seq_norm.forward(residual.clone());
        hidden_state = self.ttt.forward(hidden_state, inner, state, start_idx);

        residual = hidden_state + residual;
        hidden_state = self.ffn_norm.forward(residual.clone());
        hidden_state = self.swi_glu_mlp.forward(hidden_state);

        (hidden_state, residual)
    }
}
