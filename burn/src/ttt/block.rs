use std::sync::Arc;

use crate::ttt::cubecl_kernels::backend::FusedTttBackend;
use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::{RmsNorm, RmsNormConfig},
    tensor::Tensor,
};

use super::{
    TTTConfig,
    layer::{TTT, TTTInnerModel},
    util::{CausalConv, CausalConvConfig, SwiGluMlp, SwiGluMlpConfig},
};

// We can't write the trait bound due to a limitation of the Module derive macro
// but impls enforce it
//   Inner: TTTInnerModel<B>
#[derive(Module, Debug)]
pub struct TTTBlockWithSeq<B: FusedTttBackend, Inner> {
    pub block: TTTBlock<B>,
    pub inner: Inner,
}

impl<B: FusedTttBackend, Inner: TTTInnerModel<B>> TTTBlockWithSeq<B, Inner> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut Inner::State,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        self.block.forward(input, state, &self.inner, start_idx)
    }

    pub fn init_state(&self, batch_size: usize) -> Inner::State {
        self.inner.init_state(batch_size)
    }
}

#[derive(Module, Debug)]
pub struct TTTBlock<B: FusedTttBackend> {
    pub layer_idx: usize,
    pub conv: Option<(CausalConv<B>, RmsNorm<B>)>,
    pub config: Ignored<Arc<TTTConfig>>,
    pub seq_norm: RmsNorm<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ttt: TTT<B>,
    pub swi_glu_mlp: SwiGluMlp<B>,
}

#[derive(Config, Debug)]
pub struct TTTBlockConfig {
    ttt_config: Arc<TTTConfig>,
    layer_idx: usize,
}

impl TTTBlockConfig {
    pub fn init<B: FusedTttBackend>(self, device: &B::Device) -> TTTBlock<B> {
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

    pub fn init_with_inner<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
        self,
        inner: Inner,
        device: &B::Device,
    ) -> TTTBlockWithSeq<B, Inner> {
        let block = self.init(device);
        TTTBlockWithSeq { block, inner }
    }
}

impl<B: FusedTttBackend> TTTBlock<B> {
    pub fn forward<Inner: TTTInnerModel<B>>(
        &self,
        input: Tensor<B, 3>,
        state: &mut Inner::State,
        inner: &Inner,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        let mut hidden_states = input;

        if let Some((conv, conv_norm)) = &self.conv {
            let residual = hidden_states.clone();
            hidden_states = conv_norm.forward(hidden_states);
            hidden_states = conv.forward(hidden_states);
            hidden_states = residual + hidden_states;
        }

        let residual = hidden_states.clone();
        hidden_states = self.seq_norm.forward(hidden_states);
        hidden_states = self.ttt.forward(hidden_states, inner, state, start_idx);
        hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        hidden_states = self.ffn_norm.forward(hidden_states);
        hidden_states = self.swi_glu_mlp.forward(hidden_states);
        hidden_states = residual + hidden_states;

        hidden_states
    }
}
