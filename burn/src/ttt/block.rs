use std::sync::Arc;

use burn::{
    config::Config,
    module::{Ignored, Module},
    nn::RmsNorm,
    prelude::Backend,
    tensor::Tensor,
};

use super::{
    layer::{TTTConfig, TTTInnerModel, TTT},
    util::{CausalConv, SwiGluMlp},
};

// We can't write the trait bound due to a limitation of the Module derive macro
// but impls enforce it
//   Inner: TTTInnerModel<B>
#[derive(Module, Debug)]
pub struct Block<B: Backend, Inner> {
    layer_idx: usize,
    conv: Option<(CausalConv<B>, RmsNorm<B>)>,
    // conv_norm: Option<RmsNorm<B>>,
    config: Ignored<Arc<TTTConfig>>,
    seq_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,
    ttt: TTT<B>,
    inner: Inner,
    swi_glu_mlp: SwiGluMlp<B>,
}

#[derive(Config, Debug)]
pub struct BlockConfig {}

// fn add_opt<B: Backend>(x: Tensor<B, 3>, opt: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
//     match opt {
//         Some(a) => x + a,
//         None => x,
//     }
// }

impl<B: Backend, Inner: TTTInnerModel<B>> Block<B, Inner> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        residual: Option<Tensor<B, 3>>,
        state: &mut Inner::State,
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
        hidden_state = self
            .ttt
            .forward(hidden_state, &self.inner, state, start_idx);

        residual = hidden_state + residual;
        hidden_state = self.ffn_norm.forward(residual.clone());
        hidden_state = self.swi_glu_mlp.forward(hidden_state);

        (hidden_state, residual)
    }
}
