use burn::{
    module::Module,
    nn::{Embedding, RmsNorm},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use super::{block::TTTBlockWithSeq, layer::TTTInnerModel};

#[derive(Module, Debug)]
pub struct TTTModel<B: Backend, Inner> {
    embedding: Embedding<B>,
    layers: Vec<TTTBlockWithSeq<B, Inner>>,
    norm: RmsNorm<B>,
}

impl<B: Backend, Inner: TTTInnerModel<B>> TTTModel<B, Inner> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        start_idx: usize,
        // states: Option<Vec<Inner::State>>,
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(input);

        let [batch_size, _seq_len] = embedded.shape().dims();

        let mut hidden_states = embedded;
        let mut residual = hidden_states.zeros_like();

        // let states = states.unwrap_or_else(|| {
        let mut states = self
            .layers
            .iter()
            .map(|x| x.init_state(batch_size))
            .collect::<Vec<_>>();
        // });

        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            (hidden_states, residual) =
                layer.forward(hidden_states, Some(residual), state, start_idx);
        }

        residual = hidden_states + residual;
        hidden_states = self.norm.forward(residual);

        // TODO: This is suspect
        hidden_states = self
            .embedding
            .weight
            .val()
            .transpose()
            .unsqueeze()
            .matmul(hidden_states);

        hidden_states
    }
}
