use std::sync::Arc;

use burn::{
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Initializer, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use super::{
    TTTConfig,
    block::{TTTBlockConfig, TTTBlockWithSeq},
    layer::TTTInnerModel,
};

#[derive(Module, Debug)]
pub struct TTTModel<B: Backend, Inner> {
    pub config: Ignored<Arc<TTTConfig>>,
    pub embedding: Embedding<B>,
    pub layers: Vec<TTTBlockWithSeq<B, Inner>>,
    pub norm: RmsNorm<B>,
}

impl TTTConfig {
    pub fn init_with_inner_model<B: Backend, Inner: TTTInnerModel<B>>(
        self: &Arc<Self>,
        inner_config: &Arc<Inner::Config>,
        device: &B::Device,
    ) -> TTTModel<B, Inner> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.hidden_size)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);
        let layers = (0..self.num_hidden_layers)
            .map(|idx| {
                TTTBlockConfig::new(self.clone(), idx)
                    .init_with_inner(Inner::new(self, inner_config, device), device)
            })
            .collect();
        let norm = RmsNormConfig::new(self.hidden_size).init(device);

        TTTModel {
            config: Ignored(self.clone()),
            embedding,
            layers,
            norm,
        }
    }
}

impl<B: Backend, Inner: TTTInnerModel<B>> TTTModel<B, Inner> {
    /// Initialize fresh states for all layers
    pub fn init_states(&self, batch_size: usize) -> Vec<Inner::State> {
        self.layers
            .iter()
            .map(|x| x.init_state(batch_size))
            .collect()
    }

    /// Forward pass with fresh states (convenience method that allocates states)
    pub fn forward(&self, input: Tensor<B, 2, Int>, start_idx: usize) -> Tensor<B, 3> {
        let [batch_size, _seq_len] = input.shape().dims();
        let mut states = self.init_states(batch_size);
        self.forward_with_states(input, start_idx, &mut states)
    }

    /// Forward pass with external states (for generation with state persistence)
    pub fn forward_with_states(
        &self,
        input: Tensor<B, 2, Int>,
        start_idx: usize,
        states: &mut [Inner::State],
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(input);

        let mut hidden_states = embedded;

        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            hidden_states = layer.forward(hidden_states, state, start_idx);
        }

        hidden_states = self.norm.forward(hidden_states);

        // Output projection using tied embedding weights
        // Embedding weight is [vocab_size, hidden_size], we need [1, hidden_size, vocab_size] for matmul
        let weight = self.embedding.weight.val(); // [vocab_size, hidden_size]
        let weight = weight.unsqueeze_dim::<3>(0); // [1, vocab_size, hidden_size]
        let weight = weight.permute([0, 2, 1]); // [1, hidden_size, vocab_size]
        hidden_states = hidden_states.matmul(weight);

        hidden_states
    }
}
