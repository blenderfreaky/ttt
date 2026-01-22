use std::sync::Arc;

use burn::{
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use super::{
    PositionEncodingType, TTTConfig,
    block::{TTTBlockConfig, TTTBlockWithSeq},
    layer::TTTInnerModel,
};

#[derive(Module, Debug)]
pub struct TTTModel<B: Backend, Inner> {
    pub config: Ignored<Arc<TTTConfig>>,
    pub embedding: Embedding<B>,
    /// Optional absolute position embeddings (only present when pos_encoding is Absolute)
    pub position_embedding: Option<Embedding<B>>,
    pub layers: Vec<TTTBlockWithSeq<B, Inner>>,
    pub norm: RmsNorm<B>,
    /// Optional separate lm_head (only present when tie_word_embeddings is false).
    /// When None, uses tied embedding weights via matmul.
    /// When Some, uses separate Linear layer (more performant, single kernel call).
    pub lm_head: Option<Linear<B>>,
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

        let position_embedding = match self.pos_encoding {
            PositionEncodingType::Absolute => Some(
                EmbeddingConfig::new(self.max_seq_len, self.hidden_size)
                    .with_initializer(Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    })
                    .init(device),
            ),
            _ => None,
        };

        let layers = (0..self.num_hidden_layers)
            .map(|idx| {
                TTTBlockConfig::new(self.clone(), idx)
                    .init_with_inner(Inner::new(self, inner_config, device), device)
            })
            .collect();
        let norm = RmsNormConfig::new(self.hidden_size).init(device);

        // When tie_word_embeddings=false, use separate lm_head (more performant)
        // When tie_word_embeddings=true, lm_head is None and we use tied embedding weights
        let lm_head = if self.tie_word_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(self.hidden_size, self.vocab_size)
                    .with_bias(false)
                    .with_initializer(Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    })
                    .init(device),
            )
        };

        TTTModel {
            config: Ignored(self.clone()),
            embedding,
            position_embedding,
            layers,
            norm,
            lm_head,
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
        let [_batch_size, seq_len] = input.shape().dims();
        let device = input.device();

        let embedded = self.embedding.forward(input);

        // Add absolute position embeddings if present
        let mut hidden_states = match &self.position_embedding {
            Some(pos_emb) => {
                let positions = Tensor::<B, 1, Int>::arange(
                    start_idx as i64..(start_idx + seq_len) as i64,
                    &device,
                )
                .unsqueeze_dim::<2>(0); // [1, seq_len]
                let pos_embedded = pos_emb.forward(positions); // [1, seq_len, hidden_size]
                embedded + pos_embedded
            }
            None => embedded,
        };

        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            hidden_states = layer.forward(hidden_states, state, start_idx);
        }

        hidden_states = self.norm.forward(hidden_states);

        // Output projection: use lm_head if available (more performant), otherwise tied embedding weights
        hidden_states = if let Some(lm_head) = &self.lm_head {
            // Separate lm_head: single optimized kernel call
            lm_head.forward(hidden_states)
        } else {
            // Tied embedding weights: matmul with transposed embedding
            // Embedding weight is [vocab_size, hidden_size], we need [1, hidden_size, vocab_size] for matmul
            let weight = self.embedding.weight.val(); // [vocab_size, hidden_size]
            let weight = weight.unsqueeze_dim::<3>(0); // [1, vocab_size, hidden_size]
            let weight = weight.permute([0, 2, 1]); // [1, hidden_size, vocab_size]
            hidden_states.matmul(weight)
        };

        hidden_states
    }
}
