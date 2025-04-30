use std::sync::Arc;

use burn::{prelude::Backend, tensor::Tensor};
use tokenizers::Tokenizer;
use ttt::{linear::TTTLinearConfig, TTTConfig};

fn compute<B: Backend>() {
    let device = Default::default();

    let config = TTTConfig::new()
        .with_token_size(2048)
        .with_value_size(2048)
        .with_swi_glu_mlp_intermediate_size(5504)
        .with_num_hidden_layers(24)
        .with_num_heads(32)
        .with_conv_before_ttt(true);
    let config = Arc::new(config);

    let inner_config = Arc::new(TTTLinearConfig::new());

    let model = config.init_with_inner_model(&inner_config, device);

    let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf", None).unwrap();

    let tokens = tokenizer.encode("Hello, world!", true).unwrap();
    let token_ids = tokens.get_ids();

    let logits = model.forward(tokens, 0);

    dbg!(logits);
}

fn main() {
    compute::<burn::backend::Wgpu>();
}

mod data;
mod ttt;
