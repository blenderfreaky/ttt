use std::sync::Arc;

use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};
use tokenizers::Tokenizer;
use ttt::{
    linear::{TTTLinear, TTTLinearConfig},
    TTTConfig,
};

fn compute<B: Backend>() {
    let device = Default::default();

    // let tmp: Tensor<B, 2> = Tensor::from_data([[1., 1.], [1., 1.]], &device);
    // dbg!(tmp.clone().sum_dim(1));

    let config = TTTConfig::new()
        .with_token_size(2048)
        .with_hidden_size(2048)
        .with_swi_glu_mlp_intermediate_size(5504)
        // .with_num_hidden_layers(24)
        // .with_num_heads(32)
        .with_num_hidden_layers(2)
        .with_num_heads(2)
        // .with_num_heads(4)
        .with_conv_before_ttt(true);
    let config = Arc::new(config);

    let inner_config = Arc::new(TTTLinearConfig::new());

    let model = config.init_with_inner_model::<_, TTTLinear<_>>(&inner_config, &device);

    let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf", None).unwrap();

    let tokens = tokenizer.encode("Hello, world!", true).unwrap();
    let token_ids = Tensor::<B, 1, Int>::from_ints(tokens.get_ids(), &device);
    let token_ids = token_ids.unsqueeze();

    let logits = model.forward(token_ids, 0);

    dbg!(logits);
}

fn main() {
    compute::<burn::backend::Wgpu>();
}

mod data;
mod ttt;
