use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{Device, Tensor},
};
use tokenizers::Tokenizer;

fn compute<B: Backend>() {
    let device = Default::default();

    let tensor1 = Tensor::<B, 2>::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);

    let tensor3 = tensor1 + tensor2;

    println!("{:?}", tensor3);
}

fn main() {
    compute::<burn::backend::Wgpu>();

    let tok = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf", None).unwrap();
}

mod ttt;
