use burn::config::Config;

pub mod block;
pub mod layer;
pub mod linear;
pub mod lm;
pub mod util;

/// Configuration for the TTT layer.
#[derive(Config, Debug)]
pub struct TTTConfig {
    /// The size of token vectors.
    #[config(default = 4096)]
    pub token_size: usize,
    // /// The size of key and query vectors.
    // /// In source it seems to be token_size/2
    // key_size: usize,
    /// The size of value vectors.
    /// In source it seems to be token_size
    #[config(default = 4096)]
    pub value_size: usize,
    /// The number of TTT heads.
    #[config(default = 32)]
    pub num_heads: usize,
    /// The kernel size for the convolutional layers.
    #[config(default = 4)]
    pub conv_kernel_size: usize,
    /// The mini batch size.
    #[config(default = 16)]
    pub mini_batch_size: usize,
    // TODO: Make positional encoding generic/exchangable for different types of positional encodings
    /// The maximum sequence length (only used for rotary encoding).
    #[config(default = 2048)]
    pub max_sequence_len: usize,
    /// The theta value for the rotary encoding.
    #[config(default = 10000.0)]
    pub rope_theta: f32,
    /// The base learning rate for the TTT module.
    #[config(default = 1.0)]
    pub base_lr: f32,
    #[config(default = 1e-6)]
    pub epsilon: f64,
    #[config(default = false)]
    pub conv_before_ttt: bool,
    #[config(default = 11008)]
    pub swi_glu_mlp_intermediate_size: usize,
    #[config(default = 32)]
    pub num_hidden_layers: usize,
    #[config(default = 50277)]
    pub vocab_size: usize,
}
