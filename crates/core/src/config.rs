use burn::config::Config;

// Central backend type aliases - selected via feature flags (rocm, cuda, wgpu)
#[cfg(all(feature = "rocm", not(feature = "bf16")))]
pub type GpuBackend = burn::backend::Rocm<f32>;
#[cfg(all(feature = "rocm", feature = "bf16"))]
pub type GpuBackend = burn::backend::Rocm<half::bf16>;

#[cfg(all(feature = "cuda", not(feature = "bf16")))]
pub type GpuBackend = burn::backend::Cuda<f32>;
#[cfg(all(feature = "cuda", feature = "bf16"))]
pub type GpuBackend = burn::backend::Cuda<half::bf16>;

#[cfg(feature = "wgpu")]
pub type GpuBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
pub type GpuBackend =
    compile_error!("One of the features 'rocm', 'cuda', or 'wgpu' must be enabled");

pub type GpuAutodiffBackend = burn::backend::Autodiff<GpuBackend>;
pub type TrainingBackend = burn::backend::Autodiff<
    GpuBackend,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

/// Default vocab size for tests (GPT-2 tokenizer size)
pub const TEST_VOCAB_SIZE: usize = 50257;

/// TTT Layer Type variants
#[allow(clippy::expl_impl_clone_on_copy)]
#[derive(Config, Debug, PartialEq, Copy)]
pub enum TTTLayerType {
    Linear,
    LinearAdam,
    MLP,
    MLP2,
    MLP3,
    MLP4,
    FusedLinear,
    FusedTileLinear,
    FusedTileMultiLinear,
}

#[derive(Config, Debug, PartialEq)]
pub enum PositionEncodingType {
    /// Rotary Position Embeddings applied to Q/K
    RoPE,
    /// No position encoding
    None,
    /// Learned absolute position embeddings
    Absolute,
}

/// Configuration for the TTT layer.
#[allow(clippy::struct_excessive_bools)]
#[derive(Config, Debug)]
pub struct TTTConfig {
    /// The size of token vectors.
    #[config(default = 2048)]
    pub token_size: usize,
    /// The size of key, value, etc. across all heads.
    #[config(default = 2048)]
    pub hidden_size: usize,
    /// The number of TTT heads.
    #[config(default = 32)]
    pub num_heads: usize,
    /// The kernel size for the convolutional layers.
    #[config(default = 4)]
    pub conv_kernel_size: usize,
    /// The mini batch size.
    #[config(default = 16)]
    pub mini_batch_size: usize,
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
    #[config(default = 5504)]
    pub swi_glu_mlp_intermediate_size: usize,
    #[config(default = 24)]
    pub num_hidden_layers: usize,
    /// Vocabulary size - must match the tokenizer's vocab_size.
    /// No default: must be explicitly set to avoid tokenizer/model mismatch.
    pub vocab_size: usize,
    /// The type of TTT layer to use
    #[config(default = "TTTLayerType::Linear")]
    pub layer_type: TTTLayerType,
    /// Whether to use gating (as in Mamba backbone)
    #[config(default = false)]
    pub use_gate: bool,
    /// The type of position encoding to use
    #[config(default = "PositionEncodingType::RoPE")]
    pub pos_encoding: PositionEncodingType,
    /// Maximum sequence length for absolute position embeddings
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// MLP expansion factor (hidden_dim = expansion_factor * head_dim)
    #[config(default = 4)]
    pub mlp_expansion_factor: usize,
    /// Whether to share Q/K projection matrix.
    /// When true: uses single projection + separate conv layers for Q and K (Mamba-style).
    /// When false: uses separate Q and K projection matrices (default).
    #[config(default = false)]
    pub share_qk: bool,
    /// Whether to tie the output projection (lm_head) weights with input embeddings.
    /// When true: uses matmul with transposed embedding weights (weight sharing).
    /// When false: uses separate Linear layer for lm_head .
    #[config(default = true)]
    pub tie_word_embeddings: bool,
    /// Number of threads to use per (batch, head).
    /// Only applies to FusedTile impl. None = auto-detect.
    #[config(default = "None")]
    pub threads: Option<usize>,
}

impl TTTConfig {
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    #[must_use]
    pub fn default_12m(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(256)
            .with_token_size(256)
            .with_swi_glu_mlp_intermediate_size(512)
            .with_num_hidden_layers(6)
            .with_num_heads(4)
    }

    #[must_use]
    pub fn default_60m(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(512)
            .with_token_size(512)
            .with_swi_glu_mlp_intermediate_size(768)
            .with_num_hidden_layers(6)
            .with_num_heads(8)
    }

    #[must_use]
    pub fn default_125m(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(768)
            .with_token_size(768)
            .with_swi_glu_mlp_intermediate_size(2048)
            .with_num_hidden_layers(12)
            .with_num_heads(12)
    }

    #[must_use]
    pub fn default_350m(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(1024)
            .with_token_size(1024)
            .with_swi_glu_mlp_intermediate_size(2736)
            .with_num_hidden_layers(24)
            .with_num_heads(16)
    }

    #[must_use]
    pub fn default_760m(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(1536)
            .with_token_size(1536)
            .with_swi_glu_mlp_intermediate_size(4096)
            .with_num_hidden_layers(24)
            .with_num_heads(16)
    }

    #[must_use]
    pub fn default_1b(vocab_size: usize) -> Self {
        Self::new(vocab_size)
            .with_hidden_size(2048)
            .with_token_size(2048)
            .with_swi_glu_mlp_intermediate_size(5504)
            .with_num_hidden_layers(24)
            .with_num_heads(32)
    }
}
