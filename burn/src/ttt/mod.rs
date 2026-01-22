use burn::config::Config;

pub mod block;
pub mod cubecl_kernels;
pub mod layer;
pub mod linear;
pub mod linear_adam;
pub mod lm;
pub mod mlp;
pub mod mlp2;
pub mod mlp3;
pub mod mlp4;
pub mod util;

// Central backend type aliases - selected via feature flags (rocm, cuda, wgpu)
#[cfg(all(feature = "rocm", not(feature = "bf16")))]
pub type GpuBackend = burn::backend::Rocm<f32>;
#[cfg(all(feature = "rocm", feature = "bf16"))]
pub type GpuBackend = burn::backend::Rocm<half::bf16>;

#[cfg(feature = "cuda")]
pub type GpuBackend = burn::backend::Cuda;

#[cfg(feature = "wgpu")]
pub type GpuBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
compile_error!("One of the features 'rocm', 'cuda', or 'wgpu' must be enabled");

pub type GpuAutodiffBackend = burn::backend::Autodiff<GpuBackend>;
pub type TrainingBackend = burn::backend::Autodiff<
    GpuBackend,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
// Reference CPU backend for tests
pub type CpuBackend = burn::backend::NdArray;

/// TTT Layer Type variants
#[derive(Config, Debug, PartialEq, Copy)]
pub enum TTTLayerType {
    Linear,
    LinearAdam,
    MLP,
    MLP2,
    MLP3,
    MLP4,
    FusedLinear,
}

#[macro_export]
macro_rules! dispatch_ttt_layer_type {
    ($f:ident :: < $backend:ident, $ty:expr, $($other:ty),+ > ($($args:expr),* $(,)?)) => {
        match $ty {
            TTTLayerType::Linear => $f::<$backend, $crate::ttt::linear::TTTLinear<$backend>, $($other),+>($($args),*),
            TTTLayerType::LinearAdam => {
                $f::<$backend, $crate::ttt::linear_adam::TTTLinearAdam<$backend>, $($other),+>($($args),*)
            }
            TTTLayerType::MLP => $f::<$backend, $crate::ttt::mlp::TTTMLP<$backend>, $($other),+>($($args),*),
            TTTLayerType::MLP2 => $f::<$backend, $crate::ttt::mlp2::TTTMLP2<$backend>, $($other),+>($($args),*),
            TTTLayerType::MLP3 => $f::<$backend, $crate::ttt::mlp3::TTTMLP3<$backend>, $($other),+>($($args),*),
            TTTLayerType::MLP4 => $f::<$backend, $crate::ttt::mlp4::TTTMLP4<$backend>, $($other),+>($($args),*),
            TTTLayerType::FusedLinear => $f::<
                $backend,
                $crate::ttt::cubecl_kernels::Fused<
                    $backend,
                    $crate::ttt::linear::TTTLinear<$backend>,
                >,
                $($other),+
            >($($args),*),
        }
    };
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
#[derive(Config, Debug)]
pub struct TTTConfig {
    /// The size of token vectors.
    #[config(default = 2048)]
    pub token_size: usize,
    /// The size of key, value, etc. across all heads.
    /// In source it seems to be equal to token_size
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
    // #[config(default = 32000)]
    #[config(default = 50257)]
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
}

impl TTTConfig {
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    // Default sized from reference
    // TTT_STANDARD_CONFIGS = {
    //     "125m": {
    //         "hidden_size": 768,
    //         "intermediate_size": 2048,
    //         "num_hidden_layers": 12,
    //         "num_attention_heads": 12,
    //     },
    //     "350m": {
    //         "hidden_size": 1024,
    //         "intermediate_size": 2736,
    //         "num_hidden_layers": 24,
    //         "num_attention_heads": 16,
    //     },
    //     "760m": {
    //         "hidden_size": 1536,
    //         "intermediate_size": 4096,
    //         "num_hidden_layers": 24,
    //         "num_attention_heads": 16,
    //     },
    //     "1b": {
    //         "hidden_size": 2048,
    //         "intermediate_size": 5504,
    //         "num_hidden_layers": 24,
    //         "num_attention_heads": 32,
    //     },
    // }
    //
    //
    // def __init__(
    //     self,
    //     vocab_size=32000,
    //     hidden_size=2048,
    //     intermediate_size=5504,
    //     num_hidden_layers=24,
    //     num_attention_heads=32,
    //     hidden_act="silu",
    //     max_position_embeddings=2048,
    //     initializer_range=0.02,
    //     rms_norm_eps=1e-6,
    //     use_cache=False,
    //     pad_token_id=None,
    //     bos_token_id=1,
    //     eos_token_id=2,
    //     pretraining_tp=1,
    //     tie_word_embeddings=True,
    //     rope_theta=10000.0,
    //     use_gate=False,
    //     share_qk=False,
    //     ttt_layer_type="linear",
    //     ttt_base_lr=1.0,
    //     mini_batch_size=16,
    //     pre_conv=False,
    //     conv_kernel=4,
    //     scan_checkpoint_group_size=0,
    //     **kwargs,
    // ):

    // pub fn default() -> Self {
    //     Self {
    //         vocab_size: 32000,
    //         hidden_size: 2048,
    //         // Same as hidden_size in reference
    //         token_size: 2048,
    //         swi_glu_mlp_intermediate_size: 5504,
    //         num_hidden_layers: 24,
    //         num_heads: 32,
    //         // hidden_act="silu",
    //         max_sequence_len: 2048,
    //         // initializer_range=0.02,
    //         // rms_norm_eps: 1e-6,
    //         epsilon: 1e-6,
    //         // use_cache: False,
    //         // pad_token_id: None,
    //         // bos_token_id: 1,
    //         // eos_token_id: 2,
    //         // pretraining_tp: 1,
    //         // tie_word_embeddings: True,
    //         rope_theta: 10000.0,
    //         use_gate: false,
    //         // share_qk:False,
    //         layer_type: TTTLayerType::Linear,
    //         base_lr: 1.0,
    //         mini_batch_size: 16,
    //         conv_before_ttt: false,
    //         conv_kernel_size: 4,
    //         // scan_checkpoint_group_size:0,
    //     }
    // }

    #[must_use]
    pub fn default_12m() -> Self {
        Self::new()
            .with_hidden_size(256)
            .with_token_size(256)
            .with_swi_glu_mlp_intermediate_size(512)
            .with_num_hidden_layers(6)
            .with_num_heads(4)
    }

    #[must_use]
    pub fn default_60m() -> Self {
        Self::new()
            .with_hidden_size(512)
            .with_token_size(512)
            .with_swi_glu_mlp_intermediate_size(768)
            .with_num_hidden_layers(6)
            .with_num_heads(8)
    }

    #[must_use]
    pub fn default_125m() -> Self {
        Self::new()
            .with_hidden_size(768)
            .with_token_size(768)
            .with_swi_glu_mlp_intermediate_size(2048)
            .with_num_hidden_layers(12)
            .with_num_heads(12)
    }

    #[must_use]
    pub fn default_350m() -> Self {
        Self::new()
            .with_hidden_size(1024)
            .with_token_size(1024)
            .with_swi_glu_mlp_intermediate_size(2736)
            .with_num_hidden_layers(24)
            .with_num_heads(16)
    }

    #[must_use]
    pub fn default_760m() -> Self {
        Self::new()
            .with_hidden_size(1536)
            .with_token_size(1536)
            .with_swi_glu_mlp_intermediate_size(4096)
            .with_num_hidden_layers(24)
            .with_num_heads(16)
    }

    #[must_use]
    pub fn default_1b() -> Self {
        Self::new()
            .with_hidden_size(2048)
            .with_token_size(2048)
            .with_swi_glu_mlp_intermediate_size(5504)
            .with_num_hidden_layers(24)
            .with_num_heads(32)
    }
}
