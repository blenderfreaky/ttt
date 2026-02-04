use std::sync::Arc;
use serde::{Serialize, Deserialize};
use ttt_common::{ModelArch, TTTConfig};

/// Combined model config for inner model implementations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: Arc<ModelArch>,
    pub ttt: Arc<TTTConfig>,
}

impl ModelConfig {
    pub fn new(arch: Arc<ModelArch>, ttt: Arc<TTTConfig>) -> Self {
        Self { arch, ttt }
    }

    pub fn head_dim(&self) -> usize {
        self.arch.head_dim()
    }
}

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
