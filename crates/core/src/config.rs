use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

#[cfg(all(feature = "rocm"))]
pub type GpuBackend<F = DType> = burn::backend::Rocm<F>;

#[cfg(all(feature = "cuda"))]
pub type GpuBackend<F = DType> = burn::backend::Cuda<F>;

#[cfg(feature = "wgpu")]
pub type GpuBackend<F = DType> = burn::backend::Wgpu<F>;

#[cfg(feature = "cpu")]
pub type GpuBackend<F = DType> = burn::backend::Cpu<F>;

#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu", feature = "cpu")))]
pub type GpuBackend<F = DType> =
    compile_error!("One of the features 'rocm', 'cuda', 'wgpu' or 'cpu' must be enabled");

#[cfg(feature = "bf16")]
pub type DType = bf16;

#[cfg(feature = "f16")]
pub type DType = f16;

#[cfg(feature = "f32")]
pub type DType = f32;

#[cfg(not(any(feature = "bf16", feature = "f16", feature = "f32")))]
pub type DType = compile_error!("One of the features 'bf16', 'f16' or 'f32' must be enabled");

pub type GpuAutodiffBackend<F = DType> = burn::backend::Autodiff<GpuBackend<F>>;
pub type TrainingBackend<F = DType> = burn::backend::Autodiff<
    GpuBackend<F>,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

/// Default vocab size for tests (GPT-2 tokenizer size)
pub const TEST_VOCAB_SIZE: usize = 50257;
