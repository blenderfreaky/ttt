#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "erroneous false positives on #[cube] functions"
)]
#![allow(
    clippy::used_underscore_binding,
    clippy::pub_underscore_fields,
    reason = "False positive on Module derive"
)]
#![allow(non_camel_case_types, non_snake_case)]

use std::marker::PhantomData;

use burn::prelude::*;

use crate::ttt::layer::TTTInnerModel;

pub mod backend;
pub mod bundle;
pub mod gelu;
mod gelu_tanh;
pub mod impls;
pub mod kernel;
pub mod linear_fused;
pub mod linear_fused_tile;
pub mod ttt;
pub mod util;

#[cfg(test)]
pub mod test_utils;

// Re-export commonly used items
pub use backend::FusedTttBackend;
pub use bundle::TensorBundle;
pub use kernel::{FusedKernel, FusedKernelBackend};
pub use linear_fused::fused_ttt_forward;
pub use ttt::{TttInputs, TttKernel, TttOutputs};

use crate::ttt::linear::TTTLinear;

// ============================================================================
// Kernel marker types
// ============================================================================

/// Marker for the basic fused TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearKernel;

/// Marker for the tiled fused TTT-Linear kernel (single-stage).
/// Processes one mini-batch per kernel launch.
#[derive(Debug, Clone, Copy, Default)]
pub struct TileKernel;

/// Marker for the multi-stage tiled fused TTT-Linear kernel.
/// Processes all mini-batches in a single kernel launch.
#[derive(Debug, Clone, Copy, Default)]
pub struct TileMultiKernel;

/// Marker for the streaming tiled fused TTT-Linear kernel.
/// Runs a persistent kernel on the GPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingKernel;

/// Marker for the pointer-based streaming TTT-Linear kernel.
/// Uses zero-copy pointer tables for input.
#[derive(Debug, Clone, Copy, Default)]
pub struct PtrStreamingKernel;

// ============================================================================
// Type aliases
// ============================================================================

/// Basic fused TTT-Linear kernel.
pub type FusedLinear<B> = Fused<B, TTTLinear<B>, LinearKernel>;

/// Tiled fused TTT-Linear kernel (single-stage).
pub type FusedTile<B> = Fused<B, TTTLinear<B>, TileKernel>;

/// Multi-stage tiled fused TTT-Linear kernel.
pub type FusedTileMulti<B> = Fused<B, TTTLinear<B>, TileMultiKernel>;

/// Streaming tiled fused TTT-Linear kernel.
#[cfg(feature = "rocm")]
pub type FusedTileStreaming<B> = Fused<B, TTTLinear<B>, StreamingKernel>;

/// Pointer-based streaming TTT-Linear kernel.
#[cfg(feature = "rocm")]
pub type FusedPtrStreaming<B> = Fused<B, TTTLinear<B>, PtrStreamingKernel>;

// ============================================================================
// Fused wrapper struct
// ============================================================================

/// Wrapper for fused TTT layers.
/// The `Kernel` marker type determines which kernel implementation is used.
#[derive(Debug)]
pub struct Fused<B: Backend, Inner, Kernel> {
    pub inner: Inner,
    _phantom: PhantomData<(B, Kernel)>,
}

impl<B: Backend, Inner: Clone, Kernel> Clone for Fused<B, Inner, Kernel> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, Inner: burn::module::Module<B>, Kernel: Send + Sync + std::fmt::Debug>
    burn::module::Module<B> for Fused<B, Inner, Kernel>
{
    type Record = Inner::Record;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        self.inner.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self::new(self.inner.fork(device))
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self::new(self.inner.to_device(device))
    }

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.inner.visit(visitor);
    }

    fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self::new(self.inner.map(mapper))
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self::new(self.inner.load_record(record))
    }

    fn into_record(self) -> Self::Record {
        self.inner.into_record()
    }
}

impl<B: Backend, Inner: burn::module::ModuleDisplay, Kernel> burn::module::ModuleDisplay
    for Fused<B, Inner, Kernel>
{
    fn custom_settings(&self) -> Option<burn::module::DisplaySettings> {
        self.inner.custom_settings()
    }

    fn custom_content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        self.inner.custom_content(content)
    }
}

impl<B: Backend, Inner: burn::module::ModuleDisplayDefault, Kernel>
    burn::module::ModuleDisplayDefault for Fused<B, Inner, Kernel>
{
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        self.inner.content(content)
    }
}

impl<
    B: burn::tensor::backend::AutodiffBackend,
    Inner: burn::module::AutodiffModule<B>,
    Kernel: Send + Sync + std::fmt::Debug + Clone,
> burn::module::AutodiffModule<B> for Fused<B, Inner, Kernel>
{
    type InnerModule = Fused<B::InnerBackend, Inner::InnerModule, Kernel>;

    fn valid(&self) -> Self::InnerModule {
        Fused::new(self.inner.valid())
    }
}

impl<B: Backend, Inner, Kernel> Fused<B, Inner, Kernel> {
    /// Create a new Fused wrapper with the given inner model.
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T: TTTInnerModel<B>, K> From<T> for Fused<B, T, K> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

/// Scaling factor for converting epsilon to integer representation.
/// CubeCL requires comptime configs to implement Eq + Hash, but f32 doesn't implement these.
const EPSILON_SCALE_INV: f32 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedTttConfig {
    /// Mini-batch sequence length (CS)
    pub mini_batch_len: usize,
    /// Head dimension (F)
    pub head_dim: usize,
    /// Layer norm epsilon, stored as scaled integer (see EPSILON_SCALE_INV)
    pub epsilon_scaled: u32,
    /// Number of threads (subtiles) per cube
    pub threads: usize,
}

impl FusedTttConfig {
    #[must_use]
    pub fn new(mini_batch_len: usize, head_dim: usize, epsilon: f32, threads: usize) -> Self {
        Self {
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / EPSILON_SCALE_INV) as u32,
            threads,
        }
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * EPSILON_SCALE_INV
    }
}
