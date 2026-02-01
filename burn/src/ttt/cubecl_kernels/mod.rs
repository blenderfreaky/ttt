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

/// Type alias for the tiled fused TTT-Linear kernel (single-stage).
/// Uses double Fused wrapper to distinguish from the non-tiled fused version.
/// Processes one mini-batch per kernel launch.
pub type FusedTile<B> = Fused<B, Fused<B, TTTLinear<B>>>;

/// Type alias for the multi-stage tiled fused TTT-Linear kernel.
/// Uses triple Fused wrapper. Processes all mini-batches in a single kernel
/// launch, keeping weight/bias in shared memory between stages.
pub type FusedTileMulti<B> = Fused<B, Fused<B, Fused<B, TTTLinear<B>>>>;

/// Type alias for the streaming tiled fused TTT-Linear kernel.
/// Uses quadruple Fused wrapper. Runs a persistent kernel on the GPU and
/// feeds mini-batches incrementally via async memory transfers.
#[cfg(feature = "rocm")]
pub type FusedTileStreaming<B> = Fused<B, Fused<B, Fused<B, Fused<B, TTTLinear<B>>>>>;

/// Marker type for fused TTT layers.
/// TTTInnerModel is implemented using a fused kernel,
/// but uses the same underlying types as the regular version.
///
/// We can't write the type bound due to a limitation of the Module derive macro,
/// but all impls guarantee:
///     Inner: TTTInnerModel<B>
#[derive(Module, Debug)]
pub struct Fused<B: Backend, Inner> {
    inner: Inner,
    _backend: PhantomData<B>,
}

impl<B: Backend, T: TTTInnerModel<B>> From<T> for Fused<B, T> {
    fn from(inner: T) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
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
