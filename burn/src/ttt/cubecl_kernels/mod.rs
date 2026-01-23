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

// Re-export commonly used items
pub use backend::FusedTttBackend;
pub use bundle::TensorBundle;
pub use kernel::{FusedKernel, FusedKernelBackend};
pub use linear_fused::fused_ttt_forward;
pub use linear_fused_tile::FusedTttTileBackend;
pub use ttt::{TttInputs, TttKernel, TttOutputs};

use crate::ttt::linear::TTTLinear;

/// Type alias for the tiled fused TTT-Linear kernel.
/// Uses double Fused wrapper to distinguish from the non-tiled fused version.
/// Currently supports: seq_len=16, head_dim=64
pub type FusedTile<B> = Fused<B, Fused<B, TTTLinear<B>>>;

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
}

impl FusedTttConfig {
    #[must_use]
    pub fn new(mini_batch_len: usize, head_dim: usize, epsilon: f32) -> Self {
        Self {
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / EPSILON_SCALE_INV) as u32,
        }
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * EPSILON_SCALE_INV
    }
}
