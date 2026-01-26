//! Unified backend trait for fused TTT kernels.
//!
//! The `FusedTttBackend` trait combines all TTT kernel capabilities:
//! - Regular fused TTT kernel
//! - Tiled fused TTT kernel
//! - Multi-stage tiled kernel
//! - GELU activation kernels

use crate::ttt::cubecl_kernels::gelu::{GeluBwdKernel, GeluTanhKernel};
use crate::ttt::cubecl_kernels::kernel::FusedKernelBackend;
use crate::ttt::cubecl_kernels::linear_fused_tile::{TttTileKernel, TttTileMultiKernel};
use crate::ttt::cubecl_kernels::ttt::TttKernel;

/// This includes:
/// - `TttKernel` - the regular fused TTT kernel
/// - `TttTileKernel` - the tiled fused TTT kernel (single mini-batch)
/// - `TttTileMultiKernel` - multi-stage tiled kernel (full sequence)
/// - `GeluTanhKernel` / `GeluBwdKernel` - GELU activation kernels
pub trait FusedTttBackend:
    FusedKernelBackend<TttKernel, 9, 3>
    + FusedKernelBackend<TttTileKernel, 9, 10>
    + FusedKernelBackend<TttTileMultiKernel, 9, 10>
    + FusedKernelBackend<GeluTanhKernel, 1, 1>
    + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

impl<B> FusedTttBackend for B where
    B: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<TttTileKernel, 9, 10>
        + FusedKernelBackend<TttTileMultiKernel, 9, 10>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

/// API module with tensor-level functions for backward compatibility.
pub mod api {
    pub use crate::ttt::cubecl_kernels::gelu::{gelu_bwd, gelu_tanh};
    pub use crate::ttt::cubecl_kernels::linear_fused::fused_ttt_forward;
}
