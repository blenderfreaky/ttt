//! Backward compatibility layer for the old backend traits.
//!
//! The old `FusedTttBackend` and `GeluTanhBackend` traits have been replaced
//! with the generic `FusedKernelBackend<K, N, M>` system. This module provides
//! trait aliases and re-exports for compatibility.

use crate::ttt::cubecl_kernels::gelu::{GeluBwdKernel, GeluTanhKernel};
use crate::ttt::cubecl_kernels::kernel::FusedKernelBackend;
use crate::ttt::cubecl_kernels::ttt::TttKernel;

/// Trait alias for backends that support the fused TTT kernel.
pub trait FusedTttBackend:
    FusedKernelBackend<TttKernel, 9, 3>
    + FusedKernelBackend<GeluTanhKernel, 1, 1>
    + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

impl<B> FusedTttBackend for B where
    B: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

/// API module with tensor-level functions for backward compatibility.
pub mod api {
    pub use crate::ttt::cubecl_kernels::gelu::{gelu_bwd, gelu_tanh};
    pub use crate::ttt::cubecl_kernels::linear_fused::fused_ttt_forward;
}
