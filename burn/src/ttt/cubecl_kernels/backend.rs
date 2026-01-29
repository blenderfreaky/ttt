//! Unified backend trait for fused TTT kernels.
//!
//! The `FusedTttBackend` trait combines all TTT kernel capabilities:
//! - Regular fused TTT kernel
//! - Tiled fused TTT kernel
//! - Multi-stage tiled kernel
//! - GELU activation kernels

use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_fusion::{Fusion, FusionBackend};

use crate::ttt::cubecl_kernels::{
    gelu::{GeluBwdKernel, GeluTanhKernel},
    kernel::FusedKernelBackend,
    linear_fused_tile::{TttTileKernel, TttTileMultiKernel},
    ttt::TttKernel,
};

#[cfg(feature = "rocm")]
use crate::ttt::cubecl_kernels::linear_fused_tile::TttStreamingKernel;

/// Unified backend trait for fused TTT kernels.
///
/// This includes:
/// - `TttKernel` - the regular fused TTT kernel
/// - `TttTileKernel` - the tiled fused TTT kernel (single mini-batch)
/// - `TttTileMultiKernel` - multi-stage tiled kernel (full sequence)
/// - `TttStreamingKernel` - streaming kernel (rocm only)
/// - `GeluTanhKernel` / `GeluBwdKernel` - GELU activation kernels
///
/// Also exposes the underlying runtime types for streaming kernel support.
#[cfg(feature = "rocm")]
pub trait FusedTttBackend:
    FusedKernelBackend<TttKernel, 9, 3>
    + FusedKernelBackend<TttTileKernel, 9, 10>
    + FusedKernelBackend<TttTileMultiKernel, 9, 10>
    + FusedKernelBackend<TttStreamingKernel, 9, 10>
    + FusedKernelBackend<GeluTanhKernel, 1, 1>
    + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

/// Unified backend trait for fused TTT kernels (non-rocm version).
#[cfg(not(feature = "rocm"))]
pub trait FusedTttBackend:
    FusedKernelBackend<TttKernel, 9, 3>
    + FusedKernelBackend<TttTileKernel, 9, 10>
    + FusedKernelBackend<TttTileMultiKernel, 9, 10>
    + FusedKernelBackend<GeluTanhKernel, 1, 1>
    + FusedKernelBackend<GeluBwdKernel, 1, 1>
{
}

/// Implementation for CubeBackend (the base JIT backend).
impl<R, F, I, BT> FusedTttBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}

/// Implementation for Fusion wrapper (rocm version with streaming).
#[cfg(feature = "rocm")]
impl<B> FusedTttBackend for Fusion<B>
where
    B: FusedTttBackend + FusionBackend,
    Self: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<TttTileKernel, 9, 10>
        + FusedKernelBackend<TttTileMultiKernel, 9, 10>
        + FusedKernelBackend<TttStreamingKernel, 9, 10>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>,
{
}

/// Implementation for Fusion wrapper (non-rocm version).
#[cfg(not(feature = "rocm"))]
impl<B> FusedTttBackend for Fusion<B>
where
    B: FusedTttBackend + FusionBackend,
    Self: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<TttTileKernel, 9, 10>
        + FusedKernelBackend<TttTileMultiKernel, 9, 10>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>,
{
}

/// Implementation for Autodiff wrapper (rocm version with streaming).
#[cfg(feature = "rocm")]
impl<B, C> FusedTttBackend for Autodiff<B, C>
where
    B: FusedTttBackend,
    C: CheckpointStrategy,
    Self: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<TttTileKernel, 9, 10>
        + FusedKernelBackend<TttTileMultiKernel, 9, 10>
        + FusedKernelBackend<TttStreamingKernel, 9, 10>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>,
{
}

/// Implementation for Autodiff wrapper (non-rocm version).
#[cfg(not(feature = "rocm"))]
impl<B, C> FusedTttBackend for Autodiff<B, C>
where
    B: FusedTttBackend,
    C: CheckpointStrategy,
    Self: FusedKernelBackend<TttKernel, 9, 3>
        + FusedKernelBackend<TttTileKernel, 9, 10>
        + FusedKernelBackend<TttTileMultiKernel, 9, 10>
        + FusedKernelBackend<GeluTanhKernel, 1, 1>
        + FusedKernelBackend<GeluBwdKernel, 1, 1>,
{
}

/// API module with tensor-level functions for backward compatibility.
pub mod api {
    pub use crate::ttt::cubecl_kernels::{
        gelu::{gelu_bwd, gelu_tanh},
        linear_fused::fused_ttt_forward,
    };
}
