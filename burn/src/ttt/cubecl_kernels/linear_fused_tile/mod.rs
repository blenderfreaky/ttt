//! Fused TTT-Linear kernel (tiled implementation using thundercube).
//!
//! This module contains the tiled fused TTT-Linear implementation that uses
//! shared memory tiles from the thundercube library for optimized memory access.

mod api;
mod backward;
mod forward;
mod helpers;
mod launch;
pub mod layer_norm;
mod wrapper;

pub use api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};
pub use forward::{
    ForwardIntermediates, Inputs, Outputs, fused_ttt_forward_kernel,
    fused_ttt_forward_kernel_multi, fused_ttt_forward_stage,
};
pub use launch::{
    TttTileKernel, TttTileMultiKernel, forward, forward_multi, launch_tile_forward,
    launch_tile_forward_multi,
};
pub use layer_norm::{
    SumSqOp, layer_norm_backward, layer_norm_forward, layer_norm_forward_save_intermediates,
    layer_norm_forward_with_intermediates, layer_norm_l2_grad, layer_norm_l2_grad_backward,
    layer_norm_l2_grad_save_intermediates,
};
