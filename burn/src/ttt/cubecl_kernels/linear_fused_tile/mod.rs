//! Fused TTT-Linear kernel (tiled implementation using thundercube).
//!
//! This module contains the tiled fused TTT-Linear implementation that uses
//! shared memory tiles from the thundercube library for optimized memory access.

mod api;
mod forward;
mod launch;
mod wrapper;

pub use api::fused_ttt_tile_forward;
pub use forward::{
    extract_last_row, fused_ttt_forward_kernel, layer_norm_forward, layer_norm_l2_grad, Inputs,
    Outputs, Params, ParamsTrait,
};
pub use launch::{forward, launch_tile_forward, launch_tile_forward_64x64, TttTileKernel};
pub use wrapper::FusedTttTileBackend;
