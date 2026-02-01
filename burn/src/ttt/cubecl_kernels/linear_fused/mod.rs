//! Fused TTT-Linear kernel (scalar/per-element implementation).
//!
//! This module contains the original fused TTT-Linear implementation that operates
//! per-element using global memory with thread cooperation within cubes.

mod api;
mod backward;
mod forward;
mod launch;
mod wrapper;

pub use api::fused_ttt_forward;
pub use backward::launch_fused_ttt_backward;
pub use forward::{fused_ttt_forward_kernel, launch_fused_ttt_forward};
pub use launch::{backward, forward};
pub use super::util::empty_like;
