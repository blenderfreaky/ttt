#![warn(clippy::pedantic)]
#![allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::default_trait_access,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::type_complexity,
)]

//! TTT Layer - Outer layer implementations
//!
//! This crate provides:
//! - `TTT` - the outer TTT layer that wraps inner model implementations
//! - `TTTBlock`, `TTTBlockWithSeq` - block-level wrappers
//! - `TTTModel` - full language model implementation
//! - `dispatch_ttt_layer_type!` - macro for dispatching based on layer type

pub mod block;
pub mod lm;
pub mod ttt;

// Re-export commonly used items
pub use block::{TTTBlock, TTTBlockConfig, TTTBlockWithSeq};
pub use lm::{TTTModel, TTTConfigModelExt};
pub use ttt::{TTT, TTTConfigExt};

// Re-export TTTLayerType for use with the dispatch macro
pub use ttt_core::TTTLayerType;

/// Dispatch a function call based on TTT layer type.
///
/// This macro takes a function call with a layer type and dispatches to the
/// appropriate inner model implementation.
///
/// # Example
/// ```ignore
/// dispatch_ttt_layer_type!(train_with_inner::<B, layer_type, _>(
///     device,
///     dataset,
///     config,
/// ));
/// ```
#[macro_export]
macro_rules! dispatch_ttt_layer_type {
    ($f:ident :: < $backend:ident, $ty:expr, $($other:ty),+ > ($($args:expr),* $(,)?)) => {
        match $ty {
            $crate::TTTLayerType::Linear => {
                $f::<$backend, ttt_core::TTTLinear<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::LinearAdam => {
                $f::<$backend, ttt_core::TTTLinearAdam<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::MLP => {
                $f::<$backend, ttt_core::TTTMLP<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::MLP2 => {
                $f::<$backend, ttt_core::TTTMLP2<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::MLP3 => {
                $f::<$backend, ttt_core::TTTMLP3<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::MLP4 => {
                $f::<$backend, ttt_core::TTTMLP4<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::FusedLinear => {
                $f::<$backend, ttt_fused::FusedLinear<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::FusedTileLinear => {
                $f::<$backend, ttt_fused::FusedTile<$backend>, $($other),+>($($args),*)
            }
            $crate::TTTLayerType::FusedTileMultiLinear => {
                $f::<$backend, ttt_fused::FusedTileMulti<$backend>, $($other),+>($($args),*)
            }
        }
    };
}
