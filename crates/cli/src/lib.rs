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
    clippy::type_complexity
)]

//! TTT - Text-to-Text Transformer with Test-Time Training
//!
//! This is the main facade crate that re-exports commonly used items
//! from all sub-crates and provides the CLI binary.

pub mod artifact_info;
pub mod metrics_export;

// Re-export core types
pub use ttt_core::{
    GpuAutodiffBackend, GpuBackend, InnerModel, PosEncoding, TTTConfig, TrainingBackend,
};
// Re-export inner models
pub use ttt_core::{TTTInnerModel, TTTLinear, TTTLinearAdam, TTTMLP, TTTMLP2, TTTMLP3, TTTMLP4};
// Re-export data types
pub use ttt_data::{
    PreTokenizedDataset, TextDataset, TextGenerationBatcher, Tokenizer, TokenizerTrait,
};
// Re-export fused kernels
pub use ttt_fused::{FusedLinear, FusedTile, FusedTileMulti, FusedTttBackend};
// Re-export layer types
pub use ttt_layer::{TTT, TTTBlock, TTTBlockWithSeq, TTTModel, dispatch_ttt_layer_type};
// Re-export training types
pub use ttt_training::{
    TTTTextGenerationConfig, TTTTextGenerationModel, TTTTextGenerator, TTTTrainingConfig,
};

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu", feature = "cpu")))]
compile_error!(
    "At least one backend must be enabled, please run with `--features cuda/rocm/wgpu/cpu`"
);
