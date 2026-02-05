//! Common types shared between TTT crates.

#[cfg(feature = "burn")]
pub use burn::config::Config;
use serde::{Deserialize, Serialize};

/// Inner model type for TTT layer.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[serde(rename_all = "kebab-case")]
pub enum InnerModel {
    #[default]
    Linear,
    LinearAdam,
    Mlp,
    Mlp2,
    Mlp3,
    Mlp4,
    FusedLinear,
    FusedTileLinear,
    FusedTileMultiLinear,
    D2dStreamingLinear,
    PtrStreamingLinear,
}

/// Position encoding type.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[serde(rename_all = "kebab-case")]
pub enum PosEncoding {
    #[default]
    Rope,
    RopeGlobal,
    None,
    Absolute,
}

/// Model size presets.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    #[cfg_attr(feature = "clap", value(name = "12m"))]
    #[serde(rename = "12m")]
    M12,
    #[default]
    #[cfg_attr(feature = "clap", value(name = "60m"))]
    #[serde(rename = "60m")]
    M60,
    #[cfg_attr(feature = "clap", value(name = "125m"))]
    #[serde(rename = "125m")]
    M125,
    #[cfg_attr(feature = "clap", value(name = "350m"))]
    #[serde(rename = "350m")]
    M350,
    #[cfg_attr(feature = "clap", value(name = "760m"))]
    #[serde(rename = "760m")]
    M760,
    #[cfg_attr(feature = "clap", value(name = "1b"))]
    #[serde(rename = "1b")]
    B1,
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::M12 => write!(f, "12m"),
            Self::M60 => write!(f, "60m"),
            Self::M125 => write!(f, "125m"),
            Self::M350 => write!(f, "350m"),
            Self::M760 => write!(f, "760m"),
            Self::B1 => write!(f, "1b"),
        }
    }
}

impl std::fmt::Display for InnerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "linear"),
            Self::LinearAdam => write!(f, "linear-adam"),
            Self::Mlp => write!(f, "mlp"),
            Self::Mlp2 => write!(f, "mlp2"),
            Self::Mlp3 => write!(f, "mlp3"),
            Self::Mlp4 => write!(f, "mlp4"),
            Self::FusedLinear => write!(f, "fused-linear"),
            Self::FusedTileLinear => write!(f, "fused-tile-linear"),
            Self::FusedTileMultiLinear => write!(f, "fused-tile-multi-linear"),
            Self::D2dStreamingLinear => write!(f, "d2d-streaming-linear"),
            Self::PtrStreamingLinear => write!(f, "ptr-streaming-linear"),
        }
    }
}

impl std::fmt::Display for PosEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rope => write!(f, "rope"),
            Self::RopeGlobal => write!(f, "rope-global"),
            Self::None => write!(f, "none"),
            Self::Absolute => write!(f, "absolute"),
        }
    }
}

/// Model size presets.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
        }
    }
}
/// Model architecture (derived from ModelSize + vocab_size).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
pub struct ModelArch {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl ModelArch {
    pub fn from_size(size: ModelSize, vocab_size: usize) -> Self {
        match size {
            ModelSize::M12 => Self {
                hidden_size: 256,
                num_hidden_layers: 6,
                num_heads: 4,
                intermediate_size: 512,
                vocab_size,
            },
            ModelSize::M60 => Self {
                hidden_size: 512,
                num_hidden_layers: 6,
                num_heads: 8,
                intermediate_size: 768,
                vocab_size,
            },
            ModelSize::M125 => Self {
                hidden_size: 768,
                num_hidden_layers: 12,
                num_heads: 12,
                intermediate_size: 2048,
                vocab_size,
            },
            ModelSize::M350 => Self {
                hidden_size: 1024,
                num_hidden_layers: 24,
                num_heads: 16,
                intermediate_size: 2736,
                vocab_size,
            },
            ModelSize::M760 => Self {
                hidden_size: 1536,
                num_hidden_layers: 24,
                num_heads: 16,
                intermediate_size: 4096,
                vocab_size,
            },
            ModelSize::B1 => Self {
                hidden_size: 2048,
                num_hidden_layers: 24,
                num_heads: 32,
                intermediate_size: 5504,
                vocab_size,
            },
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// TTT layer behavioral configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TTTConfig {
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "linear"))]
    pub layer_type: InnerModel,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "rope"))]
    pub pos_encoding: PosEncoding,
    #[serde(default = "default_base_lr")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1.0"))]
    pub base_lr: f32,
    #[serde(default = "default_mini_batch_size")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "16"))]
    pub mini_batch_size: usize,
    #[serde(default = "default_max_seq_len")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2048"))]
    pub max_seq_len: usize,
    #[serde(default = "default_mlp_expansion")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "4"))]
    pub mlp_expansion_factor: usize,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub threads: Option<usize>,
    #[serde(default = "default_rope_theta")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10000.0", hide = true))]
    pub rope_theta: f32,
    #[serde(default = "default_conv_kernel")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "4", hide = true))]
    pub conv_kernel_size: usize,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub conv_before_ttt: bool,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub use_gate: bool,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub share_qk: bool,
    #[serde(default = "default_true")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "true", hide = true))]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_epsilon")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1e-6", hide = true))]
    pub epsilon: f64,
    #[serde(default = "default_dtype")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "f32"))]
    pub dtype: DType,
}

fn default_base_lr() -> f32 {
    1.0
}
fn default_mini_batch_size() -> usize {
    16
}
fn default_max_seq_len() -> usize {
    2048
}
fn default_mlp_expansion() -> usize {
    4
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_conv_kernel() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_epsilon() -> f64 {
    1e-6
}
fn default_dtype() -> DType {
    DType::F32
}

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            layer_type: InnerModel::default(),
            pos_encoding: PosEncoding::default(),
            base_lr: default_base_lr(),
            mini_batch_size: default_mini_batch_size(),
            max_seq_len: default_max_seq_len(),
            mlp_expansion_factor: default_mlp_expansion(),
            threads: None,
            rope_theta: default_rope_theta(),
            conv_kernel_size: default_conv_kernel(),
            conv_before_ttt: false,
            use_gate: false,
            share_qk: false,
            tie_word_embeddings: true,
            epsilon: default_epsilon(),
            dtype: default_dtype(),
        }
    }
}

/// Training hyperparameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TrainConfig {
    #[serde(default = "default_batch")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "32"))]
    pub batch: usize,
    #[serde(default = "default_epochs")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10"))]
    pub epochs: usize,
    #[serde(default = "default_lr")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2e-3"))]
    pub lr: f64,
    #[serde(default = "default_samples")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10000"))]
    pub samples: usize,
    #[serde(default = "default_test_samples")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1000"))]
    pub test_samples: usize,
    #[serde(default = "default_grad_accum")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1"))]
    pub grad_accum: usize,
    #[serde(default = "default_workers")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2"))]
    pub workers: usize,
    #[serde(default = "default_warmup_steps")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "5000"))]
    pub warmup_steps: usize,
    // Adam optimizer parameters
    #[serde(default = "default_beta1")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.9", hide = true))]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.999", hide = true))]
    pub beta2: f32,
    #[serde(default = "default_weight_decay")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.01", hide = true))]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip_norm")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1.0", hide = true))]
    pub grad_clip_norm: f32,
}

fn default_batch() -> usize {
    32
}
fn default_epochs() -> usize {
    10
}
fn default_lr() -> f64 {
    2e-3
}
fn default_samples() -> usize {
    10000
}
fn default_test_samples() -> usize {
    1000
}
fn default_grad_accum() -> usize {
    1
}
fn default_workers() -> usize {
    2
}
fn default_warmup_steps() -> usize {
    5000
}
fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_weight_decay() -> f32 {
    0.01
}
fn default_grad_clip_norm() -> f32 {
    1.0
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch: default_batch(),
            epochs: default_epochs(),
            lr: default_lr(),
            samples: default_samples(),
            test_samples: default_test_samples(),
            grad_accum: default_grad_accum(),
            workers: default_workers(),
            warmup_steps: default_warmup_steps(),
            beta1: default_beta1(),
            beta2: default_beta2(),
            weight_decay: default_weight_decay(),
            grad_clip_norm: default_grad_clip_norm(),
        }
    }
}

impl TrainConfig {
    pub fn samples_per_epoch(&self) -> usize {
        self.samples + self.test_samples
    }

    pub fn total_samples(&self) -> usize {
        self.samples_per_epoch() * self.epochs
    }
}

/// Full training parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TrainParams {
    #[serde(default = "default_tokenizer")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "gpt2"))]
    pub tokenizer: String,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "60m"))]
    pub size: ModelSize,
    #[serde(default, flatten)]
    #[cfg_attr(feature = "clap", command(flatten))]
    pub ttt: TTTConfig,
    #[serde(default, flatten)]
    #[cfg_attr(feature = "clap", command(flatten))]
    pub train: TrainConfig,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub seed: Option<u64>,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub dry_run: bool,
}

fn default_tokenizer() -> String {
    "gpt2".into()
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            tokenizer: default_tokenizer(),
            size: ModelSize::default(),
            ttt: TTTConfig::default(),
            train: TrainConfig::default(),
            seed: None,
            dry_run: false,
        }
    }
}

impl TrainParams {
    /// Get model architecture from size preset + vocab_size.
    pub fn arch(&self, vocab_size: usize) -> ModelArch {
        ModelArch::from_size(self.size, vocab_size)
    }

    /// Convert to CLI arguments for subprocess invocation.
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--tokenizer".into(),
            self.tokenizer.clone(),
            "--size".into(),
            self.size.to_string(),
            // TTT config
            "--layer-type".into(),
            self.ttt.layer_type.to_string(),
            "--pos-encoding".into(),
            self.ttt.pos_encoding.to_string(),
            "--base-lr".into(),
            self.ttt.base_lr.to_string(),
            "--mini-batch-size".into(),
            self.ttt.mini_batch_size.to_string(),
            "--max-seq-len".into(),
            self.ttt.max_seq_len.to_string(),
            "--mlp-expansion-factor".into(),
            self.ttt.mlp_expansion_factor.to_string(),
            "--dtype".into(),
            self.ttt.dtype.to_string(),
            // Train config
            "--batch".into(),
            self.train.batch.to_string(),
            "--epochs".into(),
            self.train.epochs.to_string(),
            "--lr".into(),
            self.train.lr.to_string(),
            "--samples".into(),
            self.train.samples.to_string(),
            "--test-samples".into(),
            self.train.test_samples.to_string(),
            "--grad-accum".into(),
            self.train.grad_accum.to_string(),
            "--workers".into(),
            self.train.workers.to_string(),
            "--warmup-steps".into(),
            self.train.warmup_steps.to_string(),
        ];
        if let Some(threads) = self.ttt.threads {
            args.extend(["--threads".into(), threads.to_string()]);
        }
        if let Some(seed) = self.seed {
            args.extend(["--seed".into(), seed.to_string()]);
        }
        if self.dry_run {
            args.push("--dry-run".into());
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_arch_from_size() {
        let arch = ModelArch::from_size(ModelSize::M60, 50257);
        assert_eq!(arch.hidden_size, 512);
        assert_eq!(arch.num_hidden_layers, 6);
        assert_eq!(arch.vocab_size, 50257);
    }

    #[test]
    fn test_train_params_default() {
        let params = TrainParams::default();
        assert_eq!(params.tokenizer, "gpt2");
        assert_eq!(params.train.epochs, 10);
        assert_eq!(params.train.batch, 32);
    }

    #[test]
    fn test_enum_serde() {
        assert_eq!(
            serde_json::from_str::<InnerModel>("\"linear\"").unwrap(),
            InnerModel::Linear
        );
        assert_eq!(
            serde_json::from_str::<ModelSize>("\"60m\"").unwrap(),
            ModelSize::M60
        );
        assert_eq!(
            serde_json::from_str::<PosEncoding>("\"rope\"").unwrap(),
            PosEncoding::Rope
        );
    }
}
