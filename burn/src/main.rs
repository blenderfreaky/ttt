use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{Shell, generate};

pub mod artifact_info;
pub mod data;
pub mod inference;
pub mod text_generation;
pub mod training;
pub mod ttt;

use burn::config::Config;
use burn::optim::AdamConfig;
use burn::tensor::backend::Backend;
use data::{Tokenizer, TokenizerTrait};
use training::TTTTrainingConfig;
use ttt::{PositionEncodingType, TTTConfig, TTTLayerType, TrainingBackend};

/// Load a tokenizer from a HuggingFace model name or local file path.
fn load_tokenizer(identifier: &str) -> Tokenizer {
    Tokenizer::load(identifier, None, None, None)
}

#[derive(Parser)]
#[command(
    name = "ttt",
    about = "TTT text generation model training and inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train on TinyStories dataset
    Train(TrainArgs),
    /// Generate text from prompt
    Generate {
        /// Artifact directory containing the trained model
        artifact_dir: String,
        /// The prompt to generate from
        prompt: String,
        /// Tokenizer: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b") or local file path
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,
    },
    /// Interactive generation session
    Interactive {
        /// Artifact directory containing the trained model
        artifact_dir: String,
        /// Tokenizer: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b") or local file path
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,
    },
    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
    /// Show information about a training run
    Info {
        /// Artifact directory to inspect
        artifact_dir: String,
        /// Show detailed metrics for each epoch
        #[arg(long, short)]
        verbose: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum InnerModel {
    /// Linear inner model
    Linear,
    /// Linear inner model with Adam optimizer
    LinearAdam,
    /// MLP inner model (standard)
    Mlp,
    /// MLP inner model (2 hidden layers)
    Mlp2,
    /// MLP inner model (3 hidden layers)
    Mlp3,
    /// MLP inner model (4 hidden layers)
    Mlp4,
    /// Fused linear (naive CubeCL impl)
    #[default]
    FusedLinear,
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum PosEncoding {
    /// Rotary Position Embeddings applied to Q/K
    #[default]
    Rope,
    /// No position encoding
    None,
    /// Learned absolute position embeddings
    Absolute,
}

impl PosEncoding {
    fn to_pos_encoding_type(self) -> PositionEncodingType {
        match self {
            PosEncoding::Rope => PositionEncodingType::RoPE,
            PosEncoding::None => PositionEncodingType::None,
            PosEncoding::Absolute => PositionEncodingType::Absolute,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum ModelSize {
    // -- custom --
    /// 20M parameter model
    #[value(name = "12m")]
    M12,
    /// 60M parameter model
    #[default]
    #[value(name = "60m")]
    M60,
    // -- below are included in the reference --
    /// 125M parameter model
    #[value(name = "125m")]
    M125,
    /// 350M parameter model
    #[value(name = "350m")]
    M350,
    /// 760M parameter model
    #[value(name = "760m")]
    M760,
    /// 1B parameter model
    #[value(name = "1b")]
    B1,
}

#[derive(Parser)]
struct TrainArgs {
    /// Tokenizer: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b") or local file path
    #[arg(long, default_value = "gpt2")]
    tokenizer: String,

    /// Inner model type
    #[arg(long, default_value = "linear")]
    inner: InnerModel,

    /// Position encoding type
    #[arg(long, default_value = "rope")]
    pos_encoding: PosEncoding,

    /// Model size
    #[arg(long, default_value = "12m")]
    size: ModelSize,

    /// Number of epochs
    #[arg(long, default_value = "10")]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value = "32")]
    batch: usize,

    /// Max sequence length
    #[arg(long, default_value = "256")]
    seq_len: usize,

    /// Learning rate
    #[arg(long, default_value = "2e-3")]
    lr: f64,

    /// Base learning rate for TTT
    #[arg(long, default_value = "1")]
    ttt_base_lr: f32,

    /// Mini-batch size
    #[arg(long, default_value = "32")]
    mini_batch_size: usize,

    /// MLP expansion factor (hidden_dim = expansion_factor * head_dim)
    #[arg(long, default_value = "4")]
    mlp_expansion: usize,

    /// Training samples per epoch
    #[arg(long, default_value = "10000")]
    samples: usize,

    /// Test samples per epoch
    #[arg(long, default_value = "1000")]
    test_samples: usize,

    /// Gradient accumulation steps
    #[arg(long, default_value = "1")]
    grad_accum: usize,

    /// Dataloader workers
    #[arg(long, default_value = "2")]
    workers: usize,

    /// Output directory
    #[arg(long, default_value = "./artifacts")]
    out: String,

    /// Use pre-tokenized dataset (faster, recommended)
    #[arg(long, default_value = "true")]
    pretokenized: bool,

    /// If set, don't perform any training, just setup
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Resume training from a checkpoint directory (same as artifact dir)
    #[arg(long)]
    resume: Option<String>,

    /// Fixed RNG seed for reproducible training
    #[arg(long)]
    seed: Option<u64>,

    /// Learning rate warmup steps
    #[arg(long, default_value = "5000")]
    warmup_steps: usize,
}

impl InnerModel {
    fn to_layer_type(self) -> TTTLayerType {
        match self {
            InnerModel::Linear => TTTLayerType::Linear,
            InnerModel::LinearAdam => TTTLayerType::LinearAdam,
            InnerModel::Mlp => TTTLayerType::MLP,
            InnerModel::Mlp2 => TTTLayerType::MLP2,
            InnerModel::Mlp3 => TTTLayerType::MLP3,
            InnerModel::Mlp4 => TTTLayerType::MLP4,
            InnerModel::FusedLinear => TTTLayerType::FusedLinear,
        }
    }
}

impl ModelSize {
    fn to_ttt_config(self, vocab_size: usize) -> TTTConfig {
        match self {
            ModelSize::M12 => TTTConfig::default_12m(vocab_size),
            ModelSize::M60 => TTTConfig::default_60m(vocab_size),
            ModelSize::M125 => TTTConfig::default_125m(vocab_size),
            ModelSize::M350 => TTTConfig::default_350m(vocab_size),
            ModelSize::M760 => TTTConfig::default_760m(vocab_size),
            ModelSize::B1 => TTTConfig::default_1b(vocab_size),
        }
    }
}

/// Find the latest checkpoint epoch in the given artifact directory
fn find_latest_checkpoint(artifact_dir: &str) -> Option<usize> {
    let checkpoint_dir = std::path::Path::new(artifact_dir).join("checkpoint");
    std::fs::read_dir(checkpoint_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.strip_prefix("model-")?
                .strip_suffix(".mpk")?
                .parse()
                .ok()
        })
        .max()
}

impl TrainArgs {
    fn into_config(self, vocab_size: usize) -> TTTTrainingConfig {
        let ttt_config = self
            .size
            .to_ttt_config(vocab_size)
            .with_layer_type(self.inner.to_layer_type())
            .with_pos_encoding(self.pos_encoding.to_pos_encoding_type())
            .with_mini_batch_size(self.mini_batch_size)
            .with_base_lr(self.ttt_base_lr)
            .with_max_seq_len(self.seq_len)
            .with_mlp_expansion_factor(self.mlp_expansion);

        TTTTrainingConfig {
            ttt_config,
            optimizer: AdamConfig::new()
                .with_beta_1(0.9)
                .with_beta_2(0.95)
                .with_epsilon(1e-8),
            batch_size: self.batch,
            num_epochs: self.epochs,
            grad_accumulation: self.grad_accum,
            warmup_steps: self.warmup_steps,
            learning_rate: self.lr,
            max_seq_len: self.seq_len,
            train_samples: self.samples,
            test_samples: self.test_samples,
            num_workers: self.workers,
            dry_run: self.dry_run,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train(args) => {
            let pretokenized = args.pretokenized;
            let resume_dir = args.resume.clone();
            let tokenizer_name = args.tokenizer.clone();
            let seed = args.seed;
            let tokenizer = load_tokenizer(&tokenizer_name);

            // Determine config and artifact_dir based on whether we're resuming
            let (config, artifact_dir, resume_epoch) = if let Some(ref resume_dir) = resume_dir {
                // Load config from resume directory
                let config_path = format!("{resume_dir}/config.json");
                let mut config = TTTTrainingConfig::load(&config_path)
                    .unwrap_or_else(|e| panic!("Failed to load config from {config_path}: {e}"));

                // Allow --epochs to override for training additional epochs
                if args.epochs != 10 {
                    // 10 is the default
                    config.num_epochs = args.epochs;
                }

                // Find latest checkpoint
                let resume_epoch = find_latest_checkpoint(resume_dir)
                    .unwrap_or_else(|| panic!("No checkpoint found in {resume_dir}/checkpoint/"));

                println!("Resuming training from epoch {resume_epoch}");
                (config, resume_dir.clone(), Some(resume_epoch))
            } else {
                let artifact_dir = args.out.clone();
                let config = args.into_config(tokenizer.vocab_size());
                (config, artifact_dir, None)
            };

            println!("Training TTT text generation model...");
            println!("Artifacts will be saved to: {artifact_dir}");
            println!(
                "Tokenizer: {tokenizer_name} (vocab_size: {})",
                tokenizer.vocab_size()
            );
            println!("Layer type: {:?}", config.ttt_config.layer_type);
            println!(
                "Model size: {} hidden, {} layers",
                config.ttt_config.hidden_size, config.ttt_config.num_hidden_layers
            );
            println!(
                "Batch size: {}, Epochs: {}, LR: {}",
                config.batch_size, config.num_epochs, config.learning_rate
            );
            println!(
                "Sequence length: {}, Samples: {}",
                config.max_seq_len, config.train_samples
            );
            println!(
                "Pre-tokenized: {} (use --no-pretokenized to disable)",
                pretokenized
            );

            let device = Default::default();

            // Set RNG seed for reproducibility if provided
            if let Some(seed) = seed {
                println!("Using fixed RNG seed: {seed}");
                TrainingBackend::seed(&device, seed);
            }

            if pretokenized {
                training::train_dataset_pretokenized::<TrainingBackend>(
                    &device,
                    &config,
                    &artifact_dir,
                    &tokenizer,
                    resume_epoch,
                );
            } else {
                training::train_dataset::<TrainingBackend>(
                    &device,
                    &config,
                    &artifact_dir,
                    tokenizer,
                    resume_epoch,
                );
            }
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            tokenizer,
        } => {
            let device = Default::default();
            let tokenizer = load_tokenizer(&tokenizer);

            match inference::generate::<TrainingBackend>(&artifact_dir, device, &prompt, tokenizer)
            {
                Ok(generated) => {
                    println!("Prompt: {}", prompt);
                    println!("Generated: {}", generated);
                }
                Err(e) => {
                    eprintln!("Error generating text: {}", e);
                }
            }
        }
        Commands::Interactive {
            artifact_dir,
            tokenizer,
        } => {
            let device = Default::default();
            let tokenizer = load_tokenizer(&tokenizer);

            match inference::interactive::<TrainingBackend>(&artifact_dir, device, tokenizer) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error starting interactive session: {}", e);
                }
            }
        }
        Commands::Completions { shell } => {
            generate(shell, &mut Cli::command(), "ttt", &mut std::io::stdout());
        }
        Commands::Info {
            artifact_dir,
            verbose,
        } => match artifact_info::ArtifactInfo::load(&artifact_dir) {
            Ok(info) => artifact_info::print_info(&info, verbose),
            Err(e) => {
                eprintln!("Error loading artifact info from {artifact_dir}: {e}");
                std::process::exit(1);
            }
        },
    }
}
