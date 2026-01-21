use clap::{Parser, Subcommand, ValueEnum};

pub mod data;
pub mod inference;
pub mod text_generation;
pub mod training;
pub mod ttt;

use burn::optim::AdamConfig;
use training::TTTTrainingConfig;
use ttt::{TTTConfig, TTTLayerType, TrainingBackend};

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
    },
    /// Interactive generation session
    Interactive {
        /// Artifact directory containing the trained model
        artifact_dir: String,
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
enum ModelSize {
    // -- custom --
    /// 20M parameter model
    #[value(name = "12m")]
    M12,
    /// 60M parameter model
    #[default]
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
    /// Inner model type
    #[arg(long, default_value = "fused-linear")]
    inner: InnerModel,

    /// Model size
    #[arg(long, default_value = "tiny")]
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
    #[arg(long, default_value = "5e-4")]
    lr: f64,

    /// Base learning rate for TTT
    #[arg(long, default_value = "1")]
    ttt_base_lr: f32,

    /// Mini-batch size
    #[arg(long, default_value = "32")]
    mini_batch_size: usize,

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
    fn to_ttt_config(self) -> TTTConfig {
        match self {
            ModelSize::M12 => TTTConfig::default_12m(),
            ModelSize::M60 => TTTConfig::default_60m(),
            ModelSize::M125 => TTTConfig::default_125m(),
            ModelSize::M350 => TTTConfig::default_350m(),
            ModelSize::M760 => TTTConfig::default_760m(),
            ModelSize::B1 => TTTConfig::default_1b(),
        }
    }
}

impl TrainArgs {
    fn into_config(self) -> TTTTrainingConfig {
        let ttt_config = self
            .size
            .to_ttt_config()
            .with_layer_type(self.inner.to_layer_type())
            .with_mini_batch_size(self.mini_batch_size)
            .with_base_lr(self.ttt_base_lr)
            .with_max_seq_len(self.seq_len);

        TTTTrainingConfig {
            ttt_config,
            optimizer: AdamConfig::new()
                .with_beta_1(0.9)
                .with_beta_2(0.95)
                .with_epsilon(1e-8),
            batch_size: self.batch,
            num_epochs: self.epochs,
            grad_accumulation: self.grad_accum,
            warmup_steps: 100,
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
            let artifact_dir = args.out.clone();
            let inner_model = args.inner;
            let pretokenized = args.pretokenized;
            let config = args.into_config();

            println!("Training TTT text generation model...");
            println!("Artifacts will be saved to: {artifact_dir}");
            println!("Inner model: {inner_model:?}");
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
            if pretokenized {
                training::train_dataset_pretokenized::<TrainingBackend>(
                    &device,
                    &config,
                    &artifact_dir,
                );
            } else {
                training::train_dataset::<TrainingBackend>(&device, &config, &artifact_dir);
            }
        }
        Commands::Generate {
            artifact_dir,
            prompt,
        } => {
            let device = Default::default();

            match inference::generate::<TrainingBackend>(&artifact_dir, device, &prompt) {
                Ok(generated) => {
                    println!("Prompt: {}", prompt);
                    println!("Generated: {}", generated);
                }
                Err(e) => {
                    eprintln!("Error generating text: {}", e);
                }
            }
        }
        Commands::Interactive { artifact_dir } => {
            let device = Default::default();

            match inference::interactive::<TrainingBackend>(&artifact_dir, device) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error starting interactive session: {}", e);
                }
            }
        }
    }
}
