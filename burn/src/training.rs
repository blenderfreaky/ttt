use crate::{
    data::{
        Gpt2Tokenizer, TextDataset, TextGenerationBatcher, TextGenerationItem, Tokenizer,
        TrainingTextGenerationBatch,
    },
    text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel},
    ttt::{
        TTTConfig,
        cubecl_kernels::{backend::FusedTttBackend, linear::FusedTTTLinear},
        layer::TTTInnerModel,
    },
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, transform::SamplerDataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::{AutodiffModule, ModuleDisplay},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, Learner, LearningParadigm, SupervisedTraining, ValidStep,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};
use std::sync::Arc;

#[derive(Config, Debug)]
pub struct TTTTrainingConfig {
    /// TTT model configuration
    pub ttt_config: TTTConfig,
    /// Optimizer configuration
    pub optimizer: AdamConfig,
    /// Training batch size
    #[config(default = 16)]
    pub batch_size: usize,
    /// Number of training epochs
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Gradient accumulation steps
    #[config(default = 4)]
    pub grad_accumulation: usize,
    /// Learning rate warmup steps
    #[config(default = 1000)]
    pub warmup_steps: usize,
    /// Peak learning rate
    #[config(default = 5e-4)]
    pub learning_rate: f64,
    /// Maximum sequence length for training
    #[config(default = 2048)]
    pub max_seq_len: usize,
}

impl TTTTrainingConfig {
    #[must_use]
    pub fn small() -> Self {
        Self {
            ttt_config: TTTConfig::default_tiny(),
            optimizer: AdamConfig::new()
                .with_beta_1(0.9)
                .with_beta_2(0.95)
                .with_epsilon(1e-8),
            batch_size: 8,
            num_epochs: 10,
            grad_accumulation: 1,
            warmup_steps: 100,
            learning_rate: 5e-4,
            max_seq_len: 1024,
        }
    }
}

/// Train TTT text generation model (non-fused, works on any backend)
pub fn train_ttt_text_generation<
    B: AutodiffBackend + FusedTttBackend,
    D: Dataset<TextGenerationItem> + 'static,
>(
    device: &B::Device,
    dataset_train: D,
    dataset_test: D,
    config: TTTTrainingConfig,
    artifact_dir: &str,
) where
    B::InnerBackend: FusedTttBackend,
{
    let tokenizer = Arc::new(Gpt2Tokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_len);

    std::fs::create_dir_all(artifact_dir).unwrap();

    let hidden_size = config.ttt_config.hidden_size;

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    assert_eq!(config.ttt_config.vocab_size, tokenizer.vocab_size());
    let model_config = TTTTextGenerationConfig {
        ttt_config: config.ttt_config,
        pad_token: tokenizer.pad_token(),
    };

    println!("Initializing model...");
    let model = model_config.init::<B, FusedTTTLinear<B>>(device);
    println!("Model initialized successfully");

    run_training(
        model,
        batcher,
        dataset_train,
        dataset_test,
        &config.optimizer,
        config.batch_size,
        config.num_epochs,
        config.grad_accumulation,
        config.warmup_steps,
        config.learning_rate,
        hidden_size,
        tokenizer.pad_token(),
        artifact_dir,
    );
}

/// Common training loop for both fused and non-fused models
fn run_training<
    B: AutodiffBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    D: Dataset<TextGenerationItem> + 'static,
>(
    model: TTTTextGenerationModel<B, Inner>,
    batcher: TextGenerationBatcher,
    dataset_train: D,
    dataset_test: D,
    optimizer_config: &AdamConfig,
    batch_size: usize,
    num_epochs: usize,
    grad_accumulation: usize,
    warmup_steps: usize,
    learning_rate: f64,
    hidden_size: usize,
    pad_token: usize,
    artifact_dir: &str,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: ValidStep<
            TrainingTextGenerationBatch<B::InnerBackend>,
            ClassificationOutput<B::InnerBackend>,
        >,
{
    // Store untrained model for debugging purposes
    DefaultRecorder::new()
        .record(
            model.clone().into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();

    println!("Creating data loaders...");
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .num_workers(8)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(8)
        .build(SamplerDataset::new(dataset_test, 1_000));
    println!("Data loaders created successfully");

    println!("Initializing optimizer...");
    let optim = optimizer_config.init();
    println!("Optimizer initialized");

    println!("Initializing learning rate scheduler...");
    println!(
        "Using Noam scheduler with {warmup_steps} warmup steps, model size {hidden_size}, learning rate factor {learning_rate}"
    );
    let lr_scheduler = NoamLrSchedulerConfig::new(learning_rate)
        .with_warmup_steps(warmup_steps.max(1))
        .with_model_size(hidden_size)
        .init()
        .expect("Failed to initialize Noam LR scheduler");

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(PerplexityMetric::new())
        .metric_valid(PerplexityMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(pad_token))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(pad_token))
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(grad_accumulation)
        .num_epochs(num_epochs)
        .summary();

    let result = training.run(Learner::new(model, optim, lr_scheduler));

    DefaultRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}

/// Quick training function with dataset
pub fn train_dataset<B: AutodiffBackend + FusedTttBackend>(
    device: &B::Device,
    config: TTTTrainingConfig,
    artifact_dir: &str,
) where
    B::InnerBackend: FusedTttBackend,
{
    println!("Loading dataset...");
    let dataset_train = TextDataset::train();
    let dataset_test = TextDataset::test();

    println!("Starting TTT text generation training...");
    train_ttt_text_generation::<B, _>(device, dataset_train, dataset_test, config, artifact_dir);

    println!("Training completed! Artifacts saved to: {artifact_dir}");
}
