use std::sync::Arc;

use burn::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, transform::SamplerDataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::{AutodiffModule, DisplaySettings, ModuleDisplay},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};

use crate::{
    data::{
        TextDataset, TextGenerationBatcher, TextGenerationItem, TokenBatcher, TokenizedItem,
        Tokenizer, TokenizerTrait, TrainingTextGenerationBatch, load_or_pretokenize,
    },
    dispatch_ttt_layer_type,
    text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel},
    ttt::{
        TTTConfig, TTTLayerType, cubecl_kernels::backend::FusedTttBackend, layer::TTTInnerModel,
    },
};

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
    #[config(default = 1)]
    pub grad_accumulation: usize,
    /// Learning rate warmup steps
    #[config(default = 5000)]
    pub warmup_steps: usize,
    /// Peak learning rate
    #[config(default = 5e-4)]
    pub learning_rate: f64,
    /// Maximum sequence length for training
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// Training samples per epoch
    #[config(default = 10000)]
    pub train_samples: usize,
    /// Test samples per epoch
    #[config(default = 1000)]
    pub test_samples: usize,
    /// Dataloader workers
    #[config(default = 2)]
    pub num_workers: usize,
    /// If set, don't perform any training, just setup
    #[config(default = false)]
    pub dry_run: bool,
}

/// Common training loop for all inner model types
#[allow(clippy::too_many_arguments)]
fn run_training<
    B: AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    Item: Clone + Send + Sync + std::fmt::Debug + 'static,
    Bat: Batcher<B, Item, TrainingTextGenerationBatch<B>>
        + Batcher<B::InnerBackend, Item, TrainingTextGenerationBatch<B::InnerBackend>>
        + Clone
        + Send
        + Sync
        + 'static,
    D: Dataset<Item> + 'static,
>(
    model: TTTTextGenerationModel<B, Inner>,
    batcher: Bat,
    dataset_train: D,
    dataset_test: D,
    config: &TTTTrainingConfig,
    pad_token: usize,
    artifact_dir: &str,
    resume_epoch: Option<usize>,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    // Store untrained model for debugging purposes (skip when resuming)
    if resume_epoch.is_none() {
        DefaultRecorder::new()
            .record(
                model.clone().into_record(),
                format!("{artifact_dir}/model").into(),
            )
            .unwrap();
    }

    println!("Creating data loaders...");
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(SamplerDataset::new(dataset_train, config.train_samples));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(SamplerDataset::new(dataset_test, config.test_samples));
    println!("Data loaders created successfully");

    println!("Initializing optimizer...");
    let optim = config.optimizer.init();
    println!("Optimizer initialized");

    let hidden_size = config.ttt_config.hidden_size;
    let warmup_steps = config.warmup_steps;
    let learning_rate = config.learning_rate;

    println!("Initializing learning rate scheduler...");
    println!(
        "Using Noam scheduler with {warmup_steps} warmup steps, model size {hidden_size}, learning rate factor {learning_rate}"
    );
    let lr_scheduler = NoamLrSchedulerConfig::new(learning_rate)
        .with_warmup_steps(warmup_steps.max(1))
        .with_model_size(hidden_size)
        .init()
        .expect("Failed to initialize Noam LR scheduler");

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(PerplexityMetric::new())
        .metric_valid(PerplexityMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(pad_token))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(pad_token))
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(config.grad_accumulation)
        .num_epochs(config.num_epochs)
        .summary();

    if let Some(epoch) = resume_epoch {
        training = training.checkpoint(epoch);
    }

    if config.dry_run {
        println!(
            "Model: {}",
            &model.format(DisplaySettings::new().with_show_num_parameters(true))
        );
        println!("Dry run completed");
        return;
    }

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    DefaultRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}

/// Train with a specific inner model type
fn train_with_inner<
    B: AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    D: Dataset<TextGenerationItem> + 'static,
>(
    device: &B::Device,
    dataset_train: D,
    dataset_test: D,
    config: &TTTTrainingConfig,
    artifact_dir: &str,
    tokenizer: Tokenizer,
    resume_epoch: Option<usize>,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    let tokenizer = Arc::new(tokenizer);
    // Batcher needs max_seq_len + 1 to account for next-token prediction
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_len + 1);

    std::fs::create_dir_all(artifact_dir).unwrap();
    config.save(format!("{artifact_dir}/config.json")).unwrap();

    let model_config = TTTTextGenerationConfig {
        ttt_config: config.ttt_config.clone(),
        pad_token: tokenizer.pad_token(),
    };

    println!("Initializing model...");
    let model: TTTTextGenerationModel<B, Inner> = model_config.init(device);
    println!("Model initialized successfully");

    let pad_token = tokenizer.pad_token();
    run_training(
        model,
        batcher,
        dataset_train,
        dataset_test,
        config,
        pad_token,
        artifact_dir,
        resume_epoch,
    );
}

pub fn train_dataset<B: AutodiffBackend + FusedTttBackend>(
    device: &B::Device,
    config: &TTTTrainingConfig,
    artifact_dir: &str,
    tokenizer: Tokenizer,
    resume_epoch: Option<usize>,
) where
    B::InnerBackend: FusedTttBackend,
{
    println!("Loading dataset...");
    let dataset_train = TextDataset::train();
    let dataset_test = TextDataset::test();

    println!("Starting TTT text generation training...");
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    let ttt_type = config.ttt_config.layer_type;
    println!("Layer type: {ttt_type:?}");

    dispatch_ttt_layer_type!(train_with_inner::<B, ttt_type, _>(
        device,
        dataset_train,
        dataset_test,
        config,
        artifact_dir,
        tokenizer,
        resume_epoch,
    ));

    println!("Training completed! Artifacts saved to: {artifact_dir}");
}

/// Train with a specific inner model type using pre-tokenized data
fn train_with_inner_pretokenized<
    B: AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    D: Dataset<TokenizedItem> + 'static,
>(
    device: &B::Device,
    dataset_train: D,
    dataset_test: D,
    config: &TTTTrainingConfig,
    artifact_dir: &str,
    pad_token: usize,
    resume_epoch: Option<usize>,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    // Batcher needs max_seq_len + 1 to account for next-token prediction
    let batcher = TokenBatcher::new(pad_token, config.max_seq_len + 1);

    std::fs::create_dir_all(artifact_dir).unwrap();
    config.save(format!("{artifact_dir}/config.json")).unwrap();

    let model_config = TTTTextGenerationConfig {
        ttt_config: config.ttt_config.clone(),
        pad_token,
    };

    println!("Initializing model...");
    let model: TTTTextGenerationModel<B, Inner> = model_config.init(device);
    println!("Model initialized successfully");

    run_training(
        model,
        batcher,
        dataset_train,
        dataset_test,
        config,
        pad_token,
        artifact_dir,
        resume_epoch,
    );
}

/// Train using pre-tokenized dataset
///
/// This function will:
/// 1. Check for existing pre-tokenized data in ~/.cache
/// 2. If not found, download and tokenize the dataset
/// 3. Train using memory-mapped pre-tokenized data
pub fn train_dataset_pretokenized<B: AutodiffBackend + FusedTttBackend>(
    device: &B::Device,
    config: &TTTTrainingConfig,
    artifact_dir: &str,
    tokenizer: &Tokenizer,
    resume_epoch: Option<usize>,
) where
    B::InnerBackend: FusedTttBackend,
{
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    let pad_token = tokenizer.pad_token();

    println!("Loading pre-tokenized datasets...");
    // Load max_seq_len + 1 tokens to account for next-token prediction shift
    // (inputs: 0..max_seq_len, targets: 1..max_seq_len+1)
    let dataset_train = load_or_pretokenize(tokenizer, "train", config.max_seq_len + 1)
        .expect("Failed to load/create pre-tokenized train dataset");
    let dataset_test = load_or_pretokenize(tokenizer, "validation", config.max_seq_len + 1)
        .expect("Failed to load/create pre-tokenized test dataset");

    println!(
        "Loaded {} train sequences, {} test sequences",
        dataset_train.len(),
        dataset_test.len()
    );

    println!("Starting TTT text generation training (pre-tokenized)...");
    let ttt_type = config.ttt_config.layer_type;
    println!("Layer type: {ttt_type:?}");

    dispatch_ttt_layer_type!(train_with_inner_pretokenized::<B, ttt_type, _>(
        device,
        dataset_train,
        dataset_test,
        config,
        artifact_dir,
        pad_token,
        resume_epoch,
    ));

    println!("Training completed! Artifacts saved to: {artifact_dir}");
}
