use crate::{
    data::{
        Gpt2Tokenizer, TextDataset, TextGenerationBatcher, TextGenerationItem, Tokenizer,
        TrainingTextGenerationBatch,
    },
    text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel},
    ttt::{
        TTTConfig, TTTLayerType,
        cubecl_kernels::{Fused, backend::FusedTttBackend},
        layer::TTTInnerModel,
        linear::TTTLinear,
        linear_adam::TTTLinearAdam,
        mlp::TTTMLP,
        mlp2::TTTMLP2,
        mlp3::TTTMLP3,
        mlp4::TTTMLP4,
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
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining,
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
    /// Training samples per epoch
    #[config(default = 10000)]
    pub train_samples: usize,
    /// Test samples per epoch
    #[config(default = 1000)]
    pub test_samples: usize,
    /// Dataloader workers
    #[config(default = 2)]
    pub num_workers: usize,
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
            train_samples: 10000,
            test_samples: 1000,
            num_workers: 2,
        }
    }
}

/// Common training loop for all inner model types
fn run_training<
    B: AutodiffBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    D: Dataset<TextGenerationItem> + 'static,
>(
    model: TTTTextGenerationModel<B, Inner>,
    batcher: TextGenerationBatcher,
    dataset_train: D,
    dataset_test: D,
    config: &TTTTrainingConfig,
    pad_token: usize,
    artifact_dir: &str,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
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

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
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
    B: AutodiffBackend,
    Inner: TTTInnerModel<B> + ModuleDisplay + AutodiffModule<B> + 'static,
    D: Dataset<TextGenerationItem> + 'static,
>(
    device: &B::Device,
    dataset_train: D,
    dataset_test: D,
    config: TTTTrainingConfig,
    artifact_dir: &str,
) where
    TTTTextGenerationModel<B, Inner>: AutodiffModule<B>,
    <Inner as AutodiffModule<B>>::InnerModule: TTTInnerModel<B::InnerBackend>,
    <TTTTextGenerationModel<B, Inner> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    let tokenizer = Arc::new(Gpt2Tokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_len);

    std::fs::create_dir_all(artifact_dir).unwrap();
    config.save(format!("{artifact_dir}/config.json")).unwrap();

    assert_eq!(config.ttt_config.vocab_size, tokenizer.vocab_size());
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
        &config,
        pad_token,
        artifact_dir,
    );
}

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
    println!("Layer type: {:?}", config.ttt_config.layer_type);

    match config.ttt_config.layer_type {
        TTTLayerType::Linear => {
            train_with_inner::<B, TTTLinear<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::LinearAdam => {
            train_with_inner::<B, TTTLinearAdam<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::MLP => {
            train_with_inner::<B, TTTMLP<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::MLP2 => {
            train_with_inner::<B, TTTMLP2<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::MLP3 => {
            train_with_inner::<B, TTTMLP3<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::MLP4 => {
            train_with_inner::<B, TTTMLP4<B>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
        TTTLayerType::FusedLinear => {
            train_with_inner::<B, Fused<B, TTTLinear<B>>, _>(
                device,
                dataset_train,
                dataset_test,
                config,
                artifact_dir,
            );
        }
    }

    println!("Training completed! Artifacts saved to: {artifact_dir}");
}
