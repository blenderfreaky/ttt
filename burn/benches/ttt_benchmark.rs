//! Generic benchmarks for TTT (Test-Time Training) inner models and full models.
//!
//! This benchmark suite provides:
//! - Inner model benchmarks (forward/backward) - just the TTT learner
//! - Full model benchmarks (forward/backward) - complete text generation model
//!
//! All benchmarks are generic over:
//! - Backend (inference vs training with autodiff)
//! - Inner model type (TTTLinear, TTTLinearAdam, TTTMLP, TTTMLP2, TTTMLP3, TTTMLP4, Fused)
//! - Model configuration (hidden size, num heads, etc.)
//! - Runtime parameters (batch size, sequence length, vocab size)
//!
//! # Usage
//!
//! Run all benchmarks:
//!   cargo bench --bench ttt_benchmark
//!
//! Run specific benchmark:
//!   cargo bench --bench ttt_benchmark -- inner_forward_linear
//!   cargo bench --bench ttt_benchmark -- full_forward

use std::sync::Arc;
use std::time::Duration;

use burn::prelude::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use paste::paste;

use ttt::data::TrainingTextGenerationBatch;
use ttt::text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel};
use ttt::ttt::TTTConfig;
use ttt::ttt::cubecl_kernels::{Fused, FusedTile, FusedTttBackend};
use ttt::ttt::layer::{Qkv, TTTInnerModel, TTTInputsInner};
use ttt::ttt::linear::TTTLinear;
use ttt::ttt::linear_adam::TTTLinearAdam;
use ttt::ttt::mlp::TTTMLP;
use ttt::ttt::mlp2::TTTMLP2;
use ttt::ttt::mlp3::TTTMLP3;
use ttt::ttt::mlp4::TTTMLP4;
use ttt::{GpuAutodiffBackend, GpuBackend};

pub fn device<B: FusedTttBackend>() -> B::Device {
    Default::default()
}

/// Force async operations to complete before returning.
fn sync<B: FusedTttBackend, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub name: &'static str,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_intermediate: usize,
    pub mini_batch_size: usize,
}

impl BenchConfig {
    pub const fn tiny() -> Self {
        Self {
            name: "tiny",
            hidden_size: 64,
            num_heads: 2,
            num_layers: 1,
            mlp_intermediate: 128,
            mini_batch_size: 16,
        }
    }

    pub const fn small() -> Self {
        Self {
            name: "small",
            hidden_size: 128,
            num_heads: 4,
            num_layers: 2,
            mlp_intermediate: 256,
            mini_batch_size: 16,
        }
    }

    pub const fn medium() -> Self {
        Self {
            name: "medium",
            hidden_size: 256,
            num_heads: 8,
            num_layers: 4,
            mlp_intermediate: 512,
            mini_batch_size: 16,
        }
    }

    pub fn to_ttt_config(&self, vocab_size: usize) -> TTTConfig {
        TTTConfig::new(vocab_size)
            .with_token_size(self.hidden_size)
            .with_hidden_size(self.hidden_size)
            .with_num_heads(self.num_heads)
            .with_num_hidden_layers(self.num_layers)
            .with_swi_glu_mlp_intermediate_size(self.mlp_intermediate)
            .with_mini_batch_size(self.mini_batch_size)
            .with_conv_before_ttt(false)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeParams {
    pub batch_size: usize,
    pub seq_length: usize,
    pub vocab_size: usize,
}

impl RuntimeParams {
    pub fn new(batch_size: usize, seq_length: usize, vocab_size: usize) -> Self {
        Self {
            batch_size,
            seq_length,
            vocab_size,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.batch_size * self.seq_length
    }

    pub fn id(&self) -> String {
        format!("b{}_s{}", self.batch_size, self.seq_length)
    }
}

fn create_inner_inputs<B: FusedTttBackend>(
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) -> TTTInputsInner<B> {
    let batch_size = params.batch_size;
    let num_heads = config.num_heads;
    let seq_len = params.seq_length;
    let head_dim = config.head_dim();

    let xq = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );
    let xk = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );
    let xv = Tensor::random(
        [batch_size, num_heads, seq_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device,
    );

    let token_eta = Tensor::arange(1..(seq_len as i64 + 1), device)
        .float()
        .recip();

    let ttt_lr_eta = Tensor::random(
        [batch_size, num_heads, seq_len],
        burn::tensor::Distribution::Uniform(0.01, 0.05),
        device,
    );

    TTTInputsInner {
        qkv: Qkv { xq, xk, xv },
        token_eta,
        ttt_lr_eta,
        start_idx: 0,
    }
}

fn random_logits<B: FusedTttBackend>(
    params: &RuntimeParams,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::random(
        [params.batch_size, params.seq_length],
        burn::tensor::Distribution::Uniform(0.0, params.vocab_size as f64 - 1.0),
        device,
    )
}

fn create_training_batch<B: FusedTttBackend>(
    params: &RuntimeParams,
    device: &B::Device,
) -> TrainingTextGenerationBatch<B> {
    let tokens_inputs = random_logits::<B>(params, device);
    let targets = random_logits::<B>(params, device);
    let mask_pad = Tensor::<B, 2, Bool>::ones([params.batch_size, params.seq_length], device);

    TrainingTextGenerationBatch {
        tokens_inputs,
        targets,
        mask_pad,
    }
}

/// Benchmark inner model forward pass
fn bench_inner_forward<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let ttt_config = Arc::new(config.to_ttt_config(params.vocab_size));
    let inner_config = Arc::new(Inner::Config::default());

    let inner: Inner = Inner::new(&ttt_config, &inner_config, device);

    // Warmup
    let warmup_inputs = create_inner_inputs::<B>(config, params, device);
    let mut warmup_state = inner.init_state(params.batch_size);
    for _ in 0..3 {
        let output = inner.forward(&mut warmup_state, warmup_inputs.clone());
        sync(output);
    }

    let group_name = format!("inner_forward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("forward", &bench_id), |b| {
        b.iter_batched(
            || {
                let inputs = create_inner_inputs::<B>(config, params, device);
                let state = inner.init_state(params.batch_size);
                (inputs, state)
            },
            |(inputs, mut state)| {
                let output = inner.forward(&mut state, inputs);
                sync(output);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark inner model backward pass (forward + backward)
fn bench_inner_backward<
    B: burn::tensor::backend::AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B>,
>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let ttt_config = Arc::new(config.to_ttt_config(params.vocab_size));
    let inner_config = Arc::new(Inner::Config::default());

    let inner: Inner = Inner::new(&ttt_config, &inner_config, device);

    // Warmup
    let warmup_inputs = create_inner_inputs::<B>(config, params, device);
    let mut warmup_state = inner.init_state(params.batch_size);
    let output = inner.forward(&mut warmup_state, warmup_inputs);
    let _grads = output.sum().backward();

    let group_name = format!("inner_backward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(30));

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("backward", &bench_id), |b| {
        b.iter_batched(
            || {
                let inputs = create_inner_inputs::<B>(config, params, device);
                let state = inner.init_state(params.batch_size);
                (inputs, state)
            },
            |(inputs, mut state)| {
                let output = inner.forward(&mut state, inputs);
                let _grads = output.sum().backward();
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark full model forward pass
fn bench_full_forward<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let ttt_config = config.to_ttt_config(params.vocab_size);
    let model_config = TTTTextGenerationConfig::new_testing(ttt_config);
    let model: TTTTextGenerationModel<B, Inner> = model_config.init(device);

    // Warmup
    let warmup_input = random_logits::<B>(params, device);
    for _ in 0..3 {
        let output = model.forward_inference(warmup_input.clone());
        sync(output);
    }

    let group_name = format!("full_forward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));

    let bench_id = format!("{}_{}", config.name, params.id());
    let input = random_logits::<B>(params, device);

    group.bench_function(BenchmarkId::new("forward", &bench_id), |b| {
        b.iter(|| {
            let output = model.forward_inference(input.clone());
            sync(output);
        });
    });

    group.finish();
}

/// Benchmark full model backward pass (forward + backward)
fn bench_full_backward<
    B: burn::tensor::backend::AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B>,
>(
    c: &mut Criterion,
    config: &BenchConfig,
    params: &RuntimeParams,
    device: &B::Device,
) {
    let ttt_config = config.to_ttt_config(params.vocab_size);
    let model_config = TTTTextGenerationConfig::new_testing(ttt_config);
    let model: TTTTextGenerationModel<B, Inner> = model_config.init(device);

    // Warmup
    let warmup_batch = create_training_batch::<B>(params, device);
    let output = model.forward_training(warmup_batch);
    let _grads = output.loss.backward();

    let group_name = format!("full_backward_{}", Inner::name());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(params.total_tokens() as u64));
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(20);

    let bench_id = format!("{}_{}", config.name, params.id());
    group.bench_function(BenchmarkId::new("backward", &bench_id), |b| {
        b.iter_batched(
            || create_training_batch::<B>(params, device),
            |batch| {
                let output = model.forward_training(batch);
                let _grads = output.loss.backward();
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

// Benchmark generation macros

/// Runtime parameters used across all benchmarks
const BENCH_PARAMS: &[RuntimeParams] = &[
    RuntimeParams {
        batch_size: 4,
        seq_length: 128,
        vocab_size: 1000,
    },
    // RuntimeParams {
    //     batch_size: 16,
    //     seq_length: 8192,
    //     vocab_size: 1000,
    // },
    // RuntimeParams {
    //     batch_size: 4,
    //     seq_length: 8192,
    //     vocab_size: 1000,
    // },
];

const BENCH_CONFIGS: &[BenchConfig] = &[
    BenchConfig::tiny(),
    // BenchConfig::small(),
    // BenchConfig::medium(),
];

/// Generate a single benchmark entry function
macro_rules! define_bench {
    // Forward benchmarks (inference backend)
    (forward, $scope:ident, $fn_name:ident, $inner:ty) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    paste! { [<bench_ $scope _forward>]::<GpuBackend, $inner>(c, &config, params, &device); }
                }
            }
        }
    };
    // Backward benchmarks (training backend)
    (backward, $scope:ident, $fn_name:ident, $inner:ty) => {
        fn $fn_name(c: &mut Criterion) {
            let device = device::<GpuAutodiffBackend>();
            for config in BENCH_CONFIGS {
                for params in BENCH_PARAMS {
                    paste! { [<bench_ $scope _backward>]::<GpuAutodiffBackend, $inner>(c, &config, params, &device); }
                }
            }
        }
    };
}

/// Generate all 4 benchmark variants (inner/full × forward/backward) for a model
macro_rules! define_model_benches {
    ($suffix:ident, $inner:ty) => {
        paste! {
            define_bench!(forward, inner, [<bench_inner_forward_ $suffix>], $inner);
            define_bench!(backward, inner, [<bench_inner_backward_ $suffix>], $inner);
            define_bench!(forward, full, [<bench_full_forward_ $suffix>], $inner);
            define_bench!(backward, full, [<bench_full_backward_ $suffix>], $inner);
        }
    };
}

/// Generate all benchmarks and criterion groups for the given models
macro_rules! define_all_benches {
    ($($suffix:ident => $inner:ty),* $(,)?) => {
        // Generate all benchmark functions
        $(define_model_benches!($suffix, $inner);)*

        paste! {
            // Generate criterion groups
            criterion_group!(inner_forward, $([<bench_inner_forward_ $suffix>]),*);
            criterion_group!(inner_backward, $([<bench_inner_backward_ $suffix>]),*);
            criterion_group!(full_forward, $([<bench_full_forward_ $suffix>]),*);
            criterion_group!(full_backward, $([<bench_full_backward_ $suffix>]),*);
        }
    };
}

define_all_benches!(
    linear => TTTLinear<_>,
    linear_adam => TTTLinearAdam<_>,
    mlp => TTTMLP<_>,
    mlp2 => TTTMLP2<_>,
    mlp3 => TTTMLP3<_>,
    mlp4 => TTTMLP4<_>,
    fused_linear => Fused<_, TTTLinear<_>>,
);

/// Generate all 4 benchmark variants for the tiled kernel
macro_rules! define_tile_model_benches {
    ($suffix:ident, $inner:ty) => {
        paste! {
            define_bench!(forward, inner, [<bench_inner_forward_ $suffix>], $inner);
            // Note: backward not yet implemented for tile kernel
            // define_tile_bench!(backward, inner, [<bench_inner_backward_ $suffix>], $inner);
            define_bench!(forward, full, [<bench_full_forward_ $suffix>], $inner);
            // define_tile_bench!(backward, full, [<bench_full_backward_ $suffix>], $inner);
        }
    };
}

// Generate tiled kernel benchmarks
define_tile_model_benches!(fused_tile, FusedTile<_>);

criterion_group!(
    tile_forward,
    bench_inner_forward_fused_tile,
    bench_full_forward_fused_tile
);

criterion_main!(
    inner_forward,
    inner_backward,
    full_forward,
    full_backward,
    tile_forward
);
