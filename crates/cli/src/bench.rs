//! Simple TTT benchmark CLI for parameter sweeps.
//!
//! Usage:
//!   ttt-bench --model-size 125m --ttt-type linear
//!   ttt-bench --model-size 125m --ttt-type fused-linear --batch 8 --json
//!   ttt-bench --model-size 125m --ttt-type linear --dtype bfloat16

use std::{sync::Arc, time::Instant};

use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use ttt_common::{InnerModel, ModelArch, ModelSize, TTTConfig};
use ttt_core::{
    TTTInnerModel,
    config::ModelConfig,
    test_utils::{TestDims, generate_test_inputs},
};
use ttt_fused::FusedTttBackend;
use ttt_layer::dispatch_ttt_layer_type;

// Backend types for runtime dtype selection
#[cfg(feature = "rocm")]
type BackendF32 = burn::backend::Rocm<f32>;
#[cfg(feature = "rocm")]
type BackendBf16 = burn::backend::Rocm<half::bf16>;

#[cfg(feature = "cuda")]
type BackendF32 = burn::backend::Cuda<f32>;
#[cfg(feature = "cuda")]
type BackendBf16 = burn::backend::Cuda<half::bf16>;

#[cfg(feature = "wgpu")]
type BackendF32 = burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
type BackendBf16 = burn::backend::Wgpu; // wgpu doesn't distinguish

type AutodiffF32 = burn::backend::Autodiff<BackendF32>;
type AutodiffBf16 = burn::backend::Autodiff<BackendBf16>;

fn parse_layer_type(s: &str) -> InnerModel {
    match s.to_lowercase().as_str() {
        "linear" => InnerModel::Linear,
        "linear-adam" | "linearadam" => InnerModel::LinearAdam,
        "mlp" => InnerModel::Mlp,
        "mlp2" => InnerModel::Mlp2,
        "mlp3" => InnerModel::Mlp3,
        "mlp4" => InnerModel::Mlp4,
        "fused" | "fused-linear" | "fusedlinear" => InnerModel::FusedLinear,
        "fused-tile" | "fusedtile" => InnerModel::FusedTileLinear,
        "fused-tile-multi" | "fusedtilemulti" => InnerModel::FusedTileMultiLinear,
        _ => panic!(
            "Unknown ttt-type: {s}. Use: linear, linear-adam, mlp, mlp2, mlp3, mlp4, fused, fused-tile, fused-tile-multi"
        ),
    }
}

fn parse_model_size(s: &str) -> ModelSize {
    match s.to_lowercase().as_str() {
        "12m" => ModelSize::M12,
        "60m" => ModelSize::M60,
        "125m" => ModelSize::M125,
        "350m" => ModelSize::M350,
        "760m" => ModelSize::M760,
        "1b" => ModelSize::B1,
        _ => panic!("Unknown model size: {s}. Use: 12m, 60m, 125m, 350m, 760m, 1b"),
    }
}

#[derive(Parser, Debug)]
#[command(name = "ttt-bench", about = "TTT benchmark for parameter sweeps")]
struct Args {
    #[arg(long, default_value = "125m")]
    model_size: String,

    #[arg(long, default_value = "linear")]
    ttt_type: String,

    #[arg(long, default_value = "false")]
    backward: bool,

    #[arg(long, default_value = "1")]
    batch: usize,

    #[arg(long, default_value = "2048")]
    seq_len: usize,

    #[arg(long, default_value = "16")]
    mini_batch: usize,

    #[arg(long, default_value = "3")]
    warmup: usize,

    #[arg(long, default_value = "5")]
    repeats: usize,

    #[arg(long, default_value = "false")]
    json: bool,

    #[arg(long, default_value = "float32")]
    dtype: String,
}

#[derive(Serialize)]
struct BenchResult {
    implementation: &'static str,
    model_size: String,
    ttt_type: String,
    backward: bool,
    batch: usize,
    seq_len: usize,
    mini_batch: usize,
    dtype: &'static str,
    time_ms: f64,
    throughput: f64,
}

fn make_dims(args: &Args, config: &ModelConfig) -> TestDims {
    TestDims {
        batch_size: args.batch,
        num_heads: config.arch.num_heads,
        head_dim: config.head_dim(),
        seq_len: args.seq_len,
        mini_batch_size: args.mini_batch,
        iterations: 1,
    }
}

fn sync<B: FusedTttBackend, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

fn bench_fwd<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
    args: &Args,
    config: &ModelConfig,
    device: &B::Device,
) -> f64 {
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(config, &inner_config, device);
    let dims = make_dims(args, config);

    for _ in 0..args.warmup {
        let inputs = generate_test_inputs(dims, device);
        let mut state = inner.init_state(args.batch);
        sync(inner.forward(&mut state, inputs));
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let inputs = generate_test_inputs(dims, device);
        let mut state = inner.init_state(args.batch);
        let start = Instant::now();
        sync(inner.forward(&mut state, inputs));
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn bench_bwd<
    B: burn::tensor::backend::AutodiffBackend + FusedTttBackend,
    Inner: TTTInnerModel<B>,
>(
    args: &Args,
    config: &ModelConfig,
    device: &B::Device,
) -> f64 {
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(config, &inner_config, device);
    let dims = make_dims(args, config);

    for _ in 0..args.warmup {
        let inputs = generate_test_inputs(dims, device);
        let mut state = inner.init_state(args.batch);
        let _ = inner.forward(&mut state, inputs).sum().backward();
    }

    let mut total = 0.0;
    for _ in 0..args.repeats {
        let inputs = generate_test_inputs(dims, device);
        let mut state = inner.init_state(args.batch);
        let start = Instant::now();
        let _ = inner.forward(&mut state, inputs).sum().backward();
        total += start.elapsed().as_secs_f64();
    }
    (total / args.repeats as f64) * 1000.0
}

fn make_config(model_size: &str, mini_batch: usize) -> ModelConfig {
    let vocab_size = 1000;
    let size = parse_model_size(model_size);
    let arch = Arc::new(ModelArch::from_size(size, vocab_size));
    let ttt = Arc::new(TTTConfig {
        mini_batch_size: mini_batch,
        ..TTTConfig::default()
    });
    ModelConfig::new(arch, ttt)
}

fn main() {
    let args = Args::parse();
    let config = make_config(&args.model_size, args.mini_batch);
    let layer_type = parse_layer_type(&args.ttt_type);

    let (time_ms, dtype_str) = match (args.dtype.as_str(), args.backward) {
        ("float32", false) => {
            let device: <BackendF32 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_fwd::<BackendF32, layer_type>(
                &args, &config, &device
            ));
            (t, "float32")
        }
        ("float32", true) => {
            let device: <AutodiffF32 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_bwd::<AutodiffF32, layer_type>(
                &args, &config, &device
            ));
            (t, "float32")
        }
        ("bfloat16", false) => {
            let device: <BackendBf16 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_fwd::<BackendBf16, layer_type>(
                &args, &config, &device
            ));
            (t, "bfloat16")
        }
        ("bfloat16", true) => {
            let device: <AutodiffBf16 as Backend>::Device = Default::default();
            let t = dispatch_ttt_layer_type!(bench_bwd::<AutodiffBf16, layer_type>(
                &args, &config, &device
            ));
            (t, "bfloat16")
        }
        _ => panic!("Unknown dtype: {}. Use: float32, bfloat16", args.dtype),
    };

    let throughput = (args.batch * args.seq_len) as f64 / (time_ms / 1000.0);

    let result = BenchResult {
        implementation: "burn",
        model_size: args.model_size.clone(),
        ttt_type: args.ttt_type.clone(),
        backward: args.backward,
        batch: args.batch,
        seq_len: args.seq_len,
        mini_batch: args.mini_batch,
        dtype: dtype_str,
        time_ms,
        throughput,
    };

    if args.json {
        println!("{}", serde_json::to_string(&result).unwrap());
    } else {
        println!("Burn TTT Benchmark");
        println!("==================");
        println!("Model: {} / {}", args.model_size, args.ttt_type);
        println!("Dtype: {}", dtype_str);
        println!("Backward: {}", args.backward);
        println!(
            "Batch: {}, Seq: {}, Mini-batch: {}",
            args.batch, args.seq_len, args.mini_batch
        );
        println!();
        println!("Time: {:.2} ms", time_ms);
        println!("Throughput: {:.0} tok/s", throughput);
    }
}
