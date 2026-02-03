//! Simple TTT benchmark CLI for parameter sweeps.
//!
//! Usage:
//!   ttt-bench --model-size 125m --ttt-type linear
//!   ttt-bench --model-size 125m --ttt-type fused-linear --batch 8 --json

use std::sync::Arc;
use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, TTTConfig, TTTInnerModel, TTTLayerType,
    test_utils::{TestDims, generate_test_inputs},
};
use ttt_fused::FusedTttBackend;
use ttt_layer::dispatch_ttt_layer_type;

fn parse_layer_type(s: &str) -> TTTLayerType {
    match s.to_lowercase().as_str() {
        "linear" => TTTLayerType::Linear,
        "linear-adam" | "linearadam" => TTTLayerType::LinearAdam,
        "mlp" => TTTLayerType::MLP,
        "mlp2" => TTTLayerType::MLP2,
        "mlp3" => TTTLayerType::MLP3,
        "mlp4" => TTTLayerType::MLP4,
        "fused" | "fused-linear" | "fusedlinear" => TTTLayerType::FusedLinear,
        "fused-tile" | "fusedtile" => TTTLayerType::FusedTileLinear,
        "fused-tile-multi" | "fusedtilemulti" => TTTLayerType::FusedTileMultiLinear,
        _ => panic!("Unknown ttt-type: {s}. Use: linear, linear-adam, mlp, mlp2, mlp3, mlp4, fused, fused-tile, fused-tile-multi"),
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

fn make_dims(args: &Args, config: &TTTConfig) -> TestDims {
    TestDims {
        batch_size: args.batch,
        num_heads: config.num_heads,
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
    config: &TTTConfig,
    device: &B::Device,
) -> f64 {
    let ttt_config = Arc::new(config.clone());
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(&ttt_config, &inner_config, device);
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

fn bench_bwd<B: burn::tensor::backend::AutodiffBackend + FusedTttBackend, Inner: TTTInnerModel<B>>(
    args: &Args,
    config: &TTTConfig,
    device: &B::Device,
) -> f64 {
    let ttt_config = Arc::new(config.clone());
    let inner_config = Arc::new(Inner::Config::default());
    let inner: Inner = Inner::new(&ttt_config, &inner_config, device);
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

fn make_config(model_size: &str, mini_batch: usize) -> TTTConfig {
    let vocab_size = 1000;
    let config = match model_size {
        "12m" => TTTConfig::default_12m(vocab_size),
        "60m" => TTTConfig::default_60m(vocab_size),
        "125m" => TTTConfig::default_125m(vocab_size),
        "350m" => TTTConfig::default_350m(vocab_size),
        "760m" => TTTConfig::default_760m(vocab_size),
        "1b" => TTTConfig::default_1b(vocab_size),
        _ => panic!("Unknown model size: {model_size}. Use: 12m, 60m, 125m, 350m, 760m, 1b"),
    };
    config.with_mini_batch_size(mini_batch)
}

fn main() {
    let args = Args::parse();
    let config = make_config(&args.model_size, args.mini_batch);
    let layer_type = parse_layer_type(&args.ttt_type);

    let time_ms = if args.backward {
        let device: <GpuAutodiffBackend as Backend>::Device = Default::default();
        dispatch_ttt_layer_type!(bench_bwd::<GpuAutodiffBackend, layer_type>(&args, &config, &device))
    } else {
        let device: <GpuBackend as Backend>::Device = Default::default();
        dispatch_ttt_layer_type!(bench_fwd::<GpuBackend, layer_type>(&args, &config, &device))
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
        dtype: "float32",
        time_ms,
        throughput,
    };

    if args.json {
        println!("{}", serde_json::to_string(&result).unwrap());
    } else {
        println!("Burn TTT Benchmark");
        println!("==================");
        println!("Model: {} / {}", args.model_size, args.ttt_type);
        println!("Backward: {}", args.backward);
        println!("Batch: {}, Seq: {}, Mini-batch: {}", args.batch, args.seq_len, args.mini_batch);
        println!();
        println!("Time: {:.2} ms", time_ms);
        println!("Throughput: {:.0} tok/s", throughput);
    }
}
