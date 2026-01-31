//! Benchmark for testing various FusedTile configurations.
//!
//! Tests different (mini_batch_len, head_dim, threads) combinations to find optimal thread counts.
//!
//! Usage:
//!   cargo bench --features rocm --bench tile_configs

use burn::tensor::Tensor;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ttt::{
    GpuBackend,
    ttt::cubecl_kernels::{
        FusedTttBackend, FusedTttConfig, linear_fused_tile::fused_ttt_tile_forward,
    },
};

fn device<B: FusedTttBackend>() -> B::Device {
    Default::default()
}

/// Force async operations to complete.
fn sync<B: FusedTttBackend, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

/// Test a specific (mini_batch_len, head_dim, threads) configuration.
fn test_tile_config<B: FusedTttBackend>(
    c: &mut Criterion,
    mini_batch_len: usize,
    head_dim: usize,
    threads: usize,
    device: &B::Device,
) {
    let batch_size = 4;
    let num_heads = 2;

    let group_name = format!("{}x{}", mini_batch_len, head_dim);

    // Create input tensors
    let xq = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let xk = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let xv = Tensor::<B, 4>::random(
        [batch_size, num_heads, mini_batch_len, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let weight = Tensor::<B, 4>::random(
        [batch_size, num_heads, head_dim, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let bias = Tensor::<B, 3>::random(
        [batch_size, num_heads, head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let token_eta = Tensor::<B, 1>::random(
        [mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ttt_lr_eta = Tensor::<B, 3>::random(
        [batch_size, num_heads, mini_batch_len],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ln_weight = Tensor::<B, 2>::random(
        [head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let ln_bias = Tensor::<B, 2>::random(
        [head_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let epsilon = 1e-6f32;
    let config = FusedTttConfig::new(mini_batch_len, head_dim, epsilon, threads);

    // Warmup
    for _ in 0..3 {
        let (output, _, _) = fused_ttt_tile_forward::<B>(
            xq.clone(),
            xk.clone(),
            xv.clone(),
            weight.clone(),
            bias.clone(),
            token_eta.clone(),
            ttt_lr_eta.clone(),
            ln_weight.clone(),
            ln_bias.clone(),
            config,
        );
        sync::<B, 4>(output);
    }

    let mut group = c.benchmark_group(&group_name);
    let total_elements = batch_size * num_heads * mini_batch_len * head_dim;
    group.throughput(Throughput::Elements(total_elements as u64));

    group.bench_function(BenchmarkId::new("threads", threads), |b| {
        b.iter(|| {
            let (output, _weight_out, _bias_out) = fused_ttt_tile_forward::<B>(
                xq.clone(),
                xk.clone(),
                xv.clone(),
                weight.clone(),
                bias.clone(),
                token_eta.clone(),
                ttt_lr_eta.clone(),
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );
            sync::<B, 4>(output);
        })
    });

    group.finish();
}

/// Test all supported tile configurations with multiple thread counts.
fn bench_tile_configs(c: &mut Criterion) {
    let device = device::<GpuBackend>();

    // Tile sizes from launch.rs supported_tile_configs
    // Note: 16x128, 32x64, 64x64 removed due to GPU shared memory limits
    let tile_sizes = [
        (8, 32),
        (8, 64),
        (16, 32),
        (16, 64),
        (32, 32),
    ];

    // Thread counts to test for each tile size
    // Cover common powers of 2 and intermediate values
    let thread_counts = [4, 8, 16, 32, 64, 128, 256];

    for (mini_batch_len, head_dim) in tile_sizes {
        println!("\nTesting tile {}x{}", mini_batch_len, head_dim);
        for threads in thread_counts {
            // Skip clearly invalid configs (threads much larger than tile)
            if threads > mini_batch_len * head_dim / 4 {
                continue;
            }

            print!("  threads={:3}... ", threads);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                test_tile_config::<GpuBackend>(c, mini_batch_len, head_dim, threads, &device);
            })) {
                Ok(_) => println!("✓"),
                Err(_) => println!("✗ (panicked)"),
            }
        }
    }
}

criterion_group!(tile_configs, bench_tile_configs);
criterion_main!(tile_configs);
