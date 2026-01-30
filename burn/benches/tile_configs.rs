//! Benchmark for testing various FusedTile configurations.
//!
//! Tests different (mini_batch_len, head_dim, threads) combinations to verify which work.
//!
//! Usage:
//!   cargo bench --features rocm --bench tile_configs

use burn::tensor::Tensor;
use criterion::{Criterion, criterion_group, criterion_main};
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
    let batch_size = 1;
    let num_heads = 1;

    let name = format!("tile_{}x{}@{}", mini_batch_len, head_dim, threads);

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

    c.bench_function(&name, |b| {
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
}

/// Test all supported tile configurations.
fn bench_tile_configs(c: &mut Criterion) {
    let device = device::<GpuBackend>();

    // Supported configurations from launch.rs:
    // (mini_batch_len, head_dim, threads)
    let configs = [(8, 32, 8), (8, 64, 8), (16, 32, 16), (32, 32, 32), (32, 64, 32)];

    for (mini_batch_len, head_dim, threads) in configs {
        test_tile_config::<GpuBackend>(c, mini_batch_len, head_dim, threads, &device);
    }
}

criterion_group!(tile_configs, bench_tile_configs);
criterion_main!(tile_configs);
