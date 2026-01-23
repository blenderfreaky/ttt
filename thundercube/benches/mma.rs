#![allow(non_snake_case)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use pollster::block_on;
use thundercube::{
    LINE_SIZE,
    plane::{load_st_direct, load_st_transpose, mma_AB, mma_AtB, store_rt_direct},
    test_utils::{client, get_strides, upload},
    tiles::{D16, D32, D4, D8, Dim, DimOrOne, Rt, St},
};

type TestRuntime = cubecl::TestRuntime;

/// Benchmark kernel for mma_AtB with configurable tile sizes.
/// C = A^T * B where A is loaded transposed.
#[cube(launch)]
fn bench_mma_AtB<
    F: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st_a = St::<F, TileK, TileM>::new();
    let mut st_b = St::<F, TileK, TileN>::new();
    let mut rt_c = Rt::<F, ThreadTileM, ThreadTileN>::new();
    rt_c.zero();

    load_st_transpose(in_a, &mut st_a, 0, 0, 0);
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma_AtB(&mut rt_c, &st_a, &st_b);

    store_rt_direct::<F, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_c, output, 0, 0, 0);
}

/// Benchmark kernel for mma_AB with configurable tile sizes.
/// C = A * B where A is loaded directly (not transposed).
#[cube(launch)]
fn bench_mma_AB<F: Float, TileM: Dim, TileK: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st_a = St::<F, TileM, TileK>::new();
    let mut st_b = St::<F, TileK, TileN>::new();
    let mut rt_c = Rt::<F, ThreadTileM, ThreadTileN>::new();
    rt_c.zero();

    load_st_direct(in_a, &mut st_a, 0, 0, 0);
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma_AB(&mut rt_c, &st_a, &st_b);

    store_rt_direct::<F, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_c, output, 0, 0, 0);
}

/// Run mma_AtB benchmark with specific tile dimensions
macro_rules! bench_mma_AtB_impl {
    ($c:expr, $group_name:expr, $tile_m:ty, $tile_k:ty, $tile_n:ty, $thread_tile:ty) => {{
        let client = client();

        let tile_m = <$tile_m>::VALUE;
        let tile_k = <$tile_k>::VALUE;
        let tile_n = <$tile_n>::VALUE;
        let thread_tile = <$thread_tile>::VALUE;

        let num_threads = (tile_m / thread_tile) * (tile_n / thread_tile);

        // Create data
        let data_a: Vec<f32> = (0..(tile_m * tile_k)).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..(tile_k * tile_n)).map(|i| i as f32).collect();
        let data_c: Vec<f32> = vec![0.0; tile_m * tile_n];

        let handle_a = upload(&client, &data_a);
        let handle_b = upload(&client, &data_b);
        let handle_c = upload(&client, &data_c);

        let shape_a = vec![tile_m, tile_k];
        let shape_b = vec![tile_k, tile_n];
        let shape_c = vec![tile_m, tile_n];

        let strides_a = get_strides(&shape_a);
        let strides_b = get_strides(&shape_b);
        let strides_c = get_strides(&shape_c);

        let flops = 2 * tile_m * tile_k * tile_n;
        let param_str = format!("M{}K{}N{}_T{}", tile_m, tile_k, tile_n, thread_tile);

        $c.throughput(Throughput::Elements(flops as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let in_a = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides_a, &shape_a, LINE_SIZE)
                };
                let in_b = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_b, &strides_b, &shape_b, LINE_SIZE)
                };
                let output = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_c, &strides_c, &shape_c, LINE_SIZE)
                };

                bench_mma_AtB::launch::<f32, $tile_k, $tile_m, $tile_n, $thread_tile, $thread_tile, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(num_threads as u32),
                    in_a,
                    in_b,
                    output,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

/// Run mma_AB benchmark with specific tile dimensions
macro_rules! bench_mma_AB_impl {
    ($c:expr, $group_name:expr, $tile_m:ty, $tile_k:ty, $tile_n:ty, $thread_tile:ty) => {{
        let client = client();

        let tile_m = <$tile_m>::VALUE;
        let tile_k = <$tile_k>::VALUE;
        let tile_n = <$tile_n>::VALUE;
        let thread_tile = <$thread_tile>::VALUE;

        let num_threads = (tile_m / thread_tile) * (tile_n / thread_tile);

        // Create data
        let data_a: Vec<f32> = (0..(tile_m * tile_k)).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..(tile_k * tile_n)).map(|i| i as f32).collect();
        let data_c: Vec<f32> = vec![0.0; tile_m * tile_n];

        let handle_a = upload(&client, &data_a);
        let handle_b = upload(&client, &data_b);
        let handle_c = upload(&client, &data_c);

        let shape_a = vec![tile_m, tile_k];
        let shape_b = vec![tile_k, tile_n];
        let shape_c = vec![tile_m, tile_n];

        let strides_a = get_strides(&shape_a);
        let strides_b = get_strides(&shape_b);
        let strides_c = get_strides(&shape_c);

        let flops = 2 * tile_m * tile_k * tile_n;
        let param_str = format!("M{}K{}N{}_T{}", tile_m, tile_k, tile_n, thread_tile);

        $c.throughput(Throughput::Elements(flops as u64));
        $c.bench_with_input(BenchmarkId::new($group_name, &param_str), &(), |b, _| {
            b.iter(|| {
                let in_a = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides_a, &shape_a, LINE_SIZE)
                };
                let in_b = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_b, &strides_b, &shape_b, LINE_SIZE)
                };
                let output = unsafe {
                    TensorArg::from_raw_parts::<Line<f32>>(&handle_c, &strides_c, &shape_c, LINE_SIZE)
                };

                bench_mma_AB::launch::<f32, $tile_m, $tile_k, $tile_n, $thread_tile, $thread_tile, TestRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(num_threads as u32),
                    in_a,
                    in_b,
                    output,
                )
                .expect("Kernel launch failed");
                block_on(client.sync()).expect("Sync failed");
            })
        });
    }};
}

fn bench_mma_AtB_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mma_AtB");

    // Small tiles (4x4)
    bench_mma_AtB_impl!(group, "mma_AtB", D4, D4, D4, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D8, D4, D8, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D8, D8, D8, D4);

    // Medium tiles
    bench_mma_AtB_impl!(group, "mma_AtB", D16, D4, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D16, D8, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D16, D16, D16, D4);

    // Larger tiles
    bench_mma_AtB_impl!(group, "mma_AtB", D32, D8, D32, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D32, D16, D32, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D32, D32, D32, D4);

    // Asymmetric tiles
    bench_mma_AtB_impl!(group, "mma_AtB", D8, D4, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D16, D4, D8, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D16, D8, D32, D4);
    bench_mma_AtB_impl!(group, "mma_AtB", D32, D8, D16, D4);

    group.finish();
}

fn bench_mma_AB_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mma_AB");

    // Small tiles
    bench_mma_AB_impl!(group, "mma_AB", D4, D4, D4, D4);
    bench_mma_AB_impl!(group, "mma_AB", D8, D4, D8, D4);
    bench_mma_AB_impl!(group, "mma_AB", D8, D8, D8, D4);

    // Medium tiles
    bench_mma_AB_impl!(group, "mma_AB", D16, D4, D16, D4);
    bench_mma_AB_impl!(group, "mma_AB", D16, D8, D16, D4);
    bench_mma_AB_impl!(group, "mma_AB", D16, D16, D16, D4);

    // Larger tiles
    bench_mma_AB_impl!(group, "mma_AB", D32, D8, D32, D4);
    bench_mma_AB_impl!(group, "mma_AB", D32, D16, D32, D4);
    bench_mma_AB_impl!(group, "mma_AB", D32, D32, D32, D4);

    // Asymmetric tiles
    bench_mma_AB_impl!(group, "mma_AB", D8, D4, D16, D4);
    bench_mma_AB_impl!(group, "mma_AB", D16, D4, D8, D4);
    bench_mma_AB_impl!(group, "mma_AB", D16, D8, D32, D4);
    bench_mma_AB_impl!(group, "mma_AB", D32, D8, D16, D4);

    group.finish();
}

fn bench_k_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mma_K_scaling");

    // Test how K dimension affects performance with fixed M=N=16
    bench_mma_AtB_impl!(group, "mma_AtB_K", D16, D4, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_K", D16, D8, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_K", D16, D16, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_K", D16, D32, D16, D4);

    group.finish();
}

fn bench_tile_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mma_tile_scaling");

    // Test square tiles with K=8 fixed
    bench_mma_AtB_impl!(group, "mma_AtB_scale", D4, D8, D4, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_scale", D8, D8, D8, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_scale", D16, D8, D16, D4);
    bench_mma_AtB_impl!(group, "mma_AtB_scale", D32, D8, D32, D4);

    group.finish();
}

criterion_group!(
    benches,
    bench_mma_AtB_tile_sizes,
    bench_mma_AB_tile_sizes,
    bench_k_dimension_scaling,
    bench_tile_size_scaling,
);
criterion_main!(benches);
