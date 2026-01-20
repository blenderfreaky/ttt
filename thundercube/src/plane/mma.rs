use crate::{
    plane::{load_st_direct, load_st_transpose, store_rt_direct},
    prelude::*,
    tiles::mma::mma,
};
use cubecl::prelude::*;

/// Cooperative matrix multiplication: C = A * B
///
/// # Layout Assumptions
/// - A: Row-major [M, K] in global memory
/// - B: Row-major [K, N] in global memory
/// - C: Row-major [M, N] in global memory
///
/// # Thread Mapping
/// Each thread computes a `(TILE_M / THREADS_M) x (TILE_N / THREADS_N)` portion of the output.
/// Threads are arranged in a 2D grid: thread (i, j) handles output block at (i, j).
///
/// # Compile-Time Parameters
/// - `TILE_M`: Tile size in M dimension (must be divisible by LINE_SIZE and THREADS_M)
/// - `TILE_N`: Tile size in N dimension (must be divisible by LINE_SIZE and THREADS_N)
/// - `TILE_K`: Tile size in K dimension (must be divisible by LINE_SIZE)
/// - `THREADS_M`: Number of threads along M dimension
/// - `THREADS_N`: Number of threads along N dimension
#[cube]
pub fn cooperative_mma<F: Float>(
    a: &Tensor<Line<F>>,
    b: &Tensor<Line<F>>,
    c: &mut Tensor<Line<F>>,
    a_base_offset: usize,
    b_base_offset: usize,
    c_base_offset: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
    #[comptime] tile_k: usize,
    #[comptime] threads_m: usize,
    #[comptime] threads_n: usize,
) {
    let thread_tile_m = comptime!(tile_m / threads_m);
    let thread_tile_n = comptime!(tile_n / threads_n);

    // Transposed
    let mut st_a = St::<F>::new(tile_k, tile_m);
    let mut st_b = St::<F>::new(tile_k, tile_n);

    let mut rt_c = Rt::<F>::new(thread_tile_m, thread_tile_n);

    rt_c.zero();

    let k_dim = a.shape(a.rank() - 1);
    let num_k_tiles = div_ceil::<usize>(k_dim, tile_k);

    let tid = UNIT_POS as usize;
    let thread_m = tid / threads_n;
    let thread_n = tid % threads_n;

    let offset_m = comptime!(thread_tile_m / LINE_SIZE) * thread_m;
    let offset_n = comptime!(thread_tile_n / LINE_SIZE) * thread_n;

    let block_m = CUBE_POS_X as usize * tile_m;
    let block_n = CUBE_POS_Y as usize * tile_n;

    for k_tile_idx in 0..num_k_tiles {
        let k_offset = k_tile_idx * tile_k;

        load_st_transpose(a, &mut st_a, a_base_offset, block_m, k_offset);

        load_st_direct(b, &mut st_b, b_base_offset, k_offset, block_n);

        sync_cube();

        mma(&mut rt_c, &st_a, &st_b, offset_m, offset_n);

        sync_cube();
    }

    let c_row = block_m + thread_m * thread_tile_m;
    let c_col = block_n + thread_n * thread_tile_n;

    store_rt_direct(&rt_c, c, c_base_offset, c_row, c_col);
}
