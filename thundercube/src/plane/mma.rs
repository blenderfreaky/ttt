use crate::{
    plane::{load_st_direct, load_st_transpose, store_rt_direct},
    prelude::*,
    tiles::{Dim, mma::mma},
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
/// Each thread computes a `ThreadTileM x ThreadTileN` portion of the output.
/// Threads are arranged in a 2D grid: thread (i, j) handles output block at (i, j).
///
/// # Type Parameters
/// - `TileK`: Tile size in K dimension
/// - `TileM`: Tile size in M dimension (shared memory)
/// - `TileN`: Tile size in N dimension (shared memory)
/// - `ThreadTileM`: Per-thread tile size in M dimension
/// - `ThreadTileN`: Per-thread tile size in N dimension
#[cube]
pub fn cooperative_mma<
    F: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    a: &Tensor<Line<F>>,
    b: &Tensor<Line<F>>,
    c: &mut Tensor<Line<F>>,
    a_base_offset: usize,
    b_base_offset: usize,
    c_base_offset: usize,
) {
    // st_a: [K, M] transposed from [M, K]
    // st_b: [K, N]
    let mut st_a = St::<F, TileK, TileM>::new();
    let mut st_b = St::<F, TileK, TileN>::new();

    let mut rt_c = Rt::<F, ThreadTileM, ThreadTileN>::new();

    rt_c.zero();

    let k_dim = a.shape(a.rank() - 1);
    let num_k_tiles = div_ceil::<usize>(k_dim, TileK::VALUE);

    let tid = UNIT_POS as usize;
    let threads_n = TileN::VALUE / ThreadTileN::VALUE;
    let thread_m = tid / threads_n;
    let thread_n = tid % threads_n;

    let offset_m = ThreadTileM::LINES * thread_m;
    let offset_n = ThreadTileN::LINES * thread_n;

    let block_m = CUBE_POS_X as usize * TileM::VALUE;
    let block_n = CUBE_POS_Y as usize * TileN::VALUE;

    for k_tile_idx in 0..num_k_tiles {
        let k_offset = k_tile_idx * TileK::VALUE;

        load_st_transpose(a, &mut st_a, a_base_offset, block_m, k_offset);

        load_st_direct(b, &mut st_b, b_base_offset, k_offset, block_n);

        sync_cube();

        mma(&mut rt_c, &st_a, &st_b, offset_m, offset_n);

        sync_cube();
    }

    let c_row = block_m + thread_m * ThreadTileM::VALUE;
    let c_col = block_n + thread_n * ThreadTileN::VALUE;

    store_rt_direct(&rt_c, c, c_base_offset, c_row, c_col);
}
