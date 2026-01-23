#![allow(non_snake_case)]

use crate::{
    prelude::*,
    tiles::{Dim, mma::{mma_AtB_rt, mma_AB_rt}},
};
use cubecl::prelude::*;

/// Computes C += A^T @ B where A is stored transposed as [K, M].
///
/// Each thread computes a sub-tile of C based on its UNIT_POS.
/// A should be loaded via load_st_transpose, B via load_st_direct.
///
/// Threads with UNIT_POS >= num_sub_tiles are safely skipped (Rt unchanged).
#[cube]
pub fn mma_AtB<F: Float, TileK: Dim, TileM: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    rt_c: &mut Rt<F, ThreadTileM, ThreadTileN>,
    st_a: &St<F, TileK, TileM>,
    st_b: &St<F, TileK, TileN>,
) {
    let threads_m = TileM::VALUE / ThreadTileM::VALUE;
    let threads_n = TileN::VALUE / ThreadTileN::VALUE;
    let num_tiles = threads_m * threads_n;

    // Guard: only threads with valid tile indices participate
    if (UNIT_POS as usize) < num_tiles {
        let tid = UNIT_POS as usize;
        let thread_m = tid / threads_n;
        let thread_n = tid % threads_n;

        let offset_m = ThreadTileM::LINES * thread_m;
        let offset_n = ThreadTileN::LINES * thread_n;

        mma_AtB_rt(rt_c, st_a, st_b, offset_m, offset_n);
    }
}

/// Computes C += A @ B where A is stored in original layout as [M, K].
///
/// Each thread computes a sub-tile of C based on its UNIT_POS.
/// Both A and B should be loaded via load_st_direct.
///
/// Threads with UNIT_POS >= num_sub_tiles are safely skipped (Rt unchanged).
#[cube]
pub fn mma_AB<F: Float, TileM: Dim, TileK: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    rt_c: &mut Rt<F, ThreadTileM, ThreadTileN>,
    st_a: &St<F, TileM, TileK>,
    st_b: &St<F, TileK, TileN>,
) {
    let threads_m = TileM::VALUE / ThreadTileM::VALUE;
    let threads_n = TileN::VALUE / ThreadTileN::VALUE;
    let num_tiles = threads_m * threads_n;

    // Guard: only threads with valid tile indices participate
    if (UNIT_POS as usize) < num_tiles {
        let tid = UNIT_POS as usize;
        let thread_m = tid / threads_n;
        let thread_n = tid % threads_n;

        let offset_m = ThreadTileM::LINES * thread_m;
        let offset_n = ThreadTileN::LINES * thread_n;

        mma_AB_rt(rt_c, st_a, st_b, offset_m, offset_n);
    }
}
