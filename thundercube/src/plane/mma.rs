#![allow(non_snake_case)]

use crate::{
    prelude::*,
    tiles::{Dim, mma::mma_AtB_rt},
};
use cubecl::prelude::*;

#[cube]
pub fn mma_AtB<F: Float, TileK: Dim, TileM: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    rt_c: &mut Rt<F, ThreadTileM, ThreadTileN>,
    st_a: &St<F, TileK, TileM>,
    st_b: &St<F, TileK, TileN>,
) {
    let tid = UNIT_POS as usize;
    let threads_n = TileN::VALUE / ThreadTileN::VALUE;
    let thread_m = tid / threads_n;
    let thread_n = tid % threads_n;

    let offset_m = ThreadTileM::LINES * thread_m;
    let offset_n = ThreadTileN::LINES * thread_n;

    mma_AtB_rt(rt_c, st_a, st_b, offset_m, offset_n);
}

#[cube]
pub fn mma_AB<F: Float, TileK: Dim, TileM: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    rt_c: &mut Rt<F, ThreadTileM, ThreadTileN>,
    st_a: &St<F, TileM, TileK>,
    st_b: &St<F, TileK, TileN>,
) {
    todo!()
}
