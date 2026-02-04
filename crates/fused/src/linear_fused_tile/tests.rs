//! Tests for the tiled fused TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, TTTLinearState,
    test_utils::{TestDims, test_backward_fmb, test_backward_fwd, test_fmb, test_fwd},
};

use crate::{FusedTile, FusedTileMulti};

// Tolerance constants for this kernel
// Tiled kernel has slightly higher tolerance due to accumulation differences
const RTOL: f32 = 1e-2;
const ATOL: f32 = 1e-3;
const BACKWARD_RTOL: f32 = 5e-2;
const BACKWARD_ATOL: f32 = 1e-3;

#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[test_case(1, 4, 32, 16 ; "batch1_heads4_dim32_seq16")]
#[test_case(2, 2, 64, 8 ; "batch2_heads2_dim64_seq8")]
fn test_fused_tile_forward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedTile<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTile",
    );
}

#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
fn test_fused_tile_backward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_backward_fmb::<GpuAutodiffBackend, FusedTile<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedTile",
    );
}

#[test_case(2, 2, 32, 8, 4 ; "batch2_heads2_dim32_mini8_stages4")]
#[test_case(1, 4, 32, 8, 2 ; "batch1_heads4_dim32_mini8_stages2")]
fn test_fused_tile_multi_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages);
    test_fwd::<GpuBackend, FusedTileMulti<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTileMulti",
    );
}

#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
fn test_fused_tile_multi_fmb_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedTileMulti<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTileMulti",
    );
}

#[test_case(2, 2, 32, 8, 2 ; "batch2_heads2_dim32_mini8_stages2")]
#[ignore]
fn test_fused_tile_multi_fmb_backward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages);
    test_backward_fmb::<GpuAutodiffBackend, FusedTileMulti<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedTileMulti",
    );
}

#[test_case(2, 2, 32, 8, 2 ; "batch2_heads2_dim32_mini8_stages2")]
#[ignore]
fn test_fused_tile_multi_backward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages);
    test_backward_fwd::<GpuAutodiffBackend, FusedTileMulti<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedTileMulti",
    );
}
