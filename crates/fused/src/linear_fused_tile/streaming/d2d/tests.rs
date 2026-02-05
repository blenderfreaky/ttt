//! Tests for the D2D streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend,
    test_utils::{TestDims, test_fmb, test_fwd},
};

use super::FusedTileD2dStreamingState;
use crate::FusedTileD2dStreaming;

const RTOL: f32 = 1e-2;
const ATOL: f32 = 1e-2;

// =============================================================================
// Forward tests (forward - multi-iteration)
// =============================================================================

#[test_case(2, 2, 32, 8, 2 ; "batch2_heads2_dim32_seq8_iter2")]
#[ignore]
fn test_fused_tile_streaming_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
    iterations: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq).with_iterations(iterations);
    test_fwd::<
        GpuBackend,
        FusedTileD2dStreaming<GpuBackend>,
        FusedTileD2dStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTileD2dStreaming");
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

#[test_case(1, 1, 64, 16 ; "batch1_heads1_dim64_seq16")]
#[test_case(1, 2, 32, 8 ; "batch1_heads2_dim32_seq8")]
#[test_case(2, 1, 32, 8 ; "batch2_heads1_dim32_seq8")]
#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[ignore]
fn test_fused_tile_streaming_fmb_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<
        GpuBackend,
        FusedTileD2dStreaming<GpuBackend>,
        FusedTileD2dStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTileD2dStreaming");
}
