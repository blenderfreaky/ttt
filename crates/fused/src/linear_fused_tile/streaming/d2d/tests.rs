//! Tests for the D2D streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend,
    test_utils::{TestDims, test_fmb, test_fwd},
};

use super::FusedTileD2dStreamingState;
use crate::FusedTileD2dStreaming;

// Tolerance constants for streaming kernel
const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-3;

// =============================================================================
// Forward tests (forward - multi-iteration)
// =============================================================================

// TODO: streaming kernel has issues, needs investigation
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
    test_fwd::<GpuBackend, FusedTileD2dStreaming<GpuBackend>, FusedTileD2dStreamingState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTileD2dStreaming",
    );
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

// TODO: streaming kernel has issues, needs investigation
#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[ignore]
fn test_fused_tile_streaming_fmb_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedTileD2dStreaming<GpuBackend>, FusedTileD2dStreamingState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTileD2dStreaming",
    );
}
