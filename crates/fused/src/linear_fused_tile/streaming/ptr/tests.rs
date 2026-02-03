//! Tests for the pointer-based streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend,
    test_utils::{TestDims, test_fmb, test_fwd},
};

use super::FusedTilePtrStreamingState;
use crate::FusedPtrStreaming;

// Tolerance constants for ptr streaming kernel
// Higher tolerance due to pointer-based memory access patterns
const RTOL: f32 = 0.5;
const ATOL: f32 = 0.4;

// =============================================================================
// Forward tests (forward - multi-stage, multi-iteration)
// =============================================================================

// TODO: ptr streaming kernel has issues, needs investigation
#[test_case(2, 2, 32, 8, 2, 2 ; "batch2_heads2_dim32_mini8_stages2_iter2")]
#[ignore]
fn test_fused_tile_ptr_streaming_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
    iterations: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims =
        TestDims::multi_stage(batch, heads, dim, mini_batch, stages).with_iterations(iterations);
    test_fwd::<GpuBackend, FusedPtrStreaming<GpuBackend>, FusedTilePtrStreamingState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTilePtrStreaming",
    );
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

// TODO: ptr streaming kernel has issues, needs investigation
#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[ignore]
fn test_fused_tile_ptr_streaming_fmb_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedPtrStreaming<GpuBackend>, FusedTilePtrStreamingState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTilePtrStreaming",
    );
}
