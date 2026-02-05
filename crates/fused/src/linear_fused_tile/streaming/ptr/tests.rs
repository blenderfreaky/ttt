//! Tests for the pointer-based streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend,
    test_utils::{TestDims, test_fmb, test_fwd},
};

use super::FusedTilePtrStreamingState;
use crate::FusedTilePtrStreaming;

const RTOL: f32 = 0.5;
const ATOL: f32 = 0.4;

// =============================================================================
// Forward tests (forward - multi-stage, multi-iteration)
// =============================================================================

// Minimal test with 1 cube to isolate visibility issues
// NOTE: Multi-iteration tests are disabled because persistent kernels on AMD GPUs
// block compute resources even when sleeping, preventing other GPU work from running.
// The kernel works correctly for single iteration (which is the normal use case).
#[test_case(1, 1, 64, 16, 2, 1 ; "batch1_heads1_dim64_mini16_stages2_iter1")]
// #[test_case(1, 1, 64, 16, 2, 2 ; "batch1_heads1_dim64_mini16_stages2_iter2")]  // blocked by persistent kernel
// #[test_case(2, 2, 32, 8, 2, 2 ; "batch2_heads2_dim32_mini8_stages2_iter2")]    // blocked by persistent kernel
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
    test_fwd::<
        GpuBackend,
        FusedTilePtrStreaming<GpuBackend>,
        FusedTilePtrStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTilePtrStreaming");
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

// NOTE: Multi-cube tests are disabled because persistent kernels on AMD GPUs
// can starve workgroups, preventing all cubes from starting.
#[test_case(1, 1, 64, 16 ; "batch1_heads1_dim64_seq16")]
// #[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]  // blocked by workgroup starvation
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
    test_fmb::<
        GpuBackend,
        FusedTilePtrStreaming<GpuBackend>,
        FusedTilePtrStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTilePtrStreaming");
}
