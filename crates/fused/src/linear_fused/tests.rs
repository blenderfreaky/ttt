//! Tests for the naive fused TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, TTTLinearState,
    test_utils::{TestDims, test_backward_fmb, test_fmb},
};

use crate::FusedLinear;

// Tolerance constants for this kernel
const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-4;
const BACKWARD_RTOL: f32 = 2e-2;
const BACKWARD_ATOL: f32 = 1e-3;

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

#[test_case(2, 4, 16, 8 ; "batch2_heads4_dim16_seq8")]
#[test_case(1, 2, 8, 4 ; "batch1_heads2_dim8_seq4")]
#[test_case(4, 8, 32, 16 ; "batch4_heads8_dim32_seq16")]
fn test_fused_linear_forward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedLinear<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedLinear",
    );
}

// =============================================================================
// Backward tests
// =============================================================================

// TODO: investigate numerical issues in backward pass
#[test_case(2, 2, 8, 4 ; "batch2_heads2_dim8_seq4")]
#[ignore]
fn test_fused_linear_backward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_backward_fmb::<GpuAutodiffBackend, FusedLinear<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedLinear",
    );
}
