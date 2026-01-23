//! Tests for store_rt_direct - storing register tiles directly to global memory.
//!
//! This file tests the scenario where we want to bypass shared memory and store
//! directly from distributed register tiles to global memory.

use crate::{
    plane::{load_rt_from_st, load_st_direct, store_rt_direct, store_rt_to_st, store_st_direct},
    prelude::*,
    test_kernel,
    util::sync_planes,
};
use cubecl::prelude::*;

/// Test kernel: Global -> St -> Rt -> Global (direct, bypassing St on the way out)
///
/// This is the pattern that was failing in the fused TTT kernel.
/// Each thread loads its portion of the tile into registers, then stores directly
/// to global memory. The function computes per-thread offsets automatically.
#[cube(launch)]
fn rw_rt_direct_16x64<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut st = St::<F, D16, D64>::new();
    let mut rt = Rt::<F, D4, D16>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_planes();

    // St -> Rt (cooperative - each thread gets its sub-tile)
    load_rt_from_st::<F, D4, D16, D16, D64>(&st, &mut rt);

    sync_planes();

    // Rt -> Global directly (function computes per-thread offsets automatically)
    store_rt_direct::<F, D4, D16, D16, D64>(&rt, output, 0, 0, 0);
}

/// Same test but for 64x64 tiles (matching the weight matrix in TTT)
#[cube(launch)]
fn rw_rt_direct_64x64<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut st = St::<F, D64, D64>::new();
    let mut rt = Rt::<F, D16, D16>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_planes();

    // St -> Rt (cooperative)
    load_rt_from_st::<F, D16, D16, D64, D64>(&st, &mut rt);

    sync_planes();

    // Rt -> Global directly (function computes per-thread offsets automatically)
    store_rt_direct::<F, D16, D16, D64, D64>(&rt, output, 0, 0, 0);
}

/// Control test: same flow but using St as intermediate (the working pattern)
#[cube(launch)]
fn rw_rt_via_st_64x64<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut st = St::<F, D64, D64>::new();
    let mut rt = Rt::<F, D16, D16>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_planes();

    // St -> Rt
    load_rt_from_st::<F, D16, D16, D64, D64>(&st, &mut rt);

    sync_planes();

    // Rt -> St (cooperative)
    store_rt_to_st::<F, D16, D16, D64, D64>(&rt, &mut st);

    sync_planes();

    // St -> Global (cooperative)
    store_st_direct(&st, output, 0, 0, 0);
}

/// Test with batch dimension - multiple tiles stored to different offsets
#[cube(launch)]
fn rw_rt_direct_batched<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let batch_idx = CUBE_POS_X as usize;

    let mut st = St::<F, D64, D64>::new();
    let mut rt = Rt::<F, D16, D16>::new();

    // Compute base offset for this batch
    let base_offset = batch_idx * D64::VALUE * D64::VALUE;

    // Global -> St
    load_st_direct(input, &mut st, base_offset, 0, 0);

    sync_planes();

    // St -> Rt
    load_rt_from_st::<F, D16, D16, D64, D64>(&st, &mut rt);

    sync_planes();

    // Rt -> Global directly with batch offset (function computes per-thread offsets)
    store_rt_direct::<F, D16, D16, D64, D64>(&rt, output, base_offset, 0, 0);
}

/// More complex test: load two tiles, do mma, then store result directly
/// This mirrors the TTT kernel's weight update pattern
#[cube(launch)]
fn mma_then_store_rt_direct<F: Float>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    // A is [64, 16], B is [16, 64], output is [64, 64]
    let mut a_st = St::<F, D64, D16>::new();
    let mut b_st = St::<F, D16, D64>::new();
    let mut c_rt = Rt::<F, D16, D16>::new();

    // Load A and B
    load_st_direct(a_input, &mut a_st, 0, 0, 0);
    load_st_direct(b_input, &mut b_st, 0, 0, 0);

    sync_planes();

    // C = A @ B using mma_AB
    c_rt.zero();
    crate::plane::mma_AB(&mut c_rt, &a_st, &b_st);

    sync_planes();

    // Store C directly to global (function computes per-thread offsets)
    store_rt_direct::<F, D16, D16, D64, D64>(&c_rt, output, 0, 0, 0);
}

/// Control: same as above but store via St
#[cube(launch)]
fn mma_then_store_via_st<F: Float>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut a_st = St::<F, D64, D16>::new();
    let mut b_st = St::<F, D16, D64>::new();
    let mut c_st = St::<F, D64, D64>::new();
    let mut c_rt = Rt::<F, D16, D16>::new();

    load_st_direct(a_input, &mut a_st, 0, 0, 0);
    load_st_direct(b_input, &mut b_st, 0, 0, 0);

    sync_planes();

    c_rt.zero();
    crate::plane::mma_AB(&mut c_rt, &a_st, &b_st);

    sync_planes();

    // Store via St (the working pattern)
    store_rt_to_st::<F, D16, D16, D64, D64>(&c_rt, &mut c_st);

    sync_planes();

    store_st_direct(&c_st, output, 0, 0, 0);
}

/// Batched mma then store_rt_direct
#[cube(launch)]
fn mma_then_store_rt_direct_batched<F: Float>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let batch_idx = CUBE_POS_X as usize;

    let mut a_st = St::<F, D64, D16>::new();
    let mut b_st = St::<F, D16, D64>::new();
    let mut c_rt = Rt::<F, D16, D16>::new();

    let a_offset = batch_idx * D64::VALUE * D16::VALUE;
    let b_offset = batch_idx * D16::VALUE * D64::VALUE;
    let c_offset = batch_idx * D64::VALUE * D64::VALUE;

    load_st_direct(a_input, &mut a_st, a_offset, 0, 0);
    load_st_direct(b_input, &mut b_st, b_offset, 0, 0);

    sync_planes();

    c_rt.zero();
    crate::plane::mma_AB(&mut c_rt, &a_st, &b_st);

    sync_planes();

    // Store directly (function computes per-thread offsets)
    store_rt_direct::<F, D16, D16, D64, D64>(&c_rt, output, c_offset, 0, 0);
}

test_kernel! {
    #[test]
    fn test_rw_rt_direct_16x64() for F in all {
        let input: Tensor = [16, 64];
        let output: Tensor = [16, 64];

        assert_eq!(
            rw_rt_direct_16x64(input(), output()) for (1, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test]
    fn test_rw_rt_direct_64x64() for F in all {
        let input: Tensor = [64, 64];
        let output: Tensor = [64, 64];

        assert_eq!(
            rw_rt_direct_64x64(input(), output()) for (1, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test]
    fn test_rw_rt_via_st_64x64_control() for F in all {
        let input: Tensor = [64, 64];
        let output: Tensor = [64, 64];

        assert_eq!(
            rw_rt_via_st_64x64(input(), output()) for (1, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test]
    fn test_rw_rt_direct_batched_2() for F in all {
        let input: Tensor = [2, 64, 64];
        let output: Tensor = [2, 64, 64];

        assert_eq!(
            rw_rt_direct_batched(input(), output()) for (2, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test]
    fn test_rw_rt_direct_batched_4() for F in all {
        let input: Tensor = [4, 64, 64];
        let output: Tensor = [4, 64, 64];

        assert_eq!(
            rw_rt_direct_batched(input(), output()) for (4, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // MMA then store tests - these mirror the TTT kernel pattern
    // Only f32/f64 to avoid half-precision arithmetic issues in CPU reference
    #[test]
    fn test_mma_then_store_rt_direct() for F in [f32, f64] {
        let a: Tensor = [64, 16] as Range;
        let b: Tensor = [16, 64] as Range;
        let output: Tensor = [64, 64];

        assert_eq!(
            mma_then_store_rt_direct(a(), b(), output()) for (1, 1, 1) @ (32),
            {
                // Compute expected C = A @ B on CPU
                for i in 0..64 {
                    for j in 0..64 {
                        let mut sum = F::from_int(0);
                        for k in 0..16 {
                            sum += a[i * 16 + k] * b[k * 64 + j];
                        }
                        output[i * 64 + j] = sum;
                    }
                }
            }
        );
    }

    #[test]
    fn test_mma_then_store_via_st_control() for F in [f32, f64] {
        let a: Tensor = [64, 16] as Range;
        let b: Tensor = [16, 64] as Range;
        let output: Tensor = [64, 64];

        assert_eq!(
            mma_then_store_via_st(a(), b(), output()) for (1, 1, 1) @ (32),
            {
                for i in 0..64 {
                    for j in 0..64 {
                        let mut sum = F::from_int(0);
                        for k in 0..16 {
                            sum += a[i * 16 + k] * b[k * 64 + j];
                        }
                        output[i * 64 + j] = sum;
                    }
                }
            }
        );
    }

    #[test]
    fn test_mma_then_store_rt_direct_batched() for F in [f32, f64] {
        let a: Tensor = [4, 64, 16] as Range;
        let b: Tensor = [4, 16, 64] as Range;
        let output: Tensor = [4, 64, 64];

        assert_eq!(
            mma_then_store_rt_direct_batched(a(), b(), output()) for (4, 1, 1) @ (32),
            {
                for batch in 0..4 {
                    let a_off = batch * 64 * 16;
                    let b_off = batch * 16 * 64;
                    let c_off = batch * 64 * 64;
                    for i in 0..64 {
                        for j in 0..64 {
                            let mut sum = F::from_int(0);
                            for k in 0..16 {
                                sum += a[a_off + i * 16 + k] * b[b_off + k * 64 + j];
                            }
                            output[c_off + i * 64 + j] = sum;
                        }
                    }
                }
            }
        );
    }
}
