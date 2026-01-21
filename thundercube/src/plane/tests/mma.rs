#![allow(non_snake_case)]

use crate::{
    plane::{load_st_direct, load_st_transpose, mma_AtB, store_rt_direct},
    prelude::*,
    test_kernel,
};
use cubecl::prelude::*;

/// Test kernel that performs C = A * B using the mma_AtB function.
///
/// Layout:
/// - A (in_a): Row-major [M, K] in global, loaded transposed into shared memory as [K, M]
/// - B (in_b): Row-major [K, N]
/// - C (output): Row-major [M, N]
///
/// The plane divides work among threads, with each thread computing a ThreadTileM x ThreadTileN
/// sub-tile of the output.
#[cube(launch)]
fn test_mma_AtB<
    F: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    // st_a is [K, M] - loaded from A which is [M, K] via transpose
    let mut st_a = St::<F, TileK, TileM>::new();
    // st_b is [K, N]
    let mut st_b = St::<F, TileK, TileN>::new();

    let mut rt_c = Rt::<F, ThreadTileM, ThreadTileN>::new();
    rt_c.zero();

    // Load A as [M, K], transpose to [K, M] in shared memory
    load_st_transpose(in_a, &mut st_a, 0, 0, 0);
    // Load B as [K, N] directly
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma_AtB(&mut rt_c, &st_a, &st_b);

    // Calculate thread offsets for storing
    let tid = UNIT_POS as usize;
    let threads_n = TileN::VALUE / ThreadTileN::VALUE;
    let thread_m = tid / threads_n;
    let thread_n = tid % threads_n;

    let offset_m = ThreadTileM::VALUE * thread_m;
    let offset_n = ThreadTileN::VALUE * thread_n;

    store_rt_direct(&rt_c, output, 0, offset_m, offset_n);
}

test_kernel! {
    #[test]
    fn test_mma_AtB() for
        F in [f32, f64]
        TileK in [D4, D8]
        TileM in [D4, D8, D16]
        TileN in [D4, D8, D16]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
        let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            test_mma_AtB(in_a(), in_b(), output()) for (1, 1, 1) @ ((TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE)),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = F::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = in_a[mi * TileK::VALUE + ki];
                            let b_val = in_b[ki * TileN::VALUE + ni];
                            sum += a_val * b_val;
                        }
                        output[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }
}
