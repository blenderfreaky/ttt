use crate::{plane::cooperative_mma, test_kernel};
use cubecl::prelude::*;
use test_case::test_matrix;

#[cube(launch)]
fn test_cooperative_mma<F: Float>(
    a: &Tensor<Line<F>>,
    b: &Tensor<Line<F>>,
    c: &mut Tensor<Line<F>>,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
    #[comptime] tile_k: usize,
    #[comptime] threads_m: usize,
    #[comptime] threads_n: usize,
) {
    cooperative_mma(
        a, b, c, 0, 0, 0, tile_m, tile_n, tile_k, threads_m, threads_n,
    );
}

test_kernel! {
    /// Test cooperative matrix multiplication with various tile sizes.
    /// Uses a single cube (1 output tile) with multiple threads.
    #[test_matrix(
        [8, 16],    // tile_m
        [8, 16],    // tile_n
        [4, 8],     // tile_k
        [2],        // threads_m
        [2]         // threads_n
    )]
    fn test_cooperative_mma_single_tile(
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        threads_m: usize,
        threads_n: usize
    ) for F in [f32] {
        let a: Tensor = [tile_m, tile_k] as Range;
        let b: Tensor = [tile_k, tile_n] as Range;
        let c: Tensor = [tile_m, tile_n] as Range;

        assert_eq!(
            test_cooperative_mma(
                a(), b(), c(),
                lit(tile_m), lit(tile_n), lit(tile_k),
                lit(threads_m), lit(threads_n)
            ) for (1, 1, 1) @ (threads_m, threads_n),
            {
                // A[m, k] = a[m * K + k]
                // B[k, n] = b[k * N + n]
                // C[m, n] = sum_k(A[m, k] * B[k, n])
                for mi in 0..tile_m {
                    for ni in 0..tile_n {
                        let mut sum = F::from_int(0);
                        for ki in 0..tile_k {
                            let a_val = a[mi * tile_k + ki];
                            let b_val = b[ki * tile_n + ni];
                            sum += a_val * b_val;
                        }
                        c[mi * tile_n + ni] = sum;
                    }
                }
            }
        );
    }
}

test_kernel! {
    /// Test cooperative matmul with larger K dimension requiring multiple K-tiles.
    #[test_matrix(
        [8],        // tile_m
        [8],        // tile_n
        [4],        // tile_k
        [2, 8],     // num_k
        [2],        // threads_m
        [2]         // threads_n
    )]
    fn test_cooperative_mma_multi_k(
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        num_k: usize,
        threads_m: usize,
        threads_n: usize
    ) for F in [f32] {
        let a: Tensor = [tile_m, tile_k*num_k];
        let b: Tensor = [tile_k*num_k, tile_n];
        let c: Tensor = [tile_m, tile_n] as Range;

        assert_eq!(
            test_cooperative_mma(
                a(), b(), c(),
                lit(tile_m), lit(tile_n), lit(tile_k),
                lit(threads_m), lit(threads_n)
            ) for (1, 1, 1) @ (threads_m, threads_n),
            {
                for mi in 0..tile_m {
                    for ni in 0..tile_n {
                        let mut sum = F::from_int(0);
                        for ki in 0..tile_k*2 {
                            let a_val = a[mi * tile_k*2 + ki];
                            let b_val = b[ki * tile_n + ni];
                            sum += a_val * b_val;
                        }
                        c[mi * tile_n + ni] = sum;
                    }
                }
            }
        );
    }
}
