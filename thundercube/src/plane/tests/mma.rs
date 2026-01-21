use crate::prelude::*;
use cubecl::prelude::*;
use test_case::test_matrix;

#[cube(launch)]
fn test_cooperative_mma<
    F: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    a: &Tensor<Line<F>>,
    b: &Tensor<Line<F>>,
    c: &mut Tensor<Line<F>>,
) {
    plane::cooperative_mma::<F, TileK, TileM, TileN, ThreadTileM, ThreadTileN>(a, b, c, 0, 0, 0);
}

test_kernel! {
    /// Test cooperative matrix multiplication with various tile sizes.
    /// Uses a single cube (1 output tile) with multiple threads.
    #[test]
    fn test_cooperative_mma_single_tile() for
        F in all
        TileK in [D8, D16]
        TileM in [D8, D16]
        TileN in [D4, D8]
        ThreadTileM in [D4]
        ThreadTileN in [D4, D8]
    {
        let a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
        let b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
        let c: Tensor = [TileM::VALUE, TileN::VALUE] as Range;

        assert_eq!(
            test_cooperative_mma(
                a(), b(), c(),
            ) for (1, 1, 1) @ (TileM::VALUE/ThreadTileM::VALUE, TileN::VALUE/ThreadTileN::VALUE),
            {
                // A[m, k] = a[m * K + k]
                // B[k, n] = b[k * N + n]
                // C[m, n] = sum_k(A[m, k] * B[k, n])
                for mi in 0..TileK::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = F::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = a[mi * TileK::VALUE + ki];
                            let b_val = b[ki * TileN::VALUE + ni];
                            sum += a_val * b_val;
                        }
                        c[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }
}

test_kernel! {
    /// Test cooperative matmul with larger K dimension requiring multiple K-tiles.
    #[test_matrix([4, 8])]
    fn test_cooperative_mma_multi_k(
        num_k: usize,
    ) for
        F in [f32]
        TileK in [D32]
        TileM in [D32]
        TileN in [D32]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let a: Tensor = [TileM::VALUE, TileK::VALUE * num_k];
        let b: Tensor = [TileK::VALUE * num_k, TileN::VALUE];
        let c: Tensor = [TileM::VALUE, TileN::VALUE] as Range;

        assert_eq!(
            test_cooperative_mma(
                a(), b(), c(),
            ) for (1, 1, 1) @ (TileM::VALUE/ThreadTileM::VALUE, TileN::VALUE/ThreadTileN::VALUE),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = F::from_int(0);
                        for ki in 0..TileK::VALUE * num_k {
                            let a_val = a[mi * TileK::VALUE * num_k + ki];
                            let b_val = b[ki * TileN::VALUE + ni];
                            sum += a_val * b_val;
                        }
                        c[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }
}
