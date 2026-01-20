use crate::{
    plane::{load_st_direct, load_st_transpose, store_rt_direct},
    prelude::*,
    test_kernel,
    tiles::mma::mma,
};
use cubecl::prelude::*;
use test_case::test_matrix;

/// Test kernel that performs C = A * B using the mma function.
///
/// Layout:
/// - A (in_a): Row-major [M, K] in global, loaded transposed into shared memory as [K, M]
/// - B (in_b): Row-major [K, N]
/// - C (output): Row-major [M, N]
#[cube(launch)]
fn test_mma<F: Float>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] m: usize,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    // transposed
    let mut st_a = St::<F>::new(k, m);
    let mut st_b = St::<F>::new(k, n);

    let mut rt_c = Rt::<F>::new(m, n);

    rt_c.zero();

    load_st_transpose(in_a, &mut st_a, 0, 0, 0);
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma(&mut rt_c, &st_a, &st_b, 0, 0);

    store_rt_direct(&rt_c, output, 0, 0, 0);
}

test_kernel! {
    /// Test matrix multiplication with various sizes
    #[test_matrix([4, 8, 16], [4, 8], [4, 8, 16])]
    fn test_mma(m: usize, k: usize, n: usize) for F in [f32, f64] {
        // All are row-major in global memory
        let in_a: Tensor = [m, k] as Range;
        let in_b: Tensor = [k, n] as Range;
        let output: Tensor = [m, n];

        assert_eq!(
            test_mma(in_a(), in_b(), output(), lit(m), lit(k), lit(n)) for (1, 1, 1) @ (1),
            {
                // A is [M, K] row-major, so A[m, k] = in_a[m * K + k]
                // B is [K, N] row-major, so B[k, n] = in_b[k * N + n]
                // C is [M, N] row-major, so C[m, n] = output[m * N + n]
                for mi in 0..m {
                    for ni in 0..n {
                        let mut sum = F::from_int(0);
                        for ki in 0..k {
                            let a_val = in_a[mi * k + ki];
                            let b_val = in_b[ki * n + ni];
                            sum += a_val * b_val;
                        }
                        output[mi * n + ni] = sum;
                    }
                }
            }
        );
    }
}
