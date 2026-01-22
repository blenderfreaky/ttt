#![allow(non_snake_case)]

use crate::{plane::swizzle, prelude::*, tiles::Dim};
use cubecl::prelude::*;

/// C += A * B
/// Supports arbitrary C-tile sizes (e.g., 8x8, 16x8).
///
/// LAYOUT ASSUMPTIONS:
/// - A: Stored in shared memory as [K, M] row-major with swizzle.
///      (Loaded from row-major [M, K] via load_st_transpose)
/// - B: Stored in shared memory as [K, N] row-major with swizzle.
///      (Loaded from row-major [K, N] via load_st_direct)
/// - C: Row-Major Register Tile [M, N].
///
/// Type parameters:
/// - CM, CN: C tile dimensions (rows, cols)
/// - K, AM: A tile dimensions (K rows, M cols) - A is transposed [K, M]
/// - K, BN: B tile dimensions (K rows, N cols)
///
/// 'offset_m/n' are the base offsets (in Lines) for the sub-tile this thread processes.
#[cube]
pub fn mma_AtB_rt<F: Float, CM: Dim, CN: Dim, K: Dim, AM: Dim, BN: Dim>(
    c: &mut Rt<F, CM, CN>,
    a: &St<F, K, AM>,
    b: &St<F, K, BN>,
    offset_m: usize,
    offset_n: usize,
) {
    // A is [K, M]: K::VALUE rows, AM::VALUE cols
    // B is [K, N]: K::VALUE rows, BN::VALUE cols
    let k_dim = K::VALUE;

    let num_m_vecs = CM::LINES;
    let num_n_vecs = CN::LINES;

    // Row-major strides (in Lines)
    let a_stride = AM::LINES; // M / LINE_SIZE
    let b_stride = BN::LINES; // N / LINE_SIZE

    // Swizzle masks
    let a_mask = a_stride - 1;
    let b_mask = b_stride - 1;

    for k in 0..k_dim {
        #[unroll(CN::LINES <= UNROLL_LIMIT_HOT)]
        for nl in 0..num_n_vecs {
            // B-Vector: row k, column (offset_n + nl) in Lines
            let b_col = offset_n + nl;
            let b_phys_col = swizzle(k, b_col, b_mask);
            let b_idx = k * b_stride + b_phys_col;
            let b_vec = b.data[b_idx];

            #[unroll(CM::LINES <= UNROLL_LIMIT_HOT)]
            for ml in 0..num_m_vecs {
                // A-Vector: row k, column (offset_m + ml) in Lines
                let a_col = offset_m + ml;
                let a_phys_col = swizzle(k, a_col, a_mask);
                let a_idx = k * a_stride + a_phys_col;
                let a_vec = a.data[a_idx];

                // 4x4 Outer Product on sub-block
                #[unroll]
                for i in 0..LINE_SIZE {
                    let a_val = a_vec[i];

                    let c_row = (ml * LINE_SIZE) + i;
                    let c_idx = (c_row * num_n_vecs) + nl;

                    c.data[c_idx] += Line::empty(LINE_SIZE).fill(a_val) * b_vec;
                }
            }
        }
    }
}

/// C += A * B (A is stored in original layout, not transposed)
/// Supports arbitrary C-tile sizes (e.g., 8x8, 16x8).
///
/// LAYOUT ASSUMPTIONS:
/// - A: Stored in shared memory as [M, K] row-major with swizzle.
///      (Loaded from row-major [M, K] via load_st_direct)
/// - B: Stored in shared memory as [K, N] row-major with swizzle.
///      (Loaded from row-major [K, N] via load_st_direct)
/// - C: Row-Major Register Tile [M, N].
///
/// Type parameters:
/// - CM, CN: C tile dimensions (rows, cols)
/// - AM, K: A tile dimensions (M rows, K cols)
/// - K, BN: B tile dimensions (K rows, N cols)
///
/// 'offset_m/n' are the base offsets (in Lines) for the sub-tile this thread processes.
#[cube]
pub fn mma_AB_rt<F: Float, CM: Dim, CN: Dim, AM: Dim, K: Dim, BN: Dim>(
    c: &mut Rt<F, CM, CN>,
    a: &St<F, AM, K>,
    b: &St<F, K, BN>,
    offset_m: usize,
    offset_n: usize,
) {
    // A is [M, K]: AM::VALUE rows, K::VALUE cols
    // B is [K, N]: K::VALUE rows, BN::VALUE cols

    let num_m_vecs = CM::LINES;
    let num_n_vecs = CN::LINES;

    // Row-major strides (in Lines)
    let a_stride = K::LINES; // K / LINE_SIZE (cols of A)
    let b_stride = BN::LINES; // N / LINE_SIZE (cols of B)

    // Swizzle masks
    let a_mask = a_stride - 1;
    let b_mask = b_stride - 1;

    // Iterate over K
    for k in 0..K::VALUE {
        let k_line = k / LINE_SIZE;
        let k_elem = k % LINE_SIZE;

        // For each output column (vectorized)
        #[unroll(CN::LINES <= UNROLL_LIMIT_HOT)]
        for nl in 0..num_n_vecs {
            // B[k, n]: row k, column (offset_n + nl) in Lines
            let b_col = offset_n + nl;
            let b_phys_col = swizzle(k, b_col, b_mask);
            let b_idx = k * b_stride + b_phys_col;
            let b_vec = b.data[b_idx];

            // For each output row (vectorized)
            #[unroll(CM::LINES <= UNROLL_LIMIT_HOT)]
            for ml in 0..num_m_vecs {
                #[unroll]
                for mi in 0..LINE_SIZE {
                    // Global row index into A
                    let m = (offset_m + ml) * LINE_SIZE + mi;

                    // A[m, k]: row m, column k (k_line-th Line, k_elem-th element)
                    let a_phys_col = swizzle(m, k_line, a_mask);
                    let a_idx = m * a_stride + a_phys_col;
                    let a_val = a.data[a_idx][k_elem];

                    // Update C[ml*LINE_SIZE + mi, :]
                    let c_row = ml * LINE_SIZE + mi;
                    let c_idx = c_row * num_n_vecs + nl;

                    c.data[c_idx] += Line::empty(LINE_SIZE).fill(a_val) * b_vec;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plane::{load_st_direct, load_st_transpose}, test_kernel};

    /// Test kernel for mma_AB_rt with offset_m=0, offset_n=0 (full tile)
    #[cube(launch)]
    fn test_mma_AB_rt_kernel<F: Float, TileM: Dim, TileK: Dim, TileN: Dim>(
        in_a: &Tensor<Line<F>>,
        in_b: &Tensor<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st_a = St::<F, TileM, TileK>::new();
        let mut st_b = St::<F, TileK, TileN>::new();

        load_st_direct(in_a, &mut st_a, 0, 0, 0);
        load_st_direct(in_b, &mut st_b, 0, 0, 0);

        let mut rt_c = Rt::<F, TileM, TileN>::new();
        rt_c.zero();

        mma_AB_rt(&mut rt_c, &st_a, &st_b, 0, 0);

        rt_c.copy_to_array(output);
    }

    /// Test kernel for mma_AB_rt with non-zero offsets (8x8 tiles, 4x4 sub-tile)
    #[cube(launch)]
    fn test_mma_AB_rt_offset_kernel<F: Float>(
        in_a: &Tensor<Line<F>>,
        in_b: &Tensor<Line<F>>,
        output: &mut Array<Line<F>>,
        offset_m: u32,
        offset_n: u32,
    ) {
        let mut st_a = St::<F, D8, D8>::new();
        let mut st_b = St::<F, D8, D8>::new();

        load_st_direct(in_a, &mut st_a, 0, 0, 0);
        load_st_direct(in_b, &mut st_b, 0, 0, 0);

        let mut rt_c = Rt::<F, D4, D4>::new();
        rt_c.zero();

        mma_AB_rt(&mut rt_c, &st_a, &st_b, offset_m as usize, offset_n as usize);

        rt_c.copy_to_array(output);
    }

    /// Test kernel for mma_AtB_rt with offset_m=0, offset_n=0 (full tile)
    #[cube(launch)]
    fn test_mma_AtB_rt_kernel<F: Float, TileK: Dim, TileM: Dim, TileN: Dim>(
        in_a: &Tensor<Line<F>>,
        in_b: &Tensor<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        // A is [M, K] in global, transposed to [K, M] in shared
        let mut st_a = St::<F, TileK, TileM>::new();
        let mut st_b = St::<F, TileK, TileN>::new();

        load_st_transpose(in_a, &mut st_a, 0, 0, 0);
        load_st_direct(in_b, &mut st_b, 0, 0, 0);

        let mut rt_c = Rt::<F, TileM, TileN>::new();
        rt_c.zero();

        mma_AtB_rt(&mut rt_c, &st_a, &st_b, 0, 0);

        rt_c.copy_to_array(output);
    }

    /// Test kernel for mma_AtB_rt with non-zero offsets (8x8 tiles, 4x4 sub-tile)
    #[cube(launch)]
    fn test_mma_AtB_rt_offset_kernel<F: Float>(
        in_a: &Tensor<Line<F>>,
        in_b: &Tensor<Line<F>>,
        output: &mut Array<Line<F>>,
        offset_m: u32,
        offset_n: u32,
    ) {
        // A is [M, K] = [8, 8] in global, transposed to [K, M] = [8, 8] in shared
        let mut st_a = St::<F, D8, D8>::new();
        let mut st_b = St::<F, D8, D8>::new();

        load_st_transpose(in_a, &mut st_a, 0, 0, 0);
        load_st_direct(in_b, &mut st_b, 0, 0, 0);

        let mut rt_c = Rt::<F, D4, D4>::new();
        rt_c.zero();

        mma_AtB_rt(&mut rt_c, &st_a, &st_b, offset_m as usize, offset_n as usize);

        rt_c.copy_to_array(output);
    }

    test_kernel! {
        #[test]
        fn test_mma_AB_rt_full() for
            F in [f32, f64]
            TileM in [D4, D8]
            TileK in [D4, D8]
            TileN in [D4, D8]
        {
            let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
            let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
            let output: Array = [TileM::VALUE * TileN::VALUE];

            assert_eq!(
                test_mma_AB_rt_kernel(in_a(), in_b(), output()) for (1, 1, 1) @ (1),
                {
                    for m in 0..TileM::VALUE {
                        for n in 0..TileN::VALUE {
                            let mut sum = F::from_int(0);
                            for k in 0..TileK::VALUE {
                                let a_val = in_a[m * TileK::VALUE + k];
                                let b_val = in_b[k * TileN::VALUE + n];
                                sum += a_val * b_val;
                            }
                            output[m * TileN::VALUE + n] = sum;
                        }
                    }
                }
            );
        }

        #[test]
        fn test_mma_AB_rt_with_offset() for F in [f32, f64] {
            let in_a: Tensor = [8, 8] as Range;
            let in_b: Tensor = [8, 8] as Range;
            let output: Array = [16];

            assert_eq!(
                test_mma_AB_rt_offset_kernel(in_a(), in_b(), output(), scalar(1u32), scalar(1u32)) for (1, 1, 1) @ (1),
                {
                    for mi in 0..4usize {
                        for ni in 0..4usize {
                            let m = 4 + mi;
                            let n = 4 + ni;
                            let mut sum = F::from_int(0);
                            for k in 0..8usize {
                                let a_val = in_a[m * 8 + k];
                                let b_val = in_b[k * 8 + n];
                                sum += a_val * b_val;
                            }
                            output[mi * 4 + ni] = sum;
                        }
                    }
                }
            );
        }

        #[test]
        fn test_mma_AtB_rt_full() for
            F in [f32, f64]
            TileK in [D4, D8]
            TileM in [D4, D8]
            TileN in [D4, D8]
        {
            let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
            let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
            let output: Array = [TileM::VALUE * TileN::VALUE];

            assert_eq!(
                test_mma_AtB_rt_kernel(in_a(), in_b(), output()) for (1, 1, 1) @ (1),
                {
                    for m in 0..TileM::VALUE {
                        for n in 0..TileN::VALUE {
                            let mut sum = F::from_int(0);
                            for k in 0..TileK::VALUE {
                                let a_val = in_a[m * TileK::VALUE + k];
                                let b_val = in_b[k * TileN::VALUE + n];
                                sum += a_val * b_val;
                            }
                            output[m * TileN::VALUE + n] = sum;
                        }
                    }
                }
            );
        }

        #[test]
        fn test_mma_AtB_rt_with_offset() for F in [f32, f64] {
            let in_a: Tensor = [8, 8] as Range;
            let in_b: Tensor = [8, 8] as Range;
            let output: Array = [16];

            assert_eq!(
                test_mma_AtB_rt_offset_kernel(in_a(), in_b(), output(), scalar(1u32), scalar(1u32)) for (1, 1, 1) @ (1),
                {
                    for mi in 0..4usize {
                        for ni in 0..4usize {
                            let m = 4 + mi;
                            let n = 4 + ni;
                            let mut sum = F::from_int(0);
                            for k in 0..8usize {
                                let a_val = in_a[m * 8 + k];
                                let b_val = in_b[k * 8 + n];
                                sum += a_val * b_val;
                            }
                            output[mi * 4 + ni] = sum;
                        }
                    }
                }
            );
        }
    }
}
