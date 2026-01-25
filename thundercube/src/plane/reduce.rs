use crate::{plane::swizzle, prelude::*, reduction_ops::*, tiles::Dim};
use cubecl::prelude::*;

/// Cooperatively reduces St across rows, producing one value per column.
/// All threads in the plane participate, each handling strided rows.
/// Result is broadcast to all threads.
///
/// St<F, R, C> -> Rv<F, C>
#[cube]
pub fn reduce_st_cols<F: Float, R: Dim, C: Dim, O: ReductionOp<F>>(
    s_mem: &St<F, R, C>,
    result: &mut Rv<F, C>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(C::LINES <= UNROLL_LIMIT)]
    for c_line in 0..C::LINES {
        // Each thread accumulates strided rows for this column
        let mut acc = O::identity();
        for r in range_stepped(tid, R::VALUE, num_threads) {
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            acc = O::combine(acc, s_mem.data[s_idx]);
        }
        // Combine across threads in plane, broadcast result
        result.data[c_line] = plane_reduce_line::<F, O>(acc);
    }
}

/// Cooperatively reduces St across columns, producing one value per row.
/// All threads in the plane participate, each handling strided columns.
/// Result is broadcast to all threads.
///
/// St<F, R, C> -> Rv<F, R>
#[cube]
pub fn reduce_st_rows<F: Float, R: Dim, C: Dim, O: ReductionOp<F>>(
    s_mem: &St<F, R, C>,
    result: &mut Rv<F, R>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(R::LINES <= UNROLL_LIMIT)]
    for r_line in 0..R::LINES {
        let mut out_line = Line::<F>::empty(LINE_SIZE);

        #[unroll]
        for i in 0..LINE_SIZE {
            let r = r_line * LINE_SIZE + i;

            // Each thread accumulates strided columns for this row
            let mut acc = O::identity();
            for c_line in range_stepped(tid, C::LINES, num_threads) {
                let phys_col = swizzle(r, c_line, mask);
                let s_idx = r * vec_stride + phys_col;
                acc = O::combine(acc, s_mem.data[s_idx]);
            }
            let local_scalar = O::finalize(acc);

            // Combine across threads in plane
            out_line[i] = O::plane_reduce(local_scalar);
        }
        result.data[r_line] = out_line;
    }
}

/// Convenience function: sum St across rows (reduce cols)
#[cube]
pub fn sum_st_cols<F: Float, R: Dim, C: Dim>(s_mem: &St<F, R, C>, result: &mut Rv<F, C>) {
    reduce_st_cols::<F, R, C, SumOp>(s_mem, result)
}

/// Convenience function: sum St across columns (reduce rows)
#[cube]
pub fn sum_st_rows<F: Float, R: Dim, C: Dim>(s_mem: &St<F, R, C>, result: &mut Rv<F, R>) {
    reduce_st_rows::<F, R, C, SumOp>(s_mem, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_utils::TestFloat, util::sync_planes};
    use test_case::test_matrix;

    const ROWS: usize = 8;
    const COLS: usize = 8;

    #[cube(launch)]
    fn test_sum_st_rows_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st = St::<F, D8, D8>::new();

        // Load input into St (with swizzle)
        let tid = UNIT_POS as usize;
        let num_threads = CUBE_DIM as usize;
        let vec_stride = D8::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, D8::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            st.data[s_idx] = input[i];
        }

        sync_planes();

        let mut result = Rv::<F, D8>::new();
        sum_st_rows::<F, D8, D8>(&st, &mut result);

        // Only first thread writes output (all have same result)
        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    #[cube(launch)]
    fn test_sum_st_cols_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st = St::<F, D8, D8>::new();

        // Load input into St (with swizzle)
        let tid = UNIT_POS as usize;
        let num_threads = CUBE_DIM as usize;
        let vec_stride = D8::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, D8::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            st.data[s_idx] = input[i];
        }

        sync_planes();

        let mut result = Rv::<F, D8>::new();
        sum_st_cols::<F, D8, D8>(&st, &mut result);

        // Only first thread writes output (all have same result)
        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    test_kernel! {
        // Note: plane reduce requires threads >= PLANE_DIM (32), so we only test 32 and 64
        #[test_matrix([32, 64])]
        fn test_sum_st_rows(threads: usize) for F in all {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS];

            assert_eq!(
                test_sum_st_rows_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        let mut sum = 0.0;
                        for c in 0..COLS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[r] = F::from_f64(sum);
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_sum_st_cols(threads: usize) for F in all {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [COLS];

            assert_eq!(
                test_sum_st_cols_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for c in 0..COLS {
                        let mut sum = 0.0;
                        for r in 0..ROWS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[c] = F::from_f64(sum);
                    }
                }
            );
        }
    }
}
