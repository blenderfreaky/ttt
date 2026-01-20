use crate::{
    plane::{
        load_direct_swizzled, load_transpose_swizzled, store_direct_swizzled,
        store_transpose_swizzled,
    },
    prelude::*,
    test_kernel,
};
use cubecl::prelude::*;
use test_case::test_matrix;

#[cube(launch)]
fn rw_direct<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] tile_rows: usize,
    #[comptime] tile_cols: usize,
) {
    let mut st = St::new(tile_rows, tile_cols);

    let r_off = CUBE_POS_X as usize * tile_rows;
    let c_off = CUBE_POS_Y as usize * tile_cols;
    load_direct_swizzled(input, &mut st, r_off, c_off);
    store_direct_swizzled(&st, output, r_off, c_off);
}

#[cube(launch)]
fn rw_transpose<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] tile_rows: usize,
    #[comptime] tile_cols: usize,
) {
    let mut st = St::new(tile_rows, tile_cols);

    let r_off = CUBE_POS_X as usize * tile_rows;
    let c_off = CUBE_POS_Y as usize * tile_cols;
    load_transpose_swizzled(input, &mut st, r_off, c_off);
    store_transpose_swizzled(&st, output, r_off, c_off);
}

// Simple identity copy kernel (no swizzling) for debugging
// Tests if basic Line indexing works
#[cube(launch)]
fn rw_simple<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] tile_rows: usize,
    #[comptime] tile_cols: usize,
) {
    let s_rows = comptime!(tile_rows);
    let s_cols = comptime!(tile_cols);
    let vec_stride = comptime!(s_cols / LINE_SIZE);
    let total_vecs = comptime!(s_rows * vec_stride);

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    // For now, test with single cube (no offset)
    for i in range_stepped(tid, total_vecs, num_threads) {
        let r = i / vec_stride;
        let c_vec = i % vec_stride;

        // Use Line-based indexing: each row has vec_stride Lines
        // Index = r * vec_stride + c_vec
        let line_idx = r * vec_stride + c_vec;

        let val = input[line_idx];
        output[line_idx] = val;
    }
}

test_kernel! {
    // Minimal test with Range values and simple kernel
    #[test_matrix([4], [4])]
    fn test_rw_simple_4x4(rows: usize, cols: usize) for F in all {
        let input: Tensor = [rows, cols] as Range;
        let output: Tensor = [rows, cols] as Range;

        assert_eq!(
            rw_simple(input(), output(), lit(4usize), lit(4usize)) for (1, 1, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32], [4], [4])]
    fn test_rw_direct(rows: usize, cols: usize, tile_rows: usize, tile_cols: usize) for F in all {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_direct(input(), output(), lit(tile_rows), lit(tile_cols)) for (rows/tile_rows, cols/tile_cols, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32], [4], [4])]
    fn test_rw_transpose(rows: usize, cols: usize, tile_rows: usize, tile_cols: usize) for F in all {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_transpose(input(), output(), lit(tile_rows), lit(tile_cols)) for (rows/tile_rows, cols/tile_cols, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }
}
