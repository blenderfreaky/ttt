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

test_kernel! {
    #[test_matrix([4, 32], [4, 32], [4], [4])]
    fn test_rw_direct(rows: usize, cols: usize, tile_rows: usize, tile_cols: usize) for F in all {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_direct(input(), output(), lit(rows), lit(cols)) for (rows/tile_rows, cols/tile_cols, 1) @ max(1),
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
            rw_transpose(input(), output(), lit(rows), lit(cols)) for (rows/tile_rows, cols/tile_cols, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }
}
