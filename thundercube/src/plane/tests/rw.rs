use crate::{
    plane::{
        load_rt_from_st, load_st_direct, load_st_transpose, store_rt_to_st, store_st_direct,
        store_st_transpose,
    },
    prelude::*,
    test_kernel,
};
use cubecl::prelude::*;
use test_case::test_matrix;

#[cube(launch)]
fn rw_direct<F: Float, TileM: Dim, TileN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();

    let r_off = CUBE_POS_X as usize * TileM::VALUE;
    let c_off = CUBE_POS_Y as usize * TileN::VALUE;
    load_st_direct(input, &mut st, 0, r_off, c_off);
    store_st_direct(&st, output, 0, r_off, c_off);
}

#[cube(launch)]
fn rw_transpose<F: Float, TileM: Dim, TileN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();

    let r_off = CUBE_POS_X as usize * TileM::VALUE;
    let c_off = CUBE_POS_Y as usize * TileN::VALUE;
    load_st_transpose(input, &mut st, 0, r_off, c_off);
    store_st_transpose(&st, output, 0, r_off, c_off);
}

/// Tests round-trip: global -> St -> Rt -> St -> global
#[cube(launch)]
fn rw_rt_st<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    let r_off = CUBE_POS_X as usize * TileM::VALUE;
    let c_off = CUBE_POS_Y as usize * TileN::VALUE;

    // Global -> St
    load_st_direct(input, &mut st, 0, r_off, c_off);

    // St -> Rt
    load_rt_from_st::<F, RtM, RtN, TileM, TileN>(&st, &mut rt);

    // Rt -> St
    store_rt_to_st::<F, RtM, RtN, TileM, TileN>(&rt, &mut st);

    // St -> Global
    store_st_direct(&st, output, 0, r_off, c_off);
}

test_kernel! {
    #[test_matrix([4, 32], [4, 32])]
    fn test_rw_direct(rows: usize, cols: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_direct(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32])]
    fn test_rw_transpose(rows: usize, cols: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_transpose(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ max(1),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 16], [4, 16])]
    fn test_rw_rt_st(rows: usize, cols: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
        RtM in [D4]
        RtN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_rt_st(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }
}
