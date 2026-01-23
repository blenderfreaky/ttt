use crate::{
    plane::{
        load_rt_from_st, load_st_direct, load_st_transpose, store_rt_to_st, store_st_direct,
        store_st_transpose,
    },
    prelude::*,
    test_kernel,
    util::sync_planes,
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

/// Tests round-trip: global -> St -> Rt -> St -> global for larger tiles
/// This matches the kernel's D16×D64 tile with D4×D16 register tiles
#[cube(launch)]
fn rw_rt_st_16x64<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, D16, D64>::new();
    let mut rt = Rt::<F, D4, D16>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_planes();

    // St -> Rt (cooperative)
    load_rt_from_st::<F, D4, D16, D16, D64>(&st, &mut rt);

    sync_planes();

    // Rt -> St (cooperative)
    store_rt_to_st::<F, D4, D16, D16, D64>(&rt, &mut st);

    sync_planes();

    // St -> Global
    store_st_direct(&st, output, 0, 0, 0);
}

test_kernel! {
    #[test]
    fn test_rw_rt_st_16x64_32threads() for F in all {
        let input: Tensor = [16, 64];
        let output: Tensor = [16, 64];

        assert_eq!(
            rw_rt_st_16x64(input(), output()) for (1, 1, 1) @ (32),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32], [1, 4, 32, 64])]
    fn test_rw_direct(rows: usize, cols: usize, threads: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_direct(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ (threads),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32], [1, 4, 32, 64])]
    fn test_rw_transpose(rows: usize, cols: usize, threads: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_transpose(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ (threads),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // Thread count fixed: must equal (TileM/RtM) * (TileN/RtN) for Rt<->St mapping
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
