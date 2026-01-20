use crate::prelude::*;
use cubecl::prelude::*;

/// Swizzle function for bank-conflict-free shared memory access.
/// Must match the swizzle used in load/store functions.
#[cube]
fn swizzle(row: usize, vec_col: usize, mask: usize) -> usize {
    vec_col ^ (row & mask)
}

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
/// 'offset_m/n' are the base offsets (in Lines) for the sub-tile this thread processes.
#[cube]
pub fn mma<F: Float>(c: &mut Rt<F>, a: &St<F>, b: &St<F>, offset_m: usize, offset_n: usize) {
    // A is [K, M]: a.rows = K, a.cols = M
    // B is [K, N]: b.rows = K, b.cols = N
    let k_dim = comptime!(a.rows);

    let num_m_vecs = comptime!(c.rows / LINE_SIZE);
    let num_n_vecs = comptime!(c.cols / LINE_SIZE);

    // Row-major strides (in Lines)
    let a_stride = comptime!(a.cols / LINE_SIZE); // M / LINE_SIZE
    let b_stride = comptime!(b.cols / LINE_SIZE); // N / LINE_SIZE

    // Swizzle masks
    let a_mask = comptime!(a_stride - 1);
    let b_mask = comptime!(b_stride - 1);

    for k in 0..k_dim {
        #[unroll]
        for nl in 0..num_n_vecs {
            // B-Vector: row k, column (offset_n + nl) in Lines
            let b_col = offset_n + nl;
            let b_phys_col = swizzle(k, b_col, b_mask);
            let b_idx = k * b_stride + b_phys_col;
            let b_vec = b.data[b_idx];

            #[unroll]
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
