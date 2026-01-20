use crate::prelude::*;
use cubecl::prelude::*;

/// C += A * B
/// Supports arbitrary C-tile sizes (e.g., 8x8, 16x8).
///
/// LAYOUT ASSUMPTIONS:
/// - A: Col-Major [M, K]. (A is transposed in Shared Mem).
/// - B: Row-Major [K, N].
/// - C: Row-Major Register Tile.
///
/// 'offset_m/n' are the Base offsets (in Lines) for the top-left of the tile.
#[cube]
pub fn mma<F: Float>(c: &mut Rt<F>, a: &St<F>, b: &St<F>, offset_m: usize, offset_n: usize) {
    let k_dim = comptime!(a.cols);

    let num_m_vecs = comptime!(c.rows / LINE_SIZE);
    let num_n_vecs = comptime!(c.cols / LINE_SIZE);

    let a_stride_k = comptime!(a.rows / LINE_SIZE);
    let b_stride_k = comptime!(b.cols / LINE_SIZE);

    for k in 0..k_dim {
        #[unroll]
        for nl in 0..num_n_vecs {
            // B-Vector (Horizontal)
            let b_idx = (offset_n + nl) + (k * b_stride_k);
            let b_vec = b.data[b_idx];

            #[unroll]
            for ml in 0..num_m_vecs {
                // A-Vector (Vertical)
                let a_idx = (offset_m + ml) + (k * a_stride_k);
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
