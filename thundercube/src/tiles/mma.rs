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

#[cube]
pub fn mma_AB_rt<F: Float, CM: Dim, CN: Dim, K: Dim, AM: Dim, BN: Dim>(
    c: &mut Rt<F, CM, CN>,
    a: &St<F, AM, K>,
    b: &St<F, K, BN>,
    offset_m: usize,
    offset_n: usize,
) {
    todo!()
}
