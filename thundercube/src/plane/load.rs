use crate::{
    plane::swizzle,
    prelude::*,
    util::{index_2d, transpose_4},
};
use cubecl::prelude::*;

#[cube]
pub fn load_direct_swizzled<F: Float>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F>,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let s_rows = comptime!(s_mem.rows);
    let s_cols = comptime!(s_mem.cols);
    let vec_stride = comptime!(s_cols / LINE_SIZE);
    let total_vecs = comptime!(s_rows * vec_stride);

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, total_vecs, num_threads) {
        let r = i / vec_stride;
        let c_vec = i % vec_stride;

        // Global Coords
        let g_r = g_offset_row + r;
        let g_c = g_offset_col + (c_vec * LINE_SIZE);

        let val = load_safe(g_mem, g_r, g_c);

        let mask = vec_stride - 1;
        let phys_c = swizzle(r, c_vec, mask);
        let s_idx = (r * vec_stride) + phys_c;

        s_mem.data[s_idx] = val;
    }
}

#[cube]
pub fn load_transpose_swizzled<F: Float>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F>,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let s_rows = comptime!(s_mem.rows);
    let s_cols = comptime!(s_mem.cols);
    let s_stride = comptime!(s_cols / LINE_SIZE);

    let patches_h = comptime!(s_rows / LINE_SIZE);
    let patches_w = comptime!(s_cols / LINE_SIZE);
    let total_patches = comptime!(patches_h * patches_w);

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, total_patches, num_threads) {
        let patch_r = i / patches_w; // Shared Row Index (in patches)
        let patch_c = i % patches_w; // Shared Col Index (in patches)

        let g_patch_r = patch_c;
        let g_patch_c = patch_r;

        let src_r = g_offset_row + (g_patch_r * LINE_SIZE);
        let src_c = g_offset_col + (g_patch_c * LINE_SIZE);

        let v0 = load_safe(g_mem, src_r + 0, src_c);
        let v1 = load_safe(g_mem, src_r + 1, src_c);
        let v2 = load_safe(g_mem, src_r + 2, src_c);
        let v3 = load_safe(g_mem, src_r + 3, src_c);

        let (t0, t1, t2, t3) = transpose_4(v0, v1, v2, v3);

        let dst_base_r = patch_r * LINE_SIZE;
        let dst_vec_c = patch_c;
        let mask = s_stride - 1;

        // We can't look here because t0, t1, etc aren't
        // an array and CubeCL doesn't like arrays.

        let r0 = dst_base_r + 0;
        let phys_c0 = swizzle(r0, dst_vec_c, mask);
        s_mem.data[r0 * s_stride + phys_c0] = t0;

        let r1 = dst_base_r + 1;
        let phys_c1 = swizzle(r1, dst_vec_c, mask);
        s_mem.data[r1 * s_stride + phys_c1] = t1;

        let r2 = dst_base_r + 2;
        let phys_c2 = swizzle(r2, dst_vec_c, mask);
        s_mem.data[r2 * s_stride + phys_c2] = t2;

        let r3 = dst_base_r + 3;
        let phys_c3 = swizzle(r3, dst_vec_c, mask);
        s_mem.data[r3 * s_stride + phys_c3] = t3;
    }
}

/// Helper to load a Line<F> from a scalar Tensor<F>.
/// Uses index_2d to handle strides properly.
#[cube]
fn load_safe<F: Float>(g: &Tensor<Line<F>>, r: usize, c: usize) -> Line<F> {
    if r < g.shape(0) && c < g.shape(1) {
        let stride_col = g.stride(1);
        let base_idx = index_2d(g, r, c);

        if stride_col == 1 {
            g[base_idx]
        } else {
            let v0 = g[base_idx][0];
            let v1 = g[base_idx + stride_col][0];
            let v2 = g[base_idx + stride_col * 2][0];
            let v3 = g[base_idx + stride_col * 3][0];

            let mut l = Line::empty(LINE_SIZE);
            l[0] = v0;
            l[1] = v1;
            l[2] = v2;
            l[3] = v3;
            l
        }
    } else {
        Line::empty(LINE_SIZE).fill(F::from_int(0))
    }
}
