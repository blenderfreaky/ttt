use crate::{
    plane::swizzle,
    prelude::*,
    util::{index_2d, transpose_4, write_into_line},
};
use cubecl::prelude::*;

#[cube]
pub fn store_direct_swizzled<F: Float>(
    s_mem: &St<F>,
    g_mem: &mut Tensor<Line<F>>,
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

        let mask = vec_stride - 1;
        let phys_c = swizzle(r, c_vec, mask);
        let s_idx = (r * vec_stride) + phys_c;

        let val = s_mem.data[s_idx];

        let g_r = g_offset_row + r;
        let g_c = g_offset_col + (c_vec * LINE_SIZE);

        store_safe(g_mem, g_r, g_c, val);
    }
}

#[cube]
pub fn store_transpose_swizzled<F: Float>(
    s_mem: &St<F>,
    g_mem: &mut Tensor<Line<F>>,
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
        let patch_r = i / patches_w;
        let patch_c = i % patches_w;

        let src_base_r = patch_r * LINE_SIZE;
        let src_vec_c = patch_c;
        let mask = s_stride - 1;

        let r0 = src_base_r + 0;
        let idx0 = r0 * s_stride + swizzle(r0, src_vec_c, mask);
        let v0 = s_mem.data[idx0];

        let r1 = src_base_r + 1;
        let idx1 = r1 * s_stride + swizzle(r1, src_vec_c, mask);
        let v1 = s_mem.data[idx1];

        let r2 = src_base_r + 2;
        let idx2 = r2 * s_stride + swizzle(r2, src_vec_c, mask);
        let v2 = s_mem.data[idx2];

        let r3 = src_base_r + 3;
        let idx3 = r3 * s_stride + swizzle(r3, src_vec_c, mask);
        let v3 = s_mem.data[idx3];

        let (t0, t1, t2, t3) = transpose_4(v0, v1, v2, v3);

        // We swap the patch coordinates to flip the block location
        let g_patch_r = patch_c;
        let g_patch_c = patch_r;

        let dst_r = g_offset_row + (g_patch_r * LINE_SIZE);
        let dst_c = g_offset_col + (g_patch_c * LINE_SIZE);

        store_safe(g_mem, dst_r + 0, dst_c, t0);
        store_safe(g_mem, dst_r + 1, dst_c, t1);
        store_safe(g_mem, dst_r + 2, dst_c, t2);
        store_safe(g_mem, dst_r + 3, dst_c, t3);
    }
}

#[cube]
pub fn store_safe<F: Float>(g: &mut Tensor<Line<F>>, r: usize, c: usize, val: Line<F>) {
    if r < g.shape(0) && c < g.shape(1) {
        let str = g.stride(1);
        let idx = index_2d(g, r, c);

        if str == 1 {
            g[idx] = val;
        } else {
            write_into_line(g.slice_mut(idx, idx), 0, val[0]);
            write_into_line(g.slice_mut(idx + str, idx + str), 0, val[1]);
            write_into_line(g.slice_mut(idx + str * 2, idx + str * 2), 0, val[2]);
            write_into_line(g.slice_mut(idx + str * 3, idx + str * 3), 0, val[3]);
        }
    }
}
