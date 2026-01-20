use crate::{
    plane::swizzle,
    prelude::*,
    util::{transpose_4, write_into_line},
};
use cubecl::prelude::*;

/// Cooperatively stores a shared memory tile to global memory without transposing.
///
/// All threads in the workgroup participate, each storing multiple `Line<F>` vectors
/// in a strided pattern to maximize memory bandwidth. Reads from shared memory use the
/// swizzled layout to avoid bank conflicts.
///
/// # Arguments
/// * `s_mem` - Source shared memory tile (`St`)
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this tile
/// * `g_offset_col` - Column offset within the matrix for this tile
#[cube]
pub fn store_st_direct<F: Float>(
    s_mem: &St<F>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
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

        store_safe(g_mem, base_offset, g_r, g_c, val);
    }
}

/// Cooperatively stores a shared memory tile to global memory while transposing.
///
/// The transpose is performed in `LINE_SIZE × LINE_SIZE` patches (4×4 by default).
/// Each thread processes entire patches: loading 4 rows from shared memory (with swizzle),
/// transposing them via register shuffles, then storing as 4 rows to global memory at
/// swapped coordinates.
///
/// # Arguments
/// * `s_mem` - Source shared memory tile (`St`)
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for the destination tile
/// * `g_offset_col` - Column offset within the matrix for the destination tile
#[cube]
pub fn store_st_transpose<F: Float>(
    s_mem: &St<F>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
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

        store_safe(g_mem, base_offset, dst_r + 0, dst_c, t0);
        store_safe(g_mem, base_offset, dst_r + 1, dst_c, t1);
        store_safe(g_mem, base_offset, dst_r + 2, dst_c, t2);
        store_safe(g_mem, base_offset, dst_r + 3, dst_c, t3);
    }
}

/// Stores a per-thread register tile to global memory without transposing.
///
/// Unlike `store_st_*`, this is not a cooperative operation - each thread stores its own
/// private register tile independently. Out-of-bounds writes are skipped, enabling safe
/// stores at matrix boundaries.
///
/// # Arguments
/// * `rt_mem` - Source register tile (`Rt`), stored row-major as `[rows, cols/LINE_SIZE]` Lines
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this thread's tile
/// * `g_offset_col` - Column offset within the matrix for this thread's tile
#[cube]
pub fn store_rt_direct<F: Float>(
    rt_mem: &Rt<F>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let rt_rows = comptime!(rt_mem.rows);
    let rt_cols = comptime!(rt_mem.cols);
    let num_n_vecs = comptime!(rt_cols / LINE_SIZE);

    let rank = g_mem.rank();
    let num_rows = g_mem.shape(rank - 2);
    let num_cols = g_mem.shape(rank - 1);
    let row_stride = g_mem.stride(rank - 2);

    #[unroll]
    for row in 0..rt_rows {
        let g_r = g_offset_row + row;
        if g_r < num_rows {
            #[unroll]
            for nl in 0..num_n_vecs {
                let g_c = g_offset_col + nl * LINE_SIZE;
                if g_c < num_cols {
                    let rt_idx = row * num_n_vecs + nl;
                    let val = rt_mem.data[rt_idx];

                    let scalar_idx = base_offset + g_r * row_stride + g_c;
                    let line_idx = scalar_idx / LINE_SIZE;
                    g_mem[line_idx] = val;
                }
            }
        }
    }
}

/// Stores a single `Line<F>` (vector of `LINE_SIZE` elements) to a global tensor.
///
/// Handles both row-major and column-major layouts:
/// - **Row-major** (`col_stride == 1`): Direct contiguous store of `LINE_SIZE` consecutive columns
/// - **Column-major**: Scatters `LINE_SIZE` scalars to non-contiguous locations
///
/// Skips writes for out-of-bounds coordinates, enabling safe boundary handling.
///
/// # Arguments
/// * `g` - Destination tensor (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Pre-computed scalar offset from batch dimensions
/// * `r` - Row index within the matrix
/// * `c` - Starting column index (must be aligned to `LINE_SIZE` for row-major)
/// * `val` - The `Line<F>` to store
#[cube]
pub fn store_safe<F: Float>(
    g: &mut Tensor<Line<F>>,
    base_offset: usize,
    r: usize,
    c: usize,
    val: Line<F>,
) {
    let rank = g.rank();
    let row_dim = rank - 2;
    let col_dim = rank - 1;

    let num_rows = g.shape(row_dim);
    let num_cols = g.shape(col_dim);
    let row_stride = g.stride(row_dim);
    let col_stride = g.stride(col_dim);

    if r < num_rows && c < num_cols {
        let scalar_idx = base_offset + r * row_stride + c * col_stride;

        if col_stride == 1 {
            // Row-major: scalar index maps to Line index by dividing by LINE_SIZE
            let line_idx = scalar_idx / LINE_SIZE;
            g[line_idx] = val;
        } else {
            // Column-major: scatter 4 scalars to different Lines
            let line_idx0 = scalar_idx / LINE_SIZE;
            let line_idx1 = (scalar_idx + col_stride) / LINE_SIZE;
            let line_idx2 = (scalar_idx + col_stride * 2) / LINE_SIZE;
            let line_idx3 = (scalar_idx + col_stride * 3) / LINE_SIZE;

            let elem0 = scalar_idx % LINE_SIZE;
            let elem1 = (scalar_idx + col_stride) % LINE_SIZE;
            let elem2 = (scalar_idx + col_stride * 2) % LINE_SIZE;
            let elem3 = (scalar_idx + col_stride * 3) % LINE_SIZE;

            write_into_line(g.slice_mut(line_idx0, line_idx0 + 1), elem0, val[0]);
            write_into_line(g.slice_mut(line_idx1, line_idx1 + 1), elem1, val[1]);
            write_into_line(g.slice_mut(line_idx2, line_idx2 + 1), elem2, val[2]);
            write_into_line(g.slice_mut(line_idx3, line_idx3 + 1), elem3, val[3]);
        }
    }
}
