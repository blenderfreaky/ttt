use crate::{plane::swizzle, prelude::*, util::transpose_4};
use cubecl::prelude::*;

/// Cooperatively loads a tile from global memory into shared memory without transposing.
///
/// All threads in the workgroup participate, each loading multiple `Line<F>` vectors
/// in a strided pattern to maximize memory bandwidth. The shared memory uses a swizzled
/// layout to avoid bank conflicts during subsequent accesses.
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `s_mem` - Destination shared memory tile (`St`)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this tile
/// * `g_offset_col` - Column offset within the matrix for this tile
#[cube]
pub fn load_st_direct<F: Float>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F>,
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

        // Global Coords
        let g_r = g_offset_row + r;
        let g_c = g_offset_col + (c_vec * LINE_SIZE);

        let val = load_safe(g_mem, base_offset, g_r, g_c);

        let mask = vec_stride - 1;
        let phys_c = swizzle(r, c_vec, mask);
        let s_idx = (r * vec_stride) + phys_c;

        s_mem.data[s_idx] = val;
    }
}

/// Cooperatively loads a tile from global memory into shared memory while transposing.
///
/// The transpose is performed in `LINE_SIZE × LINE_SIZE` patches (4×4 by default).
/// Each thread processes entire patches: loading 4 consecutive rows from global memory,
/// transposing them via register shuffles, then storing as 4 columns to shared memory.
/// The shared memory uses a swizzled layout to avoid bank conflicts.
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `s_mem` - Destination shared memory tile (`St`), will contain transposed data
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for the source tile
/// * `g_offset_col` - Column offset within the matrix for the source tile
#[cube]
pub fn load_st_transpose<F: Float>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F>,
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
        let patch_r = i / patches_w; // Shared Row Index (in patches)
        let patch_c = i % patches_w; // Shared Col Index (in patches)

        let g_patch_r = patch_c;
        let g_patch_c = patch_r;

        let src_r = g_offset_row + (g_patch_r * LINE_SIZE);
        let src_c = g_offset_col + (g_patch_c * LINE_SIZE);

        let v0 = load_safe(g_mem, base_offset, src_r + 0, src_c);
        let v1 = load_safe(g_mem, base_offset, src_r + 1, src_c);
        let v2 = load_safe(g_mem, base_offset, src_r + 2, src_c);
        let v3 = load_safe(g_mem, base_offset, src_r + 3, src_c);

        let (t0, t1, t2, t3) = transpose_4(v0, v1, v2, v3);

        let dst_base_r = patch_r * LINE_SIZE;
        let dst_vec_c = patch_c;
        let mask = s_stride - 1;

        // We can't loop here because t0, t1, etc aren't
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

/// Loads a tile from global memory directly into per-thread registers without transposing.
///
/// Unlike `load_st_*`, this is not a cooperative operation - each thread loads its own
/// private register tile independently. Out-of-bounds accesses are handled by returning
/// zeros, enabling safe loads at matrix boundaries.
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `rt_mem` - Destination register tile (`Rt`), stored row-major as `[rows, cols/LINE_SIZE]` Lines
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this thread's tile
/// * `g_offset_col` - Column offset within the matrix for this thread's tile
#[cube]
pub fn load_rt_direct<F: Float>(
    g_mem: &Tensor<Line<F>>,
    rt_mem: &mut Rt<F>,
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
        #[unroll]
        for nl in 0..num_n_vecs {
            let g_c = g_offset_col + nl * LINE_SIZE;
            let rt_idx = row * num_n_vecs + nl;

            if g_r < num_rows && g_c < num_cols {
                let scalar_idx = base_offset + g_r * row_stride + g_c;
                let line_idx = scalar_idx / LINE_SIZE;
                rt_mem.data[rt_idx] = g_mem[line_idx];
            } else {
                rt_mem.data[rt_idx] = Line::empty(LINE_SIZE).fill(F::from_int(0));
            }
        }
    }
}

/// Loads a single `Line<F>` (vector of `LINE_SIZE` elements) from a global tensor.
///
/// Handles both row-major and column-major layouts:
/// - **Row-major** (`col_stride == 1`): Direct contiguous load of `LINE_SIZE` consecutive columns
/// - **Column-major**: Gathers `LINE_SIZE` scalars from non-contiguous locations
///
/// Returns zeros for out-of-bounds coordinates, enabling safe boundary handling.
///
/// # Arguments
/// * `g` - Source tensor (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Pre-computed scalar offset from batch dimensions
/// * `r` - Row index within the matrix
/// * `c` - Starting column index (must be aligned to `LINE_SIZE` for row-major)
#[cube]
fn load_safe<F: Float>(g: &Tensor<Line<F>>, base_offset: usize, r: usize, c: usize) -> Line<F> {
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
            g[line_idx]
        } else {
            // Column-major: gather 4 scalars from different Lines
            let line_idx0 = scalar_idx / LINE_SIZE;
            let line_idx1 = (scalar_idx + col_stride) / LINE_SIZE;
            let line_idx2 = (scalar_idx + col_stride * 2) / LINE_SIZE;
            let line_idx3 = (scalar_idx + col_stride * 3) / LINE_SIZE;

            let elem0 = scalar_idx % LINE_SIZE;
            let elem1 = (scalar_idx + col_stride) % LINE_SIZE;
            let elem2 = (scalar_idx + col_stride * 2) % LINE_SIZE;
            let elem3 = (scalar_idx + col_stride * 3) % LINE_SIZE;

            let mut l = Line::empty(LINE_SIZE);
            l[0] = g[line_idx0][elem0];
            l[1] = g[line_idx1][elem1];
            l[2] = g[line_idx2][elem2];
            l[3] = g[line_idx3][elem3];
            l
        }
    } else {
        Line::empty(LINE_SIZE).fill(F::from_int(0))
    }
}
