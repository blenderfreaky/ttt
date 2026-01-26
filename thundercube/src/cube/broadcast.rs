use crate::{prelude::*, tiles::Dim};
use cubecl::prelude::*;

/// Loads from shared memory vector (Sv) into register vector (Rv).
///
/// **Broadcast operation**: All threads get the same data.
/// Each thread copies all L/LINE_SIZE Lines.
///
/// This is NOT a cooperative operation - use when all threads
/// need identical copies of a vector.
#[cube]
pub fn load_rv_from_sv<F: Float, L: Dim>(s_mem: &Sv<F, L>, r_mem: &mut Rv<F, L>) {
    #[unroll(L::LINES <= UNROLL_LIMIT)]
    for i in 0..L::LINES {
        r_mem.data[i] = s_mem.data[i];
    }
}

/// Loads a 1D vector directly from global memory into a register vector (Rv).
///
/// **Broadcast operation**: All threads load the same data.
/// Use for small vectors where all threads need the same values.
///
/// This is NOT a cooperative operation by default.
/// To make it cooperative, pass different base offsets for each thread.
///
/// # Arguments
/// * `g_mem` - Source 1D tensor of Lines
/// * `r_mem` - Destination register vector
/// * `base_offset` - Scalar offset into the source tensor
#[cube]
pub fn load_rv_direct<F: Float, L: Dim>(
    g_mem: &Tensor<Line<F>>,
    r_mem: &mut Rv<F, L>,
    base_offset: usize,
) {
    #[unroll(L::LINES <= UNROLL_LIMIT)]
    for i in 0..L::LINES {
        let line_idx = base_offset / LINE_SIZE + i;
        r_mem.data[i] = g_mem[line_idx];
    }
}

/// Stores a register vector (Rv) to global memory.
///
/// **Single-thread operation**: Only thread 0 writes (all threads have same data).
/// Use for small vectors where all threads computed the same result.
///
/// This is NOT a cooperative operation by default.
/// To make it cooperative, pass different base offsets for each thread.
///
/// # Arguments
/// * `r_mem` - Source register vector
/// * `g_mem` - Destination 1D tensor of Lines
/// * `base_offset` - Scalar offset into the destination tensor
#[cube]
pub fn store_rv_direct<F: Float, L: Dim>(
    r_mem: &Rv<F, L>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
) {
    // Only one thread needs to write (all have same data)
    if UNIT_POS == 0 {
        #[unroll(L::LINES <= UNROLL_LIMIT)]
        for i in 0..L::LINES {
            let line_idx = base_offset / LINE_SIZE + i;
            g_mem[line_idx] = r_mem.data[i];
        }
    }
}
