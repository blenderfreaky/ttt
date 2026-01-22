use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::{binary_ops::*, plane::swizzle, prelude::*, unary_ops::*};

use super::dim::Dim;

#[derive(CubeType)]
pub struct St<F: Float, R: Dim, C: Dim> {
    pub data: SharedMemory<Line<F>>,
    #[cube(comptime)]
    _phantom: PhantomData<(R, C)>,
    // This is just for ergonomics,
    // as we can't access Self::LEN due to CubeCL limitations
    #[cube(comptime)]
    len: usize,
}

pub type Sv<F, L> = St<F, L, D1>;

impl<F: Float, R: Dim, C: Dim> St<F, R, C> {
    pub const ROWS: usize = R::VALUE;
    pub const COLS: usize = C::VALUE;
    pub const SIZE: usize = R::VALUE * C::VALUE;
    pub const LEN: usize = R::VALUE * C::VALUE / LINE_SIZE;

    pub fn len() -> usize {
        Self::LEN
    }

    pub fn size() -> usize {
        Self::SIZE
    }
}

#[cube]
impl<F: Float, R: Dim, C: Dim> St<F, R, C> {
    pub fn new() -> St<F, R, C> {
        St::<F, R, C> {
            data: SharedMemory::new_lined(comptime!(Self::LEN), LINE_SIZE),
            _phantom: PhantomData,
            len: Self::LEN,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &St<F, R, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }

    pub fn apply_row_broadcast<O: BinaryOp<F>>(&mut self, op: O, row: &Rv<F, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            self.data[s_idx] = op.apply(self.data[s_idx], row.data[c_line]);
        }
    }

    pub fn apply_col_broadcast<O: BinaryOp<F>>(&mut self, op: O, col: &Rv<F, R>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            // Get the scalar for this row from the column vector
            let r_line = r / LINE_SIZE;
            let r_idx = r % LINE_SIZE;
            let col_val = col.data[r_line][r_idx];
            let broadcast = Line::<F>::empty(LINE_SIZE).fill(col_val);

            self.data[s_idx] = op.apply(self.data[s_idx], broadcast);
        }
    }

    /// Copy contents from another St (cooperative, all threads participate)
    pub fn copy_from(&mut self, other: &St<F, R, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = other.data[i];
        }
    }
}

impl<F: Float, R: Dim, C: Dim> Default for St<F, R, C> {
    fn default() -> Self {
        Self::new()
    }
}
