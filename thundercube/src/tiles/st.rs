use cubecl::prelude::*;

use crate::{binary_ops::*, prelude::*, unary_ops::*};

#[derive(CubeType)]
pub struct St<F: Float> {
    pub data: SharedMemory<Line<F>>,
    #[cube(comptime)]
    pub rows: usize,
    #[cube(comptime)]
    pub cols: usize,
}

#[cube]
impl<F: Float> St<F> {
    #[must_use]
    pub fn new(#[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        St::<F> {
            data: SharedMemory::new_lined(rows * cols, LINE_SIZE),
            rows,
            cols,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        for i in 0..self.data.len() {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &St<F>) {
        for i in 0..self.data.len() {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }
}

impl<F: Float> St<F> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols / LINE_SIZE
    }
}
