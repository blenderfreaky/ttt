use crate::{binary_ops::*, prelude::*, unary_ops::*};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Rt<F: Float> {
    pub data: Array<Line<F>>,
    #[cube(comptime)]
    pub rows: usize,
    #[cube(comptime)]
    pub cols: usize,
}

#[cube]
impl<F: Float> Rt<F> {
    #[must_use]
    pub fn new(#[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        Rt::<F> {
            data: Array::lined(rows * cols, LINE_SIZE),
            rows,
            cols,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        for i in 0..self.data.len() {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &Rt<F>) {
        for i in 0..self.data.len() {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }

    pub fn copy_from_array(&mut self, array: &Array<Line<F>>) {
        for i in 0..self.data.len() {
            self.data[i] = array[i];
        }
    }

    pub fn copy_to_array(&self, array: &mut Array<Line<F>>) {
        for i in 0..self.data.len() {
            array[i] = self.data[i];
        }
    }
}

impl<F: Float> Rt<F> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols / LINE_SIZE
    }
}
