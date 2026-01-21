use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::{binary_ops::*, prelude::*, unary_ops::*};

use super::dim::Dim;

#[derive(CubeType)]
pub struct Rt<F: Float, R: Dim, C: Dim> {
    pub data: Array<Line<F>>,
    #[cube(comptime)]
    _phantom: PhantomData<(R, C)>,
    // This is just for ergonomics,
    // as we can't access Self::LEN due to CubeCL limitations
    #[cube(comptime)]
    len: usize,
}

impl<F: Float, R: Dim, C: Dim> Rt<F, R, C> {
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
impl<F: Float, R: Dim, C: Dim> Rt<F, R, C> {
    pub fn new() -> Rt<F, R, C> {
        Rt::<F, R, C> {
            data: Array::lined(comptime!(Rt::<F, R, C>::size()), LINE_SIZE),
            _phantom: PhantomData,
            len: Self::LEN,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        #[unroll]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &Rt<F, R, C>) {
        #[unroll]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }

    pub fn copy_from_array(&mut self, array: &Array<Line<F>>) {
        #[unroll]
        for i in 0..self.len {
            self.data[i] = array[i];
        }
    }

    pub fn copy_to_array(&self, array: &mut Array<Line<F>>) {
        #[unroll]
        for i in 0..self.len {
            array[i] = self.data[i];
        }
    }
}

impl<F: Float, R: Dim, C: Dim> Default for Rt<F, R, C> {
    fn default() -> Self {
        Self::new()
    }
}

pub type RowVec<F, C> = Rt<F, super::dim::D1, C>;
pub type ColVec<F, R> = Rt<F, R, super::dim::D1>;
