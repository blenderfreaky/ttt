use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::{binary_ops::*, prelude::*, unary_ops::*};

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
        #[unroll]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &St<F, R, C>) {
        #[unroll]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }
}

impl<F: Float, R: Dim, C: Dim> Default for St<F, R, C> {
    fn default() -> Self {
        Self::new()
    }
}
