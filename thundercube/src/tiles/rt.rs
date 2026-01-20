use crate::prelude::*;
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
    pub fn new(#[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        Rt::<F> {
            data: Array::new(rows * cols / LINE_SIZE),
            rows,
            cols,
        }
    }
}
