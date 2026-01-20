use cubecl::prelude::*;

use crate::LINE_SIZE;

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
    pub fn new(#[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        St::<F> {
            data: SharedMemory::new_lined(rows * cols, LINE_SIZE),
            rows,
            cols,
        }
    }
}
