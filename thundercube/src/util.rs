use cubecl::{prelude::*, std::ReinterpretSliceMut};

use crate::LINE_SIZE;

// This is ugly, but it works
// CubeCL doesn't like [Line<F>; LINE_SIZE]
// so we have to do this monstrosity
#[cube]
pub fn transpose_4<F: Float>(
    r0: Line<F>,
    r1: Line<F>,
    r2: Line<F>,
    r3: Line<F>,
) -> (Line<F>, Line<F>, Line<F>, Line<F>) {
    let mut c0 = Line::empty(4usize);
    let mut c1 = Line::empty(4usize);
    let mut c2 = Line::empty(4usize);
    let mut c3 = Line::empty(4usize);

    c0[0] = r0[0];
    c0[1] = r1[0];
    c0[2] = r2[0];
    c0[3] = r3[0];

    c1[0] = r0[1];
    c1[1] = r1[1];
    c1[2] = r2[1];
    c1[3] = r3[1];

    c2[0] = r0[2];
    c2[1] = r1[2];
    c2[2] = r2[2];
    c2[3] = r3[2];

    c3[0] = r0[3];
    c3[1] = r1[3];
    c3[2] = r2[3];
    c3[3] = r3[3];

    (c0, c1, c2, c3)
}

#[cube]
pub fn write_into_line<F: Float>(one_slice: SliceMut<Line<F>>, idx: usize, val: F) {
    ReinterpretSliceMut::<F, F>::new(one_slice, LINE_SIZE).write(idx, val);
    // let mut l = one_slice[0];
    // l[idx] = val;
    // one_slice[0] = l;
}

#[cube]
pub fn index_1d<T: CubePrimitive>(t: &Tensor<T>, index: usize) -> usize {
    t.stride(0) * index
}

#[cube]
pub fn slice_1d<T: CubePrimitive>(t: &Tensor<T>, index: usize) -> Slice<T> {
    let start = index_1d(t, index);
    let end = index_1d(t, index + 1) - 1;
    t.slice(start, end)
}

#[cube]
pub fn index_2d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize) -> usize {
    t.stride(0) * x + t.stride(1) * y
}

#[cube]
pub fn slice_2d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize) -> Slice<T> {
    let start = index_2d(t, x, y);
    let end = index_2d(t, x + 1, y + 1) - 1;
    t.slice(start, end)
}

#[cube]
pub fn index_3d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize, z: usize) -> usize {
    t.stride(0) * x + t.stride(1) * y + t.stride(2) * z
}

#[cube]
pub fn slice_3d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize, z: usize) -> Slice<T> {
    let start = index_3d(t, x, y, z);
    let end = index_3d(t, x + 1, y + 1, z + 1) - 1;
    t.slice(start, end)
}
