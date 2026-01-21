use std::marker::PhantomData;

use crate::LINE_SIZE;

/// Marker trait for compile-time dimensions.
/// CubeCL doesn't like const generics, so we improvise.
pub trait Dim: Send + Sync + 'static {
    const VALUE: usize;
    const LINES: usize = Self::VALUE / crate::LINE_SIZE;
}

/// Dimension equal to LINE_SIZE. Use when a dimension intentionally matches the line size.
pub type DLine = D4;
const _: () = assert!(DLine::VALUE == LINE_SIZE, "DLine must equal LINE_SIZE");

/// Compile-time dimension of 1 (for vectors, do not use for tiles).
pub struct D1;
impl Dim for D1 {
    const VALUE: usize = 1;
}

/// Compile-time dimension of 4.
pub struct D4;
impl Dim for D4 {
    const VALUE: usize = 4;
}

/// Compile-time dimension of 8.
pub struct D8;
impl Dim for D8 {
    const VALUE: usize = 8;
}

/// Compile-time dimension of 16.
pub struct D16;
impl Dim for D16 {
    const VALUE: usize = 16;
}

/// Compile-time dimension of 32.
pub struct D32;
impl Dim for D32 {
    const VALUE: usize = 32;
}

/// Compile-time dimension of 64.
pub struct D64;
impl Dim for D64 {
    const VALUE: usize = 64;
}

/// Compile-time dimension of 128.
pub struct D128;
impl Dim for D128 {
    const VALUE: usize = 128;
}

/// Compile-time dimension of 256.
pub struct D256;
impl Dim for D256 {
    const VALUE: usize = 256;
}

/// Zero-sized type for carrying dimension info without runtime cost.
pub type DimPhantom<R, C> = PhantomData<(R, C)>;
