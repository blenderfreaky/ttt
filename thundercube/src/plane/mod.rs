mod load;
mod mma;
mod store;

#[cfg(test)]
mod tests;

use cubecl::cube;

#[cube]
fn swizzle(row: usize, vec_col: usize, mask: usize) -> usize {
    vec_col ^ (row & mask)
}

pub use load::*;
pub use mma::*;
pub use store::*;
