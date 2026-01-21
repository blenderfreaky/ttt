// For `addr + 0`, it makes some stuff cleaner to read
#![allow(clippy::identity_op)]
// Empty tiles aren't a thing, so this method would be confusing
#![allow(clippy::len_without_is_empty)]

pub mod binary_ops;
pub mod plane;
pub mod reduction_ops;
pub mod tiles;
pub mod unary_ops;
pub mod util;

#[cfg(test)]
#[macro_use]
mod test_utils;

// We could parametrize this
// but right now it's not worth the effort
pub const LINE_SIZE: usize = 4;

pub mod prelude {
    pub use crate::LINE_SIZE;
    pub use crate::plane;
    #[cfg(test)]
    pub use crate::test_kernel;
    pub use crate::tiles::*;
}
