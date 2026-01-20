// For `addr + 0`, it makes some stuff cleaner to read
#![allow(clippy::identity_op)]

pub mod plane;
pub mod tiles;
pub mod util;

#[cfg(test)]
#[macro_use]
mod test_utils;

// We could parametrize this
// but right now it's not worth the effort
pub const LINE_SIZE: usize = 4;

pub mod prelude {
    pub use crate::LINE_SIZE;
    pub use crate::tiles::*;
}
