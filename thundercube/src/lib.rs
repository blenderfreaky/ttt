// #![warn(clippy::pedantic)]
#![allow(
    clippy::identity_op,
    reason = "For `addr + 0`, it makes some stuff cleaner to read"
)]
#![allow(
    clippy::len_without_is_empty,
    reason = "Empty tiles aren't a thing, so this method would be confusing"
)]
#![allow(clippy::needless_range_loop)]

pub mod binary_ops;
pub mod plane;
pub mod reduction_ops;
pub mod tiles;
pub mod unary_ops;
pub mod util;

#[cfg(any(test, feature = "test-utils"))]
#[macro_use]
pub mod test_utils;

// We could parametrize this
// but right now it's not worth the effort
pub const LINE_SIZE: usize = 4;

/// Maximum loop iteration count to unconditionally unroll (general/outer loops)
pub const UNROLL_LIMIT: usize = 4;
/// Maximum loop iteration count to unconditionally unroll (hot/inner loops)
pub const UNROLL_LIMIT_HOT: usize = 8;

pub mod prelude {
    pub use crate::LINE_SIZE;
    pub use crate::UNROLL_LIMIT;
    pub use crate::UNROLL_LIMIT_HOT;
    pub use crate::plane;
    #[cfg(test)]
    pub use crate::test_kernel;
    pub use crate::tiles::*;
}
