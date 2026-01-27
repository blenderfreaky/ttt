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

#[cfg(all(
    test,
    not(any(feature = "cuda", feature = "rocm", feature = "wgpu", feature = "cpu"))
))]
compile_error!(
    "At least one backend must be enabled to run tests, please run with `--features cuda/rocm/wgpu/cpu`"
);

pub mod binary_ops;
pub mod cube;
pub mod reduction_ops;
#[cfg(feature = "rocm")]
pub mod streaming;
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
    #[cfg(test)]
    pub use crate::test_kernel;
    #[cfg(feature = "rocm")]
    pub use crate::util::gpu_sleep;
    pub use crate::{LINE_SIZE, UNROLL_LIMIT, UNROLL_LIMIT_HOT, cube, tiles::*, util::sync_planes};
}
