use std::marker::PhantomData;

use burn::prelude::*;

use crate::ttt::layer::TTTInnerModel;

pub mod backend;
pub mod linear;
mod linear_backward;
mod linear_forward;

/// Marker type for fused TTT layers.
/// TTTInnerModel is implemented using a fused kernel,
/// but uses the same underlying types as the regular version.
///
/// We can't write the type bound due to a limitation of the Module derive macro,
/// but all impls guarantee:
///     Inner: TTTInnerModel<B>
#[derive(Module, Debug)]
pub struct Fused<B: Backend, Inner> {
    inner: Inner,
    _backend: PhantomData<B>,
}

impl<B: Backend, T: TTTInnerModel<B>> From<T> for Fused<B, T> {
    fn from(inner: T) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }
}
