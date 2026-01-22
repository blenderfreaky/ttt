use std::fmt::Debug;

/// Generic trait for tensor bundles with a fixed number of elements.
///
/// `N` is the number of tensors in the bundle.
pub trait TensorBundle<T: Debug + Clone + Send, const N: usize>:
    Sized + Clone + Send + Debug
{
    type Mapped<U: Debug + Clone + Send>: TensorBundle<U, N>;

    fn map<U: Debug + Clone + Send>(self, f: impl FnMut(T) -> U) -> Self::Mapped<U>;
    fn into_array(self) -> [T; N];
    fn from_array(arr: [T; N]) -> Self;

    fn try_map<U: Debug + Clone + Send, E: Debug + Clone + Send>(
        self,
        mut f: impl FnMut(T) -> Result<U, E>,
    ) -> Result<Self::Mapped<U>, E> {
        let arr = self.into_array();
        let results: Result<Vec<U>, E> = arr.into_iter().map(|t| f(t)).collect();
        let results_arr: [U; N] = results?.try_into().ok().unwrap();
        Ok(Self::Mapped::from_array(results_arr))
    }
}

/// Declares a tensor bundle struct with automatic TensorBundle implementation.
///
/// # Example
/// ```ignore
/// tensor_bundle! {
///     /// My bundle of tensors
///     pub struct MyInputs[3] { xq, xk, xv }
///     scalars { epsilon: f32 = 0.0 }
/// }
/// ```
///
/// This generates:
/// - The struct with all fields public
/// - `TensorBundle<T, N>` impl with map, into_array, from_array
/// - `HasClient` impl for Fusion (using first field)
/// - `<scalar>()` builder methods for each scalar
#[macro_export]
macro_rules! tensor_bundle {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident [$n:literal] { $first_field:ident $(, $field:ident)* $(,)? }
        $(scalars { $($scalar:ident : $scalar_ty:ty = $scalar_default:expr),* $(,)? })?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        $vis struct $name<T> {
            pub $first_field: T,
            $(pub $field: T,)*
            $($(
                pub $scalar: $scalar_ty,
            )*)?
        }

        impl<T: std::fmt::Debug + Clone + Send> $crate::ttt::cubecl_kernels::TensorBundle<T, $n> for $name<T> {
            type Mapped<U: std::fmt::Debug + Clone + Send> = $name<U>;

            fn map<U: std::fmt::Debug + Clone + Send>(self, mut f: impl FnMut(T) -> U) -> $name<U> {
                $name {
                    $first_field: f(self.$first_field),
                    $($field: f(self.$field),)*
                    $($($scalar: self.$scalar,)*)?
                }
            }

            fn into_array(self) -> [T; $n] {
                [self.$first_field $(, self.$field)*]
            }

            fn from_array(arr: [T; $n]) -> Self {
                let [$first_field $(, $field)*] = arr;
                $name {
                    $first_field,
                    $($field,)*
                    $($($scalar: $scalar_default,)*)?
                }
            }
        }

        impl<B: burn_fusion::FusionBackend>
            $crate::ttt::cubecl_kernels::impls::HasClient<B>
            for $name<burn::tensor::ops::FloatTensor<burn_fusion::Fusion<B>>>
        {
            fn client(&self) -> &burn_fusion::client::GlobalFusionClient<B::FusionRuntime> {
                &self.$first_field.client
            }
        }

        $crate::tensor_bundle!(@setters $name $($(, $scalar : $scalar_ty)*)?);
    };

    // Generate setters for scalars
    (@setters $name:ident) => {};
    (@setters $name:ident, $($scalar:ident : $scalar_ty:ty),+) => {
        impl<T> $name<T> {
            $(
                pub fn $scalar(mut self, value: $scalar_ty) -> Self {
                    self.$scalar = value;
                    self
                }
            )+
        }
    };
}

pub use crate::tensor_bundle;
