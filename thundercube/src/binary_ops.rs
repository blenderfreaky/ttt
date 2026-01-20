use crate::prelude::*;
use cubecl::prelude::*;

#[cube]
pub trait BinaryOp<F: Float> {
    fn apply(&self, dst: Line<F>, src: Line<F>) -> Line<F>;
}

macro_rules! impl_binary_ops {
    {
    $(
        $name:ident<$t:ident>($dst:ident, $src:ident) => $body:expr;
    )+
    } => {
    $(
        ::paste::paste! {
            #[derive(CubeType)]
            pub struct [<$name Op>];

            // The CubeType derive doesn't handle unit structs too nicely,
            // so we have to hand-impl this
            impl From<[<$name Op>]> for [<$name OpExpand>] {
                fn from(_: [<$name Op>]) -> Self {
                    [<$name OpExpand>] {}
                }
            }

            #[cube]
            impl<$t: Float> BinaryOp<$t> for [<$name Op>] {
                fn apply(&self, $dst: Line<$t>, $src: Line<$t>) -> Line<$t> {
                    $body
                }
            }
        }
    )+
    };
}

macro_rules! impl_binary_convenience_fns {
    {
        for $ty:ty;
        $(
            $name:ident<$t:ident>($dst:ident, $src:ident) => $body:expr;
        )+
    }
    => {
        ::paste::paste! {
            $(
                #[cube]
                impl<$t: Float> $ty<$t> {
                    pub fn [<$name:snake>](&mut self, other: &$ty<$t>) {
                        self.apply_binary_op::<[<$name Op>]>([<$name Op>], other);
                    }
                }
            )+
        }
    };
}

macro_rules! with_binary_ops {
    ($callback:path ; $($($arg:tt)+)?) => {
        $callback! {
            $($($arg)+;)?

            Add<F>(dst, src) => dst + src;
            Sub<F>(dst, src) => dst - src;
            Mul<F>(dst, src) => dst * src;
            Div<F>(dst, src) => dst / src;

            Min<F>(dst, src) => Line::<F>::min(dst, src);
            Max<F>(dst, src) => Line::<F>::max(dst, src);

            Pow<F>(dst, src) => Line::<F>::powf(dst, src);
        }
    };
}

with_binary_ops!(impl_binary_ops;);
with_binary_ops!(impl_binary_convenience_fns; for Rt);
with_binary_ops!(impl_binary_convenience_fns; for St);

#[cfg(test)]
mod tests {
    use crate::binary_ops::*;
    use crate::test_utils::TestFloat;

    macro_rules! generate_binary_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Array<Line<F>>,
                b: &Array<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut rt_a = Rt::<F>::new(LINE_SIZE, LINE_SIZE);
                let mut rt_b = Rt::<F>::new(LINE_SIZE, LINE_SIZE);
                rt_a.copy_from_array(a);
                rt_b.copy_from_array(b);
                rt_a.$method(&rt_b);
                rt_a.copy_to_array(output);
            }
        };
    }

    generate_binary_kernel!(test_add_kernel, add);
    generate_binary_kernel!(test_sub_kernel, sub);
    generate_binary_kernel!(test_mul_kernel, mul);
    generate_binary_kernel!(test_div_kernel, div);
    generate_binary_kernel!(test_min_kernel, min);
    generate_binary_kernel!(test_max_kernel, max);
    generate_binary_kernel!(test_pow_kernel, pow);

    // ==================== TESTS ====================

    test_kernel! {
        #[test]
        fn test_add() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_add_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() + b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_sub() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_sub_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() - b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_mul() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_mul_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() * b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_div() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            // Avoid division by zero by using values away from zero
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.5, 5.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_div_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() / b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_min() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_min_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().min(b[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_max() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_max_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().max(b[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_pow() for F in all {
            // Use positive base values for pow
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.1, 5.0);
            // Use reasonable exponent range
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-2.0, 2.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_pow_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().powf(b[i].into_f64()));
                    }
                }
            );
        }
    }
}
