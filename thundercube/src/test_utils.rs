use std::fmt::Display;

use cubecl::{TestRuntime, prelude::*, server::Handle};

use half::{bf16, f16};
use rand::{Rng, rngs::StdRng};

/// A macro for testing CubeCL GPU kernels by comparing their output against reference implementations.
///
/// # Syntax
///
/// ```ignore
/// test_kernel! {
///     #[test_attributes]
///     fn test_name(test_args) for TypeName in [type_list] or all {
///         seed(optional_seed);
///
///         let var_name: VarType = [dims] as Distribution;
///
///         assert_eq!(
///             kernel_name(arg_spec, ...) for (cube_x, cube_y, cube_z) @ dim_spec,
///             { reference_code }
///         );
///     }
/// }
/// ```
///
/// # Components
///
/// - **Test attributes**: Standard Rust test attributes like `#[test]`, `#[test_case(...)]`, `#[test_matrix(...)]`
/// - **Type specification**: `for TypeName in [f32, f64]` or `for TypeName in all` (expands to bf16, f16, f32, f64).
///   Custom type aliases use `Type | alias` syntax (e.g., `::half::f16 | f16`).
/// - **Seed**: Optional `seed(u64)` for deterministic RNG. Defaults to 42.
/// - **Variables**: Declare test data with `let name: Type = [dims] as Distribution;`
///   - `Type`: `Tensor` or `Array`
///   - `dims`: Shape dimensions (e.g., `[4, 8]`)
///   - `Distribution`: `Range` (0..len) or `Uniform(start, end)`. Defaults to `Uniform(-10.0, 10.0)`.
/// - **Kernel launch**: `kernel_name(args) for (cube_count) @ dim_spec`
///   - `args`: `var_name()` passes the variable, `lit(expr)` passes a literal value
///   - `cube_count`: `(x, y, z)` cube count for launch
///   - `dim_spec`: `(x, y, z)` fixed dimensions or `max(n)` for max supported dimensions
/// - **Reference code**: A block that mutates declared variables to their expected values
///
/// # Example
///
/// ```ignore
/// test_kernel! {
///     #[test_case(4, 4)]
///     fn my_kernel_test(rows: usize, cols: usize) for F in all {
///         seed(123);
///         let input: Tensor = [rows, cols];
///         let output: Tensor = [rows, cols] as Range;
///         assert_eq!(
///             my_kernel(input(), output()) for (1, 1, 1) @ (32, 32),
///             {
///                 // Compute expected values in `output`
///                 for i in 0..output.len() {
///                     output[i] = input[i] * F::from_int(2);
///                 }
///             }
///         );
///     }
/// }
/// ```
#[macro_export]
macro_rules! test_kernel {
    // ==================== ENTRY POINT ====================
    // Parses the user-facing syntax and normalizes it into internal representation.
    // Accepts: `fn name(args) for TypeName in [types] or all { ... }`
    {
    $(
        $(#[$attr:meta])*
        fn $name:ident($($args:tt)*)
            for $tname:ident in $([$($float_type:tt),*])? $($all:ident)?
        {
            $(seed($seed:expr);)?

            $(
            let $var:ident: $vty:ty = [$($val:expr),*] $(as $distrib:ident $(($($distrib_param:expr),+))?)?;
            )*

            assert_eq!(
                $kernel:ident ($($kernel_arg_name:ident($($kernel_arg:expr)?)),*) for ($($count:expr),*) @ $($max:ident)? ($($dim:expr),*),
                $ref:expr $(,)?
            );
        }
    )*
    } => {
    $(
        // Rewrite into internal tagged format for further processing
        test_kernel! {
            types: $([$($float_type),*])? $($all)?;
            type_name: $tname;
            attrs: $(#[$attr])*;
            name: $name;
            args: ($($args)*);
            seed: ($($seed)?);
            vars: $(let $var: $vty = [$($val),*] $(as $distrib $(($($distrib_param),+))?)?;)*;
            kernel: $kernel;
            kernel_args: ($($kernel_arg_name($($kernel_arg)?)),*);
            count: ($($count),*);
            dim: $($max)? ($($dim),*);
            ref: $ref;
        }
    )*
    };

    // ==================== TYPE LIST ITERATION ====================
    // These arms iterate over the type list, generating one test per type.

    // Base case: empty type list, stop recursion
    {
        types: [];
        $($rest:tt)*
    } => {
    };

    // `all` expands to the four standard float types
    {
        types: all;
        $($rest:tt)*
    } => {
        test_kernel! {
            types: [::half::bf16 | bf16, ::half::f16 | f16, f32, f64];
            $($rest)*
        }
    };

    // Type with explicit alias: `Type | alias` (e.g., `::half::f16 | f16`)
    {
        types: [$t:ty | $tid:ident $(, $($other_types:tt)+)?];
        $($rest:tt)*
    } => {
        // Generate test for this type
        test_kernel! {
            type: $t | $tid;
            $($rest)*
        }
        // Recurse for remaining types
        test_kernel! {
            types: [$($($other_types)+)?];
            $($rest)*
        }
    };

    // Type without alias: use the type name as both type and identifier (e.g., `f32`)
    {
        types: [$t:tt $(, $($other_types:tt),+)?];
        $($rest:tt)*
    } => {
        test_kernel! {
            type: $t | $t;
            $($rest)*
        }
        // Recurse for remaining types
        test_kernel! {
            types: [$($($other_types),+)?];
            $($rest)*
        }
    };

    // ==================== FUNCTION GENERATION ====================
    // Creates the actual test function with type suffix (e.g., `test_name_f32`)
    {
        type: $t:tt | $tid:ident;
        type_name: $tname:ident;
        attrs: $(#[$attr:meta])*;
        name: $name:ident;
        args: ($($args:tt)*);
        $($rest:tt)*
    } => {
        ::paste::paste! {
            $(#[$attr])*
            #[allow(unused_mut)]
            fn [< $name _ $tid >]($($args)*) {
                // Create type alias so user can refer to the float type by name
                #[allow(dead_code)]
                type $tname = $t;
                test_kernel! {
                    @inner
                    type: $t;
                    args: ($($args)*);
                    $($rest)*
                }
            }
        }
    };

    // ==================== TEST BODY (@inner) ====================
    // Generates the actual test implementation: setup, kernel launch, and verification
    {
        @inner
        type: $t:ty;
        args: ($($args:tt)*);
        seed: ($($seed:expr)?);
        vars: $(let $var:ident: $vty:tt = [$($val:expr),*] $(as $distrib:ident $(($($distrib_param:expr),+))?)?;)*;
        kernel: $kernel:ident;
        kernel_args: ($($kernel_arg_name:ident($($kernel_arg:expr)?)),*);
        count: ($($count:expr),*);
        dim: $($max:ident)? ($($dim:expr),*);
        ref: $ref:expr;
    } => {
        ::paste::paste! {
            // 1. Setup: get compute client and RNG
            let client = $crate::test_utils::client();

            use rand::SeedableRng;
            #[allow(unused_variables)]
            let mut rng = rand::rngs::StdRng::seed_from_u64(test_kernel!{ @seed ($($seed)?) });

            // 2. Initialize variables: create data, upload to GPU, build kernel args
            //    We're passing all the identifiers here because if they were
            //    generated by the macro, they would be unique and not accessible
            //    outside of that invocation due to hygiene.
            $(
            test_kernel!{ @val($t, rng, client) $vty = [$($val),*] $(as $distrib $(($($distrib_param),+))?)?;
                [< $var _shape >], [< $var _strides >], [< $var _len >],
                $var, [< $var _handle >], [< $var _arg >]
            }
            )*

            // 3. Launch the kernel
            println!("Launching kernel");
            $kernel::launch::<$t, cubecl::TestRuntime>(
                &client,
                CubeCount::Static($(($count) as u32),*),
                test_kernel!{ @dim(client) $($max)? ($($dim),*) },
                $(
                    test_kernel!{ @arg([<$kernel_arg_name _arg>]) $kernel_arg_name($($kernel_arg)?) }
                ),*
            ).expect("Kernel launch failed");

            // 4. Compute reference values (mutates the local `$var` vectors)
            println!("Computing reference");
            $ref;

            // 5. Download GPU results and compare against reference
            println!("Checking results");
            $(
            test_kernel!{ @check(client) $var == [< $var _handle >] }
            )*
        }
    };

    // ==================== HELPER: KERNEL ARGUMENTS (@arg) ====================
    // `var()` - pass the variable's kernel arg
    { @arg($arg_name:ident) $_:ident() } => { $arg_name };
    // `lit(expr)` - pass a literal value directly
    { @arg($arg_name:ident) lit($arg:expr) } => { $arg };

    // ==================== HELPER: SEED (@seed) ====================
    { @seed () } => { 42 };  // Default seed
    { @seed ($seed:expr) } => { $seed };

    // ==================== HELPER: CUBE DIMENSIONS (@dim) ====================
    // `max(n)` - query device for max supported dimensions
    { @dim($client:ident) max($dim:expr) } => { CubeDim::new(&$client, $dim) };
    // Fixed dimensions
    { @dim($client:ident) ($x:expr) } => { CubeDim::new_1d($x) };
    { @dim($client:ident) ($x:expr, $y:expr) } => { CubeDim::new_2d($x, $y) };
    { @dim($client:ident) ($x:expr, $y:expr, $z:expr) } => { CubeDim::new_3d($x, $y, $z) };

    // ==================== HELPER: VARIABLE INITIALIZATION (@val) ====================
    // Creates shape, strides, data vector, GPU handle, and kernel argument for a variable
    { @val($t:ty, $rng:ident, $client:ident) $vty:tt = [$($val:expr),*] $(as $distrib:tt $(($($distrib_param:tt),+))?)?;
        $shape:ident, $strides:ident, $len:ident, $data:ident, $handle:ident, $arg:ident
    } => {
        let $shape = vec![$($val),*];
        println!("Initializing {} with shape {:?}", stringify!($vty), $shape);
        let $strides = $crate::test_utils::get_strides(&$shape);
        let $len: usize = $shape.iter().product();
        let mut $data = test_kernel!{ @init_val($t, $rng, $len) $(as $distrib $(($($distrib_param),+))?)? };
        let $handle = $crate::test_utils::upload(&$client, &$data);
        println!("Strides: {:?}", $strides);
        println!("Length: {}", $len);
        assert_eq!($len % $crate::LINE_SIZE, 0, "Length must be a multiple of LINE_SIZE");
        test_kernel!{ @make_arg($t) $vty; $handle, $strides, $shape, $len, $arg }
    };

    // ==================== HELPER: DATA INITIALIZATION (@init_val) ====================
    // No distribution specified: default to Uniform(-10, 10)
    { @init_val($t:ty, $rng:ident, $len:ident) } => {
        test_kernel!{ @init_val($t, $rng, $len) as Uniform(-10.0, 10.0) }
    };
    // `as Range` - sequential values 0, 1, 2, ...
    { @init_val($t:ty, $rng:ident, $len:ident) as Range } => {
        $crate::test_utils::range_vec::<$t>($len)
    };
    // `as Uniform(start, end)` - random values in [start, end)
    { @init_val($t:ty, $rng:ident, $len:ident) as Uniform($start:expr, $end:expr) } => {
        $crate::test_utils::random_vec::<$t>(&mut $rng, $len, $start, $end)
    };

    // ==================== HELPER: KERNEL ARG CONSTRUCTION (@make_arg) ====================
    // Build ArrayArg for `Array` type variables
    { @make_arg($t:ty) Array; $handle:ident, $strides:ident, $shape:ident, $len:ident, $arg:ident } => {
        let $arg: ArrayArg<'_, cubecl::TestRuntime> = unsafe {
            ArrayArg::from_raw_parts::<Line<$t>>(&$handle, $len, $crate::LINE_SIZE)
        };
    };
    // Build TensorArg for `Tensor` type variables
    { @make_arg($t:ty) Tensor; $handle:ident, $strides:ident, $shape:ident, $len:ident, $arg:ident } => {
        let $arg: TensorArg<'_, cubecl::TestRuntime> = unsafe {
            TensorArg::from_raw_parts::<Line<$t>>(&$handle, &$strides, &$shape, $crate::LINE_SIZE)
        };
    };

    // ==================== HELPER: RESULT VERIFICATION (@check) ====================
    // Downloads GPU data and compares against expected values
    { @check($client:ident) $var:ident == $handle:ident } => {
        paste::paste! {
            println!("Comparing {}", stringify!($var));
            let [< $var _kernel_data >] = $crate::test_utils::download(&$client, $handle);
            $crate::test_utils::slices_eq(&[< $var _kernel_data >], &$var, stringify!($var))
        }
    };
}

pub fn client() -> TestClient {
    TestRuntime::client(&<TestRuntime as cubecl::Runtime>::Device::default())
}

pub type TestClient = ComputeClient<TestRuntime>;

pub fn range_vec<F: TestFloat>(len: usize) -> Vec<F> {
    (0..len).map(|i| F::from_int(i as i64)).collect()
}

pub fn random_vec<F: TestFloat>(rng: &mut StdRng, len: usize, start: f64, end: f64) -> Vec<F> {
    (0..len)
        .map(|_| F::from_f64(rng.random_range(start..end)))
        .collect()
}

pub fn upload<F: TestFloat>(client: &TestClient, data: &[F]) -> Handle {
    client.create_from_slice(F::as_bytes(data))
}

pub fn download<F: TestFloat>(client: &TestClient, handle: Handle) -> Vec<F> {
    F::from_bytes(&client.read_one(handle)).to_vec()
}

pub fn get_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Assert two values are approximately equal.
pub fn approx_eq<F: TestFloat>(actual: F, expected: F) -> bool {
    let (a, e) = (actual.into_f64(), expected.into_f64());
    let diff = (a - e).abs();
    let tol = F::atol() + F::rtol() * e.abs();
    diff <= tol
}

/// Assert slices are approximately equal.
pub fn slices_eq<F: TestFloat>(actual: &[F], expected: &[F], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");

    let mut passed = true;

    let mut avg_magnitude_actual = 0.0;
    let mut avg_magnitude_expected = 0.0;

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !approx_eq(a, e) {
            passed = false;
            println!("{ctx}[{i}] failed: expected {e}, got {a}");
        }
        avg_magnitude_actual += a.into_f64().abs();
        avg_magnitude_expected += e.into_f64().abs();
    }

    println!(
        "Average magnitude of actual values: {}",
        avg_magnitude_actual / actual.len() as f64
    );
    println!(
        "Average magnitude of expected values: {}",
        avg_magnitude_expected / expected.len() as f64
    );

    if !passed {
        panic!("{} failed", ctx);
    }
}

pub trait TestFloat: CubeElement + CubePrimitive + Float + Copy + Display {
    fn into_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn rtol() -> f64;
    fn atol() -> f64;
}

impl TestFloat for f64 {
    fn into_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn rtol() -> f64 {
        1e-12
    }
    fn atol() -> f64 {
        1e-12
    }
}

impl TestFloat for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn rtol() -> f64 {
        1e-4
    }
    fn atol() -> f64 {
        1e-4
    }
}

impl TestFloat for f16 {
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn rtol() -> f64 {
        1e-2
    }
    fn atol() -> f64 {
        1e-2
    }
}

impl TestFloat for bf16 {
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }
    fn rtol() -> f64 {
        5e-2
    }
    fn atol() -> f64 {
        5e-2
    }
}

// These tests are for testing the macro, not any actual CubeCL code
#[cfg(false)]
mod test_macro_tests {
    use test_case::{test_case, test_matrix};

    use crate::LINE_SIZE;
    use crate::test_utils::TestFloat;
    use cubecl::num_traits::Zero;
    use cubecl::prelude::*;

    #[cube(launch)]
    fn noop<F: Float>(_x: Tensor<Line<F>>) {}

    #[cube(launch)]
    fn zero<F: Float>(x: &mut Tensor<Line<F>>) {
        for i in 0..x.len() {
            x[i] = Line::empty(LINE_SIZE).fill(F::from_int(0));
        }
    }

    test_kernel! {
        #[test_case(2, 2)]
        #[test_case(4, 4)]
        fn noop(a: usize, b: usize) for F in all {
            let x: Tensor = [a, b];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ (1, 1),
                {}
            );
        }

        #[test_matrix([4, 8], [4, 8])]
        fn zero(a: usize, b: usize) for F in all {
            let x: Tensor = [a, b];
            assert_eq!(
                zero(x()) for (1, 1, 1) @ (1, 1),
                {
                    x.fill(F::zero());
                }
            );
        }

        #[test]
        #[should_panic = "x failed"]
        fn zero_panic() for F in all {
            let x: Tensor = [4];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x.fill(F::zero());
                }
            );
        }

        #[test]
        #[should_panic = "Length must be a multiple of LINE_SIZE"]
        fn not_line_aligned() for F in all {
            let x: Tensor = [3, 3];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {}
            );
        }

        #[test]
        fn consistent_seed() for F in all {
            let x: Tensor = [4];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x[0] = F::from_f64(0.5311481800554763);
                }
            );
        }

        #[test_matrix([4, 16])]
        fn range_vec(n: usize) for F in all {
            let x: Tensor = [n] as Range;
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    for f in 0..n {
                        x[f] = F::from_int(f as i64);
                    }
                }
            );
        }
    }
}
