//! Parameter sweep validation tests.
//!
//! These tests run the Python reference validation with various parameter combinations,
//! then run the Rust validation to verify correctness across different configurations.
//!
//! Tests are parameterized using test_case macros and will fail if validation fails.

use std::process::Command;
use test_case::test_case;

/// Run Python validation data generation with given config, returns success status.
fn generate_validation_data(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
}

impl Config {
    fn new(b: usize, l: usize, h: usize, d: usize, mini_batch_size: usize, seed: usize) -> Self {
        Self {
            b,
            l,
            h,
            d,
            mini_batch_size,
            seed,
            use_gate: true,
            conv_kernel: 4,
            pre_conv: true,
            share_qk: true,
            tie_word_embeddings: true,
        }
    }

    fn with_flags(mut self, use_gate: bool, conv_kernel: usize, pre_conv: bool) -> Self {
        self.use_gate = use_gate;
        self.conv_kernel = conv_kernel;
        self.pre_conv = pre_conv;
        self
    }

    fn with_share_qk(mut self, share_qk: bool) -> Self {
        self.share_qk = share_qk;
        self
    }

    fn with_tie_word_embeddings(mut self, tie_word_embeddings: bool) -> Self {
        self.tie_word_embeddings = tie_word_embeddings;
        self
    }
}

/// Run Python validation data generation with given config
fn run_python_validation_data_generation(cfg: &Config) -> bool {
) -> Result<(), String> {
    let mut args = vec![
        "-m".to_string(),
        "reference.validate".to_string(),
        "--B".to_string(),
        b.to_string(),
        "--L".to_string(),
        l.to_string(),
        "--H".to_string(),
        h.to_string(),
        "--D".to_string(),
        d.to_string(),
        "--mini_batch_size".to_string(),
        mini_batch_size.to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--conv_kernel".to_string(),
        conv_kernel.to_string(),
    ];

    if use_gate {
        args.push("--use_gate".to_string());
    } else {
        args.push("--no-use_gate".to_string());
    }

    if pre_conv {
        args.push("--pre_conv".to_string());
    } else {
        args.push("--no-pre_conv".to_string());
    }

    if cfg.share_qk {
        args.push("--share_qk".to_string());
    } else {
        args.push("--no-share_qk".to_string());
    }

    if cfg.tie_word_embeddings {
        args.push("--tie_word_embeddings".to_string());
    } else {
        args.push("--no-tie_word_embeddings".to_string());
    }

    let output = Command::new("python3")
        .args(&args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("Failed to run Python: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Python validation generation failed: {stderr}"))
    }
}

/// Run Rust validation tests, returns success status with error details on failure.
fn run_rust_validation() -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["test", "--test", "validate_full", "--", "--nocapture"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("Failed to run cargo test: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Validation failed:\n{stdout}\n{stderr}"))
    }
}

/// Run a single validation sweep with the given parameters.
fn run_sweep(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
) {
    // Generate validation data
    generate_validation_data(b, l, h, d, mini_batch_size, seed, use_gate, conv_kernel, pre_conv)
        .expect("Failed to generate validation data");

    // Run Rust validation
    run_rust_validation().expect("Rust validation failed");
}

// ============================================================================
// Dimension sweep tests
// ============================================================================

// Baseline configuration
#[test_case(2, 16, 4, 16, 16, 42; "baseline")]
// Vary batch size
#[test_case(1, 16, 4, 16, 16, 42; "batch_1")]
#[test_case(3, 16, 4, 16, 16, 42; "batch_3")]
#[test_case(4, 16, 4, 16, 16, 42; "batch_4")]
// Vary num_heads
#[test_case(2, 16, 2, 16, 16, 42; "heads_2")]
#[test_case(2, 16, 3, 16, 16, 42; "heads_3")]
#[test_case(2, 16, 5, 16, 16, 42; "heads_5")]
#[test_case(2, 16, 6, 16, 16, 42; "heads_6")]
#[test_case(2, 16, 8, 16, 16, 42; "heads_8")]
// Vary head_dim
#[test_case(2, 16, 4, 8, 16, 42; "head_dim_8")]
#[test_case(2, 16, 4, 12, 16, 42; "head_dim_12")]
#[test_case(2, 16, 4, 20, 16, 42; "head_dim_20")]
#[test_case(2, 16, 4, 32, 16, 42; "head_dim_32")]
// Vary sequence length with matching mini_batch
#[test_case(2, 8, 4, 16, 8, 42; "seq_8")]
#[test_case(2, 24, 4, 16, 24, 42; "seq_24")]
#[test_case(2, 32, 4, 16, 32, 42; "seq_32")]
// Multiple mini-batches
#[test_case(2, 32, 4, 16, 16, 42; "mini_batches_2")]
#[test_case(2, 48, 4, 16, 16, 42; "mini_batches_3")]
#[ignore] // Run with: cargo test --test validate_sweep -- --ignored
fn sweep_dimensions(b: usize, l: usize, h: usize, d: usize, mini_batch_size: usize, seed: usize) {
    run_sweep(b, l, h, d, mini_batch_size, seed, true, 4, true);
}

// ============================================================================
// Feature flag sweep tests
// ============================================================================

#[test_case(true, 4, true; "all_features")]
#[test_case(false, 4, true; "no_gate")]
#[test_case(true, 4, false; "no_pre_conv")]
#[test_case(false, 4, false; "minimal")]
#[test_case(true, 2, true; "conv_kernel_2")]
#[test_case(true, 8, true; "conv_kernel_8")]
#[ignore]
fn sweep_features(use_gate: bool, conv_kernel: usize, pre_conv: bool) {
    run_sweep(2, 16, 4, 16, 16, 42, use_gate, conv_kernel, pre_conv);
}

// ============================================================================
// Combined variation tests (potential edge cases)
// ============================================================================

// Tests with default flags (gate=true, conv=4, pre_conv=true)
#[test_case(2, 16, 5, 8, 16, 42; "h5_d8")]
#[test_case(3, 20, 5, 8, 10, 42; "complex_1")]
#[test_case(2, 24, 6, 12, 12, 42; "complex_2")]
#[ignore]
fn sweep_combined(b: usize, l: usize, h: usize, d: usize, mini_batch_size: usize, seed: usize) {
    run_sweep(b, l, h, d, mini_batch_size, seed, true, 4, true);
}

// Tests with non-default flags
#[test_case(2, 32, 4, 16, 16, 42, true, 8, true; "multi_batch_large_conv")]
#[test_case(2, 16, 6, 12, 16, 42, true, 2, true; "varied_dims_small_conv")]
#[ignore]
fn sweep_combined_with_flags(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
) {
    run_sweep(b, l, h, d, mini_batch_size, seed, use_gate, conv_kernel, pre_conv);
}
