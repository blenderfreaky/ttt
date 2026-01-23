//! Parameter sweep validation tests.
//!
//! These tests run the Python reference validation with various parameter combinations,
//! then run the Rust validation to verify correctness across different configurations.
//!
//! Tests are parameterized using test_case macros and will fail if validation fails.
#![allow(clippy::too_many_arguments)]

use std::{path::PathBuf, process::Command};
use test_case::test_case;

mod validate_full;

/// Run Python validation data generation with given config, returns success status.
fn generate_validation_data(
    output_dir: String,
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
) -> Result<(), String> {
    fn bool_flag(flag: bool, name: &str) -> String {
        if flag {
            format!("--{}", name)
        } else {
            format!("--no-{}", name)
        }
    }

    let args = vec![
        "-m".to_string(),
        "reference.validate".to_string(),
        "--output_dir".to_string(),
        output_dir.to_string(),
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
        bool_flag(use_gate, "use_gate"),
        bool_flag(pre_conv, "pre_conv"),
        bool_flag(share_qk, "share_qk"),
        bool_flag(tie_word_embeddings, "tie_word_embeddings"),
    ];

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
fn run_rust_validation(dir: String) {
    validate_full::test_all(Some(PathBuf::from(dir)));
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
    share_qk: bool,
    tie_word_embeddings: bool,
) {
    let dir = format!(
        "validation_data/batch_{b}/layer_{l}/head_{h}/dim_{d}/mini_batch_{mini_batch_size}/seed_{seed}/conv_{conv_kernel}/gate_{use_gate}_pre_conv_{pre_conv}_share_qk_{share_qk}_tie_word_embeddings_{tie_word_embeddings}"
    );
    generate_validation_data(
        dir.clone(),
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        use_gate,
        conv_kernel,
        pre_conv,
        share_qk,
        tie_word_embeddings,
    )
    .expect("Failed to generate validation data");

    run_rust_validation(dir);
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
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        true,
        4,
        true,
        false,
        false,
    );
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
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        use_gate,
        conv_kernel,
        pre_conv,
        false,
        false,
    );
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
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        true,
        4,
        true,
        false,
        false,
    );
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
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        use_gate,
        conv_kernel,
        pre_conv,
        false,
        false,
    );
}

// ============================================================================
// Weight sharing sweep tests
// ============================================================================
#[test_case(false, false; "no_sharing")]
#[test_case(true, false; "share_qk_only")]
#[test_case(false, true; "tie_embeddings_only")]
#[test_case(true, true; "share_qk_and_tie_embeddings")]
#[ignore]
fn sweep_weight_sharing(share_qk: bool, tie_word_embeddings: bool) {
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        true, // use_gate
        4,    // conv_kernel
        true, // pre_conv
        share_qk,
        tie_word_embeddings,
    );
}

// ============================================================================
// Weight sharing with varied dimensions
// ============================================================================
#[test_case(1, 16, 4, 16, 16, 42, true, false; "batch_1_share_qk")]
#[test_case(2, 32, 4, 16, 16, 42, true, false; "multi_batch_share_qk")]
#[test_case(2, 16, 8, 16, 16, 42, true, false; "many_heads_share_qk")]
#[test_case(2, 16, 4, 32, 16, 42, false, true; "large_head_dim_tie_emb")]
#[test_case(3, 24, 6, 12, 12, 42, true, true; "complex_all_sharing")]
#[ignore]
fn sweep_sharing_with_dimensions(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    share_qk: bool,
    tie_word_embeddings: bool,
) {
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        true, // use_gate
        4,    // conv_kernel
        true, // pre_conv
        share_qk,
        tie_word_embeddings,
    );
}

// ============================================================================
// Full feature matrix tests (weight sharing + other flags)
// ============================================================================
#[test_case(true, 4, true, true, false; "gate_conv4_pre_shareqk")]
#[test_case(true, 4, true, false, true; "gate_conv4_pre_tieemb")]
#[test_case(true, 4, true, true, true; "gate_conv4_pre_allsharing")]
#[test_case(false, 4, true, true, false; "nogate_shareqk")]
#[test_case(true, 2, false, true, true; "conv2_nopreconv_allsharing")]
#[test_case(false, 8, false, true, true; "minimal_with_allsharing")]
#[ignore]
fn sweep_full_feature_matrix(
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
) {
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        use_gate,
        conv_kernel,
        pre_conv,
        share_qk,
        tie_word_embeddings,
    );
}
