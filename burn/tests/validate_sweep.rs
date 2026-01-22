//! Parameter sweep test to narrow down dimension-dependent bugs.
//!
//! This test runs the Python reference validation with various parameter combinations,
//! then runs the Rust validation and reports which combinations pass/fail.
//!
//! Run with: cargo test --test validate_sweep -- --nocapture --ignored

use std::process::Command;

/// A parameter configuration to test
#[derive(Debug, Clone)]
struct Config {
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
    let mut args = vec![
        "-m".to_string(),
        "reference.validate".to_string(),
        "--B".to_string(),
        cfg.b.to_string(),
        "--L".to_string(),
        cfg.l.to_string(),
        "--H".to_string(),
        cfg.h.to_string(),
        "--D".to_string(),
        cfg.d.to_string(),
        "--mini_batch_size".to_string(),
        cfg.mini_batch_size.to_string(),
        "--seed".to_string(),
        cfg.seed.to_string(),
        "--conv_kernel".to_string(),
        cfg.conv_kernel.to_string(),
    ];

    if cfg.use_gate {
        args.push("--use_gate".to_string());
    } else {
        args.push("--no-use_gate".to_string());
    }

    if cfg.pre_conv {
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
        .expect("Failed to run Python validation data generation");

    output.status.success()
}

/// Run Rust validation tests (the existing validate_full tests)
fn run_rust_validate() -> (bool, String) {
    let output = Command::new("cargo")
        .args([
            "test",
            "--test",
            "validate_full",
            "--",
            "--nocapture",
            // "test_block_forward",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to run Rust validation");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    // Extract max_diff from output if present
    // let max_diff = combined
    //     .lines()
    //     .find(|line| line.contains("max_diff="))
    //     .map(|s| s.to_string())
    //     .unwrap_or_default();
    let max_diff = if output.status.success() {
        String::new()
    } else {
        combined
    };

    (output.status.success(), max_diff)
}

#[test]
#[ignore] // Run explicitly with: cargo test --test validate_sweep -- --ignored --nocapture
fn sweep_dimensions() {
    println!("\n========================================");
    println!("Parameter Sweep Test");
    println!("========================================\n");

    // Test configurations - vary one parameter at a time from working baseline
    // Default: use_gate=true, conv_kernel=4, pre_conv=true (most complex)
    let configs = vec![
        // Baseline (known working)
        Config::new(2, 16, 4, 16, 16, 42),
        // Vary H (num_heads)
        Config::new(2, 16, 2, 16, 16, 42),
        Config::new(2, 16, 3, 16, 16, 42),
        Config::new(2, 16, 5, 16, 16, 42),
        Config::new(2, 16, 6, 16, 16, 42),
        Config::new(2, 16, 8, 16, 16, 42),
        // Vary D (head_dim)
        Config::new(2, 16, 4, 8, 16, 42),
        Config::new(2, 16, 4, 12, 16, 42),
        Config::new(2, 16, 4, 14, 16, 42),
        Config::new(2, 16, 4, 20, 16, 42),
        Config::new(2, 16, 4, 32, 16, 42),
        // Vary L (seq_len) with matching mini_batch
        Config::new(2, 8, 4, 16, 8, 42),
        Config::new(2, 24, 4, 16, 24, 42),
        Config::new(2, 32, 4, 16, 32, 42),
        // Vary B (batch_size)
        Config::new(1, 16, 4, 16, 16, 42),
        Config::new(3, 16, 4, 16, 16, 42),
        Config::new(4, 16, 4, 16, 16, 42),
        // Multiple mini-batches
        Config::new(2, 32, 4, 16, 16, 42), // 2 mini-batches
        Config::new(2, 48, 4, 16, 16, 42), // 3 mini-batches
        // Combined variations (potential failure cases)
        Config::new(2, 16, 5, 8, 16, 42), // H=5, D=8 (failed before)
        Config::new(3, 20, 5, 8, 10, 42), // Similar to failed block test
        Config::new(2, 24, 6, 12, 12, 42), // Similar to failed layer test
        // Vary conv_kernel (with gate and pre_conv on)
        Config::new(2, 16, 4, 16, 16, 42).with_flags(true, 2, true),
        Config::new(2, 16, 4, 16, 16, 42).with_flags(true, 8, true),
        // Without gate (baseline comparison)
        Config::new(2, 16, 4, 16, 16, 42).with_flags(false, 4, true),
        Config::new(2, 16, 4, 16, 16, 42).with_flags(false, 4, false),
        // Without pre_conv (baseline comparison)
        Config::new(2, 16, 4, 16, 16, 42).with_flags(true, 4, false),
        // Complex combos with flags
        Config::new(2, 32, 4, 16, 16, 42).with_flags(true, 8, true), // 2 mini-batches, large conv
        Config::new(2, 16, 6, 12, 16, 42).with_flags(true, 2, true), // different H/D, small conv
        // share_qk variations
        Config::new(2, 16, 4, 16, 16, 42).with_share_qk(false),
        Config::new(2, 16, 4, 16, 16, 42).with_share_qk(false).with_flags(false, 4, true),
        // tie_word_embeddings variations
        Config::new(2, 16, 4, 16, 16, 42).with_tie_word_embeddings(false),
        Config::new(2, 16, 4, 16, 16, 42).with_share_qk(false).with_tie_word_embeddings(false),
    ];

    let mut results = Vec::new();

    for cfg in &configs {
        print!(
            "Testing B={}, L={}, H={}, D={}, mini_batch={}, seed={}, gate={}, conv={}, pre_conv={}, share_qk={}, tie_embed={}",
            cfg.b,
            cfg.l,
            cfg.h,
            cfg.d,
            cfg.mini_batch_size,
            cfg.seed,
            cfg.use_gate,
            cfg.conv_kernel,
            cfg.pre_conv,
            cfg.share_qk,
            cfg.tie_word_embeddings
        );

        // Generate Python reference data
        if !run_python_validation_data_generation(cfg) {
            println!("PYTHON FAILED");
            results.push((cfg.clone(), false, "Python failed".to_string()));
            continue;
        }

        // Run Rust validation
        let (passed, details) = run_rust_validate();
        if passed {
            println!("PASS");
        } else {
            println!("FAIL - {}", details);
        }
        results.push((cfg.clone(), passed, details));
    }

    // Summary
    println!("\n========================================");
    println!("Summary");
    println!("========================================\n");

    let passed_count = results.iter().filter(|(_, p, _)| *p).count();
    let failed_count = results.len() - passed_count;

    println!("Passed: {}/{}", passed_count, results.len());
    println!("Failed: {}/{}", failed_count, results.len());

    if failed_count > 0 {
        println!("\nFailed configurations:");
        for (cfg, passed, details) in &results {
            if !passed {
                println!(
                    "  B={}, L={}, H={}, D={}, mini_batch={}, seed={}, gate={}, conv={}, pre_conv={}, share_qk={}, tie_embed={} - {}",
                    cfg.b,
                    cfg.l,
                    cfg.h,
                    cfg.d,
                    cfg.mini_batch_size,
                    cfg.seed,
                    cfg.use_gate,
                    cfg.conv_kernel,
                    cfg.pre_conv,
                    cfg.share_qk,
                    cfg.tie_word_embeddings,
                    details
                );
            }
        }
    }

    println!("\nPassed configurations:");
    for (cfg, passed, _) in &results {
        if *passed {
            println!(
                "  B={}, L={}, H={}, D={}, mini_batch={}, seed={}, gate={}, conv={}, pre_conv={}, share_qk={}, tie_embed={}",
                cfg.b,
                cfg.l,
                cfg.h,
                cfg.d,
                cfg.mini_batch_size,
                cfg.seed,
                cfg.use_gate,
                cfg.conv_kernel,
                cfg.pre_conv,
                cfg.share_qk,
                cfg.tie_word_embeddings
            );
        }
    }

    // Don't fail the test - just report results
    println!("\n(This test reports results but does not fail)");
}

#[test]
#[ignore]
fn sweep_h_only() {
    println!("\n========================================");
    println!("H (num_heads) Sweep");
    println!("========================================\n");

    // Keep everything else constant, vary only H
    for h in 1..=8 {
        let cfg = Config::new(2, 16, h, 16, 16, 42);
        print!("H={} ... ", h);

        if !run_python_validation_data_generation(&cfg) {
            println!("PYTHON FAILED");
            continue;
        }

        let (passed, details) = run_rust_validate();
        if passed {
            println!("PASS");
        } else {
            println!("FAIL - {}", details);
        }
    }
}

#[test]
#[ignore]
fn sweep_d_only() {
    println!("\n========================================");
    println!("D (head_dim) Sweep - must be even");
    println!("========================================\n");

    // Keep everything else constant, vary only D (must be even)
    for d in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32] {
        let cfg = Config::new(2, 16, 4, d, 16, 42);
        print!("D={} ... ", d);

        if !run_python_validation_data_generation(&cfg) {
            println!("PYTHON FAILED");
            continue;
        }

        let (passed, details) = run_rust_validate();
        if passed {
            println!("PASS");
        } else {
            println!("FAIL - {}", details);
        }
    }
}
