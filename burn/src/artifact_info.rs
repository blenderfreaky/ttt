use crate::training::TTTTrainingConfig;
use crate::ttt::TTTConfig;
use burn::optim::AdamConfig;
use serde_json::Value;
use std::path::Path;

/// Recursively merge two JSON values. `overlay` values override `base`.
fn merge_json(base: Value, overlay: Value) -> Value {
    match (base, overlay) {
        (Value::Object(mut base_map), Value::Object(overlay_map)) => {
            for (k, v) in overlay_map {
                let merged = if let Some(base_v) = base_map.remove(&k) {
                    merge_json(base_v, v)
                } else {
                    v
                };
                base_map.insert(k, merged);
            }
            Value::Object(base_map)
        }
        (_, overlay) => overlay,
    }
}

/// Create a default config for JSON merging (placeholder values get overwritten)
fn default_config() -> TTTTrainingConfig {
    TTTTrainingConfig::new(TTTConfig::new(0), AdamConfig::new())
}

/// Load config with backwards compatibility for old formats missing newer fields.
fn load_config_compat(path: &Path) -> Result<TTTTrainingConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let loaded: Value = serde_json::from_str(&content)?;

    // Get defaults by serializing a default config
    let defaults = serde_json::to_value(default_config())?;

    // Merge: loaded values override defaults
    let merged = merge_json(defaults, loaded);

    // Deserialize the complete config
    Ok(serde_json::from_value(merged)?)
}

/// Metrics parsed from an epoch's log files
#[derive(Debug, Default)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub loss: Option<f64>,
    pub perplexity: Option<f64>,
    pub accuracy: Option<f64>,
}

/// Summary of training progress and metrics
#[derive(Debug)]
pub struct ArtifactInfo {
    pub config: TTTTrainingConfig,
    pub latest_checkpoint: Option<usize>,
    pub total_epochs: usize,
    pub train_metrics: Vec<EpochMetrics>,
    pub valid_metrics: Vec<EpochMetrics>,
    pub has_final_model: bool,
}

impl ArtifactInfo {
    pub fn load(artifact_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let artifact_path = Path::new(artifact_dir);

        // Load config with backwards compatibility
        let config_path = artifact_path.join("config.json");
        let config = load_config_compat(&config_path)?;

        // Find latest checkpoint
        let checkpoint_dir = artifact_path.join("checkpoint");
        let latest_checkpoint = if checkpoint_dir.exists() {
            std::fs::read_dir(&checkpoint_dir).ok().and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.strip_prefix("model-")?
                            .strip_suffix(".mpk")?
                            .parse()
                            .ok()
                    })
                    .max()
            })
        } else {
            None
        };

        // Check for final model
        let has_final_model = artifact_path.join("model.mpk").exists();

        // Parse training metrics
        let train_metrics = parse_epoch_metrics(&artifact_path.join("train"));
        let valid_metrics = parse_epoch_metrics(&artifact_path.join("valid"));

        let total_epochs = train_metrics.len().max(valid_metrics.len());

        Ok(ArtifactInfo {
            config,
            latest_checkpoint,
            total_epochs,
            train_metrics,
            valid_metrics,
            has_final_model,
        })
    }
}

/// Parse metrics from all epochs in a metrics directory (train/ or valid/)
fn parse_epoch_metrics(metrics_dir: &Path) -> Vec<EpochMetrics> {
    let mut metrics = Vec::new();

    if !metrics_dir.exists() {
        return metrics;
    }

    let entries: Vec<_> = std::fs::read_dir(metrics_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let epoch: usize = name.strip_prefix("epoch-")?.parse().ok()?;
            Some((epoch, e.path()))
        })
        .collect();

    for (epoch, path) in entries {
        let mut epoch_metrics = EpochMetrics {
            epoch,
            ..Default::default()
        };

        // Parse Loss.log - get average of all entries
        if let Some(values) = parse_metric_log(&path.join("Loss.log")) {
            epoch_metrics.loss = Some(average(&values));
        }

        // Parse Perplexity.log - get last value (cumulative)
        if let Some(values) = parse_metric_log(&path.join("Perplexity.log")) {
            epoch_metrics.perplexity = values.last().copied();
        }

        // Parse Accuracy.log - get average
        if let Some(values) = parse_metric_log(&path.join("Accuracy.log")) {
            epoch_metrics.accuracy = Some(average(&values));
        }

        metrics.push(epoch_metrics);
    }

    metrics.sort_by_key(|m| m.epoch);
    metrics
}

/// Parse a metric log file (CSV format: value,step)
fn parse_metric_log(path: &Path) -> Option<Vec<f64>> {
    let content = std::fs::read_to_string(path).ok()?;
    let values: Vec<f64> = content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            parts.first()?.parse().ok()
        })
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

// UI helpers
const BOX_WIDTH: usize = 66;

fn box_top(title: &str) {
    let padding = BOX_WIDTH - 5 - title.len();
    println!("╭─ {title} {}╮", "─".repeat(padding));
}

fn box_bottom() {
    println!("╰{}╯", "─".repeat(BOX_WIDTH - 2));
}

fn box_row(label: &str, value: impl std::fmt::Display) {
    let content = format!(" {label:<15} {value}");
    let padding = BOX_WIDTH - 2 - content.chars().count();
    println!("│{content}{}│", " ".repeat(padding));
}

fn progress_bar(current: usize, total: usize, width: usize) -> String {
    let pct = if total > 0 {
        (current as f64 / total as f64).min(1.0)
    } else {
        0.0
    };
    let filled = (pct * width as f64) as usize;
    format!(
        "[{}{}] {:.0}%",
        "█".repeat(filled),
        "░".repeat(width - filled),
        pct * 100.0
    )
}

fn print_metrics_table(title: &str, metrics: &[EpochMetrics]) {
    box_top(title);
    println!(
        "│ {:>5} │ {:>10} │ {:>12} │ {:>10} │",
        "Epoch", "Loss", "Perplexity", "Accuracy"
    );
    println!("├───────┼────────────┼──────────────┼────────────┤");
    for m in metrics {
        let loss = m.loss.map(|v| format!("{v:.4}")).unwrap_or("-".into());
        let ppl = m
            .perplexity
            .map(|v| format!("{v:.2}"))
            .unwrap_or("-".into());
        let acc = m.accuracy.map(|v| format!("{v:.2}%")).unwrap_or("-".into());
        println!(
            "│ {:>5} │ {:>10} │ {:>12} │ {:>10} │",
            m.epoch, loss, ppl, acc
        );
    }
    box_bottom();
    println!();
}

/// Pretty-print the artifact info
pub fn print_info(info: &ArtifactInfo, verbose: bool) {
    let config = &info.config;
    let ttt = &config.ttt_config;

    println!("╭{}╮", "─".repeat(BOX_WIDTH - 2));
    println!("│{:^width$}│", "Training Run Info", width = BOX_WIDTH - 2);
    println!("╰{}╯", "─".repeat(BOX_WIDTH - 2));
    println!();

    box_top("Model Configuration");
    box_row("Layer Type:", format!("{:?}", ttt.layer_type));
    box_row("Hidden Size:", ttt.hidden_size);
    box_row("Num Layers:", ttt.num_hidden_layers);
    box_row("Num Heads:", ttt.num_heads);
    box_row("Vocab Size:", ttt.vocab_size);
    box_row("Max Seq Len:", ttt.max_seq_len);
    box_row("Position Enc:", format!("{:?}", ttt.pos_encoding));
    box_row("Mini-batch:", ttt.mini_batch_size);
    box_row("TTT Base LR:", ttt.base_lr);
    box_bottom();
    println!();

    box_top("Training Configuration");
    box_row("Batch Size:", config.batch_size);
    box_row("Num Epochs:", config.num_epochs);
    box_row("Learning Rate:", format!("{:.2e}", config.learning_rate));
    box_row("Warmup Steps:", config.warmup_steps);
    box_row("Grad Accum:", config.grad_accumulation);
    box_row("Train Samples:", config.train_samples);
    box_row("Test Samples:", config.test_samples);
    box_bottom();
    println!();

    box_top("Training Progress");
    box_row(
        "Progress:",
        progress_bar(info.total_epochs, config.num_epochs, 30),
    );
    box_row(
        "Epochs:",
        format!("{} / {}", info.total_epochs, config.num_epochs),
    );
    if let Some(cp) = info.latest_checkpoint {
        box_row("Latest Ckpt:", format!("epoch {cp}"));
    }
    box_row(
        "Final Model:",
        if info.has_final_model { "yes" } else { "no" },
    );
    box_bottom();
    println!();

    if let Some(m) = info.train_metrics.last() {
        box_top(&format!("Latest Training Metrics (epoch {})", m.epoch));
        if let Some(v) = m.loss {
            box_row("Loss:", format!("{v:.4}"));
        }
        if let Some(v) = m.perplexity {
            box_row("Perplexity:", format!("{v:.2}"));
        }
        if let Some(v) = m.accuracy {
            box_row("Accuracy:", format!("{v:.2}%"));
        }
        box_bottom();
        println!();
    }

    if let Some(m) = info.valid_metrics.last() {
        box_top(&format!("Latest Validation Metrics (epoch {})", m.epoch));
        if let Some(v) = m.loss {
            box_row("Loss:", format!("{v:.4}"));
        }
        if let Some(v) = m.perplexity {
            box_row("Perplexity:", format!("{v:.2}"));
        }
        if let Some(v) = m.accuracy {
            box_row("Accuracy:", format!("{v:.2}%"));
        }
        box_bottom();
        println!();
    }

    if verbose && !info.train_metrics.is_empty() {
        print_metrics_table("Training Metrics by Epoch", &info.train_metrics);
    }

    if verbose && !info.valid_metrics.is_empty() {
        print_metrics_table("Validation Metrics by Epoch", &info.valid_metrics);
    }
}
