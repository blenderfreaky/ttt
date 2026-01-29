//! Export training metrics to CSV for plotting.

use std::path::{Path, PathBuf};

use clap::ValueEnum;

/// Metric types that can be exported.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum MetricType {
    Loss,
    Perplexity,
    Accuracy,
    LearningRate,
}

impl MetricType {
    fn filename(&self) -> &'static str {
        match self {
            MetricType::Loss => "Loss.log",
            MetricType::Perplexity => "Perplexity.log",
            MetricType::Accuracy => "Accuracy.log",
            MetricType::LearningRate => "LearningRate.log",
        }
    }

    fn column_name(&self) -> &'static str {
        match self {
            MetricType::Loss => "loss",
            MetricType::Perplexity => "perplexity",
            MetricType::Accuracy => "accuracy",
            MetricType::LearningRate => "learning_rate",
        }
    }
}

/// Configuration for the metrics export.
pub struct ExportConfig {
    pub metrics: Vec<MetricType>,
    pub include_train: bool,
    pub include_valid: bool,
    pub target_points: Option<usize>,
    pub window: Option<usize>,
}

/// A single row in the output CSV.
#[derive(Debug)]
struct MetricRow {
    experiment: String,
    epoch: usize,
    step: usize,
    global_step: usize,
    values: Vec<Option<f64>>,
}

/// Parse a metric log file, returning (step, value) pairs.
/// Steps are 1-indexed based on line number (since the step column in log files isn't reliable).
fn parse_metric_log_with_steps(path: &Path) -> Option<Vec<(usize, f64)>> {
    let content = std::fs::read_to_string(path).ok()?;
    let values: Vec<(usize, f64)> = content
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let parts: Vec<&str> = line.split(',').collect();
            let value = parts.first()?.parse().ok()?;
            Some((i + 1, value)) // 1-indexed step based on line number
        })
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Downsample data to approximately `target` points using bucket averaging.
fn downsample_buckets(values: &[(usize, f64)], target: usize) -> Vec<(usize, f64)> {
    if values.len() <= target || target == 0 {
        return values.to_vec();
    }

    let bucket_size = values.len() / target;
    let mut result = Vec::with_capacity(target);

    for chunk in values.chunks(bucket_size) {
        if chunk.is_empty() {
            continue;
        }
        let avg_step = chunk.iter().map(|(s, _)| *s).sum::<usize>() / chunk.len();
        let avg_value = chunk.iter().map(|(_, v)| *v).sum::<f64>() / chunk.len() as f64;
        result.push((avg_step, avg_value));
    }

    result
}

/// Apply rolling average smoothing.
fn apply_rolling_average(values: &[(usize, f64)], window: usize) -> Vec<(usize, f64)> {
    if window <= 1 || values.is_empty() {
        return values.to_vec();
    }

    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(values.len());
        let slice = &values[start..end];
        let avg = slice.iter().map(|(_, v)| *v).sum::<f64>() / slice.len() as f64;
        result.push((values[i].0, avg));
    }

    result
}

/// Expand glob patterns to actual paths.
fn expand_globs(patterns: &[String]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();

    for pattern in patterns {
        let matches: Vec<_> = glob::glob(pattern)?.filter_map(|r| r.ok()).collect();

        if matches.is_empty() {
            // Treat as a literal path if no glob matches
            let path = PathBuf::from(pattern);
            if path.exists() {
                paths.push(path);
            } else {
                eprintln!("Warning: no matches for pattern '{pattern}'");
            }
        } else {
            paths.extend(matches);
        }
    }

    // Filter to only directories
    paths.retain(|p| p.is_dir());
    paths.sort();

    Ok(paths)
}

/// Get experiment name from a path (the directory name).
fn experiment_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Collect all epoch directories from a metrics dir (train/ or valid/).
fn collect_epochs(metrics_dir: &Path) -> Vec<(usize, PathBuf)> {
    let mut epochs = Vec::new();

    if !metrics_dir.exists() {
        return epochs;
    }

    if let Ok(entries) = std::fs::read_dir(metrics_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(epoch_str) = name.strip_prefix("epoch-") {
                if let Ok(epoch) = epoch_str.parse::<usize>() {
                    epochs.push((epoch, entry.path()));
                }
            }
        }
    }

    epochs.sort_by_key(|(e, _)| *e);
    epochs
}

/// Collect metrics from one artifact directory for one split.
fn collect_split_metrics(dir: &Path, split: &str, config: &ExportConfig) -> Vec<MetricRow> {
    let experiment = experiment_name(dir);
    let metrics_dir = dir.join(split);
    let epochs = collect_epochs(&metrics_dir);

    if epochs.is_empty() {
        return Vec::new();
    }

    let mut rows = Vec::new();
    let mut global_step_offset = 0;

    for (epoch, epoch_path) in epochs {
        // Load all requested metrics for this epoch
        let mut metric_data: Vec<Option<Vec<(usize, f64)>>> = Vec::new();

        for metric_type in &config.metrics {
            let log_path = epoch_path.join(metric_type.filename());
            let data = parse_metric_log_with_steps(&log_path);

            // Apply smoothing and downsampling
            let data = data.map(|mut d| {
                if let Some(window) = config.window {
                    d = apply_rolling_average(&d, window);
                }
                if let Some(target) = config.target_points {
                    d = downsample_buckets(&d, target);
                }
                d
            });

            metric_data.push(data);
        }

        // Find the maximum number of steps across all metrics
        let max_steps = metric_data
            .iter()
            .filter_map(|d| d.as_ref().map(|v| v.len()))
            .max()
            .unwrap_or(0);

        if max_steps == 0 {
            continue;
        }

        // Build index maps for each metric (step -> value)
        let index_maps: Vec<std::collections::HashMap<usize, f64>> = metric_data
            .iter()
            .map(|d| {
                d.as_ref()
                    .map(|v| v.iter().cloned().collect())
                    .unwrap_or_default()
            })
            .collect();

        // Get all unique steps across all metrics, sorted
        let mut all_steps: Vec<usize> = index_maps.iter().flat_map(|m| m.keys().cloned()).collect();
        all_steps.sort();
        all_steps.dedup();

        let max_step = all_steps.last().copied();

        // Create rows for each step
        for step in all_steps {
            let values: Vec<Option<f64>> =
                index_maps.iter().map(|m| m.get(&step).cloned()).collect();

            // Skip rows where all values are None
            if values.iter().all(|v| v.is_none()) {
                continue;
            }

            rows.push(MetricRow {
                experiment: experiment.clone(),
                epoch,
                step,
                global_step: global_step_offset + step,
                values,
            });
        }

        // Update global step offset for next epoch
        if let Some(max_step) = max_step {
            global_step_offset += max_step;
        }
    }

    rows
}

/// Write CSV output.
fn write_csv(
    rows: &[MetricRow],
    output: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(output)?;

    // Write header
    let metric_columns: Vec<&str> = config.metrics.iter().map(|m| m.column_name()).collect();
    writeln!(
        file,
        "experiment,epoch,step,global_step,{}",
        metric_columns.join(",")
    )?;

    // Write data rows
    for row in rows {
        let values_str: Vec<String> = row
            .values
            .iter()
            .map(|v| v.map(|x| format!("{x}")).unwrap_or_default())
            .collect();

        writeln!(
            file,
            "{},{},{},{},{}",
            row.experiment,
            row.epoch,
            row.step,
            row.global_step,
            values_str.join(",")
        )?;
    }

    Ok(())
}

/// Derive output path for a split from the base output path.
fn output_path_for_split(base: &str, split: &str) -> PathBuf {
    let path = Path::new(base);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy())
        .unwrap_or_default();

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if ext.is_empty() {
            parent.join(format!("{stem}_{split}"))
        } else {
            parent.join(format!("{stem}_{split}.{ext}"))
        }
    } else if ext.is_empty() {
        PathBuf::from(format!("{stem}_{split}"))
    } else {
        PathBuf::from(format!("{stem}_{split}.{ext}"))
    }
}

/// Main entry point for exporting metrics.
pub fn export_metrics(
    dirs: Vec<String>,
    output: &str,
    config: ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let paths = expand_globs(&dirs)?;

    if paths.is_empty() {
        return Err("No valid artifact directories found".into());
    }

    println!("Exporting metrics from {} directories", paths.len());

    let mut train_rows = Vec::new();
    let mut valid_rows = Vec::new();

    for path in &paths {
        let name = experiment_name(path);

        if config.include_train {
            let rows = collect_split_metrics(path, "train", &config);
            if !rows.is_empty() {
                println!("  {name}: {} train data points", rows.len());
                train_rows.extend(rows);
            }
        }

        if config.include_valid {
            let rows = collect_split_metrics(path, "valid", &config);
            if !rows.is_empty() {
                println!("  {name}: {} valid data points", rows.len());
                valid_rows.extend(rows);
            }
        }
    }

    if train_rows.is_empty() && valid_rows.is_empty() {
        return Err("No metrics data found in any directory".into());
    }

    // Write separate files for train and valid
    if !train_rows.is_empty() {
        let train_path = output_path_for_split(output, "train");
        write_csv(&train_rows, &train_path, &config)?;
        println!(
            "Wrote {} rows to {}",
            train_rows.len(),
            train_path.display()
        );
    }

    if !valid_rows.is_empty() {
        let valid_path = output_path_for_split(output, "valid");
        write_csv(&valid_rows, &valid_path, &config)?;
        println!(
            "Wrote {} rows to {}",
            valid_rows.len(),
            valid_path.display()
        );
    }

    Ok(())
}
