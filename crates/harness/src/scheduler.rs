//! Scheduling and parallel run management.

use crate::config::{FailurePolicy, HarnessConfig, RunConfig};
use crate::runner::{ProgressUpdate, RunError, RunResult, Runner};
use crate::state::{StateError, StateManager};
use crate::vram::{VramEstimate, VramEstimator};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

/// Scheduler for managing parallel training runs.
pub struct Scheduler {
    config: HarnessConfig,
    runs: Vec<RunConfig>,
    estimator: VramEstimator,
    runner: Runner,
}

/// Information about a scheduled run.
#[derive(Debug, Clone)]
pub struct ScheduledRun {
    pub name: String,
    pub config: RunConfig,
    pub vram_estimate: VramEstimate,
    pub resume_epoch: Option<usize>,
}

/// Result with VRAM info for immediate rescheduling.
struct CompletionResult {
    result: RunResult,
    vram_gb: f64,
}

impl Scheduler {
    /// Create a new scheduler.
    #[must_use]
    pub fn new(config: HarnessConfig, ttt_binary: String, state_manager: StateManager) -> Self {
        let runs = config.runs.clone();
        Self {
            config,
            runs,
            estimator: VramEstimator::default(),
            runner: Runner::new(ttt_binary, state_manager),
        }
    }

    /// Find next run that fits in available VRAM and isn't already running.
    fn find_next_run(
        &self,
        running: &HashMap<String, f64>,
        free_vram_gb: f64,
    ) -> Option<ScheduledRun> {
        let state = self.runner.state_manager().load().ok()?;

        for run in &self.runs {
            // Skip if already running
            if running.contains_key(&run.name) {
                continue;
            }

            // Check if run can be started
            let run_state = state.runs.get(&run.name);
            let can_start = run_state.is_none_or(|s| s.can_start(self.config.harness.max_retries));
            if !can_start {
                continue;
            }

            // Check if it fits
            let estimate = self.estimator.estimate(run);
            if estimate.total_gb() <= free_vram_gb {
                return Some(ScheduledRun {
                    name: run.name.clone(),
                    config: run.clone(),
                    vram_estimate: estimate,
                    resume_epoch: run_state.and_then(|s| s.checkpoint_epoch),
                });
            }
        }

        None
    }

    /// Check if all runs are finished.
    fn all_finished(&self) -> bool {
        let Ok(state) = self.runner.state_manager().load() else {
            return false;
        };
        self.runs.iter().all(|r| {
            state
                .runs
                .get(&r.name)
                .is_some_and(|s| s.is_finished(self.config.harness.max_retries))
        })
    }

    /// Run the scheduler in dry-run mode (no actual execution).
    pub fn dry_run(&self) -> Result<DryRunResult, SchedulerError> {
        let state = self.runner.state_manager().load()?;
        let available = self.config.usable_vram_gb();

        let mut scheduled = Vec::new();
        let mut too_large = Vec::new();
        let mut used_vram = 0.0;

        for run in &self.runs {
            let run_state = state.runs.get(&run.name);
            let can_start = run_state.is_none_or(|s| s.can_start(self.config.harness.max_retries));
            if !can_start {
                continue;
            }

            let estimate = self.estimator.estimate(run);
            let vram = estimate.total_gb();

            if vram > available {
                too_large.push(run.name.clone());
            } else if used_vram + vram <= available {
                scheduled.push(ScheduledRun {
                    name: run.name.clone(),
                    config: run.clone(),
                    vram_estimate: estimate,
                    resume_epoch: run_state.and_then(|s| s.checkpoint_epoch),
                });
                used_vram += vram;
            }
        }

        Ok(DryRunResult {
            scheduled,
            used_vram_gb: used_vram,
            usable_vram_gb: available,
            too_large,
        })
    }

    /// Run the main scheduling loop with immediate rescheduling.
    pub async fn run(&self) -> Result<SchedulerResult, SchedulerError> {
        let state_manager = self.runner.state_manager();

        // Initialize state for all runs
        state_manager.initialize_runs(&self.runs)?;

        // Recover any crashed runs
        let crashed = state_manager.recover_crashed_runs()?;
        if !crashed.is_empty() {
            tracing::info!("Recovered {} crashed runs: {:?}", crashed.len(), crashed);
        }

        let total_vram = self.config.usable_vram_gb();
        let multi = MultiProgress::new();

        // Track running processes: name -> vram_gb
        let mut running: HashMap<String, f64> = HashMap::new();
        let mut used_vram = 0.0;

        // Channel for completion notifications
        let (tx, mut rx) = mpsc::unbounded_channel::<CompletionResult>();

        let mut completed = 0;
        let mut failed = 0;
        let total = self.runs.len();

        loop {
            // Fill available VRAM with runs
            while let Some(scheduled) = self.find_next_run(&running, total_vram - used_vram) {
                let vram = scheduled.vram_estimate.total_gb();

                let pb = multi.add(ProgressBar::new_spinner());
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {msg}")
                        .unwrap(),
                );
                pb.set_message(format!("{}: starting...", scheduled.name));

                match self.runner.spawn(&scheduled.config, scheduled.resume_epoch) {
                    Ok(handle) => {
                        let name = handle.name.clone();
                        pb.set_message(format!("{name}: starting (PID {})", handle.pid));
                        pb.enable_steady_tick(std::time::Duration::from_millis(100));

                        running.insert(name.clone(), vram);
                        used_vram += vram;

                        tracing::info!(
                            "Started {} ({:.2} GB, total: {:.2}/{:.2} GB)",
                            name,
                            vram,
                            used_vram,
                            total_vram
                        );

                        // Progress channel for this run
                        let (progress_tx, mut progress_rx) = watch::channel(ProgressUpdate::default());
                        let progress_tx = Arc::new(progress_tx);

                        // Task to update progress bar
                        let pb_clone = pb.clone();
                        let name_clone = name.clone();
                        tokio::spawn(async move {
                            while progress_rx.changed().await.is_ok() {
                                let p = progress_rx.borrow();
                                pb_clone.set_message(format!(
                                    "{}: epoch {}/{} [{}/{}]",
                                    name_clone, p.epoch, p.epoch_total, p.items_processed, p.items_total
                                ));
                            }
                        });

                        let tx = tx.clone();
                        let state_file = self.config.harness.state_file.clone();

                        tokio::spawn(async move {
                            let sm = StateManager::new(&state_file);
                            let runner = Runner::new("ttt", sm);
                            let result = runner.wait(handle, Some(progress_tx)).await;
                            pb.finish_with_message(format!(
                                "{}: {}",
                                name,
                                if result.success { "completed" } else { "failed" }
                            ));
                            let _ = tx.send(CompletionResult { result, vram_gb: vram });
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to spawn {}: {}", scheduled.name, e);
                        pb.finish_with_message(format!("{}: spawn failed", scheduled.name));

                        match self.config.harness.on_failure {
                            FailurePolicy::Abort => return Err(SchedulerError::Run(e)),
                            FailurePolicy::Skip => {
                                state_manager.mark_skipped(&scheduled.name, &e.to_string())?;
                                failed += 1;
                            }
                            FailurePolicy::Retry => {
                                // Will retry on next iteration
                            }
                        }
                    }
                }
            }

            // Check if done
            if running.is_empty() {
                if self.all_finished() {
                    break;
                }
                // Nothing running but not finished - shouldn't happen normally
                tracing::warn!("No runs active but not all finished");
                break;
            }

            // Wait for next completion
            let Some(completion) = rx.recv().await else {
                break;
            };

            // Free the VRAM
            running.remove(&completion.result.name);
            used_vram -= completion.vram_gb;

            if completion.result.success {
                completed += 1;
                tracing::info!(
                    "{} completed (freed {:.2} GB)",
                    completion.result.name,
                    completion.vram_gb
                );
            } else {
                tracing::error!(
                    "{} failed: {}",
                    completion.result.name,
                    completion.result.error.as_deref().unwrap_or("unknown")
                );

                match self.config.harness.on_failure {
                    FailurePolicy::Abort => {
                        return Err(SchedulerError::RunFailed(
                            completion.result.name,
                            completion.result.error.unwrap_or_default(),
                        ));
                    }
                    FailurePolicy::Skip | FailurePolicy::Retry => {
                        let state = state_manager.load()?;
                        if let Some(run_state) = state.runs.get(&completion.result.name)
                            && run_state.retry_count >= self.config.harness.max_retries
                        {
                            if self.config.harness.on_failure == FailurePolicy::Skip {
                                state_manager
                                    .mark_skipped(&completion.result.name, "max retries exceeded")?;
                            }
                            failed += 1;
                        }
                        // Otherwise will retry on next loop iteration
                    }
                }
            }
            // Loop continues - immediately try to fill freed VRAM
        }

        Ok(SchedulerResult {
            total,
            completed,
            failed,
            skipped: total - completed - failed,
        })
    }

    /// Get the state manager.
    #[must_use]
    pub fn state_manager(&self) -> &StateManager {
        self.runner.state_manager()
    }
}

/// Result of a dry run.
#[derive(Debug)]
pub struct DryRunResult {
    /// Runs that would start immediately.
    pub scheduled: Vec<ScheduledRun>,
    /// Total VRAM used by scheduled runs.
    pub used_vram_gb: f64,
    /// Usable VRAM after margin.
    pub usable_vram_gb: f64,
    /// Runs too large to ever fit.
    pub too_large: Vec<String>,
}

/// Result of a scheduler run.
#[derive(Debug)]
pub struct SchedulerResult {
    /// Total number of runs.
    pub total: usize,
    /// Number of completed runs.
    pub completed: usize,
    /// Number of failed runs.
    pub failed: usize,
    /// Number of skipped runs.
    pub skipped: usize,
}

/// Errors that can occur during scheduling.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("state error: {0}")]
    State(#[from] StateError),
    #[error("run error: {0}")]
    Run(#[from] RunError),
    #[error("run {0} failed: {1}")]
    RunFailed(String, String),
}
