//! Scheduling and parallel run management.

use crate::config::{FailurePolicy, HarnessConfig, RunConfig};
use crate::runner::{new_activity_tracker, ProgressUpdate, RunError, RunResult, Runner};
use crate::state::{StateError, StateManager};
use crate::vram::{VramEstimate, VramEstimator};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
        let rust_log = config.harness.rust_log.clone();
        let runs = config.runs.clone();
        Self {
            config,
            runs,
            estimator: VramEstimator::default(),
            runner: Runner::new(ttt_binary, state_manager, rust_log),
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

        // Shared "settle until" timestamp — all watchdogs pause until this time
        let settle_until = Arc::new(AtomicU64::new(0));

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
                        let pid = handle.pid;
                        pb.set_message(format!("{name}: starting (PID {pid})"));
                        pb.enable_steady_tick(std::time::Duration::from_millis(100));

                        running.insert(name.clone(), vram);
                        used_vram += vram;

                        // Bump settle_until so all watchdogs pause while GPU settles
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        settle_until.store(
                            now + self.config.harness.settle_grace_secs,
                            Ordering::Relaxed,
                        );

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

                        // Activity tracker for idle detection
                        let activity = new_activity_tracker();

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

                        // Spawn watchdog for hang/idle detection
                        if self.config.harness.hang_timeout_secs.is_some()
                            || self.config.harness.idle_timeout_secs.is_some()
                        {
                            let hang_timeout = self.config.harness.hang_timeout_secs;
                            let idle_timeout = self.config.harness.idle_timeout_secs;
                            let settle = settle_until.clone();
                            let watch_activity = activity.clone();
                            let watch_name = name.clone();
                            // Subscribe to progress for hang detection
                            let mut watch_progress = progress_tx.subscribe();
                            tokio::spawn(async move {
                                watchdog(
                                    &watch_name,
                                    pid,
                                    hang_timeout,
                                    idle_timeout,
                                    settle,
                                    watch_activity,
                                    &mut watch_progress,
                                )
                                .await;
                            });
                        }

                        let tx = tx.clone();
                        let state_file = self.config.harness.state_file.clone();
                        let rust_log = self.config.harness.rust_log.clone();

                        tokio::spawn(async move {
                            let sm = StateManager::new(&state_file);
                            let runner = Runner::new("ttt", sm, rust_log);
                            let result =
                                runner.wait(handle, Some(progress_tx), Some(activity)).await;
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

/// Watchdog that kills a process if it hangs or goes idle.
///
/// - `hang_timeout`: seconds with no progress update (only active after the first update)
/// - `idle_timeout`: seconds with no stdout/stderr activity at all
/// - `settle_until`: shared timestamp — watchdog pauses until this time passes
async fn watchdog(
    name: &str,
    pid: u32,
    hang_timeout: Option<u64>,
    idle_timeout: Option<u64>,
    settle_until: Arc<AtomicU64>,
    last_activity: Arc<AtomicU64>,
    progress_rx: &mut watch::Receiver<ProgressUpdate>,
) {
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

    let mut got_first_progress = false;
    let mut last_progress_time = 0u64;

    loop {
        tokio::time::sleep(POLL_INTERVAL).await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Respect settle grace period
        if now < settle_until.load(Ordering::Relaxed) {
            continue;
        }

        // Check if process is still alive
        let alive = unsafe { libc::kill(pid as i32, 0) } == 0;
        if !alive {
            return;
        }

        // Check for new progress updates (non-blocking)
        if progress_rx.has_changed().unwrap_or(false) {
            let _ = progress_rx.borrow_and_update();
            if !got_first_progress {
                got_first_progress = true;
            }
            last_progress_time = now;
        }

        // Hang detection: no progress update for too long (only after first update)
        if let Some(timeout) = hang_timeout
            && got_first_progress
            && now - last_progress_time > timeout
        {
            tracing::error!(
                "{name}: no progress update for {timeout}s, killing (PID {pid})"
            );
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
            return;
        }

        // Idle detection: no stdout/stderr activity at all
        if let Some(timeout) = idle_timeout {
            let last = last_activity.load(Ordering::Relaxed);
            if now - last > timeout {
                tracing::error!(
                    "{name}: no output activity for {timeout}s, killing (PID {pid})"
                );
                unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                return;
            }
        }
    }
}
