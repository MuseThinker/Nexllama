//! Tune jobs: a measured sweep runs on its own thread, progress is read by
//! handle.
//!
//! The sweep needs exclusive use of the hardware, so the caller unloads the
//! resident instance before starting one — the job holds its own weights, and
//! two copies of one checkpoint on one GPU is how OOM happens. While a job is
//! running for a model, loading that model again is refused: the weights are
//! already resident inside the sweep, and the answer arrives in minutes.
//!
//! The registry is a plain `std` mutex because the sweep thread is not async
//! and never touches the daemon's model lock: it only writes its own status.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use infer_core::tune::{self, SweepOutcome, TuneCandidate, TuneProgress};
use infer_core::LoadConfig;

/// One tune run, from start to report.
pub struct TuneJob {
    pub id: String,
    pub model_id: String,
    pub model_path: PathBuf,
    pub base: LoadConfig,
    pub depth: String,
    pub started_at: Instant,
    pub cancel: Arc<AtomicBool>,
    status: Mutex<TuneStatus>,
}

/// Where a job is. `Running` carries the candidate being measured right now,
/// so a client can show progress without polling the engine.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TuneStatus {
    Running {
        index: usize,
        total: usize,
        label: String,
    },
    Done {
        outcome: SweepOutcome,
        elapsed_secs: f64,
    },
    Cancelled {
        partial: SweepOutcome,
        elapsed_secs: f64,
    },
    Failed {
        reason: String,
        elapsed_secs: f64,
    },
}

/// Every started job, by id. Old jobs stay listed until the daemon restarts:
/// the report is small, and it is the artefact the user compares against.
#[derive(Default)]
pub struct TuneRegistry {
    jobs: Mutex<HashMap<String, Arc<TuneJob>>>,
}

impl TuneRegistry {
    /// Start a sweep for one model on a worker thread.
    pub fn start(
        &self,
        model_id: String,
        base: LoadConfig,
        plan: Vec<TuneCandidate>,
        depth: &str,
    ) -> Arc<TuneJob> {
        let id = format!(
            "tune-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        let model_path = base.model_path.clone();
        let job = Arc::new(TuneJob {
            id: id.clone(),
            model_id,
            model_path,
            base,
            depth: depth.to_string(),
            started_at: Instant::now(),
            cancel: Arc::new(AtomicBool::new(false)),
            status: Mutex::new(TuneStatus::Running {
                index: 0,
                total: plan.len(),
                label: String::new(),
            }),
        });

        let worker = job.clone();
        let total = plan.len();
        std::thread::spawn(move || {
            let started = worker.started_at;
            let mut current_total = total;
            let outcome = tune::run_sweep(
                &worker.base,
                &plan,
                worker.depth == "deep",
                &worker.cancel,
                |ev| {
                    if let Ok(mut s) = worker.status.lock() {
                        match ev {
                            TuneProgress::Started { total } => {
                                current_total = total;
                                *s = TuneStatus::Running {
                                    index: 0,
                                    total,
                                    label: String::new(),
                                };
                            }
                            TuneProgress::Planned { total } => {
                                current_total = total;
                                if let TuneStatus::Running { index, label, .. } = &*s {
                                    *s = TuneStatus::Running {
                                        index: *index,
                                        total,
                                        label: label.clone(),
                                    };
                                }
                            }
                            TuneProgress::CandidateStarted { index, label } => {
                                *s = TuneStatus::Running {
                                    index,
                                    total: current_total,
                                    label,
                                };
                            }
                            TuneProgress::CandidateDone { .. }
                            | TuneProgress::CandidateSkipped { .. } => {}
                        }
                    }
                },
            );
            let elapsed = started.elapsed().as_secs_f64();
            let mut s = worker.status.lock().expect("status mutex");
            *s = match outcome {
                Ok(o) if worker.cancel.load(Ordering::Relaxed) => TuneStatus::Cancelled {
                    partial: o,
                    elapsed_secs: elapsed,
                },
                Ok(o) => TuneStatus::Done {
                    outcome: o,
                    elapsed_secs: elapsed,
                },
                Err(e) => TuneStatus::Failed {
                    reason: e.to_string(),
                    elapsed_secs: elapsed,
                },
            };
        });

        self.jobs
            .lock()
            .expect("registry mutex")
            .insert(id, job.clone());
        job
    }

    pub fn get(&self, id: &str) -> Option<Arc<TuneJob>> {
        self.jobs.lock().expect("registry mutex").get(id).cloned()
    }

    pub fn list(&self) -> Vec<Arc<TuneJob>> {
        let mut jobs: Vec<Arc<TuneJob>> = self
            .jobs
            .lock()
            .expect("registry mutex")
            .values()
            .cloned()
            .collect();
        jobs.sort_by_key(|a| std::cmp::Reverse(a.started_at));
        jobs
    }

    /// The model whose sweep currently owns the GPU, if any.
    ///
    /// A sweep measures one model's decode loop on a card it has to itself, so
    /// the load path refuses every load while one runs: a second resident model
    /// would take VRAM back and turn the remaining candidates into a
    /// measurement of contention.
    pub fn running_model(&self) -> Option<String> {
        self.jobs
            .lock()
            .expect("registry mutex")
            .values()
            .find(|j| {
                matches!(
                    &*j.status.lock().expect("status mutex"),
                    TuneStatus::Running { .. }
                )
            })
            .map(|j| j.model_id.clone())
    }

    /// Whether a sweep is running against this checkpoint file, whichever id
    /// it was started under.
    pub fn running_for_path(&self, path: &std::path::Path) -> bool {
        self.jobs.lock().expect("registry mutex").values().any(|j| {
            j.model_path.as_path() == path
                && matches!(
                    &*j.status.lock().expect("status mutex"),
                    TuneStatus::Running { .. }
                )
        })
    }
}

impl TuneJob {
    /// The job as the API returns it.
    pub fn to_json(&self) -> serde_json::Value {
        let status = self.status.lock().expect("status mutex");
        serde_json::json!({
            "id": self.id,
            "model": self.model_id,
            "model_path": self.model_path.display().to_string(),
            "depth": self.depth,
            "elapsed_secs": self.started_at.elapsed().as_secs_f64(),
            "cancel_requested": self.cancel.load(Ordering::Relaxed),
            "status": &*status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_running_job_blocks_its_model_and_its_path() {
        let reg = TuneRegistry::default();
        // start() spawns a thread that will fail on the missing file; the
        // status is Running until that thread finishes, which is the window
        // the load path must refuse in. Both queries must agree.
        let caps = tune::TuneCapabilities {
            n_ctx: 4096,
            has_mtp: false,
            draft_model: None,
            draft_label: "draft".into(),
            kv: infer_core::engine::KvCacheType::Q8_0,
        };
        let job = reg.start(
            "m".into(),
            LoadConfig::new("does-not-exist.gguf"),
            tune::quick_plan(&caps),
            "quick",
        );
        // Race-free enough: the thread has to load a missing file first, but a
        // fast machine could finish. Poll briefly and accept either outcome —
        // what must never happen is running_for_path disagreeing with the
        // status the same call observes.
        for _ in 0..50 {
            let running = matches!(&*job.status.lock().unwrap(), TuneStatus::Running { .. });
            assert_eq!(
                reg.running_for_path(std::path::Path::new("does-not-exist.gguf")),
                running
            );
            assert_eq!(reg.running_model().as_deref(), running.then_some("m"));
            if !running {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(matches!(
            &*job.status.lock().unwrap(),
            TuneStatus::Failed { .. } | TuneStatus::Done { .. } | TuneStatus::Cancelled { .. }
        ));
    }
}
