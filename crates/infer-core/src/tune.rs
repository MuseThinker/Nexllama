//! Measured tuning: rank load configurations by real decode speed.
//!
//! Context size is *not* a candidate. It is derived from the device and the
//! checkpoint once, by `fit::auto_ctx`, and every candidate runs at that same
//! size. What gets measured is what actually changes speed: the drafter and its
//! depth, flash attention, and the batch shape.
//!
//! Each candidate is loaded through the ordinary [`Engine`] path, so MTP and
//! DFlash/EAGLE proposals are exercised exactly as they are in a real chat. The
//! sweep owns its model instance and runs on its own thread; the daemon unloads
//! the resident instance first, so VRAM is never held twice.
//!
//! Measurement protocol (fixed, so two runs are comparable):
//!
//! - one fixed prompt, padded to roughly [`PREFILL_TARGET`] tokens;
//! - one untimed warm-up pass, then [`MEASURE_REPS`] timed passes of
//!   prefill plus [`MEASURE_GEN_TOKENS`] generated tokens, greedy sampling;
//! - the median across passes, because a single pass absorbs host noise;
//! - draft acceptance is reported alongside speed, since a drafter that is
//!   wrong is a second forward pass with nothing to show for it.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use crate::engine::{Engine, FlashAttn, LoadConfig};
use crate::error::{InferError, Result};
use crate::sampler::SamplerParams;
use crate::types::{ChatMsg, GenStats};

/// Generated tokens per timed pass.
pub const MEASURE_GEN_TOKENS: usize = 128;

/// Timed passes per candidate. Three gives a median that survives one outlier.
pub const MEASURE_REPS: usize = 3;

/// The prompt is padded to roughly this many tokens by repeating [`PROMPT_SEED`].
pub const PREFILL_TARGET: usize = 256;

/// Tokens generated during the untimed warm-up.
const WARMUP_TOKENS: usize = 16;

const PROMPT_SEED: &str = "The history of the sea is long and full of tales. \
Please continue, at length, with what came next: ";

/// Which drafter a candidate uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Drafter {
    None,
    Mtp,
    Draft,
}

/// What the model can draft with, read from the loaded instance before the
/// resident copy is unloaded.
#[derive(Debug, Clone)]
pub struct TuneCapabilities {
    /// The context every candidate runs at. Chosen by `fit::auto_ctx`, never
    /// by this module.
    pub n_ctx: u32,
    /// The checkpoint ships an MTP/NextN head.
    pub has_mtp: bool,
    /// A side-mounted DFlash/EAGLE checkpoint, when one was loaded.
    pub draft_model: Option<String>,
    /// Human-readable drafter family for report labels, e.g. "DFlash2".
    pub draft_label: String,
}

/// One configuration the sweep measures. Every field is concrete: a candidate
/// never says "auto", because the point is to find out what the machine does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TuneCandidate {
    /// A human-readable name, unique within a plan.
    pub label: String,
    /// Fixed for the whole sweep; repeated so the report stands alone.
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Flash attention forced on or off.
    pub flash_attn: bool,
    pub drafter: Drafter,
    /// Draft tokens per round. 0 for [`Drafter::None`].
    pub draft_n_max: u32,
}

/// One timed pass over the fixed workload.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RunStat {
    /// Generated tokens per second, decode only, prefill excluded.
    pub tokens_per_second: f64,
    /// Prefill plus the first sample, in milliseconds.
    pub first_token_ms: f64,
    /// Prefill throughput, tokens per second.
    pub prefill_tokens_per_second: f64,
    /// Draft tokens accepted divided by proposed, when the candidate drafted.
    pub draft_acceptance: Option<f64>,
}

/// The result of measuring one candidate.
#[derive(Debug, Clone, Serialize)]
pub struct TuneMeasurement {
    pub candidate: TuneCandidate,
    /// Median decode speed across the timed passes.
    pub tokens_per_second: f64,
    /// Median time to first token across the timed passes.
    pub first_token_ms: f64,
    /// Median prefill throughput across the timed passes.
    pub prefill_tokens_per_second: f64,
    /// Median draft acceptance across the passes that drafted.
    pub draft_acceptance: Option<f64>,
    /// Every timed pass, so a client can see the spread.
    pub runs: Vec<RunStat>,
}

/// A candidate that could not be measured, with the reason.
#[derive(Debug, Clone, Serialize)]
pub struct SkippedCandidate {
    pub candidate: TuneCandidate,
    pub reason: String,
}

/// Everything one sweep produced.
#[derive(Debug, Clone, Serialize)]
pub struct SweepOutcome {
    /// Measured candidates, fastest first.
    pub measurements: Vec<TuneMeasurement>,
    pub skipped: Vec<SkippedCandidate>,
}

/// Progress events for a running sweep.
#[derive(Debug, Clone, Serialize)]
pub enum TuneProgress {
    Started {
        total: usize,
    },
    CandidateStarted {
        index: usize,
        label: String,
    },
    CandidateDone {
        index: usize,
        tokens_per_second: f64,
    },
    CandidateSkipped {
        index: usize,
        reason: String,
    },
}

/// The matrix a sweep measures.
///
/// The drafter family and depth are the high-value knobs; flash attention and
/// the batch shape follow at the fixed context. A model with neither an MTP head
/// nor a side checkpoint gets only the non-drafting candidates.
pub fn quick_plan(caps: &TuneCapabilities) -> Vec<TuneCandidate> {
    let n_ctx = caps.n_ctx.max(2048);
    let mut plan = Vec::new();
    let push = |plan: &mut Vec<TuneCandidate>, drafter, depth, fa, n_batch, n_ubatch| {
        let name = match drafter {
            Drafter::None => "draft=none".to_string(),
            Drafter::Mtp => format!("MTP-head d={depth}"),
            Drafter::Draft => format!("{} d={depth}", caps.draft_label),
        };
        plan.push(TuneCandidate {
            label: format!(
                "{name} fa={} ubatch={n_ubatch} batch={n_batch}",
                if fa { "on" } else { "off" }
            ),
            n_ctx,
            n_batch,
            n_ubatch,
            flash_attn: fa,
            drafter,
            draft_n_max: depth,
        });
    };

    // The reference: no drafter, flash attention on, the default batch.
    push(&mut plan, Drafter::None, 0, true, 2048, 512);
    if caps.has_mtp {
        for depth in [1, 2, 3, 4, 6] {
            push(&mut plan, Drafter::Mtp, depth, true, 2048, 512);
        }
    }
    if caps.draft_model.is_some() {
        for depth in [1, 2, 3, 4, 6] {
            push(&mut plan, Drafter::Draft, depth, true, 2048, 512);
        }
    }
    // Non-drafting variants, so the winner is not judged against only one
    // flash-attention or batch setting.
    push(&mut plan, Drafter::None, 0, false, 2048, 512);
    push(&mut plan, Drafter::None, 0, true, 2048, 2048);
    push(&mut plan, Drafter::None, 0, true, 4096, 1024);
    plan
}

/// Run one sweep. `base` carries the model path and the settings shared by every
/// candidate; only the fields in [`TuneCandidate`] are overridden.
pub fn run_sweep(
    base: &LoadConfig,
    plan: &[TuneCandidate],
    cancel: &AtomicBool,
    mut progress: impl FnMut(TuneProgress),
) -> Result<SweepOutcome> {
    let mut measurements = Vec::new();
    let mut skipped = Vec::new();

    progress(TuneProgress::Started { total: plan.len() });
    for (index, candidate) in plan.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        progress(TuneProgress::CandidateStarted {
            index,
            label: candidate.label.clone(),
        });
        match measure_candidate(base, candidate, cancel) {
            Ok(m) => {
                progress(TuneProgress::CandidateDone {
                    index,
                    tokens_per_second: m.tokens_per_second,
                });
                measurements.push(m);
            }
            Err(e) => {
                let reason = e.to_string();
                progress(TuneProgress::CandidateSkipped {
                    index,
                    reason: reason.clone(),
                });
                skipped.push(SkippedCandidate {
                    candidate: candidate.clone(),
                    reason,
                });
            }
        }
    }

    measurements.sort_by(|a, b| b.tokens_per_second.total_cmp(&a.tokens_per_second));
    Ok(SweepOutcome {
        measurements,
        skipped,
    })
}

fn measure_candidate(
    base: &LoadConfig,
    candidate: &TuneCandidate,
    cancel: &AtomicBool,
) -> Result<TuneMeasurement> {
    let mut cfg = base.clone();
    cfg.n_ctx = candidate.n_ctx;
    cfg.n_batch = candidate.n_batch;
    cfg.n_ubatch = candidate.n_ubatch;
    cfg.flash_attn = if candidate.flash_attn {
        FlashAttn::On
    } else {
        FlashAttn::Off
    };
    cfg.load_mtp = candidate.drafter == Drafter::Mtp;
    cfg.mtp_n_max = if cfg.load_mtp {
        candidate.draft_n_max
    } else {
        0
    };
    cfg.draft_model = if candidate.drafter == Drafter::Draft {
        base.draft_model.clone()
    } else {
        None
    };
    cfg.draft_n_max = if candidate.drafter == Drafter::None {
        0
    } else {
        candidate.draft_n_max
    };
    // One request must not reuse another pass's state while timing.
    cfg.prefix_cache = false;
    cfg.adaptive_draft = false;

    let engine = Engine::load(cfg, |_| {})?;
    let prompt = prompt();
    let params = |max_tokens: usize| SamplerParams {
        temperature: 0.0,
        top_k: 1,
        seed: 1,
        max_tokens,
        // A no-drafter candidate must really be plain decoding. Leaving this
        // on also enables the n-gram drafter, which made the baseline look
        // faster than every real drafter and reported a fake acceptance rate.
        speculative: candidate.drafter != Drafter::None,
        // N-gram drafting is not part of this measurement even when a model
        // drafter is loaded; only the model's own proposals should count.
        ngram_m: 1,
        thinking: Some(false),
        ..SamplerParams::default()
    };

    // Warm-up: one-time driver, allocator, and graph costs must not land on a
    // timed pass.
    let _ = engine.generate(&prompt, &params(WARMUP_TOKENS), |_| true)?;

    let mut runs = Vec::with_capacity(MEASURE_REPS);
    for _ in 0..MEASURE_REPS {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let stats = engine.generate(&prompt, &params(MEASURE_GEN_TOKENS), |_| true)?;
        runs.push(run_stat(&stats));
    }
    if runs.is_empty() {
        return Err(InferError::Decode("cancelled before any timed pass".into()));
    }

    let med = |values: Vec<f64>| median(values);
    Ok(TuneMeasurement {
        candidate: candidate.clone(),
        tokens_per_second: med(runs.iter().map(|r| r.tokens_per_second).collect()),
        first_token_ms: med(runs.iter().map(|r| r.first_token_ms).collect()),
        prefill_tokens_per_second: med(runs.iter().map(|r| r.prefill_tokens_per_second).collect()),
        draft_acceptance: median_option(runs.iter().filter_map(|r| r.draft_acceptance).collect()),
        runs,
    })
}

fn run_stat(stats: &GenStats) -> RunStat {
    RunStat {
        tokens_per_second: stats.tokens_per_second(),
        first_token_ms: stats.prompt_ms + stats.first_token_ms,
        prefill_tokens_per_second: stats.prompt_tokens_per_second(),
        draft_acceptance: stats.draft_acceptance(),
    }
}

/// A fixed, prose prompt. The exact words do not matter; repeating the seed to
/// about [`PREFILL_TARGET`] tokens makes prefill cost comparable across runs.
fn prompt() -> Vec<ChatMsg> {
    let seed = PROMPT_SEED.repeat((PREFILL_TARGET / 10).max(1));
    vec![ChatMsg::new("user", seed)]
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn median_option(values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(median(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(has_mtp: bool, draft: bool) -> TuneCapabilities {
        TuneCapabilities {
            n_ctx: 32768,
            has_mtp,
            draft_model: draft.then(|| "dfash.gguf".into()),
            draft_label: "DFlash2".into(),
        }
    }

    #[test]
    fn the_plan_never_varies_context() {
        for c in [
            caps(false, false),
            caps(true, false),
            caps(false, true),
            caps(true, true),
        ] {
            let plan = quick_plan(&c);
            assert!(plan.iter().all(|p| p.n_ctx == c.n_ctx));
        }
    }

    #[test]
    fn a_model_with_no_drafter_gets_only_the_non_drafting_matrix() {
        let plan = quick_plan(&caps(false, false));
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().all(|p| p.drafter == Drafter::None));
    }

    #[test]
    fn mtp_and_side_drafters_get_their_depths() {
        let plan = quick_plan(&caps(true, true));
        let mtp: Vec<u32> = plan
            .iter()
            .filter(|p| p.drafter == Drafter::Mtp)
            .map(|p| p.draft_n_max)
            .collect();
        let side: Vec<u32> = plan
            .iter()
            .filter(|p| p.drafter == Drafter::Draft)
            .map(|p| p.draft_n_max)
            .collect();
        assert_eq!(mtp, vec![1, 2, 3, 4, 6]);
        assert_eq!(side, vec![1, 2, 3, 4, 6]);
    }

    #[test]
    fn labels_are_unique_and_concrete() {
        let plan = quick_plan(&caps(true, true));
        let mut labels: Vec<&str> = plan.iter().map(|c| c.label.as_str()).collect();
        labels.sort();
        let n = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), n, "labels must be unique");
        assert!(plan.iter().all(|c| !c.label.contains("auto")));
    }

    #[test]
    fn median_survives_one_outlier() {
        assert_eq!(median(vec![5.0]), 5.0);
        assert_eq!(median(vec![1.0, 2.0, 100.0]), 2.0);
        assert_eq!(median(vec![3.0, 1.0]), 2.0);
        assert_eq!(median(vec![]), 0.0);
        assert_eq!(median(vec![30.0, 10.0, 20.0]), 20.0);
    }

    #[test]
    fn a_cancelled_sweep_returns_an_empty_outcome() {
        let cancel = AtomicBool::new(true);
        let base = LoadConfig::new("does-not-matter.gguf");
        let plan = quick_plan(&caps(false, false));
        let outcome = run_sweep(&base, &plan, &cancel, |_| {}).expect("cancel short-circuits");
        assert!(outcome.measurements.is_empty());
        assert!(outcome.skipped.is_empty());
    }
}
