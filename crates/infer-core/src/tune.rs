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
pub const MEASURE_GEN_TOKENS: usize = 256;

/// Timed passes per candidate. Three gives a median that survives one outlier.
pub const MEASURE_REPS: usize = 5;

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
    /// Emitted after stage one, once the cross-check set is known.
    Planned {
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

/// Stage one of the sweep: the drafter family and its depth, measured at the
/// reference batch shape. Context stays whatever `fit::auto_ctx` chose.
///
/// The winner of this stage is then crossed with flash attention and the batch
/// shape by [`cross_plan`]. That two-stage shape is deliberate: crossing every
/// drafter with every batch setting would be 14 x 4 loads, most of which cannot
/// beat the winner.
pub fn quick_plan(caps: &TuneCapabilities) -> Vec<TuneCandidate> {
    let mut plan = Vec::new();
    plan.push(candidate(caps, Drafter::None, 0, true, 2048, 512));
    if caps.has_mtp {
        for depth in [1, 2, 3, 4, 6] {
            plan.push(candidate(caps, Drafter::Mtp, depth, true, 2048, 512));
        }
    }
    if caps.draft_model.is_some() {
        for depth in [1, 2, 3, 4, 6] {
            plan.push(candidate(caps, Drafter::Draft, depth, true, 2048, 512));
        }
    }
    plan
}

/// Stage two: the stage-one winner with flash attention and batch alternatives.
///
/// The drafter and depth are carried over unchanged, so the comparison answers
/// "what is the best batch shape *for this drafter*", not "what is the best
/// batch shape for a different configuration".
pub fn cross_plan(best: &TuneCandidate) -> Vec<TuneCandidate> {
    let name = best
        .label
        .split(" fa=")
        .next()
        .unwrap_or("draft")
        .to_string();
    let mut out = Vec::new();
    for (fa, n_batch, n_ubatch) in [
        (false, best.n_batch, best.n_ubatch),
        (true, best.n_batch, 2048),
        (true, best.n_batch, 1024),
        (true, 4096, 1024),
    ] {
        if fa == best.flash_attn && n_batch == best.n_batch && n_ubatch == best.n_ubatch {
            continue;
        }
        out.push(TuneCandidate {
            label: format!(
                "{name} fa={} ubatch={n_ubatch} batch={n_batch}",
                if fa { "on" } else { "off" }
            ),
            n_ctx: best.n_ctx,
            n_batch,
            n_ubatch,
            flash_attn: fa,
            drafter: best.drafter,
            draft_n_max: best.draft_n_max,
        });
    }
    out
}

/// Deep mode: the top two drafter configurations crossed with the full batch
/// matrix. Bounded on purpose — the full 11 x 12 Cartesian product is over a
/// hundred loads for differences that are usually below the noise floor.
pub fn deep_cross_plan(winners: &[TuneCandidate]) -> Vec<TuneCandidate> {
    let mut out = Vec::new();
    for base in winners.iter().take(2) {
        let name = base
            .label
            .split(" fa=")
            .next()
            .unwrap_or("draft")
            .to_string();
        for fa in [false, true] {
            for n_ubatch in [512, 1024, 2048] {
                for n_batch in [2048, 4096] {
                    if n_batch == 4096 && n_ubatch > 1024 {
                        continue;
                    }
                    if fa == base.flash_attn && n_batch == base.n_batch && n_ubatch == base.n_ubatch
                    {
                        continue;
                    }
                    out.push(TuneCandidate {
                        label: format!(
                            "{name} fa={} ubatch={n_ubatch} batch={n_batch}",
                            if fa { "on" } else { "off" }
                        ),
                        n_ctx: base.n_ctx,
                        n_batch,
                        n_ubatch,
                        flash_attn: fa,
                        drafter: base.drafter,
                        draft_n_max: base.draft_n_max,
                    });
                }
            }
        }
    }
    out
}

fn candidate(
    caps: &TuneCapabilities,
    drafter: Drafter,
    depth: u32,
    fa: bool,
    n_batch: u32,
    n_ubatch: u32,
) -> TuneCandidate {
    let name = match drafter {
        Drafter::None => "draft=none".to_string(),
        Drafter::Mtp => format!("MTP-head d={depth}"),
        Drafter::Draft => format!("{} d={depth}", caps.draft_label),
    };
    TuneCandidate {
        label: format!(
            "{name} fa={} ubatch={n_ubatch} batch={n_batch}",
            if fa { "on" } else { "off" }
        ),
        n_ctx: caps.n_ctx.max(2048),
        n_batch,
        n_ubatch,
        flash_attn: fa,
        drafter,
        draft_n_max: depth,
    }
}

/// Run one two-stage sweep. `base` carries the model path and the settings
/// shared by every candidate; only the fields in [`TuneCandidate`] are
/// overridden.
pub fn run_sweep(
    base: &LoadConfig,
    plan: &[TuneCandidate],
    deep: bool,
    cancel: &AtomicBool,
    mut progress: impl FnMut(TuneProgress),
) -> Result<SweepOutcome> {
    let mut measurements = Vec::new();
    let mut skipped = Vec::new();
    let mut index = 0usize;

    progress(TuneProgress::Started { total: plan.len() });
    if !run_stage(
        base,
        plan,
        cancel,
        &mut index,
        &mut measurements,
        &mut skipped,
        &mut progress,
    ) {
        return Ok(SweepOutcome {
            measurements,
            skipped,
        });
    }

    // Only a candidate that actually drafted may win. A configuration that
    // asked for MTP but produced no draft tokens is not a slow drafter; it is
    // a drafter that did not load, and cross-checking its batch shape would
    // just produce four more mislabeled rows.
    let best = measurements
        .iter()
        .filter(|m| m.candidate.drafter == Drafter::None || m.draft_acceptance.is_some())
        .max_by(|a, b| a.tokens_per_second.total_cmp(&b.tokens_per_second))
        .map(|m| m.candidate.clone());

    if let Some(best) = best {
        let staged = cross_plan(&best);
        if !staged.is_empty() {
            progress(TuneProgress::Planned {
                total: plan.len() + staged.len(),
            });
            let _ = run_stage(
                base,
                &staged,
                cancel,
                &mut index,
                &mut measurements,
                &mut skipped,
                &mut progress,
            );
        }

        if deep && !cancel.load(Ordering::Relaxed) {
            // Two distinct drafter/depth winners, not two batch shapes of the
            // same drafter. The point of deep mode is to test whether a second
            // drafter could win under a different batch shape.
            let mut picked: Vec<TuneCandidate> = Vec::new();
            let mut sorted: Vec<&TuneMeasurement> = measurements
                .iter()
                .filter(|m| m.candidate.drafter == Drafter::None || m.draft_acceptance.is_some())
                .collect();
            sorted.sort_by(|a, b| b.tokens_per_second.total_cmp(&a.tokens_per_second));
            for m in sorted {
                let already = picked.iter().any(|p| {
                    p.drafter == m.candidate.drafter && p.draft_n_max == m.candidate.draft_n_max
                });
                if !already {
                    picked.push(m.candidate.clone());
                }
                if picked.len() == 2 {
                    break;
                }
            }
            if !picked.is_empty() {
                let mut deep_plan = deep_cross_plan(&picked);
                deep_plan.retain(|c| {
                    !measurements.iter().any(|m| {
                        m.candidate.drafter == c.drafter
                            && m.candidate.draft_n_max == c.draft_n_max
                            && m.candidate.flash_attn == c.flash_attn
                            && m.candidate.n_batch == c.n_batch
                            && m.candidate.n_ubatch == c.n_ubatch
                    })
                });
                if !deep_plan.is_empty() {
                    progress(TuneProgress::Planned {
                        total: index + deep_plan.len(),
                    });
                    let _ = run_stage(
                        base,
                        &deep_plan,
                        cancel,
                        &mut index,
                        &mut measurements,
                        &mut skipped,
                        &mut progress,
                    );
                }
            }
        }
    }

    measurements.sort_by(|a, b| b.tokens_per_second.total_cmp(&a.tokens_per_second));
    Ok(SweepOutcome {
        measurements,
        skipped,
    })
}

/// Measure one stage. Returns false when cancellation stopped it.
#[allow(clippy::too_many_arguments)]
fn run_stage(
    base: &LoadConfig,
    plan: &[TuneCandidate],
    cancel: &AtomicBool,
    index: &mut usize,
    measurements: &mut Vec<TuneMeasurement>,
    skipped: &mut Vec<SkippedCandidate>,
    progress: &mut impl FnMut(TuneProgress),
) -> bool {
    for candidate in plan {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let current = *index;
        *index += 1;
        progress(TuneProgress::CandidateStarted {
            index: current,
            label: candidate.label.clone(),
        });
        match measure_candidate(base, candidate, cancel) {
            Ok(m) => {
                progress(TuneProgress::CandidateDone {
                    index: current,
                    tokens_per_second: m.tokens_per_second,
                });
                measurements.push(m);
            }
            Err(e) => {
                let reason = e.to_string();
                progress(TuneProgress::CandidateSkipped {
                    index: current,
                    reason: reason.clone(),
                });
                skipped.push(SkippedCandidate {
                    candidate: candidate.clone(),
                    reason,
                });
            }
        }
    }
    true
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
    if candidate.drafter != Drafter::None && runs.iter().all(|r| r.draft_acceptance.is_none()) {
        return Err(InferError::Decode(
            "the requested drafter produced no draft tokens; it may not have loaded".into(),
        ));
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
    fn a_model_with_no_drafter_starts_from_one_baseline() {
        let plan = quick_plan(&caps(false, false));
        assert_eq!(plan.len(), 1);
        assert!(plan.iter().all(|p| p.drafter == Drafter::None));
    }

    #[test]
    fn the_cross_stage_keeps_the_winner_drafter_and_depth() {
        let c = caps(true, true);
        let best = candidate(&c, Drafter::Mtp, 3, true, 2048, 512);
        let cross = cross_plan(&best);
        assert_eq!(cross.len(), 4);
        assert!(cross
            .iter()
            .all(|p| p.drafter == Drafter::Mtp && p.draft_n_max == 3 && p.n_ctx == best.n_ctx));
        assert!(cross.iter().any(|p| !p.flash_attn));
        assert!(cross.iter().any(|p| p.n_ubatch == 2048));
        assert!(cross.iter().any(|p| p.n_batch == 4096));
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
    fn the_deep_plan_crosses_two_winners_without_the_full_product() {
        let c = caps(true, true);
        let a = candidate(&c, Drafter::Mtp, 3, true, 2048, 512);
        let b = candidate(&c, Drafter::Draft, 2, true, 2048, 512);
        let deep = deep_cross_plan(&[a, b]);
        // 2 winners x (2 FA x 5 safe batch shapes - 1 already measured) = 18.
        assert_eq!(deep.len(), 18);
        assert!(deep
            .iter()
            .any(|p| p.drafter == Drafter::Mtp && p.draft_n_max == 3));
        assert!(deep
            .iter()
            .any(|p| p.drafter == Drafter::Draft && p.draft_n_max == 2));
        assert!(deep.iter().all(|p| p.n_ctx == c.n_ctx));
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
        let outcome =
            run_sweep(&base, &plan, false, &cancel, |_| {}).expect("cancel short-circuits");
        assert!(outcome.measurements.is_empty());
        assert!(outcome.skipped.is_empty());
    }
}
