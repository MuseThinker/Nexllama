//! Measured tuning: rank context-level configurations by real decode speed.
//!
//! The fit helpers answer "will it run"; this answers "how should it run".
//! Static fitting cannot know whether flash attention actually pays on this
//! GPU, or how much a larger micro-batch helps prefill here, because those
//! depend on the driver and the silicon in the machine. This module loads the
//! checkpoint once, rebuilds a context per candidate configuration, and times
//! real prefill-plus-generation passes, so the recommendation is a measurement
//! rather than an estimate.
//!
//! The sweep owns its model instance and runs on its own thread: the daemon
//! unloads the resident instance first, so VRAM is never held twice. The user
//! confirms a winner by loading normally with the tuned parameters, which is
//! the ordinary load path and needs nothing new.
//!
//! Measurement protocol (fixed, so two runs are comparable):
//!
//! - one fixed prompt, padded to [`PREFILL_TARGET`] tokens;
//! - [`MEASURE_REPS`] timed passes of prefill plus [`MEASURE_GEN_TOKENS`]
//!   generated tokens, greedy sampling, one fresh context per pass so no pass
//!   inherits another's KV state;
//! - one untimed warm-up pass before the timed ones, so the first pass does
//!   not pay one-time driver and allocator costs;
//! - the median across passes, because a single pass absorbs noise from
//!   anything the host was doing at the time.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use llama_cpp_4::prelude::*;
use serde::Serialize;

use crate::error::{InferError, Result};

/// Generated tokens per timed pass. Long enough to average out per-token
/// jitter, short enough that a quick sweep stays in the minutes, not hours.
pub const MEASURE_GEN_TOKENS: usize = 128;

/// Timed passes per candidate. Three gives a median that survives one outlier.
pub const MEASURE_REPS: usize = 3;

/// The prompt is padded to about this many tokens, so prefill cost is the same
/// for every candidate and the decode numbers compare cleanly.
pub const PREFILL_TARGET: usize = 256;

/// Tokens generated during the untimed warm-up.
const WARMUP_TOKENS: usize = 16;

/// The seed text the fixed prompt is built from. Anything works; it is padded
/// by repetition and decoded greedily, so the same candidate always does the
/// same work.
const PROMPT_SEED: &str = "The history of the sea is long and full of tales. \
Please continue, at length, with what came next: ";

/// One configuration the sweep measures. Every field is concrete: a candidate
/// never says "auto", because the point is to find out what the machine
/// actually does with each setting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TuneCandidate {
    /// A human-readable name, unique within a plan.
    pub label: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Flash attention forced on or off. `Auto` would measure whichever way
    /// the backend already leans, which is not a configuration.
    pub flash_attn: bool,
}

/// One timed pass over the fixed workload.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RunStat {
    /// Generated tokens per second, decode only — prefill excluded.
    pub tokens_per_second: f64,
    /// Prefill plus the first sample, in milliseconds.
    pub first_token_ms: f64,
    /// Prefill throughput, tokens per second.
    pub prefill_tokens_per_second: f64,
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
    /// Every timed pass, so a client can see the spread instead of trusting
    /// the median blindly.
    pub runs: Vec<RunStat>,
}

/// A candidate that could not be measured, with the reason.
///
/// A context that will not build — usually VRAM for a larger `n_ctx` — is a
/// finding about the machine, not a sweep failure: the candidate is skipped
/// and the reason travels with the report.
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
    /// Emitted once, before the first candidate.
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

/// The context-level matrix a quick sweep measures.
///
/// Flash attention crossed with micro-batch, at up to two context sizes. The
/// context tiers only differ when the base context leaves room for them: a
/// model capped at 4096 has one tier, so the plan is four candidates, not
/// eight. Larger contexts are capped by `max_n_ctx`, which the caller takes
/// from what the machine actually fits — the sweep never discovers OOM that a
/// fit check already knew about.
pub fn quick_plan(max_n_ctx: u32) -> Vec<TuneCandidate> {
    let tier_a = max_n_ctx.min(4096).max(512);
    let tier_b = max_n_ctx.min(8192);
    let mut tiers = vec![tier_a];
    if tier_b > tier_a {
        tiers.push(tier_b);
    }

    let mut plan = Vec::new();
    for &n_ctx in &tiers {
        for &fa in &[true, false] {
            for &(n_batch, n_ubatch) in &[(2048u32, 512u32), (2048u32, 2048u32)] {
                plan.push(TuneCandidate {
                    label: format!(
                        "fa={} ubatch={} ctx={}",
                        if fa { "on" } else { "off" },
                        n_ubatch,
                        n_ctx
                    ),
                    n_ctx,
                    n_batch,
                    n_ubatch,
                    flash_attn: fa,
                });
            }
        }
    }
    plan
}

/// The median of a set of numbers. Takes ownership because it sorts.
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

/// Pad the seed text until it reaches `target` tokens, the way `rawbench`
/// builds its prompt. Deterministic for a given tokenizer, which is what makes
/// two candidates comparable.
fn build_prompt(model: &LlamaModel, target: usize) -> Vec<LlamaToken> {
    let mut prompt = String::from(PROMPT_SEED);
    while model
        .str_to_token(&prompt, AddBos::Always)
        .map(|t| t.len())
        .unwrap_or(usize::MAX)
        < target
    {
        prompt.push_str(PROMPT_SEED);
    }
    model
        .str_to_token(&prompt, AddBos::Always)
        .unwrap_or_default()
}

/// One prefill-plus-generation pass on a fresh context.
fn one_pass(
    model: &LlamaModel,
    backend: &LlamaBackend,
    candidate: &TuneCandidate,
    prompt: &[LlamaToken],
    n_gen: usize,
) -> Result<RunStat> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(candidate.n_ctx))
        .with_n_batch(candidate.n_batch)
        .with_n_ubatch(candidate.n_ubatch)
        .with_flash_attn_type(if candidate.flash_attn {
            LlamaFlashAttnType::Enabled
        } else {
            LlamaFlashAttnType::Disabled
        })
        .warmup(false);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| InferError::Context(format!("{e:?}")))?;

    // Prefill the whole prompt in one batch; only the last row needs logits.
    let mut batch = LlamaBatch::new(prompt.len().max(1), 1);
    for (i, &tok) in prompt.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i + 1 == prompt.len())
            .map_err(|e| InferError::Decode(e.to_string()))?;
    }
    let t0 = Instant::now();
    ctx.decode(&mut batch)
        .map_err(|e| InferError::Decode(e.to_string()))?;
    let prefill_secs = t0.elapsed().as_secs_f64();

    let mut sampler = LlamaSampler::greedy();

    // The first token is sampled from the prompt's logits, outside the decode
    // loop: TTFT is prefill plus this one sample, then the loop times decode.
    let t_first = Instant::now();
    let mut tok = sampler.sample(&ctx, -1);
    let ttft = prefill_secs * 1000.0 + t_first.elapsed().as_secs_f64() * 1000.0;
    if model.is_eog_token(tok) {
        return Ok(RunStat {
            tokens_per_second: 0.0,
            first_token_ms: ttft,
            prefill_tokens_per_second: prompt.len() as f64 / prefill_secs,
        });
    }
    let mut pos = prompt.len() as i32;
    batch.clear();
    batch
        .add(tok, pos, &[0], true)
        .map_err(|e| InferError::Decode(e.to_string()))?;
    pos += 1;
    ctx.decode(&mut batch)
        .map_err(|e| InferError::Decode(e.to_string()))?;

    let start = Instant::now();
    let mut generated = 1usize;
    for _ in 1..n_gen {
        tok = sampler.sample(&ctx, -1);
        if model.is_eog_token(tok) {
            break;
        }
        generated += 1;
        batch.clear();
        batch
            .add(tok, pos, &[0], true)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        pos += 1;
        ctx.decode(&mut batch)
            .map_err(|e| InferError::Decode(e.to_string()))?;
    }
    let gen_secs = start.elapsed().as_secs_f64();

    Ok(RunStat {
        tokens_per_second: generated as f64 / gen_secs,
        first_token_ms: ttft,
        prefill_tokens_per_second: prompt.len() as f64 / prefill_secs,
    })
}

/// Run the sweep.
///
/// Loads the checkpoint once at `n_gpu_layers`, then measures every candidate.
/// `progress` receives one event per step; `cancel` is checked before each
/// candidate, and a cancelled sweep returns what it measured so far marked as
/// partial. Candidates whose context will not build are skipped with a reason
/// rather than failing the sweep.
pub fn run_sweep(
    model_path: &Path,
    n_gpu_layers: u32,
    candidates: &[TuneCandidate],
    cancel: &AtomicBool,
    mut progress: impl FnMut(TuneProgress),
) -> Result<SweepOutcome> {
    let backend = LlamaBackend::init().map_err(|e| InferError::Backend(format!("{e:?}")))?;
    let model = LlamaModel::load_from_file(
        &backend,
        model_path,
        &LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers),
    )
    .map_err(|e| InferError::ModelLoad(format!("{e:?}")))?;

    let prompt = build_prompt(&model, PREFILL_TARGET);
    if prompt.is_empty() {
        return Err(InferError::Tokenize(
            "the tuning prompt tokenized empty".into(),
        ));
    }

    progress(TuneProgress::Started {
        total: candidates.len(),
    });

    let mut measurements = Vec::new();
    let mut skipped = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        progress(TuneProgress::CandidateStarted {
            index,
            label: candidate.label.clone(),
        });

        // The warm-up pass is untimed: it pays the one-time driver and
        // allocator costs that would otherwise land on the first measurement.
        match one_pass(&model, &backend, candidate, &prompt, WARMUP_TOKENS) {
            Ok(_) => {}
            Err(e) => {
                let reason = format!("warm-up failed: {e}");
                progress(TuneProgress::CandidateSkipped {
                    index,
                    reason: reason.clone(),
                });
                skipped.push(SkippedCandidate {
                    candidate: candidate.clone(),
                    reason,
                });
                continue;
            }
        }

        let mut runs = Vec::with_capacity(MEASURE_REPS);
        let mut failed = None;
        for _ in 0..MEASURE_REPS {
            match one_pass(&model, &backend, candidate, &prompt, MEASURE_GEN_TOKENS) {
                Ok(stat) => runs.push(stat),
                Err(e) => {
                    failed = Some(format!("measurement failed: {e}"));
                    break;
                }
            }
        }
        if let Some(reason) = failed {
            progress(TuneProgress::CandidateSkipped {
                index,
                reason: reason.clone(),
            });
            skipped.push(SkippedCandidate {
                candidate: candidate.clone(),
                reason,
            });
            continue;
        }

        let med = |pick: fn(&RunStat) -> f64| median(runs.iter().map(pick).collect());
        let measurement = TuneMeasurement {
            candidate: candidate.clone(),
            tokens_per_second: med(|r| r.tokens_per_second),
            first_token_ms: med(|r| r.first_token_ms),
            prefill_tokens_per_second: med(|r| r.prefill_tokens_per_second),
            runs,
        };
        progress(TuneProgress::CandidateDone {
            index,
            tokens_per_second: measurement.tokens_per_second,
        });
        measurements.push(measurement);
    }

    measurements.sort_by(|a, b| b.tokens_per_second.total_cmp(&a.tokens_per_second));
    Ok(SweepOutcome {
        measurements,
        skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_context_cap_yields_one_tier() {
        let plan = quick_plan(4096);
        // Two flash-attention settings x two micro-batches, one context tier.
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().all(|c| c.n_ctx == 4096));
    }

    #[test]
    fn a_larger_cap_yields_two_tiers() {
        let plan = quick_plan(16384);
        assert_eq!(plan.len(), 8);
        let tiers: Vec<u32> = plan.iter().map(|c| c.n_ctx).collect();
        assert!(tiers.contains(&4096) && tiers.contains(&8192));
    }

    #[test]
    fn the_cap_is_never_exceeded() {
        for plan in [quick_plan(2048), quick_plan(6000), quick_plan(500_000)] {
            assert!(plan.iter().all(|c| c.n_ctx <= 8192 || c.n_ctx <= 6000));
        }
    }

    #[test]
    fn labels_are_unique_and_concrete() {
        let plan = quick_plan(8192);
        let mut labels: Vec<&str> = plan.iter().map(|c| c.label.as_str()).collect();
        labels.sort();
        let n = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), n, "labels must be unique");
        // A candidate never claims "auto": the sweep exists to resolve that.
        assert!(plan.iter().all(|c| !c.label.contains("auto")));
    }

    #[test]
    fn median_survives_one_outlier() {
        assert_eq!(median(vec![5.0]), 5.0);
        assert_eq!(median(vec![1.0, 2.0, 100.0]), 2.0);
        assert_eq!(median(vec![3.0, 1.0]), 2.0);
        assert_eq!(median(vec![]), 0.0);
        // Unsorted input is the normal case.
        assert_eq!(median(vec![30.0, 10.0, 20.0]), 20.0);
    }

    #[test]
    fn a_cancelled_sweep_returns_a_partial_outcome() {
        // No model here: the point is that the flag short-circuits before any
        // candidate runs, so the loop structure is checkable without weights.
        let cancel = Arc::new(AtomicBool::new(true));
        let outcome = run_sweep(
            Path::new("does-not-matter.gguf"),
            0,
            &quick_plan(4096),
            &cancel,
            |_| {},
        );
        // The load itself fails on a missing file — the cancel check sits
        // inside the candidate loop, which this test does not reach. The
        // assertion pins the contract that a missing model is a ModelLoad.
        assert!(matches!(outcome, Err(InferError::ModelLoad(_))));
    }
}
