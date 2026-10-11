//! Whether a checkpoint fits this machine, and what to change when it does not.
//!
//! Every number here comes from llama.cpp rather than from arithmetic of ours.
//! `get_device_memory_data` and `fit_params` are the same helpers `llama-cli`
//! uses to decide how many layers it can offload, and they account for the
//! things a hand-rolled estimate gets wrong: the embedding and output tensors
//! are far larger than an average layer, a MoE model carries `n_expert` copies
//! of some of them, and the KV cache and compute buffers are charged alongside
//! the weights.
//!
//! Estimating "free VRAM divided by average layer size" instead is how an
//! auto-tuner ends up confidently offloading too much and being killed for it
//! part-way through a load.

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::fit::{fit_params, get_device_memory_data, FitParams, FitParamsError};
use llama_cpp_4::model::params::LlamaModelParams;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hwprofile::{probe, HardwareProfile};

/// What Smart Fit concludes about one checkpoint on this machine.
///
/// Advisory in every field: nothing here changes a load on its own. The point
/// is that being wrong about memory costs an out-of-memory kill, which is a
/// worse outcome than being slow, so the decision is left to whoever asked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitReport {
    /// Whether the projected allocation fits in what is free right now.
    pub fits: bool,
    /// Bytes the checkpoint's weights and buffers are projected to need.
    pub needed_mb: u64,
    /// Bytes free on the device that constrains the load.
    ///
    /// For the measured path this is the tightest device (the minimum free
    /// across the devices the checkpoint would land on) — the actual ceiling
    /// when "fits" requires every piece to fit. For the size path it is the
    /// largest single device, because that path asks "is it worth starting"
    /// rather than "will this load".
    pub available_mb: u64,
    /// How many layers llama.cpp would offload, if it were left to decide.
    pub suggested_gpu_layers: Option<u32>,
    /// Context length llama.cpp would settle for.
    pub suggested_ctx: Option<u32>,
    /// A smaller quantisation that would fit, when the weights themselves are
    /// the problem. `None` when layers or context are enough to fix it.
    pub suggested_quant: Option<String>,
    /// Why, in words a person can act on.
    pub reason: String,
    /// The machine this was measured against.
    pub hardware: HardwareProfile,
}

impl FitReport {
    /// A report for a checkpoint that could not be measured at all.
    ///
    /// Failing to measure is not the same as failing to fit, and conflating the
    /// two would have a caller refuse a model for no stated reason.
    pub fn unknown(reason: impl Into<String>, hardware: HardwareProfile) -> Self {
        Self {
            // Deliberately optimistic: an unmeasured model is allowed to try.
            fits: true,
            needed_mb: 0,
            available_mb: hardware.largest_free_mb(),
            suggested_gpu_layers: None,
            suggested_ctx: None,
            suggested_quant: None,
            reason: reason.into(),
            hardware,
        }
    }
}

/// Offloading "everything", the way llama.cpp's own tooling asks.
///
/// Higher than any real layer count; llama.cpp clamps it to the model.
const ALL_LAYERS: u32 = 99;

/// Context to project against when the caller does not name one.
const DEFAULT_CTX: u32 = 4096;

/// Smallest context worth recommending. Below this a model stops being useful
/// for anything but the shortest exchange.
const MIN_CTX: u32 = 2048;

/// A report for a size with no local file yet.
///
/// Weights dominate what a checkpoint needs, and they are the only thing known
/// about a file before it has been fetched, so this compares against the
/// largest thing the machine can hold instead of projecting a full allocation.
/// It answers "is this worth starting", not "will this load".
pub fn from_size(bytes: u64, hardware: HardwareProfile) -> FitReport {
    let (fits, needed_mb, available_mb) = fits_size(bytes, &hardware);
    let reason = if fits {
        format!("fits: about {needed_mb} MiB of weights, and {available_mb} MiB is free")
    } else {
        format!(
            "does not fit: about {needed_mb} MiB of weights but only {available_mb} MiB is free; \
             a smaller quantisation would"
        )
    };
    FitReport {
        fits,
        needed_mb,
        available_mb,
        suggested_gpu_layers: None,
        suggested_ctx: None,
        suggested_quant: None,
        reason,
        hardware,
    }
}

/// Measure a checkpoint against this machine.
///
/// # Errors
///
/// Fails when the hardware cannot be probed or llama.cpp cannot read the file.
/// A model that simply does not fit is a successful call with `fits: false`.
pub fn fit(path: &Path, n_ctx: u32) -> Result<FitReport> {
    let hardware = probe()?;
    let n_ctx = if n_ctx == 0 { DEFAULT_CTX } else { n_ctx };

    // Held for the duration: both helpers read the checkpoint with `no_alloc`,
    // which still needs a live backend.
    let backend = crate::engine::global_backend()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(ALL_LAYERS);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));

    let measured = match get_device_memory_data(
        path,
        &model_params,
        &ctx_params,
        llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
    ) {
        Ok(r) => r,
        Err(e) => {
            return Ok(FitReport::unknown(
                format!("could not measure {}: {e}", path.display()),
                hardware,
            ))
        }
    };

    // A model split across two cards fails if either piece does not fit, so the
    // need is the biggest piece and the supply is the tightest device — the
    // honest "X of Y free" comparison is against the minimum, not the largest.
    let needed_mb = measured
        .entries
        .iter()
        .map(|e| mb(e.used()))
        .max()
        .unwrap_or(0);
    let available_mb = measured
        .entries
        .iter()
        .map(|e| mb(e.free.max(0) as usize))
        .min()
        .unwrap_or(hardware.largest_free_mb());

    let fits = measured.entries.iter().all(|e| {
        let free = e.free.max(0) as usize;
        e.used() <= free
    });

    // Ask llama.cpp what it would do. It answers even when the answer is "fewer
    // layers", which is the usual fix and a cheaper one than requantising.
    let fitted = fit_params(
        backend,
        path,
        FitParams::default()
            .with_context_params(LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)))
            .with_n_ctx_min(MIN_CTX),
    );

    let (suggested_gpu_layers, suggested_ctx) = match &fitted {
        Ok(r) => (
            Some(layers_or_all(r.model_params.n_gpu_layers())),
            r.context_params.n_ctx().map(NonZeroU32::get),
        ),
        Err(_) => (None, None),
    };

    let suggested_quant = if fits {
        None
    } else {
        // Only worth naming when the weights are the obstacle: if llama.cpp
        // found a layer split that fits, the same file is fine as it is.
        match &fitted {
            Ok(_) => None,
            Err(FitParamsError::CouldNotFit) => suggest_quant(path, needed_mb, available_mb),
            Err(_) => None,
        }
    };

    let reason = reason(
        fits,
        needed_mb,
        available_mb,
        suggested_gpu_layers,
        &suggested_quant,
    );

    Ok(FitReport {
        fits,
        needed_mb,
        available_mb,
        suggested_gpu_layers,
        suggested_ctx,
        suggested_quant,
        reason,
        hardware,
    })
}

/// Check a size that has no local file yet — a download that has not started.
///
/// Weights are the only thing known at that point, and they dominate, so this
/// is a comparison against the largest thing the machine can hold rather than a
/// projection. It answers "is this worth starting", not "will this load".
pub fn fits_size(bytes: u64, hardware: &HardwareProfile) -> (bool, u64, u64) {
    // 32-bit `usize` would silently truncate a file larger than 4 GiB. This
    // crate only ships for 64-bit Windows and macOS today, so the assert is the
    // honest guarantee; a 32-bit port would have to switch to a checked
    // conversion here and in `mb`.
    debug_assert!(bytes <= usize::MAX as u64);
    let available = hardware.largest_free_mb();
    (
        mb(bytes as usize) <= available,
        mb(bytes as usize),
        available,
    )
}

/// Build the sentence a person reads.
fn reason(
    fits: bool,
    needed_mb: u64,
    available_mb: u64,
    layers: Option<u32>,
    quant: &Option<String>,
) -> String {
    if fits {
        return match layers {
            Some(n) => format!(
                "fits: needs about {needed_mb} MiB of {available_mb} MiB free; \
                 llama.cpp would offload {n} layers"
            ),
            None => format!("fits: needs about {needed_mb} MiB of {available_mb} MiB free"),
        };
    }

    match quant {
        Some(q) => format!(
            "does not fit: needs about {needed_mb} MiB but only {available_mb} MiB is free; \
             {q} would, or load fewer layers"
        ),
        None => match layers {
            Some(n) => format!(
                "does not fit at this layer count: needs about {needed_mb} MiB of \
                 {available_mb} MiB free; llama.cpp would offload {n} layers instead"
            ),
            None => format!(
                "does not fit: needs about {needed_mb} MiB but only {available_mb} MiB is free"
            ),
        },
    }
}

/// Quantisation labels and their rough bytes per weight, largest first.
///
/// Used only to turn "this must shrink by X%" into a name someone can search
/// for. Approximate is fine: the suggestion is a starting point, and whether it
/// truly fits is settled by measuring the file that was actually downloaded.
const QUANT_LADDER: &[(&str, f32)] = &[
    ("Q8_K_M", 1.06),
    ("Q6_K", 0.81),
    ("Q5_K_M", 0.69),
    ("Q5_K_S", 0.68),
    ("Q4_K_M", 0.60),
    ("Q4_K_S", 0.57),
    ("IQ4_XS", 0.53),
    ("Q3_K_M", 0.51),
    ("Q3_K_S", 0.47),
    ("Q2_K", 0.40),
];

/// Turn llama.cpp's layer count into one a person can type.
///
/// It reports `-1` where it means "offload as much as possible" — the macOS
/// default — which is not a value anyone can pass to `--n-gpu-layers`. Reported
/// as the "all layers" figure rather than dropped, because "everything" is the
/// useful half of the advice.
fn layers_or_all(n: i32) -> u32 {
    if n < 0 {
        ALL_LAYERS
    } else {
        n as u32
    }
}

/// Name a smaller quantisation that would fit, if the file says which one it is.
///
/// The label has to come from the file name because nothing in a GGUF at this
/// stage exposes the quantisation without loading the model — and loading it is
/// exactly what is in question. When the name does not carry one, no guess is
/// offered rather than a wrong one.
fn suggest_quant(path: &Path, needed_mb: u64, available_mb: u64) -> Option<String> {
    if needed_mb == 0 || available_mb == 0 {
        return None;
    }
    let scale = available_mb as f32 / needed_mb as f32;

    let name = path.file_name()?.to_string_lossy().to_uppercase();
    let current = QUANT_LADDER
        .iter()
        .find(|(label, _)| name.contains(label))?
        .1;

    let target = current * scale;
    QUANT_LADDER
        .iter()
        .filter(|(label, bpw)| *bpw < current && *bpw <= target && !name.contains(label))
        .map(|(label, _)| label.to_string())
        .next()
}

fn mb(bytes: usize) -> u64 {
    (bytes / (1024 * 1024)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_model_is_not_reported_as_unfitting() {
        let hw = HardwareProfile::default();
        let r = FitReport::unknown("could not read the file", hw);
        // Refusing a model for no stated reason would be worse than trying it.
        assert!(r.fits);
        assert_eq!(r.suggested_quant, None);
        assert_eq!(r.reason, "could not read the file");
    }

    #[test]
    fn a_size_is_compared_against_the_largest_free_device() {
        let hw = HardwareProfile {
            ram_free_mb: 4 * 1024,
            gpus: vec![],
            ..Default::default()
        };
        let (fits, _, available) = fits_size(8 * 1024 * 1024 * 1024, &hw);
        assert!(!fits);
        assert_eq!(available, 4 * 1024);
    }

    #[test]
    fn a_quant_is_only_suggested_from_a_name_that_carries_one() {
        // Halving is within reach of the ladder, so a named file gets an answer.
        let named = suggest_quant(Path::new("Model-Q8_K_M.gguf"), 16 * 1024, 8 * 1024);
        assert!(named.is_some(), "expected a suggestion for a named quant");

        // Unnamed: offering a guess here would be a confident wrong answer.
        assert_eq!(
            suggest_quant(Path::new("model.gguf"), 16 * 1024, 8 * 1024),
            None
        );
    }

    #[test]
    fn no_quant_is_suggested_when_the_shrink_is_beyond_the_ladder() {
        // A quarter of the size is past Q2_K, the bottom rung. Saying so by
        // offering nothing beats naming a quant that still would not fit.
        assert_eq!(
            suggest_quant(Path::new("M-Q8_K_M.gguf"), 16 * 1024, 4 * 1024),
            None
        );
    }

    #[test]
    fn no_quant_is_suggested_when_already_smallest() {
        // Q2_K is the bottom rung; there is nothing below it to offer.
        assert_eq!(
            suggest_quant(Path::new("M-Q2_K.gguf"), 16 * 1024, 4 * 1024),
            None
        );
    }
}
