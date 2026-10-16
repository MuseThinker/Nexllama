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

/// Largest context an automatic size will ask for.
///
/// A ceiling on the *ladder*, not a target: a model whose own limit is lower
/// keeps its own number, and a machine that cannot hold this much is capped by
/// its memory before it is capped here.
const AUTO_CTX_CAP: u32 = 262_144;

/// Context to fall back on when the model states none of its own.
///
/// Small enough to load on almost anything, and large enough for a modern chat
/// prompt — the old 4096 default is what a 4k-plus prompt used to overflow.
pub const DEFAULT_AUTO_CTX: u32 = 8192;

/// Smallest context worth recommending. Below this a model stops being useful
/// for anything but the shortest exchange.
const MIN_CTX: u32 = 2048;

/// What a checkpoint costs before a single token of context is allocated.
///
/// Two facts, both read from the file: the weights, which occupy device memory
/// whether or not the model is used, and the KV cache each token adds. Together
/// they are what decides the largest context that still fits — which is the
/// question an automatic context is really asking, and the one the device-size
/// ladder alone cannot answer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelFootprint {
    /// Weights on disk, in MiB.
    pub weights_mb: u64,
    /// KV cache bytes per token per sequence, at llama.cpp's default cache type.
    /// `None` when the header does not state the shape.
    pub kv_bytes_per_token: Option<u64>,
}

impl ModelFootprint {
    /// Read a checkpoint's footprint from its path.
    ///
    /// Never fails: an unreadable file or a header without the shape leaves the
    /// KV figure unknown, which makes the answer narrower, not wrong — the caller
    /// then falls back to the device ladder it used before.
    pub fn of(path: &Path) -> Self {
        let weights_mb = std::fs::metadata(path)
            .map(|m| mb(m.len() as usize))
            .unwrap_or(0);
        let kv_bytes_per_token = gguf_probe::probe(path)
            .ok()
            .and_then(|p| p.kv_bytes_per_token());
        Self {
            weights_mb,
            kv_bytes_per_token,
        }
    }
}

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

/// A report for a checkpoint that was not measured, with a stated reason.
///
/// Used when measurement would be wrong to attempt: llama.cpp's fitter loads
/// the checkpoint a second time inside this process, which is not safe while
/// another model is resident — it fails the load it is measuring. Reporting
/// "not measured, and here is why" is honest in a way that a guessed number is
/// not, and an unmeasured model is still allowed to load.
pub fn fit_unmeasured(path: &Path, n_ctx: u32, reason: &str) -> Result<FitReport> {
    let hardware = probe()?;
    let suggested_ctx = if n_ctx == 0 {
        train_ctx(path).map(|c| c.clamp(MIN_CTX, AUTO_CTX_CAP))
    } else {
        None
    };
    let mut report = FitReport::unknown(reason.to_string(), hardware);
    report.suggested_ctx = suggested_ctx;
    Ok(report)
}

/// Measure a checkpoint against this machine.
///
/// # Errors
///
/// Fails when the hardware cannot be probed or llama.cpp cannot read the file.
/// A model that simply does not fit is a successful call with `fits: false`.
pub fn fit(path: &Path, n_ctx: u32) -> Result<FitReport> {
    let hardware = probe()?;

    // Held for the duration: both helpers read the checkpoint with `no_alloc`,
    // which still needs a live backend.
    let backend = crate::engine::global_backend()?;

    // What the caller asked for, or — when they left it to us (`0`) — as much as
    // the model was trained for, capped. This is the ceiling the KV cache is
    // sized against; llama.cpp then reports the largest context that fits.
    let target = if n_ctx == 0 {
        train_ctx(path)
            .unwrap_or(DEFAULT_AUTO_CTX)
            .clamp(MIN_CTX, AUTO_CTX_CAP)
    } else {
        n_ctx
    };

    // Ask llama.cpp what it would settle for. That answer is both the advice and
    // the context the report is measured at: sizing the cache for a context the
    // machine cannot hold would report a model that fits as one that does not.
    let fitted = fit_params(
        backend,
        path,
        FitParams::default()
            .with_context_params(LlamaContextParams::default().with_n_ctx(NonZeroU32::new(target)))
            .with_n_ctx_min(MIN_CTX),
    );

    let (suggested_gpu_layers, suggested_ctx) = match &fitted {
        Ok(r) => (
            Some(layers_or_all(r.model_params.n_gpu_layers())),
            r.context_params.n_ctx().map(NonZeroU32::get),
        ),
        Err(_) => (None, None),
    };

    let measured_ctx = suggested_ctx.unwrap_or(target);
    let model_params = LlamaModelParams::default().with_n_gpu_layers(ALL_LAYERS);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(measured_ctx));

    let measured = match get_device_memory_data(
        path,
        &model_params,
        &ctx_params,
        llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
    ) {
        Ok(r) => r,
        Err(e) => {
            // Measurement failed — llama.cpp's memory probe errors on some
            // architectures (MoE among them). Rather than leave an automatic
            // context at the tiny fallback, still offer the model's own training
            // length, capped: a model that says it handles 32k should not come up
            // asking for 8k just because we could not weigh it.
            let mut report = FitReport::unknown(
                format!("could not measure {}: {e}", path.display()),
                hardware,
            );
            report.suggested_ctx = train_ctx(path).map(|c| c.clamp(MIN_CTX, AUTO_CTX_CAP));
            return Ok(report);
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

/// The context the checkpoint was trained for, if its header states one.
///
/// The ceiling for an automatic size: extending past it buys a longer window
/// than the model was trained to use.
/// The largest context this model states, if it states one.
pub fn train_ctx(path: &Path) -> Option<u32> {
    let stated = gguf_probe::probe(path).ok()?.context_length?;
    u32::try_from(stated).ok().filter(|c| *c > 0)
}

/// The context to use when the caller leaves it to us.
///
/// Two things decide it, in this order: what the checkpoint says it can do, and
/// how much device memory the machine has to hold the KV cache. A checkpoint's
/// own limit is a hard ceiling — asking a model trained for 8k to run at 64k
/// buys nothing — and the device decides how far below it is worth going.
///
/// The ladder is deliberately coarse. Context is not a dial to be tuned by
/// percentage points: 32k and 36k cost about the same and differ in nothing a
/// person can perceive, while every step up is memory that a second model could
/// have used. Coarse steps also make the number predictable, which matters more
/// than optimality for something the user never asked to configure.
///
/// Encoders are not on the ladder at all. An embedding or rerank model is given
/// whole sequences, its own limit is small (typically 512–8192), and its KV
/// cache is not what a machine runs out of — so it gets its own limit, capped.
///
/// The ladder sets a ceiling from the device's *size*, and `footprint` lowers it
/// to what the *model* leaves behind. The second part matters: the ladder assumes
/// a card serves a model that fits in it, and a 27B's weights are most of a 24 GiB
/// card. Without the cap, that model was handed a 32k cache it had no room for,
/// and llama.cpp spilled layers onto the CPU to make room — which is the
/// difference between a reply that starts in a second and one that takes minutes.
pub fn auto_ctx(
    class: &str,
    train_ctx: Option<u32>,
    footprint: ModelFootprint,
    hardware: &HardwareProfile,
) -> u32 {
    const ENCODER_CAP: u32 = 8192;

    let stated = train_ctx.unwrap_or(AUTO_CTX_CAP).max(MIN_CTX);
    if class != "llm" {
        return stated.min(ENCODER_CAP);
    }

    // A card's *total* memory, not its free memory: the answer has to be stable
    // across a restart, and a context that shrank because something else was
    // resident would be a different model each run.
    //
    // Rungs are compared with slack because a card never reports the round
    // number on its box: a 24 GB card says 24575 MiB, which is one MiB short of
    // the rung and would otherwise land a whole step lower.
    const SLACK_MB: u64 = 1024;
    let gpu_mb = hardware
        .gpus
        .iter()
        .map(|g| g.vram_total_mb)
        .max()
        .unwrap_or(0);
    let vram = gpu_mb + SLACK_MB;
    let by_device = if vram >= 96 * 1024 {
        262_144
    } else if vram >= 48 * 1024 {
        131_072
    } else if vram >= 32 * 1024 {
        65_536
    } else if vram >= 24 * 1024 {
        32_768
    } else if vram >= 16 * 1024 {
        16_384
    } else if vram >= 8 * 1024 {
        8_192
    } else if hardware.gpus.is_empty() && hardware.ram_total_mb >= 64 * 1024 {
        32_768
    } else if hardware.gpus.is_empty() && hardware.ram_total_mb >= 32 * 1024 {
        16_384
    } else if hardware.gpus.is_empty() {
        MIN_CTX
    } else {
        4_096
    };

    // What the cache may spend, once the weights are somewhere. Weights that do
    // not fit leave nothing, and the context falls to the floor rather than
    // asking for a cache the machine cannot hold on top of a spill it already
    // cannot avoid.
    let total_mb = if gpu_mb > 0 {
        gpu_mb
    } else {
        hardware.ram_total_mb
    };
    // Room for the compute buffers llama.cpp allocates beside the cache.
    const WORKSPACE_MB: u64 = 1024;
    let budget_mb = total_mb
        .saturating_sub(footprint.weights_mb)
        .saturating_sub(WORKSPACE_MB);
    let by_memory = match footprint.kv_bytes_per_token {
        Some(per_token) if per_token > 0 => {
            let tokens = budget_mb.saturating_mul(1024 * 1024) / per_token;
            // Coarse on purpose, like the rungs: a number nobody will tune is
            // worth more when it is easy to predict than when it is exact.
            let rounded = tokens - (tokens % 512);
            u32::try_from(rounded).unwrap_or(u32::MAX)
        }
        // No shape in the header: the rung is the only answer available.
        _ => u32::MAX,
    };

    by_device
        .min(by_memory)
        .min(stated)
        .clamp(MIN_CTX, AUTO_CTX_CAP)
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

    /// A footprint with no shape in the header, which is the case the ladder
    /// alone still answers.
    const UNKNOWN: ModelFootprint = ModelFootprint {
        weights_mb: 0,
        kv_bytes_per_token: None,
    };

    fn machine(vram_total_mb: u64) -> HardwareProfile {
        use crate::hwprofile::GpuInfo;
        HardwareProfile {
            ram_total_mb: 64 * 1024,
            gpus: vec![GpuInfo {
                name: "test".into(),
                backend: "cuda".into(),
                vram_total_mb,
                vram_free_mb: vram_total_mb,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn an_automatic_context_follows_the_device_and_the_model() {
        // The card decides, while the model's own limit stays a hard ceiling.
        // 32 GB is the 64k rung; a 256k model is happy to be asked for 64k.
        assert_eq!(
            auto_ctx("llm", Some(262_144), UNKNOWN, &machine(32 * 1024)),
            65_536
        );
        // Same card, a model trained for 32k: its own limit wins.
        assert_eq!(
            auto_ctx("llm", Some(32_768), UNKNOWN, &machine(32 * 1024)),
            32_768
        );
        // The 24 GB rung is 32k — the number a card of that size can hold
        // alongside a second model. A 24 GB card reports 24575 MiB, so the rung
        // has to be reached by a card that is not a round number.
        assert_eq!(
            auto_ctx("llm", Some(262_144), UNKNOWN, &machine(24 * 1024)),
            32_768
        );
        assert_eq!(
            auto_ctx("llm", Some(262_144), UNKNOWN, &machine(24_575)),
            32_768
        );
        // A model that states nothing is treated generously, up to the rung.
        assert_eq!(auto_ctx("llm", None, UNKNOWN, &machine(48 * 1024)), 131_072);
        // A small card still gets a usable context rather than a token one.
        assert_eq!(
            auto_ctx("llm", Some(262_144), UNKNOWN, &machine(4 * 1024)),
            4_096
        );
        // No accelerator: system memory decides, and modestly.
        let cpu = HardwareProfile {
            ram_total_mb: 64 * 1024,
            ..Default::default()
        };
        assert_eq!(auto_ctx("llm", Some(262_144), UNKNOWN, &cpu), 32_768);
    }

    /// The device ladder does not know what is being served. A checkpoint whose
    /// weights fill most of the card has to get the context its memory allows,
    /// not the one a card of that size usually serves — that difference is a
    /// model that stays on the GPU against one llama.cpp spills onto the CPU.
    #[test]
    fn a_large_model_is_capped_to_the_memory_its_weights_leave() {
        // 27B, ~17 GiB of weights, 48 layers of 40 KV heads 128 wide, f16: about
        // 0.94 MiB of cache per token. The 24 GiB rung would ask for 32k, which
        // is 30 GiB of cache — twice what the card has left.
        let dense = ModelFootprint {
            weights_mb: 17_408,
            kv_bytes_per_token: Some(48 * 40 * 128 * 2 * 2),
        };
        let capped = auto_ctx("llm", Some(262_144), dense, &machine(24 * 1024));
        assert!(
            capped < 32_768,
            "a 27B's 32k cache does not fit a 24 GiB card, got {capped}"
        );
        // 24 GiB, less 17 GiB of weights, less 1 GiB of workspace, is ~6 GiB.
        assert_eq!(capped, 6_144);

        // The same weights with grouped-query attention: 8 KV heads is a
        // sixteenth of the cache, so the card does have room and the rung stands.
        let grouped = ModelFootprint {
            weights_mb: 17_408,
            kv_bytes_per_token: Some(48 * 8 * 128 * 2 * 2),
        };
        let roomy = auto_ctx("llm", Some(262_144), grouped, &machine(24 * 1024));
        assert!(roomy > 16_384, "GQA leaves room for 32k, got {roomy}");

        // Weights bigger than the card leave nothing: fall to the floor rather
        // than compound a spill that cannot be avoided with a cache that cannot
        // be held.
        let oversized = ModelFootprint {
            weights_mb: 40 * 1024,
            kv_bytes_per_token: Some(48 * 40 * 128 * 2 * 2),
        };
        assert_eq!(
            auto_ctx("llm", Some(262_144), oversized, &machine(24 * 1024)),
            2_048
        );
    }

    #[test]
    fn an_automatic_context_for_an_encoder_is_its_own_limit_capped() {
        // An encoder is asked for whole documents; its own limit is the answer,
        // and the cap keeps a model that states an enormous one from reserving
        // a cache it will never fill.
        assert_eq!(
            auto_ctx("embedding", Some(8_192), UNKNOWN, &machine(48 * 1024)),
            8_192
        );
        assert_eq!(
            auto_ctx("embedding", Some(512), UNKNOWN, &machine(48 * 1024)),
            2_048
        );
        assert_eq!(
            auto_ctx("rerank", Some(1_000_000), UNKNOWN, &machine(96 * 1024)),
            8_192
        );
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
