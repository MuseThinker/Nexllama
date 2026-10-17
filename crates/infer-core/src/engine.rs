use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use llama_cpp_4::common_sampler::{CommonSampler, CommonSamplerParams};
use llama_cpp_4::context::params::LlamaContextType;
use llama_cpp_4::eagle::{Eagle3Session, Eagle3SessionConfig};
use llama_cpp_4::model::params::LlamaLoadMode;
use llama_cpp_4::model::LlamaLoraAdapter;
use llama_cpp_4::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_4::mtp::{MtpSession, MtpSessionConfig};
use llama_cpp_4::ngram::ngram_simple_draft;
use llama_cpp_4::prelude::*;

use crate::error::{InferError, Result};
use crate::promptcache::PromptCache;
use crate::sampler::SamplerParams;
use crate::types::{ChatMsg, GenStats};

/// How model weights are read from disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadMode {
    #[default]
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl From<LoadMode> for LlamaLoadMode {
    fn from(m: LoadMode) -> Self {
        match m {
            LoadMode::Auto => LlamaLoadMode::Auto,
            LoadMode::None => LlamaLoadMode::None,
            LoadMode::Mmap => LlamaLoadMode::Mmap,
            LoadMode::Mlock => LlamaLoadMode::Mlock,
            LoadMode::MmapMlock => LlamaLoadMode::MmapMlock,
            LoadMode::DirectIo => LlamaLoadMode::DirectIo,
        }
    }
}

/// Flash-attention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttn {
    #[default]
    Auto,
    On,
    Off,
}

/// A LoRA adapter applied to the model at load time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraAdapter {
    pub path: PathBuf,
    #[serde(default = "one")]
    pub scale: f32,
}

fn one() -> f32 {
    1.0
}

/// Everything needed to bring a model up. Defaults are the recommended values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadConfig {
    pub model_path: PathBuf,
    /// Layers offloaded to GPU. 999 (or more) means "all layers".
    pub n_gpu_layers: u32,
    /// GPU used for the first layers.
    pub main_gpu: i32,
    /// Context size. 0 = the model's training context.
    pub n_ctx: u32,
    /// Logical batch size.
    pub n_batch: u32,
    /// Physical (micro) batch size.
    pub n_ubatch: u32,
    /// CPU threads for generation. None = llama.cpp's default (4). Raising it is
    /// a regression on this GPU-bound path, where extra threads only add host
    /// contention.
    pub n_threads: Option<i32>,
    /// CPU threads for prompt processing. None = same as n_threads.
    pub n_threads_batch: Option<i32>,
    pub load_mode: LoadMode,
    pub use_mlock: bool,
    pub flash_attn: FlashAttn,
    pub offload_kqv: bool,
    /// RoPE base frequency. 0.0 = model default.
    pub rope_freq_base: f32,
    /// RoPE frequency scaling. 0.0 = model default.
    pub rope_freq_scale: f32,
    /// Load the model's bundled MTP (NextN) head for speculative decoding.
    ///
    /// This is a request to draft, not merely to keep the weights around, so it
    /// picks up `draft_n_max` when `mtp_n_max` leaves the budget at 0.
    pub load_mtp: bool,
    /// Recurrent-state snapshots kept per sequence so a rejected draft can be
    /// rolled back. Hybrid/recurrent caches refuse partial rollback at 0.
    #[serde(default)]
    pub n_rs_seq: u32,
    /// MTP draft tokens per round (upstream `--spec-draft-n-max`).
    ///
    /// 0 takes the budget from `load_mtp`. Anything above it needs a second
    /// (draft) context and forces `load_mtp`, because the weights have to be
    /// resident to draft at all.
    #[serde(default)]
    pub mtp_n_max: u32,
    /// A **separate** draft model whose proposals the target verifies.
    ///
    /// This is the other half of speculative decoding: `load_mtp` uses heads that
    /// come bundled in the target, while this takes a second checkpoint — DFlash
    /// (block diffusion, from Z-Lab) or EAGLE-3 — that was trained to draft for
    /// it. They are complementary rather than interchangeable, because an MTP
    /// head only exists on models that shipped one, and DFlash exists precisely
    /// to cover the dense models that did not.
    #[serde(default)]
    pub draft_model: Option<PathBuf>,
    /// How many tokens a draft may propose per round.
    ///
    /// 0 picks 2. Drafting is a trade: every extra token is one more round of
    /// the drafter and a wider verification pass, and acceptance falls as the
    /// draft runs further ahead. Measured on Qwen3.8-27B (24 tok/s baseline),
    /// 2 was the peak for both a side-mounted DFlash checkpoint and a bundled
    /// MTP head; a budget of 8 left nothing on the table but time. The hard
    /// bound is `n_batch`, since the whole proposal is verified in one pass.
    #[serde(default)]
    pub draft_n_max: u32,
    /// LoRA adapters to apply.
    pub lora: Vec<LoraAdapter>,
    /// Where to keep learned n-gram draft statistics between runs.
    ///
    /// `None` keeps them in memory only, so drafts never carry across restarts.
    #[serde(default)]
    pub ngram_cache: Option<PathBuf>,
    /// Multimodal projector that turns images into tokens this model can read.
    #[serde(default)]
    pub mmproj: Option<PathBuf>,

    /// Reuse the KV state of a prefix this model has already encoded.
    ///
    /// On by default. It is a stateful optimisation, so a bug would not raise an
    /// error — it would answer from the wrong point in the conversation. That is
    /// why `prefix_check` verifies the restored state against a cache-off run,
    /// and why it is switched off automatically whenever a drafter (an MTP head
    /// or a side-mounted draft) is loaded, since those keep position state of
    /// their own.
    #[serde(default = "default_true")]
    pub prefix_cache: bool,
    /// How many prompts to remember.
    #[serde(default)]
    pub prefix_cache_entries: u32,
    /// Ceiling on the state held, in MiB.
    ///
    /// Bytes, not tokens: a remembered prompt costs its serialised KV cache,
    /// which for a large model is hundreds of kilobytes per token, so a budget
    /// counted in tokens would look small while quietly holding gigabytes.
    #[serde(default = "default_cache_mb")]
    pub prefix_cache_max_mb: u32,

    /// Stop drafting when measurement says it is not paying for itself.
    ///
    /// On by default: it only ever switches drafting off, and only after enough
    /// replies have been measured, so the worst case is that a speculative
    /// setting stops being used because it was shown not to help.
    #[serde(default = "default_true")]
    pub adaptive_draft: bool,
    /// Acceptance rate below which drafting is judged not worth it.
    #[serde(default = "default_accept_floor")]
    pub accept_floor: f32,
}

fn default_true() -> bool {
    true
}

fn default_accept_floor() -> f32 {
    0.3
}

fn default_cache_mb() -> u32 {
    512
}

impl LoadConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            n_gpu_layers: 999,
            main_gpu: 0,
            n_ctx: 4096,
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: None,
            n_threads_batch: None,
            load_mode: LoadMode::Auto,
            use_mlock: false,
            flash_attn: FlashAttn::Auto,
            offload_kqv: true,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            load_mtp: false,
            n_rs_seq: 0,
            mtp_n_max: 0,
            draft_model: None,
            draft_n_max: 0, // 0 = 2, see `draft_n_max`
            lora: Vec::new(),
            ngram_cache: None,
            mmproj: None,
            // On by default; see the note on `prefix_cache`.
            prefix_cache: true,
            prefix_cache_entries: 4,
            prefix_cache_max_mb: 512,
            adaptive_draft: true,
            accept_floor: 0.3,
        }
    }
}

/// Drafting is only worth it while the model confirms what is proposed.
///
/// A rejected proposal is a forward pass that produced nothing, so drafting at a
/// low acceptance rate is slower than not drafting at all. This decides from
/// what drafting has *actually* achieved on this model, rather than from whether
/// it was switched on.
///
/// It only ever turns drafting off, and only after enough replies to be sure —
/// which is the safe direction: the worst case is that a speculative setting
/// quietly stops being used because measurement said it was not helping.
#[derive(Debug)]
pub struct SpecGate {
    /// Smoothed share of drafted tokens the model confirmed.
    accept: Option<f64>,
    /// Replies that actually drafted, which is the sample size.
    replies: u64,
    disabled: bool,
    /// Below this, drafting is judged not to be paying for itself.
    floor: f64,
    /// Above this, it is allowed back. Higher than `floor` so a model sitting on
    /// the boundary does not flicker between drafting and not.
    resume: f64,
    /// Replies needed before the rate is trusted.
    min_replies: u64,
}

impl SpecGate {
    /// A gate that never interferes, for when the feature is off.
    fn open() -> Self {
        Self {
            accept: None,
            replies: 0,
            disabled: false,
            floor: 0.0,
            resume: 0.0,
            min_replies: u64::MAX,
        }
    }

    fn enabled(floor: f64) -> Self {
        Self {
            floor: floor.clamp(0.0, 1.0),
            resume: (floor + 0.15).clamp(0.0, 1.0),
            min_replies: 8,
            ..Self::open()
        }
    }

    /// Whether drafting should be attempted for the next reply.
    fn allows(&self) -> bool {
        !self.disabled
    }

    /// Draft length to use, scaled by the acceptance measured so far.
    ///
    /// Two drafts is the measured sweet spot on both a dense and a MoE target:
    /// the accepted-token count barely moves between one and two, so the second
    /// draft only has to survive its own verification to pay off, while a third
    /// is rejected often enough to lose. Under a poor measured acceptance even
    /// the second draft is not worth its verification pass. `max` is the
    /// configured budget — also the KV-snapshot count the contexts were built
    /// with — so the result never exceeds it.
    fn draft_len(&self, max: u32) -> u32 {
        let capped = max.clamp(1, 2);
        match self.accept {
            Some(a) if a < 0.5 => 1,
            _ => capped,
        }
    }

    /// Fold one reply's drafting results into the decision.
    fn record(&mut self, stats: &GenStats) {
        // A reply that never proposed anything says nothing about acceptance,
        // so it must not count as a rejection.
        let Some(rate) = stats.draft_acceptance() else {
            return;
        };
        self.replies += 1;
        self.accept = Some(match self.accept {
            None => rate,
            Some(prev) => prev + 0.2 * (rate - prev),
        });
        let Some(accept) = self.accept else { return };

        if self.replies >= self.min_replies && accept < self.floor {
            self.disabled = true;
        } else if self.disabled && accept >= self.resume {
            self.disabled = false;
        }
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    fn drafted(proposed: usize, accepted: usize) -> GenStats {
        GenStats {
            draft_tokens: proposed,
            accepted_draft_tokens: accepted,
            ..Default::default()
        }
    }

    #[test]
    fn a_reply_that_never_proposed_says_nothing() {
        let mut g = SpecGate::enabled(0.3);
        for _ in 0..20 {
            g.record(&GenStats::default());
        }
        // Counting these as rejections would switch drafting off for a model
        // that was never asked to draft.
        assert_eq!(g.replies, 0);
        assert!(g.allows());
    }

    #[test]
    fn a_drafter_that_confirms_nothing_is_switched_off() {
        let mut g = SpecGate::enabled(0.3);
        // Well past the samples needed before the rate is trusted.
        for _ in 0..20 {
            g.record(&drafted(8, 0));
        }
        assert!(!g.allows(), "drafting that confirms nothing must stop");
    }

    #[test]
    fn a_drafter_that_recovers_is_allowed_back() {
        let mut g = SpecGate::enabled(0.3);
        for _ in 0..20 {
            g.record(&drafted(8, 0));
        }
        assert!(!g.allows());
        for _ in 0..40 {
            g.record(&drafted(8, 8));
        }
        assert!(
            g.allows(),
            "a recovered acceptance rate must re-enable drafting"
        );
    }

    #[test]
    fn a_healthy_drafter_is_left_alone() {
        let mut g = SpecGate::enabled(0.3);
        for _ in 0..20 {
            g.record(&drafted(8, 7));
        }
        assert!(g.allows());
    }

    #[test]
    fn an_open_gate_never_interferes() {
        let mut g = SpecGate::open();
        for _ in 0..50 {
            g.record(&drafted(8, 0));
        }
        assert!(g.allows());
    }
}

/// Cap on learned n-gram contexts.
///
/// The table grows with every conversation and nothing prunes it, so a server left
/// running for months would otherwise accumulate without limit. Past this point
/// new text stops being learned; drafts from what is already known keep working.
const NGRAM_CACHE_MAX_CONTEXTS: usize = 1 << 16;

/// `(context, [(next token, times seen)])` for one entry of [`NgramModel`].
type NgramEntry = (Vec<i32>, Vec<(i32, u32)>);

/// A statistical n-gram model over token ids, used to seed speculative drafts.
///
/// llama.cpp ships `common_ngram_cache` for this, but its `update` entry point throws
/// in this build and it gives no control over the file format. This is small enough to
/// own outright: a context of `order` tokens maps to the tokens that have followed it,
/// and how often.
///
/// Entries are a sorted list rather than a map because JSON object keys have to be
/// strings, and token contexts are not. Sorting also makes lookups a binary search and
/// keeps both tie-breaking and the serialised file deterministic.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct NgramModel {
    /// Context length the table was built at. A different setting invalidates it,
    /// because the keys would no longer line up.
    pub(crate) order: usize,
    /// `(context, [(next token, times seen)])`, sorted by context then by token.
    entries: Vec<NgramEntry>,
}

impl NgramModel {
    pub(crate) fn contexts(&self) -> usize {
        self.entries.len()
    }

    /// Count every `order`-token window in `tokens` and what followed it.
    pub(crate) fn learn(&mut self, tokens: &[LlamaToken], order: usize) {
        let order = order.max(1);
        if self.order != order {
            // The keys mean something different now, so start over.
            self.order = order;
            self.entries.clear();
        }
        if self.entries.len() >= NGRAM_CACHE_MAX_CONTEXTS {
            return;
        }
        let ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();
        if ids.len() <= order {
            return;
        }

        // Rebuild through a map so counts merge, then flatten back to sorted order.
        let mut merged: BTreeMap<Vec<i32>, BTreeMap<i32, u32>> = self
            .entries
            .drain(..)
            .map(|(ctx, followers)| (ctx, followers.into_iter().collect()))
            .collect();
        for window in ids.windows(order + 1) {
            let (ctx, rest) = window.split_at(order);
            *merged
                .entry(ctx.to_vec())
                .or_default()
                .entry(rest[0])
                .or_insert(0) += 1;
        }
        self.entries = merged
            .into_iter()
            .map(|(ctx, followers)| (ctx, followers.into_iter().collect()))
            .collect();
    }

    /// Fold another table into this one, adding counts.
    ///
    /// Used when the cache file on disk has moved on since it was read: without
    /// this, writing this process's table over it discards whatever the other
    /// one learned, and the larger cache the two of them should share never
    /// forms.
    pub(crate) fn merge(&mut self, other: &NgramModel) {
        if other.order != self.order || other.entries.is_empty() {
            return;
        }
        let mut merged: BTreeMap<Vec<i32>, BTreeMap<i32, u32>> = self
            .entries
            .drain(..)
            .map(|(ctx, followers)| (ctx, followers.into_iter().collect()))
            .collect();
        for (ctx, followers) in &other.entries {
            let slot = merged.entry(ctx.clone()).or_default();
            for (tok, count) in followers {
                *slot.entry(*tok).or_insert(0) += *count;
            }
        }
        self.entries = merged
            .into_iter()
            .map(|(ctx, followers)| (ctx, followers.into_iter().collect()))
            .collect();
        // Merging can push past the cap; keeping the first `MAX` keeps the
        // bound real instead of approximate.
        self.entries.truncate(NGRAM_CACHE_MAX_CONTEXTS);
    }

    fn followers(&self, ctx: &[i32]) -> Option<&Vec<(i32, u32)>> {
        self.entries
            .binary_search_by(|(key, _)| key.as_slice().cmp(ctx))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// The most likely continuation of `hist`, greedily extended.
    ///
    /// Stops as soon as a context is unknown, because guessing past that point is what
    /// makes a speculative decoder slower than plain sampling.
    pub(crate) fn draft(&self, hist: &[LlamaToken], max: usize) -> Vec<LlamaToken> {
        if self.order == 0 || hist.len() < self.order || max == 0 {
            return Vec::new();
        }
        let mut ctx: Vec<i32> = hist[hist.len() - self.order..]
            .iter()
            .map(|t| t.0)
            .collect();
        let mut out = Vec::new();
        while out.len() < max {
            let Some(followers) = self.followers(&ctx) else {
                break;
            };
            // Sorted by token id, so the lowest id wins a count tie deterministically.
            let Some(&(next, _)) = followers
                .iter()
                .max_by_key(|&(id, count)| (count, std::cmp::Reverse(id)))
            else {
                break;
            };
            out.push(LlamaToken(next));
            ctx.remove(0);
            ctx.push(next);
        }
        out
    }
}
/// n-gram statistics that outlive a single request.
///
/// `ngram_simple_draft` only ever sees the current conversation, so it rediscovers
/// the same patterns on every message. This learns from finished replies and writes
/// the table back to disk, so drafts keep working across requests and restarts.
pub(crate) struct DraftCache {
    model: NgramModel,
    path: Option<PathBuf>,
    /// Set once learning stops, so the caller can stop paying for lookups.
    saturated: bool,
}

impl DraftCache {
    pub(crate) fn load(path: Option<PathBuf>) -> Self {
        let model = path
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|raw| serde_json::from_str::<NgramModel>(&raw).ok())
            .unwrap_or_default();
        let saturated = model.contexts() >= NGRAM_CACHE_MAX_CONTEXTS;
        Self {
            model,
            path,
            saturated,
        }
    }

    /// Propose a continuation, or nothing when there is nothing worth copying.
    ///
    /// `hist` must include the token just sampled, since that is what the context
    /// is cut from.
    pub(crate) fn draft(&self, hist: &[LlamaToken], n: u16, m: usize) -> Vec<LlamaToken> {
        if self.model.order != usize::from(n) {
            // The table was built for a different context length; relearn first.
            return Vec::new();
        }
        self.model.draft(hist, m)
    }

    /// Fold a finished request into the statistics and persist them.
    pub(crate) fn learn(&mut self, tokens: &[LlamaToken], n: u16) {
        if self.saturated || tokens.is_empty() {
            return;
        }
        self.model.learn(tokens, usize::from(n));
        self.saturated = self.model.contexts() >= NGRAM_CACHE_MAX_CONTEXTS;

        let Some(path) = &self.path else { return };
        // Before the lock: `create_new` will not make a missing directory, so
        // asking for the lock first would fail on a first run and quietly skip
        // every write.
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(dir) = parent {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!(error = %e, path = %dir.display(), "could not create the n-gram cache directory");
                return;
            }
        }
        let Some(_lock) = lock_table(path) else {
            // Another process is mid-write. The table is a cache, and this
            // process will try again after its next reply, so skipping costs
            // nothing — racing would cost both tables.
            return;
        };
        // Fold in whatever landed on disk since this table was read, so the
        // write adds to the shared cache instead of replacing it.
        if let Ok(raw) = std::fs::read_to_string(path) {
            if let Ok(on_disk) = serde_json::from_str::<NgramModel>(&raw) {
                self.model.merge(&on_disk);
            }
        }
        let raw = match serde_json::to_string(&self.model) {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!(error = %e, "could not serialise the n-gram table");
                return;
            }
        };
        // Write via a temporary file so a crash mid-write cannot leave a
        // half-written table that fails to parse on the next start.
        let tmp = path.with_extension("tmp");
        if let Err(e) = std::fs::write(&tmp, raw) {
            tracing::warn!(error = %e, path = %tmp.display(), "could not write the n-gram table");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            tracing::warn!(error = %e, path = %path.display(), "could not replace the n-gram table");
        }
    }

    /// How many distinct contexts are stored, for tests and diagnostics.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn contexts(&self) -> usize {
        self.model.contexts()
    }
}

/// A lock file held while the n-gram table is rewritten.
struct FileLock(PathBuf);

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Take an exclusive lock beside `path`, or give up.
///
/// Two engines pointed at the same cache file both read, extend and rewrite it
/// under the same temporary name. The lock makes an update exclusive; one left
/// behind by a process that died is broken once it is old enough to be stale
/// rather than active.
fn lock_table(path: &Path) -> Option<FileLock> {
    let lock = path.with_extension("lock");
    let deadline = Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(_) => return Some(FileLock(lock)),
            Err(_) => {
                let age = std::fs::metadata(&lock)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok());
                if let Some(age) = age {
                    if age > std::time::Duration::from_secs(120) {
                        let _ = std::fs::remove_file(&lock);
                        continue;
                    }
                }
                if Instant::now() >= deadline {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
}
/// Static facts about a loaded model, resolved once at load time.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EngineInfo {
    pub model_path: Option<PathBuf>,
    pub description: String,
    pub n_ctx: u32,
    pub n_ctx_train: u32,
    pub n_layer: i32,
    pub n_vocab: i32,
    pub n_params: u64,
    pub model_size: u64,
    pub ftype: Option<String>,
    /// Whether a projector was loaded, i.e. whether this model can see images.
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub mmproj_path: Option<PathBuf>,
    /// Which decoder family the GGUF carries: `llm`, `embedding` or `rerank`.
    ///
    /// A string because this travels over HTTP and through Svelte in one hop;
    /// the enum it came from lives in `gguf-probe`.
    #[serde(default)]
    pub model_class: String,
    /// Width of the vector an embedding (or rank) call returns.
    #[serde(default)]
    pub n_embd: i32,
    /// Pooling the context actually ended up with, e.g. `cls` or `rank`.
    #[serde(default)]
    pub pooling: String,
    /// Whether the GGUF ships a multi-token-prediction (NextN) head.
    ///
    /// Capability, not state: `load_mtp` says whether it was loaded.
    #[serde(default)]
    pub mtp: bool,
    /// Why a requested drafter is not in play, when it failed to load.
    ///
    /// A drafter that will not load — an MTP head llama.cpp cannot read, a
    /// checkpoint that does not match the target — is not worth failing the
    /// whole model over. The weights are fine, so they are served without it and
    /// the reason is reported here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_error: Option<String>,
}

/// Progress emitted while a model is loading.
#[derive(Debug, Clone)]
pub enum LoadEvent {
    Phase(String),
    Ready(EngineInfo),
    Error(String),
}

enum Command {
    Generate {
        msgs: Vec<ChatMsg>,
        params: SamplerParams,
        reply: Sender<GenEvent>,
    },
    Embed {
        texts: Vec<String>,
        reply: Sender<Result<EmbedResult>>,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
        reply: Sender<Result<RerankResult>>,
    },
    /// Nothing to run: a liveness probe. The send only fails once the
    /// receiving end is gone, which is what tells `/health` a worker has died.
    Ping,
    Shutdown,
}

/// Pooled vectors plus the token count the request cost.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EmbedResult {
    /// One vector per input, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Total tokens across every input.
    pub tokens: usize,
}

/// Relevance scores, one per document, in the order they were submitted.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RerankResult {
    pub scores: Vec<f32>,
    pub tokens: usize,
}

enum GenEvent {
    Token(String),
    Done(GenStats),
    /// Carried as the error itself rather than a string: the caller has to be able
    /// to tell a hung-up client apart from a real generation failure.
    Error(InferError),
}

/// A cheap, cloneable handle used to drive an already-running worker.
#[derive(Clone)]
pub struct EngineHandle {
    tx: Arc<Mutex<Sender<Command>>>,
    info: Arc<EngineInfo>,
}

impl EngineHandle {
    pub fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// Whether the worker thread is still there.
    ///
    /// The send only fails when the receiving end has been dropped, which is
    /// exactly when the worker has exited — after a panic, or after a
    /// shutdown. That is what lets a caller tell the difference between a model
    /// that is loaded and a model that can still answer, instead of reporting
    /// every request as "worker gone" with no idea when it happened.
    pub fn is_alive(&self) -> bool {
        self.tx
            .lock()
            .map(|tx| tx.send(Command::Ping).is_ok())
            .unwrap_or(false)
    }

    /// Run one completion, invoking `on_token` for every decoded text chunk.
    ///
    /// `on_token` returns whether to keep going. Returning `false` stops the
    /// generation at the next token boundary and this call returns
    /// `InferError::Cancelled` — which is what makes "the client hung up"
    /// mean something. Without it a streamed reply whose reader walked away
    /// would keep decoding until `max_tokens`, holding the model's single
    /// worker thread and every request queued behind it for the whole time.
    pub fn generate(
        &self,
        msgs: &[ChatMsg],
        params: &SamplerParams,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<GenStats> {
        let (reply_tx, reply_rx) = mpsc::channel::<GenEvent>();
        {
            let tx = self.tx.lock().map_err(|_| InferError::WorkerGone)?;
            tx.send(Command::Generate {
                msgs: msgs.to_vec(),
                params: params.clone(),
                reply: reply_tx,
            })
            .map_err(|_| InferError::WorkerGone)?;
        }
        loop {
            match reply_rx.recv() {
                Ok(GenEvent::Token(t)) => {
                    if !on_token(&t) {
                        // Dropping the receiver is how "stop" reaches the
                        // worker: its next `send` fails, and it unwinds there
                        // instead of decoding the rest of the reply.
                        drop(reply_rx);
                        return Err(InferError::Cancelled);
                    }
                }
                Ok(GenEvent::Done(stats)) => return Ok(stats),
                Ok(GenEvent::Error(e)) => return Err(e),
                Err(_) => return Err(InferError::WorkerGone),
            }
        }
    }

    /// Embed each text, returning one vector per input in the same order.
    pub fn embed(&self, texts: &[String]) -> Result<EmbedResult> {
        let (reply_tx, reply_rx) = mpsc::channel();
        {
            let tx = self.tx.lock().map_err(|_| InferError::WorkerGone)?;
            tx.send(Command::Embed {
                texts: texts.to_vec(),
                reply: reply_tx,
            })
            .map_err(|_| InferError::WorkerGone)?;
        }
        reply_rx.recv().map_err(|_| InferError::WorkerGone)?
    }

    /// Score each document against a query, one score per document.
    pub fn rerank(&self, query: &str, documents: &[String]) -> Result<RerankResult> {
        let (reply_tx, reply_rx) = mpsc::channel();
        {
            let tx = self.tx.lock().map_err(|_| InferError::WorkerGone)?;
            tx.send(Command::Rerank {
                query: query.to_string(),
                documents: documents.to_vec(),
                reply: reply_tx,
            })
            .map_err(|_| InferError::WorkerGone)?;
        }
        reply_rx.recv().map_err(|_| InferError::WorkerGone)?
    }
}

/// Owns the inference worker thread. Dropping it shuts the worker down.
pub struct Engine {
    handle: EngineHandle,
    join: Option<JoinHandle<()>>,
}

impl Engine {
    /// Load a model and start the worker. `on_progress` is called for each phase.
    pub fn load(cfg: LoadConfig, on_progress: impl FnMut(LoadEvent)) -> Result<Self> {
        let (ready_tx, ready_rx) = mpsc::channel::<LoadEvent>();
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

        let join = thread::Builder::new()
            .name("infer-core".into())
            .spawn(move || worker(cfg, ready_tx, cmd_rx))?;

        let mut on_progress = on_progress;
        let info = loop {
            match ready_rx.recv() {
                Ok(LoadEvent::Ready(info)) => break info,
                Ok(LoadEvent::Error(e)) => return Err(InferError::ModelLoad(e)),
                Ok(phase @ LoadEvent::Phase(_)) => on_progress(phase),
                Err(_) => return Err(InferError::WorkerGone),
            }
        };

        let handle = EngineHandle {
            tx: Arc::new(Mutex::new(cmd_tx)),
            info: Arc::new(info),
        };
        Ok(Self {
            handle,
            join: Some(join),
        })
    }

    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    pub fn info(&self) -> &EngineInfo {
        self.handle.info()
    }

    pub fn generate(
        &self,
        msgs: &[ChatMsg],
        params: &SamplerParams,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<GenStats> {
        self.handle.generate(msgs, params, on_token)
    }

    pub fn embed(&self, texts: &[String]) -> Result<EmbedResult> {
        self.handle.embed(texts)
    }

    pub fn rerank(&self, query: &str, documents: &[String]) -> Result<RerankResult> {
        self.handle.rerank(query, documents)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(tx) = self.handle.tx.lock() {
            let _ = tx.send(Command::Shutdown);
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// llama.cpp allows a single process-wide backend, shared by every model.
///
/// Crate-visible because the hardware probe needs an initialised backend to
/// enumerate devices, and initialising a second one is not a thing llama.cpp
/// allows.
pub(crate) fn global_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    static INIT: Mutex<()> = Mutex::new(());
    let _guard = INIT.lock().map_err(|_| InferError::WorkerGone)?;
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let backend = LlamaBackend::init().map_err(|e| InferError::Backend(format!("{e:?}")))?;
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("backend just initialized"))
}

fn worker(cfg: LoadConfig, ready: Sender<LoadEvent>, rx: Receiver<Command>) {
    let _ = ready.send(LoadEvent::Phase("initializing llama backend".into()));
    let backend = match global_backend() {
        Ok(b) => b,
        Err(e) => {
            let _ = ready.send(LoadEvent::Error(format!("backend init: {e}")));
            return;
        }
    };

    // The file name, not the path: this line goes to a log that lives on disk
    // indefinitely, and a full path is a piece of someone's directory layout
    // that nothing here needs to remember.
    let file = cfg
        .model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    let _ = ready.send(LoadEvent::Phase(format!("loading model {file}")));
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(cfg.n_gpu_layers)
        .with_main_gpu(cfg.main_gpu)
        .with_load_mode(cfg.load_mode.into())
        .with_use_mlock(cfg.use_mlock)
        .with_load_mtp(cfg.load_mtp || cfg.mtp_n_max > 0);
    let model = match LlamaModel::load_from_file(backend, &cfg.model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(LoadEvent::Error(format!("model load: {e:?}")));
            return;
        }
    };

    let _ = ready.send(LoadEvent::Phase("creating context".into()));
    let flash = match cfg.flash_attn {
        FlashAttn::Auto => LlamaFlashAttnType::Auto,
        FlashAttn::On => LlamaFlashAttnType::Enabled,
        FlashAttn::Off => LlamaFlashAttnType::Disabled,
    };
    // The family decides how the context is built, so it has to be known before
    // the context exists: encoders need embeddings enabled and, for rankers, the
    // ranking head attached.
    let probed = gguf_probe::probe(&cfg.model_path).ok();
    let class = probed
        .as_ref()
        .map(gguf_probe::Probe::classify)
        .unwrap_or(gguf_probe::ModelClass::Llm);

    // The model's chat templates, parsed once and kept for the worker's life —
    // parsing them is not free. They are what lets a request turn reasoning on or
    // off (`enable_thinking`), which the legacy apply path cannot express. `None`
    // means the model ships no usable template, and every prompt falls back to
    // the legacy path unchanged.
    let templates = ChatTemplates::from_model(&model, None).ok();

    // Rejecting a draft walks the cache backwards one token at a time, and a
    // hybrid or recurrent cache only gives those rows back if it was built with
    // a snapshot per drafted token. Whichever drafter this load asked for
    // therefore decides the snapshot count before either context exists; leaving
    // it at 0 fails later, at verification time, with no way to recover.
    let draft_n_max = if cfg.draft_n_max > 0 {
        cfg.draft_n_max
    } else {
        2
    };
    // `load_mtp` says "draft with the bundled head", not "keep its weights
    // resident", so an unset budget falls back to the same one a side-mounted
    // drafter gets. Resolved here rather than at draft time: the snapshot count
    // below is read while the contexts are built and cannot change afterwards.
    let mtp_n_max = if cfg.load_mtp {
        cfg.mtp_n_max.max(draft_n_max)
    } else {
        cfg.mtp_n_max
    };
    let mut rs_seq = cfg.n_rs_seq;
    if cfg.draft_model.is_some() {
        rs_seq = rs_seq.max(draft_n_max);
    }
    if mtp_n_max > 0 {
        rs_seq = rs_seq.max((mtp_n_max as i32).max(4) as u32);
    }

    let build_ctx = |pooling: Option<LlamaPoolingType>| {
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(cfg.n_ctx))
            .with_n_batch(cfg.n_batch)
            .with_n_ubatch(cfg.n_ubatch)
            .with_flash_attn_type(flash)
            .with_offload_kqv(cfg.offload_kqv)
            .with_n_rs_seq(rs_seq);
        if class != gguf_probe::ModelClass::Llm {
            ctx_params = ctx_params.with_embeddings(true);
            // An encoder is fed a whole sequence at once, and llama.cpp
            // asserts `n_ubatch >= n_tokens` rather than splitting one. With the
            // generation defaults (512) a 512-token document is fine and a
            // 513-token one kills the process with an assertion — which reads to
            // the caller as the service simply vanishing. The micro-batch is
            // raised to the context instead, so any input the context accepts
            // can be encoded.
            ctx_params = ctx_params.with_n_ubatch(crate::encoder_ubatch(cfg.n_ubatch, cfg.n_ctx));
        }
        if let Some(p) = pooling {
            ctx_params = ctx_params.with_pooling_type(p);
        }
        if cfg.rope_freq_base > 0.0 {
            ctx_params = ctx_params.with_rope_freq_base(cfg.rope_freq_base);
        }
        if cfg.rope_freq_scale > 0.0 {
            ctx_params = ctx_params.with_rope_freq_scale(cfg.rope_freq_scale);
        }
        if let Some(t) = cfg.n_threads {
            ctx_params = ctx_params.with_n_threads(t);
        }
        if let Some(t) = cfg.n_threads_batch.or(cfg.n_threads) {
            ctx_params = ctx_params.with_n_threads_batch(t);
        }
        ctx_params
    };

    // Declared before the context on purpose. Drops run in reverse order, and
    // the context holds raw pointers to the adapters it was given: the other
    // way round frees an adapter the context may still touch while it drains
    // its queued work.
    let mut lora_adapters: Vec<LlamaLoraAdapter> = Vec::new();

    // A ranker needs the rank pooling type; without it the head never reaches the
    // graph and every score would be identical. Encoders normally declare their
    // own pooling in the GGUF, so that is left alone unless it turns out to be
    // `none`, which would make `embeddings_seq_ith` fail.
    let pooling = match class {
        gguf_probe::ModelClass::Rerank => Some(LlamaPoolingType::Rank),
        _ => None,
    };
    let mut ctx = match model.new_context(backend, build_ctx(pooling)) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready.send(LoadEvent::Error(format!("context: {e:?}")));
            return;
        }
    };
    if class == gguf_probe::ModelClass::Embedding && ctx.pooling_type() == LlamaPoolingType::None {
        let _ = ready.send(LoadEvent::Phase(
            "model declares no pooling type; falling back to mean".into(),
        ));
        drop(ctx);
        match model.new_context(backend, build_ctx(Some(LlamaPoolingType::Mean))) {
            Ok(c) => ctx = c,
            Err(e) => {
                let _ = ready.send(LoadEvent::Error(format!("context: {e:?}")));
                return;
            }
        }
    }
    ctx.set_warmup(false);

    // MTP drafts with a second context over the same weights, so it is built
    // here rather than per request: it costs a second KV cache, and rebuilding
    // it per request would throw that allocation away every time.
    //
    // Both sides need recurrent-state snapshots sized to the draft length, or
    // a rejected draft cannot be rolled back and verification cannot commit it.
    let mut mtp_draft_ctx = if mtp_n_max == 0 {
        None
    } else {
        if class != gguf_probe::ModelClass::Llm {
            let _ = ready.send(LoadEvent::Error(
                "MTP speculative decoding needs a chat model (class llm)".into(),
            ));
            return;
        }
        if !probed.as_ref().is_some_and(gguf_probe::Probe::has_mtp) {
            let _ = ready.send(LoadEvent::Error(format!(
                "{} ships no MTP head, so there is nothing to draft with",
                cfg.model_path.display()
            )));
            return;
        }
        let n_rs = (mtp_n_max as i32).max(4) as u32;
        let need = mtp_n_max + 1;
        if cfg.n_batch < need || cfg.n_ubatch < need {
            let _ = ready.send(LoadEvent::Error(format!(
                "MTP with n_max={mtp_n_max} needs --n-batch and --n-ubatch of at least {need}"
            )));
            return;
        }

        let _ = ready.send(LoadEvent::Phase(format!(
            "creating MTP draft context (n_max={mtp_n_max})"
        )));
        let mut draft_params = build_ctx(None)
            .with_ctx_type(LlamaContextType::Mtp)
            .with_n_rs_seq(n_rs)
            .with_n_seq_max(1);
        draft_params = draft_params.with_n_ctx(NonZeroU32::new(cfg.n_ctx));
        match model.new_context(backend, draft_params) {
            Ok(mut c) => {
                c.set_warmup(false);
                Some(c)
            }
            Err(e) => {
                let _ = ready.send(LoadEvent::Error(format!("MTP draft context: {e:?}")));
                return;
            }
        }
    };

    // A separately published draft model is its own set of weights, so it gets
    // its own model and context — built here rather than per request, because a
    // second KV cache is not something to rebuild for every reply.
    let draft_kind = cfg.draft_model.as_deref().map(draft_kind);
    // Set when a requested drafter cannot be brought up. The model still loads;
    // this is reported so the caller knows drafting is off and why.
    let mut draft_error: Option<String> = None;
    let draft_model = match cfg.draft_model.as_ref() {
        Some(path) => {
            if class != gguf_probe::ModelClass::Llm {
                let _ = ready.send(LoadEvent::Error(
                    "a separate draft model needs a chat model (class llm) to draft for".into(),
                ));
                return;
            }
            if mtp_n_max > 0 {
                let _ = ready.send(LoadEvent::Error(
                    "MTP and a separate draft model are two ways to draft; load one or the \
                     other, not both"
                        .into(),
                ));
                return;
            }
            let _ = ready.send(LoadEvent::Phase("loading draft model".into()));
            // An MTP checkpoint ships its head as ordinary tensors that llama.cpp
            // skips unless asked for them, so asking is the difference between a
            // draft head and a file full of weights nothing points at.
            // Same placement and read-strategy questions as the main model: a
            // draft on the wrong GPU is worse than no draft, and mmap/mlock/dio
            // mean just as much for a second multi-gigabyte file.
            let mut params = LlamaModelParams::default()
                .with_n_gpu_layers(cfg.n_gpu_layers)
                .with_main_gpu(cfg.main_gpu)
                .with_load_mode(cfg.load_mode.into())
                .with_use_mlock(cfg.use_mlock);
            if draft_kind == Some(DraftKind::Mtp) {
                params = params.with_load_mtp(true);
            }
            match LlamaModel::load_from_file(backend, path, &params) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(error = ?e, "draft model failed to load; loading without it");
                    draft_error = Some(format!(
                        "the draft model could not be loaded, so speculative decoding is off ({e})"
                    ));
                    None
                }
            }
        }
        None => None,
    };
    let mut draft_ctx = match draft_model.as_ref() {
        Some(dm) => {
            let _ = ready.send(LoadEvent::Phase("creating draft context".into()));
            // An ordinary context rather than `LlamaContextType::Mtp` — unless the
            // checkpoint *is* an MTP head. That one is a model whose trunk was
            // stripped away: nothing but embeddings, the last block and the
            // next-token layer, which only the MTP graph knows how to walk. Built
            // as a normal context it crashes inside graph reserve, because every
            // trunk layer it reaches for is absent.
            //
            // A `DFlash` checkpoint may ship without its own embedding and output
            // heads — it borrows the target's through `ctx_other` — so the paired
            // context has to be named here or llama.cpp refuses to build the
            // graph at all.
            let draft_rs = if dm.is_recurrent() || dm.is_hybrid() {
                rs_seq
            } else {
                0
            };
            let mut params = build_ctx(None)
                .with_n_ctx(NonZeroU32::new(cfg.n_ctx))
                .with_n_rs_seq(draft_rs)
                .with_ctx_other(&ctx);
            if draft_kind == Some(DraftKind::Mtp) {
                params = params
                    .with_ctx_type(LlamaContextType::Mtp)
                    .with_n_seq_max(1);
            }
            match dm.new_context(backend, params) {
                Ok(mut c) => {
                    c.set_warmup(false);
                    Some(c)
                }
                Err(e) => {
                    tracing::warn!(error = ?e, "draft context failed; loading without a drafter");
                    draft_error = Some(format!(
                        "the draft model loaded but its context could not be created, so \
                         speculative decoding is off ({e:?})"
                    ));
                    None
                }
            }
        }
        None => None,
    };

    // The context keeps a *list* of adapters, and the binding's setter replaces
    // that whole list with the single adapter it is handed — so looping over
    // the list used to leave only the last one applied, with nothing to show
    // for it. Saying so is better than silently answering with one adapter.
    if cfg.lora.len() > 1 {
        let _ = ready.send(LoadEvent::Error(
            "only one LoRA adapter can be applied at a time; this build cannot set \
             several at once"
                .into(),
        ));
        return;
    }
    for spec in &cfg.lora {
        let mut adapter = match model.lora_adapter_init(&spec.path) {
            Ok(a) => a,
            Err(e) => {
                let _ = ready.send(LoadEvent::Error(format!(
                    "lora {}: {e:?}",
                    spec.path.display()
                )));
                return;
            }
        };
        if let Err(e) = ctx.lora_adapter_set(&mut adapter, spec.scale) {
            let _ = ready.send(LoadEvent::Error(format!("lora set: {e:?}")));
            return;
        }
        lora_adapters.push(adapter);
    }

    let mut cache = DraftCache::load(cfg.ngram_cache.clone());

    // Prefix reuse. A drafting turn keeps its own draft-side cache alongside the
    // target's, and the two have to move together: `PromptCache` therefore
    // remembers both halves of a drafting snapshot and only ever hands a
    // drafting snapshot back to a drafting turn (and a plain one to a plain
    // turn). Restoring one without the other is what used to make this whole
    // feature switch off the moment a drafter was loaded.
    let mut prefix = if cfg.prefix_cache {
        PromptCache::new(
            cfg.prefix_cache_entries.max(1) as usize,
            cfg.prefix_cache_max_mb as usize * 1024 * 1024,
        )
    } else {
        PromptCache::disabled()
    };

    // The ids currently resident in the target cache, reply included. The next
    // turn extends this sequence rather than rebuilding it whenever its prompt
    // begins with exactly these — which is the shape every chat turn takes. It
    // is the cheap half of prefix reuse: nothing is copied to keep it, which is
    // what makes it work on a hybrid model whose serialised state is enormous.
    // `live_draft` says whether the draft cache holds those same positions too,
    // so a drafting turn never resumes a prefix a plain turn left behind.
    let mut live: Vec<i32> = Vec::new();
    let mut live_draft = false;

    let mut spec = if cfg.adaptive_draft {
        SpecGate::enabled(cfg.accept_floor as f64)
    } else {
        SpecGate::open()
    };
    // Latched off the first time the KV cache refuses to drop rejected draft
    // tokens (hybrid models), so only one request pays for finding out.
    let drafting = Arc::new(AtomicBool::new(true));

    // The projector has to be built against the already-loaded text model, and it
    // is optional: without one the model simply cannot see images.
    let vision = match &cfg.mmproj {
        Some(path) => {
            let params = MtmdContextParams::default()
                .with_flash_attn_type(match cfg.flash_attn {
                    FlashAttn::Auto => LlamaFlashAttnType::Auto,
                    FlashAttn::On => LlamaFlashAttnType::Enabled,
                    FlashAttn::Off => LlamaFlashAttnType::Disabled,
                })
                .print_timings(false)
                .warmup(false);
            match MtmdContext::init_from_file(path, &model, params) {
                Ok(ctx) if !ctx.supports_vision() => {
                    let _ = ready.send(LoadEvent::Error(format!(
                        "{} has no vision projector (audio/video only)",
                        path.display()
                    )));
                    return;
                }
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    let _ = ready.send(LoadEvent::Error(format!("mmproj {}: {e}", path.display())));
                    return;
                }
            }
        }
        None => None,
    };

    // llama.cpp builds contexts with warmup disabled and leaves the run to the
    // caller (`llama_set_warmup` is deprecated). Without it the first real
    // decode pays for CUDA graph capture and scratch-buffer setup — the "first
    // message hangs, later ones are instant" a fresh load shows. Warm here, on a
    // throwaway batch, so that cost lands on Load rather than the first reply.
    //
    // Generation models only. An encoder has no decode path at all — asking one
    // to decode is what turned a load of a 0.6 GB embedding model into an
    // assertion failure that took the whole process with it, which the caller
    // saw as the service disappearing rather than as a bad warmup.
    if class == gguf_probe::ModelClass::Llm {
        let _ = ready.send(LoadEvent::Phase("warming up".into()));
        if let Err(e) = warmup_context(&mut ctx, &model, cfg.n_batch.max(1) as usize) {
            tracing::warn!("warmup decode failed, continuing without it: {e}");
        }
    }
    // The draft and verification graphs are shapes of their own, captured on
    // first use. One throwaway generation runs the exact path the first real
    // reply will take, so a speculative model warms it too — with its own cache,
    // prefix cache and gate, so the warmup leaves no trace in the real ones.
    if class == gguf_probe::ModelClass::Llm && (mtp_n_max > 0 || draft_ctx.is_some()) {
        let mut warm_cache = DraftCache::load(None);
        let warm_drafting = AtomicBool::new(true);
        let mut warm_prefix = PromptCache::disabled();
        let mut warm_live: Vec<i32> = Vec::new();
        let mut warm_live_draft = false;
        let mut warm_spec = SpecGate::open();
        let (reply, _rx) = mpsc::channel();
        let msgs = [ChatMsg::new("user", "hi")];
        let params = SamplerParams {
            thinking: Some(false),
            speculative: true,
            max_tokens: 1,
            ..SamplerParams::default()
        };
        if let Err(e) = run_generate(
            &model,
            &mut ctx,
            &vision,
            templates.as_ref(),
            &msgs,
            &params,
            &reply,
            &mut warm_cache,
            &warm_drafting,
            &mut warm_prefix,
            &mut warm_live,
            &mut warm_live_draft,
            &mut warm_spec,
            &mut mtp_draft_ctx,
            mtp_n_max,
            &mut draft_ctx,
            draft_kind,
            draft_n_max,
        ) {
            tracing::warn!("warmup generation failed, continuing without it: {e}");
        }
    }

    let info = EngineInfo {
        model_path: Some(cfg.model_path.clone()),
        description: model.desc(256).unwrap_or_default(),
        n_ctx: ctx.n_ctx(),
        n_ctx_train: model.n_ctx_train(),
        n_layer: model.n_layer(),
        n_vocab: model.n_vocab(),
        n_params: model.n_params(),
        model_size: model.model_size(),
        ftype: model.ftype().map(|f| format!("{f:?}")),
        vision: vision.is_some(),
        mmproj_path: cfg.mmproj.clone(),
        model_class: class.as_str().to_string(),
        // What an embedding or rank call actually returns: a pooled embedding
        // is `n_embd_out` wide and a rank head `n_cls_out`, so reporting the
        // model's internal `n_embd` would disagree with the vectors served.
        n_embd: sequence_floats(&model, ctx.pooling_type()) as i32,
        pooling: format!("{:?}", ctx.pooling_type()).to_ascii_lowercase(),
        mtp: probed.as_ref().is_some_and(gguf_probe::Probe::has_mtp),
        draft_error,
    };
    let _ = ready.send(LoadEvent::Ready(info));

    // The class decided how this context was built, so it also decides what the
    // context may be asked for. Without the check, embedding a ranker reads a
    // one-float buffer as if it were `n_embd` floats wide, and generating from
    // an encoder samples logits that were never computed.
    while let Ok(cmd) = rx.recv() {
        // Probes and shutdown are handled here, outside the catch: neither can
        // panic worth catching, and `break` cannot be reached from inside the
        // closure anyway.
        if matches!(cmd, Command::Ping) {
            continue;
        }
        if matches!(cmd, Command::Shutdown) {
            break;
        }

        // A panic anywhere below — a sampler handed a context that cannot
        // sample, a shape llama.cpp did not expect — used to take this thread
        // down silently. The model stayed in the daemon's map, every later
        // request failed with "worker gone", and nothing said when or why.
        // Catching it keeps the thread alive and reports the failure instead.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match cmd {
            Command::Ping => {}
            Command::Generate {
                msgs,
                params,
                reply,
            } => {
                if class != gguf_probe::ModelClass::Llm {
                    let _ = reply.send(GenEvent::Error(InferError::Unsupported {
                        class: class.as_str().to_string(),
                        what: "generate text; load a chat model (class llm)".into(),
                    }));
                } else if let Err(e) = params.validate() {
                    let _ = reply.send(GenEvent::Error(e));
                } else if let Err(e) = run_generate(
                    &model,
                    &mut ctx,
                    &vision,
                    templates.as_ref(),
                    &msgs,
                    &params,
                    &reply,
                    &mut cache,
                    &drafting,
                    &mut prefix,
                    &mut live,
                    &mut live_draft,
                    &mut spec,
                    &mut mtp_draft_ctx,
                    mtp_n_max,
                    &mut draft_ctx,
                    draft_kind,
                    draft_n_max,
                ) {
                    let _ = reply.send(GenEvent::Error(e));
                }
            }
            Command::Embed { texts, reply } => {
                let result = if class != gguf_probe::ModelClass::Embedding {
                    Err(InferError::Unsupported {
                        class: class.as_str().to_string(),
                        what: "produce embeddings; load an embedding model".into(),
                    })
                } else {
                    run_embed(&model, &mut ctx, &texts)
                };
                let _ = reply.send(result);
            }
            Command::Rerank {
                query,
                documents,
                reply,
            } => {
                let result = if class != gguf_probe::ModelClass::Rerank {
                    Err(InferError::Unsupported {
                        class: class.as_str().to_string(),
                        what: "score documents; load a rerank model".into(),
                    })
                } else {
                    run_rerank(&model, &mut ctx, &query, &documents)
                };
                let _ = reply.send(result);
            }
            Command::Shutdown => {}
        }));

        if let Err(payload) = outcome {
            tracing::error!("inference worker panicked: {}", panic_message(&payload));
        }
    }
}

/// The message a panic payload carries, if it has one.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "no message".to_string()
}

/// How many floats one sequence's output buffer actually holds.
///
/// llama.cpp sizes that buffer from the pooling type, not from `n_embd`: a
/// pooled embedding gets `n_embd_out` floats, while a rank head gets
/// `n_cls_out` — a single float for bge-reranker. Taking the width from the
/// wrong branch walks off the end of the allocation, so it is derived from the
/// pooling type here instead of being passed in.
fn sequence_floats(model: &LlamaModel, pooling: LlamaPoolingType) -> usize {
    let n = match pooling {
        LlamaPoolingType::Rank => model.n_cls_out() as usize,
        _ => usize::try_from(model.n_embd_out()).unwrap_or(0),
    };
    n.max(1)
}

/// Read sequence 0's pooled output, sized for this context's pooling type.
///
/// Not `LlamaContext::embeddings_seq_ith`: that slices `n_embd` floats, which
/// matches neither branch above. On a rank head it reads hundreds of floats
/// past a one-float buffer.
fn sequence_output(ctx: &LlamaContext, model: &LlamaModel) -> Result<Vec<f32>> {
    let pooling = ctx.pooling_type();
    if pooling == LlamaPoolingType::None {
        return Err(InferError::Embedding(
            "this context has no pooling type, so it produces no sequence embeddings".into(),
        ));
    }
    let n = sequence_floats(model, pooling);
    let ptr = unsafe { llama_cpp_sys_4::llama_get_embeddings_seq(ctx.as_ptr(), 0) };
    if ptr.is_null() {
        return Err(InferError::Embedding(
            "no sequence embeddings in this context".into(),
        ));
    }
    // SAFETY: `ptr` belongs to the context that was just decoded, and `n` is
    // the width llama.cpp allocated for this pooling type.
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
}

/// Scale a vector to unit length, leaving a zero vector alone.
fn normalize(v: &mut [f32]) {
    let sum: f32 = v.iter().map(|x| x * x).sum();
    if sum > 0.0 && sum.is_finite() {
        let inv = sum.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Embed a batch of texts, one forward pass each.
///
/// The texts are embedded one at a time rather than as parallel sequences: that
/// would need `n_seq_max` on the context and a batch laid out per sequence, and
/// the extra bookkeeping buys nothing for the request sizes this serves.
fn run_embed(model: &LlamaModel, ctx: &mut LlamaContext, texts: &[String]) -> Result<EmbedResult> {
    // A non-causal encoder needs every token in one *physical* batch, and
    // llama.cpp enforces that with `GGML_ASSERT(n_ubatch >= n_tokens)` rather
    // than an error — an assert there aborts the process and every model in
    // it. With the 2048/512 defaults, an input of 513..2048 tokens clears the
    // Rust batch's capacity check and then trips the assert, so the batch is
    // sized to `n_ubatch` and the limit is reported up front instead.
    let n_batch = ctx.n_batch().min(ctx.n_ubatch()).max(1) as usize;
    let n_ctx = (ctx.n_ctx() as usize).min(n_batch);
    let mut vectors = Vec::with_capacity(texts.len());
    let mut tokens_total = 0usize;

    for text in texts {
        let tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| InferError::Tokenize(e.to_string()))?;
        if tokens.is_empty() {
            vectors.push(Vec::new());
            continue;
        }
        // Nothing is generated here, so the input may fill the context exactly.
        if tokens.len() > n_ctx {
            return Err(InferError::ContextFull {
                used: tokens.len(),
                n_ctx,
            });
        }
        tokens_total += tokens.len();

        ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(n_batch, 1);
        for (i, &tok) in tokens.iter().enumerate() {
            // Logits are only needed on the last token; pooling reads the
            // sequence state, not the per-token outputs.
            batch
                .add(tok, i as i32, &[0], i + 1 == tokens.len())
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch).map_err(|e| {
            InferError::Decode(format!(
                "embedding decode (n_tokens={}): {e}",
                batch.n_tokens()
            ))
        })?;

        let mut vector = sequence_output(ctx, model)?;
        // Unit length, which is what the OpenAI-shaped API and llama.cpp's own
        // server return. Raw pooled output makes a dot product depend on how
        // long the text was, so two documents would compare differently than
        // any client doing cosine similarity expects.
        normalize(&mut vector);
        vectors.push(vector);
    }

    Ok(EmbedResult {
        vectors,
        tokens: tokens_total,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_generate<'m, 'd>(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'m>,
    vision: &Option<MtmdContext>,
    templates: Option<&ChatTemplates>,
    msgs: &[ChatMsg],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    cache: &mut DraftCache,
    drafting: &AtomicBool,
    prefix: &mut PromptCache,
    live: &mut Vec<i32>,
    live_draft: &mut bool,
    spec: &mut SpecGate,
    mtp_draft_ctx: &mut Option<LlamaContext<'m>>,
    // (see `run_generate`)
    mtp_n_max: u32,
    draft_ctx: &mut Option<LlamaContext<'d>>,
    draft_kind: Option<DraftKind>,
    draft_n_max: u32,
) -> Result<()> {
    let n_ctx = ctx.n_ctx() as usize;
    let n_batch = ctx.n_batch().max(1) as usize;

    // The reply text is accumulated thread-locally; start this turn clean.
    reset_emitted();

    // The draft heads work from the target's hidden states, and an image has
    // none of those. Rather than fail the request somebody attached a picture
    // to, a conversation with images takes the plain path and simply does not
    // draft this turn — the same answer, one pass per token.
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();

    // Whether to draft at all. A gate that has measured drafting as not paying
    // for itself closes here, before any draft context is touched.
    let speculate = params.speculative && spec.allows();

    // Which drafter this turn would actually use, resolved before the prefix
    // lookup: a drafting turn has to put the draft side of a snapshot back too,
    // so it may only match a snapshot that carries one.
    let mtp_path = speculate && mtp_n_max > 0 && images == 0 && mtp_draft_ctx.is_some();
    let side_path =
        speculate && !mtp_path && images == 0 && draft_kind.is_some() && draft_ctx.is_some();
    let need_draft = mtp_path || side_path;

    // A DFlash/EAGLE side drafter keeps continuation state this binding cannot
    // serialise, so a turn that would use one never resumes a prefix — it clears
    // and encodes whole, exactly as it did before any of this existed.
    let reuse_capable = !(side_path && draft_kind != Some(DraftKind::Mtp));

    // Prefix reuse, in the shape that actually pays on a hybrid model. The usual
    // prompt is the previous turn's prompt-plus-reply with one user message
    // appended, and all of it is still in the cache — so leaving it there and
    // encoding only the tail is the entire saving. Serialising that cache to
    // copy it back costs *more* than re-encoding, because the recurrent state is
    // the bulk of it, so the snapshot below is a fallback for a diverged prompt
    // only and is paid for only when one occurs.
    let had_live = !live.is_empty();
    let mut start = 0usize;
    let mut reuse = false;
    let mut save_snapshot = false;
    if prefix.is_enabled() && images == 0 && reuse_capable {
        if let Ok(tokens) = tokenize_text_prompt(templates, model, msgs, params) {
            let ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();
            if !live.is_empty()
                && live.len() < ids.len()
                && ids[..live.len()] == live[..]
                && (!need_draft || *live_draft)
            {
                // A strict extension of what is resident: keep the cache and
                // encode only the suffix. An exact match is refused — the last
                // token has to be decoded again for its logits, and re-decoding
                // a position the cache already covers is the one arrangement
                // here that has not been shown to be safe.
                start = live.len();
                reuse = true;
            } else if let Some(len) = prefix.longest_prefix(&ids, need_draft) {
                if len < ids.len() {
                    // A restore needs an empty cache to land in. If it does not
                    // take, `reuse` stays false and the prompt is encoded whole
                    // below — half a restore is worse than none.
                    ctx.clear_kv_cache();
                    if let Some(entry) = prefix.snapshot_for(len) {
                        if unsafe { restore_state(ctx, &entry.state) } {
                            // The draft side has to come back with the target's,
                            // or the head would sit behind the target and reject
                            // every proposal. A snapshot with no draft state is
                            // simply not used on a drafting turn.
                            let draft_ok = match entry.draft.as_deref() {
                                None => true,
                                Some(bytes) if mtp_path => mtp_draft_ctx
                                    .as_mut()
                                    .is_some_and(|c| unsafe { restore_state(c, bytes) }),
                                Some(bytes) if side_path => draft_ctx
                                    .as_mut()
                                    .is_some_and(|c| unsafe { restore_state(c, bytes) }),
                                Some(_) => false,
                            };
                            if draft_ok {
                                start = len;
                                reuse = true;
                            }
                        }
                    }
                }
            }
        }
        // A prompt that diverged from a conversation already in progress is the
        // one case a snapshot can still help later (an edit, a branch, a
        // regenerated answer), so it is saved here and nowhere else.
        save_snapshot = !reuse && had_live;
    }
    if !reuse {
        ctx.clear_kv_cache();
    }

    // MTP prefills through its own session, because the draft head needs the
    // target's hidden state for *every* prompt token and only the session can
    // hand that over. Prefilling here first would leave the head with nothing
    // to draft from.
    if mtp_path {
        let mut resident: Vec<i32> = Vec::new();
        let stats = {
            let draft = mtp_draft_ctx
                .as_mut()
                .expect("mtp_path implies a draft context");
            generate_mtp(
                model,
                ctx,
                draft,
                vision,
                templates,
                msgs,
                params,
                reply,
                n_ctx,
                spec.draft_len(mtp_n_max) as i32,
                start,
                &mut resident,
            )?
        };
        // Both caches now hold prompt plus reply at the same positions, so the
        // ids are what the next turn continues from. Only a later diverged turn
        // pays for a serialised copy, and only because this one was mid-thread.
        if save_snapshot {
            let draft_state = mtp_draft_ctx
                .as_ref()
                .and_then(|c| unsafe { save_state(c) });
            if let Some(state) = unsafe { save_state(ctx) } {
                prefix.insert(resident.clone(), state, draft_state);
            }
        }
        if prefix.is_enabled() && images == 0 {
            *live = resident;
            *live_draft = true;
        } else {
            live.clear();
            *live_draft = false;
        }
        spec.record(&stats);
        let mut stats = stats;
        let parsed = finish_reply(reply);
        stats.tool_calls = parsed.tool_calls;
        stats.parsed_content = parsed.content;
        stats.reasoning_content = parsed.reasoning;
        let _ = reply.send(GenEvent::Done(stats));
        return Ok(());
    }

    // A separate draft model outranks the n-gram drafter as well: it proposes
    // from learned weights rather than from repeats in this conversation, so
    // when one is loaded it is the better guess.
    if side_path {
        let kind = draft_kind.expect("side_path implies a draft kind");
        let mut resident: Vec<i32> = Vec::new();
        let stats = {
            let draft = draft_ctx
                .as_mut()
                .expect("side_path implies a draft context");
            if kind == DraftKind::Mtp {
                // An MTP head drafted from a side-mounted checkpoint is still an
                // MTP head — same loop, same session, only the weights it reads
                // are a second file rather than the target's own.
                //
                // `MtpSession` is typed as though both contexts came out of one
                // model, which describes the bundled case and nothing else.
                // Upstream pairs a target with a separate MTP checkpoint just as
                // happily — `--model-draft` alongside `--spec-type draft-mtp` —
                // because the session only ever hands raw context pointers to C
                // and never assumes the two share weights. The draft model
                // loaded here outlives the session this borrow feeds, so the
                // only thing being widened is the binding's own type.
                //
                // SAFETY: the draft context's data outlives this borrow, so the
                // lifetime extension is sound; no other use of `draft` follows.
                let side: &mut LlamaContext<'m> = unsafe {
                    std::mem::transmute::<&mut LlamaContext<'d>, &mut LlamaContext<'m>>(draft)
                };
                generate_mtp(
                    model,
                    ctx,
                    side,
                    vision,
                    templates,
                    msgs,
                    params,
                    reply,
                    n_ctx,
                    spec.draft_len(draft_n_max) as i32,
                    start,
                    &mut resident,
                )?
            } else {
                generate_draft_model(
                    model,
                    ctx,
                    draft,
                    kind,
                    templates,
                    msgs,
                    params,
                    reply,
                    n_ctx,
                    spec.draft_len(draft_n_max) as i32,
                )?
            }
        };
        // A DFlash/EAGLE checkpoint keeps continuation state this binding does
        // not serialise, so those turns leave no snapshot and simply re-encode
        // next time; a side-mounted MTP head moves in step with the target, so
        // its ids are kept for the next turn to continue from.
        if kind == DraftKind::Mtp && save_snapshot {
            let draft_state = draft_ctx.as_ref().and_then(|c| unsafe { save_state(c) });
            if let Some(state) = unsafe { save_state(ctx) } {
                prefix.insert(resident.clone(), state, draft_state);
            }
        }
        if kind == DraftKind::Mtp && prefix.is_enabled() && images == 0 {
            *live = resident;
            *live_draft = true;
        } else {
            // DFlash/EAGLE leaves the cache holding prompt plus reply but no
            // serialisable continuation, so nothing is resumed from it.
            live.clear();
            *live_draft = false;
        }
        spec.record(&stats);
        let mut stats = stats;
        let parsed = finish_reply(reply);
        stats.tool_calls = parsed.tool_calls;
        stats.parsed_content = parsed.content;
        stats.reasoning_content = parsed.reasoning;
        let _ = reply.send(GenEvent::Done(stats));
        return Ok(());
    }

    let t_start = Instant::now();
    let (prompt_tokens, n_past, hist) = fill_prompt(
        model, ctx, vision, templates, params, msgs, n_ctx, n_batch, start,
    )?;
    let prompt_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // The generators report back the exact ids they left in the cache, reply
    // included, so the next turn can skip re-encoding the answer to the previous
    // question as well as the conversation before it.
    let mut resident: Vec<i32> = hist.iter().map(|t| t.0).collect();

    let mut stats = if speculate {
        generate_speculative(
            model,
            ctx,
            &hist,
            params,
            reply,
            n_ctx,
            n_past,
            prompt_tokens,
            prompt_ms,
            cache,
            drafting,
            &mut resident,
        )?
    } else {
        generate_plain(
            model,
            ctx,
            &hist,
            params,
            reply,
            n_ctx,
            n_past,
            prompt_tokens,
            prompt_ms,
            &mut resident,
        )?
    };

    // Remember what is in the cache now, once the reply is part of it: the ids
    // are the cheap continuation, the serialised snapshot only the fallback.
    if save_snapshot {
        if let Some(state) = unsafe { save_state(ctx) } {
            prefix.insert(resident.clone(), state, None);
        }
    }
    if prefix.is_enabled() && images == 0 {
        *live = resident;
        *live_draft = false;
    } else {
        live.clear();
        *live_draft = false;
    }
    let parsed = finish_reply(reply);
    stats.tool_calls = parsed.tool_calls;
    stats.parsed_content = parsed.content;
    stats.reasoning_content = parsed.reasoning;
    let _ = reply.send(GenEvent::Done(stats));
    Ok(())
}

/// Serialise sequence 0's KV state, so a later prompt can skip re-encoding it.
///
/// `llama_state_seq_get_data` covers one sequence's cache — not the whole
/// context — which is exactly the granularity a prompt prefix needs.
///
/// # Safety
///
/// `ctx` must be a live context that has just finished encoding.
unsafe fn save_state(ctx: &LlamaContext) -> Option<Vec<u8>> {
    let ptr = ctx.as_ptr();
    let size = llama_cpp_sys_4::llama_state_seq_get_size(ptr, 0);
    if size == 0 {
        return None;
    }
    let mut buf = vec![0u8; size];
    let wrote = llama_cpp_sys_4::llama_state_seq_get_data(ptr, buf.as_mut_ptr(), size, 0);
    // A short read means the state changed under us; a partial buffer would
    // restore a cache that is silently wrong, so it is refused outright.
    (wrote == size).then_some(buf)
}

/// Put a saved state back, so encoding can resume part-way through a prompt.
///
/// # Safety
///
/// `ctx` must be a live context whose cache has just been cleared, and `state`
/// must have come from [`save_state`] on a context with the same shape.
unsafe fn restore_state(ctx: &mut LlamaContext, state: &[u8]) -> bool {
    let read =
        llama_cpp_sys_4::llama_state_seq_set_data(ctx.as_ptr(), state.as_ptr(), state.len(), 0);
    read == state.len()
}

/// Decode one throwaway batch, then drop its KV state.
///
/// Current llama.cpp sets up graphs and scratch buffers per shape on the first
/// decode of that shape; doing one here means the first real request reuses that
/// work rather than paying for it. Best-effort: a warmup that fails leaves the
/// model loaded and merely slow on the first reply — which is where it started.
fn warmup_context(ctx: &mut LlamaContext, model: &LlamaModel, n_batch: usize) -> Result<()> {
    // Never past the context: a warmup that asked for more rows than the cache
    // holds would fail, and it is the cache shape we are trying to warm anyway.
    let tokens = n_batch.clamp(1, ctx.n_ctx().max(1) as usize);
    let mut batch = LlamaBatch::new(tokens, 1);
    let bos = model.token_bos();
    for i in 0..tokens {
        // The values are discarded; logits on the last row keep the batch shape
        // the decode path expects.
        let logits = i + 1 == tokens;
        batch
            .add(bos, i as i32, &[0], logits)
            .map_err(|e| InferError::Decode(e.to_string()))?;
    }
    let decoded = ctx.decode(&mut batch);
    // Whatever happened, leave no warmup tokens behind.
    ctx.clear_kv_cache();
    decoded.map_err(|e| InferError::Decode(e.to_string()))
}

/// Encode the conversation into the KV cache and report where it ended.
///
/// Returns the prompt length in tokens, the new `n_past`, and the token sequence
/// itself — the generators need the history to seed the n-gram drafters and to
/// learn from.
///
/// Text-only conversations take the plain batching path. As soon as an image is
/// present the prompt goes through libmtmd instead, because the tokens standing
/// in for a picture are image *embeddings*, which a token batch cannot carry.
#[allow(clippy::too_many_arguments)]
fn fill_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    vision: &Option<MtmdContext>,
    templates: Option<&ChatTemplates>,
    params: &SamplerParams,
    msgs: &[ChatMsg],
    n_ctx: usize,
    n_batch: usize,
    start: usize,
) -> Result<(usize, i32, Vec<LlamaToken>)> {
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();
    let Some(vision) = vision.as_ref().filter(|_| images > 0) else {
        if images > 0 {
            return Err(InferError::VisionUnavailable { images });
        }
        return fill_text_prompt(model, ctx, templates, params, msgs, n_ctx, n_batch, start);
    };
    // An image prompt takes the mtmd path, which owns its own encoding and has
    // no token prefix to match against, so any cached offset is ignored here.
    // It also keeps the legacy template path: libmtmd renders the conversation
    // itself, and reasoning control is a text-only concern for now.

    let prompt = render_prompt(model, msgs, vision)?;

    let mut bitmaps = Vec::with_capacity(images);
    for msg in msgs {
        for data in &msg.images {
            bitmaps.push(
                MtmdBitmap::from_buf(vision, data)
                    .map_err(|e| InferError::Vision(format!("image: {e}")))?,
            );
        }
    }
    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

    let text = MtmdInputText::new(&prompt, true, true);
    let mut chunks = MtmdInputChunks::new();
    vision
        .tokenize(&text, &bitmap_refs, &mut chunks)
        .map_err(|e| InferError::Vision(format!("tokenize prompt: {e}")))?;

    let mut n_past = 0i32;
    vision
        .eval_chunks(
            ctx.as_ptr(),
            &chunks,
            0,
            0,
            n_batch as i32,
            true,
            &mut n_past,
        )
        .map_err(|e| InferError::Vision(format!("encode image: {e}")))?;

    if n_past as usize >= n_ctx {
        return Err(InferError::ContextFull {
            used: n_past as usize,
            n_ctx,
        });
    }

    // The prompt is already in the cache, so the generators only need its tokens
    // for history and drafting. Image positions have no token id to recover, so
    // this reports an empty history: the n-gram drafters stay switched off for
    // image prompts rather than working from a sequence that does not match the
    // one in the cache. Verification makes that a performance choice, not a
    // correctness one.
    Ok((n_past as usize, n_past, Vec::new()))
}

/// Render and tokenize a text-only conversation.
///
/// Split out because the prefix cache needs the token ids *before* anything is
/// encoded, and encoding is precisely what it exists to avoid.
fn tokenize_text_prompt(
    templates: Option<&ChatTemplates>,
    model: &LlamaModel,
    msgs: &[ChatMsg],
    params: &SamplerParams,
) -> Result<Vec<LlamaToken>> {
    let prompt = render_chat(templates, model, msgs, params)?;

    model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| InferError::Tokenize(e.to_string()))
}

/// One message as the OpenAI shape `ChatApplyParams` expects, carrying the
/// tool-call fields when the turn is part of a tool exchange.
fn message_json(m: &ChatMsg) -> serde_json::Value {
    let mut v = serde_json::json!({ "role": m.role, "content": m.content });
    if let Some(calls) = &m.tool_calls {
        v["tool_calls"] = calls.clone();
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = serde_json::json!(id);
    }
    v
}

/// Render a conversation to a prompt string.
///
/// Takes the Jinja template path when the caller names a thinking mode or
/// declares tools — the only two things the legacy apply path cannot express.
/// Otherwise the legacy path is used unchanged, so a request that says nothing
/// about thinking renders exactly as it always did. A template that does not
/// understand a flag ignores it.
fn render_chat(
    templates: Option<&ChatTemplates>,
    model: &LlamaModel,
    msgs: &[ChatMsg],
    params: &SamplerParams,
) -> Result<String> {
    let tools = params
        .tools_json
        .as_deref()
        .filter(|t| !t.trim().is_empty());
    // Tools need the Jinja path: the legacy call has no way to hand them over,
    // so the model would never see them and the reply would never contain a
    // call. A thinking mode needs it for the same reason.
    if let Some(templates) = templates.filter(|_| params.thinking.is_some() || tools.is_some()) {
        let wire: Vec<serde_json::Value> = msgs.iter().map(message_json).collect();
        let json =
            serde_json::to_string(&wire).map_err(|e| InferError::ChatTemplate(e.to_string()))?;
        let mut apply = ChatApplyParams::new(json)
            .with_enable_thinking(params.thinking.unwrap_or(true))
            .with_add_generation_prompt(true);
        if let Some(tools) = tools {
            apply = apply
                .with_tools(tools.to_owned())
                // Parse-time setting: split thinking blocks into
                // `reasoning_content`, so the answer is not polluted by them.
                .with_reasoning_format(ReasoningFormat::Auto);
            if let Some(choice) = params
                .tool_choice
                .as_deref()
                .and_then(|c| ToolChoice::parse_oaicompat(c).ok())
            {
                apply = apply.with_tool_choice(choice);
            }
            apply = apply.with_parallel_tool_calls(params.parallel_tool_calls);
        }
        let applied = templates
            .apply(&apply)
            .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
        let prompt = applied.prompt.clone();
        // Held only when tools were declared: such a reply is not streamed token
        // by token, and this is what splits it apart once it finishes.
        PARSER.with(|p| {
            *p.borrow_mut() = if tools.is_some() {
                Some(ReplyParser { chat: applied })
            } else {
                None
            };
        });
        return Ok(prompt);
    }
    PARSER.with(|p| *p.borrow_mut() = None);

    let messages = msgs
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
    model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| InferError::ChatTemplate(e.to_string()))
}

/// The ordinary text path: tokenize, then decode in `n_batch` slices.
///
/// `start` is how many leading tokens are already in the KV cache, contributed
/// by the prefix cache. Those are skipped; the rest are decoded as usual.
#[allow(clippy::too_many_arguments)]
fn fill_text_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    templates: Option<&ChatTemplates>,
    params: &SamplerParams,
    msgs: &[ChatMsg],
    n_ctx: usize,
    n_batch: usize,
    start: usize,
) -> Result<(usize, i32, Vec<LlamaToken>)> {
    let tokens = tokenize_text_prompt(templates, model, msgs, params)?;

    if tokens.len() >= n_ctx {
        return Err(InferError::ContextFull {
            used: tokens.len(),
            n_ctx,
        });
    }

    let start = start.min(tokens.len());
    let mut batch = LlamaBatch::new(n_batch, 1);
    let total = tokens.len();
    let mut n_past: i32 = start as i32;

    for (chunk_idx, chunk) in tokens[start..].chunks(n_batch).enumerate() {
        batch.clear();
        let base = start + chunk_idx * n_batch;
        for (i, &tok) in chunk.iter().enumerate() {
            let pos = (base + i) as i32;
            let logits = base + i == total - 1;
            batch
                .add(tok, pos, &[0], logits)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        n_past = (base + chunk.len()) as i32;
    }

    Ok((total, n_past, tokens))
}

/// Apply the chat template, making sure the media marker count matches the images.
///
/// libmtmd pairs each marker with one bitmap and refuses a mismatch, but the
/// marker is just text as far as the template is concerned: some templates emit
/// one themselves, most do not. Counting after templating and topping up is more
/// robust than trying to predict which is which.
fn render_prompt(model: &LlamaModel, msgs: &[ChatMsg], vision: &MtmdContext) -> Result<String> {
    let marker = vision.marker().to_string();
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();

    // Put each marker in the message that carries the picture. Appending the
    // shortfall to the whole prompt instead used to hoist every image above the
    // system prompt and out of its own turn, which is not a conversation the
    // model was trained on — the turn the picture belongs to is the point.
    let messages = msgs
        .iter()
        .map(|m| {
            let mut content = m.content.clone();
            let have = content.matches(&marker).count();
            let want = m.image_count();
            if want > have {
                content.push_str(&marker.repeat(want - have));
            }
            LlamaChatMessage::new(m.role.clone(), content)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;

    let prompt = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;

    let present = prompt.matches(&marker).count();
    match present.cmp(&images) {
        std::cmp::Ordering::Equal => Ok(prompt),
        std::cmp::Ordering::Less => {
            // The template dropped the markers it was handed. The images still
            // have to go somewhere, and the end of the prompt at least leaves
            // every turn intact.
            let extra = marker.repeat(images - present);
            Ok(format!("{prompt}{extra}"))
        }
        std::cmp::Ordering::Greater => Err(InferError::Vision(format!(
            "the chat template emits {present} media markers but {images} images were attached"
        ))),
    }
}

/// Score each document against a query, one forward pass each.
///
/// The input is the encoder convention llama.cpp uses when no `rerank` chat
/// template is present: `BOS query SEP document EOS`. For a BERT reranker those
/// tokens are `[CLS]` and `[SEP]`, which is exactly the sentence pair the head was
/// trained on, so the scores are meaningful rather than merely ordered.
fn run_rerank(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    query: &str,
    documents: &[String],
) -> Result<RerankResult> {
    if ctx.pooling_type() != LlamaPoolingType::Rank {
        return Err(InferError::Rerank(
            "this model was not loaded with rank pooling, so it has no relevance head".into(),
        ));
    }
    // A model carrying its own rerank template needs that exact format, which is
    // not what is built below; refusing beats returning confidently wrong scores.
    if model.chat_template(Some("rerank")).is_ok() {
        return Err(InferError::Rerank(
            "this model ships a rerank chat template, which this build does not apply yet".into(),
        ));
    }

    // Same single-ubatch requirement as `run_embed`: a non-causal encoder
    // asserts on `n_ubatch < n_tokens`, so the batch is capped at `n_ubatch`.
    let n_batch = ctx.n_batch().min(ctx.n_ubatch()).max(1) as usize;
    let n_ctx = (ctx.n_ctx() as usize).min(n_batch);
    let bos = model.token_bos();
    let sep = model.token_sep();
    let eos = model.token_eos();

    let mut scores = Vec::with_capacity(documents.len());
    let mut tokens_total = 0usize;

    for document in documents {
        let mut tokens: Vec<LlamaToken> = Vec::new();
        if bos.0 >= 0 {
            tokens.push(bos);
        }
        tokens.extend(
            model
                .str_to_token(query, AddBos::Never)
                .map_err(|e| InferError::Tokenize(e.to_string()))?,
        );
        if sep.0 >= 0 {
            tokens.push(sep);
        }
        tokens.extend(
            model
                .str_to_token(document, AddBos::Never)
                .map_err(|e| InferError::Tokenize(e.to_string()))?,
        );
        if eos.0 >= 0 {
            tokens.push(eos);
        }
        // Scoring generates nothing, so the pair may fill the context exactly.
        if tokens.len() > n_ctx {
            return Err(InferError::ContextFull {
                used: tokens.len(),
                n_ctx,
            });
        }
        tokens_total += tokens.len();

        ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(n_batch, 1);
        for (i, &tok) in tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], false)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch).map_err(|e| {
            InferError::Decode(format!(
                "rerank decode (n_tokens={}): {e}",
                batch.n_tokens()
            ))
        })?;

        // A head that produced nothing is a failure, not a score: returning NaN
        // here used to surface as a successful response that sorted last, which
        // reads as a weak match rather than a broken one.
        let score = sequence_output(ctx, model)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Rerank("the rank head produced no score".into()))?;
        scores.push(score);
    }

    Ok(RerankResult {
        scores,
        tokens: tokens_total,
    })
}

/// UI exposes are tuned against.
#[allow(clippy::too_many_arguments)]
fn generate_plain(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    prompt: &[LlamaToken],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    n_ctx: usize,
    mut n_past: i32,
    prompt_tokens: usize,
    prompt_ms: f64,
    // Ids the cache holds when this returns: the prompt, then every token that
    // reached the KV. The caller keys the next turn's reusable prefix on it.
    resident: &mut Vec<i32>,
) -> Result<GenStats> {
    let n_batch = ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut sampler = build_sampler(params, model.n_vocab());

    // The repetition penalties have to see the prompt, the same way the
    // speculative path feeds it. Without this the sampler's history starts
    // empty, so a repeated phrase in the prompt is not penalised — and the two
    // paths disagree about the very first token of an identical request.
    for &tok in prompt {
        sampler.accept(tok);
    }

    let t_gen = Instant::now();
    let mut first_token_ms = 0.0f64;
    let mut pending: Vec<u8> = Vec::new();
    let mut generated = 0usize;
    // Zero means "until the model stops"; the context length is the real bound.
    let cap = params.token_cap();

    while generated < cap && (n_past as usize) < n_ctx {
        let token = sampler.sample(ctx, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        emit_token(model, token, &mut pending, reply)?;

        if generated == 0 {
            first_token_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        }
        generated += 1;

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        n_past += 1;
        ctx.decode(&mut batch)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        resident.push(token.0);
    }
    emit_tail(&mut pending, reply)?;

    Ok(GenStats {
        prompt_tokens,
        generated_tokens: generated,
        prompt_ms,
        gen_ms: t_gen.elapsed().as_secs_f64() * 1000.0,
        first_token_ms,
        hit_token_cap: params.hit_token_cap(generated),
        ..Default::default()
    })
}

/// Speculative decoding with the model's own MTP (next-token-prediction) head.
///
/// This is the same shape as the n-gram path, but the proposals come from the
/// extra layer the model was trained with rather than from a lookup, so they are
/// cheap *and* accurate: the head is one layer of a model that already predicts
/// its own continuation. Each round asks the head for up to `mtp_n_max` tokens,
/// decodes them together with the sampled token in a single target pass, and
/// keeps only the prefix the target agreed with.
///
/// Three things make this correct rather than merely fast:
///
/// * `process` is called after every target decode so the draft context's
///   recurrent state tracks the target's KV cache.
/// * `accept` reports how many drafts survived, which is what rolls the draft
///   head's recurrent state back to the committed point.
/// * the target KV cache is trimmed back to the accepted prefix. That trim is
///   why the contexts were built with `n_rs_seq` snapshots: a hybrid model's
///   recurrent state has no other way to rewind.
#[allow(clippy::too_many_arguments)]
fn generate_mtp<'m>(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'m>,
    draft_ctx: &mut LlamaContext<'m>,
    vision: &Option<MtmdContext>,
    templates: Option<&ChatTemplates>,
    msgs: &[ChatMsg],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    n_ctx: usize,
    mtp_n_max: i32,
    // First `start` prompt tokens are already in both caches, restored by the
    // caller; only the suffix is prefilled here. Zero means nothing is in yet.
    start: usize,
    // Ids both caches hold when this returns, prompt then reply.
    resident: &mut Vec<i32>,
) -> Result<GenStats> {
    let n_batch = ctx.n_batch().max(1) as usize;
    let k_max = (mtp_n_max.max(1) as usize).min(n_batch.saturating_sub(1));

    // The MTP head consumes the target's hidden states, and image tokens have
    // none, so image prompts are refused outright rather than silently
    // generating from a draft head that never saw them.
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();
    if images > 0 {
        return Err(InferError::Unsupported {
            class: "mtp".into(),
            what: format!("draft with images ({images} attached); MTP is text-only"),
        });
    }
    let _ = vision;

    // The drafter must see the same prompt the target would, thinking mode
    // included, or the two disagree about the conversation from token zero.
    let prompt = tokenize_text_prompt(templates, model, msgs, params)?;
    if prompt.len() >= n_ctx {
        return Err(InferError::ContextFull {
            used: prompt.len(),
            n_ctx,
        });
    }

    let mut cs_params = CommonSamplerParams::new();
    let mut sc = cs_params.scalars();
    sc.min_keep = 1;
    sc.temp = params.temperature;
    sc.top_k = params.top_k;
    sc.top_p = params.top_p;
    sc.min_p = params.min_p;
    sc.penalty_last_n = params.repeat_last_n;
    sc.penalty_repeat = params.repeat_penalty;
    sc.penalty_freq = params.repeat_freq;
    sc.penalty_present = params.repeat_present;
    sc.seed = params.seed;
    cs_params.set_scalars(&sc);
    let mut sampler =
        CommonSampler::new(model, &mut cs_params).map_err(|e| InferError::Decode(e.to_string()))?;
    for tok in &prompt {
        sampler.accept(*tok, false);
    }

    let config = MtpSessionConfig::new(1, mtp_n_max);
    let mut session = MtpSession::new_with_config(ctx, draft_ctx, config)
        .map_err(|e| InferError::Decode(format!("MTP session: {e}")))?;

    // A resumed request already has the head's cache in step with the target's,
    // put there by the caller, and emptying it here would throw that away. Only
    // a fresh prefill starts from an empty draft cache: the session borrows both
    // contexts, so the cache can only be emptied through it, and a request that
    // failed mid-round leaves the head's positions ahead of the target's, which
    // the next prefill would then reject.
    if start == 0 {
        session
            .clear_draft_kv_cache_seq(None, None, None)
            .map_err(|e| InferError::Decode(format!("MTP draft cache reset: {e}")))?;
    }

    // The draft head needs the whole prompt to have gone through the target, so
    // prefill goes through the session rather than straight to the context.
    // Every ubatch has to pass `process`, not just the last one: the head keeps
    // a hidden-state row per token, and a ubatch it never saw is a hole in it.
    // A resumed request encodes only the suffix — the prefix is already in both
    // caches, which is the whole point of having remembered it.
    let t_start = Instant::now();
    let mut batch = LlamaBatch::new(n_batch.max(k_max + 1), 1);
    let prompt_tokens = prompt.len();
    let total = prompt.len();
    let mut n_past = start as i32;
    resident.clear();
    resident.extend(prompt.iter().map(|t| t.0));
    for (chunk_idx, chunk) in prompt[start..].chunks(n_batch).enumerate() {
        batch.clear();
        let base = start + chunk_idx * n_batch;
        for (i, &tok) in chunk.iter().enumerate() {
            batch
                .add(tok, (base + i) as i32, &[0], base + i == total - 1)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        session.decode_target_and_process(&mut batch).map_err(|e| {
            InferError::Decode(format!(
                "MTP prefill ubatch {chunk_idx} (n_tokens={}, n_past={base}): {e}",
                chunk.len()
            ))
        })?;
        n_past = (base + chunk.len()) as i32;
    }
    let prompt_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    session
        .begin(0, &prompt)
        .map_err(|e| InferError::Decode(format!("MTP begin: {e}")))?;

    let t_gen = Instant::now();
    let mut first_token_ms = 0.0f64;
    let mut pending: Vec<u8> = Vec::new();
    let mut generated = 0usize;
    let mut draft_tokens = 0usize;
    let mut accepted_draft = 0usize;
    let mut draft_rounds = 0usize;
    // Zero means "until the model stops"; the context length is the real bound.
    let cap = params.token_cap();

    // `next` is sampled but not yet in the target cache.
    let mut next = sampler
        .sample(session.target_context_mut(), -1, false)
        .map_err(|e| InferError::Decode(format!("MTP sample first token: {e}")))?;
    sampler.accept(next, true);

    loop {
        if generated >= cap || model.is_eog_token(next) {
            break;
        }

        emit_token(model, next, &mut pending, reply)?;
        if generated == 0 {
            first_token_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        }
        generated += 1;

        // One slot must stay free for the token that comes after this round.
        let room = (n_ctx as i32 - n_past - 1).max(0) as usize;
        if room == 0 {
            break;
        }

        let proposals = if generated < cap {
            session
                .draft(0, n_past, next)
                .map_err(|e| InferError::Decode(format!("MTP draft: {e}")))?
        } else {
            Vec::new()
        };
        let k = proposals.len().min(room).min(k_max);

        // `draft` already pushed the proposal positions into the head's own
        // cache, but those rows were speculative: `process` has to rebuild them
        // from the target's shifted hidden states, which is also what records
        // the rows `accept` later reads. Without this reset the two writers
        // collide and the head is rejected outright.
        session
            .clear_draft_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .map_err(|e| InferError::Decode(format!("MTP draft kv reset: {e}")))?;

        if k == 0 {
            // No proposal: fall through to the single-token decode below.
            batch.clear();
            batch
                .add(next, n_past, &[0], true)
                .map_err(|e| InferError::Decode(e.to_string()))?;
            session
                .decode_target_and_process(&mut batch)
                .map_err(|e| InferError::Decode(format!("MTP decode: {e}")))?;
            n_past += 1;
            resident.push(next.0);
            next = sampler
                .sample(session.target_context_mut(), -1, false)
                .map_err(|e| InferError::Decode(format!("MTP sample: {e}")))?;
            sampler.accept(next, true);
            continue;
        }

        let draft = &proposals[..k];
        batch.clear();
        batch
            .add(next, n_past, &[0], true)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        for (i, &tok) in draft.iter().enumerate() {
            batch
                .add(tok, n_past + 1 + i as i32, &[0], true)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        // The target decode has to be followed by `process`: it is the only
        // place that reads the target's hidden states back and stores the rows
        // `accept` picks the new pending row from. It runs against the head's
        // cache, not the target's, so the logits the verification below reads
        // are untouched.
        session
            .decode_target_and_process(&mut batch)
            .map_err(|e| InferError::Decode(format!("MTP verify: {e}")))?;

        draft_tokens += k;
        draft_rounds += 1;

        // Row 0 of the batch is the token that produced the proposal, so row j
        // validates draft j-1, and the row after the last draft gives the token
        // that follows. `verify_draft` is the same routine the n-gram path
        // relies on, so both draughters stay under one verified implementation.
        let out = verify_draft(&mut sampler, session.target_context_mut(), draft)
            .map_err(|e| InferError::Decode(format!("MTP verify sample: {e}")))?;
        if out.is_empty() {
            return Err(InferError::Decode(
                "MTP: verification returned no tokens".into(),
            ));
        }
        // `out` is the accepted drafts followed by the token sampled after the
        // last confirmed position, so it always ends in the new `next`.
        let accepted = out.len() - 1;
        let (kept_drafts, sampled) = out.split_at(accepted);
        // The batch above was anchored on the previous `next`, which is what the
        // cache gained first; `next` is about to become the token after it.
        let anchor = next;
        next = sampled[0];

        // Drop the rejected tail from the target cache. `p1 = None` means "to
        // the end", which is the only form a recurrent cache can honour.
        let kept = 1 + accepted as i32;
        if accepted < k {
            let p0 = (n_past + kept) as u32;
            // Both caches advanced through the rejected tokens and both have to
            // give them back, or the draft head's positions stop lining up with
            // the target's and the next `process` is rejected.
            let trimmed = session
                .clear_target_kv_cache_seq(Some(0), Some(p0), None)
                .map_err(|e| InferError::Decode(format!("MTP target kv trim: {e}")))?;
            let trimmed_draft = session
                .clear_draft_kv_cache_seq(Some(0), Some(p0), None)
                .map_err(|e| InferError::Decode(format!("MTP draft kv trim: {e}")))?;
            if !trimmed || !trimmed_draft {
                return Err(InferError::Decode(format!(
                    "MTP: the KV cache would not drop {} rejected draft token(s) even with \
                     n_rs_seq snapshots (target={trimmed}, draft={trimmed_draft}); raise \
                     --n-rs-seq or lower the draft length",
                    k - accepted
                )));
            }
        }

        // Roll the draft head's recurrent state to what survived.
        session
            .accept(0, accepted as u16)
            .map_err(|e| InferError::Decode(format!("MTP accept: {e}")))?;

        accepted_draft += accepted;
        n_past += kept;
        resident.push(anchor.0);
        resident.extend(kept_drafts.iter().map(|t| t.0));

        // Emit the accepted drafts; the trailing sampled token goes out at the
        // top of the next round, which is where it is also checked against the
        // stop conditions. A draft may itself contain the end-of-generation
        // token, so the round has to be able to stop mid-prefix like the
        // n-gram path does.
        let mut stop = false;
        for &tok in kept_drafts {
            if generated >= cap || model.is_eog_token(tok) {
                stop = true;
                break;
            }
            emit_token(model, tok, &mut pending, reply)?;
            generated += 1;
        }
        if stop {
            break;
        }
    }
    emit_tail(&mut pending, reply)?;

    Ok(GenStats {
        prompt_tokens,
        generated_tokens: generated,
        prompt_ms,
        gen_ms: t_gen.elapsed().as_secs_f64() * 1000.0,
        first_token_ms,
        draft_tokens,
        accepted_draft_tokens: accepted_draft,
        draft_rounds,
        hit_token_cap: params.hit_token_cap(generated),
        ..Default::default()
    })
}

/// Which protocol a separately published draft model speaks.
///
/// The two are not interchangeable: they differ in what the checkpoint has to
/// contain and in how the session validates it, so it has to be told which one
/// it is driving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DraftKind {
    /// Z-Lab's block-diffusion drafter. DFlash2 checkpoints — the ones that add
    /// grouped dynamic convolution and a candidate selector — speak the same
    /// protocol and are detected from the header, so there is no separate
    /// variant for them.
    DFlash,
    /// EAGLE-3, which needs three target-extract layers in the draft model.
    Eagle3,
    /// A side-mounted MTP (NextN) head: a second checkpoint that carries the
    /// target's own extra prediction layer, stripped of the trunk it was cut
    /// from. Qwen3.8 publishes its MTP this way rather than bundling it.
    Mtp,
}

impl DraftKind {
    fn name(self) -> &'static str {
        match self {
            DraftKind::DFlash => "DFlash",
            DraftKind::Eagle3 => "EAGLE-3",
            DraftKind::Mtp => "MTP",
        }
    }
}

/// Read the draft protocol off the checkpoint, falling back to its file name.
///
/// All three are published with the protocol in the name, which would usually
/// be enough — but the header is what decides, and an EAGLE-3 checkpoint handed
/// to the DFlash session fails somewhere deep in the draft graph. The nextn
/// tensors are the one signal that survives being renamed, so an MTP head is
/// recognised from them before the file name is even looked at.
fn draft_kind(path: &Path) -> DraftKind {
    if let Ok(p) = gguf_probe::probe(path) {
        let arch = p
            .architecture
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if arch.contains("dflash") {
            return DraftKind::DFlash;
        }
        if arch.contains("eagle") {
            return DraftKind::Eagle3;
        }
        if p.has_mtp() {
            return DraftKind::Mtp;
        }
    }
    let name = path.to_string_lossy().to_lowercase();
    // Order matters: publishers name checkpoints after the family they serve as
    // often as after the head they are, so "eagle" is checked before "mtp" — a
    // file mentioning both is much more likely to be the EAGLE checkpoint. The
    // NextN spelling is what several MTP-head releases actually use.
    if name.contains("dflash") {
        DraftKind::DFlash
    } else if name.contains("eagle") {
        DraftKind::Eagle3
    } else if name.contains("mtp") || name.contains("nextn") {
        DraftKind::Mtp
    } else {
        DraftKind::Eagle3
    }
}

/// Speculative decoding driven by a **separate** draft model.
///
/// The counterpart to [`generate_mtp`]: there the proposals come from heads
/// bundled in the target, here from a second checkpoint trained to draft for it.
/// The loop is the same shape — prefill through the session so the drafter sees
/// every prompt position, then per round: propose, verify the whole proposal in
/// one batched target decode, keep the prefix the target agreed with, and trim
/// both caches back to it.
///
/// Both exist because they cover different models rather than because one is
/// better: an MTP head only exists on models that shipped one, and DFlash's
/// block diffusion was built for dense models that did not — Qwen3.8-27B has no
/// MTP head to load, but there is a DFlash2 checkpoint for it.
#[allow(clippy::too_many_arguments)]
fn generate_draft_model<'m, 'd>(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'m>,
    draft_ctx: &mut LlamaContext<'d>,
    kind: DraftKind,
    templates: Option<&ChatTemplates>,
    msgs: &[ChatMsg],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    n_ctx: usize,
    n_draft_max: i32,
) -> Result<GenStats> {
    let who = kind.name();
    let n_batch = ctx.n_batch().max(1) as usize;
    let k_max = (n_draft_max.max(1) as usize).min(n_batch.saturating_sub(1));

    // The drafter works from the target's hidden states, and image positions
    // have none, so an image prompt is refused rather than silently drafting
    // from a history that does not match what is in the cache.
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();
    if images > 0 {
        return Err(InferError::Unsupported {
            class: who.to_ascii_lowercase(),
            what: format!("draft with images ({images} attached); {who} is text-only"),
        });
    }

    // The drafter must see the same prompt the target would, thinking mode
    // included, or the two disagree about the conversation from token zero.
    let prompt = tokenize_text_prompt(templates, model, msgs, params)?;
    if prompt.len() >= n_ctx {
        return Err(InferError::ContextFull {
            used: prompt.len(),
            n_ctx,
        });
    }

    let mut cs_params = CommonSamplerParams::new();
    let mut sc = cs_params.scalars();
    sc.min_keep = 1;
    sc.temp = params.temperature;
    sc.top_k = params.top_k;
    sc.top_p = params.top_p;
    sc.min_p = params.min_p;
    sc.penalty_last_n = params.repeat_last_n;
    sc.penalty_repeat = params.repeat_penalty;
    sc.penalty_freq = params.repeat_freq;
    sc.penalty_present = params.repeat_present;
    sc.seed = params.seed;
    cs_params.set_scalars(&sc);
    let mut sampler =
        CommonSampler::new(model, &mut cs_params).map_err(|e| InferError::Decode(e.to_string()))?;
    for tok in &prompt {
        sampler.accept(*tok, false);
    }

    let config = Eagle3SessionConfig::new(1, n_draft_max);
    let mut session = match kind {
        DraftKind::DFlash => Eagle3Session::new_dflash_with_config(ctx, draft_ctx, config),
        DraftKind::Eagle3 => Eagle3Session::new_with_config(ctx, draft_ctx, config),
        // Unreachable: an MTP head is routed to `generate_mtp` before this
        // function is ever entered, and would not survive this session's
        // validation if it were.
        DraftKind::Mtp => {
            return Err(InferError::Decode(
                "an MTP head cannot be driven by a separate-draft session".into(),
            ));
        }
    }
    .map_err(|e| InferError::Decode(format!("{who} session: {e}")))?;

    // A request that failed mid-round leaves the drafter ahead of the target,
    // and the next prefill would then be rejected outright.
    session
        .clear_draft_kv_cache_seq(None, None, None)
        .map_err(|e| InferError::Decode(format!("{who} draft cache reset: {e}")))?;

    // The whole prompt has to pass through the session, not just its last
    // ubatch: the drafter keeps a row per token, and one it never saw is a
    // hole it cannot draft past.
    let t_start = Instant::now();
    let mut batch = LlamaBatch::new(n_batch.max(k_max + 1), 1);
    let prompt_tokens = prompt.len();
    let total = prompt.len();
    let mut n_past = 0i32;
    for (chunk_idx, chunk) in prompt.chunks(n_batch).enumerate() {
        batch.clear();
        let base = chunk_idx * n_batch;
        for (i, &tok) in chunk.iter().enumerate() {
            batch
                .add(tok, (base + i) as i32, &[0], base + i == total - 1)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        session.decode_target_and_process(&mut batch).map_err(|e| {
            InferError::Decode(format!(
                "{who} prefill ubatch {chunk_idx} (n_tokens={}, n_past={base}): {e}",
                chunk.len()
            ))
        })?;
        n_past = (base + chunk.len()) as i32;
    }
    let prompt_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    session
        .begin(0, &prompt)
        .map_err(|e| InferError::Decode(format!("{who} begin: {e}")))?;

    let t_gen = Instant::now();
    let mut first_token_ms = 0.0f64;
    let mut pending: Vec<u8> = Vec::new();
    let mut generated = 0usize;
    let mut draft_tokens = 0usize;
    let mut accepted_draft = 0usize;
    let mut draft_rounds = 0usize;
    let cap = params.token_cap();

    let mut next = sampler
        .sample(session.target_context_mut(), -1, false)
        .map_err(|e| InferError::Decode(format!("{who} sample first token: {e}")))?;
    sampler.accept(next, true);

    loop {
        if generated >= cap || model.is_eog_token(next) {
            break;
        }

        emit_token(model, next, &mut pending, reply)?;
        if generated == 0 {
            first_token_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        }
        generated += 1;

        let room = (n_ctx as i32 - n_past - 1).max(0) as usize;
        if room == 0 {
            break;
        }

        let proposals = if generated < cap {
            session
                .draft(0, n_past, next)
                .map_err(|e| InferError::Decode(format!("{who} draft: {e}")))?
        } else {
            Vec::new()
        };
        let k = proposals.len().min(room).min(k_max);

        // `draft` pushed the proposal positions into the drafter's own cache,
        // but those rows were speculative: `process` has to rebuild them from
        // the target's hidden states, which is also what records the rows
        // `accept` reads back.
        session
            .clear_draft_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .map_err(|e| InferError::Decode(format!("{who} draft kv reset: {e}")))?;

        if k == 0 {
            batch.clear();
            batch
                .add(next, n_past, &[0], true)
                .map_err(|e| InferError::Decode(e.to_string()))?;
            session
                .decode_target_and_process(&mut batch)
                .map_err(|e| InferError::Decode(format!("{who} decode: {e}")))?;
            n_past += 1;
            next = sampler
                .sample(session.target_context_mut(), -1, false)
                .map_err(|e| InferError::Decode(format!("{who} sample: {e}")))?;
            sampler.accept(next, true);
            continue;
        }

        let draft = &proposals[..k];
        batch.clear();
        batch
            .add(next, n_past, &[0], true)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        for (i, &tok) in draft.iter().enumerate() {
            batch
                .add(tok, n_past + 1 + i as i32, &[0], true)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        session
            .decode_target_and_process(&mut batch)
            .map_err(|e| InferError::Decode(format!("{who} verify: {e}")))?;

        draft_tokens += k;
        draft_rounds += 1;

        // Row 0 produced the proposal, so row j validates draft j-1, and the
        // row after the last draft yields the token that follows.
        let out = verify_draft(&mut sampler, session.target_context_mut(), draft)
            .map_err(|e| InferError::Decode(format!("{who} verify sample: {e}")))?;
        if out.is_empty() {
            return Err(InferError::Decode(format!(
                "{who}: verification returned no tokens"
            )));
        }
        let accepted = out.len() - 1;
        let (kept_drafts, sampled) = out.split_at(accepted);
        next = sampled[0];

        let kept = 1 + accepted as i32;
        if accepted < k {
            let p0 = (n_past + kept) as u32;
            // Both caches walked through the rejected tokens and both have to
            // give them back, or the drafter's positions stop lining up with
            // the target's and the next `process` is refused.
            let trimmed = session
                .clear_target_kv_cache_seq(Some(0), Some(p0), None)
                .map_err(|e| InferError::Decode(format!("{who} target kv trim: {e}")))?;
            let trimmed_draft = session
                .clear_draft_kv_cache_seq(Some(0), Some(p0), None)
                .map_err(|e| InferError::Decode(format!("{who} draft kv trim: {e}")))?;
            if !trimmed || !trimmed_draft {
                return Err(InferError::Decode(format!(
                    "{who}: the KV cache would not drop {} rejected draft token(s) \
                     (target={trimmed}, draft={trimmed_draft}); raise --n-rs-seq or lower \
                     the draft length",
                    k - accepted
                )));
            }
        }

        session
            .accept(0, accepted as u16)
            .map_err(|e| InferError::Decode(format!("{who} accept: {e}")))?;

        accepted_draft += accepted;
        n_past += kept;

        let mut stop = false;
        for &tok in kept_drafts {
            if generated >= cap || model.is_eog_token(tok) {
                stop = true;
                break;
            }
            emit_token(model, tok, &mut pending, reply)?;
            generated += 1;
        }
        if stop {
            break;
        }
    }
    emit_tail(&mut pending, reply)?;

    Ok(GenStats {
        prompt_tokens,
        generated_tokens: generated,
        prompt_ms,
        gen_ms: t_gen.elapsed().as_secs_f64() * 1000.0,
        first_token_ms,
        draft_tokens,
        accepted_draft_tokens: accepted_draft,
        draft_rounds,
        hit_token_cap: params.hit_token_cap(generated),
        ..Default::default()
    })
}

/// Speculative decoding with an n-gram drafter.
///
/// The drafter proposes a continuation by finding the most recent repeat of the
/// trailing n-gram, the model verifies the whole proposal in a single batched
/// decode, and only the tokens it confirmed are kept. That makes this exact —
/// the accepted prefix is what the model would have sampled anyway — while
/// turning several sequential decodes into one batched pass. Prose usually
/// offers no repeat to copy and drafts nothing, which costs nothing at all.
///
/// The loop keeps one invariant: `next` has been sampled but is **not** in the
/// KV cache yet. That is what lets a round decode `next` together with the
/// draft in a single batch and then read `draft.len() + 1` logits back, which is
/// exactly what `sample_and_accept_n` indexes.
#[allow(clippy::too_many_arguments)]
fn generate_speculative(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    prompt: &[LlamaToken],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    n_ctx: usize,
    mut n_past: i32,
    prompt_tokens: usize,
    prompt_ms: f64,
    cache: &mut DraftCache,
    drafting: &AtomicBool,
    // Ids the cache holds when this returns. Kept in step with `n_past` at every
    // commit point, because the caller keys the next turn's prefix on it.
    resident: &mut Vec<i32>,
) -> Result<GenStats> {
    let n = params.ngram_n.max(1);
    let m = params.ngram_m.max(1);
    let k_max = m as usize;
    let n_batch = ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(n_batch.max(k_max + 1), 1);

    // Verifying a draft needs `CommonSampler`, so this path cannot reuse the
    // hand-rolled chain in `build_sampler`. Its default chain (penalties →
    // top_k → typical_p → top_p → min_p → xtc → temperature) collapses to the
    // same behaviour once the no-op samplers take their default scalars.
    let mut cs_params = CommonSamplerParams::new();
    let mut sc = cs_params.scalars();
    sc.min_keep = 1; // the plain path samples with a minimum of one candidate
    sc.temp = params.temperature;
    sc.top_k = params.top_k;
    sc.top_p = params.top_p;
    sc.min_p = params.min_p;
    sc.penalty_last_n = params.repeat_last_n;
    sc.penalty_repeat = params.repeat_penalty;
    sc.penalty_freq = params.repeat_freq;
    sc.penalty_present = params.repeat_present;
    sc.seed = params.seed;
    cs_params.set_scalars(&sc);
    let mut sampler =
        CommonSampler::new(model, &mut cs_params).map_err(|e| InferError::Decode(e.to_string()))?;

    // Repetition penalties have to see the prompt, but as prompt tokens: the
    // sampler treats them differently from what the model generated.
    for &tok in prompt {
        sampler.accept(tok, false);
    }

    let t_gen = Instant::now();
    let mut first_token_ms = 0.0f64;
    let mut pending: Vec<u8> = Vec::new();
    let mut generated = 0usize;
    let mut draft_tokens = 0usize;
    let mut accepted_draft = 0usize;
    let mut draft_rounds = 0usize;
    // A round that gets nothing accepted has already paid for a forward pass,
    // so wait a step before drafting again instead of paying twice per token.
    let mut draft_pause = 0usize;
    // Zero means "until the model stops"; the context length is the real bound.
    let cap = params.token_cap();

    // Every committed token, searched for repeats. `next` is deliberately kept
    // out of it: `ngram_simple_draft` appends the sampled token to the pattern.
    let mut hist: Vec<LlamaToken> = prompt.to_vec();

    let mut next = sampler
        .sample(ctx, -1, false)
        .map_err(|e| InferError::Decode(format!("sample first token: {e}")))?;
    sampler.accept(next, true);

    loop {
        if generated >= cap || model.is_eog_token(next) {
            break;
        }

        emit_token(model, next, &mut pending, reply)?;
        if generated == 0 {
            first_token_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        }
        generated += 1;
        hist.push(next);

        // One slot past `n_past` has to stay free, otherwise there is nowhere
        // to put the token that comes next.
        let room = (n_ctx as i32 - n_past - 1).max(0) as usize;
        if room == 0 {
            break;
        }

        // The emitted token that still has to reach the KV cache. Normally that
        // is `next` itself; a drafted round leaves `out`'s last token in that
        // state instead, because everything in `out` is already emitted.
        let mut to_decode = next;

        // `ngram_simple_draft` needs more history than the pattern it searches
        // for, otherwise there is nothing to match against.
        let can_draft = drafting.load(Ordering::Relaxed)
            && draft_pause == 0
            && hist.len() > n as usize + m as usize;
        if can_draft {
            // Two drafters, unioned. The stateless one finds repeats inside this
            // conversation; the cache knows what followed these n-grams in earlier
            // ones, including before a restart. Both are cheap lookups.
            let drafted = ngram_simple_draft(n, m, &hist[..hist.len() - 1], next)
                .map_err(|e| InferError::Decode(format!("ngram draft: {e}")))?;
            let drafted = if drafted.is_empty() {
                cache.draft(&hist, n, m as usize)
            } else {
                drafted
            };
            let k = draft_len(drafted.len(), room, k_max, n_batch);

            if k > 0 {
                let draft = &drafted[..k];
                batch.clear();
                batch
                    .add(next, n_past, &[0], true)
                    .map_err(|e| InferError::Decode(e.to_string()))?;
                for (i, &tok) in draft.iter().enumerate() {
                    batch
                        .add(tok, n_past + 1 + i as i32, &[0], true)
                        .map_err(|e| InferError::Decode(e.to_string()))?;
                }
                ctx.decode(&mut batch).map_err(|e| {
                    InferError::Decode(format!(
                        "draft verify decode (n_tokens={}, n_past={}, draft={}): {e}",
                        batch.n_tokens(),
                        n_past,
                        k
                    ))
                })?;
                // Logit row i predicts draft[i], so the batch above yields the
                // `k + 1` rows this reads.
                let out = verify_draft(&mut sampler, ctx, draft).map_err(|e| {
                    InferError::Decode(format!(
                        "verify draft (draft={}, n_past={}): {e}",
                        k, n_past
                    ))
                })?;
                if out.is_empty() {
                    break;
                }
                let accepted = out.len() - 1;
                draft_rounds += 1;
                draft_tokens += k;
                accepted_draft += accepted;
                draft_pause = usize::from(accepted == 0);

                if accepted < k {
                    // The rejected draft tokens are still sitting in the KV
                    // cache above the accepted prefix, and llama.cpp requires
                    // every batch to start exactly one position past the cache's
                    // last one. The rejected tail is always at the *end* of the
                    // sequence, so this is an open-ended removal ("to the end").
                    let p0 = (n_past + 1 + accepted as i32) as u32;
                    let trimmed = ctx
                        .clear_kv_cache_seq(Some(0), Some(p0), None)
                        .map_err(|e| {
                            InferError::Decode(format!("speculative: kv cache trim: {e}"))
                        })?;

                    if trimmed {
                        n_past += 1 + accepted as i32;
                        // The anchor and the accepted drafts are in the cache;
                        // the token that follows them lands there below.
                        resident.push(next.0);
                        resident.extend(draft[..accepted].iter().map(|t| t.0));
                    } else {
                        // Hybrid models (recurrent + attention layers) refuse to
                        // drop a range from the cache, so there is no way to
                        // undo a rejected draft. That is not a failure worth
                        // failing the request over: rebuild the cache from what
                        // has actually been committed, stop drafting for this
                        // engine, and finish the reply one token at a time.
                        drafting.store(false, Ordering::Relaxed);

                        let mut replay: Vec<LlamaToken> = hist[..hist.len() - 1].to_vec();
                        if let Some(&last) = hist.last() {
                            replay.push(last);
                        }
                        replay.extend_from_slice(&out[..accepted]);

                        ctx.clear_kv_cache();
                        decode_sequence(ctx, &replay)?;
                        n_past = replay.len() as i32;
                        *resident = replay.iter().map(|t| t.0).collect();
                    }
                } else {
                    n_past += 1 + accepted as i32;
                }

                let mut stop = false;
                for &tok in &out {
                    if generated >= cap || model.is_eog_token(tok) {
                        stop = true;
                        break;
                    }
                    emit_token(model, tok, &mut pending, reply)?;
                    if generated == 0 {
                        first_token_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
                    }
                    generated += 1;
                    hist.push(tok);
                }
                // `out` is entirely emitted now, and its last token is the one
                // still missing from the cache. Emitting it again as `next`
                // would duplicate it, so decode it instead.
                to_decode = out[out.len() - 1];
                if stop {
                    break;
                }
            }
        }

        draft_pause = draft_pause.saturating_sub(1);
        // A draft that was accepted whole can leave the context exactly full.
        // The token still to decode has already been emitted, so ending here is
        // correct — decoding one position past the end is what used to turn a
        // filled context into a failed request whose text had already gone out.
        if n_past as usize >= n_ctx {
            break;
        }
        batch.clear();
        batch
            .add(to_decode, n_past, &[0], true)
            .map_err(|e| InferError::Decode(e.to_string()))?;
        n_past += 1;
        ctx.decode(&mut batch).map_err(|e| {
            InferError::Decode(format!(
                "speculative step decode (n_tokens={}, n_past={}): {e}",
                batch.n_tokens(),
                n_past
            ))
        })?;
        resident.push(to_decode.0);
        next = sampler
            .sample(ctx, -1, false)
            .map_err(|e| InferError::Decode(format!("sample next token: {e}")))?;
        sampler.accept(next, true);
    }

    emit_tail(&mut pending, reply)?;

    // Whatever this reply produced is what the next one should be able to copy.
    cache.learn(&hist, n);

    Ok(GenStats {
        prompt_tokens,
        generated_tokens: generated,
        prompt_ms,
        gen_ms: t_gen.elapsed().as_secs_f64() * 1000.0,
        first_token_ms,
        draft_tokens,
        accepted_draft_tokens: accepted_draft,
        draft_rounds,
        hit_token_cap: params.hit_token_cap(generated),
        ..Default::default()
    })
}

/// Check a draft against the model's own logits, one row at a time.
///
/// This is `common_sampler_sample_and_accept_n` written out by hand, and it has
/// to be: the Rust binding reaches that function through a size-then-fill
/// wrapper, which runs the whole verification **twice**. The first pass feeds
/// every token it checks back into the sampler, so the second pass can disagree
/// with the first and report a longer result than the buffer sized for the first —
/// which surfaces as an opaque failure partway through a reply. Sampling row by
/// row has no output buffer to size, and it also stops the sampler's history from
/// being double-counted.
///
/// Logit row `i` predicts `draft[i]`, so the caller must have decoded the anchor
/// token plus the whole draft in one batch. The result is the accepted draft
/// prefix followed by exactly one freshly sampled token, and is never empty.
fn verify_draft(
    sampler: &mut CommonSampler,
    ctx: &mut LlamaContext,
    draft: &[LlamaToken],
) -> std::result::Result<Vec<LlamaToken>, String> {
    let mut out = Vec::with_capacity(draft.len() + 1);
    for (i, &proposed) in draft.iter().enumerate() {
        let token = sampler
            .sample(ctx, i as i32, false)
            .map_err(|e| e.to_string())?;
        sampler.accept(token, true);
        let agrees = token.0 == proposed.0;
        out.push(token);
        if !agrees {
            return Ok(out);
        }
    }
    // The whole draft held up, so sample the token that follows it.
    let token = sampler
        .sample(ctx, draft.len() as i32, false)
        .map_err(|e| e.to_string())?;
    sampler.accept(token, true);
    out.push(token);
    Ok(out)
}

/// Decode a whole token sequence into the context, logits on the last token.
///
/// Used to rebuild the KV cache from committed tokens when a speculative round
/// has to be undone and the cache cannot drop the rejected tail.
fn decode_sequence(ctx: &mut LlamaContext, tokens: &[LlamaToken]) -> Result<()> {
    let n_batch = ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let total = tokens.len();
    for (chunk_idx, chunk) in tokens.chunks(n_batch).enumerate() {
        batch.clear();
        let base = chunk_idx * n_batch;
        for (i, &tok) in chunk.iter().enumerate() {
            let pos = (base + i) as i32;
            let logits = base + i == total - 1;
            batch
                .add(tok, pos, &[0], logits)
                .map_err(|e| InferError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferError::Decode(e.to_string()))?;
    }
    Ok(())
}

/// Clamp a draft to what one verification batch can hold.
/// The batch carries the anchor token plus the draft, so it needs one free slot
/// in the context (`room`) and one free slot in the batch itself.
pub(crate) fn draft_len(drafted: usize, room: usize, k_max: usize, n_batch: usize) -> usize {
    drafted.min(room).min(k_max).min(n_batch.saturating_sub(1))
}

// The bytes emitted so far on this worker thread, and the parser for the prompt
// they came from.
//
// Every token the reply sends passes through `emit_token`, which makes it the
// one place that sees the whole reply. When the request declared tools the reply
// is parsed once it is complete — the template's parser splits it into content,
// reasoning and tool calls — so the bytes and that parser are kept here, on the
// worker's own thread (the only thread that ever runs a generation), rather than
// threaded through every generator's signature.
thread_local! {
    static EMITTED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static PARSER: std::cell::RefCell<Option<ReplyParser>> = const { std::cell::RefCell::new(None) };
}

/// The parser for the tool-bearing request in flight.
///
/// A tool call is only recognisable once its closing marker arrives, and text
/// already sent cannot be taken back, so a reply that might contain one is not
/// streamed token by token: nothing is sent during generation and the parsed
/// content is emitted once, just before `Done`.
struct ReplyParser {
    chat: ChatParams,
}

/// Whether the generation in flight is holding its reply back for parsing.
fn streaming_held() -> bool {
    PARSER.with(|p| p.borrow().is_some())
}

/// Forget the previous reply's text and parser before a new generation.
fn reset_emitted() {
    EMITTED.with(|e| e.borrow_mut().clear());
    PARSER.with(|p| *p.borrow_mut() = None);
}

fn take_emitted() -> String {
    EMITTED.with(|e| {
        let bytes = std::mem::take(&mut *e.borrow_mut());
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// What a finished reply contributes to the response.
#[derive(Default)]
struct ParsedReply {
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<serde_json::Value>,
}

/// Parse the finished reply into content, reasoning and tool calls.
///
/// A reply with no tools is returned empty and left untouched: its text was
/// already streamed as it arrived, and there is nothing to pull apart. For a
/// tool-bearing reply the held-back content is emitted here, before `Done`.
fn finish_reply(reply: &Sender<GenEvent>) -> ParsedReply {
    let Some(chat) = PARSER.with(|p| p.borrow().as_ref().map(|r| r.chat.clone())) else {
        return ParsedReply::default();
    };
    let text = take_emitted();
    let Ok(parsed) = chat.parse(&text, false) else {
        // Parsing is presentation, not substance: if it fails the raw text
        // stands in, so a reply is never lost to it. Logged because a failed
        // parse of a tool-bearing reply looks, to the caller, like the model
        // simply chose not to call anything.
        tracing::warn!("tool reply did not parse; returning it as plain text");
        let _ = reply.send(GenEvent::Token(text));
        return ParsedReply::default();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&parsed) else {
        return ParsedReply::default();
    };
    let field = |key: &str| {
        value
            .get(key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let content = field("content");
    let reasoning = field("reasoning_content");
    let tool_calls = value
        .get("tool_calls")
        .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        .cloned();
    if let Some(content) = &content {
        let _ = reply.send(GenEvent::Token(content.clone()));
    }
    ParsedReply {
        content,
        reasoning,
        tool_calls,
    }
}

/// Append one token's bytes to the stream, emitting every complete UTF-8 prefix.
fn emit_token(
    model: &LlamaModel,
    token: LlamaToken,
    pending: &mut Vec<u8>,
    reply: &Sender<GenEvent>,
) -> Result<()> {
    let piece = model
        .token_to_bytes(token, Special::Plaintext)
        .map_err(|e| InferError::Tokenize(e.to_string()))?;
    // Only a held-back reply is read back as a whole, so only it is kept.
    if streaming_held() {
        EMITTED.with(|e| e.borrow_mut().extend_from_slice(&piece));
    }
    pending.extend_from_slice(&piece);
    emit_utf8(pending, reply)
}

fn build_sampler(params: &SamplerParams, n_vocab: i32) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::with_capacity(6);
    if params.repeat_penalty != 1.0 || params.repeat_freq != 0.0 || params.repeat_present != 0.0 {
        chain.push(LlamaSampler::penalties(
            n_vocab,
            params.repeat_last_n,
            params.repeat_penalty,
            params.repeat_freq,
            params.repeat_present,
        ));
    }
    if params.temperature <= 0.0 {
        // Greedy still has to run behind the penalties: they change which logit
        // is the largest, so dropping them changes the answer. Returning early
        // used to skip them, which is why temperature 0 disagreed with the
        // speculative path — that one keeps its penalties on.
        chain.push(LlamaSampler::greedy());
        return LlamaSampler::chain_simple(chain);
    }
    chain.push(LlamaSampler::top_k(params.top_k));
    chain.push(LlamaSampler::top_p(params.top_p, 1));
    chain.push(LlamaSampler::min_p(params.min_p, 1));
    chain.push(LlamaSampler::temp(params.temperature));
    chain.push(LlamaSampler::dist(params.seed));
    LlamaSampler::chain_simple(chain)
}

/// Emit every complete UTF-8 prefix currently buffered, keeping any partial tail.
///
/// A send failure means the caller dropped the receiver — the browser closed the
/// tab, the request was cancelled — and is reported rather than swallowed, so the
/// generation loop stops instead of spending the rest of `max_tokens` on output
/// nobody will read.
fn emit_utf8(pending: &mut Vec<u8>, reply: &Sender<GenEvent>) -> Result<()> {
    // A tool-bearing reply is held back and emitted whole once parsed, so its
    // text is dropped here rather than sent as it arrives.
    if streaming_held() {
        pending.clear();
        return Ok(());
    }
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                if !s.is_empty() {
                    reply
                        .send(GenEvent::Token(s.to_string()))
                        .map_err(|_| InferError::ClientGone)?;
                    pending.clear();
                }
                return Ok(());
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    let s = unsafe { std::str::from_utf8_unchecked(&pending[..valid]) }.to_string();
                    reply
                        .send(GenEvent::Token(s))
                        .map_err(|_| InferError::ClientGone)?;
                    pending.drain(..valid);
                }
                if e.error_len().is_some() {
                    let s = String::from_utf8_lossy(&pending[..1]).into_owned();
                    reply
                        .send(GenEvent::Token(s))
                        .map_err(|_| InferError::ClientGone)?;
                    pending.remove(0);
                    continue;
                }
                return Ok(());
            }
        }
    }
}

fn emit_tail(pending: &mut Vec<u8>, reply: &Sender<GenEvent>) -> Result<()> {
    if streaming_held() {
        pending.clear();
        return Ok(());
    }
    if !pending.is_empty() {
        let s = String::from_utf8_lossy(pending).into_owned();
        reply
            .send(GenEvent::Token(s))
            .map_err(|_| InferError::ClientGone)?;
        pending.clear();
    }
    Ok(())
}
