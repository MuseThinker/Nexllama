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
    /// CPU threads for generation. None = auto (physical cores).
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
    pub load_mtp: bool,
    /// Recurrent-state snapshots kept per sequence so a rejected draft can be
    /// rolled back. Hybrid/recurrent caches refuse partial rollback at 0.
    #[serde(default)]
    pub n_rs_seq: u32,
    /// MTP draft tokens per round (upstream `--spec-draft-n-max`).
    ///
    /// 0 turns MTP off. Anything above it needs a second (draft) context and
    /// forces `load_mtp`, because the weights have to be resident to draft at all.
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
    /// How many tokens a separate draft model may propose per round.
    ///
    /// 0 picks a conservative default; the real bound is `n_batch`, since the
    /// whole proposal has to be verified in one pass.
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
            draft_n_max: 0,
            lora: Vec::new(),
            ngram_cache: None,
            mmproj: None,
        }
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
fn global_backend() -> Result<&'static LlamaBackend> {
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

    let build_ctx = |pooling: Option<LlamaPoolingType>| {
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(cfg.n_ctx))
            .with_n_batch(cfg.n_batch)
            .with_n_ubatch(cfg.n_ubatch)
            .with_flash_attn_type(flash)
            .with_offload_kqv(cfg.offload_kqv)
            .with_n_rs_seq(cfg.n_rs_seq);
        if class != gguf_probe::ModelClass::Llm {
            ctx_params = ctx_params.with_embeddings(true);
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
    let mtp_n_max = cfg.mtp_n_max;
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
            match LlamaModel::load_from_file(
                backend,
                path,
                &LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers),
            ) {
                Ok(m) => Some(m),
                Err(e) => {
                    let _ = ready.send(LoadEvent::Error(format!("draft model load: {e:?}")));
                    return;
                }
            }
        }
        None => None,
    };
    let mut draft_ctx = match draft_model.as_ref() {
        Some(dm) => {
            let _ = ready.send(LoadEvent::Phase("creating draft context".into()));
            // An ordinary context, not `LlamaContextType::Mtp`: that one is for a
            // model drafting from heads bundled in itself, whereas this reads the
            // target's hidden states through the session instead.
            match dm.new_context(
                backend,
                build_ctx(None).with_n_ctx(NonZeroU32::new(cfg.n_ctx)),
            ) {
                Ok(mut c) => {
                    c.set_warmup(false);
                    Some(c)
                }
                Err(e) => {
                    let _ = ready.send(LoadEvent::Error(format!("draft context: {e:?}")));
                    return;
                }
            }
        }
        None => None,
    };
    let draft_kind = cfg.draft_model.as_deref().map(draft_kind);
    let draft_n_max = if cfg.draft_n_max > 0 {
        cfg.draft_n_max
    } else {
        8
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
                    &msgs,
                    &params,
                    &reply,
                    &mut cache,
                    &drafting,
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
    msgs: &[ChatMsg],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    cache: &mut DraftCache,
    drafting: &AtomicBool,
    mtp_draft_ctx: &mut Option<LlamaContext<'m>>,
    // (see `run_generate`)
    mtp_n_max: u32,
    draft_ctx: &mut Option<LlamaContext<'d>>,
    draft_kind: Option<DraftKind>,
    draft_n_max: u32,
) -> Result<()> {
    let n_ctx = ctx.n_ctx() as usize;
    let n_batch = ctx.n_batch().max(1) as usize;

    ctx.clear_kv_cache();

    // MTP prefills through its own session, because the draft head needs the
    // target's hidden state for *every* prompt token and only the session can
    // hand that over. Prefilling here first would leave the head with nothing
    // to draft from.
    if params.speculative && mtp_n_max > 0 {
        if let Some(draft) = mtp_draft_ctx.as_mut() {
            let stats = generate_mtp(
                model,
                ctx,
                draft,
                vision,
                msgs,
                params,
                reply,
                n_ctx,
                mtp_n_max as i32,
            )?;
            let _ = reply.send(GenEvent::Done(stats));
            return Ok(());
        }
    }

    // A separate draft model outranks the n-gram drafter as well: it proposes
    // from learned weights rather than from repeats in this conversation, so
    // when one is loaded it is the better guess.
    if params.speculative {
        if let (Some(kind), Some(draft)) = (draft_kind, draft_ctx.as_mut()) {
            let stats = generate_draft_model(
                model,
                ctx,
                draft,
                kind,
                msgs,
                params,
                reply,
                n_ctx,
                draft_n_max as i32,
            )?;
            let _ = reply.send(GenEvent::Done(stats));
            return Ok(());
        }
    }

    let t_start = Instant::now();
    let (prompt_tokens, n_past, hist) = fill_prompt(model, ctx, vision, msgs, n_ctx, n_batch)?;
    let prompt_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let stats = if params.speculative {
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
        )?
    };
    let _ = reply.send(GenEvent::Done(stats));
    Ok(())
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
fn fill_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    vision: &Option<MtmdContext>,
    msgs: &[ChatMsg],
    n_ctx: usize,
    n_batch: usize,
) -> Result<(usize, i32, Vec<LlamaToken>)> {
    let images: usize = msgs.iter().map(ChatMsg::image_count).sum();
    let Some(vision) = vision.as_ref().filter(|_| images > 0) else {
        if images > 0 {
            return Err(InferError::VisionUnavailable { images });
        }
        return fill_text_prompt(model, ctx, msgs, n_ctx, n_batch);
    };

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

/// The ordinary text path: tokenize, then decode in `n_batch` slices.
fn fill_text_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    msgs: &[ChatMsg],
    n_ctx: usize,
    n_batch: usize,
) -> Result<(usize, i32, Vec<LlamaToken>)> {
    let messages = msgs
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;

    let prompt = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| InferError::Tokenize(e.to_string()))?;

    if tokens.len() >= n_ctx {
        return Err(InferError::ContextFull {
            used: tokens.len(),
            n_ctx,
        });
    }

    let mut batch = LlamaBatch::new(n_batch, 1);
    let total = tokens.len();
    let mut n_past: i32 = 0;

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
    msgs: &[ChatMsg],
    params: &SamplerParams,
    reply: &Sender<GenEvent>,
    n_ctx: usize,
    mtp_n_max: i32,
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

    let messages = msgs
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
    let rendered = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
    let prompt = model
        .str_to_token(&rendered, AddBos::Always)
        .map_err(|e| InferError::Tokenize(e.to_string()))?;
    if prompt.len() >= n_ctx {
        return Err(InferError::ContextFull {
            used: prompt.len(),
            n_ctx,
        });
    }

    // Verification has to sample from every position in the batch, so this
    // reuses the n-gram path's reasoning: the plain hand-rolled chain cannot
    // read a row other than -1.
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

    // The session borrows both contexts, so the draft cache can only be emptied
    // through it. A request that failed mid-round leaves the head's positions
    // ahead of the target's, and the next prefill would then be rejected.
    session
        .clear_draft_kv_cache_seq(None, None, None)
        .map_err(|e| InferError::Decode(format!("MTP draft cache reset: {e}")))?;

    // The draft head needs the whole prompt to have gone through the target, so
    // prefill goes through the session rather than straight to the context.
    // Every ubatch has to pass `process`, not just the last one: the head keeps
    // a hidden-state row per token, and a ubatch it never saw is a hole in it.
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
}

impl DraftKind {
    fn name(self) -> &'static str {
        match self {
            DraftKind::DFlash => "DFlash",
            DraftKind::Eagle3 => "EAGLE-3",
        }
    }
}

/// Read the draft protocol off the checkpoint, falling back to its file name.
///
/// Both kinds are published with the protocol in the name, which would usually
/// be enough — but the header is what decides, and an EAGLE-3 checkpoint handed
/// to the DFlash session fails somewhere deep in the draft graph.
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
    }
    if path.to_string_lossy().to_lowercase().contains("dflash") {
        DraftKind::DFlash
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

    let messages = msgs
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
    let rendered = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| InferError::ChatTemplate(e.to_string()))?;
    let prompt = model
        .str_to_token(&rendered, AddBos::Always)
        .map_err(|e| InferError::Tokenize(e.to_string()))?;
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
    if !pending.is_empty() {
        let s = String::from_utf8_lossy(pending).into_owned();
        reply
            .send(GenEvent::Token(s))
            .map_err(|_| InferError::ClientGone)?;
        pending.clear();
    }
    Ok(())
}
