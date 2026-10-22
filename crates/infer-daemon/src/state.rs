use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_core::{Engine, EngineInfo, GenStats, LoadConfig};
use tokio::sync::Mutex;

pub struct Loaded {
    pub id: String,
    pub path: PathBuf,
    pub engine: Engine,
    pub info: EngineInfo,
    /// How this model was loaded, kept so it can be loaded again after the idle
    /// reaper frees it (see [`Dormant`]).
    pub config: LoadConfig,
    pub loaded_at: Instant,
    pub last_used: Instant,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub load_mtp: bool,
    pub mmproj: Option<String>,
    pub draft_model: Option<String>,
    /// Default reasoning mode for this model, from the load request.
    ///
    /// `None` keeps the model's own default. A chat request that does not name
    /// its own `thinking` inherits this, so "never think on this model" is set
    /// once, at load, rather than on every message.
    pub thinking: Option<bool>,
    pub loras: Vec<String>,
    /// Requests being served by this model right now.
    ///
    /// The idle timer alone used to decide when to unload, and `last_used` is
    /// only touched when a request *starts* — so a reply that ran longer than
    /// keep_alive had its model taken out from under it mid-generation.
    pub in_flight: Arc<AtomicUsize>,
    /// Running performance figures, updated after every reply.
    ///
    /// Kept behind a `std` mutex rather than reached through `DaemonState`
    /// because a reply finishes on a blocking thread, which can hold this but
    /// not the async lock guarding the model map.
    pub stats: Arc<std::sync::Mutex<RuntimeStats>>,
}

impl Loaded {
    /// Whether any request is being served by this model at this moment.
    pub fn is_busy(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) > 0
    }
}

/// A model the daemon can load again but is not holding right now.
///
/// The idle reaper moves a freed model here instead of forgetting it, so the
/// next request that names it is served by loading it back rather than failing
/// as unknown. Everything about the model except the engine itself is kept.
#[derive(Clone)]
pub struct Dormant {
    pub id: String,
    pub path: PathBuf,
    pub config: LoadConfig,
    pub info: EngineInfo,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub load_mtp: bool,
    pub mmproj: Option<String>,
    pub draft_model: Option<String>,
    pub thinking: Option<bool>,
    pub loras: Vec<String>,
}

impl Dormant {
    /// Capture everything about a live model except the weights and contexts.
    pub fn from_loaded(l: &Loaded) -> Self {
        Self {
            id: l.id.clone(),
            path: l.path.clone(),
            config: l.config.clone(),
            info: l.info.clone(),
            n_gpu_layers: l.n_gpu_layers,
            n_ctx: l.n_ctx,
            load_mtp: l.load_mtp,
            mmproj: l.mmproj.clone(),
            draft_model: l.draft_model.clone(),
            thinking: l.thinking,
            loras: l.loras.clone(),
        }
    }
}

/// Keeps a model resident for as long as a request is running.
///
/// Handed out when a request starts and released when the reply is finished,
/// which is what stops the reaper from unloading a model that is busy. It is
/// cloneable because a streamed reply outlives the handler that started it.
pub struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter.clone())
    }
}

impl Clone for InFlight {
    /// A clone is another holder, not another view of the same one: each has to
    /// be released, or the model reads as idle while a stream is still open.
    fn clone(&self) -> Self {
        Self::new(&self.0)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Weight of the newest sample once enough have arrived.
///
/// Low enough that one freak request cannot swing the answer, high enough that
/// a real change shows up within a handful of replies.
const EMA_ALPHA: f64 = 0.2;

/// How many samples to treat as unreliable, and so weight more heavily.
///
/// Applied as `1/n`, the usual bias correction: the first few replies are
/// mostly noise, and letting the very first one anchor the average would make
/// the figure wrong for longer than the warm-up lasts.
const EMA_WARMUP: u64 = 5;

/// One step of an exponential moving average.
fn ema(prev: f64, sample: f64, n: u64) -> f64 {
    let alpha = if n <= EMA_WARMUP {
        1.0 / n as f64
    } else {
        EMA_ALPHA
    };
    prev + alpha * (sample - prev)
}

/// How this model has actually been performing.
///
/// Averages rather than totals: what makes a model feel fast or slow is how it
/// is doing now, and a running total would let one enormous request dominate
/// the answer for the rest of the session.
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    /// Replies completed since this model was loaded.
    pub requests: u64,
    /// Tokens produced since this model was loaded.
    pub tokens: u64,
    /// Smoothed decode rate. Zero until a reply has been timed.
    pub tokens_per_second: f64,
    /// Smoothed time to first token, in milliseconds.
    pub first_token_ms: f64,
    /// Smoothed share of drafted tokens the model confirmed.
    ///
    /// `None` until a reply actually drafted something, which is the only
    /// honest answer for a model that has not tried.
    pub draft_acceptance: Option<f64>,
}

impl RuntimeStats {
    /// Fold one finished reply into the averages.
    pub fn record(&mut self, s: &GenStats) {
        self.requests += 1;
        self.tokens += s.generated_tokens as u64;

        // The first sample has nothing to average with, so it seeds rather than
        // easing up from zero; easing would report a model as slow for the
        // first few replies purely because it started at 0.
        if self.requests == 1 {
            self.tokens_per_second = s.tokens_per_second();
            self.first_token_ms = s.first_token_ms;
        } else {
            self.tokens_per_second =
                ema(self.tokens_per_second, s.tokens_per_second(), self.requests);
            self.first_token_ms = ema(self.first_token_ms, s.first_token_ms, self.requests);
        }

        // Drafting is decided per request. A reply that never proposed anything
        // must not drag the acceptance down as though it had proposed and lost.
        if let Some(rate) = s.draft_acceptance() {
            self.draft_acceptance = Some(match self.draft_acceptance {
                None => rate,
                Some(prev) => ema(prev, rate, self.requests),
            });
        }
    }
}

/// Fold a reply into a model's figures, if there is a slot for them.
///
/// A poisoned lock is dropped rather than propagated: a reply has already been
/// delivered by the time this runs, and failing it for the sake of a statistic
/// would be a worse outcome than not recording one.
pub fn record_stats(slot: Option<&Arc<std::sync::Mutex<RuntimeStats>>>, s: &GenStats) {
    if let Some(st) = slot {
        if let Ok(mut g) = st.lock() {
            g.record(s);
        }
    }
}

pub struct DaemonState {
    pub models: HashMap<String, Loaded>,
    /// Models that were loaded and then freed, remembered so a later request can
    /// load them again rather than fail. Only the config and metadata survive.
    pub dormant: HashMap<String, Dormant>,
    pub default_model: Option<String>,
    /// Zero means keep every model resident until explicitly unloaded.
    pub keep_alive: Duration,
    pub base_cfg: LoadConfig,
}

impl DaemonState {
    /// Start serving a request: refresh the idle timer and hold the model.
    ///
    /// Returns `None` only if the model is gone, which the caller has already
    /// ruled out by resolving an id.
    pub fn begin(&mut self, id: &str) -> Option<InFlight> {
        let l = self.models.get_mut(id)?;
        l.last_used = Instant::now();
        Some(InFlight::new(&l.in_flight))
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.models.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Resolve which model a request refers to, falling back to the only model
    /// of the wanted class.
    ///
    /// An exact name wins when it is loaded (or remembered) and is the right
    /// class. A name that matches nothing falls back to the sole model of that
    /// class, so a client holding a stale name — an agent configured before the
    /// desktop app swapped the loaded model — is still served rather than
    /// refused. Two or more models of the class make the choice ambiguous, and
    /// the caller is told to name one.
    pub fn resolve_class(&self, requested: Option<&str>, class: &str) -> Option<String> {
        let of_that_class = |id: &str| -> bool {
            self.models
                .get(id)
                .map(|l| l.info.model_class == class)
                .or_else(|| self.dormant.get(id).map(|d| d.info.model_class == class))
                .unwrap_or(false)
        };
        if let Some(id) = requested.filter(|m| !m.trim().is_empty()) {
            if of_that_class(id) {
                return Some(id.to_string());
            }
        }
        let mut matches = self
            .models
            .iter()
            .filter(|(_, l)| l.info.model_class == class)
            .map(|(k, _)| k.clone())
            .chain(
                self.dormant
                    .iter()
                    .filter(|(_, d)| d.info.model_class == class)
                    .map(|(k, _)| k.clone()),
            );
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }
}

/// The `auto` alias's policy, set by the app over loopback.
///
/// `auto` is a routing strategy, not a model: when `enabled`, a request naming
/// `auto` runs on the local model when it can and falls back to the cloud when
/// it cannot. When disabled, `auto` is neither listed nor resolved.
#[derive(Clone)]
pub struct Routing {
    pub enabled: bool,
    /// `"local"` (the default) or `"cloud"`.
    pub prefer: String,
}

impl Default for Routing {
    fn default() -> Self {
        Self {
            enabled: false,
            prefer: "local".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Mutex<DaemonState>>,
    /// Configured cloud providers, with their keys. Independent of `inner` so a
    /// request that routes to a cloud provider never contends with (or holds)
    /// the model lock while its upstream call is in flight.
    pub cloud: crate::cloud::Registry,
    /// The `auto` alias's policy: whether it is on, and its preference
    /// (`"local"` or `"cloud"`). Set by the app over loopback.
    pub routing: Arc<std::sync::RwLock<Routing>>,
    /// Shared HTTP client for cloud proxying: one connection pool, reused.
    pub client: reqwest::Client,
    /// Folder `GET /admin/models` lists, when the daemon was pointed at one.
    ///
    /// Separate from the models that are *loaded* here: those arrive by path
    /// (`--model`, `POST /admin/load`) and are the engine's business. This is
    /// the library folder a browser UI or the desktop app browses, which is why
    /// it is optional — a daemon that was handed models by path has no folder to
    /// list and says so instead of inventing one.
    pub models_dir: Option<Arc<PathBuf>>,
    /// HuggingFace transfers started over HTTP, and their progress reports.
    pub hub: Arc<crate::hub::HubState>,
}

impl AppState {
    pub fn new(keep_alive: Duration, base_cfg: LoadConfig) -> Self {
        Self {
            cloud: Arc::new(std::sync::RwLock::new(Vec::new())),
            routing: Arc::new(std::sync::RwLock::new(Routing::default())),
            // One pooled client for every upstream call: a reused connection
            // skips the TCP + TLS handshake that otherwise adds a beat to the
            // first token of each cloud reply. `tcp_nodelay` keeps small SSE
            // chunks from being held back by Nagle.
            client: reqwest::Client::builder()
                .tcp_nodelay(true)
                .tcp_keepalive(Duration::from_secs(60))
                .pool_max_idle_per_host(8)
                .pool_idle_timeout(Duration::from_secs(300))
                .build()
                .unwrap_or_default(),
            models_dir: None,
            hub: Arc::new(crate::hub::HubState::new()),
            inner: Arc::new(Mutex::new(DaemonState {
                models: HashMap::new(),
                dormant: HashMap::new(),
                default_model: None,
                keep_alive,
                base_cfg,
            })),
        }
    }

    /// Point `GET /admin/models` at a folder. Left unset for a daemon given
    /// models by path.
    pub fn with_models_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.models_dir = dir.map(Arc::new);
        self
    }

    /// Hand out a model's performance slot.
    ///
    /// Kept apart from [`DaemonState::begin`] because a caller may want the
    /// figures without starting a request, and because the caller that records
    /// them runs on a blocking thread — which can hold the returned `std`
    /// mutex, but not the async one this takes.
    pub async fn runtime_stats(&self, id: &str) -> Option<Arc<std::sync::Mutex<RuntimeStats>>> {
        let guard = self.inner.lock().await;
        guard.models.get(id).map(|l| l.stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(tokens: usize, gen_ms: f64, ttft: f64) -> GenStats {
        GenStats {
            generated_tokens: tokens,
            gen_ms,
            first_token_ms: ttft,
            ..Default::default()
        }
    }

    #[test]
    fn the_first_reply_seeds_rather_than_easing_up_from_zero() {
        let mut s = RuntimeStats::default();
        s.record(&reply(100, 1000.0, 50.0));
        assert_eq!((s.requests, s.tokens), (1, 100));
        assert_eq!(s.tokens_per_second, 100.0);
        assert_eq!(s.first_token_ms, 50.0);
        // Nothing drafted, so there is no rate to report.
        assert_eq!(s.draft_acceptance, None);
    }

    #[test]
    fn a_reply_that_never_drafted_does_not_count_as_rejected() {
        let mut s = RuntimeStats::default();
        s.record(&reply(10, 100.0, 5.0));
        s.record(&reply(10, 100.0, 5.0));
        assert_eq!(s.requests, 2);
        // Reporting 0 here would claim every draft had been rejected, when in
        // fact no draft was ever proposed.
        assert_eq!(s.draft_acceptance, None);
    }

    #[test]
    fn acceptance_moves_toward_a_new_sample_without_jumping() {
        let mut s = RuntimeStats::default();
        // Every draft rejected.
        s.record(&GenStats {
            draft_tokens: 4,
            accepted_draft_tokens: 0,
            ..Default::default()
        });
        assert_eq!(s.draft_acceptance, Some(0.0));

        // Every draft confirmed. The average has to rise toward it, not become
        // it: one good reply does not erase a bad one.
        s.record(&GenStats {
            draft_tokens: 4,
            accepted_draft_tokens: 4,
            ..Default::default()
        });
        let rate = s
            .draft_acceptance
            .expect("acceptance set by a drafting reply");
        assert!(rate > 0.0 && rate < 1.0, "expected smoothing, got {rate}");
    }
}
