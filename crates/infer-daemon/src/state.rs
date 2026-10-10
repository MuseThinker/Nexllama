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
    pub loaded_at: Instant,
    pub last_used: Instant,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub load_mtp: bool,
    pub mmproj: Option<String>,
    pub draft_model: Option<String>,
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

    /// Resolve which model a request refers to.
    pub fn resolve(&self, requested: Option<&str>) -> Option<String> {
        if let Some(id) = requested {
            return self.models.contains_key(id).then(|| id.to_string());
        }
        if let Some(d) = &self.default_model {
            if self.models.contains_key(d) {
                return Some(d.clone());
            }
        }
        if self.models.len() == 1 {
            return self.models.keys().next().cloned();
        }
        None
    }
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Mutex<DaemonState>>,
}

impl AppState {
    pub fn new(keep_alive: Duration, base_cfg: LoadConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(DaemonState {
                models: HashMap::new(),
                default_model: None,
                keep_alive,
                base_cfg,
            })),
        }
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
