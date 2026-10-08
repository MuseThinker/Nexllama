use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_core::{Engine, EngineInfo, LoadConfig};
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
}
