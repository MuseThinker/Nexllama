use std::collections::HashMap;
use std::path::PathBuf;
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
}

pub struct DaemonState {
    pub models: HashMap<String, Loaded>,
    pub default_model: Option<String>,
    /// Zero means keep every model resident until explicitly unloaded.
    pub keep_alive: Duration,
    pub base_cfg: LoadConfig,
}

impl DaemonState {
    pub fn touch(&mut self, id: &str) {
        if let Some(l) = self.models.get_mut(id) {
            l.last_used = Instant::now();
        }
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
