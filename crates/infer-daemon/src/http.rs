use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::{header, header::AUTHORIZATION, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;

use infer_core::{
    ChatMsg, Engine, EngineHandle, FlashAttn, InferError, LoadMode, LoraAdapter, SamplerParams,
};

use crate::state::{AppState, DaemonState, InFlight, Loaded};

/// Shared secret required on every request when set.
pub type ApiKey = Option<Arc<String>>;

/// Upper bounds on what a request may ask this server to allocate or repeat.
///
/// These are not tuning values. `/admin/load` hands `n_ctx` and `n_batch`
/// straight to llama.cpp, which sizes the KV cache and the batch from them: a
/// request carrying `n_ctx: 4294967295` does not fail cleanly, it asks the
/// allocator for terabytes and takes the whole process — and every model
/// loaded in it — down. Checking here turns that into a 400.
mod limits {
    /// Largest context a load may ask for. The KV cache is
    /// O(n_ctx × n_layer × n_embd), so this is the one that matters most.
    pub const N_CTX: u32 = 1 << 20;
    /// Largest logical batch.
    pub const N_BATCH: u32 = 1 << 16;
    /// Largest physical (micro) batch.
    pub const N_UBATCH: u32 = 1 << 16;
    /// Most layers that can be offloaded.
    pub const N_GPU_LAYERS: u32 = 4096;
    /// Most threads that can be requested.
    pub const THREADS: i32 = 1024;
    /// Largest request body accepted.
    ///
    /// A chat transcript carries images as base64, so this has to be generous:
    /// 64 MB is far past any real prompt, and it stops a client from pushing an
    /// unbounded blob into memory a megabyte at a time.
    pub const BODY_BYTES: usize = 64 * 1024 * 1024;
}

pub fn router(state: AppState, api_key: ApiKey, cors_origins: Vec<String>) -> Router {
    let protected = Router::new()
        // OpenAI-compatible surface.
        .route("/v1", get(list_models))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        // Upstream llama.cpp registers the same four rerank spellings.
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        .route("/v1/reranking", post(rerank))
        .route("/reranking", post(rerank))
        // Ollama-compatible surface, so an example written against Ollama works
        // here by changing the port and nothing else.
        .route("/api/tags", get(ollama_tags))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/embed", post(ollama_embed))
        .route("/api/embeddings", post(ollama_embeddings_legacy))
        .route("/admin/status", get(status))
        .route("/admin/load", post(load))
        .route("/admin/unload", post(unload))
        .route_layer(middleware::from_fn_with_state(api_key, require_key));

    Router::new()
        // Open like Ollama's banner: says what this is and where to go, and
        // deliberately no model names, so exposing the port does not leak them.
        .route("/", get(root))
        .route("/openapi.json", get(openapi))
        .route("/docs", get(docs))
        .route("/api/version", get(ollama_version))
        .route("/health", get(health))
        .merge(protected)
        .layer(cors_layer(&cors_origins))
        .layer(RequestBodyLimitLayer::new(limits::BODY_BYTES))
        .with_state(state)
}

/// Which browser origins may call this server.
///
/// `CorsLayer::permissive()` answered every cross-origin request, which turned
/// "open a web page" into "drive this daemon": any site the machine visited
/// could load a model, burn GPU time and read the replies, whether or not a
/// token was configured. Only origins that are this app — the WebView it
/// serves, the dev server, or one the operator named with `--cors-origin` —
/// are answered.
///
/// Programmatic clients are unaffected either way: curl, Python and the desktop
/// app send no `Origin` header, and CORS is a browser rule, not a server one.
fn cors_layer(extra: &[String]) -> CorsLayer {
    const LOCAL: &[&str] = &[
        // Tauri's WebView, and the dev server that feeds it.
        "http://asset.localhost",
        "tauri://localhost",
        "http://tauri.localhost",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ];
    let allowed: Vec<HeaderValue> = LOCAL
        .iter()
        .copied()
        .chain(extra.iter().map(String::as_str))
        .filter_map(|o| HeaderValue::from_str(o).ok())
        .collect();

    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin: &HeaderValue, _| {
            allowed.contains(origin)
        }))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([AUTHORIZATION, header::CONTENT_TYPE])
        .max_age(std::time::Duration::from_secs(600))
}

/// `GET /openapi.json` — the specification, parsed once and handed out as-is.
///
/// Parsing at first use keeps a malformed document from booting, and the drift
/// test below keeps it from describing routes that do not exist.
async fn openapi() -> Json<serde_json::Value> {
    static SPEC: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    Json(
        SPEC.get_or_init(|| {
            serde_json::from_str(include_str!("../openapi.json")).expect("openapi.json parses")
        })
        .clone(),
    )
}

/// `GET /docs` — a browsable list built from the specification.
///
/// The page renders itself from `/openapi.json` with no script or style fetched
/// from anywhere: this runs on machines that may have no internet at all.
async fn docs() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("docs.html"),
    )
}

/// `GET /` — the banner.
///
/// Pointing a browser or a curl at the base URL is the first thing anyone does;
/// answering with a JSON menu beats a 404.
async fn root(State(state): State<AppState>) -> Json<serde_json::Value> {
    let loaded = state.inner.lock().await.models.len();
    Json(json!({
        "name": "Nexllama",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "ok",
        "models_loaded": loaded,
        "endpoints": {
            "list_models": ["GET /v1", "GET /v1/models", "GET /api/tags"],
            "chat": ["POST /v1/chat/completions", "POST /api/chat", "POST /api/generate"],
            "embeddings": ["POST /v1/embeddings", "POST /api/embed", "POST /api/embeddings"],
            "rerank": ["POST /v1/rerank", "POST /rerank"],
            "admin": ["GET /health", "GET /admin/status", "POST /admin/load", "POST /admin/unload"]
        }
    }))
}

/// `GET /api/version` — Ollama clients ask for this before anything else.
async fn ollama_version() -> Json<serde_json::Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// `GET /api/tags` — Ollama's model list.
async fn ollama_tags(State(state): State<AppState>) -> Json<serde_json::Value> {
    let guard = state.inner.lock().await;
    let models: Vec<serde_json::Value> = guard
        .models
        .values()
        .map(|l| {
            json!({
                "name": l.id,
                "model": l.id,
                "size": l.info.model_size,
                "digest": "",
                "modified_at": "",
                "details": {
                    "family": l.info.description,
                    "parameter_size": format!("{:.2}B", l.info.n_params as f64 / 1e9),
                    "quantization_level": l.info.ftype,
                    "class": l.info.model_class,
                },
            })
        })
        .collect();
    Json(json!({ "models": models }))
}

/// Reject requests that do not carry the configured bearer token.
///
/// Binding to a routable interface hands this server to the whole network, and
/// it has no accounts of its own, so the token is the only thing standing
/// between a stranger and someone's GPU. `/health` stays open for readiness
/// probes; it exposes loaded model names and nothing more.
async fn require_key(State(key): State<ApiKey>, request: Request, next: Next) -> Response {
    let Some(expected) = key else {
        return next.run(request).await;
    };

    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);

    // A plain compare is fine here: this guards a local model server, not a
    // timing-sensitive secret, and keeping it constant-length avoids leaking
    // anything through early returns.
    let ok = match presented {
        Some(got) => {
            let (a, b) = (got.as_bytes(), expected.as_bytes());
            a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
        }
        None => false,
    };

    if ok {
        return next.run(request).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        [(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"nexllama\""),
        )],
        Json(json!({ "error": "missing or invalid bearer token" })),
    )
        .into_response()
}

#[derive(Debug)]
pub enum ApiError {
    NotLoaded,
    NotFound(String),
    /// A request asked for something outside what this server will do.
    BadRequest(String),
    Model(String),
    Worker(String),
}

impl ApiError {
    fn code(&self) -> StatusCode {
        match self {
            ApiError::NotLoaded => StatusCode::CONFLICT,
            ApiError::NotFound(_) | ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Model(_) | ApiError::Worker(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn message(&self) -> String {
        match self {
            ApiError::NotLoaded => "no model loaded".to_string(),
            ApiError::NotFound(m)
            | ApiError::BadRequest(m)
            | ApiError::Model(m)
            | ApiError::Worker(m) => m.clone(),
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.code(), Json(json!({ "error": self.message() }))).into_response()
    }
}

impl From<InferError> for ApiError {
    fn from(e: InferError) -> Self {
        ApiError::Model(e.to_string())
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(e: tokio::task::JoinError) -> Self {
        ApiError::Worker(e.to_string())
    }
}

async fn health(State(state): State<AppState>) -> Response {
    let guard = state.inner.lock().await;
    // A model can sit in the map and still be unable to answer: its worker
    // thread may have exited. Asking costs nothing, and answering "ok" while
    // every loaded model is dead is what makes a readiness probe useless — the
    // orchestrator never learns to restart this.
    let dead: Vec<String> = guard
        .models
        .iter()
        .filter(|(_, l)| !l.engine.handle().is_alive())
        .map(|(id, _)| id.clone())
        .collect();
    if !dead.is_empty() {
        tracing::error!(?dead, "loaded models are no longer answering");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "degraded", "unresponsive": dead })),
        )
            .into_response();
    }
    Json(json!({
        "status": "ok",
        // Lets the app check that the process answering here is the one it
        // started, rather than a leftover holding the port.
        "pid": std::process::id(),
        "models_loaded": guard.models.len(),
        "models": guard.ids(),
        "default_model": guard.default_model,
        "keep_alive_secs": guard.keep_alive.as_secs(),
    }))
    .into_response()
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let guard = state.inner.lock().await;
    let models: Vec<serde_json::Value> = guard
        .models
        .values()
        .map(|l| {
            json!({
                "id": l.id,
                "path": l.path,
                "info": l.info,
                "n_gpu_layers": l.n_gpu_layers,
                "n_ctx": l.n_ctx,
                "load_mtp": l.load_mtp,
                "mmproj": l.mmproj,
                "draft_model": l.draft_model,
                "loras": l.loras,
                "loaded_secs_ago": l.loaded_at.elapsed().as_secs(),
                "idle_secs": l.last_used.elapsed().as_secs(),
            })
        })
        .collect();
    Json(json!({
        "models": models,
        "default_model": guard.default_model,
        "keep_alive_secs": guard.keep_alive.as_secs(),
    }))
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let guard = state.inner.lock().await;
    let data: Vec<serde_json::Value> = guard
        .models
        .values()
        .map(|l| {
            json!({
                "id": l.id,
                "object": "model",
                "path": l.path,
                "info": l.info,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

#[derive(Deserialize)]
struct LoadRequest {
    model: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    n_gpu_layers: Option<u32>,
    #[serde(default)]
    main_gpu: Option<i32>,
    #[serde(default)]
    n_ctx: Option<u32>,
    #[serde(default)]
    n_batch: Option<u32>,
    #[serde(default)]
    n_ubatch: Option<u32>,
    #[serde(default)]
    threads: Option<i32>,
    #[serde(default)]
    threads_batch: Option<i32>,
    #[serde(default)]
    load_mode: Option<LoadMode>,
    #[serde(default)]
    mmap: Option<bool>,
    #[serde(default)]
    use_mlock: Option<bool>,
    #[serde(default)]
    flash_attn: Option<FlashAttn>,
    #[serde(default)]
    offload_kqv: Option<bool>,
    #[serde(default)]
    rope_freq_base: Option<f32>,
    #[serde(default)]
    rope_freq_scale: Option<f32>,
    #[serde(default)]
    load_mtp: Option<bool>,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    draft_model: Option<String>,
    #[serde(default)]
    draft_n_max: Option<u32>,
    #[serde(default)]
    lora: Option<Vec<String>>,
    #[serde(default)]
    keep_alive_secs: Option<u64>,
    #[serde(default)]
    set_default: Option<bool>,
}

fn model_id(req: &LoadRequest) -> String {
    if let Some(name) = req.name.as_ref().filter(|s| !s.trim().is_empty()) {
        return name.trim().to_string();
    }
    Path::new(&req.model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string()
}

async fn load(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let id = model_id(&req);
    let mut cfg;
    {
        let guard = state.inner.lock().await;
        cfg = guard.base_cfg.clone();
    }
    // Reject the sizes before they reach llama.cpp, where they stop being a
    // bad number and start being an allocation. Done before the file check so a
    // request that cannot be honoured does not touch the disk at all.
    for (name, v, max) in [
        ("n_ctx", req.n_ctx, limits::N_CTX),
        ("n_batch", req.n_batch, limits::N_BATCH),
        ("n_ubatch", req.n_ubatch, limits::N_UBATCH),
    ] {
        if let Some(v) = v {
            if v == 0 || v > max {
                return Err(ApiError::BadRequest(format!(
                    "{name} must be between 1 and {max}, got {v}"
                )));
            }
        }
    }
    if let (Some(batch), Some(ubatch)) = (req.n_batch, req.n_ubatch) {
        if ubatch > batch {
            return Err(ApiError::BadRequest(format!(
                "n_ubatch ({ubatch}) must not exceed n_batch ({batch})"
            )));
        }
    }
    for (name, v) in [
        ("threads", req.threads),
        ("threads_batch", req.threads_batch),
    ] {
        if let Some(v) = v {
            if !(0..=limits::THREADS).contains(&v) {
                return Err(ApiError::BadRequest(format!(
                    "{name} must be between 0 and {}, got {v}",
                    limits::THREADS
                )));
            }
        }
    }

    if let Some(v) = req.n_gpu_layers {
        if v > limits::N_GPU_LAYERS {
            return Err(ApiError::BadRequest(format!(
                "n_gpu_layers must be <= {}, got {v}",
                limits::N_GPU_LAYERS
            )));
        }
        cfg.n_gpu_layers = v;
    }
    if let Some(v) = req.main_gpu {
        if !(0..=64).contains(&v) {
            return Err(ApiError::BadRequest(format!(
                "main_gpu must be between 0 and 64, got {v}"
            )));
        }
        cfg.main_gpu = v;
    }
    cfg.model_path = req.model.clone().into();
    if !Path::new(&req.model).is_file() {
        return Err(ApiError::NotFound(format!(
            "model file not found: {}",
            req.model
        )));
    }
    if let Some(v) = req.n_ctx {
        cfg.n_ctx = v;
    }
    if let Some(v) = req.n_batch {
        cfg.n_batch = v;
    }
    if let Some(v) = req.n_ubatch {
        cfg.n_ubatch = v;
    }
    if let Some(v) = req.threads {
        cfg.n_threads = (v > 0).then_some(v);
    }
    if let Some(v) = req.threads_batch {
        cfg.n_threads_batch = (v > 0).then_some(v);
    }
    if let Some(v) = req.load_mode {
        cfg.load_mode = v;
    }
    if let Some(v) = req.mmap {
        cfg.load_mode = if v { LoadMode::Auto } else { LoadMode::None };
    }
    if let Some(v) = req.use_mlock {
        cfg.use_mlock = v;
    }
    if let Some(v) = req.flash_attn {
        cfg.flash_attn = v;
    }
    if let Some(v) = req.offload_kqv {
        cfg.offload_kqv = v;
    }
    if let Some(v) = req.rope_freq_base {
        cfg.rope_freq_base = v;
    }
    if let Some(v) = req.rope_freq_scale {
        cfg.rope_freq_scale = v;
    }
    if let Some(v) = req.load_mtp {
        cfg.load_mtp = v;
    }
    if let Some(v) = req.lora.as_ref() {
        cfg.lora = v
            .iter()
            .map(|p| LoraAdapter {
                path: p.clone().into(),
                scale: 1.0,
            })
            .collect();
    }
    if let Some(v) = req.mmproj.as_ref() {
        if !Path::new(v).exists() {
            return Err(ApiError::NotFound(format!("mmproj not found: {v}")));
        }
        cfg.mmproj = Some(v.clone().into());
    }
    // A separate checkpoint that drafts for this model — DFlash or EAGLE-3. It
    // is the other half of speculative decoding: `load_mtp` uses heads bundled
    // in the target, this uses a second set of weights trained to predict it.
    if let Some(v) = req.draft_model.as_ref() {
        if !Path::new(v).is_file() {
            return Err(ApiError::NotFound(format!("draft model not found: {v}")));
        }
        cfg.draft_model = Some(v.clone().into());
        if let Some(n) = req.draft_n_max {
            cfg.draft_n_max = n;
        }
    }
    let n_gpu_layers = cfg.n_gpu_layers;
    let n_ctx = cfg.n_ctx;
    let load_mtp = cfg.load_mtp;

    let mut notes: Vec<String> = Vec::new();
    if req.draft_model.is_some() {
        notes.push(
            "a draft model is loaded; turn on speculative decoding per request to have it \
             draft"
                .into(),
        );
    }

    // Drop any existing model with the same id before loading the new one.
    let old = {
        let mut guard = state.inner.lock().await;
        guard.models.remove(&id)
    };
    let _ = tokio::task::spawn_blocking(move || drop(old)).await;

    let path = cfg.model_path.clone();
    let log_id = id.clone();
    let engine = tokio::task::spawn_blocking(move || {
        Engine::load(cfg, |ev| tracing::info!(?log_id, ?ev, "load"))
    })
    .await??;

    let info = engine.info().clone();
    let mut guard = state.inner.lock().await;
    if let Some(secs) = req.keep_alive_secs {
        guard.keep_alive = std::time::Duration::from_secs(secs);
    }
    let now = Instant::now();
    // `insert` hands back whatever it displaced. Dropping an `Engine` waits for
    // its worker thread to finish the reply it is decoding, which on this
    // thread would block the async runtime — so a concurrent load of the same
    // id could stall every other request for as long as a generation takes.
    let displaced = guard.models.insert(
        id.clone(),
        Loaded {
            id: id.clone(),
            path,
            engine,
            info: info.clone(),
            loaded_at: now,
            last_used: now,
            n_gpu_layers,
            n_ctx,
            load_mtp,
            mmproj: req.mmproj.clone(),
            draft_model: req.draft_model.clone(),
            loras: req.lora.clone().unwrap_or_default(),
            in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        },
    );
    if guard.default_model.is_none() || req.set_default.unwrap_or(false) {
        guard.default_model = Some(id.clone());
    }
    drop(guard);
    if let Some(old) = displaced {
        let _ = tokio::task::spawn_blocking(move || drop(old)).await;
    }

    Ok(Json(
        json!({ "status": "ok", "id": id, "info": info, "notes": notes }),
    ))
}

#[derive(Deserialize, Default)]
struct UnloadRequest {
    #[serde(default)]
    model: Option<String>,
}

async fn unload(
    State(state): State<AppState>,
    body: Option<Json<UnloadRequest>>,
) -> Json<serde_json::Value> {
    let req = body.map(|Json(r)| r).unwrap_or_default();
    let removed: Vec<Loaded> = {
        let mut guard = state.inner.lock().await;
        match req.model.as_deref() {
            Some(id) => {
                let r = guard.models.remove(id).into_iter().collect();
                if guard.default_model.as_deref() == Some(id) {
                    guard.default_model = None;
                }
                r
            }
            None => {
                let all: Vec<Loaded> = guard.models.drain().map(|(_, l)| l).collect();
                guard.default_model = None;
                all
            }
        }
    };
    let n = removed.len();
    let _ = tokio::task::spawn_blocking(move || drop(removed)).await;
    Json(json!({ "status": "ok", "unloaded": n }))
}

/// One message, with `content` either plain text or OpenAI-style parts.
///
/// ```json
/// {"role": "user", "content": "hi"}
/// {"role": "user", "content": [
///     {"type": "text", "text": "what is this?"},
///     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
/// ]}
/// ```
#[derive(Deserialize)]
struct InMsg {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct InPart {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<InImage>,
}

#[derive(Deserialize)]
struct InImage {
    #[serde(default)]
    url: Option<String>,
}

/// Decode one `image_url` payload into raw bytes.
///
/// `data:` URLs are the only form accepted. An `http(s)` URL would turn every
/// request into an SSRF primitive against whatever the host can reach, and a bare
/// path would let a caller name files on this machine.
fn decode_image_url(url: &str) -> std::result::Result<Vec<u8>, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "image_url.url must be a data: URL".to_string())?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| "malformed data: URL".to_string())?;
    if !meta.ends_with(";base64") {
        return Err("only base64 data: URLs are supported".into());
    }
    base64_decode(payload).ok_or_else(|| "image data is not valid base64".to_string())
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn sextet(b: u8) -> Option<u8> {
        Some(match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }

    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &b in input.as_bytes() {
        // Data URLs in the wild arrive line-wrapped, so whitespace is skipped
        // rather than treated as corruption.
        if b.is_ascii_whitespace() {
            continue;
        }
        if b == b'=' {
            break;
        }
        let v = sextet(b)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<InMsg>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i32>,
    #[serde(default)]
    repeat_freq: Option<f32>,
    #[serde(default)]
    repeat_present: Option<f32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    speculative: Option<bool>,
    #[serde(default)]
    ngram_n: Option<u16>,
    #[serde(default)]
    ngram_m: Option<u16>,
}

impl ChatRequest {
    fn sampler(&self) -> SamplerParams {
        let mut s = SamplerParams::default();
        if let Some(v) = self.temperature {
            s.temperature = v;
        }
        if let Some(v) = self.top_p {
            s.top_p = v;
        }
        if let Some(v) = self.top_k {
            s.top_k = v;
        }
        if let Some(v) = self.min_p {
            s.min_p = v;
        }
        if let Some(v) = self.repeat_penalty {
            s.repeat_penalty = v;
        }
        if let Some(v) = self.repeat_last_n {
            s.repeat_last_n = v;
        }
        if let Some(v) = self.repeat_freq {
            s.repeat_freq = v;
        }
        if let Some(v) = self.repeat_present {
            s.repeat_present = v;
        }
        if let Some(v) = self.seed {
            s.seed = v;
        }
        if let Some(v) = self.max_tokens {
            s.max_tokens = v;
        }
        if let Some(v) = self.speculative {
            s.speculative = v;
        }
        if let Some(v) = self.ngram_n {
            s.ngram_n = v;
        }
        if let Some(v) = self.ngram_m {
            s.ngram_m = v;
        }
        s
    }

    /// Flatten the request into engine messages, pulling out any attached images.
    fn chat_msgs(&self) -> ApiResult<Vec<ChatMsg>> {
        self.messages
            .iter()
            .map(|m| {
                let (text, images) = flatten_content(&m.content)?;
                Ok(ChatMsg::new(m.role.clone(), text).with_images(images))
            })
            .collect()
    }
}

/// Split a message's `content` into its text and its images.
fn flatten_content(content: &serde_json::Value) -> ApiResult<(String, Vec<Vec<u8>>)> {
    match content {
        serde_json::Value::String(s) => Ok((s.clone(), Vec::new())),
        serde_json::Value::Null => Ok((String::new(), Vec::new())),
        serde_json::Value::Array(raw) => {
            let parts: Vec<InPart> = serde_json::from_value(serde_json::Value::Array(raw.clone()))
                .map_err(|e| ApiError::NotFound(format!("malformed content parts: {e}")))?;
            let mut text = String::new();
            let mut images = Vec::new();
            for part in parts {
                match part.kind.as_deref() {
                    Some("text") => text.push_str(part.text.as_deref().unwrap_or_default()),
                    Some("image_url") | Some("input_image") => {
                        let url = part
                            .image_url
                            .as_ref()
                            .and_then(|i| i.url.as_deref())
                            .ok_or_else(|| {
                                ApiError::NotFound("image_url part has no url".into())
                            })?;
                        images.push(decode_image_url(url).map_err(ApiError::NotFound)?);
                    }
                    Some(other) => {
                        return Err(ApiError::NotFound(format!(
                            "unsupported content part type: {other}"
                        )));
                    }
                    // Tolerate a bare {"text": …} or {"image_url": …} part.
                    None => {
                        if let Some(t) = part.text.as_deref() {
                            text.push_str(t);
                        } else if let Some(url) =
                            part.image_url.as_ref().and_then(|i| i.url.as_deref())
                        {
                            images.push(decode_image_url(url).map_err(ApiError::NotFound)?);
                        }
                    }
                }
            }
            Ok((text, images))
        }
        _ => Err(ApiError::NotFound(
            "content must be a string or an array of parts".into(),
        )),
    }
}

#[derive(Deserialize)]
struct EmbedRequest {
    /// A single string or a list of them, as OpenAI's API accepts.
    input: serde_json::Value,
    #[serde(default)]
    model: Option<String>,
}

/// Resolve a model for generation, refusing anything that cannot generate.
async fn chat_engine(
    state: &AppState,
    requested: Option<&str>,
) -> ApiResult<(EngineHandle, String, InFlight)> {
    let mut guard = state.inner.lock().await;
    let id = require_model(&guard, requested)?;
    // Encoder models decode nothing: their tokens are probabilities, not a
    // continuation. Failing fast says so, instead of streaming word salad.
    let class = guard
        .models
        .get(&id)
        .map(|l| l.info.model_class.clone())
        .unwrap_or_default();
    if class != "llm" {
        return Err(ApiError::NotFound(format!(
            "{id} is a {class} model — it cannot generate text; load a chat model (class llm)",
        )));
    }
    let engine = guard.models.get(&id).expect("resolved").engine.handle();
    let active = guard.begin(&id).expect("resolved");
    Ok((engine, id, active))
}

/// `YYYY-MM-DDThh:mm:ssZ` from the system clock.
///
/// Ollama stamps every streamed line, and clients parse it, so a real timestamp
/// is worth the twenty lines. Days-from-civil is Howard Hinnant's algorithm.
fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// One NDJSON line per event, which is how Ollama streams.
fn ndjson(value: serde_json::Value) -> Result<axum::body::Bytes, Infallible> {
    let mut line = value.to_string();
    line.push('\n');
    Ok(axum::body::Bytes::from(line))
}

/// Map Ollama's `options` onto the sampler knobs we have.
fn sampler_from_options(
    base: &SamplerParams,
    options: Option<&serde_json::Value>,
) -> SamplerParams {
    let mut p = base.clone();
    let Some(o) = options else { return p };
    let num = |key: &str| {
        o.get(key)
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
    };
    if let Some(v) = num("temperature") {
        p.temperature = v;
    }
    if let Some(v) = num("top_k") {
        p.top_k = v as i32;
    }
    if let Some(v) = num("top_p") {
        p.top_p = v;
    }
    if let Some(v) = num("min_p") {
        p.min_p = v;
    }
    if let Some(v) = num("repeat_penalty") {
        p.repeat_penalty = v;
    }
    if let Some(v) = num("repeat_last_n") {
        p.repeat_last_n = v as i32;
    }
    // Ollama's limit on generated tokens.
    if let Some(v) = num("num_predict").map(|v| v as i64) {
        if v > 0 {
            p.max_tokens = v as usize;
        }
    }
    if let Some(v) = o.get("seed").and_then(serde_json::Value::as_u64) {
        p.seed = v as u32;
    }
    p
}

#[derive(Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    #[serde(default = "default_true")]
    stream: bool,
    #[serde(default)]
    options: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OllamaGenerateRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default = "default_true")]
    stream: bool,
    #[serde(default)]
    options: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

/// `POST /api/chat` — Ollama's chat route.
async fn ollama_chat(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, ApiError> {
    let (handle, model_id, active) = chat_engine(&state, req.model.as_deref()).await?;
    let msgs: Vec<ChatMsg> = req
        .messages
        .iter()
        .map(|m| ChatMsg::new(m.role.clone(), m.content.clone()))
        .collect();
    let params = sampler_from_options(&SamplerParams::default(), req.options.as_ref());
    Ok(ollama_stream(handle, model_id, active, msgs, params, req.stream, true).await)
}

/// `POST /api/generate` — Ollama's single-prompt route.
///
/// Ollama keeps it distinct from chat, so this wraps the prompt as one user turn
/// rather than routing it to `/api/chat`.
async fn ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, ApiError> {
    let (handle, model_id, active) = chat_engine(&state, req.model.as_deref()).await?;
    let mut msgs = Vec::new();
    if let Some(system) = req.system.filter(|s| !s.trim().is_empty()) {
        msgs.push(ChatMsg::new("system", system));
    }
    msgs.push(ChatMsg::new("user", req.prompt.clone()));
    let params = sampler_from_options(&SamplerParams::default(), req.options.as_ref());
    Ok(ollama_stream(handle, model_id, active, msgs, params, req.stream, false).await)
}

/// Build one Ollama reply object. A present `stats` marks the final line.
fn ollama_reply(
    model_id: &str,
    chat: bool,
    content: &str,
    stats: Option<&infer_core::GenStats>,
) -> serde_json::Value {
    let mut v = json!({
        "model": model_id,
        "created_at": rfc3339_now(),
        "done": stats.is_some(),
    });
    if chat {
        v["message"] = json!({ "role": "assistant", "content": content });
    } else {
        v["response"] = json!(content);
    }
    if let Some(s) = stats {
        v["done_reason"] = json!("stop");
        v["prompt_eval_count"] = json!(s.prompt_tokens);
        v["eval_count"] = json!(s.generated_tokens);
    }
    v
}

/// Shared body for the two Ollama generation routes.
///
/// `chat` picks the message-shaped reply; otherwise the reply carries `response`,
/// which is what `/api/generate` clients read. Streaming here is newline-delimited
/// JSON rather than SSE — the one place Ollama and OpenAI genuinely differ.
async fn ollama_stream(
    handle: EngineHandle,
    model_id: String,
    active: InFlight,
    msgs: Vec<ChatMsg>,
    params: SamplerParams,
    stream: bool,
    chat: bool,
) -> Response {
    if !stream {
        let generated = tokio::task::spawn_blocking(move || {
            // Held until the reply is finished: the idle timer cannot be
            // allowed to unload this model while it is still generating.
            let _active = active;
            let mut out = String::new();
            let stats = handle.generate(&msgs, &params, |t| {
                out.push_str(t);
                true
            })?;
            Ok::<_, InferError>((out, stats))
        })
        .await;

        return match generated {
            Ok(Ok((content, stats))) => {
                Json(ollama_reply(&model_id, chat, &content, Some(&stats))).into_response()
            }
            Ok(Err(e)) => api_error(&e),
            Err(e) => api_error(&InferError::Decode(e.to_string())),
        };
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, Infallible>>(64);
    tokio::task::spawn_blocking(move || {
        // Held for as long as the stream runs, which outlives this handler by
        // however long the reply takes.
        let _active = active;
        let stream_tx = tx.clone();
        let result = handle.generate(&msgs, &params, |t| {
            // Returning `false` stops the generation. The send fails exactly
            // when the reader is gone, which is the moment carrying on would
            // cost the model's worker thread the rest of `max_tokens` for
            // nothing.
            stream_tx
                .blocking_send(ndjson(ollama_reply(&model_id, chat, t, None)))
                .is_ok()
        });
        // A reader that hung up is how a stream normally ends, not a failure.
        if matches!(result, Err(InferError::ClientGone | InferError::Cancelled)) {
            return;
        }
        let final_line = match result {
            Ok(stats) => ndjson(ollama_reply(&model_id, chat, "", Some(&stats))),
            Err(e) => ndjson(json!({ "error": e.to_string(), "done": true })),
        };
        let _ = tx.blocking_send(final_line);
    });

    // NDJSON, not the default `application/octet-stream`: Ollama clients — and
    // this project's own /docs page — branch on the content type to decide
    // whether to read the reply a line at a time or wait for all of it.
    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

/// A fresh identifier for one streamed reply, as `chatcmpl-<n>`.
fn next_stream_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("chatcmpl-{n}")
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn api_error(e: &InferError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": e.to_string() })),
    )
        .into_response()
}

/// Resolve the model a request named, refusing to guess.
///
/// There is deliberately no default: one machine can hold a chat model, an
/// embedder and a reranker at once, and silently picking one turns a typo or a
/// forgotten field into the wrong model answering. The error lists what is
/// loaded, so the fix is one line away.
fn require_model(guard: &DaemonState, requested: Option<&str>) -> ApiResult<String> {
    if guard.models.is_empty() {
        return Err(ApiError::NotLoaded);
    }
    let Some(requested) = requested.filter(|m| !m.trim().is_empty()) else {
        let mut loaded: Vec<&str> = guard.models.keys().map(String::as_str).collect();
        loaded.sort_unstable();
        return Err(ApiError::NotFound(format!(
            "\"model\" is required; loaded: {}",
            loaded.join(", ")
        )));
    };
    guard
        .resolve(Some(requested))
        .ok_or_else(|| ApiError::NotFound(format!("model not found: {requested:?}")))
}

/// OpenAI-shaped `/v1/embeddings`.
///
/// The model has to be an encoder: a chat model loaded here has embeddings
/// disabled, so the request fails with a message that says which to load.
async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let texts = embed_inputs(&req.input)?;
    let (handle, model_id, active) = embed_engine(&state, req.model.as_deref()).await?;

    let result = tokio::task::spawn_blocking(move || {
        let _active = active;
        handle.embed(&texts)
    })
    .await??;

    let data: Vec<serde_json::Value> = result
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": v,
            })
        })
        .collect();

    Ok(Json(json!({
        "object": "list",
        "model": model_id,
        "data": data,
        "usage": {
            "prompt_tokens": result.tokens,
            "total_tokens": result.tokens,
        },
    })))
}

/// Accept `input` as a string or an array of strings, which is all OpenAI allows.
fn embed_inputs(input: &serde_json::Value) -> Result<Vec<String>, ApiError> {
    match input {
        serde_json::Value::String(s) => Ok(vec![s.clone()]),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    ApiError::NotFound("input array must contain only strings".into())
                })
            })
            .collect(),
        _ => Err(ApiError::NotFound(
            "input must be a string or an array of strings".into(),
        )),
    }
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    model: Option<String>,
    /// Keep only the best `top_n` results, as Cohere's API allows.
    #[serde(default)]
    top_n: Option<usize>,
}

/// Order documents by relevance, keeping submission order for ties.
///
/// A NaN score sorts last rather than first: `f32::total_cmp` puts NaN above
/// infinity, so a broken head would otherwise be reported as the best match. A
/// non-panicking comparator still matters — a panic here would take the server
/// down.
fn rank_order(scores: &[f32], top_n: Option<usize>) -> Vec<(usize, f32)> {
    let key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| key(b.1).total_cmp(&key(a.1)).then(a.0.cmp(&b.0)));
    if let Some(n) = top_n {
        ranked.truncate(n);
    }
    ranked
}

/// Cohere/Jina-shaped `/v1/rerank`.
async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.documents.is_empty() {
        return Err(ApiError::NotFound("documents must not be empty".into()));
    }

    let (handle, model_id, active) = {
        let mut guard = state.inner.lock().await;
        let id = require_model(&guard, req.model.as_deref())?;
        let class = guard
            .models
            .get(&id)
            .map(|l| l.info.model_class.clone())
            .unwrap_or_default();
        if class != "rerank" {
            return Err(ApiError::NotFound(format!(
                "{id} is a {class} model — load a reranker (class rerank) to rank documents",
            )));
        }
        let handle = guard.models.get(&id).expect("resolved").engine.handle();
        let active = guard.begin(&id).expect("resolved");
        (handle, id, active)
    };

    let query = req.query.clone();
    let documents = req.documents.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _active = active;
        handle.rerank(&query, &documents)
    })
    .await??;

    let ranked = rank_order(&result.scores, req.top_n);

    let results: Vec<serde_json::Value> = ranked
        .iter()
        .map(|(index, score)| {
            json!({
                "index": index,
                "relevance_score": score,
                "document": req.documents.get(*index),
            })
        })
        .collect();

    Ok(Json(json!({
        "model": model_id,
        "results": results,
        "usage": { "total_tokens": result.tokens },
    })))
}

#[derive(Deserialize)]
struct OllamaEmbedRequest {
    #[serde(default)]
    model: Option<String>,
    /// Ollama takes a string or a list here, like OpenAI.
    input: serde_json::Value,
}

/// `POST /api/embed` — Ollama's current embeddings route.
async fn ollama_embed(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let texts = embed_inputs(&req.input)?;
    let (handle, model_id, active) = embed_engine(&state, req.model.as_deref()).await?;
    let result = tokio::task::spawn_blocking(move || {
        let _active = active;
        handle.embed(&texts)
    })
    .await??;
    Ok(Json(json!({
        "model": model_id,
        "embeddings": result.vectors,
        "prompt_eval_count": result.tokens,
    })))
}

#[derive(Deserialize)]
struct OllamaLegacyEmbedRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
}

/// `POST /api/embeddings` — the original single-prompt route Ollama still serves.
async fn ollama_embeddings_legacy(
    State(state): State<AppState>,
    Json(req): Json<OllamaLegacyEmbedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (handle, _, active) = embed_engine(&state, req.model.as_deref()).await?;
    let texts = vec![req.prompt.clone()];
    let result = tokio::task::spawn_blocking(move || {
        let _active = active;
        handle.embed(&texts)
    })
    .await??;
    let embedding = result.vectors.into_iter().next().unwrap_or_default();
    Ok(Json(json!({ "embedding": embedding })))
}

/// Resolve a model for embedding, refusing anything that cannot embed.
async fn embed_engine(
    state: &AppState,
    requested: Option<&str>,
) -> ApiResult<(EngineHandle, String, InFlight)> {
    let mut guard = state.inner.lock().await;
    let id = require_model(&guard, requested)?;
    let class = guard
        .models
        .get(&id)
        .map(|l| l.info.model_class.clone())
        .unwrap_or_default();
    if class != "embedding" {
        return Err(ApiError::NotFound(format!(
            "{id} is a {class} model — load an embedding model (class embedding) to embed",
        )));
    }
    let engine = guard.models.get(&id).expect("resolved").engine.handle();
    let active = guard.begin(&id).expect("resolved");
    Ok((engine, id, active))
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let (handle, model_id, active) = chat_engine(&state, req.model.as_deref()).await?;

    let params = req.sampler();
    let msgs = req.chat_msgs()?;

    if !req.stream {
        let (text, stats) = tokio::task::spawn_blocking(move || {
            let _active = active;
            let mut out = String::new();
            let stats = handle.generate(&msgs, &params, |t| {
                out.push_str(t);
                true
            })?;
            Ok::<_, InferError>((out, stats))
        })
        .await??;

        return Ok(Json(json!({
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                // A reply cut off by `max_tokens` is not a finished reply, and
                // a client that stitches the pieces together has to be able to
                // tell the two apart.
                "finish_reason": if stats.hit_token_cap { "length" } else { "stop" },
            }],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.generated_tokens,
            },
            "stats": stats,
        }))
        .into_response());
    }

    // OpenAI's stream shape carries these on every chunk, and clients read
    // them: an `id` to group the reply, `object` to recognise it, `created` to
    // order it, and `model` to know what answered.
    let chat_id = next_stream_id();
    let created = unix_secs();
    let stream_id = chat_id.clone();
    let stream_model = model_id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::task::spawn_blocking(move || {
        // Held for as long as the stream runs, which outlives this handler by
        // however long the reply takes.
        let _active = active;
        let stream_tx = tx.clone();
        let mut first_chunk = true;
        let result = handle.generate(&msgs, &params, |t| {
            let mut delta = json!({ "content": t });
            if first_chunk {
                // The first chunk names the role, so a client assembling the
                // message knows who is speaking before any text arrives.
                delta["role"] = json!("assistant");
                first_chunk = false;
            }
            let ev = Event::default().data(
                json!({
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": stream_model,
                    "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                })
                .to_string(),
            );
            // `false` stops the generation. The send fails precisely when the
            // reader is gone, which is the moment carrying on would hold the
            // model's worker thread — and every request queued behind it — for
            // the rest of `max_tokens`.
            stream_tx.blocking_send(Ok(ev)).is_ok()
        });
        // A reader that hung up is how a stream normally ends, not a failure.
        if matches!(result, Err(InferError::ClientGone | InferError::Cancelled)) {
            return;
        }
        let final_ev = match result {
            Ok(stats) => {
                // See the non-streaming path: a truncated reply reports itself.
                let finish_reason = if stats.hit_token_cap {
                    "length"
                } else {
                    "stop"
                };
                Event::default().data(
                    json!({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
                        "stats": stats,
                    })
                    .to_string(),
                )
            }
            // Still an `event: error`, but the body is JSON now: a plain-text
            // payload cannot be parsed by the reader that is waiting for
            // chunks, so the failure would arrive as silence.
            Err(e) => Event::default().event("error").data(
                json!({ "error": { "message": e.to_string(), "type": "server_error" } })
                    .to_string(),
            ),
        };
        let _ = tx.blocking_send(Ok(final_ev));
        // The terminator OpenAI clients wait for. Without it they read until
        // the socket closes and report the stream as truncated.
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use infer_core::LoadConfig;
    use std::time::Duration;
    use tower::ServiceExt;

    type TestRequest = axum::http::Request<Body>;

    fn app(key: Option<&str>) -> Router {
        let state = AppState::new(Duration::from_secs(0), LoadConfig::new(""));
        router(state, key.map(|k| Arc::new(k.to_string())), Vec::new())
    }

    /// `get` would shadow `axum::routing::get`, so this is named `probe`.
    fn probe(path: &str, token: Option<&str>) -> TestRequest {
        let mut req = TestRequest::new(Body::empty());
        *req.uri_mut() = path.parse().expect("valid uri");
        if let Some(t) = token {
            let value = HeaderValue::from_str(&format!("Bearer {t}")).expect("valid header");
            req.headers_mut().insert(AUTHORIZATION, value);
        }
        req
    }

    async fn status_of(key: Option<&str>, path: &str, token: Option<&str>) -> StatusCode {
        app(key).oneshot(probe(path, token)).await.unwrap().status()
    }

    #[tokio::test]
    async fn protected_routes_reject_missing_and_wrong_token() {
        assert_eq!(
            status_of(Some("secret"), "/admin/status", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_of(Some("secret"), "/admin/status", Some("nope")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_of(Some("secret"), "/v1/models", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn protected_routes_accept_the_right_token() {
        assert_eq!(
            status_of(Some("secret"), "/admin/status", Some("secret")).await,
            StatusCode::OK
        );
        // A prefix of the token must not pass.
        assert_eq!(
            status_of(Some("secret"), "/admin/status", Some("sec")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_of(Some("secret"), "/admin/status", Some("secretx")).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn health_stays_open_for_probes() {
        assert_eq!(
            status_of(Some("secret"), "/health", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn no_token_means_no_auth() {
        assert_eq!(status_of(None, "/admin/status", None).await, StatusCode::OK);
        assert_eq!(
            status_of(None, "/admin/status", Some("anything")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn unauthorized_body_explains_itself() {
        let res = app(Some("secret"))
            .oneshot(probe("/admin/status", None))
            .await
            .unwrap();
        assert_eq!(
            res.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer realm=\"nexllama\""
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("bearer token"));
    }

    fn parts(json: serde_json::Value) -> ApiResult<(String, Vec<Vec<u8>>)> {
        flatten_content(&json)
    }

    #[test]
    fn plain_string_content_passes_through() {
        let (text, images) = parts(json!("hello")).unwrap();
        assert_eq!(text, "hello");
        assert!(images.is_empty());
    }

    #[test]
    fn text_and_image_parts_are_split() {
        // "hi" in base64 is a 3-byte payload.
        let (text, images) = parts(json!([
            { "type": "text", "text": "what is this?" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGk=" } },
        ]))
        .unwrap();
        assert_eq!(text, "what is this?");
        assert_eq!(images, vec![b"hi".to_vec()]);
    }

    #[test]
    fn several_images_keep_their_order() {
        let (_, images) = parts(json!([
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,YQ==" } },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,Yg==" } },
        ]))
        .unwrap();
        assert_eq!(images, vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn remote_and_local_urls_are_refused() {
        for url in [
            "http://169.254.169.254/latest/meta-data/",
            "https://example.com/cat.png",
            "file:///C:/Windows/win.ini",
            "C:\\Windows\\win.ini",
            "/etc/passwd",
        ] {
            let err = decode_image_url(url).unwrap_err();
            assert!(err.contains("data:"), "{url} should be refused: {err}");
        }
    }

    #[test]
    fn base64_decoding_handles_padding_and_whitespace() {
        assert_eq!(base64_decode("aGk=").unwrap(), b"hi");
        assert_eq!(base64_decode("aGk").unwrap(), b"hi");
        assert_eq!(base64_decode("aG\nk=").unwrap(), b"hi");
        assert_eq!(base64_decode("YQ==").unwrap(), b"a");
        assert!(base64_decode("!!").is_none());
    }

    #[test]
    fn unknown_part_types_are_named_in_the_error() {
        let err = parts(json!([{ "type": "video_url", "video_url": { "url": "x" } }])).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("video_url"), "{msg}");
    }

    #[test]
    fn embed_input_accepts_a_string_or_a_list() {
        assert_eq!(embed_inputs(&json!("hello")).unwrap(), vec!["hello"]);
        assert_eq!(
            embed_inputs(&json!(["a", "b"])).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(embed_inputs(&json!([])).unwrap().is_empty());
    }

    #[test]
    fn embed_input_refuses_anything_else() {
        for bad in [
            json!(12),
            json!({"a": 1}),
            json!([1, 2]),
            json!(["ok", 3]),
            json!(null),
        ] {
            assert!(embed_inputs(&bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn rank_order_puts_the_best_document_first() {
        let order = rank_order(&[0.1, 0.9, 0.5], None);
        assert_eq!(order, vec![(1, 0.9), (2, 0.5), (0, 0.1)]);
    }

    #[test]
    fn rank_order_keeps_submission_order_for_ties() {
        assert_eq!(
            rank_order(&[0.5, 0.5, 0.5], None),
            vec![(0, 0.5), (1, 0.5), (2, 0.5)]
        );
    }

    #[test]
    fn rank_order_honours_top_n() {
        assert_eq!(
            rank_order(&[0.1, 0.9, 0.5], Some(2)),
            vec![(1, 0.9), (2, 0.5)]
        );
        assert!(rank_order(&[0.1, 0.9, 0.5], Some(0)).is_empty());
        // Asking for more than there is returns everything.
        assert_eq!(rank_order(&[0.1, 0.9], Some(99)).len(), 2);
    }

    #[test]
    fn rank_order_survives_a_nan_without_panicking() {
        let order = rank_order(&[f32::NAN, 0.4], None);
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], (1, 0.4), "a NaN score must not top the list");
        assert!(order[1].1.is_nan());
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        let now = rfc3339_now();
        // 2026-10-08T12:34:56Z
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.ends_with('Z'), "{now}");
        assert_eq!(&now[4..5], "-", "{now}");
        assert_eq!(&now[10..11], "T", "{now}");
        let year: i32 = now[..4].parse().unwrap_or(0);
        assert!((2020..2100).contains(&year), "implausible year in {now}");
    }

    #[test]
    fn ollama_options_map_onto_the_sampler() {
        let base = SamplerParams::default();
        let options = json!({
            "temperature": 0.1,
            "top_k": 7,
            "num_predict": 33,
            "seed": 9,
            "ignored": "nope"
        });
        let p = sampler_from_options(&base, Some(&options));
        assert_eq!(p.temperature, 0.1);
        assert_eq!(p.top_k, 7);
        assert_eq!(p.max_tokens, 33);
        assert_eq!(p.seed, 9);
        // Untouched knobs keep their defaults.
        assert_eq!(p.top_p, base.top_p);
    }

    #[test]
    fn ollama_options_never_shrink_the_token_budget_to_nothing() {
        let base = SamplerParams::default();
        // Ollama treats num_predict <= 0 as "until it stops", and so does a
        // max_tokens of 0 here: neither is a literal zero-token budget.
        let p = sampler_from_options(&base, Some(&json!({ "num_predict": 0 })));
        assert_eq!(p.max_tokens, base.max_tokens);
        assert_eq!(p.token_cap(), usize::MAX);
        let p = sampler_from_options(&base, Some(&json!({ "num_predict": -1 })));
        assert_eq!(p.max_tokens, base.max_tokens);
        // No options at all is the base as-is.
        assert_eq!(
            sampler_from_options(&base, None).max_tokens,
            base.max_tokens
        );
    }

    fn spec() -> serde_json::Value {
        serde_json::from_str(include_str!("../openapi.json")).expect("openapi.json parses")
    }

    #[test]
    fn the_specification_is_valid_json_and_self_describing() {
        let s = spec();
        assert_eq!(s["openapi"], "3.1.0");
        assert_eq!(s["info"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(s["paths"].as_object().is_some_and(|p| p.len() > 10));
        // Everything that references a schema must reference one that exists.
        let schemas = s["components"]["schemas"]
            .as_object()
            .expect("schemas")
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let raw = include_str!("../openapi.json");
        for part in raw.split("\"#/components/schemas/").skip(1) {
            let name = part.split('"').next().unwrap_or_default();
            assert!(schemas.contains(name), "dangling schema reference: {name}");
        }
    }

    /// Every documented route must actually be routed.
    ///
    /// Nothing introspects axum's router, so this walks the specification and
    /// checks each entry does not 404 — the failure that would otherwise ship a
    /// doc describing an endpoint that was renamed away.
    #[tokio::test]
    async fn every_documented_route_exists() {
        let s = spec();
        let paths = s["paths"].as_object().expect("paths").clone();
        let app = app(None);

        for (path, methods) in paths {
            for method in methods.as_object().expect("methods").keys() {
                let verb = match method.as_str() {
                    "get" => axum::http::Method::GET,
                    "post" => axum::http::Method::POST,
                    other => panic!("unexpected method {other}"),
                };
                let mut req = TestRequest::new(Body::empty());
                *req.method_mut() = verb;
                *req.uri_mut() = path.parse().expect("uri");
                req.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );

                let status = app.clone().oneshot(req).await.unwrap().status();
                assert_ne!(
                    status,
                    StatusCode::NOT_FOUND,
                    "{method} {path} is documented but not routed"
                );
            }
        }
    }

    #[tokio::test]
    async fn the_banner_and_spec_are_open_even_with_a_token_set() {
        // Clients bootstrap from these before they have the key.
        for path in ["/", "/openapi.json", "/docs", "/api/version", "/health"] {
            assert_eq!(
                status_of(Some("secret"), path, None).await,
                StatusCode::OK,
                "{path} should be open"
            );
        }
    }

    async fn post_status(path: &str, body: serde_json::Value) -> StatusCode {
        let mut req = TestRequest::new(Body::from(serde_json::to_vec(&body).expect("body")));
        *req.method_mut() = Method::POST;
        *req.uri_mut() = path.parse().expect("uri");
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        app(None).oneshot(req).await.unwrap().status()
    }

    /// A size that cannot be allocated is refused here rather than handed to
    /// llama.cpp, where an oversized `n_ctx` is an allocation failure that
    /// takes the whole process down.
    #[tokio::test]
    async fn load_refuses_sizes_that_cannot_be_allocated() {
        for n_ctx in [0u32, limits::N_CTX + 1, u32::MAX] {
            assert_eq!(
                post_status("/admin/load", json!({ "model": "m.gguf", "n_ctx": n_ctx })).await,
                StatusCode::BAD_REQUEST,
                "n_ctx {n_ctx} must be refused"
            );
        }
        assert_eq!(
            post_status(
                "/admin/load",
                json!({ "model": "m.gguf", "n_batch": 128, "n_ubatch": 512 })
            )
            .await,
            StatusCode::BAD_REQUEST,
            "a micro-batch larger than the batch must be refused"
        );
        assert_eq!(
            post_status("/admin/load", json!({ "model": "m.gguf", "threads": -1 })).await,
            StatusCode::BAD_REQUEST,
            "a negative thread count must be refused"
        );
    }

    /// A browser page from another site must not be able to drive this server,
    /// which is what `CorsLayer::permissive()` allowed.
    #[tokio::test]
    async fn cors_answers_only_known_origins() {
        let from = |origin: &str| {
            let mut req = TestRequest::new(Body::empty());
            *req.uri_mut() = "/v1/models".parse().expect("uri");
            req.headers_mut().insert(
                header::ORIGIN,
                HeaderValue::from_str(origin).expect("origin"),
            );
            req
        };

        let res = app(None)
            .oneshot(from("https://elsewhere.example"))
            .await
            .unwrap();
        assert!(
            res.headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none(),
            "a site that is not this app gets no cross-origin grant"
        );

        let res = app(None)
            .oneshot(from("http://127.0.0.1:5173"))
            .await
            .unwrap();
        assert_eq!(
            res.headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .and_then(|v| v.to_str().ok()),
            Some("http://127.0.0.1:5173"),
            "the dev server is one of ours"
        );
    }
}
