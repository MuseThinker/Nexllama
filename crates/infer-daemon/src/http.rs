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
    ChatMsg, Engine, EngineHandle, FitReport, FlashAttn, InferError, LoadConfig, LoadMode,
    LoraAdapter, SamplerParams,
};

use crate::cloud;
use crate::state::{record_stats, AppState, DaemonState, Dormant, InFlight, Loaded, RuntimeStats};

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

pub fn router(
    state: AppState,
    api_key: ApiKey,
    cors_origins: Vec<String>,
    ui_dir: Option<std::path::PathBuf>,
) -> Router {
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
        .route("/admin/fit", post(fit_check))
        // The models folder, read with the same rules the desktop library uses
        // (`model-index`), so a browser opened against this daemon shows the
        // library the app would show. Only when `--models-dir` named one.
        .route("/admin/models", get(list_models_folder))
        // Cloud providers: pushed by the desktop app over loopback. The key is
        // write-only — `GET` lists the providers but never their keys.
        .route("/admin/providers", get(list_providers).post(set_providers))
        .route("/admin/providers/test", post(test_provider))
        // The `auto` alias's preference: local-first (default) or cloud-first.
        .route("/admin/routing", get(get_routing).post(set_routing))
        .route_layer(middleware::from_fn_with_state(api_key, require_key));

    let api = Router::new()
        .route("/openapi.json", get(openapi))
        .route("/docs", get(docs))
        .route("/api/version", get(ollama_version))
        .route("/health", get(health))
        .merge(protected);

    // With a UI to serve, `/` is the app rather than the banner: the banner is
    // where you go when there is no UI, and an API-only daemon on a port with
    // nothing else on it should still say what it is. The API is merged on top
    // either way, so the UI and the engine are one port.
    let app = match ui_dir {
        Some(dir) => crate::ui::router(dir).merge(api),
        None => api.route("/", get(root)),
    };

    app.layer(cors_layer(&cors_origins))
        .layer(RequestBodyLimitLayer::new(limits::BODY_BYTES))
        .with_state(state)
}

/// The models folder behind `GET /admin/models`.
///
/// Answers `{"dir": null, "models": []}` when no folder was configured rather
/// than a 404: the caller is a UI asking "what is there?", and "there is no
/// folder configured" is a fact about this daemon, not a missing endpoint. A
/// folder that cannot be read is an error with the reason — that is a volume
/// that went missing, and saying so beats an empty list that looks like an empty
/// library.
async fn list_models_folder(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(dir) = state.models_dir.as_deref().cloned() else {
        return Ok(Json(json!({ "dir": null, "models": [] })));
    };
    // Reading GGUF headers is file IO, and a library on a slow disk can hold a
    // worker for a while. Off the async runtime so a scan cannot delay the
    // chat completions running next to it.
    let models = tokio::task::spawn_blocking(move || model_index::scan(&dir))
        .await
        .map_err(|e| ApiError::Worker(format!("listing models failed: {e}")))?
        .map_err(ApiError::Worker)?;
    Ok(Json(json!({
        "dir": state.models_dir.as_deref().map(|d| d.to_string_lossy().to_string()),
        "models": models,
    })))
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
            "admin": ["GET /health", "GET /admin/status", "POST /admin/load", "POST /admin/unload", "POST /admin/fit"]
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
    /// A cloud provider returned an error (or could not be reached). Carries the
    /// provider's own status so the client sees, say, a 401 as a 401.
    Upstream(StatusCode, String),
}

impl ApiError {
    fn code(&self) -> StatusCode {
        match self {
            ApiError::NotLoaded => StatusCode::CONFLICT,
            ApiError::NotFound(_) | ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Model(_) | ApiError::Worker(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Upstream(code, _) => *code,
        }
    }

    fn message(&self) -> String {
        match self {
            ApiError::NotLoaded => "no model loaded".to_string(),
            ApiError::NotFound(m)
            | ApiError::BadRequest(m)
            | ApiError::Model(m)
            | ApiError::Worker(m)
            | ApiError::Upstream(_, m) => m.clone(),
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
    let mut models: Vec<serde_json::Value> = guard
        .models
        .values()
        .map(|l| {
            // Read under its own lock and tolerate poisoning: a model that has
            // already answered should not be able to take /admin/status down
            // with it over a statistic.
            let perf = l.stats.lock().ok().map(|s| {
                json!({
                    "requests": s.requests,
                    "tokens": s.tokens,
                    "tokens_per_second": round2(s.tokens_per_second),
                    "first_token_ms": round2(s.first_token_ms),
                    "draft_acceptance": s.draft_acceptance.map(round2),
                })
            });
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
                "perf": perf,
                "loaded": true,
            })
        })
        .collect();
    // Models freed while idle are still names a client can ask for — the first
    // request reloads them — so they are listed too, marked as not loaded.
    models.extend(guard.dormant.values().map(|d| {
        json!({
            "id": d.id,
            "path": d.path,
            "info": d.info,
            "n_gpu_layers": d.n_gpu_layers,
            "n_ctx": d.n_ctx,
            "load_mtp": d.load_mtp,
            "mmproj": d.mmproj,
            "draft_model": d.draft_model,
            "loras": d.loras,
            "loaded_secs_ago": 0,
            "idle_secs": 0,
            "perf": serde_json::Value::Null,
            "loaded": false,
        })
    }));
    // Cloud providers are callable through this same port, so list them here
    // too — with `model_class: llm` so a chat client keeps them, and `cloud`
    // set so the UI can tell them from the local ones. The `auto` alias is
    // deliberately *not* here: it is an API-side routing policy, not a model to
    // pick from the library or the chat dropdown.
    let providers = state
        .cloud
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for p in providers.iter() {
        for m in &p.models {
            models.push(json!({
                "id": format!("{}/{}", p.id, m),
                "info": { "model_class": "llm" },
                "cloud": true,
                "provider": p.name,
                "loaded": true,
            }));
        }
    }
    Json(json!({
        "models": models,
        "default_model": guard.default_model,
        "keep_alive_secs": guard.keep_alive.as_secs(),
    }))
}

#[derive(Deserialize)]
struct FitRequest {
    /// Path to a GGUF already on disk.
    #[serde(default)]
    model: Option<String>,
    /// Context to measure against; `0` (the default) sizes it automatically.
    #[serde(default)]
    n_ctx: Option<u32>,
    /// Size in bytes of a file that is *not* on disk yet.
    ///
    /// Given instead of `model` when there is nothing to open — a download that
    /// has not started. Checking that is the only way to warn before several
    /// gigabytes have already been transferred.
    #[serde(default)]
    size_bytes: Option<u64>,
}

/// `POST /admin/fit` — measure one checkpoint against this machine.
///
/// Advisory in every field, and it never refuses a model on its own. The whole
/// point of measuring is that guessing has already been tried: an auto-tuner
/// that offloads too much gets the process killed part-way through a load,
/// which is a worse outcome than loading something that turns out slow. So this
/// reports, and the caller decides.
async fn fit_check(
    State(state): State<AppState>,
    Json(req): Json<FitRequest>,
) -> Result<Json<FitReport>, ApiError> {
    // A size needs no file, which is the point: it can be checked before the
    // download that would have brought the file.
    if let Some(bytes) = req.size_bytes {
        let hardware = tokio::task::spawn_blocking(infer_core::hwprofile::probe)
            .await
            .map_err(|e| ApiError::Model(format!("hardware probe failed: {e}")))?
            .map_err(|e| ApiError::Model(e.to_string()))?;
        return Ok(Json(infer_core::fit::from_size(bytes, hardware)));
    }

    let Some(model) = req.model else {
        return Err(ApiError::BadRequest(
            "fit needs either `model` (a path) or `size_bytes` (a size)".into(),
        ));
    };
    let path = std::path::PathBuf::from(model);
    let n_ctx = req.n_ctx.unwrap_or(0);
    // llama.cpp's fitter loads the checkpoint a second time inside this
    // process, which fails while another model is resident — so an honest
    // "not measured" beats an attempt that ends in a null result.
    let resident = { state.inner.lock().await.models.len() };
    // Measuring reads the checkpoint's metadata, which is real file input on a
    // multi-gigabyte file; it must not run on the async runtime.
    let report = tokio::task::spawn_blocking(move || {
        if resident > 0 {
            infer_core::fit::fit_unmeasured(
                &path,
                n_ctx,
                "not measured: a model is resident, and measuring would load this checkpoint a second time in the same process; unload to measure",
            )
        } else {
            infer_core::fit::fit(&path, n_ctx)
        }
    })
    .await
    .map_err(|e| ApiError::Model(format!("fit task failed: {e}")))?
    .map_err(|e| ApiError::Model(e.to_string()))?;
    Ok(Json(report))
}

/// A residency slot: one model of each kind stays resident, and loading
/// another of the same kind releases the first.
///
/// Chat and vision are separate slots even though both are `llm` class,
/// because a projector model answers a different kind of request and the two
/// are routinely wanted at once — an OCR pass feeding a chat turn.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelSlot {
    Chat,
    Vision,
    Embedding,
    Rerank,
}

impl ModelSlot {
    fn label(self) -> &'static str {
        match self {
            ModelSlot::Chat => "the other chat model",
            ModelSlot::Vision => "the other vision model",
            ModelSlot::Embedding => "the other embedding model",
            ModelSlot::Rerank => "the other rerank model",
        }
    }
}

/// The slot a checkpoint occupies, read from its header alone.
///
/// `vision` is passed in when the caller already knows — a bundle being loaded
/// with its projector — because the projector file is what makes it one, and
/// that pairing is only known at load time.
fn model_slot_for(path: &std::path::Path, vision: Option<bool>) -> Option<ModelSlot> {
    let class = infer_core::model_class_of(path);
    match class {
        "embedding" => Some(ModelSlot::Embedding),
        "rerank" => Some(ModelSlot::Rerank),
        // An unreadable file classifies as a chat model, which is the safe
        // default: it will simply be asked to serve as one.
        _ => Some(if vision.unwrap_or(false) {
            ModelSlot::Vision
        } else {
            ModelSlot::Chat
        }),
    }
}

/// The slot of a model that is already loaded.
fn slot_of(info: &infer_core::EngineInfo) -> ModelSlot {
    match info.model_class.as_str() {
        "embedding" => ModelSlot::Embedding,
        "rerank" => ModelSlot::Rerank,
        _ if info.vision => ModelSlot::Vision,
        _ => ModelSlot::Chat,
    }
}

/// Release every resident model sharing `slot` but `keep`, and return them.
async fn park_models_in_slot(state: &AppState, keep: &str, slot: ModelSlot) -> Vec<Loaded> {
    let mut guard = state.inner.lock().await;
    let stale: Vec<String> = guard
        .models
        .iter()
        .filter(|(k, m)| k.as_str() != keep && slot_of(&m.info) == slot)
        .map(|(k, _)| k.clone())
        .collect();
    let mut parked = Vec::new();
    for other in stale {
        if let Some(l) = guard.models.remove(&other) {
            tracing::info!(%other, "released: one model per slot is resident at a time");
            if guard.default_model.as_deref() == Some(other.as_str()) {
                // The chat panel should follow the model the caller just asked
                // for, not keep pointing at the one it replaced.
                guard.default_model = None;
            }
            guard.dormant.insert(other, Dormant::from_loaded(&l));
            parked.push(l);
        }
    }
    parked
}

/// Release every resident model except `keep`, parking rather than dropping.
///
/// Used when a load has already failed: the device can be full because *another
/// slot's* model is sitting on it, and a caller switching models should not have
/// to know which one is in the way. Parking keeps them reloadable, so the worst
/// case is that they are listed as released rather than resident.
async fn release_all_except(state: &AppState, keep: &str) -> Vec<Loaded> {
    let mut guard = state.inner.lock().await;
    let others: Vec<String> = guard
        .models
        .keys()
        .filter(|k| k.as_str() != keep)
        .cloned()
        .collect();
    let mut freed = Vec::new();
    for id in others {
        if let Some(l) = guard.models.remove(&id) {
            if guard.default_model.as_deref() == Some(id.as_str()) {
                guard.default_model = None;
            }
            guard.dormant.insert(id, Dormant::from_loaded(&l));
            freed.push(l);
        }
    }
    freed
}

/// Load on a worker thread, naming the model in the log either way.
async fn spawn_engine(cfg: LoadConfig, log_id: &str) -> Result<infer_core::Engine, InferError> {
    let log_id = log_id.to_string();
    match tokio::task::spawn_blocking(move || {
        Engine::load(cfg, |ev| tracing::info!(?log_id, ?ev, "load"))
    })
    .await
    {
        Ok(result) => result,
        Err(e) => Err(InferError::Backend(format!("load task failed: {e}"))),
    }
}

/// What to tell someone whose load just failed.
///
/// llama.cpp reports a failed allocation the same way whatever the cause — the
/// binding surfaces it as `NullResult` — so the useful information has to be
/// gathered here: what is already resident, how much device memory is left, and
/// whether the file is still where it was. A load that fails with the whole
/// picture in one line can be acted on; the same failure without it can only be
/// repeated.
async fn load_failure_context(path: &std::path::Path, state: &AppState) -> String {
    let resident = {
        let guard = state.inner.lock().await;
        guard
            .models
            .values()
            .map(|m| m.id.clone())
            .collect::<Vec<_>>()
    };
    let (gpu_free, ram_free) = match infer_core::hwprofile::probe() {
        Ok(h) => (
            h.best_gpu().map(|g| g.vram_free_mb).unwrap_or(0),
            h.ram_free_mb,
        ),
        Err(_) => (0, 0),
    };

    let mut facts = Vec::new();
    if !resident.is_empty() {
        facts.push(format!(
            "{} model(s) resident: {}",
            resident.len(),
            resident.join(", ")
        ));
    }
    if gpu_free > 0 {
        facts.push(format!("{gpu_free} MiB of video memory free"));
    }
    if ram_free > 0 {
        facts.push(format!("{ram_free} MiB of system memory free"));
    }
    if !path.exists() {
        facts.push(format!("{} is no longer at that path", path.display()));
    }
    if facts.is_empty() {
        return String::new();
    }
    format!(
        " — {}. Offload fewer layers, use a smaller quantisation, or unload a model first.",
        facts.join("; ")
    )
}

/// The local generation model to route to, if one is loaded or remembered.
fn local_llm(guard: &DaemonState) -> Option<String> {
    guard
        .models
        .values()
        .find(|l| l.info.model_class == "llm")
        .map(|l| l.id.clone())
        .or_else(|| {
            guard
                .dormant
                .values()
                .find(|d| d.info.model_class == "llm")
                .map(|d| d.id.clone())
        })
}

/// The first cloud model, as `<provider>/<model>`, if any provider is configured.
fn first_cloud(providers: &[cloud::Provider]) -> Option<String> {
    providers
        .iter()
        .find_map(|p| p.models.first().map(|m| format!("{}/{}", p.id, m)))
}

/// The `auto` alias, when it is enabled. It names a policy rather than a
/// checkpoint, so it is only offered to API clients (`/v1/models`) while on.
fn alias_entries(
    guard: &DaemonState,
    providers: &[cloud::Provider],
    enabled: bool,
) -> Vec<serde_json::Value> {
    if !enabled {
        return Vec::new();
    }
    let local = local_llm(guard).unwrap_or_else(|| "—".to_string());
    let cloud = first_cloud(providers).unwrap_or_else(|| "—".to_string());
    vec![serde_json::json!({
        "id": "auto",
        "object": "model",
        "alias": true,
        "info": { "model_class": "llm" },
        "loaded": true,
        "routes_to": format!("{local} / {cloud}"),
    })]
}

/// Whether a loaded or remembered local model can see images.
fn local_has_vision(guard: &DaemonState, id: &str) -> bool {
    guard
        .models
        .get(id)
        .map(|l| l.info.vision)
        .or_else(|| guard.dormant.get(id).map(|d| d.info.vision))
        .unwrap_or(false)
}

/// Whether a request carries any image, read cheaply from the raw body.
fn body_has_images(body: &serde_json::Value) -> bool {
    body.get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_array())
                    .is_some_and(|parts| {
                        parts.iter().any(|p| {
                            matches!(
                                p.get("type").and_then(|t| t.as_str()),
                                Some("image_url") | Some("input_image")
                            )
                        })
                    })
            })
        })
}

/// Decide which concrete model `auto` means.
///
/// Local-first unless `prefer` is `cloud`, and skips a local model that cannot
/// see an image the request carries.
fn route_alias(
    guard: &DaemonState,
    providers: &[cloud::Provider],
    prefer: &str,
    has_images: bool,
) -> Option<String> {
    let local = local_llm(guard);
    let cloud = first_cloud(providers);
    let local_vision = local
        .as_deref()
        .is_some_and(|id| local_has_vision(guard, id));
    route_choice(prefer, local, cloud, local_vision, has_images)
}

/// The pure decision behind [`route_alias`], split out so it can be tested
/// without a loaded engine.
fn route_choice(
    prefer: &str,
    local: Option<String>,
    cloud: Option<String>,
    local_vision: bool,
    has_images: bool,
) -> Option<String> {
    if prefer == "cloud" && cloud.is_some() {
        return cloud;
    }
    local.filter(|_| !has_images || local_vision).or(cloud)
}

/// What [`expand_alias`] did, so `chat` can fall back the right way.
#[derive(Default, Clone, Copy)]
struct AliasRoute {
    /// The request named `auto` (and it is enabled).
    auto: bool,
    /// `auto` resolved to a local model.
    local: bool,
}

/// Expand the `auto` alias in `body["model"]` to a concrete id.
async fn expand_alias(state: &AppState, body: &mut serde_json::Value) -> ApiResult<AliasRoute> {
    if body.get("model").and_then(serde_json::Value::as_str) != Some("auto") {
        return Ok(AliasRoute::default());
    }
    let (enabled, prefer) = state
        .routing
        .read()
        .map(|r| (r.enabled, r.prefer.clone()))
        .unwrap_or_else(|_| (true, "local".to_string()));
    if !enabled {
        return Err(ApiError::NotFound(
            "auto routing is off; enable it in the app, or name a concrete model".into(),
        ));
    }
    let has_images = body_has_images(body);
    let target = {
        let guard = state.inner.lock().await;
        let providers = state
            .cloud
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        route_alias(&guard, providers.as_slice(), &prefer, has_images)
    };
    match target {
        Some(id) => {
            // A cloud id carries a provider prefix; a local one has no slash.
            let local = !id.contains('/');
            body["model"] = json!(id);
            Ok(AliasRoute { auto: true, local })
        }
        None => Err(ApiError::NotFound(
            "auto: nothing to route to — load a local model or add a cloud provider".into(),
        )),
    }
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let guard = state.inner.lock().await;
    let mut data: Vec<serde_json::Value> = guard
        .models
        .values()
        .map(|l| {
            json!({
                "id": l.id,
                "object": "model",
                "path": l.path,
                "info": l.info,
                "loaded": true,
            })
        })
        .collect();
    // Freed models are still names a client can request (they reload on use).
    data.extend(guard.dormant.values().map(|d| {
        json!({
            "id": d.id,
            "object": "model",
            "path": d.path,
            "info": d.info,
            "loaded": false,
        })
    }));
    drop(guard);
    // Cloud providers, snapped so the alias list can describe them too.
    let providers: Vec<cloud::Provider> = state.cloud.read().map(|p| p.clone()).unwrap_or_default();
    // The `auto` alias, listed after the concrete names while it is enabled, so
    // an API client can ask for the routing policy by name.
    let enabled = state.routing.read().map(|r| r.enabled).unwrap_or(true);
    let aliases = {
        let guard = state.inner.lock().await;
        alias_entries(&guard, &providers, enabled)
    };
    data.extend(aliases);
    for p in &providers {
        for m in &p.models {
            data.push(json!({
                "id": format!("{}/{}", p.id, m),
                "object": "model",
                "owned_by": p.id,
                "provider": p.name,
                "cloud": true,
            }));
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

/// `POST /admin/providers` — replace the cloud provider set.
///
/// The desktop app owns this list; it reads the keys from the OS credential
/// store and pushes them here over loopback. Keys live only in memory.
async fn set_providers(
    State(state): State<AppState>,
    Json(list): Json<Vec<cloud::ProviderIn>>,
) -> ApiResult<Json<serde_json::Value>> {
    let providers: Vec<cloud::Provider> = list
        .into_iter()
        .filter(|p| !p.id.trim().is_empty() && !p.base_url.trim().is_empty())
        .map(|p| cloud::Provider {
            id: p.id,
            name: if p.name.trim().is_empty() {
                p.base_url.clone()
            } else {
                p.name
            },
            base_url: p.base_url,
            api_key: p.api_key,
            models: p.models,
        })
        .collect();
    let count = providers.len();
    let ids: Vec<String> = providers.iter().map(|p| p.id.clone()).collect();
    *state
        .cloud
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = providers;
    // Ids only — never a key.
    tracing::info!(providers = ?ids, "cloud providers updated");
    Ok(Json(json!({ "status": "ok", "count": count })))
}

/// `GET /admin/providers` — the set, without any key.
async fn list_providers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let providers: Vec<serde_json::Value> = state
        .cloud
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|p| {
            json!({
                "id": p.id,
                "name": p.name,
                "base_url": p.base_url,
                "models": p.models,
                "has_key": !p.api_key.is_empty(),
            })
        })
        .collect();
    Json(json!({ "providers": providers }))
}

#[derive(Deserialize)]
struct RoutingRequest {
    /// Turn the `auto` alias on or off. Omitted leaves it unchanged.
    #[serde(default)]
    enabled: Option<bool>,
    /// `"local"` (default) or `"cloud"`. Omitted leaves it unchanged.
    #[serde(default)]
    prefer: Option<String>,
}

/// `GET /admin/routing` — the `auto` alias's current on/off and preference.
async fn get_routing(State(state): State<AppState>) -> Json<serde_json::Value> {
    let (enabled, prefer) = state
        .routing
        .read()
        .map(|r| (r.enabled, r.prefer.clone()))
        .unwrap_or_else(|_| (true, "local".to_string()));
    Json(json!({ "enabled": enabled, "prefer": prefer }))
}

/// `POST /admin/routing` — set the `auto` alias's policy (partial update).
async fn set_routing(
    State(state): State<AppState>,
    Json(req): Json<RoutingRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let mut routing = state
        .routing
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(enabled) = req.enabled {
        routing.enabled = enabled;
    }
    if let Some(prefer) = req.prefer.as_deref() {
        // Anything but an explicit "cloud" means local-first: the safe, cheap,
        // private default.
        routing.prefer = if prefer.trim() == "cloud" {
            "cloud"
        } else {
            "local"
        }
        .to_string();
    }
    Ok(Json(json!({
        "status": "ok",
        "enabled": routing.enabled,
        "prefer": routing.prefer,
    })))
}

#[derive(Deserialize)]
struct ProviderTest {
    base_url: String,
    #[serde(default)]
    api_key: String,
}

/// `POST /admin/providers/test` — ask a provider what it serves, before saving.
///
/// Takes an unsaved base URL and key so the UI can populate the model list (and
/// catch a bad key) while the user is still editing.
async fn test_provider(
    State(state): State<AppState>,
    Json(req): Json<ProviderTest>,
) -> ApiResult<Json<serde_json::Value>> {
    let models = cloud::fetch_models(&state.client, &req.base_url, &req.api_key).await?;
    Ok(Json(json!({ "status": "ok", "models": models })))
}

#[derive(Deserialize)]
struct LoadRequest {
    model: String,
    #[serde(default)]
    name: Option<String>,
    /// Refuse the load when the checkpoint measures as not fitting.
    ///
    /// Off by default. The measurement is advisory because being wrong about
    /// memory costs an out-of-memory kill mid-load, while a wrong warning costs
    /// nothing — so this only refuses when the caller explicitly asks it to.
    #[serde(default)]
    strict_fit: Option<bool>,
    /// Reuse the KV state of an already-encoded prefix for this model.
    #[serde(default)]
    prefix_cache: Option<bool>,
    /// How many prompts to remember.
    #[serde(default)]
    prefix_cache_entries: Option<u32>,
    /// Ceiling on remembered state, in MiB.
    #[serde(default)]
    prefix_cache_max_mb: Option<u32>,
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
    mtp_n_max: Option<u32>,
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
    /// Default reasoning mode for this model: `false` renders every prompt with
    /// `enable_thinking` off, `true`/omitted leaves it on. A per-model setting
    /// that a request may still override with its own `thinking`.
    #[serde(default)]
    thinking: Option<bool>,
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

/// Drop the MTP flags a request only inherited.
///
/// `cfg` is cloned from the daemon's startup config, which carries
/// `load_mtp`/`mtp_n_max` for the model the daemon came up on. A request that
/// supplies its own draft model is choosing the other way to draft, so those
/// inherited flags must not ride along — the engine refuses MTP and a separate
/// drafter together, and the error would name a conflict this request never
/// asked for. When the request *did* ask for MTP itself, both flags stay and
/// the engine reports the genuine conflict.
fn drop_inherited_mtp(cfg: &mut LoadConfig, asked_mtp: bool) {
    if cfg.draft_model.is_some() && !asked_mtp {
        cfg.load_mtp = false;
        cfg.mtp_n_max = 0;
    }
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
    // `n_ctx: 0` is not a size, it is "size it to the machine" — let it through
    // and the fit measurement below fills in what actually fits.
    if let Some(v) = req.n_ctx {
        if v > limits::N_CTX {
            return Err(ApiError::BadRequest(format!(
                "n_ctx must be between 0 and {}, got {v}",
                limits::N_CTX
            )));
        }
    }
    for (name, v, max) in [
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
    // Whether *this* request asked for MTP, as opposed to inheriting the flag
    // from the daemon's startup config. Read off the request before `cfg` is
    // touched.
    let asked_mtp = req.load_mtp.unwrap_or(false) || req.mtp_n_max.unwrap_or(0) > 0;
    if let Some(v) = req.load_mtp {
        cfg.load_mtp = v;
    }
    if let Some(v) = req.mtp_n_max {
        cfg.mtp_n_max = v;
        // Asking for a budget is asking for drafting, whatever the flag next
        // to it said; leaving `load_mtp` false would load nothing to draft with.
        if v > 0 {
            cfg.load_mtp = true;
        }
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
        drop_inherited_mtp(&mut cfg, asked_mtp);
    }
    let n_gpu_layers = cfg.n_gpu_layers;
    let load_mtp = cfg.load_mtp;

    let mut notes: Vec<String> = Vec::new();
    if load_mtp {
        notes.push(
            "the model's bundled MTP head is loaded; turn on speculative decoding per \
             request to draft with it"
                .into(),
        );
    }

    // Prefix reuse is opt-in per load, and off unless asked for. The budget is
    // in MiB because what a remembered prompt costs is its serialised KV state,
    // not its token count.
    if let Some(on) = req.prefix_cache {
        cfg.prefix_cache = on;
    }
    if let Some(n) = req.prefix_cache_entries {
        cfg.prefix_cache_entries = n;
    }
    if let Some(mb) = req.prefix_cache_max_mb {
        cfg.prefix_cache_max_mb = mb;
    }

    // Measured before anything is displaced: refusing a load must not cost the
    // caller the model that is already serving. A measurement that fails is
    // advisory too — `fit.rs` documents "never refuses a model on its own" and
    // an `Err` here would contradict that. We log and continue without a
    // report, so a transient backend/probe failure cannot brick loading.
    //
    // Measured only when the measurement is actually consulted — an automatic
    // context has to be sized from it, and `strict_fit` is by definition a
    // request to be told — and only when nothing else is resident. llama.cpp's
    // Fit advice is only consulted when `strict_fit` asks for it, because the
    // fitter answers by loading the checkpoint a second time inside this
    // process. That second load fails while another model is resident — it ends
    // in `invalid vector subscript` from inside llama.cpp and reaches the caller
    // as `model load: NullResult` — so it is refused rather than attempted, with
    // a sentence saying what to do instead.
    let fit: Option<FitReport> = if req.strict_fit.unwrap_or(false) {
        let others_resident = {
            let guard = state.inner.lock().await;
            guard.models.keys().any(|k| k != &id)
        };
        if others_resident {
            return Err(ApiError::BadRequest(
                "strict_fit asks for a measurement, and a measurement loads the checkpoint a \
                 second time in this process — which cannot be done while another model is \
                 resident. Unload the other models, or drop strict_fit and let the load be \
                 tried."
                    .into(),
            ));
        }
        match tokio::task::spawn_blocking({
            let path = cfg.model_path.clone();
            let n_ctx = cfg.n_ctx;
            move || infer_core::fit::fit(&path, n_ctx)
        })
        .await
        {
            Ok(Ok(r)) => Some(r),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "fit: could not measure; loading without fit advice");
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "fit task failed; loading without fit advice");
                None
            }
        }
    } else {
        None
    };

    if let Some(fit) = fit.as_ref() {
        if !fit.fits {
            let hint = match (fit.suggested_gpu_layers, fit.suggested_ctx) {
                (Some(layers), Some(ctx)) => {
                    format!(" (llama.cpp suggests {layers} layers, ctx {ctx})")
                }
                (Some(layers), None) => format!(" (llama.cpp suggests {layers} layers)"),
                _ => String::new(),
            };
            if req.strict_fit.unwrap_or(false) {
                return Err(ApiError::BadRequest(format!(
                    "{}{hint}; not loaded because strict_fit was requested",
                    fit.reason
                )));
            }
            notes.push(format!("warning: {}{hint}", fit.reason));
        }
    }

    // An automatic context (`n_ctx: 0`) is sized from the model's own limit and
    // the device it will run on — see `fit::auto_ctx`. Not from the fitter: that
    // would measure the checkpoint by loading it, which is both the reason a
    // second model could not be held and a second full load on every cold start.
    // The ladder needs nothing measured, and gives the same answer every time.
    let inbound_class = infer_core::model_class_of(std::path::Path::new(&req.model));
    if cfg.n_ctx == 0 {
        let hardware = infer_core::hwprofile::probe()
            .map_err(|e| ApiError::Model(format!("could not read this machine's memory: {e}")))?;
        cfg.n_ctx = infer_core::fit::auto_ctx(
            inbound_class,
            infer_core::fit::train_ctx(std::path::Path::new(&req.model)),
            infer_core::fit::ModelFootprint::of(std::path::Path::new(&req.model)),
            &hardware,
        );
        notes.push(format!(
            "context sized to {} tokens automatically ({} model, {} of video memory)",
            cfg.n_ctx,
            inbound_class,
            hardware
                .gpus
                .iter()
                .map(|g| g.vram_total_mb)
                .max()
                .map(|mb| format!("{} MiB", mb))
                .unwrap_or_else(|| "no dedicated".to_string()),
        ));
    }
    let n_ctx = cfg.n_ctx;

    // Drop any existing model with the same id before loading the new one.
    let old = {
        let mut guard = state.inner.lock().await;
        guard.models.remove(&id)
    };
    let _ = tokio::task::spawn_blocking(move || drop(old)).await;

    let path = cfg.model_path.clone();

    // One model per slot, where a slot is what a caller actually asks for. A
    // second chat model is wanted *instead of* the first, not beside it: two
    // 13 GB models do not share a 24 GB card, and the failure that produces is a
    // null result from deep inside the engine rather than anything the caller
    // can act on. Parked models keep their configuration in `dormant`, so the
    // next request that names one reloads it — the same path the idle reaper
    // uses, and nothing is lost if the load below fails.
    let incoming_slot = model_slot_for(&path, req.mmproj.as_ref().map(|_| true));
    if let Some(slot) = incoming_slot {
        let parked = park_models_in_slot(&state, &id, slot).await;
        if !parked.is_empty() {
            let names: Vec<String> = parked.iter().map(|m| m.id.clone()).collect();
            notes.push(format!(
                "released {} to make room (it reloads automatically when a request names it): {}",
                slot.label(),
                names.join(", ")
            ));
            let _ = tokio::task::spawn_blocking(move || drop(parked)).await;
        }
    }

    // Kept so the model can be loaded again if the idle reaper frees it.
    let config = cfg.clone();
    let log_id = id.clone();
    let engine = match spawn_engine(cfg.clone(), &log_id).await {
        Ok(engine) => engine,
        Err(first) => {
            // The slot rule above already released the model this one replaces,
            // so a failure here means the device itself is full. For a model the
            // caller is switching to, that is worth one retry with everything
            // else released — "switch to this model" should not require knowing
            // which other model is in the way. It is *not* worth it for an
            // encoder: those are small, and tearing down a chat model to make
            // room for 600 MB is a worse answer than saying what is missing.
            let may_release_all = matches!(
                incoming_slot,
                Some(ModelSlot::Chat) | Some(ModelSlot::Vision)
            );
            let freed = if may_release_all {
                release_all_except(&state, &id).await
            } else {
                Vec::new()
            };
            if freed.is_empty() {
                let context = load_failure_context(&path, &state).await;
                return Err(ApiError::Model(format!("{first}{context}")));
            }
            let names: Vec<String> = freed.iter().map(|m| m.id.clone()).collect();
            notes.push(format!(
                "released {} and retried: the device was too full to load alongside them",
                names.join(", ")
            ));
            let _ = tokio::task::spawn_blocking(move || drop(freed)).await;
            match spawn_engine(cfg, &log_id).await {
                Ok(engine) => engine,
                Err(second) => {
                    let context = load_failure_context(&path, &state).await;
                    return Err(ApiError::Model(format!("{second}{context}")));
                }
            }
        }
    };

    let info = engine.info().clone();
    // A drafter that would not load is reported rather than failing the load:
    // the model is up and usable, just without speculative decoding.
    if let Some(reason) = &info.draft_error {
        notes.push(reason.clone());
    } else if req.draft_model.is_some() {
        notes.push(
            "a draft model is loaded; turn on speculative decoding per request to have it \
             draft"
                .into(),
        );
    }
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
            config,
            loaded_at: now,
            last_used: now,
            n_gpu_layers,
            n_ctx,
            load_mtp,
            mmproj: req.mmproj.clone(),
            draft_model: req.draft_model.clone(),
            thinking: req.thinking,
            loras: req.lora.clone().unwrap_or_default(),
            in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            stats: std::sync::Arc::new(std::sync::Mutex::new(RuntimeStats::default())),
        },
    );
    // It is live again, so it is no longer a candidate for reload.
    guard.dormant.remove(&id);
    if guard.default_model.is_none() || req.set_default.unwrap_or(false) {
        guard.default_model = Some(id.clone());
    }
    drop(guard);
    if let Some(old) = displaced {
        let _ = tokio::task::spawn_blocking(move || drop(old)).await;
    }

    Ok(Json(
        json!({ "status": "ok", "id": id, "info": info, "notes": notes, "fit": fit }),
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
    let (removed, forgotten): (Vec<Loaded>, Vec<String>) = {
        let mut guard = state.inner.lock().await;
        match req.model.as_deref() {
            Some(id) => {
                let resident = guard.models.remove(id).into_iter().collect();
                if guard.default_model.as_deref() == Some(id) {
                    guard.default_model = None;
                }
                // Unloading is the user saying "I am done with this model", so
                // it also forgets the parked copy. Keeping it would leave the
                // model in `/v1/models` and would let a later request reload it,
                // which is the opposite of what unloading means.
                let forgotten: Vec<String> = guard
                    .dormant
                    .remove(id)
                    .into_iter()
                    .map(|_| id.to_string())
                    .collect();
                (resident, forgotten)
            }
            None => {
                let all: Vec<Loaded> = guard.models.drain().map(|(_, l)| l).collect();
                let forgotten: Vec<String> = guard.dormant.drain().map(|(id, _)| id).collect();
                guard.default_model = None;
                (all, forgotten)
            }
        }
    };
    let n = removed.len() + forgotten.len();
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
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
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
    /// Whether a reasoning model should think before answering. Omitted keeps
    /// the model's own default.
    #[serde(default)]
    thinking: Option<bool>,
    /// OpenAI `tools` array. Handed to the chat template so the model knows the
    /// tools exist and formats calls the way the template expects.
    #[serde(default)]
    tools: Option<serde_json::Value>,
    /// OpenAI `tool_choice` (`"auto"`, `"required"` or `"none"`).
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
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
        if let Some(v) = self.thinking {
            s.thinking = Some(v);
        }
        if let Some(tools) = &self.tools {
            if tools.as_array().is_some_and(|a| !a.is_empty()) {
                s.tools_json = Some(tools.to_string());
            }
        }
        // Only the string forms reach the parser; a named-function object is
        // left to the model's own judgement (the template still sees the tools).
        if let Some(serde_json::Value::String(choice)) = &self.tool_choice {
            s.tool_choice = Some(choice.clone());
        }
        if let Some(p) = self.parallel_tool_calls {
            s.parallel_tool_calls = p;
        }
        s
    }

    /// Flatten the request into engine messages, pulling out any attached images.
    fn chat_msgs(&self) -> ApiResult<Vec<ChatMsg>> {
        self.messages
            .iter()
            .map(|m| {
                let (text, images) = flatten_content(&m.content)?;
                let mut msg = ChatMsg::new(m.role.clone(), text).with_images(images);
                if let Some(calls) = &m.tool_calls {
                    msg = msg.with_tool_calls(calls.clone());
                }
                if let Some(id) = &m.tool_call_id {
                    msg = msg.with_tool_call_id(id.clone());
                }
                Ok(msg)
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
/// Load a model back if the idle reaper freed it.
///
/// Returns immediately when the model is already loaded, or when the id is
/// unknown — the caller's own checks decide what an unknown id means. The first
/// request after a model was freed therefore blocks for its load, which is the
/// price of freeing VRAM while idle.
async fn ensure_ready(state: &AppState, id: &str) -> ApiResult<()> {
    let dormant = {
        let guard = state.inner.lock().await;
        if guard.models.contains_key(id) {
            return Ok(());
        }
        guard.dormant.get(id).cloned()
    };
    let Some(d) = dormant else {
        return Ok(());
    };

    tracing::info!(%id, "reloading a model freed while idle");
    // One model per slot applies here too: naming a released model is a request
    // to switch to it, and the device has no room to hold both.
    if let Some(slot) = model_slot_for(&d.path, Some(d.info.vision)) {
        let parked = park_models_in_slot(state, id, slot).await;
        if !parked.is_empty() {
            let _ = tokio::task::spawn_blocking(move || drop(parked)).await;
        }
    }
    let cfg = d.config.clone();
    let engine = tokio::task::spawn_blocking(move || Engine::load(cfg, |_| {}))
        .await
        .map_err(|e| ApiError::Model(format!("reload task failed: {e}")))?
        .map_err(|e| ApiError::Model(e.to_string()))?;
    let info = engine.info().clone();
    let now = Instant::now();

    let mut guard = state.inner.lock().await;
    // A racing request may have loaded it already; keep the live one.
    if guard.models.contains_key(id) {
        return Ok(());
    }
    guard.dormant.remove(id);
    guard.models.insert(
        id.to_string(),
        Loaded {
            id: id.to_string(),
            path: d.path,
            engine,
            info,
            config: d.config,
            loaded_at: now,
            last_used: now,
            n_gpu_layers: d.n_gpu_layers,
            n_ctx: d.n_ctx,
            load_mtp: d.load_mtp,
            mmproj: d.mmproj,
            draft_model: d.draft_model,
            thinking: d.thinking,
            loras: d.loras,
            in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            stats: std::sync::Arc::new(std::sync::Mutex::new(RuntimeStats::default())),
        },
    );
    Ok(())
}

async fn chat_engine(
    state: &AppState,
    requested: Option<&str>,
) -> ApiResult<(EngineHandle, String, InFlight, Option<bool>)> {
    let id = {
        let guard = state.inner.lock().await;
        require_model(&guard, requested, "llm")?
    };
    ensure_ready(state, &id).await?;

    let mut guard = state.inner.lock().await;
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
    let (engine, thinking) = {
        let loaded = guard.models.get(&id).ok_or(ApiError::NotLoaded)?;
        (loaded.engine.handle(), loaded.thinking)
    };
    let active = guard.begin(&id).ok_or(ApiError::NotLoaded)?;
    Ok((engine, id, active, thinking))
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

/// Two decimal places.
///
/// These are smoothed figures shown to a person, not measurements to act on:
/// the third decimal is noise, and printing it invites reading meaning into it.
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
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
    let (handle, model_id, active, thinking) = chat_engine(&state, req.model.as_deref()).await?;
    let msgs: Vec<ChatMsg> = req
        .messages
        .iter()
        .map(|m| ChatMsg::new(m.role.clone(), m.content.clone()))
        .collect();
    let mut params = sampler_from_options(&SamplerParams::default(), req.options.as_ref());
    // Ollama has no thinking option, so the model's own default always applies.
    params.thinking = thinking;
    // Taken before `model_id` is handed to the stream, which owns it from there.
    let perf = state.runtime_stats(&model_id).await;
    Ok(ollama_stream(
        handle, model_id, active, msgs, params, req.stream, true, perf,
    )
    .await)
}

/// `POST /api/generate` — Ollama's single-prompt route.
///
/// Ollama keeps it distinct from chat, so this wraps the prompt as one user turn
/// rather than routing it to `/api/chat`.
async fn ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, ApiError> {
    let (handle, model_id, active, thinking) = chat_engine(&state, req.model.as_deref()).await?;
    let mut msgs = Vec::new();
    if let Some(system) = req.system.filter(|s| !s.trim().is_empty()) {
        msgs.push(ChatMsg::new("system", system));
    }
    msgs.push(ChatMsg::new("user", req.prompt.clone()));
    let mut params = sampler_from_options(&SamplerParams::default(), req.options.as_ref());
    // Ollama has no thinking option, so the model's own default always applies.
    params.thinking = thinking;
    let perf = state.runtime_stats(&model_id).await;
    Ok(ollama_stream(
        handle, model_id, active, msgs, params, req.stream, false, perf,
    )
    .await)
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
#[allow(clippy::too_many_arguments)]
async fn ollama_stream(
    handle: EngineHandle,
    model_id: String,
    active: InFlight,
    msgs: Vec<ChatMsg>,
    params: SamplerParams,
    stream: bool,
    chat: bool,
    perf: Option<Arc<std::sync::Mutex<RuntimeStats>>>,
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
                record_stats(perf.as_ref(), &stats);
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
            Ok(stats) => {
                record_stats(perf.as_ref(), &stats);
                ndjson(ollama_reply(&model_id, chat, "", Some(&stats)))
            }
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
/// Pick the model a request should run on, of the kind the route needs.
///
/// An exact name wins when it is that kind. A name that matches nothing — a
/// stale id held by an agent after the loaded model was swapped — falls back to
/// the only model of that kind, so a client does not have to be reconfigured
/// every time the desktop app changes models.
fn require_model(guard: &DaemonState, requested: Option<&str>, class: &str) -> ApiResult<String> {
    if let Some(id) = guard.resolve_class(requested, class) {
        return Ok(id);
    }
    if guard.models.is_empty() && guard.dormant.is_empty() {
        return Err(ApiError::NotLoaded);
    }
    let mut known: Vec<&str> = guard
        .models
        .keys()
        .chain(guard.dormant.keys())
        .map(String::as_str)
        .collect();
    known.sort_unstable();
    match requested.filter(|m| !m.trim().is_empty()) {
        // Several models of this kind are loaded and the name matched none:
        // naming one is the only way to choose.
        Some(id) => Err(ApiError::NotFound(format!(
            "model {id:?} not found and more than one {class} model is loaded: {}",
            known.join(", ")
        ))),
        None => Err(ApiError::NotFound(format!(
            "\"model\" is required; loaded: {}",
            known.join(", ")
        ))),
    }
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

    let id = {
        let guard = state.inner.lock().await;
        require_model(&guard, req.model.as_deref(), "rerank")?
    };
    ensure_ready(&state, &id).await?;

    let (handle, model_id, active) = {
        let mut guard = state.inner.lock().await;
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
        let handle = guard
            .models
            .get(&id)
            .ok_or(ApiError::NotLoaded)?
            .engine
            .handle();
        let active = guard.begin(&id).ok_or(ApiError::NotLoaded)?;
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
    let id = {
        let guard = state.inner.lock().await;
        require_model(&guard, requested, "embedding")?
    };
    ensure_ready(state, &id).await?;

    let mut guard = state.inner.lock().await;
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
    let engine = guard
        .models
        .get(&id)
        .ok_or(ApiError::NotLoaded)?
        .engine
        .handle();
    let active = guard.begin(&id).ok_or(ApiError::NotLoaded)?;
    Ok((engine, id, active))
}

/// Tag each tool call with its position, the way OpenAI's stream expects.
///
/// The non-streaming shape carries no `index`, but a client assembling a
/// streamed reply orders the calls by it.
fn stream_tool_calls(calls: &serde_json::Value) -> serde_json::Value {
    let indexed: Vec<serde_json::Value> = calls
        .as_array()
        .map(|a| {
            a.iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut v = c.clone();
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("index".into(), json!(i));
                    }
                    v
                })
                .collect()
        })
        .unwrap_or_default();
    serde_json::Value::Array(indexed)
}

async fn chat(
    State(state): State<AppState>,
    Json(mut body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    // The `auto` alias names a policy, not a model; decide which concrete one it
    // means now. `route` records what happened so `auto` can fall back either
    // way — local to cloud when the local model cannot serve the request, and
    // cloud to local when the provider is unreachable.
    let route = expand_alias(&state, &mut body).await?;

    // A model named `<providerId>/<model>` is served by that cloud provider;
    // anything else is a local model. Decided here, before any local state is
    // touched, so a cloud request never contends with the model lock.
    let routed = {
        let providers = state
            .cloud
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cloud::route(
            providers.as_slice(),
            body.get("model").and_then(serde_json::Value::as_str),
        )
    };
    if let Some((provider, native)) = routed {
        match cloud::proxy_chat(&state.client, &provider, native, body.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(e) if route.auto => {
                // Cloud-first `auto` (or an `auto` with no local model) could not
                // reach the provider. Fall back to the local model rather than
                // fail, which is the stability `auto` exists for.
                let local = {
                    let guard = state.inner.lock().await;
                    local_llm(&guard)
                };
                match local {
                    Some(id) => body["model"] = json!(id),
                    None => return Err(e),
                }
            }
            Err(e) => return Err(e),
        }
    }

    let req: ChatRequest = serde_json::from_value(body.clone())
        .map_err(|e| ApiError::BadRequest(format!("bad chat request: {e}")))?;
    let engine = chat_engine(&state, req.model.as_deref()).await;
    let (handle, model_id, active, thinking) = match engine {
        Ok(triple) => triple,
        Err(e) if route.local => {
            // `auto`'s local model could not take the request — not loaded,
            // context full, no vision — and no token has been produced yet, so
            // the cloud can answer it whole. Only `auto` does this; a concrete
            // local name that fails is the honest answer.
            let cloud = {
                let providers = state
                    .cloud
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                first_cloud(providers.as_slice())
                    .and_then(|cid| cloud::route(providers.as_slice(), Some(&cid)))
            };
            match cloud {
                Some((provider, native)) => {
                    body["model"] = json!(format!("{}/{}", provider.id, native));
                    return cloud::proxy_chat(&state.client, &provider, native, body).await;
                }
                None => return Err(e),
            }
        }
        Err(e) => return Err(e),
    };
    // Taken while `model_id` is still ours; the streaming branch moves it into
    // the blocking task that finishes the reply.
    let perf = state.runtime_stats(&model_id).await;

    let mut params = req.sampler();
    // A request that names no thinking mode inherits the model's own default,
    // set at load time.
    if params.thinking.is_none() {
        params.thinking = thinking;
    }
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

        record_stats(perf.as_ref(), &stats);

        // A reply that asked for tools reports `tool_calls`, not `stop`: the
        // caller has to run them and send the results back.
        let mut message = json!({ "role": "assistant", "content": text });
        if let Some(calls) = &stats.tool_calls {
            message["tool_calls"] = calls.clone();
        }
        if let Some(reasoning) = &stats.reasoning_content {
            message["reasoning_content"] = json!(reasoning);
        }
        let finish_reason = if stats.tool_calls.is_some() {
            "tool_calls"
        } else if stats.hit_token_cap {
            "length"
        } else {
            "stop"
        };

        return Ok(Json(json!({
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": message,
                // A reply cut off by `max_tokens` is not a finished reply, and
                // a client that stitches the pieces together has to be able to
                // tell the two apart.
                "finish_reason": finish_reason,
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
                record_stats(perf.as_ref(), &stats);
                // See the non-streaming path: a truncated reply reports itself,
                // and a reply that asked for tools reports `tool_calls`.
                let finish_reason = if stats.tool_calls.is_some() {
                    "tool_calls"
                } else if stats.hit_token_cap {
                    "length"
                } else {
                    "stop"
                };
                let mut delta = json!({});
                if let Some(calls) = &stats.tool_calls {
                    // OpenAI's stream carries the calls in the last chunk, each
                    // tagged with its `index` so an assembling client can order
                    // them.
                    delta["tool_calls"] = stream_tool_calls(calls);
                }
                if let Some(reasoning) = &stats.reasoning_content {
                    delta["reasoning_content"] = json!(reasoning);
                }
                Event::default().data(
                    json!({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
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
    use std::time::Duration;
    use tower::ServiceExt;

    type TestRequest = axum::http::Request<Body>;

    fn app(key: Option<&str>) -> Router {
        let state = AppState::new(Duration::from_secs(0), LoadConfig::new(""));
        router(
            state,
            key.map(|k| Arc::new(k.to_string())),
            Vec::new(),
            None,
        )
    }

    /// The same server with a models folder and a UI directory, for the paths
    /// that only exist when they were pointed at one.
    fn app_with_dirs(key: Option<&str>, models: Option<&std::path::Path>, ui: Option<&std::path::Path>) -> Router {
        let mut state = AppState::new(Duration::from_secs(0), LoadConfig::new(""));
        state = state.with_models_dir(models.map(|p| p.to_path_buf()));
        router(
            state,
            key.map(|k| Arc::new(k.to_string())),
            Vec::new(),
            ui.map(|p| p.to_path_buf()),
        )
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

    #[test]
    fn tools_map_onto_the_sampler() {
        let req: ChatRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "get_weather", "parameters": {} } }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": true,
        }))
        .expect("chat request");
        let p = req.sampler();
        let tools = p.tools_json.expect("tools carried through");
        assert!(tools.contains("get_weather"));
        assert_eq!(p.tool_choice.as_deref(), Some("required"));
        assert!(p.parallel_tool_calls);
    }

    #[test]
    fn tools_absent_or_unnamed_leaves_the_sampler_plain() {
        let req: ChatRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [],
        }))
        .expect("chat request");
        assert!(req.sampler().tools_json.is_none());

        // A named-function object is not one of the string forms the template
        // understands, so it is dropped rather than mangled into a grammar.
        let req: ChatRequest = serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": { "type": "function", "function": { "name": "f" } },
        }))
        .expect("chat request");
        assert!(req.sampler().tool_choice.is_none());
    }

    #[test]
    fn tool_messages_keep_their_calls_and_ids() {
        let req: ChatRequest = serde_json::from_value(json!({
            "messages": [
                { "role": "assistant", "content": null, "tool_calls": [
                    { "id": "call_1", "type": "function",
                      "function": { "name": "f", "arguments": "{}" } }
                ] },
                { "role": "tool", "tool_call_id": "call_1", "content": "42" }
            ]
        }))
        .expect("chat request");
        let msgs = req.chat_msgs().expect("messages");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].tool_calls.is_some());
        assert_eq!(msgs[1].tool_call_id.as_deref(), Some("call_1"));
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

    /// The response body, for the cases where two different failures share one
    /// status code and only the message tells them apart.
    async fn post_error(path: &str, body: serde_json::Value) -> String {
        let mut req = TestRequest::new(Body::from(serde_json::to_vec(&body).expect("body")));
        *req.method_mut() = Method::POST;
        *req.uri_mut() = path.parse().expect("uri");
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let res = app(None).oneshot(req).await.unwrap();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The provider list round-trips through the daemon, but the key never comes
    /// back out, and the configured models show up in `/v1/models`.
    #[tokio::test]
    async fn providers_round_trip_without_leaking_keys() {
        let router = app(None);

        let body = json!([{
            "id": "deepseek",
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-secret",
            "models": ["deepseek-chat"]
        }]);
        let mut req = TestRequest::new(Body::from(serde_json::to_vec(&body).expect("body")));
        *req.method_mut() = Method::POST;
        *req.uri_mut() = "/admin/providers".parse().expect("uri");
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        assert_eq!(
            router.clone().oneshot(req).await.unwrap().status(),
            StatusCode::OK
        );

        let res = router
            .clone()
            .oneshot(probe("/admin/providers", None))
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes())
            .into_owned();
        assert!(text.contains("\"has_key\":true"), "{text}");
        assert!(!text.contains("sk-secret"), "key leaked: {text}");

        let res = router
            .clone()
            .oneshot(probe("/v1/models", None))
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes())
            .into_owned();
        assert!(text.contains("deepseek/deepseek-chat"), "{text}");
        assert!(!text.contains("sk-secret"), "key leaked in models: {text}");

        // The chat client reads /admin/status, so cloud models must be there too.
        let res = router.oneshot(probe("/admin/status", None)).await.unwrap();
        let text = String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes())
            .into_owned();
        assert!(text.contains("deepseek/deepseek-chat"), "{text}");
        assert!(text.contains("\"cloud\":true"), "{text}");
    }

    fn l() -> Option<String> {
        Some("local-model".to_string())
    }
    fn c() -> Option<String> {
        Some("deepseek/deepseek-chat".to_string())
    }

    #[test]
    fn auto_is_local_first_and_cloud_is_the_fallback() {
        // Local first with both available.
        assert_eq!(
            route_choice("local", l(), c(), false, false),
            Some("local-model".into())
        );
        // Cloud first when asked.
        assert_eq!(
            route_choice("cloud", l(), c(), false, false),
            Some("deepseek/deepseek-chat".into())
        );
        // No local: use cloud.
        assert_eq!(
            route_choice("local", None, c(), false, false),
            Some("deepseek/deepseek-chat".into())
        );
        // An image a non-vision local cannot read goes to the cloud.
        assert_eq!(
            route_choice("local", l(), c(), false, true),
            Some("deepseek/deepseek-chat".into())
        );
        // A vision local keeps it.
        assert_eq!(
            route_choice("local", l(), c(), true, true),
            Some("local-model".into())
        );
        // Nothing to route to.
        assert_eq!(route_choice("local", None, None, false, false), None);
    }

    #[test]
    fn image_detection_reads_content_parts() {
        assert!(!body_has_images(&json!({
            "messages": [{ "role": "user", "content": "hi" }]
        })));
        assert!(body_has_images(&json!({
            "messages": [{ "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "data:x" } }
            ] }]
        })));
    }

    /// A size that cannot be allocated is refused here rather than handed to
    /// llama.cpp, where an oversized `n_ctx` is an allocation failure that
    /// takes the whole process down.
    #[tokio::test]
    async fn load_refuses_sizes_that_cannot_be_allocated() {
        for n_ctx in [limits::N_CTX + 1, u32::MAX] {
            assert_eq!(
                post_status("/admin/load", json!({ "model": "m.gguf", "n_ctx": n_ctx })).await,
                StatusCode::BAD_REQUEST,
                "n_ctx {n_ctx} must be refused"
            );
        }
        // `0` asks for an automatic size, so it is a valid request; the missing
        // file is what stops it. NotFound shares the 400 status, so the message
        // is what proves the context was not the reason.
        let body = post_error("/admin/load", json!({ "model": "m.gguf", "n_ctx": 0 })).await;
        assert!(
            body.contains("model file not found"),
            "an automatic context must be accepted, got: {body}"
        );
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

    /// The daemon comes up with `--load-mtp` for its startup model, so every
    /// later load inherits that flag through `base_cfg`. A load that brings its
    /// own drafter is choosing the other way to draft, and must not then be
    /// told it asked for both.
    #[test]
    fn a_draft_model_drops_an_inherited_mtp_head() {
        let mut cfg = LoadConfig::new("m.gguf");
        cfg.load_mtp = true;
        cfg.mtp_n_max = 4;
        cfg.draft_model = Some("d.gguf".into());
        drop_inherited_mtp(&mut cfg, false);
        assert!(!cfg.load_mtp, "the inherited head must be dropped");
        assert_eq!(cfg.mtp_n_max, 0);

        // Asked for both: keep them, so the engine reports the real conflict.
        let mut cfg = LoadConfig::new("m.gguf");
        cfg.load_mtp = true;
        cfg.draft_model = Some("d.gguf".into());
        drop_inherited_mtp(&mut cfg, true);
        assert!(cfg.load_mtp);

        // No drafter: nothing to conflict with, so the default survives.
        let mut cfg = LoadConfig::new("m.gguf");
        cfg.load_mtp = true;
        drop_inherited_mtp(&mut cfg, false);
        assert!(cfg.load_mtp);
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

    /// A scratch folder for these tests, named so parallel runs cannot collide.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nexllama-http-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    async fn body_of(res: axum::response::Response) -> Vec<u8> {
        axum::body::to_bytes(res.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body")
            .to_vec()
    }

    /// A daemon given no folder answers "no library" rather than a 404 — the UI
    /// asks this on every open, and a 404 there would read as a broken server.
    #[tokio::test]
    async fn the_models_endpoint_answers_empty_without_a_folder() {
        let res = app(None)
            .oneshot(probe("/admin/models", None))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body_of(res).await).expect("json");
        assert!(json["dir"].is_null());
        assert_eq!(json["models"].as_array().map(Vec::len), Some(0));
    }

    /// With a folder, the same endpoint lists it, and it is the same listing the
    /// desktop library would show: one row per model, projector and drafter
    /// paired with their weights.
    #[tokio::test]
    async fn the_models_endpoint_lists_the_folder_it_was_pointed_at() {
        let dir = scratch("models");
        let model = dir.join("my-model");
        std::fs::create_dir_all(&model).expect("model dir");
        std::fs::write(model.join("my-model.gguf"), b"gguf").expect("weights");
        std::fs::write(model.join("my-model-mmproj.gguf"), b"gguf").expect("projector");

        let res = app_with_dirs(None, Some(&dir), None)
            .oneshot(probe("/admin/models", None))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body_of(res).await).expect("json");
        assert_eq!(json["models"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["models"][0]["name"], "my-model");
        assert_eq!(json["models"][0]["kind"], "vision");
        assert!(json["dir"]
            .as_str()
            .is_some_and(|d| std::path::Path::new(d) == dir));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The folder lists models; it does not expose the files in it.
    #[tokio::test]
    async fn the_models_endpoint_needs_the_token_like_the_rest() {
        assert_eq!(
            status_of(Some("secret"), "/admin/models", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    /// With `--ui-dir`, `/` is the app and the API is still there — the point of
    /// serving both on one port.
    #[tokio::test]
    async fn a_ui_directory_replaces_the_banner_and_keeps_the_api() {
        let dir = scratch("ui");
        std::fs::write(dir.join("index.html"), "<!doctype html><title>Nexllama</title>")
            .expect("index");
        std::fs::create_dir_all(dir.join("assets")).expect("assets");
        std::fs::write(dir.join("assets").join("app.js"), "console.log(1)").expect("asset");

        let app = app_with_dirs(None, None, Some(&dir));

        let res = app.clone().oneshot(probe("/", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let html = String::from_utf8(body_of(res).await).expect("utf8");
        assert!(html.contains("<title>Nexllama</title>"), "`/` should be the app");

        let res = app
            .clone()
            .oneshot(probe("/assets/app.js", None))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "built assets are served as they are");

        // A path that is not a file still opens the app, so a bookmark survives.
        let res = app.clone().oneshot(probe("/chat", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // And the engine is on the same port.
        let res = app.oneshot(probe("/health", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        std::fs::remove_dir_all(&dir).ok();
    }
}
