mod http;
mod lifecycle;
mod state;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use infer_core::{Engine, FlashAttn, LoadConfig, LoadMode};

use crate::state::{AppState, Loaded, RuntimeStats};

#[derive(Parser, Debug)]
#[command(name = "infer-daemon", about = "Local llama.cpp inference daemon")]
struct Args {
    /// GGUF model to load at startup. Can be repeated for multiple models.
    #[arg(long)]
    model: Vec<String>,

    /// TCP port to listen on.
    #[arg(long, default_value_t = 15656)]
    port: u16,

    /// Interface to bind. Use 0.0.0.0 to accept connections from the network.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Require this bearer token on every request except /health.
    #[arg(long)]
    api_key: Option<String>,

    /// Run on a routable interface without a token.
    ///
    /// The server has no accounts, no rate limit and no quota, so anyone who
    /// can reach the port can load models onto the GPU and spend its time.
    /// That combination is refused unless it is asked for by name — this is the
    /// flag a container or reverse proxy sets when something in front of it
    /// already authenticates callers.
    #[arg(long, default_value_t = false)]
    allow_unauthenticated: bool,

    /// Extra browser origin allowed to call this server, e.g.
    /// `https://example.com`. Repeatable; this machine's own origins are always
    /// allowed.
    #[arg(long = "cors-origin")]
    cors_origin: Vec<String>,

    /// Layers to offload to the GPU (999 = all).
    #[arg(long, default_value_t = 999)]
    n_gpu_layers: u32,

    /// GPU used for the first layers.
    #[arg(long, default_value_t = 0)]
    main_gpu: i32,

    /// Context size.
    #[arg(long, default_value_t = 4096)]
    n_ctx: u32,

    /// Logical batch size.
    #[arg(long, default_value_t = 2048)]
    n_batch: u32,

    /// Physical (micro) batch size.
    #[arg(long, default_value_t = 512)]
    n_ubatch: u32,

    /// CPU threads (0 = llama.cpp default).
    #[arg(long, default_value_t = 0)]
    threads: i32,

    /// CPU threads for prompt processing (0 = same as --threads).
    #[arg(long, default_value_t = 0)]
    threads_batch: i32,

    /// Weight load mode: auto | none | mmap | mlock | mmap_mlock | dio.
    #[arg(long, default_value = "auto")]
    load_mode: String,

    /// Lock the model in RAM.
    #[arg(long, default_value_t = false)]
    mlock: bool,

    /// Flash attention: auto | on | off.
    #[arg(long, default_value = "auto")]
    flash_attn: String,

    /// Disable offloading the KV cache to the GPU.
    #[arg(long, default_value_t = false)]
    no_offload_kqv: bool,

    /// Load the model's bundled MTP head for speculative decoding.
    #[arg(long, default_value_t = false)]
    load_mtp: bool,

    /// Recurrent-state snapshots per sequence for draft rollback.
    #[arg(long, default_value_t = 0)]
    n_rs_seq: u32,

    /// MTP speculative decoding: draft tokens per round (upstream
    /// --spec-draft-n-max). 0 takes the budget from --load-mtp, so it only
    /// disables MTP when --load-mtp is not set. Implies --load-mtp and
    /// --n-rs-seq.
    #[arg(long, default_value_t = 0)]
    mtp_n_max: u32,

    /// Multimodal projector that lets the model read images.
    ///
    /// Pass it once to serve every startup model, or once per --model to pair
    /// them by position. It is a startup option only: a model loaded later from
    /// the library brings its own.
    #[arg(long)]
    mmproj: Vec<PathBuf>,

    /// A separate draft checkpoint (DFlash / EAGLE-3) that proposes tokens for
    /// the model it sits next to.
    ///
    /// Pairing works the same way as --mmproj. This is not the bundled MTP
    /// head, which needs no file: see --load-mtp and --mtp-n-max.
    #[arg(long)]
    draft_model: Vec<PathBuf>,

    /// Draft tokens per round for --draft-model (0 = the engine's default).
    #[arg(long, default_value_t = 0)]
    draft_n_max: u32,

    /// RoPE base frequency (0 = model default).
    #[arg(long, default_value_t = 0.0)]
    rope_freq_base: f32,

    /// RoPE frequency scaling (0 = model default).
    #[arg(long, default_value_t = 0.0)]
    rope_freq_scale: f32,

    /// Seconds to keep an idle model resident (0 = forever).
    #[arg(long, default_value_t = 0)]
    keep_alive_secs: u64,

    /// File backing the cross-session n-gram draft cache.
    ///
    /// Speculative decoding only. Learned continuations are written back here so
    /// drafts keep working after a restart; without it the cache is in-memory.
    #[arg(long)]
    ngram_cache: Option<PathBuf>,
}

fn parse_load_mode(s: &str) -> LoadMode {
    match s.to_ascii_lowercase().as_str() {
        "none" => LoadMode::None,
        "mmap" => LoadMode::Mmap,
        "mlock" => LoadMode::Mlock,
        "mmap_mlock" | "mmap+mlock" => LoadMode::MmapMlock,
        "dio" | "direct_io" | "directio" => LoadMode::DirectIo,
        _ => LoadMode::Auto,
    }
}

fn parse_flash(s: &str) -> FlashAttn {
    match s.to_ascii_lowercase().as_str() {
        "on" | "enabled" => FlashAttn::On,
        "off" | "disabled" => FlashAttn::Off,
        _ => FlashAttn::Auto,
    }
}

/// Whether a bind address only accepts connections from this machine.
///
/// Blanket "not loopback" would treat `127.0.0.2` or `::1` as exposed, so the
/// common spellings are checked explicitly and anything unrecognised is assumed
/// to be routable — the safe direction for a warning.
fn is_loopback(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = Args::parse();
    watch_parent();

    let mut base = LoadConfig::new("");
    base.n_gpu_layers = args.n_gpu_layers;
    base.main_gpu = args.main_gpu;
    base.n_ctx = args.n_ctx;
    base.n_batch = args.n_batch;
    base.n_ubatch = args.n_ubatch;
    base.n_threads = (args.threads > 0).then_some(args.threads);
    base.n_threads_batch = (args.threads_batch > 0).then_some(args.threads_batch);
    base.load_mode = parse_load_mode(&args.load_mode);
    base.use_mlock = args.mlock;
    base.flash_attn = parse_flash(&args.flash_attn);
    base.offload_kqv = !args.no_offload_kqv;
    base.load_mtp = args.load_mtp;
    base.n_rs_seq = args.n_rs_seq;
    base.mtp_n_max = args.mtp_n_max;
    if args.mtp_n_max > 0 {
        base.n_rs_seq = base.n_rs_seq.max((args.mtp_n_max as i32).max(4) as u32);
        base.load_mtp = true;
    }
    base.rope_freq_base = args.rope_freq_base;
    base.rope_freq_scale = args.rope_freq_scale;
    base.ngram_cache = args.ngram_cache.clone();

    // Checked before anything is loaded: a mispaired list would otherwise only
    // surface after the first model was already resident in VRAM.
    for (flag, n) in [
        ("--mmproj", args.mmproj.len()),
        ("--draft-model", args.draft_model.len()),
    ] {
        if n > 1 && n != args.model.len() {
            anyhow::bail!(
                "{flag} was given {n} times but there are {} startup models",
                args.model.len()
            );
        }
    }

    let state = AppState::new(Duration::from_secs(args.keep_alive_secs), base.clone());

    let mut first = true;
    for (i, model) in args.model.iter().enumerate() {
        // Projector and draft file describe *this* model, so they are resolved
        // per model and never left in `base`: `base` is also where a later load
        // from the library starts from, and it has its own files to bring.
        let one = |values: &[PathBuf]| {
            if values.len() == 1 {
                values.first().cloned()
            } else {
                values.get(i).cloned()
            }
        };
        let mut cfg = base.clone();
        cfg.model_path = model.clone().into();
        cfg.mmproj = one(&args.mmproj);
        cfg.draft_model = one(&args.draft_model);
        cfg.draft_n_max = args.draft_n_max;
        let path = cfg.model_path.clone();
        let draft_model = cfg
            .draft_model
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());
        let mmproj = cfg.mmproj.as_ref().map(|p| p.to_string_lossy().to_string());
        let id = Path::new(model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let n_gpu_layers = cfg.n_gpu_layers;
        let n_ctx = cfg.n_ctx;
        let load_mtp = cfg.load_mtp;

        // The file name reaches the log; the full path does not. A log that
        // lives on disk indefinitely does not need a record of where the user
        // keeps their models.
        let draft = draft_model
            .as_deref()
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or("none");
        tracing::info!(%id, %draft, "loading startup model");
        // A startup model is a convenience, not a requirement: the file may have
        // been moved since it was configured, and a daemon that exits on that
        // leaves the UI unable to load anything else. Log it and keep serving —
        // the library can still load whatever is actually on disk.
        let engine = match tokio::task::spawn_blocking(move || {
            Engine::load(cfg, |ev| tracing::info!(?ev, "load"))
        })
        .await
        {
            Ok(Ok(engine)) => engine,
            Ok(Err(e)) => {
                tracing::error!(%id, error = %e, "startup model failed to load; serving without it");
                continue;
            }
            Err(e) => {
                tracing::error!(%id, error = %e, "startup model load task failed; serving without it");
                continue;
            }
        };

        let info = engine.info().clone();
        let mut guard = state.inner.lock().await;
        let now = Instant::now();
        guard.models.insert(
            id.clone(),
            Loaded {
                id: id.clone(),
                path,
                engine,
                info,
                loaded_at: now,
                last_used: now,
                n_gpu_layers,
                n_ctx,
                load_mtp,
                mmproj,
                draft_model,
                loras: Vec::new(),
                in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                stats: std::sync::Arc::new(std::sync::Mutex::new(RuntimeStats::default())),
            },
        );
        if first {
            guard.default_model = Some(id);
            first = false;
        }
    }

    lifecycle::spawn_reaper(state.clone(), Duration::from_secs(5));

    // An empty token used to be read as "no authentication", which quietly
    // turned the flag meant to protect the server into the one that opens it.
    // Say so instead: either there is a token or there is not, and the caller
    // gets to find out which.
    let api_key = match args.api_key.as_deref().map(str::trim) {
        Some("") => {
            anyhow::bail!(
                "--api-key was given but is blank; pass a token, or drop the flag and pass \
                 --allow-unauthenticated"
            );
        }
        Some(k) => Some(Arc::new(k.to_string())),
        // The desktop app passes the token this way: an environment variable
        // is not visible in `ps` or the task manager, which an `--api-key`
        // argument is.
        None => std::env::var("NEXLLAMA_API_KEY")
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .map(Arc::new),
    };
    if api_key.is_none() && !is_loopback(&args.host) && !args.allow_unauthenticated {
        anyhow::bail!(
            "refusing to listen on {} without --api-key: anyone who can reach this port can \
             load models and spend GPU time on them. Pass --api-key <token>, bind to \
             127.0.0.1, or pass --allow-unauthenticated if something else guards the port",
            args.host
        );
    }

    let app = http::router(state, api_key, args.cors_origin.clone());
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("infer-daemon listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Exit when the process that started us is gone.
///
/// A desktop app that is killed rather than closed used to leave its daemon
/// running: the model stayed in memory and the port stayed taken. On Windows a
/// job object handles that from outside; the other platforms have no equivalent
/// we can attach to a child, so the daemon watches for its parent instead.
#[cfg(unix)]
fn watch_parent() {
    let parent = std::env::var("NEXLLAMA_PARENT_PID")
        .ok()
        .and_then(|s| s.parse::<libc::pid_t>().ok())
        .filter(|p| *p > 1);
    let Some(parent) = parent else {
        return;
    };
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(2));
        // Re-parented to init once the starter is gone, so the id simply stops
        // matching; no signal handling needed.
        if unsafe { libc::getppid() } != parent {
            tracing::info!("the app that started this daemon is gone; shutting down");
            std::process::exit(0);
        }
    });
}

#[cfg(not(unix))]
fn watch_parent() {}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_detection() {
        for host in [
            "127.0.0.1",
            "127.0.0.53",
            "localhost",
            "LOCALHOST",
            "::1",
            "[::1]",
        ] {
            assert!(is_loopback(host), "{host} should count as loopback");
        }
        for host in [
            "0.0.0.0",
            "192.168.1.20",
            "10.0.0.5",
            "::",
            "example.local",
            "",
        ] {
            assert!(!is_loopback(host), "{host} should count as routable");
        }
    }
}
