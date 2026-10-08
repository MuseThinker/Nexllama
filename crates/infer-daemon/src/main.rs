mod http;
mod lifecycle;
mod state;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use infer_core::{Engine, FlashAttn, LoadConfig, LoadMode};

use crate::state::{AppState, Loaded};

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

    /// Multimodal projector that lets the model read images.
    #[arg(long)]
    mmproj: Option<PathBuf>,

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
    base.rope_freq_base = args.rope_freq_base;
    base.rope_freq_scale = args.rope_freq_scale;
    base.ngram_cache = args.ngram_cache.clone();
    base.mmproj = args.mmproj.clone();

    let state = AppState::new(Duration::from_secs(args.keep_alive_secs), base.clone());

    let mut first = true;
    for model in &args.model {
        let mut cfg = base.clone();
        cfg.model_path = model.clone().into();
        let path = cfg.model_path.clone();
        let id = Path::new(model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let n_gpu_layers = cfg.n_gpu_layers;
        let n_ctx = cfg.n_ctx;
        let load_mtp = cfg.load_mtp;

        tracing::info!(%id, ?path, "loading startup model");
        let engine = tokio::task::spawn_blocking(move || {
            Engine::load(cfg, |ev| tracing::info!(?ev, "load"))
        })
        .await??;

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
                mmproj: None,
                draft_model: None,
                loras: Vec::new(),
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
        None => None,
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
