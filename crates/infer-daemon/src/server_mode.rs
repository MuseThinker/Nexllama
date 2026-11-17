//! Hosting a llama.cpp server process for multi-user serving.
//!
//! The in-process engine answers one request at a time: a second caller waits
//! for the first reply to finish. `llama-server` from the same llama.cpp tree
//! does continuous batching with `--parallel N` slots instead, which is what a
//! machine shared by several people needs (see `docs/benchmarks/concurrency.md`
//! for what that buys on real hardware).
//!
//! This module owns that child process: translating a [`LoadConfig`] into
//! `llama-server` arguments, picking a port, waiting for the server to answer,
//! and killing it on drop. The HTTP surface in `http.rs` forwards `/v1/*` to it
//! while it runs, so clients keep talking to one address.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use infer_core::engine::{FlashAttn, KvCacheType, LoadConfig};

/// A running `llama-server` child process.
pub struct ServerProc {
    child: Child,
    port: u16,
    pub model_id: String,
    pub model_path: PathBuf,
    pub parallel: u32,
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
    /// Projector this server was started with, so the model entry can say
    /// whether it can see or hear.
    pub mmproj: Option<PathBuf>,
}

/// The daemon keeps this in shared state, so it has to cross threads.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ServerProc>();
};

impl ServerProc {
    /// Start a server for `cfg` on `port` and wait until it answers `/health`.
    ///
    /// The wait is the load time: `llama-server` reads the weights before it
    /// binds, so a large model takes as long here as it does in process. A
    /// child that exits during that wait is reported with its exit status
    /// rather than as a timeout, because "out of memory" and "still loading"
    /// need different answers.
    pub async fn spawn(
        exe: &Path,
        cfg: &LoadConfig,
        model_id: String,
        port: u16,
        parallel: u32,
    ) -> Result<Self, String> {
        let mut cmd = Command::new(exe);
        cmd.args(server_args(cfg, port, parallel))
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // No console window: this is spawned by a service, not by a shell.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", exe.display()))?;
        let mut proc = Self {
            child,
            port,
            model_id,
            model_path: cfg.model_path.clone(),
            parallel,
            n_ctx: cfg.n_ctx,
            n_gpu_layers: cfg.n_gpu_layers,
            mmproj: cfg.mmproj.clone(),
        };
        proc.wait_ready().await?;
        Ok(proc)
    }

    pub fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Whether the child is still running. Reaps it when it is not.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    async fn wait_ready(&mut self) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| e.to_string())?;
        let url = format!("{}/health", self.base());
        // A 35B model on a cold page cache is minutes, not seconds.
        let deadline = Instant::now() + Duration::from_secs(900);
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(format!("llama-server exited with {status}"));
            }
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                return Err("llama-server did not become ready within 15 minutes".into());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Translate a load configuration into `llama-server` arguments.
///
/// Every flag here is one the tuning sweep also measures, so a translation that
/// drifts would make the sweep and the server disagree about the same model.
/// Kept pure and separate for exactly that reason.
pub fn server_args(cfg: &LoadConfig, port: u16, parallel: u32) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    a.push("-m".into());
    a.push(cfg.model_path.display().to_string());
    if let Some(mmproj) = &cfg.mmproj {
        a.push("--mmproj".into());
        a.push(mmproj.display().to_string());
    }
    a.push("-ngl".into());
    a.push(cfg.n_gpu_layers.to_string());
    // The pool is shared between slots, so the per-conversation budget is this
    // divided by `parallel`.
    a.push("-c".into());
    a.push(cfg.n_ctx.to_string());
    a.push("-np".into());
    a.push(parallel.max(1).to_string());
    a.push("-b".into());
    a.push(cfg.n_batch.to_string());
    a.push("-ub".into());
    a.push(cfg.n_ubatch.to_string());
    a.push("-fa".into());
    a.push(
        match cfg.flash_attn {
            FlashAttn::Auto => "auto",
            FlashAttn::On => "on",
            FlashAttn::Off => "off",
        }
        .into(),
    );
    // Quantized V needs flash attention; passing it with attention off would
    // make the server refuse the load instead of falling back the way the
    // in-process engine does.
    if !matches!(cfg.flash_attn, FlashAttn::Off) {
        a.push("-ctk".into());
        a.push(kv_name(cfg.cache_type_k).into());
        a.push("-ctv".into());
        a.push(kv_name(cfg.cache_type_v).into());
    }
    if cfg.n_cpu_moe > 0 {
        a.push("-ncmoe".into());
        a.push(cfg.n_cpu_moe.to_string());
    }
    if cfg.load_mtp {
        a.push("--spec-type".into());
        a.push("draft-mtp".into());
    }
    a.push("--cont-batching".into());
    a.push("--host".into());
    a.push("127.0.0.1".into());
    a.push("--port".into());
    a.push(port.to_string());
    a
}

fn kv_name(t: KvCacheType) -> &'static str {
    match t {
        KvCacheType::F16 => "f16",
        KvCacheType::Q8_0 => "q8_0",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LoadConfig {
        let mut c = LoadConfig::new("C:/models/qwen.gguf");
        c.n_gpu_layers = 999;
        c.n_ctx = 32768;
        c.n_batch = 2048;
        c.n_ubatch = 512;
        c.flash_attn = FlashAttn::On;
        c.cache_type_k = KvCacheType::F16;
        c.cache_type_v = KvCacheType::F16;
        c.load_mtp = true;
        c
    }

    /// Flag/value pairs, skipping the lone switches (`--cont-batching`), which
    /// would otherwise shift every pair after them.
    fn pairs(args: &[String]) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 1 < args.len() {
            if args[i].starts_with('-') && !args[i + 1].starts_with('-') {
                out.push((args[i].clone(), args[i + 1].clone()));
                i += 2;
            } else {
                i += 1;
            }
        }
        out
    }

    #[test]
    fn the_model_and_its_shape_survive_translation() {
        let args = server_args(&cfg(), 15810, 4);
        let p = pairs(&args);
        let get = |k: &str| {
            p.iter()
                .find(|(a, _)| a == k)
                .map(|(_, v)| v.as_str())
                .unwrap_or("")
        };
        assert_eq!(get("-m"), "C:/models/qwen.gguf");
        assert_eq!(get("-ngl"), "999");
        assert_eq!(get("-c"), "32768");
        assert_eq!(get("-np"), "4");
        assert_eq!(get("-b"), "2048");
        assert_eq!(get("-ub"), "512");
        assert_eq!(get("-fa"), "on");
        assert_eq!(get("-ctk"), "f16");
        assert_eq!(get("-ctv"), "f16");
        assert_eq!(get("--port"), "15810");
        assert!(args.iter().any(|a| a == "--cont-batching"));
    }

    #[test]
    fn mtp_is_requested_by_name_and_only_when_loaded() {
        let args = server_args(&cfg(), 1, 1);
        let i = args
            .iter()
            .position(|a| a == "--spec-type")
            .expect("spec type");
        assert_eq!(args[i + 1], "draft-mtp");

        let mut plain = cfg();
        plain.load_mtp = false;
        assert!(!server_args(&plain, 1, 1).iter().any(|a| a == "--spec-type"));
    }

    /// Attention off cannot carry a quantized V cache, so the flags are left
    /// out entirely rather than sent and refused.
    #[test]
    fn attention_off_drops_the_cache_type_flags() {
        let mut c = cfg();
        c.flash_attn = FlashAttn::Off;
        c.cache_type_k = KvCacheType::Q8_0;
        c.cache_type_v = KvCacheType::Q8_0;
        let args = server_args(&c, 1, 1);
        assert!(!args.iter().any(|a| a == "-ctk" || a == "-ctv"));
        assert!(args.iter().any(|a| a == "off"));
    }

    #[test]
    fn the_projector_and_cpu_experts_are_carried_over() {
        let mut c = cfg();
        c.mmproj = Some(PathBuf::from("C:/models/mmproj.gguf"));
        c.n_cpu_moe = 12;
        let args = server_args(&c, 1, 1);
        let i = args.iter().position(|a| a == "--mmproj").expect("mmproj");
        assert_eq!(args[i + 1], "C:/models/mmproj.gguf");
        let j = args.iter().position(|a| a == "-ncmoe").expect("ncmoe");
        assert_eq!(args[j + 1], "12");
    }
}
