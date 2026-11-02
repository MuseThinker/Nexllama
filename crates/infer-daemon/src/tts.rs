//! The resident speech worker.
//!
//! `llama-tts` is a one-shot CLI: every invocation reloads the weights. This
//! module owns a long-lived `llama-tts-worker` process instead, so the weights
//! and the mmproj stay loaded until the TTS model is unloaded. The worker reads
//! one JSON request per line on stdin and answers with one JSON line per
//! request; its first line after loading is `READY`.
//!
//! The daemon never exposes the worker's stdin/stdout to the network. Speech
//! requests still arrive on `15656`; this is only the internal channel.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// How long the model may take to load before the worker is declared failed.
///
/// A cold Q8 load on a slow disk was measured at about 70 seconds; five minutes
/// leaves room for a slower machine without hiding a genuinely dead worker.
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// How long one synthesis may run. Long text on a slow CPU is still minutes,
/// not hours.
const SPEAK_TIMEOUT: Duration = Duration::from_secs(900);

#[derive(Debug, Clone)]
pub enum Status {
    Idle,
    Loading,
    Ready,
    Failed(String),
}

impl Status {
    pub fn label(&self) -> &'static str {
        match self {
            Status::Idle => "idle",
            Status::Loading => "loading",
            Status::Ready => "ready",
            Status::Failed(_) => "failed",
        }
    }

    pub fn error(&self) -> Option<&str> {
        match self {
            Status::Failed(e) => Some(e),
            _ => None,
        }
    }
}

struct Inner {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    status: Status,
    model_id: Option<String>,
}

pub struct TtsWorker {
    bin: Option<PathBuf>,
    inner: Arc<Mutex<Inner>>,
}

impl TtsWorker {
    pub fn new(bin: Option<PathBuf>) -> Self {
        Self {
            bin,
            inner: Arc::new(Mutex::new(Inner {
                child: None,
                stdin: None,
                stdout: None,
                status: Status::Idle,
                model_id: None,
            })),
        }
    }

    /// Whether a worker executable was found beside the daemon.
    pub fn is_configured(&self) -> bool {
        self.bin.as_ref().map(|p| p.is_file()).unwrap_or(false)
    }

    pub async fn status(&self) -> Status {
        self.inner.lock().await.status.clone()
    }

    /// Kill any worker and forget its model. Called on unload.
    pub async fn stop(&self) {
        let child = {
            let mut guard = self.inner.lock().await;
            guard.status = Status::Idle;
            guard.model_id = None;
            guard.stdin = None;
            guard.stdout = None;
            guard.child.take()
        };
        if let Some(mut child) = child {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }

    /// Start (or restart) the worker for one model.
    ///
    /// Returns immediately; the model loads in the background and the first
    /// [`wait_ready`] call waits for it. Starting twice for the same model is a
    /// no-op, which keeps a repeated Load click from loading a second copy.
    pub async fn start(
        &self,
        model_id: String,
        weights: PathBuf,
        mmproj: PathBuf,
        cfg: &infer_core::LoadConfig,
    ) -> Result<(), String> {
        let Some(bin) = self.bin.as_ref().filter(|p| p.is_file()) else {
            return Err("llama-tts-worker.exe was not found beside the daemon".into());
        };

        {
            let guard = self.inner.lock().await;
            if guard.model_id.as_deref() == Some(model_id.as_str())
                && matches!(guard.status, Status::Ready | Status::Loading)
            {
                return Ok(());
            }
        }

        // A new model (or a retry) replaces the old process, which is what
        // releases the previous model's VRAM.
        self.stop().await;

        let mut cmd = tokio::process::Command::new(bin);
        cmd.arg("-m")
            .arg(&weights)
            .arg("-mm")
            .arg(&mmproj)
            .arg("-ngl")
            .arg(cfg.n_gpu_layers.to_string())
            .arg("-c")
            .arg((if cfg.n_ctx == 0 { 4096 } else { cfg.n_ctx }).to_string())
            .arg("-b")
            .arg(cfg.n_batch.max(1).to_string())
            .arg("-ub")
            .arg(cfg.n_ubatch.max(1).to_string());
        if let Some(threads) = cfg.n_threads {
            cmd.arg("-t").arg(threads.to_string());
        }
        if let Some(threads) = cfg.n_threads_batch {
            cmd.arg("-tb").arg(threads.to_string());
        }
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            // The daemon's own stderr is the log file; worker warnings and
            // model-load errors belong there.
            .stderr(std::process::Stdio::inherit());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", bin.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().map(BufReader::new);

        {
            let mut guard = self.inner.lock().await;
            guard.child = Some(child);
            guard.stdin = stdin;
            guard.stdout = stdout;
            guard.status = Status::Loading;
            guard.model_id = Some(model_id);
        }

        let inner = self.inner.clone();
        tokio::spawn(async move {
            wait_ready(inner).await;
        });
        Ok(())
    }

    /// Wait until the model is loaded, a failure is known, or the timeout wins.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.inner.lock().await.status.clone();
            match status {
                Status::Ready => return Ok(()),
                Status::Failed(e) => return Err(e),
                Status::Idle => return Err("the speech worker is not running".into()),
                Status::Loading => {
                    if Instant::now() >= deadline {
                        return Err(format!(
                            "the speech worker did not become ready within {} seconds",
                            timeout.as_secs()
                        ));
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }

    /// Send one request and wait for its answer.
    ///
    /// The worker processes one request at a time; holding the inner lock for
    /// the duration is what serialises them. `out` is written by the worker.
    pub async fn speak(
        &self,
        input: &str,
        language: &str,
        voice: Option<&Path>,
        out: &Path,
        max_frames: u32,
    ) -> Result<(), String> {
        let voice = voice
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let output = out.to_string_lossy();
        let (input, lang, voice, output) = (
            input.as_bytes(),
            language.as_bytes(),
            voice.as_bytes(),
            output.as_bytes(),
        );
        let mut message = format!(
            "TTS1 {} {} {} {} {}\n",
            input.len(),
            lang.len(),
            voice.len(),
            output.len(),
            max_frames
        )
        .into_bytes();
        for part in [input, lang, voice, output] {
            message.extend_from_slice(part);
        }

        let mut guard = self.inner.lock().await;
        match &guard.status {
            Status::Ready => {}
            Status::Failed(e) => return Err(e.clone()),
            Status::Loading => return Err("the speech model is still loading".into()),
            Status::Idle => return Err("the speech worker is not running".into()),
        }

        let failure = async {
            let stdin = guard
                .stdin
                .as_mut()
                .ok_or_else(|| "the speech worker has no stdin".to_string())?;
            stdin
                .write_all(&message)
                .await
                .map_err(|e| format!("could not send the speech request: {e}"))?;
            stdin
                .flush()
                .await
                .map_err(|e| format!("could not flush the speech request: {e}"))?;

            let mut response = String::new();
            let read = tokio::time::timeout(SPEAK_TIMEOUT, async {
                let stdout = guard
                    .stdout
                    .as_mut()
                    .ok_or_else(|| "the speech worker has no stdout".to_string())?;
                stdout
                    .read_line(&mut response)
                    .await
                    .map_err(|e| format!("could not read the speech answer: {e}"))
            })
            .await
            .map_err(|_| "speech timed out".to_string())??;
            if read == 0 {
                return Err("the speech worker exited".into());
            }
            let response = response.trim();
            if let Some(bytes) = response.strip_prefix("OK ") {
                bytes
                    .parse::<u64>()
                    .map_err(|_| format!("bad speech answer: {response:?}"))?;
                Ok(())
            } else if let Some(error) = response.strip_prefix("ERR ") {
                Err(error.to_string())
            } else {
                Err(format!("bad speech answer: {response:?}"))
            }
        }
        .await;

        if let Err(e) = &failure {
            // A protocol failure means the process is no longer trustworthy.
            // A model-level error ("set_input failed") leaves it usable, but
            // it is safer to restart on the next request than to keep writing
            // into a half-broken stream.
            guard.status = Status::Failed(e.clone());
            guard.stdin = None;
            guard.stdout = None;
            if let Some(mut child) = guard.child.take() {
                let _ = child.start_kill();
            }
        }
        failure
    }
}

/// Read the worker's startup handshake.
async fn wait_ready(inner: Arc<Mutex<Inner>>) {
    let stdout = {
        let mut guard = inner.lock().await;
        guard.stdout.take()
    };
    let Some(mut stdout) = stdout else {
        fail(inner, "the speech worker has no stdout".to_string()).await;
        return;
    };

    let mut line = String::new();
    let outcome = loop {
        let read = tokio::time::timeout(LOAD_TIMEOUT, stdout.read_line(&mut line)).await;
        match read {
            Ok(Ok(0)) => break Err("the speech worker exited before loading the model".to_string()),
            Ok(Ok(_)) => {
                let text = line.trim();
                if text.is_empty() {
                    line.clear();
                    continue;
                }
                if text == "READY" {
                    break Ok(());
                }
                // The worker reports a load error as one `ERR ...` line; keep
                // any other line as the diagnostic rather than discarding it.
                break Err(text.strip_prefix("ERR ").unwrap_or(text).to_string());
            }
            Ok(Err(e)) => break Err(format!("could not read the worker handshake: {e}")),
            Err(_) => {
                break Err(format!(
                    "the speech worker did not load the model within {} seconds",
                    LOAD_TIMEOUT.as_secs()
                ))
            }
        }
    };

    match outcome {
        Ok(()) => {
            let mut guard = inner.lock().await;
            guard.status = Status::Ready;
            guard.stdout = Some(stdout);
        }
        Err(e) => {
            fail(inner, e).await;
        }
    }
}

async fn fail(inner: Arc<Mutex<Inner>>, message: String) {
    let child = {
        let mut guard = inner.lock().await;
        guard.status = Status::Failed(message);
        guard.stdin = None;
        guard.stdout = None;
        guard.child.take()
    };
    if let Some(mut child) = child {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }
}
