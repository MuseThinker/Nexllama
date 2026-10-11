use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use infer_hub::{GgufFile, HubClient, RepoSummary};
use tauri::{Emitter, Manager};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Spawn a helper process without ever showing a console window.
fn hidden_command(program: &str) -> Command {
    let mut cmd = Command::new(program);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// Keep a spawned daemon inside a job object, so it cannot outlive the app.
///
/// `KILL_ON_JOB_CLOSE` means the daemons are terminated when the job's last
/// handle goes away — and the OS closes that handle when this process ends,
/// whatever the reason. Depending on `RunEvent::Exit` alone leaves an orphan
/// behind every time the app is killed instead of closed, and an orphan holds
/// the model in VRAM and the port.
#[cfg(windows)]
mod job {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    // The handle as an integer: a raw `HANDLE` pointer is not `Sync`, and a
    // `static` has to be.
    static JOB: OnceLock<usize> = OnceLock::new();

    fn handle() -> Option<HANDLE> {
        let raw = *JOB.get_or_init(|| unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const std::ffi::c_void,
                std::mem::size_of_val(&info) as u32,
            );
            if ok == 0 {
                CloseHandle(job);
                return 0;
            }
            job as usize
        });
        (raw != 0).then_some(raw as HANDLE)
    }

    pub fn assign(pid: u32) {
        if let Some(job) = handle() {
            unsafe {
                let proc = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if !proc.is_null() {
                    AssignProcessToJobObject(job, proc);
                    CloseHandle(proc);
                }
            }
        }
    }
}

const DEFAULT_PORT: u16 = 15656;
/// Backends this app knows how to build a path for.
///
/// `metal` is Apple's own GPU API. On Apple silicon it is the only path that
/// runs a model on the GPU directly; Vulkan there is MoltenVK translating on
/// top of Metal, which works and costs a layer.
const BACKENDS: [&str; 4] = ["cuda", "vulkan", "metal", "cpu"];

/// Daemon executable name for the current platform.
#[cfg(windows)]
const DAEMON_EXE: &str = "infer-daemon.exe";
#[cfg(not(windows))]
const DAEMON_EXE: &str = "infer-daemon";

/// Whether `name` is one of the backends this app knows how to build a path
/// for.
///
/// Backend names end up in `Path::join` and, one step later, in
/// `Command::new`: they name a directory *and* the process launched from it.
/// An unchecked name is therefore a traversal primitive — `..` segments or an
/// absolute path would let any caller run whichever `infer-daemon.exe` it
/// likes. Everything that turns a name into a path goes through here first.
fn is_backend_name(name: &str) -> bool {
    BACKENDS.contains(&name)
}

/// Whether `candidate` really lives under `root`, following symlinks.
///
/// `Path::join` plus a name check is still not enough on its own: if either
/// side is a symlink or a junction, a canonical-looking path can point
/// somewhere else entirely. Canonicalizing both is what actually pins it down.
fn is_within(root: &Path, candidate: &Path) -> bool {
    let (Ok(root), Ok(candidate)) = (root.canonicalize(), candidate.canonicalize()) else {
        return false;
    };
    candidate.starts_with(root)
}

/// Canonicalize `root`, then build `root/<name>` from it.
///
/// Joining onto a canonicalized root keeps an absolute or traversal-bearing
/// `name` from escaping, even before `is_backend_name` is consulted.
fn child_of(root: &Path, name: &str) -> PathBuf {
    root.canonicalize()
        .unwrap_or_else(|_| root.to_path_buf())
        .join(name)
}

/// Interface the daemon binds. Loopback by default; the settings page can widen
/// it to every interface, which is what makes the server reachable from other
/// machines on the network.
fn bind_host(settings: &serde_json::Value) -> String {
    settings
        .get("bindHost")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("127.0.0.1")
        .to_string()
}

/// Bearer token handed to the daemon, if the user configured one.
/// Where the daemon's access token is kept in the OS credential store.
const SECRET_SERVICE: &str = "com.nexllama.desktop";
const SECRET_ACCOUNT: &str = "daemon-api-key";

/// Read the access token from the OS credential store.
///
/// Settings live in a JSON file beside the app, readable by anything running as
/// this user — which is where the token used to sit. Credential Manager, the
/// keychain and the Secret Service are what a secret is for.
fn secret() -> Option<String> {
    let entry = keyring::Entry::new(SECRET_SERVICE, SECRET_ACCOUNT).ok()?;
    entry.get_password().ok().filter(|s| !s.trim().is_empty())
}

/// Write the access token to the credential store, or remove it when empty.
fn store_secret(value: &str) -> Result<(), String> {
    let entry = keyring::Entry::new(SECRET_SERVICE, SECRET_ACCOUNT).map_err(|e| e.to_string())?;
    if value.is_empty() {
        return match entry.delete_credential() {
            Ok(()) => Ok(()),
            // Nothing stored yet is the ordinary case for clearing a token.
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        };
    }
    entry.set_password(value).map_err(|e| e.to_string())
}

/// The token to hand the daemon.
///
/// The credential store wins; the copy in `settings.json` is only a fallback
/// for a token saved before it moved, and is migrated out on the next save.
fn api_key(settings: &serde_json::Value) -> Option<String> {
    secret().or_else(|| {
        settings
            .get("apiKey")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    })
}

/// Whether the current bind address is reachable from off this machine.
///
/// `0.0.0.0` and `::` are unspecified rather than loopback, which is exactly the
/// point: they accept every interface, so they count as exposed.
fn is_exposed(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return false;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => !ip.is_loopback(),
        // A hostname resolves somewhere; without resolving it we must assume the
        // risky answer.
        Err(_) => true,
    }
}

/// Where the daemon is in its life, tracked separately from the handle.
///
/// "Is it running" and "start it" used to be two steps with nothing between
/// them, so two callers could both find nothing running and each spawn one:
/// the second would die on the port, and the handle kept would be the dead
/// one while the live process went untracked and unkillable. The phase makes
/// the gap explicit, so a start that is already under way says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Phase {
    #[default]
    Stopped,
    Starting,
    Running,
    Stopping,
}

#[derive(Default)]
struct DaemonProc {
    phase: Mutex<Phase>,
    child: Mutex<Option<Child>>,
    /// Why the last start attempt failed.
    ///
    /// The first thing a launch does is start the daemon, and it does so with
    /// no window yet to report to: without this, a refused start — the bind
    /// address is exposed and no access token is set, say — reached a log line
    /// nobody sees, and every later call failed as an unreachable port.
    last_error: Mutex<Option<String>>,
}

/// Lock helper that reports poisoning instead of panicking.
///
/// `unwrap` on a poisoned mutex takes the command thread down, and every
/// command that touches the daemon would then fail the same way — turning one
/// earlier panic into an app that can no longer stop or start anything.
fn lock<T>(m: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    m.lock()
        .map_err(|_| "the daemon state lock is poisoned; restart the app".to_string())
}

/// Remember why a start failed, or clear it now that one worked.
fn note_start(state: &DaemonProc, result: &Result<u16, String>) {
    if let Ok(mut slot) = lock(&state.last_error) {
        *slot = result.as_ref().err().cloned();
    }
}

fn backends_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("backends");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Prefer an updated backend in the writable app-data dir, then the bundled resource.
///
/// Returns `None` for anything that is not a known backend name, so callers
/// can treat "no such backend" and "refused" the same way.
fn backend_exe(app: &tauri::AppHandle, name: &str) -> Option<PathBuf> {
    if !is_backend_name(name) {
        return None;
    }
    if let Ok(root) = backends_root(app) {
        let exe = child_of(&root, name).join(DAEMON_EXE);
        if exe.is_file() && is_within(&root, &exe) {
            return Some(exe);
        }
    }
    let dir = app
        .path()
        .resolve(
            format!("backends/{name}"),
            tauri::path::BaseDirectory::Resource,
        )
        .ok()?;
    let exe = dir.join(DAEMON_EXE);
    exe.is_file().then_some(exe)
}

fn dev_daemon() -> Option<PathBuf> {
    for profile in ["debug", "release"] {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../target/{profile}/{DAEMON_EXE}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

static GPU_NAMES: OnceLock<Vec<String>> = OnceLock::new();
static PREFERRED: OnceLock<&'static str> = OnceLock::new();

fn gpu_names() -> Vec<String> {
    GPU_NAMES
        .get_or_init(|| {
            let text = platform_gpu_text();
            text.lines()
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .clone()
}

/// Ask this platform what graphics hardware it has.
///
/// The branch is decided at run time rather than with `#[cfg(target_os)]` on
/// purpose: a `cfg` branch is not compiled on the platforms it does not match,
/// so the macOS and Linux arms would only ever be checked by whoever happened
/// to build there. This way every arm is type-checked on every build, and a
/// platform is only ever *executed* where it applies.
fn platform_gpu_text() -> String {
    let out = match std::env::consts::OS {
        // `system_profiler` reports each display controller, with the
        // interesting lines shaped like "Chipset Model: Apple M2 Pro".
        "macos" => hidden_command("system_profiler")
            .args(["SPDisplaysDataType"])
            .output()
            .map(|o| {
                let raw = String::from_utf8_lossy(&o.stdout);
                raw.lines()
                    .filter_map(|line| line.split_once(':'))
                    .map(|(key, value)| (key.trim(), value.trim()))
                    .filter(|(key, _)| *key == "Chipset Model" || *key == "Vendor")
                    .map(|(_, value)| value.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
        // `lspci` is the closest thing Linux has to a standard answer. The
        // lines worth reading name a VGA, 3D or display controller.
        "linux" => hidden_command("lspci")
            .args(["-nn"])
            .output()
            .map(|o| {
                let raw = String::from_utf8_lossy(&o.stdout);
                raw.lines()
                    .filter(|line| {
                        let l = line.to_ascii_lowercase();
                        l.contains("vga")
                            || l.contains("3d controller")
                            || l.contains("display controller")
                    })
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
        // Windows: CIM through PowerShell, which needs no extra dependency and
        // is present on every supported version.
        _ => hidden_command("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default(),
    };
    out
}

fn has_nvidia() -> bool {
    // `nvidia-smi` ships with the driver on every platform that has one.
    let smi = hidden_command("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();
    matches!(smi, Ok(o) if o.status.success())
}

fn preferred_backend() -> &'static str {
    PREFERRED.get_or_init(|| {
        let names = gpu_names();
        // Apple's own GPU: Metal reaches it directly, so prefer it to Vulkan's
        // translation layer whenever the bundle actually carries it.
        if cfg!(target_os = "macos") && names.iter().any(|n| n.contains("apple")) {
            return "metal";
        }
        let nvidia = names.iter().any(|n| n.contains("nvidia")) || has_nvidia();
        if nvidia {
            return "cuda";
        }
        let other_gpu = names.iter().any(|n| {
            n.contains("amd") || n.contains("radeon") || n.contains("intel") || n.contains("arc")
        });
        if other_gpu {
            return "vulkan";
        }
        "cpu"
    })
}

fn available_backends(app: &tauri::AppHandle) -> Vec<&'static str> {
    BACKENDS
        .iter()
        .copied()
        .filter(|b| backend_exe(app, b).is_some())
        .collect()
}

fn resolve_backend(app: &tauri::AppHandle, requested: &str) -> String {
    // `backend_exe` already refuses unknown names, so anything else falls
    // through to auto-detection rather than being used to build a path.
    if requested != "auto" && backend_exe(app, requested).is_some() {
        return requested.to_string();
    }
    let preferred = preferred_backend();
    // Try the platform's own answer first, then whatever else is bundled.
    let order: [&str; 4] = match preferred {
        "metal" => ["metal", "vulkan", "cuda", "cpu"],
        "cuda" => ["cuda", "vulkan", "metal", "cpu"],
        "vulkan" => ["vulkan", "metal", "cuda", "cpu"],
        _ => ["metal", "cuda", "vulkan", "cpu"],
    };
    for b in order {
        if backend_exe(app, b).is_some() {
            return b.to_string();
        }
    }
    "cpu".to_string()
}

fn daemon_binary(app: &tauri::AppHandle, backend: &str) -> Result<PathBuf, String> {
    if let Some(exe) = backend_exe(app, backend) {
        return Ok(exe);
    }
    if let Some(dev) = dev_daemon() {
        return Ok(dev);
    }
    Err(
        "no infer-daemon backend found. Run scripts/build-daemon.ps1, or drop one into \
         %APPDATA%/com.nexllama.desktop/backends/<name>/."
            .into(),
    )
}

fn child_alive(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(None))
}

#[tauri::command]
fn backend_info(app: tauri::AppHandle) -> serde_json::Value {
    let requested = load_settings(&app)
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    serde_json::json!({
        "preferred": preferred_backend(),
        "requested": requested,
        "resolved": resolve_backend(&app, &requested),
        "available": available_backends(&app),
        "gpus": gpu_names(),
    })
}

#[tauri::command]
fn daemon_binary_info(app: tauri::AppHandle) -> serde_json::Value {
    let requested = load_settings(&app)
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let backend = resolve_backend(&app, &requested);
    serde_json::json!({
        "backend": backend,
        "binary": daemon_binary(&app, &backend).ok().map(|p| p.to_string_lossy().to_string()),
    })
}

#[tauri::command]
fn daemon_running(state: tauri::State<DaemonProc>) -> bool {
    let mut guard = match lock(&state.child) {
        Ok(g) => g,
        Err(_) => return false,
    };
    match guard.as_mut() {
        Some(child) => {
            if child_alive(child) {
                true
            } else {
                *guard = None;
                if let Ok(mut p) = lock(&state.phase) {
                    *p = Phase::Stopped;
                }
                false
            }
        }
        None => false,
    }
}

/// Where the daemon's stdout and stderr are appended.
///
/// The app starts it without a console, so this file is the only record of why a
/// load failed or why the process vanished on startup.
/// Largest daemon log kept before it is rotated away.
const MAX_LOG_BYTES: u64 = 4 * 1024 * 1024;

fn daemon_log(app: &tauri::AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_data_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("daemon.log");
    rotate_log(&path);
    Some(path)
}

/// Keep the log from growing without bound.
///
/// It is opened in append mode and was never trimmed, so a long-lived install
/// accumulated every load and every error forever — including model paths,
/// which is someone's directory layout sitting on disk indefinitely. One
/// generation is enough: the point is a bound, not an archive.
fn rotate_log(path: &Path) {
    let Ok(size) = std::fs::metadata(path).map(|m| m.len()) else {
        return;
    };
    if size < MAX_LOG_BYTES {
        return;
    }
    let backup = path.with_extension("log.1");
    let _ = std::fs::remove_file(&backup);
    let _ = std::fs::rename(path, &backup);
}

fn spawn_daemon(
    app: &tauri::AppHandle,
    model: Option<String>,
    keep_alive_secs: Option<u64>,
    n_gpu_layers: Option<u32>,
    n_ctx: Option<u32>,
) -> Result<u16, String> {
    let settings = load_settings(app);
    let requested = settings
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let backend = resolve_backend(app, &requested);
    let bin = daemon_binary(app, &backend)?;
    let workdir = bin.parent().map(|p| p.to_path_buf());

    let mut cmd = Command::new(&bin);
    if let Some(dir) = workdir {
        cmd.current_dir(dir);
    }
    // The daemon has no console to print to, and discarding its output makes every
    // failure undiagnosable — a missing runtime DLL kills the process before it can
    // say anything. Appending to a log keeps the evidence around.
    let log = daemon_log(app)
        .and_then(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
        })
        .and_then(|f| f.try_clone().ok().map(|out| (out, f)));
    match log {
        Some((out, err)) => {
            cmd.stdout(Stdio::from(out));
            cmd.stderr(Stdio::from(err));
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let host = bind_host(&settings);
    // The daemon refuses to start unauthenticated on a routable address, which
    // would otherwise show up here as a process that exits immediately and a
    // log nobody reads. Fail before spawning, where the fix can be named.
    if is_exposed(&host) && api_key(&settings).is_none() {
        return Err(
            "listening on every interface needs an access token; set one in Settings → \
             Network, or bind to 127.0.0.1"
                .into(),
        );
    }
    cmd.arg("--port").arg(DEFAULT_PORT.to_string());
    cmd.arg("--host").arg(&host);
    // Passed through the environment rather than on the command line: a
    // process's arguments are readable by every other process on this machine,
    // and this is the one value here worth keeping out of them.
    if let Some(key) = api_key(&settings) {
        cmd.env("NEXLLAMA_API_KEY", key);
    }
    // Speculative decoding learns which n-grams tend to be followed by what;
    // keeping that next to the settings means drafts survive a restart.
    if let Ok(dir) = app.path().app_data_dir() {
        cmd.arg("--ngram-cache").arg(dir.join("ngram-cache.bin"));
    }

    let mut models: Vec<String> = settings
        .get("autoLoad")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if models.is_empty() {
        if let Some(m) = settings
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            models.push(m.to_string());
        }
    }
    if let Some(m) = model {
        if !models.contains(&m) {
            models.push(m);
        }
    }
    // A bundle is several files: the projector is what lets the model read
    // images, the draft checkpoint is what speculative decoding drafts with,
    // and a bundled MTP head needs neither. Startup loading used to take the
    // main file alone, so an auto-loaded bundle came up half a model — nothing
    // to draft with — until someone pressed Load in the library and sent the
    // same files over HTTP instead.
    let root = models_dir(app).ok();
    let root = root.as_deref();
    let files: Vec<StartupFiles> = models.iter().map(|m| startup_files(m, root)).collect();
    // The daemon pairs these by position, so they are only passed when every
    // startup model has the file: one --draft-model against two --model values
    // would otherwise give the first model's drafter to the second.
    let paired = |get: fn(&StartupFiles) -> Option<String>| -> Option<Vec<String>> {
        if !files.iter().all(|f| get(f).is_some()) {
            return None;
        }
        Some(files.iter().filter_map(get).collect())
    };
    // The head is a daemon-wide flag, so it is only used when it can only mean
    // one model.
    let mtp_head = models.len() == 1 && files.first().is_some_and(|f| f.mtp);

    for m in &models {
        cmd.arg("--model").arg(m);
    }
    if let Some(list) = paired(|f| f.projector.clone()) {
        for p in list {
            cmd.arg("--mmproj").arg(p);
        }
    }
    // Bundled head and a side-mounted checkpoint draft the same way, and the
    // engine only ever takes one: the head is both faster and free of a second
    // set of weights, so it wins whenever the file has one.
    if mtp_head {
        cmd.arg("--load-mtp");
    } else if let Some(list) = paired(|f| f.draft.clone()) {
        for d in list {
            cmd.arg("--draft-model").arg(d);
        }
    }

    let num = |key: &str| settings.get(key).and_then(|v| v.as_i64());
    let uint = |key: &str| settings.get(key).and_then(|v| v.as_u64());

    if let Some(v) = keep_alive_secs.or_else(|| uint("keepAliveSecs")) {
        cmd.arg("--keep-alive-secs").arg(v.to_string());
    }
    if let Some(v) = n_gpu_layers.map(|x| x as i64).or_else(|| num("nGpuLayers")) {
        cmd.arg("--n-gpu-layers").arg(v.to_string());
    }
    if let Some(v) = n_ctx.map(|x| x as i64).or_else(|| num("nCtx")) {
        cmd.arg("--n-ctx").arg(v.to_string());
    }
    if let Some(v) = num("nBatch") {
        cmd.arg("--n-batch").arg(v.to_string());
    }
    if let Some(v) = num("nUbatch") {
        cmd.arg("--n-ubatch").arg(v.to_string());
    }
    if let Some(v) = num("mainGpu") {
        cmd.arg("--main-gpu").arg(v.to_string());
    }
    if let Some(v) = num("threads") {
        cmd.arg("--threads").arg(v.to_string());
    }
    if let Some(v) = num("threadsBatch") {
        cmd.arg("--threads-batch").arg(v.to_string());
    }
    if let Some(v) = settings.get("loadMode").and_then(|v| v.as_str()) {
        cmd.arg("--load-mode").arg(v);
    }
    if settings
        .get("useMlock")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        cmd.arg("--mlock");
    }
    if let Some(v) = settings.get("flashAttn").and_then(|v| v.as_str()) {
        cmd.arg("--flash-attn").arg(v);
    }
    if settings.get("offloadKqv").and_then(|v| v.as_bool()) == Some(false) {
        cmd.arg("--no-offload-kqv");
    }
    if let Some(v) = settings.get("ropeFreqBase").and_then(|v| v.as_f64()) {
        if v > 0.0 {
            cmd.arg("--rope-freq-base").arg(v.to_string());
        }
    }
    if let Some(v) = settings.get("ropeFreqScale").and_then(|v| v.as_f64()) {
        if v > 0.0 {
            cmd.arg("--rope-freq-scale").arg(v.to_string());
        }
    }

    // Lets the daemon notice when this app goes away, even when it is killed
    // rather than closed. Windows additionally puts the child in a job object
    // below; elsewhere this is the only mechanism there is.
    cmd.env("NEXLLAMA_PARENT_PID", std::process::id().to_string());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    #[cfg(windows)]
    job::assign(child.id());

    // A spawned process is not a running server. Reporting success the moment
    // the process existed meant a daemon that died on a missing DLL — or never
    // got the port because something else already held it — looked exactly
    // like a healthy start, right up to the point nothing could reach it.
    let ready = wait_ready(&mut child, &host, Duration::from_secs(30));

    let state = app.state::<DaemonProc>();
    *lock(&state.child)? = Some(child);
    *lock(&state.phase)? = Phase::Running;
    ready?;
    Ok(DEFAULT_PORT)
}

/// Wait until the daemon answers on its port, or until it is plainly not going
/// to.
///
/// Returns `Ok` both when the port answers and when the process is still alive
/// but slow: loading a startup model happens before the socket is bound, and
/// failing a start that is merely slow would be worse than not checking.
fn wait_ready(child: &mut Child, host: &str, timeout: Duration) -> Result<(), String> {
    let probe = match host.trim() {
        // Unspecified addresses accept every interface, so loopback answers.
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        other => other,
    };
    let ip: std::net::IpAddr = match probe.parse() {
        Ok(ip) => ip,
        // A hostname would have to be resolved before it could be probed;
        // skipping the check beats guessing where to connect.
        Err(_) => return Ok(()),
    };
    let addr = std::net::SocketAddr::new(ip, DEFAULT_PORT);
    let deadline = Instant::now() + timeout;
    loop {
        // An exit is the case worth reporting, and it says everything: the log
        // holds the reason.
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "the daemon exited before it was ready (status: {status}); see daemon.log"
            ));
        }

        // Reachability is not enough. A leftover daemon from a hard kill answers
        // the port just as readily, and adopting it silently means every setting
        // the user just changed — the access token, the bind address — quietly
        // fails to apply, because that process was started long before them.
        if let Some(pid) = health_pid(addr) {
            return match pid {
                Some(pid) if pid == child.id() => Ok(()),
                Some(pid) => Err(format!(
                    "port {DEFAULT_PORT} is already served by another process (pid {pid}); \
                     close the other Nexllama window or stop that daemon, then start again"
                )),
                // A server that is not ours answering without a pid is still not ours.
                None => Err(format!(
                    "port {DEFAULT_PORT} is already in use by a process that is not this \
                     daemon's; stop it, then start again"
                )),
            };
        }

        if Instant::now() >= deadline {
            // Alive but not answering: give it the benefit of the doubt rather
            // than failing a start that is merely slow to load a model.
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `Some(Some(pid))` when a Nexllama daemon answers, `Some(None)` when something
/// else does, `None` when nothing answers yet.
fn health_pid(addr: std::net::SocketAddr) -> Option<Option<u32>> {
    use std::io::{Read, Write};

    let mut stream =
        std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(250)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.write_all(
        format!("GET /health HTTP/1.0\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
    );
    let mut body = String::new();
    let _ = stream.read_to_string(&mut body);
    Some(parse_pid(&body))
}

/// Pull the pid out of a health response body.
fn parse_pid(body: &str) -> Option<u32> {
    let rest = body.split("\"pid\"").nth(1)?;
    let rest = rest.split(':').nth(1)?;
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Stop the daemon and reap it.
///
/// `kill` alone leaves the process un-reaped: on Unix it lingers as a zombie,
/// and on Windows the backend DLLs it holds stay locked long enough to break an
/// update that replaces them immediately afterwards.
fn stop_child(state: &DaemonProc) -> Result<(), String> {
    let child = lock(&state.child)?.take();
    if let Ok(mut phase) = lock(&state.phase) {
        *phase = Phase::Stopped;
    }
    match child {
        Some(mut child) => child
            .kill()
            .and_then(|_| child.wait().map(|_| ()))
            .map_err(|e| e.to_string()),
        None => Ok(()),
    }
}

#[tauri::command]
fn daemon_start(
    app: tauri::AppHandle,
    state: tauri::State<DaemonProc>,
    model: Option<String>,
    keep_alive_secs: Option<u64>,
    n_gpu_layers: Option<u32>,
    n_ctx: Option<u32>,
) -> Result<u16, String> {
    {
        let mut phase = lock(&state.phase)?;
        match *phase {
            Phase::Starting => return Err("the daemon is already starting".into()),
            Phase::Stopping => {
                return Err("the daemon is shutting down; try again in a moment".into())
            }
            Phase::Running => {
                let alive = lock(&state.child)?
                    .as_mut()
                    .map(child_alive)
                    .unwrap_or(false);
                if alive {
                    // Already up. The model and settings passed here cannot be
                    // applied to a process that is running, and reporting
                    // success anyway is how a change looks applied when it was
                    // quietly dropped.
                    return Ok(DEFAULT_PORT);
                }
                *phase = Phase::Stopped;
            }
            Phase::Stopped => {}
        }
        // Claim the start before spawning: the spawn itself is slow and holds
        // no lock, and this is what stops a second caller from starting one
        // alongside it.
        *phase = Phase::Starting;
    }

    let result = spawn_daemon(&app, model, keep_alive_secs, n_gpu_layers, n_ctx);
    note_start(&state, &result);
    if let Ok(mut phase) = lock(&state.phase) {
        *phase = if result.is_ok() {
            Phase::Running
        } else {
            Phase::Stopped
        };
    }
    result
}

#[tauri::command]
fn daemon_stop(state: tauri::State<DaemonProc>) -> Result<(), String> {
    if *lock(&state.phase)? == Phase::Starting {
        return Err("the daemon is still starting; wait a moment".into());
    }
    // Marked before the kill so a start arriving while this is waiting for the
    // process to be reaped is told to wait instead of racing it.
    *lock(&state.phase)? = Phase::Stopping;
    stop_child(&state)
}

#[tauri::command]
fn daemon_status(app: tauri::AppHandle, state: tauri::State<DaemonProc>) -> serde_json::Value {
    let running = match lock(&state.child) {
        Ok(mut guard) => match guard.as_mut() {
            Some(child) => {
                if child_alive(child) {
                    true
                } else {
                    *guard = None;
                    false
                }
            }
            None => false,
        },
        Err(_) => false,
    };
    let requested = load_settings(&app)
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let backend = resolve_backend(&app, &requested);
    serde_json::json!({
        "running": running,
        "port": DEFAULT_PORT,
        "backend": backend,
        "host": bind_host(&load_settings(&app)),
        "binary": daemon_binary(&app, &backend).ok().map(|p| p.to_string_lossy().to_string()),
        // A start refused at launch has nowhere else to surface itself, so the
        // status carries it and the library card can say what to fix.
        "last_error": lock(&state.last_error).ok().and_then(|slot| slot.clone()),
    })
}

fn settings_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join("settings.json"))
}

fn load_settings(app: &tauri::AppHandle) -> serde_json::Value {
    let path = match settings_path(app) {
        Ok(p) => p,
        Err(_) => return serde_json::json!({}),
    };
    // Every failure used to collapse into "no settings": a half-written file, a
    // permission problem and a first run all looked the same, and the UI would
    // then write the defaults straight back over whatever was really there.
    match std::fs::read_to_string(&path) {
        Ok(s) => match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "nexllama: settings at {} are not valid JSON: {e}",
                    path.display()
                );
                serde_json::json!({})
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => {
            eprintln!(
                "nexllama: could not read settings at {}: {e}",
                path.display()
            );
            serde_json::json!({})
        }
    }
}

/// Write the settings file atomically.
///
/// Writing in place truncates first, so a crash or a restart at the wrong
/// moment leaves half a JSON document behind — which then reads as "no
/// settings" and gets overwritten with defaults. Writing alongside and renaming
/// is atomic within a directory on every platform this runs on.
fn write_settings(app: &tauri::AppHandle, settings: &serde_json::Value) -> Result<(), String> {
    let path = settings_path(app)?;
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    match std::fs::rename(&tmp, &path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e.to_string())
        }
    }
}

#[tauri::command]
fn get_settings(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let mut settings = load_settings(&app);
    // The token is not in the file, but the UI still has to show and copy it.
    if let Some(obj) = settings.as_object_mut() {
        obj.insert(
            "apiKey".to_string(),
            serde_json::Value::String(secret().unwrap_or_default()),
        );
    }
    Ok(settings)
}

#[tauri::command]
fn set_settings(app: tauri::AppHandle, settings: serde_json::Value) -> Result<(), String> {
    let mut stored = settings;
    // The token never reaches the file: it goes to the OS credential store, and
    // any copy left there by an earlier version is dropped here.
    if let Some(key) = stored.get("apiKey").and_then(|v| v.as_str()) {
        store_secret(key.trim())?;
        if let Some(obj) = stored.as_object_mut() {
            obj.remove("apiKey");
        }
    }
    write_settings(&app, &stored)
}

#[tauri::command]
fn set_autostart(app: tauri::AppHandle, enabled: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let manager = app.autolaunch();
    if enabled {
        manager.enable().map_err(|e| e.to_string())
    } else {
        manager.disable().map_err(|e| e.to_string())
    }
}

fn models_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let configured = load_settings(app)
        .get("modelsDir")
        .and_then(|v| v.as_str())
        .filter(|d| !d.trim().is_empty())
        .map(PathBuf::from);
    let dir = match configured {
        Some(d) => d,
        None => app
            .path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models"),
    };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

#[tauri::command]
fn models_dir_cmd(app: tauri::AppHandle) -> Result<String, String> {
    models_dir(&app).map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn open_models_dir(app: tauri::AppHandle) -> Result<(), String> {
    let dir = models_dir(&app)?;
    reveal_in_file_manager(&dir).map_err(|e| e.to_string())
}

/// Open a folder in whatever this desktop uses to show folders.
///
/// Run-time for the same reason as [`platform_gpu_text`]: every arm gets
/// compiled everywhere, so a typo in the Linux one is not a surprise saved for
/// whoever first builds there.
fn reveal_in_file_manager(dir: &Path) -> std::io::Result<()> {
    let opener = match std::env::consts::OS {
        "macos" => "open",
        "linux" => "xdg-open",
        _ => "explorer",
    };
    Command::new(opener).arg(dir).spawn().map(|_| ())
}

/// What comes with an auto-loaded model.
struct StartupFiles {
    projector: Option<String>,
    draft: Option<String>,
    mtp: bool,
}

/// Reads a model's own files the way the library does, so that starting the
/// app on a model and clicking Load on it in the library cannot disagree about
/// what belongs together.
fn startup_files(model: &str, root: Option<&Path>) -> StartupFiles {
    let mut out = StartupFiles {
        projector: None,
        draft: None,
        mtp: false,
    };
    out.mtp = gguf_probe::probe(model)
        .map(|p| p.has_mtp() && p.classify() == gguf_probe::ModelClass::Llm)
        .unwrap_or(false);
    let path = Path::new(model);
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => return out,
    };
    // The library gives a file in the models root a bundle of its own when the
    // root holds several models: scanning it for a projector would hand this
    // model somebody else's. One model in a folder of its own — weights with
    // their projector and draft beside them — is the other half of that rule,
    // and is exactly the case where there is nothing to get wrong.
    let at_root = root.is_some_and(|r| same_dir(dir, r));
    let mut files = Vec::new();
    collect_gguf_files(dir, dir, &mut files);
    if at_root {
        // The root also contains whole subfolders of other models, and those
        // are none of this file's business.
        files.retain(|f| {
            f["path"]
                .as_str()
                .is_some_and(|p| Path::new(p).parent() == Some(dir))
        });
        if files.iter().filter(|f| f["role"] == "main").count() != 1 {
            return out;
        }
    }
    // `read_dir` order is a property of the filesystem, not of the model: the
    // same folder has to keep yielding the same drafter.
    sort_files(&mut files);
    let pick = |role: &str| {
        files
            .iter()
            .filter(|f| f["path"].as_str() != Some(model))
            .find(|f| f["role"] == role)
            .and_then(|f| f["path"].as_str().map(str::to_string))
    };
    out.projector = pick("projector");
    out.draft = pick("draft");
    out
}

/// Windows paths are case-insensitive and the models folder is set by hand, so
/// equality here cannot be byte equality.
fn same_dir(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

fn sort_files(files: &mut [serde_json::Value]) {
    files.sort_by(|a, b| {
        a["filename"]
            .as_str()
            .unwrap_or("")
            .cmp(b["filename"].as_str().unwrap_or(""))
    });
}

/// A short label for the drafter a side-mounted file implements, from its name.
///
/// The engine makes the same call from the checkpoint header when it loads;
/// this is the list's guess from the filename, so the choice is readable
/// before anything has been loaded.
fn draft_kind_label(filename: &str) -> &'static str {
    let n = filename.to_ascii_lowercase();
    if n.contains("dflash") {
        "DFlash"
    } else if n.contains("eagle") {
        "EAGLE-3"
    } else if n.contains("mtp") || n.contains("nextn") {
        "MTP"
    } else {
        "draft"
    }
}

fn role_of(filename: &str) -> &'static str {
    let n = filename.to_ascii_lowercase();
    if n.contains("mmproj") || n.contains("projector") {
        "projector"
    } else if n.contains("draft")
        || n.contains("eagle")
        || n.contains("mtp")
        || n.contains("dflash")
    {
        "draft"
    } else if n.contains("lora") || n.contains("adapter") {
        "lora"
    } else if n.contains("vocoder") {
        "vocoder"
    } else {
        "main"
    }
}

fn collect_gguf_files(dir: &Path, base: &Path, out: &mut Vec<serde_json::Value>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, base, out);
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        let rel = path.strip_prefix(base).unwrap_or(&path);
        let filename = rel.to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(serde_json::json!({
            "filename": filename,
            "role": role_of(&filename),
            "size": size,
            "path": path.to_string_lossy(),
        }));
    }
}

fn bundle(name: &str, dir: &str, files: Vec<serde_json::Value>) -> serde_json::Value {
    let pick = |role: &str| {
        files
            .iter()
            .find(|f| f["role"] == role)
            .map(|f| f["path"].clone())
    };
    // The name is a hint; the header is the verdict. Main weights get named
    // after the family they serve — "…-MTP", "…-EAGLE" — exactly the way draft
    // heads are, so a folder can hold two files whose names both say "draft"
    // while only one is a full model. Among the candidates, the largest one
    // whose header can actually be read is the main file: a stripped head is
    // never the biggest GGUF in its own folder, and loading the second-biggest
    // file as the model is the one outcome worse than offering nothing.
    let main = {
        let mut candidates: Vec<&serde_json::Value> = files
            .iter()
            .filter(|f| matches!(f["role"].as_str(), Some("main") | Some("draft")))
            .collect();
        candidates.sort_by_key(|f| std::cmp::Reverse(f["size"].as_u64().unwrap_or(0)));
        candidates
            .iter()
            .find(|f| {
                f["path"]
                    .as_str()
                    .map(|p| gguf_probe::probe(p).is_ok())
                    .unwrap_or(false)
            })
            .map(|f| f["path"].clone())
    };
    let loras: Vec<serde_json::Value> = files
        .iter()
        .filter(|f| f["role"] == "lora")
        .map(|f| f["path"].clone())
        .collect();
    let size: u64 = files.iter().filter_map(|f| f["size"].as_u64()).sum();
    // Read the family out of the main file's header so the library can group it
    // and chat can refuse the models that cannot chat.
    let probe = main
        .as_ref()
        .and_then(|p| p.as_str())
        .and_then(|p| gguf_probe::probe(p).ok());
    let class = probe
        .as_ref()
        .map(|p| p.classify().as_str().to_string())
        .unwrap_or_else(|| gguf_probe::ModelClass::Llm.as_str().to_string());
    let mtp = probe.as_ref().is_some_and(gguf_probe::Probe::has_mtp);
    serde_json::json!({
        "name": name,
        "dir": dir,
        "size": size,
        "main": main,
        "projector": pick("projector"),
        "draft": pick("draft"),
        "loras": loras,
        "class": class,
        "mtp": mtp,
        "draft_kind": pick("draft").and_then(|d| d.as_str().map(draft_kind_label)),
        "kind": if pick("projector").is_some() {
            "vision"
        } else if pick("draft").is_some() {
            "speculative"
        } else {
            class.as_str()
        },
        "files": files,
    })
}

#[tauri::command]
fn list_local_models(app: tauri::AppHandle) -> Result<Vec<serde_json::Value>, String> {
    let dir = models_dir(&app)?;
    let mut bundles = Vec::new();
    let mut loose: Vec<serde_json::Value> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            let mut files = Vec::new();
            collect_gguf_files(&path, &path, &mut files);
            sort_files(&mut files);
            if !files.is_empty() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model")
                    .to_string();
                bundles.push(bundle(&name, &name, files));
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model")
                .to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            loose.push(serde_json::json!({
                "filename": filename,
                "role": role_of(&filename),
                "size": size,
                "path": path.to_string_lossy(),
            }));
        }
    }
    // The folder the user pointed the app at is often a single model's own
    // folder — projector, draft checkpoint and bundled head beside the weights.
    // Listing those as five separate models asked which one to load, and every
    // answer except the weights was wrong.
    sort_files(&mut loose);
    bundles.extend(root_bundles(loose));
    bundles.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
    });
    Ok(bundles)
}

/// Turns the loose `.gguf` files sitting in the models folder itself into rows.
///
/// Exactly one of them being the main file means the rest are that model's own
/// files, so they belong in one row. Two or more main files means a folder of
/// unrelated models with no way to tell whose projector is whose, so each file
/// keeps a row of its own.
fn root_bundles(files: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let mut mains = files.iter().filter(|f| f["role"] == "main");
    let (Some(main), None) = (mains.next(), mains.next()) else {
        return files
            .into_iter()
            .map(|f| {
                let name = stem_of(f["filename"].as_str());
                bundle(&name, "", vec![f])
            })
            .collect();
    };
    let name = stem_of(main["filename"].as_str());
    vec![bundle(&name, "", files)]
}

/// The row's name is what a file is called without its extension.
fn stem_of(filename: Option<&str>) -> String {
    let filename = filename.unwrap_or("model");
    Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
        .to_string()
}

#[tauri::command]
async fn hub_search(
    query: String,
    author: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<RepoSummary>, String> {
    let client = HubClient::new();
    let limit = limit.unwrap_or(20);
    match author.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
        // Naming the author turns "hundreds of qwen repos" into a list worth
        // reading; see `TRUSTED_AUTHORS` for why those three.
        Some(author) => client.search_by_author(author, &query, limit).await,
        None => client.search(&query, limit).await,
    }
    .map_err(|e| e.to_string())
}

/// The authors the download page offers first.
#[tauri::command]
fn hub_trusted_authors() -> Vec<&'static str> {
    infer_hub::TRUSTED_AUTHORS.to_vec()
}

#[tauri::command]
async fn hub_list_gguf(repo: String) -> Result<Vec<GgufFile>, String> {
    HubClient::new()
        .list_gguf(&repo)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn hub_download(
    app: tauri::AppHandle,
    repo: String,
    filename: String,
) -> Result<String, String> {
    // The destination is always the models directory. It used to be a caller
    // supplied path, which made this command an arbitrary-file-write primitive:
    // aim it at the backend folder and the next daemon start would run a
    // downloaded `infer-daemon.exe`.
    let dir = models_dir(&app)?;
    let filename = filename.trim().to_string();
    if !filename.to_ascii_lowercase().ends_with(".gguf") {
        return Err("only .gguf files can be downloaded".to_string());
    }
    let client = HubClient::new();
    let handle = app.clone();
    let repo_ev = repo.clone();
    let file_ev = filename.clone();
    let path = client
        .download(&repo, &filename, &dir, move |p| {
            let _ = handle.emit(
                "download://progress",
                serde_json::json!({
                    "repo": &repo_ev,
                    "filename": &file_ev,
                    "downloaded": p.downloaded,
                    "total": p.total,
                }),
            );
        })
        .await
        .map_err(|e| e.to_string())?;

    let out = path.to_string_lossy().to_string();
    let _ = app.emit("download://done", serde_json::json!({ "path": out }));
    Ok(out)
}

#[derive(serde::Deserialize)]
struct BackendManifest {
    version: String,
    #[serde(default)]
    llama_cpp: Option<String>,
    #[serde(default)]
    backends: std::collections::HashMap<String, BackendEntry>,
    /// Ed25519 signature over `manifest_payload`, hex-encoded.
    #[serde(default)]
    signature: Option<String>,
}

#[derive(serde::Deserialize)]
struct BackendEntry {
    url: String,
    sha256: String,
}

/// Largest manifest accepted. It is a few hundred bytes of JSON.
const MAX_MANIFEST_BYTES: u64 = 1 << 20;
/// Largest backend archive accepted.
const MAX_BACKEND_BYTES: u64 = 1 << 30;
/// Largest total an archive may expand to once unpacked.
const MAX_UNPACKED_BYTES: u64 = 1 << 31;

/// The bytes an update signature covers, rebuilt from the parsed manifest.
///
/// Signing the parsed structure rather than the raw text means the check does
/// not depend on key order or formatting, which a text signature would not
/// survive being re-serialised by whatever produced it.
fn manifest_payload(m: &BackendManifest) -> String {
    let mut parts = vec![
        format!("version={}", m.version),
        format!("llama_cpp={}", m.llama_cpp.as_deref().unwrap_or_default()),
    ];
    let mut names: Vec<&String> = m.backends.keys().collect();
    names.sort();
    for name in names {
        let e = &m.backends[name];
        parts.push(format!(
            "backend={name};url={};sha256={}",
            e.url,
            e.sha256.to_ascii_lowercase()
        ));
    }
    parts.join("\n")
}

/// The Ed25519 public key trusted to sign update manifests.
///
/// A checksum only proves the archive matches the manifest that named it — and
/// both come from the same request. Whoever can replace one can replace the
/// other, so without a signature the check is theatre: a hijacked manifest
/// simply hands over a matching hash and the app installs it.
///
/// No key is compiled in by default, on purpose. An updater that trusts any
/// key is an updater that checks none. Set one in Settings → Backend updates,
/// or bake it in with `NEXLLAMA_UPDATE_KEY` at build time.
fn trusted_update_key(app: &tauri::AppHandle) -> Result<[u8; 32], String> {
    let hex_key = load_settings(app)
        .get("updatePublicKey")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .or_else(|| option_env!("NEXLLAMA_UPDATE_KEY").map(str::to_string))
        .ok_or_else(|| {
            "no update signing key configured: paste the public key into Settings → Backend \
             updates (make one with `cargo run --example new-update-key`)"
                .to_string()
        })?;
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_key.trim(), &mut out)
        .map_err(|_| "the configured update key is not 32 hex-encoded bytes".to_string())?;
    Ok(out)
}

/// Check a manifest's signature before anything in it is acted upon.
fn verify_manifest(app: &tauri::AppHandle, m: &BackendManifest) -> Result<(), String> {
    verify_signature(&trusted_update_key(app)?, m)
}

/// Check a manifest's signature against a key.
fn verify_signature(key_bytes: &[u8; 32], m: &BackendManifest) -> Result<(), String> {
    let key = ed25519_dalek::VerifyingKey::from_bytes(key_bytes)
        .map_err(|_| "the configured update key is not a valid Ed25519 key".to_string())?;
    let sig_hex = m
        .signature
        .as_deref()
        .ok_or_else(|| "the update manifest carries no signature".to_string())?;
    let mut sig_bytes = [0u8; 64];
    hex::decode_to_slice(sig_hex.trim(), &mut sig_bytes)
        .map_err(|_| "the manifest signature is not 64 hex-encoded bytes".to_string())?;
    use ed25519_dalek::Verifier;
    key.verify(
        manifest_payload(m).as_bytes(),
        &ed25519_dalek::Signature::from_bytes(&sig_bytes),
    )
    .map_err(|_| "the manifest signature does not match this key".to_string())
}

/// Refuse any update URL that is not HTTPS.
///
/// The manifest URL is typed by hand and the archive URL comes from the
/// manifest, so neither is trusted: over plain HTTP either can be rewritten in
/// transit, and the checksum cannot tell, because it is rewritten too.
fn require_https(url: &str) -> Result<(), String> {
    match reqwest::Url::parse(url) {
        Ok(u) if u.scheme() == "https" => Ok(()),
        Ok(_) => Err(format!("refusing to fetch over a non-HTTPS URL: {url}")),
        Err(_) => Err(format!("not a valid URL: {url}")),
    }
}

/// Fetch an update artifact over HTTPS, size-capped.
async fn fetch_https(url: &str, max_bytes: u64) -> Result<Vec<u8>, String> {
    require_https(url)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    // Redirects are followed, so the URL that was checked above is not
    // necessarily the one that answered.
    if resp.url().scheme() != "https" {
        return Err(format!("redirected to a non-HTTPS URL: {}", resp.url()));
    }
    if let Some(len) = resp.content_length() {
        if len > max_bytes {
            return Err(format!("{len} bytes is over the {max_bytes} byte limit"));
        }
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
    if bytes.len() as u64 > max_bytes {
        return Err(format!("the download exceeded the {max_bytes} byte limit"));
    }
    Ok(bytes)
}

/// Fetch and verify an update manifest.
async fn fetch_manifest(app: &tauri::AppHandle, url: &str) -> Result<BackendManifest, String> {
    let bytes = fetch_https(url, MAX_MANIFEST_BYTES).await?;
    let manifest: BackendManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("the manifest is not valid JSON: {e}"))?;
    verify_manifest(app, &manifest)?;
    Ok(manifest)
}

/// Whether `candidate` is at least `installed`, comparing dotted numbers.
///
/// Anything unparsable compares as older, so a version that changes shape
/// cannot be used to sneak an earlier build back in.
fn version_at_least(candidate: &str, installed: &str) -> bool {
    let parts = |s: &str| -> Vec<u64> {
        s.split('.')
            .map(|p| p.trim().parse::<u64>().unwrap_or(0))
            .collect()
    };
    let (a, b) = (parts(candidate), parts(installed));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x > y;
        }
    }
    true
}

fn manifest_url(app: &tauri::AppHandle) -> Option<String> {
    load_settings(app)
        .get("updateManifestUrl")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
}

fn installed_version(app: &tauri::AppHandle) -> String {
    backends_root(app)
        .ok()
        .and_then(|r| std::fs::read_to_string(r.join("version.json")).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("version")
                .and_then(|x| x.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "bundled".to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn extract_zip(bytes: &[u8], dest: &Path) -> Result<(), String> {
    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| e.to_string())?;
    if dest.exists() {
        std::fs::remove_dir_all(dest).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    // A zip declares its own sizes, so the running total is what stops a
    // few-hundred-kilobyte download from unpacking into a disk full of zeros.
    let mut unpacked = 0u64;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(rel) = file.enclosed_name() else {
            continue;
        };
        let out = dest.join(rel);
        if file.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
        } else {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
            unpacked += std::io::copy(&mut file, &mut f).map_err(|e| e.to_string())?;
            if unpacked > MAX_UNPACKED_BYTES {
                // Leave nothing half-written behind for the next start to find.
                let _ = std::fs::remove_dir_all(dest);
                return Err(format!(
                    "the archive expands past {MAX_UNPACKED_BYTES} bytes"
                ));
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn backend_update_status(app: tauri::AppHandle) -> serde_json::Value {
    serde_json::json!({
        "current": installed_version(&app),
        "manifest_url": manifest_url(&app),
        "available": available_backends(&app),
    })
}

#[tauri::command]
async fn check_backend_update(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let url = manifest_url(&app).ok_or("no update manifest URL configured")?;
    let manifest = fetch_manifest(&app, &url).await?;
    let current = installed_version(&app);
    // Only advertise names we recognise: the keys are arbitrary strings chosen
    // by whoever hosts the manifest, and listing one is an invitation to click
    // it.
    let mut backends: Vec<&String> = manifest
        .backends
        .keys()
        .filter(|b| is_backend_name(b))
        .collect();
    backends.sort();
    // "bundled" is not a version, so any signed release counts as newer than
    // it. Otherwise only offer something that is actually ahead, so an older
    // manifest cannot be used to roll the backend back.
    let ahead = current == "bundled" || version_at_least(&manifest.version, &current);
    Ok(serde_json::json!({
        "current": current,
        "remote": manifest.version,
        "llama_cpp": manifest.llama_cpp,
        "backends": backends,
        "update_available": ahead && manifest.version != current,
        "signed": manifest.signature.is_some(),
    }))
}

#[tauri::command]
async fn apply_backend_update(
    app: tauri::AppHandle,
    state: tauri::State<'_, DaemonProc>,
    backend: String,
) -> Result<serde_json::Value, String> {
    let url = manifest_url(&app).ok_or("no update manifest URL configured")?;
    let manifest = fetch_manifest(&app, &url).await?;
    // Downgrades are how a known-bad build gets reinstalled, and "newer" is
    // decided locally against what is already on disk rather than trusted from
    // the manifest.
    let current = installed_version(&app);
    if current != "bundled" && !version_at_least(&manifest.version, &current) {
        return Err(format!(
            "refusing to downgrade: {current} is installed, the manifest offers {}",
            manifest.version
        ));
    }
    // `backend` is a manifest key and therefore remote-controlled. It is about
    // to become a directory that gets deleted and unpacked into, so it has to
    // be one of ours before anything else happens.
    if !is_backend_name(&backend) {
        return Err(format!(
            "unknown backend '{backend}'; expected one of: {}",
            BACKENDS.join(", ")
        ));
    }
    let entry = manifest
        .backends
        .get(&backend)
        .ok_or_else(|| format!("backend '{backend}' not present in manifest"))?;

    let bytes = fetch_https(&entry.url, MAX_BACKEND_BYTES).await?;
    let got = sha256_hex(&bytes);
    if !got.eq_ignore_ascii_case(&entry.sha256) {
        return Err(format!(
            "checksum mismatch for {backend}: expected {}, got {got}",
            entry.sha256
        ));
    }

    // The daemon holds the backend DLLs, so it has to be gone before they are
    // replaced — and stopped for real: an update that cannot stop it must fail
    // rather than overwrite files out from under a running process.
    let was_running = *lock(&state.phase)? == Phase::Running;
    stop_child(&state)?;

    let root = backends_root(&app)?;
    let dest = child_of(&root, &backend);
    if dest.exists() && !is_within(&root, &dest) {
        return Err(format!(
            "refusing to unpack outside the backends folder: {}",
            dest.display()
        ));
    }
    extract_zip(&bytes, &dest)?;
    std::fs::write(
        root.join("version.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "version": manifest.version,
            "llama_cpp": manifest.llama_cpp,
        }))
        .unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;

    // Only bring it back if it was up before: stopping the daemon and then
    // having an update start it anyway is the kind of surprise that puts a
    // model back in VRAM when the user had taken it out.
    let restarted = if was_running {
        spawn_daemon(&app, None, None, None, None)?;
        true
    } else {
        false
    };

    Ok(serde_json::json!({
        "status": "ok",
        "backend": backend,
        "version": manifest.version,
        "restarted": restarted,
    }))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--autostart"]),
        ))
        .manage(DaemonProc::default())
        .setup(|app| {
            let handle = app.handle().clone();

            if std::env::args().any(|a| a == "--autostart") {
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.minimize();
                }
            }

            let settings = load_settings(&handle);
            let auto = settings
                .get("autoStartDaemon")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if auto {
                let model = settings
                    .get("model")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let keep = settings.get("keepAliveSecs").and_then(|v| v.as_u64());
                let ngl = settings
                    .get("nGpuLayers")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let nctx = settings
                    .get("nCtx")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let result = spawn_daemon(&handle, model, keep, ngl, nctx);
                if let Err(e) = &result {
                    eprintln!("[nexllama] daemon autostart skipped: {e}");
                }
                // The window is not up yet, so this is the only chance to keep
                // the reason: the library card reads it back from the status.
                note_start(&handle.state::<DaemonProc>(), &result);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend_info,
            daemon_binary_info,
            daemon_running,
            daemon_start,
            daemon_stop,
            daemon_status,
            get_settings,
            set_settings,
            set_autostart,
            models_dir_cmd,
            open_models_dir,
            list_local_models,
            backend_update_status,
            check_backend_update,
            apply_backend_update,
            hub_search,
            hub_list_gguf,
            hub_download,
            hub_trusted_authors,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app.try_state::<DaemonProc>() {
                    let _ = stop_child(&state);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A runtime library name, so the fixtures below name a real file on
    /// whatever platform the test is running on.
    #[cfg(windows)]
    const TEST_LIB: &str = "llama.dll";
    #[cfg(target_os = "macos")]
    const TEST_LIB: &str = "libllama.dylib";
    #[cfg(not(any(windows, target_os = "macos")))]
    const TEST_LIB: &str = "libllama.so";

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// The models folder is often one model's own folder: weights with a
    /// projector and a draft beside them and nothing else, in which case a
    /// start has to take all three. A folder holding two models has no way to
    /// say whose projector is whose, so there a file brings nothing along —
    /// and a model in its own subfolder under the root keeps working as ever.
    #[test]
    fn startup_files_follow_the_bundles_the_library_would_build() {
        let here = std::env::temp_dir().join(format!("nexllama-here-{}", std::process::id()));
        let there = std::env::temp_dir().join(format!("nexllama-there-{}", std::process::id()));
        let sub = here.join("qwen");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(&there).unwrap();
        let weights = here.join("flat.gguf");
        let projector = here.join("mmproj-x.gguf");
        let draft = here.join("dflash-x.gguf");
        let main = sub.join("main.gguf");
        let sub_proj = sub.join("mmproj-y.gguf");
        let sub_draft = sub.join("dflash-y.gguf");
        let first = there.join("a.gguf");
        let second = there.join("b.gguf");
        let shared_proj = there.join("mmproj-x.gguf");
        let files = [
            &weights,
            &projector,
            &draft,
            &main,
            &sub_proj,
            &sub_draft,
            &first,
            &second,
            &shared_proj,
        ];
        for p in files {
            std::fs::write(p, b"not a gguf").unwrap();
        }
        let here = here.to_string_lossy().to_string();
        let here = Path::new(&here);
        let there = there.to_string_lossy().to_string();
        let there = Path::new(&there);

        let flat = startup_files(&weights.to_string_lossy(), Some(here));
        assert_eq!(
            (flat.projector.as_deref(), flat.draft.as_deref()),
            (projector.to_str(), draft.to_str())
        );

        let bundled = startup_files(&main.to_string_lossy(), Some(here));
        assert_eq!(
            (bundled.projector.as_deref(), bundled.draft.as_deref()),
            (sub_proj.to_str(), sub_draft.to_str())
        );

        for model in [&first, &second] {
            let alone = startup_files(&model.to_string_lossy(), Some(there));
            assert_eq!((alone.projector, alone.draft), (None, None));
        }
        std::fs::remove_dir_all(here).ok();
        std::fs::remove_dir_all(there).ok();
    }

    /// One model plus its own files is one row in the library; a folder of
    /// unrelated models keeps one row each.
    #[test]
    fn loose_files_in_the_models_folder_form_one_row() {
        let entry = |filename: &str| {
            serde_json::json!({
                "filename": filename,
                "role": role_of(filename),
                "size": 1u64,
                "path": format!("C:/models/{filename}"),
            })
        };
        let bundle = |files: Vec<serde_json::Value>| root_bundles(files);

        let one = bundle(vec![
            entry("mmproj-x.gguf"),
            entry("base-model.gguf"),
            entry("dflash-x.gguf"),
        ]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0]["name"], "base-model");
        assert_eq!(
            one[0]["main"].as_str().and_then(|p| p.rsplit('/').next()),
            Some("base-model.gguf")
        );
        assert!(one[0]["projector"].is_string());
        assert!(one[0]["draft"].is_string());
        assert_eq!(one[0]["files"].as_array().map(Vec::len), Some(3));

        let many = bundle(vec![entry("a-model.gguf"), entry("b-model.gguf")]);
        assert_eq!(many.len(), 2);
        assert_eq!(many[0]["name"], "a-model");
        assert_eq!(many[1]["name"], "b-model");
    }

    #[test]
    fn zip_roundtrip() {
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            w.start_file(DAEMON_EXE, opts).unwrap();
            std::io::Write::write_all(&mut w, b"binary").unwrap();
            w.start_file(TEST_LIB, opts).unwrap();
            std::io::Write::write_all(&mut w, b"lib").unwrap();
            w.finish().unwrap();
        }
        let dir = std::env::temp_dir().join("nexllama-zip-test");
        let _ = std::fs::remove_dir_all(&dir);
        extract_zip(&buf, &dir).unwrap();
        assert_eq!(std::fs::read(dir.join(DAEMON_EXE)).unwrap(), b"binary");
        assert_eq!(std::fs::read(dir.join(TEST_LIB)).unwrap(), b"lib");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bind_host_defaults_to_loopback() {
        let settings = serde_json::json!({});
        assert_eq!(bind_host(&settings), "127.0.0.1");
        assert!(!is_exposed(&bind_host(&settings)));

        let settings = serde_json::json!({ "bindHost": "  " });
        assert_eq!(bind_host(&settings), "127.0.0.1");
    }

    #[test]
    fn wildcards_count_as_exposed() {
        for host in ["0.0.0.0", "::", "[::]", "192.168.0.5", "example.local"] {
            assert!(is_exposed(host), "{host} should count as exposed");
        }
        for host in ["127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"] {
            assert!(!is_exposed(host), "{host} should count as local");
        }
    }

    fn entry(url: &str, sha256: &str) -> BackendEntry {
        BackendEntry {
            url: url.to_string(),
            sha256: sha256.to_string(),
        }
    }

    fn manifest(version: &str, sha256: &str) -> BackendManifest {
        BackendManifest {
            version: version.to_string(),
            llama_cpp: Some("b1".to_string()),
            backends: [("cpu".to_string(), entry("https://x/cpu.zip", sha256))]
                .into_iter()
                .collect(),
            signature: None,
        }
    }

    #[test]
    fn backend_names_must_come_from_the_whitelist() {
        for ok in ["cpu", "cuda", "vulkan"] {
            assert!(is_backend_name(ok), "{ok} is a backend");
        }
        // Anything else would be joined onto a path and then executed.
        for bad in [
            "",
            "..",
            "../..",
            "CPU",
            "cpu/../cuda",
            "C:\\evil",
            "cpu.exe",
        ] {
            assert!(!is_backend_name(bad), "{bad} must not be a backend");
        }
    }

    #[test]
    fn version_comparison_blocks_downgrades() {
        assert!(version_at_least("0.2.0", "0.1.0"));
        assert!(
            version_at_least("0.1.0", "0.1.0"),
            "the same version is fine"
        );
        assert!(
            !version_at_least("0.1.0", "0.2.0"),
            "an older build must lose"
        );
        assert!(version_at_least("0.10.0", "0.9.0"), "numeric, not lexical");
        // "bundled" carries no numbers, so it compares as zero — which is
        // exactly why the callers test for it by name instead of comparing.
        assert!(version_at_least("0.1.0", "bundled"));
    }

    #[test]
    fn a_signature_covers_the_whole_manifest() {
        assert_ne!(
            manifest_payload(&manifest("0.2.0", "abc")),
            manifest_payload(&manifest("0.3.0", "abc")),
            "the version is signed"
        );
        assert_ne!(
            manifest_payload(&manifest("0.2.0", "abc")),
            manifest_payload(&manifest("0.2.0", "def")),
            "the archive hash is signed"
        );
    }

    #[test]
    fn an_unsigned_or_tampered_manifest_is_refused() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let key = signing.verifying_key().to_bytes();

        let mut m = manifest("0.2.0", "abc");
        assert!(
            verify_signature(&key, &m).is_err(),
            "a manifest with no signature must not install"
        );

        m.signature = Some(hex::encode(
            signing.sign(manifest_payload(&m).as_bytes()).to_bytes(),
        ));
        assert!(verify_signature(&key, &m).is_ok());

        // Same signature, different archive: editing the manifest after
        // signing has to invalidate it, or the signature means nothing.
        let mut tampered = manifest("0.2.0", "def");
        tampered.signature = m.signature.clone();
        assert!(verify_signature(&key, &tampered).is_err());

        // A different key must not validate either.
        let other = SigningKey::from_bytes(&[9u8; 32])
            .verifying_key()
            .to_bytes();
        assert!(verify_signature(&other, &m).is_err());
    }

    #[test]
    fn only_https_update_urls_are_fetched() {
        for bad in [
            "http://x/manifest.json",
            "file:///x",
            "ftp://x/y",
            "not a url",
        ] {
            assert!(require_https(bad).is_err(), "{bad} must be refused");
        }
        assert!(require_https("https://x/manifest.json").is_ok());
    }

    #[test]
    fn blank_api_key_means_no_auth() {
        assert_eq!(api_key(&serde_json::json!({})), None);
        assert_eq!(api_key(&serde_json::json!({ "apiKey": "" })), None);
        assert_eq!(api_key(&serde_json::json!({ "apiKey": "  " })), None);
        assert_eq!(
            api_key(&serde_json::json!({ "apiKey": " secret " })).as_deref(),
            Some("secret")
        );
    }

    #[test]
    fn health_pid_is_read_from_a_real_response() {
        let body = "HTTP/1.1 200 OK\r\n\r\n{\"status\":\"ok\",\"pid\":39504,\"models_loaded\":0}";
        assert_eq!(parse_pid(body), Some(39504));
        // Ordered differently, and with no spaces.
        assert_eq!(parse_pid("{\"pid\":7,\"status\":\"ok\"}"), Some(7));
    }

    #[test]
    fn health_pid_is_none_for_other_servers() {
        assert_eq!(parse_pid("{}"), None);
        assert_eq!(parse_pid("<html>hello</html>"), None);
        // A `pid` that is not a number is not a pid.
        assert_eq!(parse_pid("{\"pid\": \"abc\"}"), None);
    }
}
