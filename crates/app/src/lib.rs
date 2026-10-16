use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use infer_hub::{GgufFile, HubClient, RepoSummary};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
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

/// The port the daemon should listen on.
///
/// A stored value that is missing, zero, out of range, or not a number falls
/// back to the default: a typo here would otherwise leave the app unable to
/// reach its own daemon, with the port it actually took only in the log. Both a
/// JSON number and a numeric string are accepted, since the settings page may
/// have written either.
fn configured_port(settings: &serde_json::Value) -> u16 {
    let raw = settings.get("port");
    let value = raw.and_then(serde_json::Value::as_u64).or_else(|| {
        raw.and_then(serde_json::Value::as_str)
            .and_then(|s| s.trim().parse().ok())
    });
    value
        .and_then(|v| u16::try_from(v).ok())
        .filter(|p| *p != 0)
        .unwrap_or(DEFAULT_PORT)
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
    secret().or_else(|| settings_api_key(settings))
}

/// The access token carried in the settings file, if it has one.
///
/// Split out from [`api_key`] so it can be tested on its own: the credential
/// store it otherwise consults first is per-machine state, and a test that
/// reads it asserts whatever happens to be on the machine running it.
fn settings_api_key(settings: &serde_json::Value) -> Option<String> {
    settings
        .get("apiKey")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Credential-store account for a cloud provider's key.
///
/// One entry per provider, so removing a provider does not disturb the others.
fn cloud_account(id: &str) -> String {
    format!("cloud:{id}")
}

/// Read a cloud provider's key/token from the OS credential store.
fn cloud_secret(id: &str) -> Option<String> {
    let entry = keyring::Entry::new(SECRET_SERVICE, &cloud_account(id)).ok()?;
    entry.get_password().ok().filter(|s| !s.trim().is_empty())
}

/// Store a cloud provider's key/token, or clear it when empty.
fn store_cloud_secret(id: &str, value: &str) -> Result<(), String> {
    let entry =
        keyring::Entry::new(SECRET_SERVICE, &cloud_account(id)).map_err(|e| e.to_string())?;
    if value.is_empty() {
        return match entry.delete_credential() {
            Ok(()) => Ok(()),
            // Nothing stored yet is the ordinary case for clearing a key.
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        };
    }
    entry.set_password(value).map_err(|e| e.to_string())
}

/// The daemon's base URL as this app reaches it: a wildcard bind is dialled on
/// loopback, since that always answers.
fn daemon_base(settings: &serde_json::Value) -> String {
    let host = bind_host(settings);
    let host = match host.trim() {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
        other => other.to_string(),
    };
    format!("http://{host}:{}", configured_port(settings))
}

/// A cloud provider as the UI sends it: the non-secret parts plus, when the user
/// has just typed one, the key. The key is written to the credential store and
/// forwarded to the daemon; it never reaches the settings file.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CloudProviderArg {
    id: String,
    #[serde(default)]
    name: String,
    base_url: String,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    key: String,
}

/// The saved providers, with keys left blank (they are read from the store).
fn providers_from_settings(app: &tauri::AppHandle) -> Vec<CloudProviderArg> {
    load_settings(app)
        .get("cloudProviders")
        .and_then(|v| v.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|p| {
                    let id = p.get("id").and_then(|v| v.as_str())?.to_string();
                    if p.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
                        return None;
                    }
                    let base_url = p.get("baseUrl").and_then(|v| v.as_str())?.to_string();
                    let name = p
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&id)
                        .to_string();
                    let models = p
                        .get("models")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|m| m.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(CloudProviderArg {
                        id,
                        name,
                        base_url,
                        models,
                        key: String::new(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Push cloud providers to the daemon, keys included.
///
/// A key that arrives with the list is written to the credential store (so a
/// later restart finds it) and forwarded; a provider without one falls back to
/// whatever the store already holds. Keys travel only over loopback and never
/// touch the settings file. The daemon keeps them in memory, so this runs again
/// after every daemon start. Returns how many providers were sent.
async fn push_providers(
    app: &tauri::AppHandle,
    list: Vec<CloudProviderArg>,
) -> Result<(usize, usize), String> {
    let settings = load_settings(app);
    let mut payload: Vec<serde_json::Value> = Vec::new();
    let mut with_key = 0usize;
    for p in list {
        if p.id.trim().is_empty() || p.base_url.trim().is_empty() {
            continue;
        }
        let typed = p.key.trim();
        let api_key = if typed.is_empty() {
            // Nothing typed: take what the store already holds for this provider.
            cloud_secret(&p.id).unwrap_or_default()
        } else {
            // Persist for the next start, but a credential-store failure must not
            // stop the key reaching the daemon now.
            if let Err(e) = store_cloud_secret(&p.id, typed) {
                eprintln!("Nexllama: could not store the key for {}: {e}", p.id);
            }
            typed.to_string()
        };
        if !api_key.is_empty() {
            with_key += 1;
        }
        let name = if p.name.trim().is_empty() {
            p.base_url.clone()
        } else {
            p.name
        };
        payload.push(serde_json::json!({
            "id": p.id,
            "name": name,
            "base_url": p.base_url,
            "api_key": api_key,
            "models": p.models,
        }));
    }

    let url = format!("{}/admin/providers", daemon_base(&settings));
    let mut req = reqwest::Client::new().post(&url).json(&payload);
    if let Some(token) = api_key(&settings) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("{status}: {body}"));
    }
    Ok((payload.len(), with_key))
}

/// The `auto` alias preference stored in settings: `"local"` (default) or `"cloud"`.
fn synergy_prefer(settings: &serde_json::Value) -> String {
    settings
        .get("synergyPrefer")
        .and_then(|v| v.as_str())
        .filter(|s| *s == "cloud" || *s == "local")
        .unwrap_or("local")
        .to_string()
}

/// Whether `auto` routing is on (default false: the user opts in).
fn synergy_enabled(settings: &serde_json::Value) -> bool {
    settings
        .get("synergyEnabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Tell the daemon how the `auto` alias should behave. Best-effort.
async fn push_routing(app: &tauri::AppHandle, enabled: bool, prefer: &str) -> Result<(), String> {
    let settings = load_settings(app);
    let url = format!("{}/admin/routing", daemon_base(&settings));
    let mut req = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({ "enabled": enabled, "prefer": prefer }));
    if let Some(token) = api_key(&settings) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{}: {}", resp.status(), resp.text().await.unwrap_or_default()));
    }
    Ok(())
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
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = exe.parent().map(|d| d.join(DAEMON_EXE)) {
            if p.is_file() {
                return Some(p);
            }
        }
    }
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

/// The GPU backend this machine should run on, whether or not it is installed.
///
/// Unlike [`resolve_backend`], this ignores what is on disk: it is what drives
/// the "download the missing GPU backend" step, so an accelerator the installer
/// did not carry is fetched rather than given up on.
fn desired_gpu_backend() -> Option<&'static str> {
    let names = gpu_names();
    let nvidia = names.iter().any(|n| n.contains("nvidia")) || has_nvidia();
    if nvidia {
        return Some("cuda");
    }
    let other = names.iter().any(|n| {
        n.contains("amd") || n.contains("radeon") || n.contains("intel") || n.contains("arc")
    });
    if other {
        return Some("vulkan");
    }
    if cfg!(target_os = "macos") && names.iter().any(|n| n.contains("apple")) {
        return Some("metal");
    }
    None
}

/// Fetch the machine's GPU backend if it is not present yet. Best-effort.
///
/// Returns the backend it installed, or `None` when there was nothing to do
/// (no GPU, backend already present, or no signed manifest configured).
async fn ensure_gpu_backend(app: &tauri::AppHandle, state: &DaemonProc) -> Option<String> {
    let backend = desired_gpu_backend()?;
    if backend_exe(app, backend).is_some() {
        return None;
    }
    // Nothing to download from; CPU it is, and the UI can say why.
    manifest_url(app)?;
    match install_backend(app, state, backend).await {
        Ok(_) => {
            eprintln!("Nexllama: installed the missing {backend} backend");
            Some(backend.to_string())
        }
        Err(e) => {
            eprintln!("Nexllama: could not install the {backend} backend: {e}");
            None
        }
    }
}

fn available_backends(app: &tauri::AppHandle) -> Vec<&'static str> {
    BACKENDS
        .iter()
        .copied()
        .filter(|b| backend_exe(app, b).is_some())
        .collect()
}

/// The backends that suit this machine, best first, `cpu` always last.
///
/// A CUDA build hard-links NVIDIA's libraries (`cublas64_*`, `cudart64_*`), so
/// it is only a candidate when an NVIDIA GPU is actually present. Listing it
/// unconditionally meant a machine with only an Intel iGPU — whose preference is
/// Vulkan — fell through to CUDA on an installer that bundles `cpu` and `cuda`
/// and nothing else, and the daemon then died on a missing DLL at startup.
fn backend_order(nvidia: bool, other_gpu: bool, apple: bool) -> Vec<&'static str> {
    let mut order = Vec::new();
    if apple {
        order.push("metal");
    }
    if nvidia {
        order.push("cuda");
    }
    if other_gpu {
        order.push("vulkan");
    }
    order.push("cpu");
    order
}

fn resolve_backend(app: &tauri::AppHandle, requested: &str) -> String {
    // `backend_exe` already refuses unknown names, so anything else falls
    // through to auto-detection rather than being used to build a path.
    if requested != "auto" && backend_exe(app, requested).is_some() {
        return requested.to_string();
    }
    let names = gpu_names();
    let nvidia = names.iter().any(|n| n.contains("nvidia")) || has_nvidia();
    let other_gpu = names.iter().any(|n| {
        n.contains("amd") || n.contains("radeon") || n.contains("intel") || n.contains("arc")
    });
    let apple = cfg!(target_os = "macos") && names.iter().any(|n| n.contains("apple"));

    for b in backend_order(nvidia, other_gpu, apple) {
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
        // The backend this machine's GPU wants, installed or not — what the UI
        // offers to download when it is missing.
        "gpu_backend": desired_gpu_backend(),
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
    // Try the chosen backend, and if it cannot come up — a GPU build whose
    // runtime is missing on this machine, say — fall back to CPU rather than
    // failing the start. A slow answer beats a dialog nobody can act on.
    match try_spawn_backend(
        app,
        &settings,
        &backend,
        model.clone(),
        keep_alive_secs,
        n_gpu_layers,
        n_ctx,
    ) {
        Ok(port) => Ok(port),
        Err(e) if backend != "cpu" && backend_exe(app, "cpu").is_some() => {
            eprintln!("Nexllama: {backend} backend failed to start ({e}); falling back to cpu");
            try_spawn_backend(
                app,
                &settings,
                "cpu",
                model,
                keep_alive_secs,
                n_gpu_layers,
                n_ctx,
            )
        }
        Err(e) => Err(e),
    }
}

/// Spawn one specific backend and wait for it to answer.
#[allow(clippy::too_many_arguments)]
fn try_spawn_backend(
    app: &tauri::AppHandle,
    settings: &serde_json::Value,
    backend: &str,
    model: Option<String>,
    keep_alive_secs: Option<u64>,
    n_gpu_layers: Option<u32>,
    n_ctx: Option<u32>,
) -> Result<u16, String> {
    let bin = daemon_binary(app, backend)?;
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
    let host = bind_host(settings);
    let port = configured_port(settings);
    // The daemon refuses to start unauthenticated on a routable address, which
    // would otherwise show up here as a process that exits immediately and a
    // log nobody reads. Fail before spawning, where the fix can be named.
    if is_exposed(&host) && api_key(settings).is_none() {
        return Err(
            "listening on every interface needs an access token; set one in Settings → \
             Network, or bind to 127.0.0.1"
                .into(),
        );
    }
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--host").arg(&host);
    // Passed through the environment rather than on the command line: a
    // process's arguments are readable by every other process on this machine,
    // and this is the one value here worth keeping out of them.
    if let Some(key) = api_key(settings) {
        cmd.env("NEXLLAMA_API_KEY", key);
    }
    // Speculative decoding learns which n-grams tend to be followed by what;
    // keeping that next to the settings means drafts survive a restart.
    if let Ok(dir) = app.path().app_data_dir() {
        cmd.arg("--ngram-cache").arg(dir.join("ngram-cache.bin"));
    }

    // Model(s) to load at start. An explicit request wins; otherwise the ones
    // the user had loaded last time are brought back, so a restart (including
    // after an update) returns to the same setup. A path that has moved or been
    // deleted is skipped — the daemon already tolerates a startup model that
    // will not load — rather than failing the whole start.
    let mut models: Vec<String> = Vec::new();
    if let Some(m) = model {
        models.push(m);
    } else if let Some(list) = settings.get("autoLoad").and_then(|v| v.as_array()) {
        for p in list.iter().filter_map(|v| v.as_str()) {
            if Path::new(p).exists() {
                models.push(p.to_string());
            }
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
    // Draft length for a startup model, on whichever drafter it uses: the
    // bundled head reads `--mtp-n-max`, a side checkpoint `--draft-n-max`.
    // Passing the head's flag for a side checkpoint would load the head too.
    if let Some(v) = num("draftNMax") {
        if v > 0 {
            if mtp_head {
                cmd.arg("--mtp-n-max").arg(v.to_string());
            } else if files.iter().any(|f| f.draft.is_some()) {
                cmd.arg("--draft-n-max").arg(v.to_string());
            }
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
    let ready = wait_ready(&mut child, &host, port, Duration::from_secs(30));

    let state = app.state::<DaemonProc>();
    *lock(&state.child)? = Some(child);
    *lock(&state.phase)? = Phase::Running;
    ready?;
    // Cloud providers live only in the daemon's memory, so every start re-sends
    // them. Best-effort: a failure here leaves local inference untouched, and the
    // app pushes again whenever the user edits a provider.
    {
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            let list = providers_from_settings(&app);
            if let Err(e) = push_providers(&app, list).await {
                eprintln!("Nexllama: pushing cloud providers failed: {e}");
            }
            let prefer = synergy_prefer(&load_settings(&app));
            let enabled = synergy_enabled(&load_settings(&app));
            if let Err(e) = push_routing(&app, enabled, &prefer).await {
                eprintln!("Nexllama: pushing routing preference failed: {e}");
            }
        });
    }
    Ok(port)
}

/// Wait until the daemon answers on its port, or until it is plainly not going
/// to.
///
/// Returns `Ok` both when the port answers and when the process is still alive
/// but slow: loading a startup model happens before the socket is bound, and
/// failing a start that is merely slow would be worse than not checking.
fn wait_ready(child: &mut Child, host: &str, port: u16, timeout: Duration) -> Result<(), String> {
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
    let addr = std::net::SocketAddr::new(ip, port);
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
                    "port {port} is already served by another process (pid {pid}); \
                     close the other Nexllama window or stop that daemon, then start again"
                )),
                // A server that is not ours answering without a pid is still not ours.
                None => Err(format!(
                    "port {port} is already in use by a process that is not this \
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
async fn daemon_start(
    app: tauri::AppHandle,
    state: tauri::State<'_, DaemonProc>,
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
                    return Ok(configured_port(&load_settings(&app)));
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

    // A machine with a GPU whose backend is not installed (this build did not
    // carry it, or an older install) fetches it now — signed and checksummed —
    // so the reply runs on the GPU instead of quietly falling back to CPU.
    ensure_gpu_backend(&app, &state).await;

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
    let settings = load_settings(&app);
    let requested = settings
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let backend = resolve_backend(&app, &requested);
    serde_json::json!({
        "running": running,
        "port": configured_port(&settings),
        "backend": backend,
        "gpu_backend": desired_gpu_backend(),
        "host": bind_host(&settings),
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

/// Store a cloud provider's key/token in the OS credential store.
#[tauri::command]
fn cloud_key_set(id: String, key: String) -> Result<(), String> {
    store_cloud_secret(&id, &key)
}

/// Forget a cloud provider's key/token.
#[tauri::command]
fn cloud_key_clear(id: String) -> Result<(), String> {
    store_cloud_secret(&id, "")
}

/// Which of the named providers already have a key stored (ids only — never the
/// value, so the UI can show "set / not set" without the secret crossing IPC).
#[tauri::command]
fn cloud_key_present(ids: Vec<String>) -> Vec<String> {
    ids.into_iter()
        .filter(|id| cloud_secret(id).is_some())
        .collect()
}

/// Send the configured providers (with their keys) to the daemon.
///
/// The UI passes the current list, keys included, so a key the user just typed
/// is stored and forwarded in the same step; with no list, the saved providers
/// are read back and their keys taken from the credential store.
#[tauri::command]
async fn cloud_push_providers(
    app: tauri::AppHandle,
    providers: Option<Vec<CloudProviderArg>>,
) -> Result<serde_json::Value, String> {
    let list = match providers {
        Some(list) => list,
        None => providers_from_settings(&app),
    };
    let (count, with_key) = push_providers(&app, list).await?;
    // `withKey` lets the UI tell "saved" from "saved, but the key did not make
    // it" — the failure this whole path exists to avoid.
    Ok(serde_json::json!({ "count": count, "withKey": with_key }))
}

/// Ask a provider what it serves, via the daemon, before the user saves it.
/// Arguments for `cloud_fetch_models`, as one object.
///
/// A single struct argument rather than two positional ones: Tauri renames the
/// command's arguments, and an object with explicit camelCase fields leaves no
/// room for `base_url`/`key` to be bound to each other by mistake.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CloudProbe {
    base_url: String,
    #[serde(default)]
    key: String,
}

#[tauri::command]
async fn cloud_fetch_models(
    app: tauri::AppHandle,
    probe: CloudProbe,
) -> Result<serde_json::Value, String> {
    let settings = load_settings(&app);
    let url = format!("{}/admin/providers/test", daemon_base(&settings));
    let mut req = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({ "base_url": &probe.base_url, "api_key": &probe.key }));
    if let Some(token) = api_key(&settings) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let value: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    if !status.is_success() {
        let message = value
            .get("error")
            .and_then(|e| e.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| status.to_string());
        return Err(message);
    }
    Ok(value)
}

/// Set the `auto` alias policy (on/off and `"local"`/`"cloud"`) and push it.
///
/// Either field may be omitted to leave it unchanged.
#[tauri::command]
async fn set_routing(
    app: tauri::AppHandle,
    enabled: Option<bool>,
    prefer: Option<String>,
) -> Result<(), String> {
    let mut settings = load_settings(&app);
    if let Some(obj) = settings.as_object_mut() {
        if let Some(e) = enabled {
            obj.insert("synergyEnabled".to_string(), serde_json::Value::Bool(e));
        }
        if let Some(p) = &prefer {
            let p = if p.trim() == "cloud" { "cloud" } else { "local" };
            obj.insert(
                "synergyPrefer".to_string(),
                serde_json::Value::String(p.to_string()),
            );
        }
    }
    write_settings(&app, &settings)?;
    push_routing(&app, synergy_enabled(&settings), &synergy_prefer(&settings)).await
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
        // Fresh install: keep models on the system drive, where the model
        // libraries people already have usually live.
        None if cfg!(windows) => PathBuf::from(r"C:\models"),
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
    model_index::collect_gguf_files(dir, dir, &mut files);
    if at_root {
        // The root also contains whole subfolders of other models, and those
        // are none of this file's business — except one named for a drafter
        // (`MTP/`, an EAGLE or DFlash checkpoint), which belongs to the weights
        // beside it. The relative filename carries the folder name, so files
        // collected from there already classify as drafts.
        files.retain(|f| {
            f["path"].as_str().is_some_and(|p| {
                let parent = Path::new(p).parent();
                parent == Some(dir)
                    || parent
                        .and_then(Path::file_name)
                        .and_then(|n| n.to_str())
                        .is_some_and(model_index::is_drafter_dir)
            })
        });
        if files.iter().filter(|f| f["role"] == "main").count() != 1 {
            return out;
        }
    }
    // `read_dir` order is a property of the filesystem, not of the model: the
    // same folder has to keep yielding the same drafter.
    model_index::sort_files(&mut files);
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

/// A safe folder name for a repository id.
///
/// `owner/name` becomes `owner--name`: a separator the filesystem accepts on
/// every platform, so a repository id can be used as a directory name as it
/// stands. Anything that could escape the models folder is dropped rather than
/// sanitised into something ambiguous.
fn repo_folder_name(repo: &str) -> String {
    let cleaned: String = repo
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect();
    let cleaned = cleaned.trim_matches('.').to_string();
    if cleaned.is_empty() {
        "model".to_string()
    } else {
        cleaned
    }
}

#[tauri::command]
fn list_local_models(app: tauri::AppHandle) -> Result<Vec<serde_json::Value>, String> {
    let dir = models_dir(&app)?;
    // The same rules the daemon uses for GET /admin/models, so a library opened
    // in a browser and one opened in the app cannot disagree about what a model is.
    model_index::scan(&dir)
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
        // reading; see `TRUSTED_AUTHORS` for why those four.
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
       // One folder per model, named after the repository. A model is several
       // files — weights, sometimes a projector and a draft — and dropping them
       // into the models root mixes them with whatever else is there: the
       // library can no longer tell whose projector is whose, and the download
       // list stops being readable. The library treats a folder as a model, so
       // this is also what makes a freshly downloaded model appear as one entry.
       let dir = dir.join(repo_folder_name(&repo));
       std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
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

/// Like [`fetch_https`], but emits an `event` after every chunk so the UI can
/// show a progress bar for a multi-hundred-megabyte backend download.
async fn fetch_https_progress(
    app: &tauri::AppHandle,
    url: &str,
    max_bytes: u64,
    event: &str,
) -> Result<Vec<u8>, String> {
    use tauri::Emitter;
    require_https(url)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    if resp.url().scheme() != "https" {
        return Err(format!("redirected to a non-HTTPS URL: {}", resp.url()));
    }
    let total = resp.content_length();
    if let Some(len) = total {
        if len > max_bytes {
            return Err(format!("{len} bytes is over the {max_bytes} byte limit"));
        }
    }
    let mut out: Vec<u8> = Vec::with_capacity(total.unwrap_or(0).min(max_bytes) as usize);
    let mut downloaded: u64 = 0;
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        downloaded += chunk.len() as u64;
        if downloaded > max_bytes {
            return Err(format!("the download exceeded the {max_bytes} byte limit"));
        }
        out.extend_from_slice(&chunk);
        let _ = app.emit(
            event,
            serde_json::json!({ "downloaded": downloaded, "total": total }),
        );
    }
    Ok(out)
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
        // A URL baked in at build time means the app ships with an update
        // source and the user never has to type one.
        .or_else(|| option_env!("NEXLLAMA_UPDATE_URL").map(str::to_string))
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
        // Whether a source is compiled in, so the UI can say the user does not
        // have to type one.
        "builtin_url": option_env!("NEXLLAMA_UPDATE_URL"),
        "key_set": trusted_update_key(&app).is_ok(),
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
    install_backend(&app, &state, &backend).await
}

/// Download, verify and unpack one backend from the signed manifest.
///
/// Shared by the manual "apply update" command and the automatic GPU-backend
/// install that runs before a start, so a machine whose GPU backend is not in
/// the installer fetches it instead of silently falling back to CPU.
async fn install_backend(
    app: &tauri::AppHandle,
    state: &DaemonProc,
    backend: &str,
) -> Result<serde_json::Value, String> {
    let url = manifest_url(app).ok_or("no update manifest URL configured")?;
    let manifest = fetch_manifest(app, &url).await?;
    // Downgrades are how a known-bad build gets reinstalled, and "newer" is
    // decided locally against what is already on disk rather than trusted from
    // the manifest.
    let current = installed_version(app);
    if current != "bundled" && !version_at_least(&manifest.version, &current) {
        return Err(format!(
            "refusing to downgrade: {current} is installed, the manifest offers {}",
            manifest.version
        ));
    }
    // `backend` is a manifest key and therefore remote-controlled. It is about
    // to become a directory that gets deleted and unpacked into, so it has to
    // be one of ours before anything else happens.
    if !is_backend_name(backend) {
        return Err(format!(
            "unknown backend '{backend}'; expected one of: {}",
            BACKENDS.join(", ")
        ));
    }
    let entry = manifest
        .backends
        .get(backend)
        .ok_or_else(|| format!("backend '{backend}' not present in manifest"))?;

    let bytes = fetch_https_progress(app, &entry.url, MAX_BACKEND_BYTES, "update://progress").await?;
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
    stop_child(state)?;

    let root = backends_root(app)?;
    let dest = child_of(&root, backend);
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
        spawn_daemon(app, None, None, None, None)?;
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

/// Bring the main window back from the tray.
fn show_main(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
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
                // The service comes up with nothing loaded; the model is chosen
                // in the library. Only the daemon-level knobs are handed over.
                let keep = settings.get("keepAliveSecs").and_then(|v| v.as_u64());
                let ngl = settings
                    .get("nGpuLayers")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let nctx = settings
                    .get("nCtx")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let result = spawn_daemon(&handle, None, keep, ngl, nctx);
                if let Err(e) = &result {
                    eprintln!("[nexllama] daemon autostart skipped: {e}");
                }
                // The window is not up yet, so this is the only chance to keep
                // the reason: the library card reads it back from the status.
                note_start(&handle.state::<DaemonProc>(), &result);
            }

            // Closing the window hides the app to the tray rather than quitting
            // it, so the service (and any loaded model) stays resident the way a
            // background daemon should. Only "Quit" from the tray leaves, and
            // that path runs the exit handler that stops the daemon.
            let open = MenuItem::with_id(app, "open", "Open Nexllama", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Nexllama", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
            TrayIconBuilder::with_id("main")
                .icon(
                    app.default_window_icon()
                        .expect("window icon is bundled")
                        .clone(),
                )
                .tooltip("Nexllama")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main(app),
                    // A real exit: the run-loop teardown stops the daemon.
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main(tray.app_handle());
                    }
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
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
            cloud_key_set,
            cloud_key_clear,
            cloud_key_present,
            cloud_push_providers,
            cloud_fetch_models,
            set_routing,
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

    /// A HuggingFace download of an MTP head lands in its own `MTP/` folder
    /// beside the weights. That folder is the model's drafter, so starting on
    /// the weights has to bring it along.
    #[test]
    fn a_drafter_folder_beside_the_weights_is_picked_up() {
        let root = std::env::temp_dir().join(format!("nexllama-mtpdir-{}", std::process::id()));
        let mtp = root.join("MTP");
        std::fs::create_dir_all(&mtp).unwrap();
        let weights = root.join("model.gguf");
        let head = mtp.join("mtp-model-Q4_0.gguf");
        std::fs::write(&weights, b"not a gguf").unwrap();
        std::fs::write(&head, b"not a gguf").unwrap();

        let root_s = root.to_string_lossy().to_string();
        let files = startup_files(&weights.to_string_lossy(), Some(Path::new(&root_s)));
        assert_eq!(files.draft.as_deref(), head.to_str());
        assert_eq!(files.projector, None);

        std::fs::remove_dir_all(&root).ok();
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

    /// CUDA is only a candidate when an NVIDIA GPU is present; a machine with no
    /// discrete GPU must never fall through to it (its daemon would not load).
    #[test]
    fn cuda_is_only_offered_when_nvidia_is_present() {
        assert_eq!(backend_order(false, false, false), vec!["cpu"]);
        assert_eq!(backend_order(true, false, false), vec!["cuda", "cpu"]);
        assert_eq!(backend_order(false, true, false), vec!["vulkan", "cpu"]);
        assert_eq!(
            backend_order(true, true, false),
            vec!["cuda", "vulkan", "cpu"]
        );
        assert_eq!(backend_order(false, false, true), vec!["metal", "cpu"]);
        // No GPU at all is CPU first and only — never CUDA.
        assert!(!backend_order(false, false, false).contains(&"cuda"));
    }

    /// The credential store must actually persist a value, not fall back to an
    /// inert backend. Ignored by default because it writes to the OS store and
    /// needs one present (a headless CI box may not have it).
    #[test]
    #[ignore = "writes to the OS credential store"]
    fn keyring_round_trips_a_secret() {
        store_cloud_secret("probe-test", "hello-123").expect("store");
        assert_eq!(cloud_secret("probe-test").as_deref(), Some("hello-123"));
        store_cloud_secret("probe-test", "").expect("clear");
        assert!(cloud_secret("probe-test").is_none());
    }

    /// A port the app cannot interpret must not leave it unable to reach its own
    /// daemon; every bad shape falls back to the default.
    #[test]
    fn the_port_defaults_unless_a_valid_one_is_stored() {
        assert_eq!(configured_port(&serde_json::json!({})), DEFAULT_PORT);
        // A string, as a settings page may have written it.
        assert_eq!(
            configured_port(&serde_json::json!({ "port": "8080" })),
            8080
        );
        assert_eq!(configured_port(&serde_json::json!({ "port": 8080 })), 8080);
        // Out of range or nonsense.
        assert_eq!(
            configured_port(&serde_json::json!({ "port": 0 })),
            DEFAULT_PORT
        );
        assert_eq!(
            configured_port(&serde_json::json!({ "port": 70000 })),
            DEFAULT_PORT
        );
        assert_eq!(
            configured_port(&serde_json::json!({ "port": -1 })),
            DEFAULT_PORT
        );
        assert_eq!(
            configured_port(&serde_json::json!({ "port": "abc" })),
            DEFAULT_PORT
        );
        assert_eq!(
            configured_port(&serde_json::json!({ "port": "  " })),
            DEFAULT_PORT
        );
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
        assert_eq!(settings_api_key(&serde_json::json!({})), None);
        assert_eq!(settings_api_key(&serde_json::json!({ "apiKey": "" })), None);
        assert_eq!(settings_api_key(&serde_json::json!({ "apiKey": "  " })), None);
        assert_eq!(
            settings_api_key(&serde_json::json!({ "apiKey": " secret " })).as_deref(),
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
