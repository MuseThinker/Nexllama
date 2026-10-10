//! What this machine can offer a model, measured by llama.cpp's own rules.
//!
//! Everything here is read from ggml's device registry rather than from the
//! platform. That is deliberate: llama.cpp is what actually allocates the
//! memory, so the numbers it reports about itself are the only ones worth
//! fitting a model against. A hand-written probe would have to ask Windows,
//! macOS and Linux three separate questions and would come back with figures
//! that disagree with the allocator anyway.
//!
//! The CPU device is the useful surprise: ggml reports host RAM through it
//! (`GlobalMemoryStatusEx` on Windows, `sysconf` on Unix), so system memory
//! comes from the same enumeration as the GPUs and needs no platform code.

use llama_cpp_4::model::LlamaBackendDeviceType;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One compute device ggml can place tensors on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// ggml's device name, e.g. `CUDA0`, `Vulkan1`, `Metal`.
    pub name: String,
    /// Coarse family: `cuda`, `vulkan`, `metal`, `blas`, or `other`.
    ///
    /// Derived from the name because ggml reports no backend id, and the name
    /// is what llama.cpp's own tooling prints and matches on.
    pub backend: String,
    /// Total device memory in MiB as reported at probe time.
    pub vram_total_mb: u64,
    /// Free device memory in MiB as reported at probe time.
    pub vram_free_mb: u64,
}

/// A snapshot of the machine's capacity for running a model.
///
/// Deliberately model-independent: it says what exists, not what fits. Fitting
/// a specific checkpoint is a separate question that needs the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Host RAM in MiB, taken from the CPU device.
    pub ram_total_mb: u64,
    /// Available host RAM in MiB at probe time.
    pub ram_free_mb: u64,
    /// Usable parallelism, which is what a thread count should default near.
    pub cpu_cores: u32,
    /// Accelerators ggml can offload to. Empty on a CPU-only machine.
    pub gpus: Vec<GpuInfo>,
}

impl HardwareProfile {
    /// Whether any accelerator can take layers at all.
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// The accelerator most worth offloading to: the one with the most free
    /// memory right now.
    ///
    /// Free rather than total, because the question being answered is "where
    /// should this model go", and a card already holding another model is not
    /// the answer however large it is.
    pub fn best_gpu(&self) -> Option<&GpuInfo> {
        self.gpus.iter().max_by_key(|g| g.vram_free_mb)
    }

    /// Largest single allocation any device can take, in MiB.
    ///
    /// This is the ceiling on "will it load at all": a model spread across
    /// cards needs each piece to fit, but a model on one device needs the whole
    /// thing to fit here. Falls back to host RAM when there is no accelerator,
    /// which is exactly the machine where RAM is the only thing that matters.
    pub fn largest_free_mb(&self) -> u64 {
        self.gpus
            .iter()
            .map(|g| g.vram_free_mb)
            .max()
            .unwrap_or(self.ram_free_mb)
    }
}

/// Read the machine's capacity.
///
/// The llama backend is initialised on first use and shared process-wide; this
/// initialises it if nothing else has yet, so a daemon that has never loaded a
/// model still gets a real device list.
///
/// # Errors
///
/// Fails only if the llama backend cannot be initialised. A machine with no
/// accelerator reports an empty `gpus` and is not an error.
pub fn probe() -> Result<HardwareProfile> {
    // Held across the enumeration: the device registry is only populated while
    // a backend exists.
    let _backend = crate::engine::global_backend()?;
    Ok(unsafe { enumerate() })
}

/// Walk ggml's device registry.
///
/// # Safety
///
/// Only sound while a llama backend is alive, which [`probe`] guarantees by
/// holding one across the call. Device pointers belong to ggml and stay valid
/// for the life of the process.
unsafe fn enumerate() -> HardwareProfile {
    let mut profile = HardwareProfile {
        cpu_cores: std::thread::available_parallelism()
            .map(|n| u32::try_from(n.get()).unwrap_or(u32::MAX))
            .unwrap_or(1),
        ..Default::default()
    };

    let count = llama_cpp_sys_4::ggml_backend_dev_count();
    for i in 0..count {
        let dev = llama_cpp_sys_4::ggml_backend_dev_get(i);
        if dev.is_null() {
            continue;
        }

        let mut free: usize = 0;
        let mut total: usize = 0;
        llama_cpp_sys_4::ggml_backend_dev_memory(dev, &mut free, &mut total);

        match LlamaBackendDeviceType::from(llama_cpp_sys_4::ggml_backend_dev_type(dev)) {
            // The host is not an accelerator: its memory is the system's, so it
            // is recorded as RAM rather than listed alongside the GPUs.
            LlamaBackendDeviceType::Cpu => {
                profile.ram_total_mb = mb(total);
                profile.ram_free_mb = mb(free);
            }
            // A meta device aggregates the ones underneath it; listing it would
            // double-count every card it wraps.
            LlamaBackendDeviceType::Meta => {}
            _ => {
                let name = device_name(dev);
                profile.gpus.push(GpuInfo {
                    backend: backend_of(&name),
                    name,
                    vram_total_mb: mb(total),
                    vram_free_mb: mb(free),
                });
            }
        }
    }

    profile
}

/// The device's name, or an empty string if ggml will not give one.
///
/// # Safety
///
/// `dev` must be a live ggml device pointer.
unsafe fn device_name(dev: llama_cpp_sys_4::ggml_backend_dev_t) -> String {
    let ptr = llama_cpp_sys_4::ggml_backend_dev_name(dev);
    if ptr.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

/// Which backend a ggml device name belongs to.
///
/// Case-insensitive because the names are assembled from backend prefixes that
/// have not been consistent across llama.cpp releases.
fn backend_of(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for (needle, label) in [
        ("cuda", "cuda"),
        ("vulkan", "vulkan"),
        ("metal", "metal"),
        ("blas", "blas"),
        ("cann", "cann"),
        ("rocm", "rocm"),
        ("sycl", "sycl"),
    ] {
        if lower.contains(needle) {
            return label.to_string();
        }
    }
    "other".to_string()
}

/// Bytes to MiB, rounding down.
fn mb(bytes: usize) -> u64 {
    (bytes / (1024 * 1024)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, free: u64, total: u64) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            backend: backend_of(name),
            vram_total_mb: total,
            vram_free_mb: free,
        }
    }

    #[test]
    fn backend_names_map_to_families() {
        assert_eq!(backend_of("CUDA0"), "cuda");
        assert_eq!(backend_of("cuda0"), "cuda");
        assert_eq!(backend_of("Metal"), "metal");
        assert_eq!(backend_of("Vulkan1"), "vulkan");
        assert_eq!(backend_of("BLAS"), "blas");
        // An unrecognised name is still a usable device, just an unlabelled one.
        assert_eq!(backend_of("Something"), "other");
    }

    #[test]
    fn no_gpu_machine_falls_back_to_ram() {
        let p = HardwareProfile {
            ram_free_mb: 8 * 1024,
            ..Default::default()
        };
        assert!(!p.has_gpu());
        assert_eq!(p.best_gpu(), None);
        // With nowhere to offload, host RAM is the only ceiling there is.
        assert_eq!(p.largest_free_mb(), 8 * 1024);
    }

    #[test]
    fn best_gpu_prefers_free_over_total() {
        // A large card that is nearly full loses to a small empty one.
        let big_but_full = gpu("CUDA0", 512, 24 * 1024);
        let small_and_empty = gpu("CUDA1", 6 * 1024, 6 * 1024);
        let p = HardwareProfile {
            gpus: vec![big_but_full.clone(), small_and_empty.clone()],
            ..Default::default()
        };
        assert!(p.has_gpu());
        assert_eq!(p.best_gpu(), Some(&small_and_empty));
        assert_eq!(p.largest_free_mb(), 6 * 1024);
    }

    #[test]
    fn mb_rounds_down() {
        assert_eq!(mb(0), 0);
        assert_eq!(mb(1024 * 1024), 1);
        assert_eq!(mb(3 * 1024 * 1024 + 512 * 1024), 3);
    }
}
