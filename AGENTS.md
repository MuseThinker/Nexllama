# AGENTS.md

Notes for working in this repository. The product is **Nexllama**; the crate
directory is `crates/app` and it is **its own Cargo workspace** (excluded from
the root workspace), so app commands run from inside `crates/app`.

## Layout

- `crates/infer-core` — inference engine (speculative decoding, EAGLE-3, etc.)
- `crates/infer-daemon` — HTTP daemon that hosts a loaded model
- `crates/infer-hub` — HuggingFace search/list/download client
- `crates/model-index`, `crates/gguf-probe` — model metadata helpers
- `crates/app` — Tauri desktop app (separate workspace; `nexllama-app`)
- `frontend/` — SvelteKit UI

## Build, test, lint

Root workspace (all crates except the app):

```powershell
cargo fmt --all
cargo clippy -p infer-hub --all-targets -- -D warnings   # per crate as needed
cargo test
```

App (from `crates/app`):

```powershell
cargo clippy -- -D warnings
```

Frontend (from repo root):

```powershell
npm --prefix frontend run build     # production build into frontend/dist
npm --prefix frontend run dev       # dev server
```

There is no separate frontend typecheck/lint script; `vite build` is the gate
(it prints pre-existing a11y warnings that are not failures).

## Backends

Each backend is a self-contained `infer-daemon` build staged into
`crates/app/backends/<backend>/` (gitignored, never committed):

```powershell
scripts\build-daemon.ps1 -Backend cpu|cuda|vulkan -Release
```

It sets `CARGO_TARGET_DIR=target/backend-<backend>` so backends do not share a
target dir. `package.ps1` clears that variable before the desktop build so the
bundle lands under the app's own `target/`.

## Packaging

```powershell
scripts\package.ps1 -Edition full|lite|both    # default full
```

- `-Edition both` is required to get **both** installers from one run. A lone
  `-Edition lite` run overwrites the plain-named installer before renaming it,
  which deletes the full one.
- Output: `crates/app/target/release/bundle/nsis/Nexllama_0.1.0_x64[-lite]-setup.exe`.
- `-Native` is off by default on purpose: shipped backends must run on any CPU,
  so they are built against a baseline, not the build host's ISA.
- Default backends are `cpu` + `cuda`, plus `vulkan` **when a Vulkan SDK is
  found**. When none is found the AMD/Intel path silently uses CPU.

## Vulkan SDK (for the `vulkan` backend)

`ggml-vulkan` needs `glslc`, the Vulkan headers, `vulkan-1.lib`, and the
SPIRV-Headers CMake config (`CMakeLists.txt` under `ggml/src/ggml-vulkan`).
`build.rs` locates the SDK via `$env:VULKAN_SDK` (validated by the presence of
`Lib\vulkan-1.lib`), then the registry, then `C:\VulkanSDK\<version>`.

The LunarG download CDN is not reachable from this environment, so a **portable
SDK** is assembled under `%TEMP%\opencode\vulkan\sdk` (not committed):

```
sdk\
  Include\vulkan\ + Include\vk_video\     # Vulkan-Headers
  include\spirv\                          # SPIRV-Headers
  share\cmake\SPIRV-Headers\              # SPIRV-HeadersConfig.cmake
  Lib\vulkan-1.lib                        # synthesized from C:\Windows\System32\vulkan-1.dll
  Bin\glslc.exe + shaderc/glslang/SPIRV-Tools DLLs
```

To build or package the Vulkan backend:

```powershell
$sdk = "C:\Users\ADMINI~1\AppData\Local\Temp\opencode\vulkan\sdk"
$env:VULKAN_SDK = $sdk
$env:Path = "$sdk\Bin;$env:Path"
scripts\build-daemon.ps1 -Backend vulkan -Release
scripts\package.ps1 -Edition both
```

`vulkan-1.lib` only names symbols; the runtime `vulkan-1.dll` comes from the
installed GPU driver and is not redistributed. If the SDK directory is cleared,
it must be reassembled from Vulkan-Headers, SPIRV-Headers, and MSYS2
`shaderc`/`glslang`/`spirv-tools` packages.

## Never commit

`crates/app/backends/`, `target/`, `frontend/dist/`, `models/`, and any
`*.gguf`/`*.bin`/`*.safetensors` (see `.gitignore`).
