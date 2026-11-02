# llama-tts sidecar

`llama-tts.exe` is llama.cpp's text-to-speech frontend, built from the same
llama.cpp revision as the DLLs in each backend (`6fbe4b2`, build 25). It is a
thin executable over `llama.dll`, `llama-common.dll` and `mtmd.dll`, so it adds
about 50 KB to a backend rather than a second runtime.

`llama-tts-worker.exe` is the resident version used by the daemon by default.
It is `llama-tts.cpp` with the one-shot prompt replaced by a line protocol:
the model and mmproj load once, then each JSON line on stdin asks for one WAV.
The daemon kills it on unload, which is what releases its VRAM. The one-shot
`llama-tts.exe` remains as a fallback when the worker is missing.

The Rust crate builds llama.cpp with `LLAMA_BUILD_TOOLS=OFF`, so neither binary
is produced by `cargo build`. They are checked in here and copied next to
`infer-daemon.exe` by `scripts/build-daemon.ps1`. Both must stay on the same
llama.cpp revision as the DLLs; if the backend is bumped, rebuild them from
`tools/tts` in the matching source tree and update this note.

`llama-tts-worker.exe` is built from `llama-tts-worker.cpp` with the same
include paths and import libraries as `llama-tts`. `scripts/build-tts-worker.ps1`
takes the llama.cpp source, CMake install tree and CMake build tree:

```powershell
scripts\build-tts-worker.ps1 `
  -LlamaSource <llama.cpp source> `
  -LlamaLib <cmake install tree> `
  -LlamaBuild <cmake build tree>
```

At runtime the daemon exposes `POST /v1/audio/speech`; the worker is an internal
process and never opens a network port.
