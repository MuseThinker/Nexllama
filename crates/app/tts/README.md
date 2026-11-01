# llama-tts sidecar

`llama-tts.exe` is llama.cpp's text-to-speech frontend, built from the same
llama.cpp revision as the DLLs in each backend (`6fbe4b2`, build 25). It is a
thin executable over `llama.dll`, `llama-common.dll` and `mtmd.dll`, so it adds
about 50 KB to a backend rather than a second runtime.

The Rust crate builds llama.cpp with `LLAMA_BUILD_TOOLS=OFF`, so this binary is
not produced by `cargo build`. It is checked in here and copied next to
`infer-daemon.exe` by `scripts/build-daemon.ps1`. It must stay on the same
llama.cpp revision as the DLLs; if the backend is bumped, rebuild this file from
`tools/tts` in the matching source tree and update this note.

At runtime the daemon exposes it as `POST /v1/audio/speech`; the daemon passes
`--tts-bin` only when a different executable is wanted.
