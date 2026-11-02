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
include paths and import libraries as `llama-tts` (see `build.ninja` in the
llama.cpp build directory). For example, from a VS x64 prompt, with
`LLAMA_SRC` pointing at the llama.cpp source and `LLAMA_LIB` at the CMake
install tree:

```text
cl /nologo /MD /Brepro /W0 /utf-8 /O2 /DNDEBUG /std:c++17 /EHsc ^
  /DGGML_BACKEND_SHARED /DGGML_SHARED /DGGML_USE_CPU /DLLAMA_SHARED ^
  /DLLAMA_SUBPROCESS /D_CRT_SECURE_NO_WARNINGS ^
  /I%LLAMA_SRC%\include /I%LLAMA_SRC%\ggml\include /I%LLAMA_SRC%\common ^
  /I%LLAMA_SRC%\vendor /I%LLAMA_SRC%\tools\mtmd /I%LLAMA_LIB%\include ^
  llama-tts-worker.cpp /Fe:llama-tts-worker.exe /link /SUBSYSTEM:CONSOLE ^
  /INCREMENTAL:NO %LLAMA_LIB%\lib\llama-common.lib ^
  %LLAMA_LIB%\lib\mtmd.lib %LLAMA_LIB%\lib\llama.lib ^
  %LLAMA_LIB%\lib\ggml.lib %LLAMA_LIB%\lib\ggml-cpu.lib ^
  %LLAMA_LIB%\lib\ggml-base.lib kernel32.lib user32.lib gdi32.lib ^
  winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib ^
  comdlg32.lib advapi32.lib
```

At runtime the daemon exposes `POST /v1/audio/speech`; the worker is an internal
process and never opens a network port.
