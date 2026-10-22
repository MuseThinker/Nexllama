# Nexllama

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms: Windows | Linux](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-lightgrey.svg)]
[![Engine: llama.cpp](https://img.shields.io/badge/engine-llama.cpp-blue.svg)]
[![Docs: OpenAPI 3.1](https://img.shields.io/badge/docs-OpenAPI%203.1-green.svg)]
[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red.svg)](README.zh.md)

**Nexllama: all-in-one desktop app & inference API wrapping llama.cpp — auto
hardware fitting, speculative decoding, prefix cache, one local/cloud endpoint.**

## ✨ Nexllama

Nexllama is a hardware-aware all-in-one desktop inference and API gateway,
losslessly wrapping llama.cpp. It supports auto-download and one-click import for
GGUF models with a built-in model repository and OpenAI-compatible API. No manual
compilation or GPU layer tuning; hardware resources are auto-detected and
optimized. Native support for MTP, DFlash2, EAGLE-3, n-gram speculative decoding
and prefix caching, with an evaluator to toggle acceleration. Optimized for Agents,
it supports stream cancellation, native tool calling and model name parsing. Unify
local GGUF and cloud LLMs under one API endpoint. Optional routing falls back to
cloud when local hardware hits limits, disabled by default, vendor-agnostic while
preserving full llama.cpp performance.

## 🚀 Core highlights

| Feature | What it is |
|---|---|
| **All-in-One** | llama.cpp used as it is, behind a desktop interface instead of a command line, with the model library, the downloader and the API included. There is nothing else to install and no flags to learn. |
| **Adaptive hardware fitting** | Layer counts, context length and offload come from llama.cpp's own fitting helpers rather than from an estimate; the backend is chosen from the hardware actually present; a backend the machine needs but the package does not carry is downloaded, verified and installed from a signed manifest on demand. |
| **Blazing-fast inference** | Native MTP, DFlash2, EAGLE-3 and n-gram speculative decoding, behind an adaptive gate that switches drafting off the moment it stops paying. A prefix cache removes the repeated prefill that dominates time-to-first-token in a long conversation — the two take effect per model, never stacked. |
| **Built for agents** | An OpenAI-compatible endpoint that OpenClaw, Hermes and SDK clients drive unchanged, native tool calling, streaming that can be cancelled, model names that still resolve when slightly stale, and failures reported with the facts needed to act on them. |
| **Local and cloud, unified** | Local models and direct cloud vendors are served from the same port, and an optional `auto` policy keeps work on the machine and moves only what the machine cannot do well — a lower bill at the same quality, with no provider lock-in. |

> Repository description (for GitHub): *Nexllama is a hardware-aware all-in-one
> desktop inference and API gateway, losslessly wrapping llama.cpp. It supports
> auto-download and one-click import for GGUF models with a built-in model repository
> and OpenAI-compatible API. No manual compilation or GPU layer tuning; hardware
> resources are auto-detected and optimized. Native support for MTP, DFlash2,
> EAGLE-3, n-gram speculative decoding and prefix caching, with an evaluator to
> toggle acceleration. Optimized for Agents, it supports stream cancellation, native
> tool calling and model name parsing. Unify local GGUF and cloud LLMs under one API
> endpoint. Optional routing falls back to cloud when local hardware hits limits,
> disabled by default, vendor-agnostic while preserving full llama.cpp performance.*

---

## Installation

### Windows

Two installers, and the difference is one thing: whether NVIDIA's runtime comes with it.

| Installer | Size | Which machine |
|---|---|---|
| **Standard** `Nexllama_*_x64-setup.exe` | ~483 MB | **The one to download.** It carries a portable CPU build and NVIDIA's CUDA runtime (`cudart64`, `cublas64`, `cublasLt64`), so it needs nothing installed beforehand. A new machine, a laptop, or one you are not sure about belongs here: an NVIDIA card runs on the GPU and anything else falls back to the CPU. |
| **Lite** `Nexllama_*_x64-lite-setup.exe` | ~130 MB | Only for a machine that **already has the CUDA runtime** — usually a development box. It leaves out those three NVIDIA libraries, one of which (`cublasLt64_13.dll`) is 428 MB on its own. **A new computer, or one where CUDA was never installed, should use the standard installer**: without the runtime the CUDA backend cannot load, and the app falls back to the CPU — it works, but far more slowly. |

Both installers share a product name, install location and startup entry, so
installing one over the other replaces it instead of leaving two copies behind.

Run the installer and open Nexllama. The app picks a backend for the machine at first
launch and starts the daemon on `http://127.0.0.1:15656`.

The installer asks what it should leave behind on its options page — a desktop
shortcut and starting at login, both checked by default. Both are changeable later
in Settings, which reads the real state rather than its own memory of it.

Models are read from `C:\models` by default — change the location in Settings, or drop
a GGUF there before the first launch.

### Linux

Three installers come out of the same tree: `.deb` (Debian/Ubuntu), `.rpm`
(Fedora/openSUSE) and `.AppImage` (any distribution, nothing installed):

```bash
chmod +x Nexllama-*.AppImage && ./Nexllama-*.AppImage      # no root needed
sudo apt install ./Nexllama_*.deb                          # Debian/Ubuntu
sudo dnf install ./Nexllama-*.rpm                          # Fedora/openSUSE
```

They are built against Ubuntu 22.04 / Debian 12 libraries (glibc and
webkit2gtk 4.1), so anything from that generation or newer runs them.

Building them yourself:

```bash
./scripts/package.sh cpu vulkan    # Linux: .deb + .rpm + .AppImage
```

which needs the toolchain plus the packages the bundler stages and links
against:

```bash
sudo apt install build-essential cmake ninja-build pkg-config \
    libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev \
    libayatana-appindicator3-dev patchelf rpm zip
```

The daemon can also be run on its own, which is the usual shape for a server:

```bash
./infer-daemon --model /srv/models/model.gguf --port 15656 --n-gpu-layers 999
```

### Docker

GPU is the default. The image serves the UI, the API, and Hub searches/downloads on
one port, and `docker compose up -d` offloads all layers to the GPU:

```bash
docker compose up -d          # http://localhost:8090
```

The host needs the NVIDIA Container Toolkit. `docker-compose.yml` builds
`Dockerfile.web.cuda`. On a host without an NVIDIA GPU, use the CPU image instead:

```bash
docker compose -f docker-compose.cpu.yml up -d
```

That builds `Dockerfile.web.cpu`, which keeps every layer on the CPU. The default
compose file is still the GPU one.

Open `http://localhost:8090`; put a `.gguf` in `./models` (or set `MODELS_DIR`) and it
shows up in the library. Press **Load** to offload it, then chat. Search and download
on the Models page also write into `./models`, so the mount is writable. `docker compose down`
stops it.

By default the port has no token, which is right on a laptop. Anywhere else, set one —
the daemon reads `NEXLLAMA_API_KEY` itself, and the app asks for the token on its
Settings page:

```bash
API_KEY=$(openssl rand -hex 16) docker compose up -d
```

Models are read from and written to `./models`; on Linux hosts the mounted folder
must be writable by uid `10001` for Hub downloads. Environment overrides: `PORT`,
`MODELS_DIR`, `API_KEY`, `RUST_LOG`, `CUDA_VERSION`, and the build argument
`CRATES_MIRROR` for a crates.io mirror.

### From source

```powershell
npm --prefix frontend install
.\scripts\package.ps1                 # standard: cpu + cuda + NVIDIA runtime
.\scripts\package.ps1 -Edition lite   # lite: leaves the runtime out
.\scripts\package.ps1 -Edition both   # both, from one daemon build
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-setup.exe
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-lite-setup.exe
```

`-Native` bakes the build machine's instruction set in — faster CPU inference, but
the result only runs on that machine or one of the same generation. Fine for a
personal build, not for one you hand to someone else.

<details>
<summary>Build prerequisites</summary>

| Requirement | Needed for |
|---|---|
| Rust (stable), Node.js 22+ | everything |
| Visual Studio Build Tools (C++), LLVM (`clang`, `libclang`), CMake, Ninja | `llama-cpp-4` bindgen and the native build |
| CUDA Toolkit | the `cuda` backend |
| LunarG Vulkan SDK (`glslc`) | the `vulkan` backend |

`scripts/build.ps1` locates these components and configures `LIBCLANG_PATH`,
`CFLAGS=/utf-8`, `CMAKE_GENERATOR=Ninja` and the MSVC environment automatically.
</details>

---

## Quick start

1. **Select a models folder** — Settings → *Models folder* (`C:\models` by default).
2. **Load a model** — press **Load** in the library. A model loaded before a restart or
   an update is restored automatically; one whose file has moved is skipped rather
   than treated as a failure.
3. **Use it** — from the Chat panel, or from any client:

```bash
curl http://127.0.0.1:15656/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"alias","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:15656/v1", api_key="unused-on-loopback")
client.chat.completions.create(
    model="alias",
    messages=[{"role": "user", "content": "hello"}],
)
```

```bash
# Ollama-compatible clients work against the same port:
#   /api/tags   /api/chat   /api/generate   /api/embed
```

`http://127.0.0.1:15656/docs` serves an interactive API page that runs entirely
offline.

---

## Features

### One endpoint for local and cloud models

A provider is added from a built-in catalogue on *Models → Cloud models*: choose the
vendor, paste the key, fetch the model list, save. Its models then appear in
`/v1/models` beside the local ones as `<provider>/<model>`, and a request naming one is
proxied to that vendor.

- **Direct vendors only.** DeepSeek, OpenAI, Anthropic, Google, xAI, Mistral, Cohere,
  智谱 GLM, Kimi, 通义 Qwen and MiniMax, each with the China and international
  endpoint where the two differ. Token-plan variants (xAI CLI, MiniMax, 智谱 Coding)
  accept a token in place of an API key.
- **Streaming is forwarded unchanged**, terminator included; nothing is re-chunked in
  transit. HTTP/2 and a warm connection pool keep time-to-first-token close to the
  provider's own.
- **Cloud traffic is isolated from local inference.** The provider registry has its
  own lock, proxying is asynchronous, and a cloud request never touches the model
  worker thread or the GPU — a slow provider can only delay its own request.
- **Cloud requests leave the machine.** The configuration page states this, and cloud
  entries are marked in the model list so a client can tell them apart.

### The `auto` routing policy

Disabled by default. When enabled, `auto` appears in `/v1/models` and clients may send
`model: "auto"`:

| Situation | Result |
|---|---|
| Local-first (default), local model resident | The local model answers |
| Local-first, local model absent or fails to start | The request goes to the cloud |
| Cloud-first | The cloud model answers |
| Cloud-first, cloud unreachable | Falls back to the local model |
| The selected side failed **after** output began | The error is returned; no second answer is substituted |

Set it from the application or headlessly with
`POST /admin/routing {"enabled":true,"prefer":"local"}`.

### Speculative decoding

*Chat → Parameters → Speculative* lets a model draft its own continuation and verify
the entire draft in one batched forward pass. Three families of proposals are supported,
and the library shows a **drafter chooser** per bundle — `auto` prefers a side-mounted
DFlash/EAGLE checkpoint over the bundled MTP head, because a published draft file is
the drafter made for that exact model:

- **N-gram drafting** — proposes what followed the most recent repeat of the last
  `ngram_n` tokens. The learned table persists across requests and restarts.
- **MTP (NextN head)** — the multi-token-prediction head some models ship inside the
  same GGUF (`qwen35.nextn_predict_layers`), or published separately as
  `mtp-<model>-<quant>.gguf` and placed beside the model.
- **DFlash and EAGLE-3** — a second, smaller checkpoint trained to predict the target,
  configured with `draft_model` and `draft_n_max`.

An adaptive gate (`adaptive_draft`, enabled by default, acceptance floor 0.3) disables
drafting once it stops paying and re-enables it when the acceptance rate recovers. It
can only ever turn drafting *off*, which makes leaving it enabled the safe direction.
Accepted tokens are tokens the model would itself have sampled, so the optimisation is
exact rather than approximate.

### Measured configuration

`POST /admin/fit` accepts a GGUF path — or only a `size_bytes` for a file not yet
downloaded — and answers with `fits`, memory required against memory free, the
`n_gpu_layers` and `n_ctx` llama.cpp would select, a smaller quantisation when the
weights themselves are the problem, and a `reason` in plain language. The application
exposes **Check fit** and **Adopt** per model. The verdict is advice by default;
`strict_fit: true` refuses a load outright on request.

### Prefix cache and runtime figures

Because each turn contains the previous one as a prefix, re-encoding that prefix is
most of the latency before the first token. With `prefix_cache` enabled — the default —
the KV state is reused and only the new suffix is encoded. It is disabled automatically
when a draft model or MTP head is loaded, because those track position separately from
the target.

```bash
cargo run -p infer-core --example prefix_check -- models/your-model.gguf 0
```

`/admin/status` reports per-model `perf` — requests, tokens, smoothed
`tokens_per_second`, `first_token_ms` and `draft_acceptance` — so every claim above can
be verified rather than trusted.

### Models folder and multi-file models

Models are discovered by scanning the configured folder recursively for `*.gguf`
(`C:\models` by default). Downloads are written to the same location. The GGUF header is
read without loading the model, so the class, an MTP head and a size hint are known
before anything is loaded.

A model shipped as several files lives in its own subdirectory:

```
models/Qwen2.5-VL-7B/
  Qwen2.5-VL-7B-Q4_K_M.gguf        # main weights
  Qwen2.5-VL-7B.mmproj-f16.gguf    # vision projector
  Qwen2.5-VL-7B.draft.gguf         # draft model (speculative decoding)
```

Roles are identified from the file name (`mmproj`/`projector`,
`draft`/`eagle`/`mtp`/`dflash`, `lora`/`adapter`) and a bundle is loaded as a unit, so
a model is never left without its projector or its drafter. A directory containing
several unrelated models is listed file by file, since ownership cannot be inferred.

**Model classes** are read from the GGUF header rather than guessed from names:

| Class | Recognised by | Behaviour |
|---|---|---|
| `llm` | everything else | offered in the chat panel; vision bundles are included here |
| `embedding` | a known encoder architecture (`bert`, `nomic-bert`, `jina-bert-v2`, …) | API only — never offered for chat, refused by `/v1/chat/completions` |
| `rerank` | a `rerank.head_count` key with heads | API only |

A chat model is something one converses with to evaluate it; an encoder is something a
program points at. Neither is offered where the other belongs.

### Embeddings, reranking and vision

- **`/v1/embeddings`** — OpenAI response shape with unit-length vectors; raw pooled
  output would make a dot product depend on text length. The context is configured from
  the model's class, not from a flag. Input longer than the context is an error naming
  the limit, never a silent truncation.
- **`/v1/rerank`** — Cohere/Jina response shape. The input is built as
  `BOS query SEP document EOS`, the sentence pair the classification head was trained
  on; a NaN score sorts last, so a broken head cannot be reported as the best match.
- **Vision** — a bundle containing a projector accepts OpenAI-style `image_url` parts on
  `/v1/chat/completions`. Only `data:` URLs are accepted: an `http(s)` URL would turn
  every request into an SSRF primitive and a bare path would name files on the host.
  The projector runs on the GPU when the model is offloaded — roughly 200 ms against
  50 s on CPU for a 336×252 image on a 40 GB card.

### Reasoning control and tool calling

**Thinking** is set per model in the advanced panel and may be overridden per request
with `"thinking": true|false`. The switch reaches the chat template as
`enable_thinking`, so the template that understands it is the one that applies it.

**Tool calling** accepts an OpenAI-shaped `tools` array with optional `tool_choice` and
`parallel_tool_calls`, and answers as an ordinary OpenAI message:

```bash
curl http://127.0.0.1:15656/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.8-27B",
  "messages": [{"role":"user","content":"What is the weather in Paris?"}],
  "tools": [{"type":"function","function":{
    "name":"get_weather","description":"Get the current weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                  "required":["city"]}}}]
}'
```

```json
{"choices":[{"index":0,"finish_reason":"tool_calls","message":{
  "role":"assistant","content":"",
  "reasoning_content":"The user is asking about Paris…",
  "tool_calls":[{"type":"function",
    "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}
```

The template formats the call and the same template's parser reads it back, so whatever
call syntax a model family uses is understood without imposing a grammar on its
output. `content` contains only the answer, the chain of thought is separated into
`reasoning_content`, and raw call markup never reaches the caller. Streaming carries
the same structure: the final chunk before `[DONE]` carries `tool_calls`, each tagged
with its `index`, and `finish_reason: "tool_calls"`.

### Streaming and cancellation

`"stream": true` produces Server-Sent Events in the OpenAI shape — `id`, `object`,
`created` and `model` on every chunk, `data: [DONE]` at the end — so the official SDKs
work unchanged. Cancellation is honoured from both ends: closing the response body
stops generation at the next token, as does the application's stop button. An
abandoned reader would otherwise leave the model decoding the remainder of
`max_tokens` into a socket nobody is reading, and because each model owns one worker
thread, every queued request waits for it.

### Model name resolution

One machine can hold a chat model, an embedder and a reranker simultaneously, and each
route resolves against the kind of model it requires. A request naming a resident model
of that kind uses it; when the name matches nothing and exactly one model of that kind
is resident, that model answers, so a stale identifier does not break a client. Only
genuine ambiguity is refused, and the error names the candidates:

```json
{"error": "model \"typo\" not found and more than one llm model is loaded: llama3, qwen2.5"}
```

### The desktop application

- **Tray-resident.** Closing the window hides it; models stay loaded and connected
  clients stay attached. Quit from the tray menu.
- **Models return after a restart or an update.** The set loaded previously is restored
  automatically; a file that has moved is skipped.
- **Idle unload with transparent reload.** With `keep_alive_secs` set, a model is
  released after that many idle seconds and reloaded by the next request.
- **A per-model advanced panel.** GPU layers, context, batching, KV offload, RoPE
  scaling, load mode, memory locking, thinking, MTP budget and prefix cache, in one
  place.
- **GPU configuration without guesswork.** Settings reports the detected adapter, the
  backend that would be selected and the backend in use, and installs a missing backend
  on request.

---

## HTTP API

The daemon listens on `http://127.0.0.1:15656` and serves both an OpenAI-compatible and
an Ollama-compatible surface, so an integration written for either works after changing
only the port.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Banner and endpoint index (public; no model names) |
| GET | `/openapi.json`, `/docs` | OpenAPI 3.1 specification and an offline interactive page (public) |
| GET | `/v1`, `/v1/models` | OpenAI-style model list — local, cloud, and `auto` when enabled |
| GET | `/api/tags`, `/api/version` | Ollama-style model list and version |
| GET | `/health` | Liveness and the loaded model (public) |
| GET | `/admin/status` | Model information, classes, runtime `perf`, idle time |
| POST | `/admin/load` | `{ "model": "path.gguf", "name": "alias", "n_gpu_layers": 999, "n_ctx": 4096, … }` |
| POST | `/admin/unload` | `{ "model": "alias" }`, or `{}` for every model |
| POST | `/admin/fit` | Fit a checkpoint, or a `size_bytes`, before downloading |
| GET | `/admin/models` | The models folder, as `model-index` reads it |
| GET | `/admin/hub/authors` | Trusted HuggingFace publishers offered as search filters |
| GET | `/admin/hub/search` | `?q=…&author=…&limit=20`; list GGUF repositories |
| GET | `/admin/hub/files` | `?repo=owner/name`; list a repository's GGUF files |
| GET, POST | `/admin/hub/download` | `{ "repo": "owner/name", "filename": "x.gguf" }`; `GET` with the same pair reports progress |
| POST | `/admin/hub/download/pause`, `/cancel` | Stop a running transfer, keeping or deleting the partial file |
| GET | `/admin/hub/downloads` | Every transfer this daemon knows about |
| GET, POST | `/admin/providers` | List and configure cloud providers; keys are never returned |
| POST | `/admin/providers/test` | Retrieve a provider's model list to validate a key |
| GET, POST | `/admin/routing` | The `auto` policy: `{ "enabled": true, "prefer": "local" }` |
| POST | `/v1/chat/completions` | OpenAI chat; `"stream": true` for Server-Sent Events |
| POST | `/api/chat`, `/api/generate` | Ollama chat and single prompt; NDJSON when streaming |
| POST | `/v1/embeddings`, `/api/embed`, `/api/embeddings` | Embeddings in OpenAI and Ollama shapes |
| POST | `/v1/rerank`, `/rerank`, `/v1/reranking`, `/reranking` | `{ "model": "…", "query": "…", "documents": ["…"], "top_n": 3 }` |

Ollama's `options` object maps onto the sampler: `temperature`, `top_k`, `top_p`,
`min_p`, `repeat_penalty`, `repeat_last_n`, `num_predict` (generated-token ceiling;
values `<= 0` mean "until the model stops") and `seed`. Responses carry the customary
`prompt_eval_count` and `eval_count`, and each streamed line a `created_at`.

`GET /docs` renders the specification as an interactive page with no external scripts
or styles: a token field retained in that browser, a **Send** control per operation
seeded from the specification's example, streaming routes consumed chunk by chunk, and
examples that name a model which is actually loaded. Tests walk the specification and
fail if a documented route is absent, if a schema reference is undefined, or if the
`/docs` flow regresses in a real browser.

### Configuration

Defaults are the recommended values; every option is also available in Settings and in
the per-model **Advanced** panel. `POST /admin/load` accepts:

| Group | Parameters |
|---|---|
| Model | `n_gpu_layers` (999 = all), `main_gpu`, `load_mode` (`auto｜none｜mmap｜mlock｜mmap_mlock｜dio`), `use_mlock` |
| Context | `n_ctx` (0 = automatic, sized to the machine), `n_batch` (2048), `n_ubatch` (512), `flash_attn` (`auto｜on｜off`), `offload_kqv`, `rope_freq_base`, `rope_freq_scale` |
| CPU | `threads`, `threads_batch` (0 = automatic) |
| Adapters | `load_mtp` (bundled MTP head), `lora` (list of adapter paths) |
| Sampling (per request) | `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `repeat_freq`, `repeat_present`, `seed`, `max_tokens` |
| Speculative (per request) | `speculative`, `ngram_n`, `ngram_m` |

The daemon exposes the same options as command-line flags (`--load-mode`,
`--flash-attn`, `--n-batch`, `--no-offload-kqv`, …) for headless and container use.

Two flags exist to serve the app itself rather than an API in front of something
else, which is what the web container does:

| Flag | Effect |
|---|---|
| `--ui-dir <dir>` | Serve the built UI from `<dir>`: `/` answers the app, its assets are served from the paths they were built at, and the API is merged on the same port. Without it, `/` is the banner that says what the server is. |
| `--models-dir <dir>` | The folder `GET /admin/models` lists, read with the same rules the desktop library uses (`model-index`), so a library opened in a browser and one opened in the app cannot disagree about what a model is. Unset answers `{"dir": null, "models": []}` rather than a 404. |

`--ui-dir` and `--models-dir` are why the web image needs no nginx: one process, one
port, and the browser talks to the same origin that serves the app.

### Network exposure and security

The daemon binds `127.0.0.1` by default, so nothing beyond the machine can reach it.
Settings → *Network* can widen the binding to `0.0.0.0`, which requires a bearer token:

```powershell
infer-daemon.exe --host 0.0.0.0 --port 15656 --api-key <token> --model model.gguf
```

The server has no accounts, no rate limiting and no quota; anyone who can reach the
port can load models onto the machine's GPU and consume the electricity. The token is
compared in constant time and required on every route except `/health`, which remains
public for readiness probes and reveals only the loaded model name. Binding to a
routable address **without** a token is refused:

```
refusing to listen on 0.0.0.0 without --api-key: anyone who can reach this port
can load models and spend GPU time on them. Pass --api-key <token>, bind to
127.0.0.1, or pass --allow-unauthenticated if something else guards the port
```

`--allow-unauthenticated` exists for a container or reverse proxy that authenticates
callers itself; an empty `--api-key ""` is an error rather than a quiet way to disable
authentication. The application stores its token in the operating system's credential
store and passes it to the daemon through `NEXLLAMA_API_KEY`, so it appears neither
beside the settings file nor in a process command line. Cross-origin requests are
answered only for the application's own origins; others are added with
`--cors-origin <origin>`.

Image inputs are restricted to `data:` URLs, and the application's capability manifest
grants nothing beyond `opener:default`.

---

## Model downloads

Search is unrestricted; the download page offers the three catalogues that cover nearly
everything worth running locally as one-click filters. The desktop app and the Docker
web UI both use it; in the container the transfer runs in the daemon, so closing the
browser does not cancel it.

- **bartowski** — the broadest catalogue, one model across many quantisations
- **Unsloth** — dynamic quantisations and the releases that carry the heads MTP needs
- **z-lab** — the DFlash draft models (UCSD's Z-Lab, which proposed and leads the method)

Quantisation is a craft, and a poor one costs accuracy that no amount of context
recovers; everything else on the Hub remains available. Downloads are resumable and are
checked against free memory before the transfer begins. The client defaults to
`https://hf-mirror.com`; override with `HF_ENDPOINT`, and authenticate gated
repositories with `HF_TOKEN`.

## Backend updates

Settings → *Backend updates* accepts a manifest URL. The app checks the manifest,
downloads the matching backend archive, verifies its SHA-256, replaces
`%APPDATA%/com.nexllama.desktop/backends/<name>/` and restarts the daemon. An updated
backend takes precedence over the bundled one.

A manifest must be **signed**: an Ed25519 signature over its version, llama.cpp build
and each backend's URL and hash. A checksum alone only proves the archive matches the
manifest that named it, and both arrive from the same request, so hash verification
alone would let anyone who can serve one serve the other. The public key is taken from
Settings or from `NEXLLAMA_UPDATE_KEY`; no key is compiled in, because an updater that
trusts any key verifies nothing. Fetches are HTTPS-only including after redirects, a
version older than the installed one is refused, and manifest, archive and unpacked
totals are size-capped so a malformed release cannot exhaust the disk.

```powershell
cargo run -p nexllama-app --example new-update-key     # public and private key, hex
cargo run -p nexllama-app --example sign-manifest -- manifest.json <private-hex>
.\scripts\make-backend-release.ps1 -Version 0.2.0 -LlamaCpp b10502 `
  -BaseUrl https://github.com/<owner>/<repo>/releases/download/v0.2.0
```

---

## Architecture

```
frontend/          Svelte 5 + Vite interface (chat, models, downloads, settings)
crates/
  infer-core/      safe wrapper over llama.cpp (llama-cpp-4): load, chat, streaming
  infer-daemon/    axum HTTP service: /health, /admin/*, /v1/* (SSE), cloud proxy
  infer-hub/       HuggingFace client: search, file listing, resumable download
  gguf-probe/      GGUF header reader: model class, MTP head detection, size hints
  app/             Tauri 2 desktop shell
                   · probes the machine and selects a backend
                   · bundles each backend under backends/<name>/
                   · stores cloud keys in the OS credential store
```

The interface drives the daemon over local HTTP. `infer-core` executes llama.cpp on a
dedicated worker thread per model and streams decoded UTF-8; cloud traffic is proxied
asynchronously and never enters that thread.

---

## Troubleshooting

**The installer is flagged by SmartScreen.** Builds are not signed by default.

**The wrong backend was selected.** Settings → GPU reports the detected adapter, the
backend that would be used and the one in use, and can install a missing backend on
demand. The selection can be overridden with Auto / CUDA / Vulkan / CPU; a machine with
no NVIDIA adapter falls back to CPU rather than starting a CUDA build that cannot work.

**A model will not load.** Run **Check fit** on it: it reports memory required against
memory free and suggests a layer count, a context length or a smaller quantisation.
`/admin/status` carries the refusal reason.

**A model has disappeared from the library.** It lies outside the configured folder.
Settings → *Models folder* shows the path in use, and *Open folder* opens it.

**`model not found` after moving files.** Point the folder at the new location. A
remembered model whose file has moved is skipped at start-up rather than failing it,
and the library lists whatever is actually present.

**Network restrictions or a slow crates.io.** Configure a mirror in
`~/.cargo/config.toml`:

```toml
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
```

---

## Current notes

- With a draft model or MTP head active, the prefix cache is disabled because those
  paths track position independently of the target; such models re-encode the
  conversation each turn.
- Hardware acceleration is provided by the backends llama.cpp offers — CUDA and
  Vulkan. Vendor-specific runtimes such as CANN or MUSA are not currently supported.
- Cloud requests transmit their content to the named provider.

## Contributing

Issues and pull requests are welcome. The same scripts used for packaging build and
test the workspace:

```powershell
.\scripts\build.ps1 fmt --all
.\scripts\build.ps1 clippy --all-targets
.\scripts\build.ps1 test --workspace
```

Contributions are licensed under the MIT License.

## Community

- 🐛 [Issues](issues) — defects and feature requests
- 💡 [Discussions](discussions) — questions, design proposals and demonstrations

## License

Licensed under the MIT License ([LICENSE](LICENSE)).

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work shall be licensed as above, without any additional terms
or conditions. Third-party components — llama.cpp and GGML among them — are listed in
[NOTICE](NOTICE).
