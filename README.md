# Nexllama

A lean, self-contained local LLM desktop app: download a GGUF model, it loads
automatically, and it runs at llama.cpp speed because it *is* llama.cpp.

- **No manual compile for users.** You ship prebuilt binaries; users just install.
- **Auto-loading.** A resident daemon keeps the model in RAM/VRAM for instant opens.
- **Downloader built in.** HuggingFace search + resumable download.
- **Fast.** The native build uses host CPU features (AVX2/AVX-512); GPU via CUDA/Vulkan.

## Architecture

```
frontend/  Svelte 5 + Vite UI (chat, models, download, settings)
crates/
  infer-core/   safe wrapper over llama.cpp (llama-cpp-4): load / chat / streaming
  infer-daemon/ axum HTTP daemon: /health, /admin/*, /v1/chat/completions (SSE)
  infer-hub/    HuggingFace client: search, list GGUF, resumable download
  app/          Tauri 2 desktop shell (own workspace)
                - probes the machine, picks the right backend automatically
                - bundles multiple backends under backends/<name>/
```

The UI talks to the daemon over local HTTP (default `http://127.0.0.1:15656`).
`infer-core` runs llama.cpp on a dedicated worker thread and streams decoded UTF-8.

## One-click installer with GPU auto-detection

A single installer can ship several backends (`cpu`, `cuda`, `vulkan`, `metal`). At
launch the app probes the machine and starts the matching daemon, so users don't choose
anything:

1. NVIDIA GPU (`nvidia-smi`, WMI) → `cuda`
2. Apple GPU (`system_profiler`) → `metal` — Metal reaches Apple's GPUs directly,
   where Vulkan would be MoltenVK translating on top of it
3. other GPU (AMD/Intel, from `lspci`) → `vulkan`
4. otherwise → `cpu`

Each backend is a self-contained folder `backends/<name>/{infer-daemon.exe, *.dll}`, so
different backends never clash over DLL names. Users can override the choice in Settings
(Auto / CUDA / Vulkan / CPU); the resolved backend is shown there.

Build the installer with the backends you can compile:

```powershell
.\scripts\package.ps1 -Backends cpu,cuda
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-setup.exe
```

Building a backend requires its toolchain: `cuda` needs the CUDA Toolkit, `vulkan` needs
the LunarG Vulkan SDK (`glslc`); `cpu` needs nothing extra.


## Prerequisites (Windows)

- Rust (stable) via [rustup](https://rustup.rs)
- LLVM (`clang`, `libclang` for bindgen)
- CMake + Ninja
- Visual Studio Build Tools with "Desktop development with C++"
- Node.js 22+ (frontend / Tauri CLI)
- CUDA Toolkit (only for the `cuda` build)

`scripts/build.ps1` locates all of the above and sets up the environment
(`LIBCLANG_PATH`, `CFLAGS=/utf-8`, `CMAKE_GENERATOR=Ninja`, vcvars) automatically.

## Build & run

Builds go through `scripts/build.ps1`, which wraps `cargo` with the right env.
Features: `native` (host CPU), `vulkan`, `cuda`.

```powershell
# daemon
.\scripts\build.ps1 build -p infer-daemon --features native
.\scripts\build.ps1 run   -p infer-daemon -- --model models\model.gguf --port 15656 --n-gpu-layers 0

# frontend (dev server on http://127.0.0.1:5173)
npm --prefix frontend install
npm --prefix frontend run dev

# benchmark (load time, TTFT, tok/s)
.\scripts\build.ps1 run -p infer-core --features native --example bench -- models\model.gguf 0
```

### Desktop installer

```powershell
npm --prefix frontend install                     # once, provides the Tauri CLI
.\scripts\package.ps1 -Backend vulkan             # builds daemon (release+native) + NSIS installer
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-setup.exe
```

| Platform | What you get | How to build |
|---|---|---|
| Windows | NSIS `.exe` | `.\scripts\package.ps1 -Backends cpu,cuda` |
| macOS | `.app` and `.dmg` | `./scripts/package.sh metal cpu` |
| Linux | `.deb` and `.AppImage` | `./scripts/package.sh cpu vulkan` |

`scripts/build-daemon.ps1 -Backend <cpu|vulkan|cuda> [-Native] [-Release]` stages the
sidecar plus its runtime libraries into `crates/app/backends/<name>/` for the
desktop bundle; `scripts/build-daemon.sh` is the same thing for macOS and Linux,
where the runtime libraries are `.dylib` and `.so` instead of `.dll`.

The macOS and Linux scripts need `cmake`, `ninja` and a C++ toolchain. Linux
additionally needs `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `librsvg2-dev` and
`libayatana-appindicator3-dev`.

**Nothing is signed by default.** Windows shows a SmartScreen prompt; macOS
refuses to open an unsigned app until the user right-clicks it once. Signing and
notarising a macOS build needs an Apple Developer account and these secrets,
which the packaging step picks up when they are present:

```bash
APPLE_SIGNING_IDENTITY   # e.g. "Developer ID Application: …"
APPLE_CERTIFICATE        # base64 .p12
APPLE_CERTIFICATE_PASSWORD
APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID   # for notarisation
```

## HTTP API (daemon)

The daemon listens on **`http://127.0.0.1:15656`**. `GET /` answers with a banner
and an endpoint index; `GET /v1` and `GET /api/tags` list the loaded models, so
pointing a client at the base URL is enough to see what is there. Both an
OpenAI-compatible and an Ollama-compatible surface are served, so an example
written for either works after changing only the port.

| Method | Path | Description |
|---|---|---|
| GET | `/` | banner + endpoint index (open, no model names) |
| GET | `/openapi.json`, `/docs` | OpenAPI 3.1 spec and a browsable page (open) |
| GET | `/v1`, `/v1/models` | OpenAI-style model list |
| GET | `/api/tags` | Ollama-style model list |
| GET | `/api/version` | Ollama-style version |
| GET | `/health` | liveness + loaded model (open) |
| GET | `/admin/status` | model info, idle time |
| POST | `/admin/load` | `{ "model": "path.gguf", "name": "alias", "n_gpu_layers": 999, "n_ctx": 4096, ... }` |
| POST | `/admin/unload` | `{ "model": "alias" }` or `{}` to unload all |
| POST | `/v1/chat/completions` | OpenAI chat; `"stream": true` for SSE |
| POST | `/api/chat`, `/api/generate` | Ollama chat / single prompt; `"stream": true` for NDJSON |
| POST | `/v1/embeddings` | `{ "model": "bge-small", "input": "text" }` or `"input": ["a", "b"]` |
| POST | `/api/embed`, `/api/embeddings` | Ollama embeddings (list / legacy single) |
| POST | `/v1/rerank` (`/rerank`) | `{ "model": "bge-reranker", "query": "…", "documents": ["…"], "top_n": 3 }` |

Ollama's `options` map onto the sampler: `temperature`, `top_k`, `top_p`,
`min_p`, `repeat_penalty`, `repeat_last_n`, `num_predict` (generated-token cap;
`<= 0` means "until it stops"), and `seed`. Replies carry the usual
`prompt_eval_count` / `eval_count`, and each streamed line a `created_at`.

### OpenAPI

`GET /openapi.json` serves an OpenAPI 3.1 specification covering every route on
both surfaces, so editors, Postman and client generators can read the API
directly. Request bodies carry `example` payloads, which those tools pick up too.

`GET /docs` renders that specification as a browsable, **interactive** page with
no external scripts or styles — it works on a machine with no internet access:

- a token field (kept in that browser) so protected routes can be called once a
  key is set;
- a **Send** button per operation, an editable body seeded from the spec's
  example, and a response pane showing status and payload;
- streaming routes (SSE and NDJSON) are read chunk by chunk, so you watch tokens
  arrive rather than waiting for the reply to finish;
- **examples name a model that is actually loaded**, picked per route class
  (chat → llm, embeddings → embedding, rerank → rerank). The spec's placeholders
  would otherwise answer 400 out of the box.

Both pages are open, like the banner, so a client can bootstrap before it has the
API key. A test walks the specification and fails if any documented route does not
actually exist, another fails on a `$ref` to a schema that is not defined, and the
`/docs` flow — open the page, send an operation, read the reply — is exercised in
a real browser over the DevTools protocol.

`keep_alive_secs = 0` keeps models resident (instant opens); a positive value
unloads them after that many idle seconds.

**Every API route requires `model` explicitly.** One machine can hold a chat
model, an embedder and a reranker at once, and guessing which one a request meant
turns a typo into the wrong model answering. Omit it and the error lists what is
loaded:

```json
{"error": "\"model\" is required; loaded: bge-reranker, bge-small, qwen2.5"}
```

The chat panel is the exception: it defaults to the model you last used there,
falling back to the first-loaded generation model, and remembers your choice.

### Network access

The daemon binds `127.0.0.1` by default, so nothing but this machine can reach
it. Settings → **Network** can widen that to `0.0.0.0`, which accepts
connections from every interface:

```powershell
infer-daemon.exe --host 0.0.0.0 --port 15656 --api-key <token> --model model.gguf
```

**You have to set an access token when you do.** The server has no accounts, no
rate limit and no quota — anyone who can reach the port can load models onto
your GPU and burn your electricity. The token is a bearer value, compared in
constant time, and required on every route except `/health` (left open for
readiness probes; it reveals loaded model names and nothing else).

Binding to a routable address without one is **refused**, not warned about:

```
refusing to listen on 0.0.0.0 without --api-key: anyone who can reach this port
can load models and spend GPU time on them. Pass --api-key <token>, bind to
127.0.0.1, or pass --allow-unauthenticated if something else guards the port
```

`--allow-unauthenticated` is the escape hatch for a container or a reverse
proxy that authenticates callers itself. An empty `--api-key ""` is an error
rather than a silent way to switch authentication off.

The desktop app keeps the token in the OS credential store — Credential
Manager, the keychain, the Secret Service — and passes it to the daemon through
`NEXLLAMA_API_KEY`, so it is neither written next to the settings file nor
visible in the process's command line. The same variable works for a daemon you
start yourself.

Cross-origin requests are answered only for this app's own origins (the WebView
it serves and the Vite dev server); add others with `--cors-origin <origin>`.
The point is that opening a web page should not hand that page a way to drive
the daemon.

From another machine:

```bash
curl http://192.168.1.20:15656/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"alias","messages":[{"role":"user","content":"hi"}]}'
```

The desktop app fills in the token from Settings automatically, so it keeps
working after you set one.

## Parameters

Defaults are the recommended values; everything can be tuned in Settings and in the
per-model **Advanced** panel on the Models page. Loading accepts:

- Model: `n_gpu_layers` (999 = all), `main_gpu`, `load_mode` (`auto|none|mmap|mlock|mmap_mlock|dio`), `use_mlock`
- Context: `n_ctx` (0 = model default), `n_batch` (2048), `n_ubatch` (512), `flash_attn` (`auto|on|off`), `offload_kqv`, `rope_freq_base`, `rope_freq_scale`
- CPU: `threads`, `threads_batch` (0 = auto)
- Adapters: `load_mtp` (bundled MTP head), `lora` (list of adapter paths)
- Sampling (per request): `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `repeat_freq`, `repeat_present`, `seed`, `max_tokens`
- Speculative (per request): `speculative`, `ngram_n`, `ngram_m`

The daemon exposes the same options as CLI flags (`--load-mode`, `--flash-attn`,
`--n-batch`, `--no-offload-kqv`, ...) for headless/Docker use.

## Models folder

Models live in a folder scanned recursively for `*.gguf`. The path defaults to
`%APPDATA%/com.nexllama.desktop/models` (app data dir) and is chosen in Settings with a
native folder picker (**Select folder** / **Open folder**). The library lists every GGUF
it finds; click **Load** to bring one up, or tick one or more under **Auto-load at start**
to keep them resident from launch. Downloads land in the same folder.

### Multi-file models

A model that ships several files lives in its own subfolder:

```
models/Qwen2.5-VL-7B/
  Qwen2.5-VL-7B-Q4_K_M.gguf        # main
  Qwen2.5-VL-7B.mmproj-f16.gguf    # projector
  Qwen2.5-VL-7B.draft.gguf         # draft (speculative)
```

Roles are recognized from the filename (`mmproj`/`projector`, `draft`/`eagle`/`mtp`/`dflash`,
`lora`/`adapter`) and the bundle is loaded together. Wired today: **LoRA adapters** and the
**MTP flag** (`load_mtp`, same-file NextN head). Vision projectors and draft-model speculative
decoding are accepted but not yet used for generation — the load response returns a `notes`
field saying so rather than silently ignoring them.

### Model classes

Every GGUF is classified when it is scanned, from its own header — no name
guessing, no extension heuristics:

| class | how it is recognised | what the app does with it |
|---|---|---|
| `llm` | everything else | the chat panel's dropdown — for testing and tuning; vision models land here too |
| `embedding` | a known encoder architecture (`bert`, `nomic-bert`, `jina-bert-v2`, …) | **API only** — library badge, never in the chat dropdown, refused by `/v1/chat/completions` |
| `rerank` | a `rerank.head_count` key with heads | **API only**, same as embedding |

The split is the point: a chat model is something you test and tune by talking to
it, and an encoder is something you point a program at. Neither appears where the
other belongs, and the composer says so when the only models loaded are API ones.
| vision | a projector file in the bundle | chat, and it can accept images |

The class travels in `/admin/status` (`info.model_class`) and is enforced in the
daemon too, so an API client pointing at an embedding model gets

```json
{"error": "bge-small-en-v1.5-q8_0 is a embedding model — it cannot generate text; load a chat model (class llm)"}
```

instead of three hundred tokens of word salad. Classification covers what the
header can honestly express: an embedding model that reuses a generation
architecture (gte is `qwen2` under the hood) still classifies as `llm`, which is
the safe direction here — it only affects what the UI offers, and generation is
the only thing this app actually runs.

## Embeddings

An encoder model loaded from the library serves `/v1/embeddings` with the
OpenAI shape, so anything already pointed at that API works:

```bash
curl http://127.0.0.1:15656/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "bge-small", "input": ["a cat is sleeping", "a kitten naps", "stocks fell"]}'
```

```json
{"object":"list","model":"bge-small",
 "data":[{"object":"embedding","index":0,"embedding":[...384 floats...]}, ...],
 "usage":{"prompt_tokens":18,"total_tokens":18}}
```

- **The context is built from the class**, not from a flag: an encoder gets
  embeddings enabled, a ranker additionally gets the `rank` pooling type (without
  it the head never reaches the graph and every score would be identical), and a
  chat model is left alone. `info.pooling` in `/admin/status` reports what the
  context ended up with — bge-small comes back `cls`, read from its own metadata
  rather than overridden, because forcing mean pooling there would quietly
  degrade the vectors.
- **Chat and embeddings refuse each other**, each naming what to load instead.
  An encoder cannot generate, and a chat model loaded here has embeddings
  disabled.
- **Texts are embedded one forward pass at a time.** Parallel sequences would
  need `n_seq_max` and a per-sequence batch layout for no gain at these request
  sizes.
- **Vectors are returned at unit length**, as the OpenAI-shaped API and
  llama.cpp's own server do. Raw pooled output would make a dot product depend
  on the length of the text, so a client comparing two documents by cosine
  similarity would get a different answer than it expects.
- A text longer than the context is an error naming the limit, not a silent
  truncation — encoder contexts are small (512 for bge-small) and quietly
  cutting a document in half would give a confidently wrong vector.

Reranking is classified (`rerank` in `/admin/status`) but has no endpoint yet:
the cross-encoder input format depends on a `rerank` chat template the model may
or may not carry, and it has not been verified against a real reranker here.

## Reranking

A reranker serves Cohere/Jina-shaped `/v1/rerank` (also `/rerank`,
`/v1/reranking`, `/reranking`, matching upstream llama.cpp):

```bash
curl http://127.0.0.1:15656/v1/rerank -H "Content-Type: application/json" \
  -d '{"model": "bge-reranker", "query": "What is the capital of France?",
       "documents": ["Berlin is the capital of Germany.",
                     "Paris is the capital of France.",
                     "Bananas are a good source of potassium."]}'
```

```json
{"model":"bge-reranker",
 "results":[{"index":1,"relevance_score":7.3092,"document":"Paris is the capital of France."},
            {"index":0,"relevance_score":-5.7594,"document":"Berlin is the capital of Germany."},
            {"index":2,"relevance_score":-11.0402,"document":"Bananas are a good source of potassium."}],
 "usage":{"total_tokens":64}}
```

Those are real numbers from bge-reranker-v2-m3: the right document leads by a wide
margin, and the hard negative — a capital city, wrong country — lands clearly
below it. `top_n` trims the list; ties keep submission order; a NaN score sorts
last instead of first, so a broken head cannot be reported as the best match.

- **The context gets `rank` pooling**, without which the head never reaches the
  graph and every score would be identical. `info.pooling` reports `rank`.
- **The input is `BOS query SEP document EOS`** — for a BERT reranker that is
  `[CLS] query [SEP] document [SEP]`, the sentence pair the head was trained on.
  A model that ships its own `rerank` chat template needs a different format this
  build does not apply yet, so it is refused rather than scored wrongly.
- **The rank output is read straight from the context**, not through
  `LlamaContext::embeddings_seq_ith`, which slices `n_embd` floats: right for a
  pooled embedding, but a rank head emits `n_cls_out` (one float for
  bge-reranker), so that call would read hundreds of floats past the buffer.

### Telling a reranker from its embedding sibling

bge-reranker-v2-m3 and bge-m3 share an architecture, a name (`Bge M3`) and every
KV key — llama.cpp itself needs the caller to declare `--pooling rank` for them.
The only thing that differs is the classification head, so the classifier reads
the tensor names and looks for llama.cpp's `cls.*` head. `gguf-probe` ships a
diagnostic for exactly this:

```powershell
cargo run -p gguf-probe --example probe -- model.gguf --keys
```

## Vision

Load a model with `--mmproj <projector.gguf>` (or click **Load** on a bundle
that contains one) and the same `/v1/chat/completions` endpoint accepts
OpenAI-style image parts:

```bash
curl http://127.0.0.1:15656/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "SmolVLM-256M-Instruct-Q8_0",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What color is the bar at the bottom?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}]
  }'
```

Details worth knowing:

- **Only `data:` URLs are accepted.** An `http(s)` URL would make every request
  an SSRF primitive against whatever the host can reach, and a bare path would
  name files on this machine; both are refused. `image_url` and `input_image`
  are the recognised part types, `text` the rest.
- **The projector runs on the GPU** when the model is offloaded — a 336×252
  image takes ~200 ms on a 40 GB card versus ~50 s on CPU, so this is where the
  difference actually lives.
- **N-gram drafting is disabled for image prompts.** Image positions have no
  token ids, so the drafters would work from a history that does not match what
  is in the KV cache; drafting is a performance optimisation and verification
  would keep it correct either way, but there is nothing honest to fast-forward.
- The composer's paperclip accepts images up to 4 MB and previews them inline;
  a conversation re-sends its images with every turn, which is why the limit
  exists.
- `info.vision` in `/admin/status` says whether a model can see.

## Speculative decoding

Chat → **Params** → *Speculative* lets the model draft its own continuation and
then check the whole guess in one batched forward pass:

1. `ngram_simple_draft` finds the most recent repeat of the last `ngram_n`
   tokens and proposes the `ngram_m` tokens that followed it last time.
2. The target model decodes the anchor token and the whole draft in a **single**
   batch, which yields `draft + 1` logit rows.
3. Verification walks those rows one at a time, keeps the draft prefix the model
   agrees with, and samples one replacement token at the first point of
   disagreement.
4. The rejected tail is dropped from the KV cache, because llama.cpp requires
   every batch to begin exactly one position past the cache's last one.

Two drafters feed it, and their proposals are unioned: the stateless one finds
repeats inside the current conversation, and a small statistical model learned
from earlier replies proposes what usually followed the same n-grams. The
learned table lives in `%APPDATA%/com.nexllama.desktop/ngram-cache.json`
(`--ngram-cache <path>`), so drafts keep working across requests and restarts.
It stops learning past 65536 distinct contexts, which bounds a long-running
server; drafts from what it already knows keep working.

Verification is written out by hand rather than calling
`common_sampler_sample_and_accept_n`, because the Rust binding reaches that
function through a size-then-fill wrapper that runs the whole check **twice**.
The first pass feeds every token it verifies back into the sampler, so the
second pass can disagree and need more output space than the first one sized —
an opaque failure partway through a reply. Sampling row by row has no buffer to
size and does not double-count the sampler's history.

Accepted tokens are tokens the model itself would have sampled, so this is an
exact optimisation — not an approximation. Two practical notes:

- **It only pays off on repetitive text.** Prose usually has no repeat to copy,
  the drafter returns nothing, and the round degenerates to the normal
  one-token-per-pass path. Repetitive output (code, logs, JSON, lists) is where
  it wins — asking the same question twice in a row hits 100% acceptance. A
  round that gets nothing accepted pauses drafting for one step so a bad guess
  costs one wasted pass rather than two.
- **Text is not bit-identical to a non-speculative run.** Verifying a draft
  means computing a batch instead of a single token, and floating-point sums in
  a different order, so a near-tie can resolve the other way. Both paths are
  individually deterministic; with drafting effectively disabled the two produce
  byte-identical output.

Every reply reports the hit rate, e.g. `draft 6/32 (19%)`.

### MTP (built-in NextN head)

Some models ship a multi-token-prediction head inside the same GGUF — Qwen3.6's
`qwen35moe.nextn_predict_layers`, for instance. llama.cpp will only load that head
when asked, and it is what the reference command uses to go fast:

```bash
llama-server -m model.gguf -ngl 99 --spec-type draft-mtp --spec-draft-n-max 3
```

The app detects the head from the GGUF (`nextn_predict_layers`, or `nextn` tensor
names) and reports it as `info.mtp` in `/admin/status`, with a badge in the
library. **Loading it is a separate step**, and `load_mtp` says whether it
happened — the honest state today is usually `mtp=true, load_mtp=false`.

That is deliberate for now. `--load-mtp` only loads the head; the speedup comes
from *drafting* with it and verifying against the model, and that loop is not
implemented yet. Enabling the flag without the loop would spend the head's memory
for no gain, so the app neither auto-enables it nor hides the fact that the model
has it. `mtp=true` is the signal that the draft loop would pay off.

Note also that the n-gram drafter above is refused on models whose KV cache cannot
drop rejected draft tokens (Qwen3.6's hybrid cache is one), so speculative
decoding is off there until the MTP path exists — the daemon says so instead of
returning wrong tokens.

An abandoned stream is abandoned properly: when the reader goes away the engine
stops at its next emitted token instead of finishing `max_tokens` into a socket
nobody is reading. Cancelling a 600-token reply used to block the next request
for the remainder of the generation; now the following request starts within a
second.

This is self-speculation, so it needs no draft model. A bundled MTP head is
loaded via `load_mtp`, and `draft`/`eagle` models are recognised but not yet
wired into generation.

## Backend updates

The llama.cpp backends are versioned separately from the app, so an engine update does not
require reinstalling. Point Settings → **Backend updates** at a manifest URL; the app checks
it, downloads the matching backend zip, verifies its SHA-256, swaps
`%APPDATA%/com.nexllama.desktop/backends/<name>/`, and restarts the daemon. Updated backends
take precedence over the ones bundled in the installer.

**A manifest has to be signed.** A checksum only proves the archive matches the
manifest that named it, and both arrive from the same request — so hash alone
would let anyone who can serve one serve the other. The app therefore:

- requires an **Ed25519 signature** over the manifest's version, llama.cpp build
  and every backend's URL and hash;
- reads the public key from Settings → **Backend updates**, or from
  `NEXLLAMA_UPDATE_KEY` at build time — there is no key compiled in by default,
  because an updater that trusts any key is an updater that checks none;
- fetches the manifest and the archive over **HTTPS only**, including after
  redirects;
- refuses a version that is **older** than what is installed.

Make a keypair and sign a manifest with:

```powershell
cargo run -p nexllama-app --example new-update-key     # public + private hex
cargo run -p nexllama-app --example sign-manifest -- manifest.json <private-hex>
```

The connection is also capped in size (manifest, archive and unpacked total) so
a bad release cannot fill the disk.

Produce the artifacts with:

```powershell
.\scripts\make-backend-release.ps1 -Version 0.2.0 -LlamaCpp b10502 `
  -BaseUrl https://github.com/<owner>/<repo>/releases/download/v0.2.0
```

then sign `manifest.json` and upload it with the `*.zip` files to that release.


## Docker

```bash
docker compose build
MODEL=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf docker compose up   # CPU
docker compose -f docker-compose.cuda.yml up                       # CUDA (needs nvidia-container-toolkit)
```

Models are mounted read-only from `./models`. Env overrides: `PORT`, `MODEL`,
`N_GPU_LAYERS`, `N_CTX`, `KEEP_ALIVE_SECS`, `MODELS_DIR`, and build arg
`CRATES_MIRROR` for a crates.io mirror.

The image binds `0.0.0.0` so the host can reach it, and passes
`--allow-unauthenticated` because nothing in front of it authenticates callers.
That is fine for `docker compose up` on a laptop; put `--api-key <token>` (or a
reverse proxy) in front before exposing the port to a network. Both images run
as an unprivileged user and install the llama.cpp shared libraries into
`/usr/local/lib`.

## Streaming

`POST /v1/chat/completions` with `"stream": true` answers Server-Sent Events in
the OpenAI shape — `id`, `object`, `created`, `model` on every chunk, and
`data: [DONE]` at the end — so the official SDKs work unchanged.

Cancelling works from either end. Closing the response body stops generation at
the next token, and the Chat panel's stop button does the same. This matters
because of what happened otherwise: an abandoned reader left the model decoding
the rest of `max_tokens` into a socket nobody was reading, and since each model
has one worker thread, every request queued behind it waited too.

## License

Licensed under either of

- the MIT license ([LICENSE-MIT](LICENSE-MIT))
- the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work shall be dual licensed as above, without any
additional terms or conditions.

Third-party components — llama.cpp and GGML among them — are listed in
[NOTICE](NOTICE).

## Notes

- The HuggingFace client defaults to `https://hf-mirror.com`; override with
  `HF_ENDPOINT`, authenticate gated repos with `HF_TOKEN`.
- `llama-cpp-4` links shared libraries; the desktop app stages them next to the
  sidecar on first launch, and the Docker image copies the `.so` files.
- On a restricted network, set the crates.io mirror in `~/.cargo/config.toml`:
  ```toml
  [source.crates-io]
  replace-with = "rsproxy-sparse"
  [source.rsproxy-sparse]
  registry = "sparse+https://rsproxy.cn/index/"
  ```
