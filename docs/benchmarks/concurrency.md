# Serving several people from one card

How many concurrent users a single GPU can carry, measured rather than guessed.
The numbers below come from one machine; the method is what this page is for.

## What was measured

Nexllama's own daemon runs one decode loop per loaded model: requests queue, and
the second caller waits for the first reply to finish. `llama-server` from the
same llama.cpp tree does continuous batching with `--parallel N` slots instead.
This benchmark compares the two shapes and finds where the card stops scaling.

Protocol, for every `--parallel` value and every drafter setting:

1. Start `llama-server` with the model, `-c 32768` (the pool is divided between
   slots), `-fa on`, `-ctk f16 -ctv f16`, `-b 2048 -ub 512`.
2. Warm up with one short request.
3. Measure a **single** request while idle: 128 tokens, temperature 0.8.
4. Fire **N requests at once**, each with its own conversation so no slot cache
   is shared, and record the aggregate tokens/second, the median speed each
   request saw, and the wall time for all N.

## Hardware

| | |
|---|---|
| GPU | Quadro RTX 6000 (Turing, sm_75), 24 GB, 672 GB/s |
| Model | Qwen3.6-35B-A3B-UD-IQ4_XS (MoE, 3B active) |
| Build | llama.cpp b11829, CUDA 13, `CMAKE_CUDA_ARCHITECTURES=75` |

## Results

| Drafter | Slots | Single req (tok/s) | Aggregate (tok/s) | Each request (tok/s) | Wall for N (s) | VRAM (MiB) |
|---|---|---|---|---|---|---|
| none | 1 | 93.1 | 83.8 | 93.0 | 1.53 | 18945 |
| none | 2 | 93.1 | 137.6 | 74.4 | 1.86 | 18929 |
| none | 4 | 95.0 | 197.0 | 53.6 | 2.60 | 19049 |
| none | 8 | 94.9 | **224.5** | 30.2 | 4.56 | 19295 |
| MTP d=2 | 1 | 115.2 | 105.4 | 120.2 | 1.21 | 19629 |
| MTP d=2 | 2 | 127.7 | 143.2 | 87.9 | 1.79 | 19886 |
| MTP d=2 | 4 | **132.7** | 198.6 | 59.7 | 2.58 | 20401 |
| MTP d=2 | 8 | 116.9 | 220.7 | 34.8 | 4.64 | 21419 |

Three things worth keeping:

- **Throughput stops paying at four slots.** 1→2 is +64%, 2→4 is +43%, 4→8 is
  +14%. The card's ceiling here is about 200–225 tok/s of aggregate decode.
- **MTP is a single-user optimisation, not a concurrency one.** At four slots it
  lifts each request from 95 to 133 tok/s, but at eight slots the aggregate is
  slightly *lower* than without it (220.7 vs 224.5) and it costs 2 GB of VRAM.
- **Drafter and slots compose.** Speculative decoding works with
  `--parallel 8`; there is no need to choose between them at the flag level.

## What that means for 20 people

At 300 tokens per reply:

| Slots | Each of 8 callers | One round of 20 questions | Verdict |
|---|---|---|---|
| 4 | 60 tok/s (5 s for 300 tokens) | ~30 s | one question per person per minute fits |
| 8 | 30 tok/s (10 s for 300 tokens) | ~27 s | same capacity, slower per person |

For a 24 GB card the practical setting is **four slots with MTP** for up to four
people at a time, or **eight slots without it** when more people share the
machine and the aggregate matters more than any one reply. Above eight slots the
VRAM budget (21.4 GB with MTP at eight) leaves nothing for the desktop.

## Turning it on in the app

Settings → **Server mode** (服务端模式) starts the daemon with a `llama-server` it
hosts itself:

- It needs a `llama-server.exe`. The installer does not bundle one; the setting
  takes a path, and a build placed beside the backend's `infer-daemon.exe` is
  found without one. When neither exists the page says so and the switch has no
  effect.
- **Concurrent slots** is `--parallel`. The context setting is the whole pool,
  so each conversation gets it divided by the slot count.
- The switch applies on the next daemon start, so restart the service after
  changing it. Loading a model then starts the server, and `/admin/status` lists
  it with its port and slot count; unloading stops it and frees the card.
- Loading a speech, embedding or reranking model is refused while server mode is
  on: one `llama-server` serves one chat model at a time.

## Reproducing it

```powershell
# 1. Build a server for this GPU. The official Windows CUDA zip is compiled for
#    Ampere and newer and dies with `CUDA error` on Turing, so build from source.
#    The copy inside the llama-cpp-sys crate has tools/server/main.cpp stripped,
#    so fetch the full tree at the commit the crate pins:
#      llama-server.exe --version  ->  build 11829, commit 184f65360
curl.exe -L -o llama.tar.gz https://codeload.github.com/ggml-org/llama.cpp/tar.gz/184f65360
tar.exe -xzf llama.tar.gz

cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 `
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j 20

# 2. Stop any other daemon: the benchmark needs the whole card.
# 3. Measure.
.\scripts\bench-concurrency.ps1 `
    -ServerExe .\build\bin\llama-server.exe `
    -Model C:\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf `
    -Parallel 1,2,4,8 -Spec none,mtp
```

## Notes and limits

- The benchmark measures **decode**, not prefill. With many people interleaving
  conversations the prompt cache holds only a few of them, so prefill cost grows
  with conversation length; a long chat pays for it on every turn.
- Numbers are for this card and this quantisation. A Blackwell card has 1.33x
  the bandwidth and much faster per-SM execution, so expect more than the
  bandwidth ratio suggests — measure it rather than scaling these.
- `-c 32768` is the total pool: at eight slots each conversation gets 4k. Raise
  it only if the card has room.
