# Nexllama

[![许可证: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![下载量](https://img.shields.io/github/downloads/MuseThinker/Nexllama/total?label=downloads)](https://github.com/MuseThinker/Nexllama/releases)
[![平台: Windows | Linux](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-lightgrey.svg)]
[![引擎: llama.cpp](https://img.shields.io/badge/engine-llama.cpp-blue.svg)]
[![文档: OpenAPI 3.1](https://img.shields.io/badge/docs-OpenAPI%203.1-green.svg)]
[![English](https://img.shields.io/badge/lang-English-lightgrey.svg)](README.md)

**Nexllama：本地大模型的一体化推理桌面与 API 网关——原样封装 llama.cpp，硬件自动适配、投机解码加速、前缀缓存，本地与云端共用一个端口。**

## ✨ Nexllama

Nexllama 是一个对硬件有感知能力的一体化本地推理桌面与 API 网关，对 llama.cpp 只做封装、不改动内核。GGUF 模型可以自动下载、一键导入，模型库与 OpenAI 兼容接口一应俱全；不需要自己编译，也不用手工去调 GPU 卸载层数与后端——程序会探测本机硬件并据此完成适配。原生支持 MTP、DFlash2、EAGLE-3、n-gram 投机解码与前缀缓存，并能判断收益、在划不来时自动停用加速。面向 Agent 场景做了优化：流式输出可中断、原生工具调用、模型名容错解析。本地 GGUF 与云端大模型共用一个 API 端点，可选的算力路由能在本地吃不消时平滑切到云端；路由默认关闭、不绑定任何厂商，同时完整保留 llama.cpp 的原生推理性能。

## 🚀 核心亮点

| 特性 | 说明 |
|---|---|
| **一体化 All-in-One** | 直接使用原生 llama.cpp，只是把命令行换成了图形界面：模型库、下载器与 OpenAI 兼容 API 全部内置。装完即用，不必再配依赖，也不用记各种参数。 |
| **硬件自适应** | 推理层数、上下文长度与显存卸载，都由 llama.cpp 自带的 fit 助手推算，不靠估摸；后端按实际硬件来选；本机需要、而安装包里没有的后端，会依据签名清单按需下载、校验并安装。 |
| **推理够快** | 原生支持 MTP、DFlash2、EAGLE-3 与 n-gram 投机解码，自适应闸门会在收益消失的瞬间停用加速；前缀缓存则消掉长对话中占大头的重复预填充——两者按模型择一生效，从不叠加。 |
| **为 Agent 而生** | 提供 OpenAI 兼容端点，OpenClaw、Hermes 以及各类 SDK 客户端无需改动即可接入；原生工具调用、可取消的流式输出、略微过期的模型名也能解析，出错时给出足以据以处理的信息。 |
| **本地与云端统一** | 本地模型与云厂商直连服务跑在同一个端口上，可选的 `auto` 策略让活儿尽量留在本机、只把本机不擅长的部分送出去——同样的质量、更低的账单，且不锁定任何厂商。 |

> 仓库简介（英文 / GitHub repo description）：*Nexllama is a hardware-aware
> all-in-one desktop inference and API gateway, losslessly wrapping llama.cpp. It
> supports auto-download and one-click import for GGUF models with a built-in model
> repository and OpenAI-compatible API. No manual compilation or GPU layer tuning;
> hardware resources are auto-detected and optimized. Native support for MTP,
> DFlash2, EAGLE-3, n-gram speculative decoding and prefix caching, with an
> evaluator to toggle acceleration. Optimized for Agents, it supports stream
> cancellation, native tool calling and model name parsing. Unify local GGUF and
> cloud LLMs under one API endpoint. Optional routing falls back to cloud when local
> hardware hits limits, disabled by default, vendor-agnostic while preserving full
> llama.cpp performance.*

---

## 安装

### Windows

两种安装包，差别只有一点：要不要把英伟达运行库一起打包。

| 安装包 | 体积 | 适用机器 |
|---|---|---|
| **标准版** [`Nexllama_0.1.1_x64-setup.exe`](https://github.com/MuseThinker/Nexllama/releases/download/v0.1.1/Nexllama_0.1.1_x64-setup.exe) | ~495 MB | **推荐首选。** 内置一份可移植的 CPU 构建和英伟达 CUDA 运行库（`cudart64`、`cublas64`、`cublasLt64`），不需要预装任何东西。新电脑、笔记本，或者你不确定装没装过 CUDA 的机器，都选它：有 N 卡就走 GPU，没有就自动用 CPU。 |
| **精简版** [`Nexllama_0.1.1_x64-lite-setup.exe`](https://github.com/MuseThinker/Nexllama/releases/download/v0.1.1/Nexllama_0.1.1_x64-lite-setup.exe) | ~143 MB | 只适合**已经装好 CUDA 运行库**的机器，通常是开发机。它省掉了那三个英伟达 DLL——其中 `cublasLt64_13.dll` 一个就有 428 MB。**新电脑或从没装过 CUDA 的机器请用标准版**：缺运行库时 CUDA 后端起不来，应用会退回 CPU 运行（能用，但慢得多）。 |

两个安装包的产品名、安装目录与开机启动项一致，装另一个会直接覆盖，不会留下两份。

安装包未做代码签名，Windows 可能提示“未知发布者”。请选择 **更多信息 → 仍要运行**，并用 [`SHA256SUMS.txt`](https://github.com/MuseThinker/Nexllama/releases/download/v0.1.1/SHA256SUMS.txt) 校验下载文件。

运行安装程序并打开 Nexllama。首次启动时，应用会按本机情况挑一个后端，并在 `http://127.0.0.1:15656` 上拉起守护进程。

安装器的选项页会问你要留下什么——桌面快捷方式和开机自启，默认都勾选，之后随时可在设置里改。

模型默认从 `C:\models` 读取；想换位置就在设置里改，或者在首次启动前把一个 GGUF 放进去。

### Linux

同一套源码会产出三种安装包：`.deb`（Debian/Ubuntu）、`.rpm`（Fedora/openSUSE）和 `.AppImage`（任意发行版，免安装）：

```bash
chmod +x Nexllama-*.AppImage && ./Nexllama-*.AppImage      # 免 root
sudo apt install ./Nexllama_*.deb                          # Debian/Ubuntu
sudo dnf install ./Nexllama-*.rpm                          # Fedora/openSUSE
```

它们基于 Ubuntu 22.04 / Debian 12 的库（glibc 与 webkit2gtk 4.1）构建，因此这一代及更新的系统都能运行。

自己构建：

```bash
./scripts/package.sh cpu vulkan    # Linux：.deb + .rpm + .AppImage
```

除工具链外，还需要打包器会用到并链接的一些包：

```bash
sudo apt install build-essential cmake ninja-build pkg-config \
    libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev \
    libayatana-appindicator3-dev patchelf rpm zip
```

守护进程也可以单独运行，这通常是服务器上的用法：

```bash
./infer-daemon --model /srv/models/model.gguf --port 15656 --n-gpu-layers 999
```

### Docker

默认使用 GPU。镜像在同一端口提供界面、API 和 HuggingFace 搜索/下载，`docker compose up -d` 会把所有层放到 GPU：

```bash
docker compose up -d          # http://localhost:8090
```

宿主机需要安装 NVIDIA Container Toolkit。`docker-compose.yml` 构建
`Dockerfile.web.cuda`。没有 NVIDIA GPU 的机器改用 CPU 镜像：

```bash
docker compose -f docker-compose.cpu.yml up -d
```

这会构建 `Dockerfile.web.cpu`，所有层都在 CPU 上运行；默认仍然是 GPU。

打开 `http://localhost:8090`；把 `.gguf` 放进 `./models`（或设置 `MODELS_DIR`）后，模型会出现在模型库中。点“加载”把它放到 GPU，然后即可对话。模型页的搜索和下载也会写入 `./models`，所以挂载是可写的。`docker compose down` 停止服务。

默认端口没有 token，笔记本上这样最方便。其他环境请设置 token——守护进程自己读取 `NEXLLAMA_API_KEY`，应用会在设置页要求输入：

```bash
API_KEY=$(openssl rand -hex 16) docker compose up -d
```

模型从 `./models` 读取并写入；在 Linux 主机上，挂载目录需要对 uid `10001` 可写，Hub 下载才能成功。可覆盖的环境变量：`PORT`、`MODELS_DIR`、`API_KEY`、`RUST_LOG`、`CUDA_VERSION`，以及 crates.io 镜像构建参数 `CRATES_MIRROR`。

### 从源码构建

```powershell
npm --prefix frontend install
.\scripts\package.ps1                 # 标准版：cpu + cuda + 英伟达运行库
.\scripts\package.ps1 -Edition lite   # 精简版：不打包英伟达运行库
.\scripts\package.ps1 -Edition both   # 两者都出，只编译一次守护进程
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.1_x64-setup.exe
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.1_x64-lite-setup.exe
```

`-Native` 会把构建机的指令集编进去——CPU 推理更快，但产物只能在同款或同代 CPU 上运行。个人自用无所谓，要发给别人就别加。

<details>
<summary>构建前置依赖</summary>

| 依赖 | 用途 |
|---|---|
| Rust（stable）、Node.js 22+ | 全部 |
| Visual Studio Build Tools (C++)、LLVM（`clang`、`libclang`）、CMake、Ninja | `llama-cpp-4` 的 bindgen 与原生构建 |
| CUDA Toolkit | `cuda` 后端 |
| LunarG Vulkan SDK（`glslc`） | `vulkan` 后端 |

`scripts/build.ps1` 会自动找到这些组件，并配好 `LIBCLANG_PATH`、`CFLAGS=/utf-8`、`CMAKE_GENERATOR=Ninja` 与 MSVC 环境。
</details>

---

## 快速开始

1. **选模型文件夹**——设置 → *模型文件夹*（默认 `C:\models`）。
2. **加载模型**——在模型库里点**加载**。重启或更新前已加载的模型会自动恢复；文件被挪走了的会被跳过，而非当作失败。
3. **用起来**——在对话面板里，或者用任意客户端：

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
# Ollama 兼容客户端也能用同一端口：
#   /api/tags   /api/chat   /api/generate   /api/embed
```

`http://127.0.0.1:15656/docs` 提供一个完全离线运行的交互式 API 页面。

---

## 功能

### 本地与云端共用一个端点

在 *模型 → 云端模型* 里从内置目录添加服务商：选厂商、贴密钥、拉取模型列表、保存。之后它的模型会以 `<provider>/<model>` 的形式出现在 `/v1/models` 中，和本地模型并列；请求里点到哪个，就会被转发到对应厂商。

- **只直连厂商。** 包括 DeepSeek、OpenAI、Anthropic、Google、xAI、Mistral、Cohere、智谱 GLM、Kimi、通义 Qwen、MiniMax；国内外端点不同的，分别提供。Token 套餐（xAI CLI、MiniMax、智谱 Coding）可以用 token 代替 API key。
- **流式原样转发**，连终止符一起，中途不做任何重新分块。HTTP/2 加预热连接池，让首 token 延迟贴近厂商本身。
- **云端流量与本地推理互不干扰。** 服务商注册表单独加锁，代理走异步，云端请求永远碰不到模型工作线程或 GPU——某个服务商再慢，也只拖慢它自己的请求。
- **云端请求会离开本机。** 配置页写明了这一点，模型列表里也把云端条目标了出来，方便客户端分辨。

### `auto` 路由策略

默认关闭。开启后，`auto` 会出现在 `/v1/models` 中，客户端可以发送 `model: "auto"`：

| 情形 | 结果 |
|---|---|
| 本地优先（默认），本地模型已驻留 | 由本地模型回答 |
| 本地优先，本地模型缺失或启动失败 | 请求转到云端 |
| 云端优先 | 由云端模型回答 |
| 云端优先，云端不可达 | 回退到本地模型 |
| 选定的一侧在**开始出字之后**失败 | 直接返回错误，不会拿另一个答案顶上 |

可以在应用里开启，也可以免交互地：

```bash
POST /admin/routing {"enabled":true,"prefer":"local"}
```

### 投机解码

*对话 → 参数 → 投机解码* 让模型先起草自己的续写，再用一次批处理前向把整段草稿验证掉。支持三类草稿，模型库还会为每个模型包提供**起草器选择器**——`auto` 优先挑外挂的 DFlash/EAGLE 草稿文件，而不是内置的 MTP 头，因为一份公开发布的草稿文件，正是为这个模型专门训练出来的：

- **n-gram 起草**——按最近重复出现过的 `ngram_n` 个 token 之后的内容来提案。学到的表会跨请求、跨重启保留。
- **MTP（NextN 头）**——有些模型在同一份 GGUF 里就自带多 token 预测头（`qwen35.nextn_predict_layers`），也可以单独发布成 `mtp-<model>-<quant>.gguf` 放在模型旁边。
- **DFlash 与 EAGLE-3**——一个更小的、专门训练来预测目标的检查点，用 `draft_model` 与 `draft_n_max` 配置。

自适应闸门（`adaptive_draft`，默认开启，接受率下限 0.3）会在划不来时关掉起草，等接受率回升再打开。它只可能把起草**关掉**，所以一直开着最稳妥。被接受的 token，本来就是模型自己也会采出来的 token，因此这个优化是精确的，不是近似的。

### 加载前的适配检查

`POST /admin/fit` 接受一个 GGUF 路径——文件还没下载时，只给一个 `size_bytes` 也行——回答你：装不装得下、需要多少内存对比还剩多少、llama.cpp 会选多大的 `n_gpu_layers` 与 `n_ctx`、如果是权重本身太大就建议换一个更小的量化，以及一句人话的 `reason`。应用里对每个模型提供**检查适配**与**采用**。默认只作建议；打开 `strict_fit: true` 就会直接拒绝加载。

### 前缀缓存与运行指标

因为每一轮都把上一轮当作前缀，重新编码这段前缀，往往占掉首 token 之前的大部分时间。开启 `prefix_cache`（默认开启）后，KV 状态被复用，只编码新增的后缀。加载了草稿模型或 MTP 头时会自动关闭，因为它们的定位与目标各自独立。

```bash
cargo run -p infer-core --example prefix_check -- models/your-model.gguf 0
```

`/admin/status` 会报告每个模型的 `perf`——请求数、token 数、平滑过的 `tokens_per_second`、`first_token_ms` 与 `draft_acceptance`，所以上面的每一条说法都可以自己去验证，而不必凭信任接受。

### 模型文件夹与多文件模型

模型是通过递归扫描配置文件夹里的 `*.gguf` 发现的（默认 `C:\models`），下载也写到这里。GGUF 头无需加载模型就能读取，因此在加载之前，模型类别、有没有 MTP 头、大概多大，就已经知道了。

拆成多个文件的模型会放在自己的子目录里：

```
models/Qwen2.5-VL-7B/
  Qwen2.5-VL-7B-Q4_K_M.gguf        # 主权重
  Qwen2.5-VL-7B.mmproj-f16.gguf    # 视觉投影器
  Qwen2.5-VL-7B.draft.gguf         # 草稿模型（投机解码）
```

角色由文件名判断（`mmproj`/`projector`、`draft`/`eagle`/`mtp`/`dflash`、`lora`/`adapter`），整个模型包作为一个整体加载，所以模型不会出现丢了投影器或起草器的情况。如果目录里放着多个不相干的模型，就逐个文件列出，因为无从判断谁归谁。

**模型类别**从 GGUF 头里读，而不是猜名字：

| 类别 | 识别依据 | 行为 |
|---|---|---|
| `llm` | 其余全部 | 在对话面板中提供；视觉模型包也归在此列 |
| `embedding` | 已知的编码器架构（`bert`、`nomic-bert`、`jina-bert-v2` …） | 仅供 API——绝不出现在对话中，会被 `/v1/chat/completions` 拒绝 |
| `rerank` | 带 `rerank.head_count` 键且有头数 | 仅供 API |

对话模型是拿来聊、拿来评估的；编码器是给程序指过去用的。两者都不会出现在不该出现的地方。

### 向量、重排与视觉

- **`/v1/embeddings`**——OpenAI 的响应格式，向量为单位长度；如果直接给出池化后的原始输出，点积就会随文本长短而变。上下文按模型类别配置，而不是靠某个开关。超过上下文的输入会报错并写明上限，绝不悄悄截断。
- **`/v1/rerank`**——Cohere/Jina 的响应格式。输入按 `BOS query SEP document EOS` 拼好，也就是分类头训练时用的句对；NaN 分数排在最后，坏掉的头不会被评为最佳匹配。
- **视觉**——含投影器的模型包，在 `/v1/chat/completions` 上接受 OpenAI 风格的 `image_url` 片段。只收 `data:` URL：`http(s)` URL 会把每个请求变成 SSRF 的入口，而裸路径则会指向主机上的文件。模型卸载到 GPU 后，投影器也在 GPU 上跑——一张 336×252 的图在 40 GB 卡上约 200 ms，纯 CPU 则要 50 s 左右。

### 思考开关与工具调用

**思考（Thinking）**按模型在高级面板里设置，也可以在请求里用 `"thinking": true|false` 覆盖。这个开关会以 `enable_thinking` 传进聊天模板，所以真正认得它的模板，就是真正会应用它的模板。

**工具调用**接受 OpenAI 格式的 `tools` 数组（`tool_choice` 与 `parallel_tool_calls` 可选），并以一条普通的 OpenAI 消息作答：

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

模板负责把调用格式化，同一个模板的解析器再把它读回来，这样无论哪个模型家族用的是什么调用语法，都能被理解，而不必给它的输出强加一套文法。`content` 里只有答案，思维链被单独放进 `reasoning_content`，原始调用标记不会流到调用方。流式下结构相同：`[DONE]` 前的最后一块带着 `tool_calls`，每个都带 `index`，并置 `finish_reason: "tool_calls"`。

### 流式与取消

`"stream": true` 会产出 OpenAI 格式的 Server-Sent Events——每块都带 `id`、`object`、`created` 与 `model`，末尾是 `data: [DONE]`，因此官方 SDK 不用改就能用。取消两端都认：客户端关掉响应体，会在下一个 token 处停止生成；应用里的停止按钮也一样。否则，一个中途走掉的读取方，会让模型把剩下的 `max_tokens` 一直解码进没人看的套接字里；而每个模型只有一个工作线程，后面排队的请求都得等它。

### 模型名解析

同一台机器可以同时驻留对话模型、嵌入模型和重排模型，每条路由按自己需要的类别来解析。请求里点到的模型已驻留，就用它；名字对不上，而该类别又恰好只驻留了一个模型时，就由那一个来回答，这样过期的标识符不会把客户端卡死。只有真正出现歧义时才拒绝，并且错误里会列出候选：

```json
{"error": "model \"typo\" not found and more than one llm model is loaded: llama3, qwen2.5"}
```

### 桌面应用

- **常驻托盘。** 关窗口只是隐藏；模型照样加载着，已连上的客户端也不掉线。要退出，从托盘菜单走。
- **重启或更新后模型还在。** 之前加载的那一组会自动恢复；文件挪走了的则跳过。
- **空闲卸载、透明重载。** 设好 `keep_alive_secs` 后，模型空闲到点就释放，下次请求再自动加载回来。
- **每个模型一个高级面板。** GPU 层数、上下文、批处理、KV 卸载、RoPE 缩放、加载模式、内存锁定、思考、MTP 预算、前缀缓存，全在一处。
- **GPU 配置不用猜。** 设置里会同时告诉你探测到的显卡、将选用的后端与正在使用的后端，缺后端还能一键安装。

---

## HTTP API

守护进程监听 `http://127.0.0.1:15656`，同时提供 OpenAI 兼容和 Ollama 兼容两套接口——为其中任一套写好的集成，改个端口就能用。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 横幅与端点索引（公开，不含模型名） |
| GET | `/openapi.json`、`/docs` | OpenAPI 3.1 规范与离线交互页面（公开） |
| GET | `/v1`、`/v1/models` | OpenAI 风格的模型列表——本地、云端，以及启用后的 `auto` |
| GET | `/api/tags`、`/api/version` | Ollama 风格的模型列表与版本 |
| GET | `/health` | 存活状态与已加载的模型（公开） |
| GET | `/admin/status` | 模型信息、类别、运行时 `perf`、空闲时间 |
| POST | `/admin/load` | `{ "model": "path.gguf", "name": "alias", "n_gpu_layers": 999, "n_ctx": 4096, … }` |
| POST | `/admin/unload` | `{ "model": "alias" }`，或 `{}` 表示全部 |
| POST | `/admin/fit` | 适配检查点，或下载前的 `size_bytes` |
| GET, POST | `/admin/providers` | 列出与配置云端服务商；密钥绝不返回 |
| POST | `/admin/providers/test` | 拉取服务商的模型列表以校验密钥 |
| GET, POST | `/admin/routing` | `auto` 策略：`{ "enabled": true, "prefer": "local" }` |
| POST | `/v1/chat/completions` | OpenAI 对话；`"stream": true` 时为 Server-Sent Events |
| POST | `/api/chat`、`/api/generate` | Ollama 对话与单次提示；流式为 NDJSON |
| POST | `/v1/embeddings`、`/api/embed`、`/api/embeddings` | OpenAI 与 Ollama 两种形态的向量 |
| POST | `/v1/rerank`、`/rerank`、`/v1/reranking`、`/reranking` | `{ "model": "…", "query": "…", "documents": ["…"], "top_n": 3 }` |

Ollama 的 `options` 会映射到采样器：`temperature`、`top_k`、`top_p`、`min_p`、`repeat_penalty`、`repeat_last_n`、`num_predict`（生成 token 的上限；`<= 0` 表示"直到模型自己停下来"）与 `seed`。响应会带上惯用的 `prompt_eval_count` 与 `eval_count`，每条流式行都带 `created_at`。

`GET /docs` 会把规范渲染成一个不依赖任何外部脚本或样式的交互页面：token 输入框在该浏览器里保留、每个操作都有一个由规范示例预填好的**发送**按钮、流式路由逐块消费，示例里点名的模型也确实处于加载状态。测试会遍历整份规范：某个文档化路由缺失、某个 schema 引用没有定义、或者 `/docs` 的流程在真实浏览器里退化，都会导致失败。

### 配置

默认值即推荐值；每个选项在设置和按模型的**高级**面板里都能找到。`POST /admin/load` 接受：

| 分组 | 参数 |
|---|---|
| 模型 | `n_gpu_layers`（999 = 全部）、`main_gpu`、`load_mode`（`auto｜none｜mmap｜mlock｜mmap_mlock｜dio`）、`use_mlock` |
| 上下文 | `n_ctx`（0 = 自动，按机器定大小）、`n_batch`（2048）、`n_ubatch`（512）、`flash_attn`（`auto｜on｜off`）、`offload_kqv`、`rope_freq_base`、`rope_freq_scale` |
| CPU | `threads`、`threads_batch`（0 = 自动） |
| 适配器 | `load_mtp`（内置 MTP 头）、`lora`（适配器路径列表） |
| 采样（按请求） | `temperature`、`top_k`、`top_p`、`min_p`、`repeat_penalty`、`repeat_last_n`、`repeat_freq`、`repeat_present`、`seed`、`max_tokens` |
| 投机（按请求） | `speculative`、`ngram_n`、`ngram_m` |

守护进程也把这些选项以命令行参数的形式暴露出来（`--load-mode`、`--flash-attn`、`--n-batch`、`--no-offload-kqv` …），方便无界面与容器场景。

### 对外暴露与安全

守护进程默认只绑 `127.0.0.1`，机器之外碰不到。设置 → *网络* 里可以放宽到 `0.0.0.0`，但这要求带一个 bearer token：

```powershell
infer-daemon.exe --host 0.0.0.0 --port 15656 --api-key <token> --model model.gguf
```

服务器没有账号、没有限流、也没有配额；谁能碰到这个端口，谁就能往本机 GPU 上加载模型、把你的电费烧掉。token 用固定时间比较，除 `/health` 外每条路由都要求它——`/health` 保持公开，供就绪探针使用，只暴露已加载模型的名字。绑到可路由地址却**不带** token 会被拒绝：

```
refusing to listen on 0.0.0.0 without --api-key: anyone who can reach this port
can load models and spend GPU time on them. Pass --api-key <token>, bind to
127.0.0.1, or pass --allow-unauthenticated if something else guards the port
```

`--allow-unauthenticated` 是给那些自己会做认证的容器或反向代理用的；把 `--api-key ""` 传空是错误，而不是悄悄关掉认证的办法。应用把 token 存在操作系统的凭据库里，再通过 `NEXLLAMA_API_KEY` 交给守护进程，所以它既不出现在设置文件旁边，也不出现在进程命令行里。跨域请求只对应用自己的源应答；要加别的源，用 `--cors-origin <origin>`。

图片输入只允许 `data:` URL；应用的能力清单除 `opener:default` 外不授予任何权限。

---

## 模型下载

搜索没有任何限制；下载页把几乎覆盖了所有值得本地一跑的来源，整理成三个一键筛选的目录：

- **bartowski**——覆盖最广，一个模型配齐多种量化
- **Unsloth**——动态量化，以及带 MTP 所需头的那些发布
- **z-lab**——DFlash 草稿模型（来自 UCSD 的 Z-Lab，正是它提出并主导了这套方法）

量化是手艺活，一个差劲的量化丢掉的精度，再多上下文也补不回来；Hub 上其余的内容依然可以取。下载支持断点续传，并在开始传输前按空闲内存做个预检。客户端默认走 `https://hf-mirror.com`；可用 `HF_ENDPOINT` 覆盖，用 `HF_TOKEN` 认证受限仓库。

## 后端更新

设置 → *后端更新* 里填入清单 URL。应用会检查清单、下载匹配的后端压缩包、校验其 SHA-256、替换 `%APPDATA%/com.nexllama.desktop/backends/<name>/`，然后重启守护进程。更新过的后端优先级高于内置的那份。

清单必须**签名**：用 Ed25519 对它的版本、llama.cpp 构建号，以及每个后端的 URL 与哈希签名。光有校验和，只能证明压缩包与指定它的清单对得上，而这两样又来自同一个响应，所以只校哈希，等于谁能给出其中一个，谁就能给出另一个。公钥取自设置或 `NEXLLAMA_UPDATE_KEY`；不预置任何密钥，因为一个谁都信任的更新器等于什么都没验证。抓取强制走 HTTPS（重定向之后也是），版本低于已装的会被拒绝，清单、压缩包与解包后的总量都有上限，畸形发布无法把磁盘塞爆。

```powershell
cargo run -p nexllama-app --example new-update-key     # 公钥与私钥，十六进制
cargo run -p nexllama-app --example sign-manifest -- manifest.json <private-hex>
.\scripts\make-backend-release.ps1 -Version 0.2.0 -LlamaCpp b10502 `
  -BaseUrl https://github.com/<owner>/<repo>/releases/download/v0.2.0
```

---

## 架构

```
frontend/          Svelte 5 + Vite 界面（对话、模型、下载、设置）
crates/
  infer-core/      对 llama.cpp（llama-cpp-4）的安全封装：加载、对话、流式
  infer-daemon/    axum HTTP 服务：/health、/admin/*、/v1/*（SSE）、云端代理
  infer-hub/       HuggingFace 客户端：搜索、文件列表、断点续传下载
  gguf-probe/      GGUF 头部读取：模型类别、MTP 头探测、大小提示
  app/             Tauri 2 桌面外壳
                   · 探测机器并选择后端
                   · 把各后端打包在 backends/<name>/ 下
                   · 把云端密钥存于操作系统凭据库
```

界面通过本地 HTTP 驱动守护进程。`infer-core` 为每个模型开一个专门的工作线程来跑 llama.cpp，并流式输出解码后的 UTF-8；云端流量异步代理，绝不进这个线程。

---

## 故障排查

**安装包被 SmartScreen 拦下。** 默认构建不签名。

**后端选错了。** 设置 → GPU 里会显示探测到的显卡、将选用的后端与正在使用的后端，缺后端还能一键安装。也可以手动指定 Auto / CUDA / Vulkan / CPU；没有 N 卡的机器会退回 CPU，而不是硬启动一个跑不起来的 CUDA 构建。

**模型加载不了。** 对它跑一次**检查适配**：它会给出所需内存对比空闲内存，并建议层数、上下文长度或更小的量化。`/admin/status` 里带着拒绝的原因。

**模型从库里消失了。** 它不在配置的文件夹里。设置 → *模型文件夹* 会显示正在使用的路径，*打开文件夹* 就能打开它。

**挪了文件之后报 `model not found`。** 把文件夹指到新位置即可。文件挪走了的已记忆模型，启动时会被跳过，而不是让启动失败；库里列出的，就是实际存在的东西。

**网络受限，或者 crates.io 太慢。** 在 `~/.cargo/config.toml` 里配个镜像：

```toml
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
```

---

## 当前说明

- 加载草稿模型或 MTP 头时，前缀缓存会被禁用——这些路径与目标各自独立地定位，因此这类模型每一轮都要重新编码对话。
- 硬件加速来自 llama.cpp 提供的后端——CUDA 与 Vulkan。CANN、MUSA 这类厂商专用运行库目前不支持。
- 云端请求会把内容发送给它所指向的服务商。

## 参与贡献

欢迎提交 Issue 和 Pull Request。打包用的同一套脚本，也用来构建和测试整个工作区：

```powershell
.\scripts\build.ps1 fmt --all
.\scripts\build.ps1 clippy --all-targets
.\scripts\build.ps1 test --workspace
```

贡献内容以 MIT 许可证授权。

## 社区

- 🐛 [Issues](issues)——问题反馈与功能建议
- 💡 [Discussions](discussions)——使用疑问、设计讨论与成果展示

## 许可证

本项目以 MIT 许可证授权（[LICENSE](LICENSE)）。

除另有明示外，你为纳入本项目而有意提交的贡献，均按上述许可证授权，不附加任何其他条款。第三方组件——包括 llama.cpp 与 GGML——列于 [NOTICE](NOTICE)。
