# Nexllama

[![许可证: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![平台: Windows | macOS | Linux](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]
[![引擎: llama.cpp](https://img.shields.io/badge/engine-llama.cpp-blue.svg)]
[![文档: OpenAPI 3.1](https://img.shields.io/badge/docs-OpenAPI%203.1-green.svg)]
[![English](https://img.shields.io/badge/lang-English-lightgrey.svg)](README.md)

**Nexllama：一体化本地大模型推理桌面与 API 服务——无损封装 llama.cpp，自动硬件适配、投机解码加速、前缀缓存，本地与云端同一端口。**

## ✨ Nexllama

Nexllama是硬件感知的一体化本地大模型桌面推理与API网关，无损封装llama.cpp，支持GGUF模型自动下载、一键导入，内置模型仓库与OpenAI兼容接口，无需编译源码、手动调试GPU卸载层数与后端，自动探测硬件资源并完成适配调度。原生支持MTP、DFlash2、EAGLE-3、n-gram投机解码及前缀缓存，内置收益判断逻辑智能启停加速策略，专为Agent场景优化，提供流式中断、原生工具调用与模型容错解析。实现本地GGUF与云端大模型共用同一API端点，可选算力路由，本地算力不足时平滑切换云端，路由默认关闭，无厂商绑定，完整保留llama.cpp原生推理性能。

## 🚀 核心亮点

| 特性 | 介绍 |
|---|---|
| **一体化封装 All-in-One** | 无损封装原生 llama.cpp，图形界面内置模型库、下载器与 OpenAI 兼容 API。一次安装，不再额外配置依赖，告别繁琐命令行操作。 |
| **智能硬件自适应** | 自动探测 GPU/CPU/内存，推理层数、上下文窗口、显存卸载策略由 llama.cpp 自有 fit 助手拟合；缺失的后端经签名清单按需下载、校验并安装。 |
| **极速推理体验** | 原生支持 MTP、DFlash2、EAGLE-3 与 n-gram 投机解码，自适应闸门在收益不足时自动关闭；多轮长对话靠前缀缓存压低首 token 延迟——两者按模型择一生效，从不叠加。实测 Qwen3.8-27B 解码提速最高 38%。 |
| **为 Agent 深度优化** | 原生 OpenAI 兼容，内置流式取消、模板原生工具调用、模型名容错解析，专为多轮智能体调用降低报错。 |
| **本地与云端协同** | 统一 API 网关：优先本地 GPU，本地模型缺失或失败时回退云端（或反向），不绑定任何云厂商；默认关闭，按需开启。 |

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

两种安装包，区别只有一件事：英伟达运行库要不要一起打包。

| 安装包 | 体积 | 适用机器 |
|---|---|---|
| **标准版** `Nexllama_*_x64-setup.exe` | ~483 MB | **推荐用这个。** 自带可移植 CPU 构建和英伟达 CUDA 运行库（`cudart64`、`cublas64`、`cublasLt64`），**不需要**预先安装任何东西。新电脑、笔记本、或者你不确定装没装过 CUDA 的机器都选它：有 N 卡就走 GPU，没有就自动用 CPU。 |
| **精简版** `Nexllama_*_x64-lite-setup.exe` | ~130 MB | 只适合**本机已经装好 CUDA 运行库**的机器（通常是开发机）。它不打包那三个英伟达 DLL——其中 `cublasLt64_13.dll` 一个就有 428 MB。**新电脑或从来没装过 CUDA 的机器请用标准版**：没有运行库时 CUDA 后端起不来，应用会自动退回 CPU 运行（能用，但慢很多）。 |

两个安装包的产品名、安装位置、开机启动项都相同，装另一个会直接覆盖，不会装成两份。

运行安装程序并打开 Nexllama。应用会在首次启动时按本机情况选择后端，并在 `http://127.0.0.1:15656` 启动守护进程。

安装器会在选项页询问要留下什么——桌面快捷方式和开机启动，默认都勾选，之后可在设置中随时更改。

默认从 `C:\models` 读取模型——可在设置中更改位置，或在首次启动前把一个 GGUF 放到那里。

### macOS 与 Linux

```bash
./scripts/package.sh metal cpu     # macOS: .app / .dmg
./scripts/package.sh cpu vulkan    # Linux: .deb / .rpm / .AppImage
```

守护进程也可独立运行，通常作为服务器使用：

```bash
./infer-daemon --model /srv/models/model.gguf --port 15656 --n-gpu-layers 999
```

Linux 打包还需 `libwebkit2gtk-4.1-dev`、`libgtk-3-dev`、`librsvg2-dev` 与
`libayatana-appindicator3-dev`。

### Docker

```bash
docker compose build
MODEL=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf docker compose up           # CPU
docker compose -f docker-compose.cuda.yml up                                 # CUDA
```

模型以只读方式挂载自 `./models`。环境变量覆盖项：`PORT`、`MODEL`、`N_GPU_LAYERS`、
`N_CTX`、`KEEP_ALIVE_SECS`、`MODELS_DIR`，以及构建参数 `CRATES_MIRROR`（crates.io 镜像）。

### 从源码构建

```powershell
npm --prefix frontend install
.\scripts\package.ps1                 # 标准版：cpu + cuda + 英伟达运行库
.\scripts\package.ps1 -Edition lite   # 精简版：不打包英伟达运行库
.\scripts\package.ps1 -Edition both   # 两种都出（只编译一次守护进程）
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-setup.exe
# -> crates\app\target\release\bundle\nsis\Nexllama_0.1.0_x64-lite-setup.exe
```

`-Native` 会把本机指令集编进去（CPU 推理更快），但只能在本机/同代 CPU 上运行：
个人自用可以加，要发布给别人就别加。

<details>
<summary>构建前置依赖</summary>

| 依赖 | 用途 |
|---|---|
| Rust（stable）、Node.js 22+ | 全部 |
| Visual Studio Build Tools (C++)、LLVM（`clang`、`libclang`）、CMake、Ninja | `llama-cpp-4` 的 bindgen 与原生构建 |
| CUDA Toolkit | `cuda` 后端 |
| LunarG Vulkan SDK（`glslc`） | `vulkan` 后端 |

`scripts/build.ps1` 会自动定位这些组件并配置 `LIBCLANG_PATH`、`CFLAGS=/utf-8`、
`CMAKE_GENERATOR=Ninja` 与 MSVC 环境。
</details>

---

## 快速开始

1. **选择模型文件夹**——设置 → *模型文件夹*（默认为 `C:\models`）。
2. **加载模型**——在模型库中点击**加载**。重启或更新前已加载的模型会自动恢复；文件被移动过的会被跳过而非视为失败。
3. **使用**——通过对话面板，或任意客户端：

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

### 本地与云端模型共用一个端点

在 *模型 → 云端模型* 中从内置目录添加服务商：选择厂商、粘贴密钥、拉取模型列表、保存。其模型随后以 `<provider>/<model>` 出现在 `/v1/models` 中，与本地模型并列；命名其一的请求会被代理到对应厂商。

- **仅直连厂商。** DeepSeek、OpenAI、Anthropic、Google、xAI、Mistral、Cohere、智谱 GLM、Kimi、通义 Qwen、MiniMax，国内外端点不同时分别提供。Token 套餐（xAI CLI、MiniMax、智谱 Coding）可用 token 代替 API key。
- **流式转发原样不变**，含终止符；传输中不做任何重分块。HTTP/2 与预热连接池让首 token 延迟接近厂商自身。
- **云端流量与本地推理隔离。** 服务商注册表使用独立锁，代理异步进行，云端请求绝不触及模型工作线程或 GPU——慢速服务商只会拖慢它自己的请求。
- **云端请求会离开本机。** 配置页已明示，且云端条目在模型列表中标记，便于客户端区分。

### `auto` 路由策略

默认关闭。启用后，`auto` 出现在 `/v1/models` 中，客户端可发送 `model: "auto"`：

| 情形 | 结果 |
|---|---|
| 本地优先（默认），本地模型驻留 | 由本地模型回答 |
| 本地优先，本地模型缺失或加载失败 | 请求转至云端 |
| 云端优先 | 由云端模型回答 |
| 云端优先，云端不可达 | 回退至本地模型 |
| 选定一侧在**输出开始之后**失败 | 返回错误；不会用另一个答案替代 |

可在应用中开启，或免交互地：

```bash
POST /admin/routing {"enabled":true,"prefer":"local"}
```

### 投机解码

*对话 → 参数 → 投机解码* 让模型起草自己的续写，并在一次批处理前向中验证整段草稿。支持三族提案，且模型库对每个模型包提供**起草器选择器**——`auto` 优先选用外挂的 DFlash/EAGLE 草稿文件而非内置 MTP 头，因为一份公开发布的草稿文件正是为该模型专门训练的：

- **N-gram 起草**——根据最近重复出现的 `ngram_n` 个 token 之后的内容提案。学习到的表跨请求与重启持久保留。
- **MTP（NextN 头）**——部分模型在同一 GGUF 内自带的多 token 预测头（`qwen35.nextn_predict_layers`），或单独发布为 `mtp-<model>-<quant>.gguf` 并放在模型旁。
- **DFlash 与 EAGLE-3**——第二个更小的、经训练用以预测目标的检查点，通过 `draft_model` 与 `draft_n_max` 配置。

自适应闸门（`adaptive_draft`，默认开启，接受率下限 0.3）在收益停止时关闭起草，并在接受率恢复时重新开启。它只会把起草**关掉**，因此保持开启是最安全的选择。被接受的 token 正是模型自身本就会采样出的 token，所以这一优化是精确而非近似的。

<details>
<summary>基准：同一模型的五种起草配置</summary>

Qwen3.8-27B-UD-IQ4_XS（13.3 GB）于 24 GB Quadro RTX 6000，`n_ctx=4096`，
`n_gpu_layers=999`，`max_tokens=256`，temperature 1.0 / top_p 0.95 / top_k 20，seed 42，
固定单条提示。第 1–4 行与第 6 行为三次会话（一次预热 + 七次计时）；第 5 行为一次会话五次。
*Decode* 为所有生成时间内的生成 token 数。

| # | 起草器 | `draft_n_max` | 解码 tok/s | 加速 | TTFT | prefill tok/s | 接受率 | 每轮接受 |
|---|---|---|---|---|---|---|---|---|
| 1 | 无（基线） | – | **24.3**（24.0–24.7） | — | 126–128 ms | 414–433 | – | – |
| 2 | DFlash `Q4_0`（draft_model） | 2 | 27.9（25.9–30.9） | +15 % | 178–206 ms | 197–223 | 62.7 % | 1.25 |
| 3 | DFlash2 `Q4_K_M`（draft_model） | 3 | 27.6（26.2–30.1） | +14 % | 204–242 ms | 174–199 | 56.5 % | 1.69 |
| 4 | MTP 外挂文件 `Q4_0`（draft_model） | 2 | 32.6（32.1–33.7） | +34 % | 178–215 ms | 187–224 | 64.7 % | 1.29 |
| 5 | MTP 内置头，`--mtp-n-max` | 2 | **33.9** | **+38 %** | 172 ms | 231 | 64.9 % | 1.30 |
| 6 | 对照：DFlash2 驻留，`speculative: false` | – | 24.1（24.0–24.1） | 0 % | 131–134 ms | 398–403 | – | – |

- **第 6 行是对照。** 驻留的起草器空闲时零成本：解码与 prefill 完全一致。第 2–5 行的首 token 与 prefill 退步，是起草*处于活动*所致——提示也要经由起草器做 prefill，吞吐大约减半。
- **MTP 两族中更强且稳定。** 两个 MTP 行在每次会话都保持在 32–34 tok/s，而 DFlash 行在 26–31 间波动。MTP 从目标自身隐藏状态、经紧邻末块的一个头起草，提案更贴近目标，且只花一个前向块而非一整个第二模型。
- **草稿长度有小幅最优。** DFlash `Q4_0` 在 2 最快，DFlash2 在 3，MTP 在 2。把 DFlash2 提到 4 会把接受率从 56.5 % 降到 45.6 %，退回 `n_max=2` 的吞吐。
- **文本与无草稿运行并非逐位一致。** 验证草稿走的是批处理而非单 token；浮点求和顺序不同，近平局可能反向解析。每条路径各自确定。
</details>

### 实测配置

`POST /admin/fit` 接受 GGUF 路径——或仅 `size_bytes`（针对尚未下载的文件）——并回答：能否装下、所需内存相对空闲内存、llama.cpp 会选择的 `n_gpu_layers` 与 `n_ctx`、当权重本身才是问题时建议的更小量化，以及一段人话 `reason`。应用对每个模型提供**检查适配**与**采用**。默认只是建议；`strict_fit: true` 会直接拒绝加载。

### 前缀缓存与运行指标

由于每一轮都包含前一轮作为前缀，重新编码该前缀占据了首 token 之前的大部分延迟。开启 `prefix_cache`（默认开启）后，KV 状态被复用，只编码新的后缀。当加载了草稿模型或 MTP 头时会自动关闭，因为它们与目标各自独立地追踪位置。

```bash
cargo run -p infer-core --example prefix_check -- models/your-model.gguf 0
```

`/admin/status` 报告每个模型的 `perf`——请求数、token 数、平滑后的 `tokens_per_second`、
`first_token_ms` 与 `draft_acceptance`——因此上面的每一条主张都可验证而非仅靠信任。

### 模型文件夹与多文件模型

通过递归扫描配置文件夹中的 `*.gguf`（默认 `C:\models`）发现模型。下载也写入同一位置。GGUF 头部会在不加载模型的情况下被读取，因此在加载前即可获知模型类别、是否带 MTP 头以及大小提示。

一个拆分为多个文件的模型放在自己的子目录中：

```
models/Qwen2.5-VL-7B/
  Qwen2.5-VL-7B-Q4_K_M.gguf        # 主权重
  Qwen2.5-VL-7B.mmproj-f16.gguf    # 视觉投影器
  Qwen2.5-VL-7B.draft.gguf         # 草稿模型（投机解码）
```

角色由文件名识别（`mmproj`/`projector`、`draft`/`eagle`/`mtp`/`dflash`、`lora`/`adapter`），
模型包作为一个整体加载，因此模型绝不会丢掉它的投影器或起草器。包含多个无关模型的目录按文件逐一列出，因为无法推断归属。

**模型类别**从 GGUF 头部读取，而非靠名字猜测：

| 类别 | 识别依据 | 行为 |
|---|---|---|
| `llm` | 其余所有 | 在对话面板提供；视觉模型包也在此列 |
| `embedding` | 已知编码器架构（`bert`、`nomic-bert`、`jina-bert-v2` …） | 仅 API——绝不在对话中提供，被 `/v1/chat/completions` 拒绝 |
| `rerank` | 带 `rerank.head_count` 键且有头数 | 仅 API |

对话模型是用来评估、与之交谈的；编码器是程序指向的。二者都不在对方该出现的地方被提供。

### 嵌入、重排序与视觉

- **`/v1/embeddings`**——OpenAI 响应形态，向量为单位长度；原始池化输出会让点积依赖于文本长度。上下文由模型类别配置，而非某个标志。超过上下文的输入是命名了上限的错误，绝不静默截断。
- **`/v1/rerank`**——Cohere/Jina 响应形态。输入构造为 `BOS query SEP document EOS`，即分类头训练所用的句对；NaN 分数排到最后，坏掉的头不会被报为最佳匹配。
- **视觉**——含投影器的模型包在 `/v1/chat/completions` 接受 OpenAI 风格的 `image_url` 片段。只接受 `data:` URL：一个 `http(s)` URL 会把每次请求变成 SSRF 原语，裸路径则会指名主机上的文件。投影器在模型卸载到 GPU 时于 GPU 上运行——一张 336×252 图在 40 GB 卡上约 200 ms，CPU 上约 50 s。

### 推理控制与工具调用

**思考**在高级面板中按模型设置，也可在请求中以 `"thinking": true|false` 覆盖。该开关以
`enable_thinking` 抵达聊天模板，因此理解它的模板就是应用它的模板。

**工具调用**接受 OpenAI 形态的 `tools` 数组（可选 `tool_choice` 与 `parallel_tool_calls`），并以普通 OpenAI 消息作答：

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

模板负责格式化调用，同一个模板的解析器负责读回，因此任一模型家族所用的调用语法都能被理解，而无需对其输出强加文法。`content` 只含答案，思维链被分离到 `reasoning_content`，原始调用标记从不抵达调用方。流式也携带相同结构：在 `[DONE]` 之前的最后一块携带 `tool_calls`，各带 `index`，且 `finish_reason: "tool_calls"`。

### 流式与取消

`"stream": true` 产生 OpenAI 形态的 Server-Sent Events——每块都带 `id`、`object`、`created` 与 `model`，末尾 `data: [DONE]`——因此官方 SDK 无需改动即可工作。取消从两端都被尊重：关闭响应体在下个 token 处停止生成，应用的停止按钮亦然。否则被遗弃的读取者会让模型把 `max_tokens` 的余下部分解码进无人读取的套接字，而每个模型独占一个工作线程，每个排队请求都要等它。

### 模型名解析

一台机器可同时驻留对话模型、嵌入模型与重排序模型，每条路由按所需模型类别解析。命名了某驻留模型的请求使用它；当名字匹配不到且恰好只有一个该类别模型驻留时，由该模型回答，因此陈旧的标识符不会弄垮客户端。只有真正歧义才被拒绝，且错误会列出候选：

```json
{"error": "model \"typo\" not found and more than one llm model is loaded: llama3, qwen2.5"}
```

### 桌面应用

- **驻留托盘。** 关闭窗口只是隐藏；模型保持加载，已连接的客户端保持附着。从托盘菜单退出。
- **重启或更新后模型回归。** 之前加载的集合自动恢复；被移动的文件被跳过。
- **空闲卸载与透明重载。** 设置 `keep_alive_secs` 后，模型在空闲若干秒后释放，并在下次请求时重载。
- **按模型的统一高级面板。** GPU 层数、上下文、批处理、KV 卸载、RoPE 缩放、加载模式、内存锁定、思考、MTP 预算与前缀缓存，尽在一处。
- **无猜测的 GPU 配置。** 设置在检测到的适配器、将选用的后端、正在使用的后端之间报告，并可按需安装缺失后端。

---

## HTTP API

守护进程监听 `http://127.0.0.1:15656`，同时提供 OpenAI 兼容与 Ollama 兼容两套接口，因此为任一者写的集成只需改端口即可工作。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 横幅与端点索引（公开；不含模型名） |
| GET | `/openapi.json`、`/docs` | OpenAPI 3.1 规范与离线交互页面（公开） |
| GET | `/v1`、`/v1/models` | OpenAI 风格模型列表——本地、云端，以及启用时的 `auto` |
| GET | `/api/tags`、`/api/version` | Ollama 风格模型列表与版本 |
| GET | `/health` | 存活与已加载模型（公开） |
| GET | `/admin/status` | 模型信息、类别、运行时 `perf`、空闲时间 |
| POST | `/admin/load` | `{ "model": "path.gguf", "name": "alias", "n_gpu_layers": 999, "n_ctx": 4096, … }` |
| POST | `/admin/unload` | `{ "model": "alias" }`，或 `{}` 表示全部 |
| POST | `/admin/fit` | 适配检查点，或下载前的 `size_bytes` |
| GET, POST | `/admin/providers` | 列出与配置云端服务商；密钥绝不返回 |
| POST | `/admin/providers/test` | 拉取服务商模型列表以校验密钥 |
| GET, POST | `/admin/routing` | `auto` 策略：`{ "enabled": true, "prefer": "local" }` |
| POST | `/v1/chat/completions` | OpenAI 对话；`"stream": true` 为 Server-Sent Events |
| POST | `/api/chat`、`/api/generate` | Ollama 对话与单提示；流式为 NDJSON |
| POST | `/v1/embeddings`、`/api/embed`、`/api/embeddings` | OpenAI 与 Ollama 形态的嵌入 |
| POST | `/v1/rerank`、`/rerank`、`/v1/reranking`、`/reranking` | `{ "model": "…", "query": "…", "documents": ["…"], "top_n": 3 }` |

Ollama 的 `options` 对象映射到采样器：`temperature`、`top_k`、`top_p`、`min_p`、
`repeat_penalty`、`repeat_last_n`、`num_predict`（生成 token 上限；`<= 0` 表示"直到模型停止"）与 `seed`。响应携带惯常的 `prompt_eval_count` 与 `eval_count`，每条流式行携带 `created_at`。

`GET /docs` 将规范渲染为无外部脚本或样式的交互页面：在该浏览器中保留的 token 字段、由规范示例播种的每操作**发送**控件、逐块消费的流式路由，以及命名了实际已加载模型的示例。测试会遍历规范，若某文档化路由缺失、某 schema 引用未定义、或 `/docs` 流程在真实浏览器中退化，则失败。

### 配置

默认即推荐值；每个选项也可在设置与按模型**高级**面板中找到。`POST /admin/load` 接受：

| 分组 | 参数 |
|---|---|
| 模型 | `n_gpu_layers`（999 = 全部）、`main_gpu`、`load_mode`（`auto｜none｜mmap｜mlock｜mmap_mlock｜dio`）、`use_mlock` |
| 上下文 | `n_ctx`（0 = 自动，按机器定大小）、`n_batch`（2048）、`n_ubatch`（512）、`flash_attn`（`auto｜on｜off`）、`offload_kqv`、`rope_freq_base`、`rope_freq_scale` |
| CPU | `threads`、`threads_batch`（0 = 自动） |
| 适配器 | `load_mtp`（内置 MTP 头）、`lora`（适配器路径列表） |
| 采样（按请求） | `temperature`、`top_k`、`top_p`、`min_p`、`repeat_penalty`、`repeat_last_n`、`repeat_freq`、`repeat_present`、`seed`、`max_tokens` |
| 投机（按请求） | `speculative`、`ngram_n`、`ngram_m` |

守护进程以相同选项作为命令行标志暴露（`--load-mode`、`--flash-attn`、`--n-batch`、
`--no-offload-kqv` …），便于无界面与容器使用。

### 网络暴露与安全

守护进程默认绑定 `127.0.0.1`，因此机器之外无法触及。设置 → *网络* 可放宽到 `0.0.0.0`，
但这需要 bearer token：

```powershell
infer-daemon.exe --host 0.0.0.0 --port 15656 --api-key <token> --model model.gguf
```

服务器没有账号、没有限流、没有配额；任何能触及该端口的人都可以把模型加载到机器 GPU 上并消耗电费。token 以常量时间比较，且除 `/health` 外每条路由都要求它——`/health` 保持公开用于就绪探针，只暴露已加载模型名。绑定到可路由地址却**不带** token 会被拒绝：

```
refusing to listen on 0.0.0.0 without --api-key: anyone who can reach this port
can load models and spend GPU time on them. Pass --api-key <token>, bind to
127.0.0.1, or pass --allow-unauthenticated if something else guards the port
```

`--allow-unauthenticated` 用于自行做认证的容器或反向代理；空的 `--api-key ""` 是错误而非静默关闭认证的方式。应用把 token 存于操作系统凭据库，并通过 `NEXLLAMA_API_KEY` 传给守护进程，因此它既不出现在设置文件旁，也不出现在进程命令行中。跨域请求只对应用自身的源应答；其他源用 `--cors-origin <origin>` 添加。

图像输入仅限于 `data:` URL，且应用的能力清单除 `opener:default` 外不授予任何权限。

---

## 模型下载

搜索不受限制；下载页提供覆盖本地值得运行几乎一切的三种目录作为一键过滤：

- **bartowski**——最广泛的目录，一个模型覆盖多种量化
- **Unsloth**——动态量化，以及携带 MTP 所需头的发布
- **z-lab**——DFlash 草稿模型（UCSD 的 Z-Lab，提出并主导该方法）

量化是门手艺，糟糕的量化会损失精度，再多上下文也补不回来；Hub 上的其他内容仍可获取。下载可断点续传，并在传输开始前按空闲内存预检。客户端默认 `https://hf-mirror.com`；用 `HF_ENDPOINT` 覆盖，用 `HF_TOKEN` 认证受限仓库。

## 后端更新

设置 → *后端更新* 接受清单 URL。应用检查清单、下载匹配的后端归档、校验其 SHA-256、
替换 `%APPDATA%/com.nexllama.desktop/backends/<name>/` 并重启守护进程。更新后的后端优先于内置者。

清单必须**签名**：对其版本、llama.cpp 构建以及每个后端的 URL 与哈希做 Ed25519 签名。
仅校验和只能证明归档匹配命名它的清单，而二者来自同一次请求，因此仅靠哈希校验会让任何能服务其一者也能服务另一个。公钥取自设置或 `NEXLLAMA_UPDATE_KEY`；不编译进任何密钥，因为一个信任任意密钥的更新器什么都没验证。抓取仅限 HTTPS（含重定向后），低于已装版本的版本被拒绝，且清单、归档与解包总量都有大小上限，畸形发布无法耗尽磁盘。

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

界面通过本地 HTTP 驱动守护进程。`infer-core` 在每模型专属的工作线程上执行 llama.cpp 并流式输出解码后的 UTF-8；云端流量异步代理，绝不进入该线程。

---

## 故障排查

**安装程序被 SmartScreen 拦截，或 macOS 拒绝打开应用。** 默认构建不签名。签署并公证 macOS 构建需要 Apple Developer 账号及 `APPLE_SIGNING_IDENTITY`、`APPLE_CERTIFICATE`、`APPLE_CERTIFICATE_PASSWORD`、`APPLE_ID`、`APPLE_PASSWORD`、`APPLE_TEAM_ID`，打包步骤在它们存在时使用。

**选错了后端。** 设置 → GPU 报告检测到的适配器、将使用的后端与正在使用的后端，并可按需安装缺失后端。可用 Auto / CUDA / Vulkan / CPU 覆盖选择；无 NVIDIA 适配器的机器回退到 CPU，而非启动一个无法工作的 CUDA 构建。

**模型加载不了。** 对它运行**检查适配**：它会报告所需内存相对空闲内存，并建议层数、上下文长度或更小的量化。`/admin/status` 携带拒绝原因。

**模型从库中消失了。** 它在配置文件夹之外。设置 → *模型文件夹* 显示正在使用的路径，*打开文件夹* 可打开它。

**移动文件后 `model not found`。** 把文件夹指向新位置。文件被移动的已记忆模型在启动时跳过而非弄垮它，库里列出实际存在的内容。

**网络限制或 crates.io 慢。** 在 `~/.cargo/config.toml` 配置镜像：

```toml
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
```

---

## 当前说明

- 加载草稿模型或 MTP 头时，前缀缓存会被禁用，因为这些路径与目标各自独立地追踪位置；此类模型每轮重新编码对话。
- 硬件加速由 llama.cpp 提供的后端给出——CUDA、Vulkan 与 Metal。CANN、MUSA 等厂商专用运行库当前不支持。
- 云端请求会将其内容传送给命名的服务商。

## 贡献

欢迎 Issue 与 Pull Request。打包所用的同一套脚本也负责构建与测试工作区：

```powershell
.\scripts\build.ps1 fmt --all
.\scripts\build.ps1 clippy --all-targets
.\scripts\build.ps1 test --workspace
```

贡献内容以 MIT 许可证授权。

## 社区

- 🐛 [Issues](issues)——缺陷反馈与功能需求
- 💡 [Discussions](discussions)——使用问题、设计讨论与成果展示

## 许可证

根据 MIT 许可证授权（[LICENSE](LICENSE)）。

除另有明示外，本项目中有意提交的贡献内容，均按上述许可授权，不附加其他条款。第三方组件——其中包括 llama.cpp 与 GGML——列于
[NOTICE](NOTICE)。
