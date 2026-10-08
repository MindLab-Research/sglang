# 报告：客户端 TTFT 高（10–70s）的根因 —— 服务端 TTFT 正常，瓶颈在信道【上传方向】

日期：2026-10-07 · 环境：1101（8×B300，GLM-5.3-Flash，PD 分离 4+4）
分支：`pr17` · 仓库：`MindLab-Research/sglang-private`（**私有仓库，非 fork**）

---

## 1. 现象

客户端以 SSE 测得 TTFT **10s / 19s / 23s / 47s**，均出现在 **200k–280k prompt token**、
**缓存命中 95–99.5%** 的请求上；而**全冷的 200k 请求（命中 0）反而更快**。
本地直连服务端测同一形状请求，TTFT 只有 **0.5s**。

## 2. 方法（关键：在真实入口埋点）

在**真实入口**部署 SSE 反向代理并逐条记录：

```
客户端 ──SSH 反向隧道（sshd-session）──► 127.0.0.1:31300 【SSE 反代】──► 127.0.0.1:31400 (router)
                                                     └─ 逐条记录 TTFT / TPOT / 上传耗时 / body 成分
```

> 注意：隧道是**直连 31300**（不是 13100 上的 socat），因此反代必须占用 **31300**，router 让位到 31400。

反代每条记录下列字段（写入 `proxy_ttft.jsonl` / `proxy_ttft.log`）：

| 字段 | 含义 |
|---|---|
| `body_bytes` | 请求体字节数 |
| `text_bytes` / `n_img` / `img_bytes` | body 成分：文本字节 / 图片个数 / 图片字节（base64 或 URL） |
| `req_read_s` | **连接建立 → 请求体收完**（= 客户端上传耗时，经隧道） |
| `first_pkt_s` | 请求体收完 → **服务端发出第一个 SSE 包** |
| `ttft_s` | 请求体收完 → 首个 token（含 `reasoning_content`） |
| `reason_ttft_s` / `content_ttft_s` | 分别量测思维链首 token 与正文首 token |
| `tpot_ms` / `out_tokens` / `prompt_tokens` / `cached_tokens` | 常规 |

> ⚠️ 必须解析 `delta.reasoning_content`。只认 `delta.content` 会把**思维链跑完的时刻**误当 TTFT
> （曾据此误报「服务端 6.9s / 14.4s」，实为 reasoning 时长，已撤回）。

## 3. 证据

### 3.1 逐条分解（用户真实流量）

| 完成时刻 | body | **上传 `req_read_s`** | 上传速率 | **服务端 `first_pkt_s`** | 连接→首包 |
|---|---|---|---|---|---|
| 20:30:14 | 2.68 MB | 2.82s | 0.95 MB/s | 3.22s | 6.04s |
| 20:31:03 | 2.69 MB | **51.42s** | 52 KB/s | 3.34s | 54.76s |
| 20:31:33 | **0.73 MB** | **69.47s** | **10 KB/s** | 0.93s | 70.41s |
| **20:32:06** | **12.43 MB** | **43.74s** | 284 KB/s | **3.51s** | **47.25s** |
| 20:27:05 | — | — | — | 3.52s | — |

**⇒ 客户端 47s ≈ 43.74s 上传 + 3.51s 服务端。**

### 3.2 上传速率与体积无关 ⇒ 信道抖动，不是请求大小

0.73 MB 传 **69.5s**（10 KB/s），2.68 MB 只传 **2.8s**（0.95 MB/s）——**相差 95 倍**。

### 3.3 并发互相挤占（多模态大 body 是驱动因素）

倒推每条的连接建立时刻（`accept ≈ 完成时刻 − req_read_s − total_s`）：

| 完成时刻 | body | req_read_s | total_s | **倒推 accept** |
|---|---|---|---|---|
| 20:31:03.831 | 2.69 MB | 51.42s | 8.58s | **≈20:30:03.8** |
| 20:31:33.011 | 0.73 MB | 69.47s | 19.70s | **≈20:30:03.8** |
| 20:32:06.456 | 12.43 MB | 43.74s | 6.85s | ≈20:31:15.9 |

两条**同一瞬间发起**：0.73 MB（纯文本）本该 ~1s 传完，却用了 **69.5s**——因为它
**在与 2.69 MB 那条争抢同一条隧道的上传带宽**；随后 12.43 MB 继续挤占。
⇒ **大体积多模态 body 占满上传管道，并把并发请求一并拖慢。**

### 3.4 体积放大（多模态）

273,226 prompt token 的**纯文本**约 1.2 MB（≈4.4 B/token），实测 body **12.43 MB**（≈45 B/token）
⇒ **放大 ≈10×**，含大块非文本负载（图片 base64）。`analyze_body` 单测：2.67 MB 多模态 body →
`img_bytes=2,666,695 / text_bytes=1,514 / n_img=1`；纯文本对照 `img_bytes=0`。

### 3.5 反代与服务端无问题（对照组）

| 路径 | 首包 | 首 token |
|---|---|---|
| 直连 router :31400 | 0.522s / 0.552s | 0.522s / 0.552s |
| **经反代 :31300** | **0.510s / 0.539s** | 0.510s / 0.539s |

⇒ **反代零额外延迟**；服务端真 TTFT **0.5–3.5s**，健康。

## 4. 结论

1. 客户端 TTFT ≈ **上传耗时（经隧道）+ 服务端 TTFT**；服务端仅占 **0.5–3.5s**，
   其余（最大 ~70s）**全部发生在客户端→机器的上传方向**。
2. 上传耗时的**驱动因素 = 多模态 body 体积 + 并发挤占**（体积大 → 占满管道 → 拖慢并发），
   **不是**纯文本长度，也**不是** sglang 内部。
3. 若按「首个 `content` 非空」计时，还会把**思维链时长**计入；应按**首个 SSE 包 / 首个
   `reasoning_content`** 计时。

## 5. 修复方向（信道 / 客户端侧；sglang 无需改动）

1. **压图**：降分辨率/质量，禁用无压缩 PNG；或用 **URL 让服务端拉取**，不塞 base64
   （base64 额外 +33% 体积）。
2. **避免重复回传历史图片**：多轮对话每轮重发整段历史含图 ⇒ 体积随轮次线性膨胀（已达 12.4 MB）。
3. **隧道**：SSH 开压缩（`-C`）或改用带宽更稳的通道；排查 RTT / 丢包 / 带宽抖动。
4. **度量口径**：客户端把「上传完成」与「首 token」分开计时，看板 TTFT 取首个 SSE 包
   （或首个 reasoning token），不要取首个 `content`。

## 6. 本分支附带的服务端通用修复（与本次 TTFT 问题独立，但同批提交）

两处修复均位于**模型无关的共享代码**，因此 **DeepSeek（dpsk）与 GLM-5.3-Flash 同时受益**：

1. **MoE-LoRA 按「适配器是否真含专家权重」选路**（`lora/lora.py`、`lora/utils.py`、
   `lora/lora_manager.py`、`lora/layers.py`、`models/deepseek_v2.py`、
   `model_executor/runner_utils/capture_mode.py`、`model_executor/runner/decode_cuda_graph_runner.py`）：
   原判据是「batch 内有无 LoRA」，导致**纯注意力适配器（甚至全零适配器）也被迫走 decomposed MoE 路径**；
   改为看适配器是否命中 `\.experts\.`，并新增第三种 CUDA-graph 变体 `lora_nomoe`。
   实测：**attn-only 3491 → 4156 tok/s（+19.0%），TPOT 16.46 → 14.01 ms**；MoE 组不变。
2. **`hiradix_cache.load_back` 有界恢复**（`mem_cache/hiradix_cache.py`）：
   原「全有或全无 + 失败后 `evict(len(host_indices))` 无界重试」改为仅当
   `need <= evictable_size_` 才驱逐恢复（否则快速失败交冷 prefill），并加 `[load_back]` 耗时告警。
   实测未破坏 HiCache（819k 命中回载 TTFT 1.81s；190k 命中 0.58s）。
