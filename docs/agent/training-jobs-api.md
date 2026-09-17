# smg 训练任务接口使用文档（v1）

> B300 bf16 PD 集群 · smg router 训练任务提交/轮询/下载接口
> 版本：2026-08-21 · 适用二进制：smg（含 Training job manager, concurrency=64）

---

## 1. 概述

训练侧通过 **control 接口**批量提交推理任务，异步执行后轮询状态、下载结果。每个任务返回：

- **完整输出文本**（output_text）
- **token id 数组**（output_ids）——模型实际生成的 token 序列
- **逐 token logprob**（output_token_logprobs / output_logprob_entries）——与 token id 逐位对齐

**执行通道**：内部走 smg 自身 `/generate`（sglang 原生 API），PD 双调度（bootstrap→prefill→decode），带 LoRA 时按任务透传 `lora_path`。

### 基础信息

| 项 | 值 |
|---|---|
| 公网入口 | `http://8.213.214.14:18888` |
| 认证 | `Authorization: Bearer sk-control-pd-2026` |
| 路径前缀 | `/v1/control/jobs` |
| 并发上限 | 64（同 job 的任务共享，多个 job 同时提交也在 64 内排队） |
| 请求超时 | **仅空闲超时**（env `SMG_JOBS_REQUEST_TIMEOUT_SECS` 可调，默认 3600s，建议 7200+）：引擎每吐一段新内容就重置计时器，**只要持续出内容就永不超时**；仅当整整该时长无任何数据才判定上游卡死断开。**无固定 wall-clock 总 deadline** |
| 结果保留 | **48 小时**（磁盘持久化，router 重启自动恢复）。超过保留期的**终态** job（completed/failed/cancelled/partial）会被后台 GC 删除；`queued`/`running` 永不因超龄被删。env `SMG_JOBS_RETENTION_HOURS` 可调（小数小时可用；**0 = 关闭 GC**），扫描间隔 `SMG_JOBS_GC_INTERVAL_SECS`（默认 1800s） |
| 单 job 任务数上限 | 4096 |

---

## 2. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/control/jobs` | 提交任务（数组或对象包裹） |
| GET | `/v1/control/jobs` | 列出所有 job（简况） |
| GET | `/v1/control/jobs/{job_id}` | 轮询单个 job 状态（含实时进度 token 数） |
| GET | `/v1/control/jobs/{job_id}/result` | 下载整个 job 结果（终态/已取消均可） |
| GET | `/v1/control/jobs/{job_id}/tasks/{task_id}/result` | 下载单个任务结果（取消任务返回空 samples） |
| POST | `/v1/control/jobs/{job_id}/cancel` | 取消 job（排队任务立即 `cancelled`，运行中任务保留已生成 token） |
| DELETE | `/v1/control/jobs/{job_id}` | 删除 job（内存+磁盘） |

所有接口都需要 Bearer 认证（control key）。

---

## 3. 提交任务

### 3.1 请求体格式

**格式 A：裸数组**（推荐，即训练侧现有的 gateway-req 格式数组）

```json
[
  {
    "model": "zai-org/GLM-5.2",
    "prompt": "Solve the problem step by step...",
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "n": 1,
    "stream": true,
    "stream_options": {"include_usage": true},
    "logprobs": 1,
    "lora_path": "/root/glm52_local/loras/L0"
  }
]
```

**格式 B：对象包裹**（job 级默认 LoRA，任务可单独覆盖）

```json
{
  "lora_path": "/root/glm52_local/loras/L0",
  "requests": [
    {"prompt": "...", "max_tokens": 4096},
    {"prompt": "...", "max_tokens": 4096, "lora_path": "/root/glm52_local/loras/L2"},
    {"prompt": "...", "max_tokens": 4096, "lora_path": null}
  ]
}
```

### 3.2 字段说明

> **输入三选一**：`prompt`（文本）/ `input_ids`（token）/ `continue_from`（引用已有任务续跑）。**必须且只能给一个**，否则 400。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | string | ✅（三选一） | 生成提示词（空串/纯空白会被拒绝） |
| `input_ids` | int[] | ✅（三选一） | **token 级输入**，替代 `prompt`（引擎不再做 tokenize，序列逐位确定） |
| `continue_from` | object | ✅（三选一） | **续跑引用**，服务端自己拼输入，客户端不需要回传任何 token，见 §10 |
| `max_tokens` | int | ❌ | 最大生成 token 数，默认 4096 |
| `temperature` | float | ❌ | 采样温度，默认服务端值（续跑任务省略时继承源任务） |
| `top_p` | float | ❌ | nucleus 采样，默认服务端值（续跑任务省略时继承源任务） |
| `n` | int | ❌ | 采样次数（1-64），默认 1；结果为数组 `samples[n]` |
| `lora_path` | string\|null | ❌ | **LoRA adapter 路径；不传=基模**。job 级设置可被任务级覆盖，任务级 `null` 强制基模；续跑任务省略时继承源任务 |
| `model` | string | ❌ | 兼容字段（接受训练侧模型名，如 `zai-org/GLM-5.2`，单模型部署忽略） |
| `stream` / `stream_options` / `logprobs` | any | ❌ | 兼容字段（内部固定流式执行 + 返回 logprob，无需设置） |

### 3.3 提交响应

```json
{
  "job_id": "job_1787282349702_001_6026",
  "status_url": "/v1/control/jobs/job_1787282349702_001_6026",
  "result_url": "/v1/control/jobs/job_1787282349702_001_6026/result",
  "task_count": 1,
  "tasks": [
    {
      "task_id": "job_1787282349702_001_6026_t0000",
      "index": 0,
      "status": "queued",
      "lora_path": null,
      "prompt_chars": 2366
    }
  ]
}
```

HTTP 状态码：`202 Accepted`（成功）；`400`（JSON 解析失败/空数组/空 prompt/超限）；`401`（认证失败）。

### 3.4 提交示例（curl）

```bash
curl -X POST http://8.213.214.14:18888/v1/control/jobs \
  -H 'Authorization: Bearer sk-control-pd-2026' \
  -H 'Content-Type: application/json' \
  -d '[
    {
      "model": "zai-org/GLM-5.2",
      "prompt": "Solve the problem step by step. ...\nAnswer:",
      "max_tokens": 4096,
      "temperature": 0.7,
      "top_p": 0.9,
      "n": 1,
      "lora_path": "/root/glm52_local/loras/L0"
    }
  ]'
```

从文件提交（推荐，长 prompt 友好）：

```bash
# 单任务文件（gateway-req-02.json 格式）
jq -s '.' gateway-req-02.json > body.json   # 包成数组，或直接:
curl -X POST ... -d @<(echo "[$(cat gateway-req-02.json)]")

# 多任务文件
curl -X POST ... -d @tasks.json
```

---

## 4. 轮询状态

```bash
curl -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.214.14:18888/v1/control/jobs/{job_id}
```

### 响应

```json
{
  "job_id": "job_1787282349702_001_6026",
  "status": "running",
  "progress": {"done": 3, "failed": 0, "cancelled": 0, "running": 1, "queued": 6, "total": 10},
  "total_output_tokens": 8231,
  "tasks": [
    {"task_id": "..._t0000", "index": 0, "status": "completed", "token_count": 2048, "error": null},
    {"task_id": "..._t0001", "index": 1, "status": "running",   "token_count": 512,   "error": null},
    {"task_id": "..._t0002", "index": 2, "status": "failed",    "token_count": 0,     "error": "generate returned 400: ..."}
  ]
}
```

### 状态语义

| job status | 含义 |
|---|---|
| `queued` | 已接收，未开始（并发满时排队） |
| `running` | 至少一个任务执行中 |
| `completed` | 全部任务成功 |
| `partial` | 部分成功部分失败/部分取消 |
| `failed` | 全部失败 |
| `cancelled` | 全部任务被取消（`POST .../cancel` 后） |

task status：`queued` / `running` / `completed` / `failed` / `cancelled`。

- `token_count`：已完成任务为结果累计 token 数；**运行中任务为实时已生成 token 数**（后端 SSE 增量更新），可用于进度展示
- `total_output_tokens`：所有已完成/取消任务的结果 token + 运行中任务实时 token 之和

**轮询建议**：每 5-10 秒一次；`status` 进入终态（completed/partial/failed/cancelled）后即可下载。

---

## 4.5 取消任务

```bash
curl -X POST -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.214.14:18888/v1/control/jobs/{job_id}/cancel
```

**语义**：
- 排队中（`queued`）的任务立即置为 `cancelled`（从未开始，无结果文件）
- 运行中（`running`）的任务在下一个 SSE chunk 边界停止，**已生成的 token ids + logprobs 会被保留并持久化**（结果结构与完成任务的完全一致，`finish_reason: "cancelled"`），可正常下载
- 已完成/失败的任务不受影响
- 取消后 job 状态为 `cancelled` 或 `partial`（视是否全部取消），result 端点可正常下载

响应：

```json
{
  "job_id": "job_1787282349702_001_6026",
  "status": "running",
  "running_cancelled": 1
}
```

（`running_cancelled` = 本次取消动作所标记的运行中任务数；轮询 status 确认收敛到终态。）

---

## 5. 下载结果

### 5.1 整个 job

```bash
curl -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.214.14:18888/v1/control/jobs/{job_id}/result -o result.json
```

**任何时候都可以下载**（运行中也可以）：

- 已完成/已取消的任务 → 给出**持久化结果**；
- 运行中的任务 → 给出**内存里的实时 partial 快照**（截至当前已生成的 `output_text` + `output_ids` + 逐 token logprobs），可用它做 partial rollout 的"先训已生成 prefix"；
- 响应里的 `status` 字段标明这批结果是终态（`completed` / `partial` / `cancelled` / `failed`）还是仍在跑（`running` / `queued`）。尚无任何 token 的运行中任务不会出现在 `results` 里。

⚠️ 返回 `409 Conflict` 的是**单个任务**端点（下面 5.2）：该任务未完成时才会 409。**整个 job 的 result 不再 409。**

### 5.2 单个任务

```bash
curl -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.214.14:18888/v1/control/jobs/{job_id}/tasks/{task_id}/result -o task.json
```

任务未完成时返回 `409 Conflict`（提示先轮询 status）；已取消的任务返回空 `samples`。

### 5.3 结果结构（核心）

```json
{
  "job_id": "job_..._...",
  "status": "completed",
  "task_count": 1,
  "results": [
    {
      "task_id": "job_..._t0000",
      "index": 0,
      "samples": [
        {
          "output_text": "We are given a problem about a mouse...",
          "output_ids": [98314, 100438, 98386, "..."],
          "output_token_logprobs": [-0.565, -4.562, -0.0007, "..."],
          "output_logprob_entries": [
            [-0.5655841827392578, 98314, null],
            [-4.56245231628418,   100438, null],
            [-0.0006675875629298389, 98386, null]
          ],
          "output_top_logprobs": [null, null, null],
          "finish_reason": "length",
          "prompt_tokens": 516,
          "completion_tokens": 4096
        }
      ]
    }
  ]
}
```

### 5.4 字段对齐保证（重要）

对每个 sample，以下四个数组**逐位 1:1 对齐**（长度恒相等）：

```
output_ids[i]               第 i 个生成 token 的 id
output_token_logprobs[i]    该 token 的 logprob（float）
output_logprob_entries[i]   原始三元组 [logprob, token_id, top_logprobs|null]
output_top_logprobs[i]      top-k logprob 详情（请求 top-k>1 时非 null）
```

且 `len(output_ids) == completion_tokens`。

**消费建议**：训练侧直接使用 `output_ids`（token id）与 `output_token_logprobs` 做对数似然计算——**不要**对 `output_text` 重新 encode（BPE 合并会导致与生成 token 序列不一致，例如此前 60000 vs 59986 的 14 token 差异即源于此）。

---

### 5.5 压缩传输（推荐，2026-08-22 起）

result 端点支持按 `Accept-Encoding` 自动协商压缩（仅 jobs 控制面路由，推理路径不受影响）：

| Accept-Encoding | 响应 Content-Encoding | 32.7MB result 实测 |
|---|---|---|
| `gzip` | `gzip` | ~6.6MB（20.3%，降 5 倍） |
| `zstd` | `zstd` | 更小（~17%） |
| 不发 | 不压缩（原始） | 32.7MB |

Python `requests` / curl 默认自动带 gzip（`requests` 发 `Accept-Encoding: gzip, deflate`，自动解压；curl 加 `--compressed`）。**弱网/跨境链路下务必让 driver 带上压缩**（曾出现 32.7MB 原始传输在 18KB/s 链路上滴流 30 分钟）。

## 6. 完整调用流程示例（Python）

```python
import json, time, requests

BASE = "http://8.213.214.14:18888"
HEADERS = {"Authorization": "Bearer sk-control-pd-2026", "Content-Type": "application/json"}

# 1) 提交：任务数组；lora_path 可选（不传=基模）
tasks = [
    {"model": "zai-org/GLM-5.2",
     "prompt": "Solve the problem step by step. ...\nAnswer:",
     "max_tokens": 4096, "temperature": 0.7, "top_p": 0.9, "n": 1,
     "lora_path": "/root/glm52_local/loras/L0"},          # 删掉此行即用基模
]
r = requests.post(f"{BASE}/v1/control/jobs", headers=HEADERS,
                  json=tasks, timeout=60)
r.raise_for_status()
job_id = r.json()["job_id"]
print("job:", job_id)

# 2) 轮询
while True:
    st = requests.get(f"{BASE}/v1/control/jobs/{job_id}",
                      headers=HEADERS, timeout=30).json()
    print(st["status"], st["progress"])
    if st["status"] in ("completed", "partial", "failed"):
        break
    time.sleep(10)

# 3) 下载
result = requests.get(f"{BASE}/v1/control/jobs/{job_id}/result",
                      headers=HEADERS, timeout=300).json()

for tr in result["results"]:
    for s in tr["samples"]:
        assert len(s["output_ids"]) == len(s["output_token_logprobs"]) \
               == len(s["output_logprob_entries"]) == s["completion_tokens"]
        ids   = s["output_ids"]                # token id 序列
        lps   = s["output_token_logprobs"]     # 逐位对齐的 logprob
        text  = s["output_text"]               # 仅作参考/展示

# 4)（可选）清理
requests.delete(f"{BASE}/v1/control/jobs/{job_id}", headers=HEADERS)
```

---

## 7. 错误处理

| 场景 | 表现 | 处理 |
|---|---|---|
| 认证失败 | `401` | 检查 Bearer key |
| body 非法 | `400` + `{"error": "invalid task at index N: ..."}` | 按提示修正对应任务 |
| job 未完成下载 | `409` | 先轮询到终态 |
| 单任务执行失败 | task `status=failed` + `error` 字段 | 不影响同 job 其他任务（job 变 `partial`）；失败任务无 result |
| n>1 中途失败 | 已完成 samples 保留，job 正常 completed | 检查 error 日志 |
| router 重启 | running/queued 任务标记 `failed`（"interrupted by router restart"），completed 结果保留 | 重新提交失败部分 |
| SSE 中断（已有部分数据） | 保留已生成部分并 completed | 如需严格完整可检查 finish_reason |
| 取消运行中任务 | task `status=cancelled`，已生成 tokens/logprobs 保留（`finish_reason: "cancelled"`） | 结果结构与 completed 完全一致，可正常下载 |
| 取消排队任务 | task `status=cancelled`，从未开始 | 下载单任务返回空 samples 数组（200），格式与完成一致 |

**失败任务重试**：当前无自动重试；重新提交仅含失败任务的子数组即可（prompt 原样重用）。

---

## 8. 运维备注（服务端）

- 结果落盘：`/root/smg-jobs/{job_id}/`（`job.json` + `task_XXXX.json`，原子写）；router 启动时自动恢复
- 相关环境变量（启动 router 时）：`SMG_JOBS_DIR`（默认 ./smg-jobs）、`SMG_JOBS_MAX_CONCURRENCY`（默认 64）、`SMG_JOBS_REQUEST_TIMEOUT_SECS`（默认 3600）、`SMG_JOBS_SELF_URL`（默认 `http://127.0.0.1:{port}`）、**`SMG_JOBS_ENGINE_URLS`**（逗号分隔的引擎地址，`continue_from` 续跑把源 prompt 还原成 token id 时用；不设则回退到 router 启动参数里的 worker URL，见 §10）、**`SMG_JOBS_RETENTION_HOURS`（默认 48，0=关闭 GC）**、**`SMG_JOBS_GC_INTERVAL_SECS`（默认 1800）**
- **本地 GC（2026-09-15 加入）**：后台任务定期扫描 `SMG_JOBS_DIR`，把**最后活动时间**超过保留期的 job 目录删掉（`job.json` 每次状态变更都会重写、`task_*.json` 随任务完成落盘，所以该时间即真实进度）。判据：只删终态；`running`/`queued` 一律保留；删除会打日志（含释放体积），例如
  `jobs: gc removed job_xxx (idle 52.1h > retention 48.0h, freed 91.2 MB)`。
  启动时也会立即扫一次（把停机期间积压的旧数据回收）。
- 当前部署命令（B300-1）：
  ```bash
  SMG_JOBS_DIR=/root/smg-jobs setsid nohup /usr/local/bin/smg launch \
    --pd-disaggregation \
    --prefill http://10.0.0.75:30100 8998 \
    --decode http://10.0.0.67:30200 \
    --port 31000 --model-path /root/glm52_local/bf16 \
    --api-key sk-glm52-pd \
    --control-plane-api-keys '1:admin:admin:sk-control-pd-2026' \
    > /root/smg_router.log 2>&1 &
  ```
- 日志：`/root/smg_router.log`（`jobs:` 前缀行为训练任务相关）
- 代码位置：`sgl-model-gateway/src/control_plane/jobs.rs`

---

## 9. 验收记录（2026-08-21）

| 验证项 | 结果 |
|---|---|
| 小样例（48 tokens） | ids=48 / logprobs=48 / entries=48 完全对齐 ✅ |
| 真实样例（gateway-req-02.json，4096 tokens，62s） | ids=4096 / logprobs=4096 / entries=4096，completion_tokens=4096，三元组 token_id 逐位吻合 ✅ |
| token id 还原 | bf16 tokenizer decode(output_ids) 与 output_text 逐位一致 ✅ |
| 认证 | 无 key 401 / control key 200 ✅ |

---

## 10. 续跑：cancel → continue（2026-09-14 新增）

> **结果方向不变**：每个任务的 `output_text` / `output_ids` / 逐 token logprob **照常回传**（那是训练数据本身）。
> 本节只解决**请求方向**的体积问题：客户端要"接着往下生成"时，**不需要把之前的 token 回传**，只发一个引用即可。

### 10.1 为什么需要它

- cancel 是**终结语义**：引擎侧 abort → rid 销毁、KV 释放；sglang **没有 un-cancel / 恢复同一 rid** 的接口。所以"续跑"必然是**新请求**重走 prefill。
- 那次的正确姿势是：`新输入 = 原来的输入 token ++ 已经生成的 token`（token 级拼接，**不能拼文本** —— 重新 tokenize 会在 thinking 段/特殊 token 边界漂移一两个 token，续跑就不是同一条轨迹了）。
- 这些 token 客户端本来就有（`output_ids` + 它自己发过的 prompt），但**让客户端回传**意味着把几万～几十万 token 塞进请求体（体积大、还要自己保证逐位正确）。本特性把这件事**移到服务端**：客户端只发 `job_id`（可再指定 task / sample）。

### 10.2 用法 A：整个 job 续跑（最小请求体）

```bash
curl -X POST http://8.213.214.14:18888/v1/control/jobs \
  -H 'Authorization: Bearer <control-key>' -H 'Content-Type: application/json' \
  -d '{"continue_from": {"job_id": "job_1787282349702_001_6026"}, "max_tokens": 512}'
```

- 新 job 与源 job **任务数一一对应**（顺序一致），第 i 个任务续跑源 job 的第 i 个任务；
- `max_tokens` / `temperature` / `top_p` / `lora_path` / `n` 是本级覆盖项；`temperature`/`top_p`/`lora_path` 省略时**继承源任务**（同 adapter、同采样），`max_tokens` 不继承（新预算是新请求的事）；
- `{"continue_from": {...}, "requests": [...]}` 同时出现 → 400（语义冲突）。

### 10.3 用法 B：任务级续跑

```json
[
  {"continue_from": {"job_id": "job_x", "task_id": "job_x_t0000"}, "max_tokens": 512},
  {"continue_from": {"job_id": "job_x", "index": 1, "sample_index": 1}, "temperature": 0.0}
]
```

| `continue_from` 字段 | 必填 | 说明 |
|---|---|---|
| `job_id` | ✅ | 源 job |
| `task_id` | ❌ | 源任务 id（全局唯一，提交响应里给的就是它） |
| `index` | ❌ | 源任务序号（`task_id` 的替代写法） |
| `sample_index` | ❌ | 从第几个 sample 续（`n>1` 时用），默认 0 |

- 源 job 只有 1 个任务时可以都不写；
- 源 job 有多个任务且不指定 → 400（提示补 `task_id`/`index`）。

### 10.4 用法 C：显式 token（仍支持）

客户端自己拼好整段时照旧可发 `input_ids`（与 `prompt` 二选一）：

```json
[{"input_ids": [128000, 9906, ...], "max_tokens": 512}]
```

`input_ids` 与 `continue_from` 的服务端语义完全一致（都走 native `/generate` 的 `input_ids` 路径），区别只是"谁来拼"。

### 10.5 语义与保真

| 项 | 行为 |
|---|---|
| 结果内容 | 续跑任务的结果**只含本次新生成的 token**（旧 token 客户端已经有），`samples[]` 结构不变 |
| 逐位一致 | 服务端拼的是 **token 数组**，不是文本 → 边界不发生重 tokenize |
| 续跑链 | "续跑的续跑"支持：组合后的输入会落盘 `input_XXXX.json`，下一跳直接复用，不依赖源任务是否还在动 |
| 历史数据 | **完全支持**：本特性上线前已落盘的 job/task 无需迁移即可作为源。文本 `prompt` 由引擎 tokenizer 现场还原为 id（与当初 `/generate` 同一 tokenizer 对象）；已完成的 `task_XXXX.json` 结果照旧参与拼接 |
| 运行中的源 | 源任务还在跑时也能续（取实时 partial 快照）；源任务一条都没生成时返回明确错误，不会静默产空样本 |

### 10.6 成本（重要）

- 原 prompt 部分：prefill 端命中 radix / HiCache 前缀缓存，基本不重算；
- **已生成的 token 必须重新 prefill**（它们从没进过 prefill 的缓存）；且 decode 端 `--disable-radix-cache`（红线）→ **整个上下文的 KV 都要从 prefill 传过去**，传输量 ∝ 上下文长度，与"续跑新增多少 token"无关。
- 结论：续几万 token 的 rollout 很划算；100k+ 上下文会明显（传输是大头）。

### 10.7 错误码

| 现象 | 原因 |
|---|---|
| 400 `task must provide exactly one of 'prompt', 'input_ids' or 'continue_from'` | 输入来源给了 0 个或 ≥2 个 |
| 400 `continue_from: unknown job_id 'xxx'` | 源 job 已被 delete（或超过 48h 保留期） |
| 400 `continue_from: job 'x' has N tasks — specify 'task_id' or 'index'` | 多任务 job 未指定源任务 |
| 任务级 `error`：`continue_from: task 'x' has no generated tokens yet` | 源任务还没生成任何 token |
| 任务级 `error`：`continue_from: task 'x' has no recoverable input tokens` | 源任务的组合输入既无 `input_ids` 也无 `input_XXXX.json`（异常场景） |
| 任务级 `error`：`no engine URL is configured` | router 未配置引擎地址：设置 `SMG_JOBS_ENGINE_URLS`（逗号分隔）或按 worker URL 启动 router |
