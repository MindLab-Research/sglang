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
| 公网入口 | `http://8.213.215.2:18888` |
| 认证 | `Authorization: Bearer sk-control-pd-2026` |
| 路径前缀 | `/v1/control/jobs` |
| 并发上限 | 64（同 job 的任务共享，多个 job 同时提交也在 64 内排队） |
| 单请求超时 | 3600 秒（env `SMG_JOBS_REQUEST_TIMEOUT_SECS` 可调） |
| 结果保留 | 48 小时（磁盘持久化，router 重启自动恢复） |
| 单 job 任务数上限 | 4096 |

---

## 2. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/control/jobs` | 提交任务（数组或对象包裹） |
| GET | `/v1/control/jobs` | 列出所有 job（简况） |
| GET | `/v1/control/jobs/{job_id}` | 轮询单个 job 状态 |
| GET | `/v1/control/jobs/{job_id}/result` | 下载整个 job 结果 |
| GET | `/v1/control/jobs/{job_id}/tasks/{task_id}/result` | 下载单个任务结果 |
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

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | string | ✅ | 生成提示词（空串/纯空白会被拒绝） |
| `max_tokens` | int | ❌ | 最大生成 token 数，默认 4096 |
| `temperature` | float | ❌ | 采样温度，默认服务端值 |
| `top_p` | float | ❌ | nucleus 采样，默认服务端值 |
| `n` | int | ❌ | 采样次数（1-64），默认 1；结果为数组 `samples[n]` |
| `lora_path` | string\|null | ❌ | **LoRA adapter 路径；不传=基模**。job 级设置可被任务级覆盖，任务级 `null` 强制基模 |
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
curl -X POST http://8.213.215.2:18888/v1/control/jobs \
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
  http://8.213.215.2:18888/v1/control/jobs/{job_id}
```

### 响应

```json
{
  "job_id": "job_1787282349702_001_6026",
  "status": "running",
  "progress": {"done": 3, "failed": 0, "total": 10},
  "total_output_tokens": 8231,
  "tasks": [
    {"task_id": "..._t0000", "index": 0, "status": "completed", "token_count": 2048, "error": null},
    {"task_id": "..._t0001", "index": 1, "status": "running",   "token_count": 0,    "error": null},
    {"task_id": "..._t0002", "index": 2, "status": "failed",    "token_count": 0,    "error": "generate returned 400: ..."}
  ]
}
```

### 状态语义

| job status | 含义 |
|---|---|
| `queued` | 已接收，未开始（并发满时排队） |
| `running` | 至少一个任务执行中 |
| `completed` | 全部任务成功 |
| `partial` | 部分成功部分失败 |
| `failed` | 全部失败 |

task status：`queued` / `running` / `completed` / `failed`。

**轮询建议**：每 5-10 秒一次；`status` 进入终态（completed/partial/failed）后即可下载。

---

## 5. 下载结果

### 5.1 整个 job

```bash
curl -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.215.2:18888/v1/control/jobs/{job_id}/result -o result.json
```

job 未完成时返回 `409 Conflict`（提示先轮询）。

### 5.2 单个任务

```bash
curl -H 'Authorization: Bearer sk-control-pd-2026' \
  http://8.213.215.2:18888/v1/control/jobs/{job_id}/tasks/{task_id}/result -o task.json
```

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

## 6. 完整调用流程示例（Python）

```python
import json, time, requests

BASE = "http://8.213.215.2:18888"
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

**失败任务重试**：当前无自动重试；重新提交仅含失败任务的子数组即可（prompt 原样重用）。

---

## 8. 运维备注（服务端）

- 结果落盘：`/root/smg-jobs/{job_id}/`（`job.json` + `task_XXXX.json`，原子写）；router 启动时自动恢复
- 相关环境变量（启动 router 时）：`SMG_JOBS_DIR`（默认 ./smg-jobs）、`SMG_JOBS_MAX_CONCURRENCY`（默认 64）、`SMG_JOBS_REQUEST_TIMEOUT_SECS`（默认 3600）、`SMG_JOBS_SELF_URL`（默认 `http://127.0.0.1:{port}`）
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
