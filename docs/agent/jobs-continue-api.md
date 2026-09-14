# 训练 Job 续写 API（`continue_from`）

> **一句话**：取消或截断后的 job 想接着生成，客户端**只发一个 job 引用**即可 —— 服务端自己把
> `源任务输入 token ++ 源任务已生成 token` 拼成上下文重走 prefill，**不需要客户端回传任何 token**。
>
> 适用对象：训练/RL 侧同学。实现见 `sgl-model-gateway/src/control_plane/jobs.rs`（`continue_from`
> / `resolve_input_ids` / `ensure_lora_loaded`）。

---

## 0. 两个方向，别搞混（最重要的一条）

| 方向 | 内容 | 本 API 是否改变 |
|---|---|---|
| **结果方向**（服务端 → 客户端） | `output_text` / `output_ids` / `output_token_logprobs` | **完全没有变**。训练数据一个字段都不少，照常返回。 |
| **请求方向**（客户端 → 服务端） | 续写时要让模型"接着上次往下写" | **这里省掉了 token 回传**：客户端给引用，服务端拼上下文。 |

即：**续写 ≠ 不给客户端 token**。客户端拿到的仍是完整 token + 逐 token logprob；省掉的只是"续写请求里"那份几万～几十万 token 的 JSON。

---

## 1. 30 秒上手

```bash
# ① 提交一个普通 job（老接口，不变）
curl -sS -X POST "$BASE/v1/control/jobs" \
  -H "Authorization: Bearer $CONTROL_KEY" -H 'Content-Type: application/json' \
  -d '[{"prompt":"用一句话说明什么是张量并行。","max_tokens":512,"temperature":0.0}]'
# → {"job_id":"job_1789375041513_000_7705","status":"queued",...}

# ② 结束（或先 cancel）后，直接续写：请求体里只有一个 job id
curl -sS -X POST "$BASE/v1/control/jobs" \
  -H "Authorization: Bearer $CONTROL_KEY" -H 'Content-Type: application/json' \
  -d '{"continue_from":{"job_id":"job_1789375041513_000_7705"},"max_tokens":512}'
```

服务端会构造 `input_ids = 源任务输入 token ++ 源任务已生成 token` 再发 `prefill → decode`。

> 实测（B300 1021/1022，GLM-5.3 MoL）：
> 源任务 `prompt_tokens=8`、产出 24 token → 续写任务 `prompt_tokens=32 == 8 + 24` ✅
> 源任务 `prompt_tokens=21`、产出 966 token → 续写任务 `prompt_tokens=987 == 21 + 966` ✅

---

## 2. 输入的三种形态（三选一，必须且只能给一个）

| 形态 | 字段 | 用途 |
|---|---|---|
| 文本 | `prompt` | 普通请求（原行为） |
| token | `input_ids` | 客户端自己拼好了整段 token |
| **引用** | **`continue_from`** | **本 API 主角**：服务端自己拼，客户端不回传 token |

三者同时给或都不给 → **400**，报文明确：
`task must provide exactly one of 'prompt', 'input_ids' or 'continue_from'`。

---

## 3. `continue_from` 的两级用法

### 3.1 job 级（最小请求体，推荐）

```json
{"continue_from": {"job_id": "job_1789375041513_000_7705"}, "max_tokens": 512}
```

- 与源 job **任务一一对应**（顺序一致，`tasks[i]` 续写源 `tasks[i]`）；
- 可同时给覆盖项：`max_tokens` / `temperature` / `top_p` / `lora_path` / `n`；
- 与 `requests` 同时出现 → 400（语义冲突）。

### 3.2 任务级（可精确指定）

```json
[
  {"continue_from": {"job_id": "job_x", "task_id": "job_x_t0000"}, "max_tokens": 512},
  {"continue_from": {"job_id": "job_x", "index": 1, "sample_index": 1}, "temperature": 0.0}
]
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `job_id` | ✅ | 源 job |
| `task_id` | ❌ | 源任务 id（全局唯一，提交响应里就有） |
| `index` | ❌ | 源任务序号（`task_id` 的替代写法） |
| `sample_index` | ❌ | `n>1` 时从第几个 sample 续，默认 0 |

源 job 只有 1 个任务时可都省略；**多任务且不指定 → 400**（提示补 `task_id`/`index`）。

---

## 4. 参数继承：省略 = 沿用源任务

| 参数 | 省略时的行为 |
|---|---|
| `temperature` / `top_p` | **继承源任务**（同一条实验的分布） |
| `lora_path` | **继承源任务**（同 adapter） |
| `max_tokens` | **不继承**，默认 4096 —— 新预算是新请求的事 |
| `n` | 不继承，默认 1 |

想强制基模：显式 `"lora_path": null`。

---

## 5. 结果形态

```json
{"results":[{"task_id":"...","index":0,"samples":[{
  "output_text":" …续写的正文…",
  "output_ids":[...],                 // 只含【本次新增】token
  "output_token_logprobs":[...],      // 与 output_ids 一一对齐
  "finish_reason":"stop",
  "prompt_tokens":32,                 // = 源 prompt_tokens + 源已生成 token 数
  "completion_tokens":24
}]}]}
```

- **只含本次新增 token**：历史 token 客户端上次已经拿到了，不重复；
- `prompt_tokens` 是**自检锚点**：应当等于 `源.prompt_tokens + len(源.output_ids)`，
  不等就说明上下文拼错了（目前实现下逐位吻合）。

---

## 6. 端到端：cancel → 续写（Python）

```python
import json, time, urllib.request

BASE, KEY = "http://<smg-host>:31000", "<CONTROL_KEY>"

def call(method, path, payload=None):
    r = urllib.request.Request(BASE + path,
                               data=None if payload is None else json.dumps(payload).encode(),
                               method=method)
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read().decode() or "null")

job = call("POST", "/v1/control/jobs",
           [{"prompt": "写一个 3000 行的 CSV 生成器说明。", "max_tokens": 4096}])
jid = job["job_id"]

time.sleep(5)
call("POST", f"/v1/control/jobs/{jid}/cancel")          # 取消（已生成的 token 会保留）

partial = call("GET", f"/v1/control/jobs/{jid}/result")  # 需要的话看看已生成了什么
s0 = partial["results"][0]["samples"][0]
print("已生成 token:", len(s0["output_ids"]), "| logprob 数:", len(s0["output_token_logprobs"]))

# 续写：只发引用，不回传 token
cont = call("POST", "/v1/control/jobs",
            {"continue_from": {"job_id": jid}, "max_tokens": 2048})
cid = cont["job_id"]
while True:
    st = call("GET", f"/v1/control/jobs/{cid}")
    if st["status"] in ("completed", "partial", "failed", "cancelled"):
        break
    time.sleep(2)

s1 = call("GET", f"/v1/control/jobs/{cid}/result")["results"][0]["samples"][0]
assert s1["prompt_tokens"] == s0["prompt_tokens"] + len(s0["output_ids"])   # 上下文自检
print("续写新增 token:", len(s1["output_ids"]), "| logprob 数:", len(s1["output_token_logprobs"]))
```

---

## 7. 语义与保真

| 项 | 行为 |
|---|---|
| 拼接粒度 | **token 级**（不经文本）→ 不存在边界重 tokenize 导致的漂移 |
| 续写的续写 | 支持：组合后的上下文会落盘（`input_XXXX.json`），下一跳直接复用 |
| 运行中的源 | 支持：源还在跑时取**实时 partial** 快照 |
| 源一条都没生成 | 明确报错（不会静默产空样本） |
| 历史数据 | **上线前落盘的 job 无需迁移**即可作为源（文本 prompt 由引擎 tokenizer 现场还原成 id） |
| 采样 | 要"同一条轨迹"请用贪心（`temperature=0`）或固定 seed；KV 复用 ≠ 采样事件复用 |

---

## 8. 错误码

| HTTP / 字段 | 含义 |
|---|---|
| 400 `task must provide exactly one of 'prompt', 'input_ids' or 'continue_from'` | 输入来源 0 个或 ≥2 个 |
| 400 `continue_from: unknown job_id 'xxx'` | 源 job 已删除或超出保留期 |
| 400 `continue_from: job 'x' has N tasks — specify 'task_id' or 'index'` | 多任务 job 未指定源任务 |
| 400 `body cannot contain both 'requests' and 'continue_from'` | 两种形态混用 |
| 任务级 error `no generated tokens yet` | 源任务还没产出任何 token |
| 任务级 error **`LoRA adapter is not loaded on N of M engine(s): ...`** | **见 §9**：adapter 未在每个引擎加载 |

---

## 9. ⚠️ LoRA adapter 前置条件（PD 下必读）

PD（prefill/decode 分离）下，**adapter 必须在每个引擎都处于已加载状态**，否则请求会在
bootstrap 握手里干等 600s 才失败（历史行为；现已改为**秒级失败**）。

现在的行为：

- 引擎侧：PD 模式下**禁止"请求触发自动重载 adapter"**，未加载 → 立刻 400 并给出指引；
- jobs 侧：派发前逐引擎 `GET /v1/models` 预检，缺失 → **任务立即失败**并给出同一条指引：

```
LoRA adapter is not loaded on 2 of 2 engine(s): http://prefill:30100, http://decode:30200.
A PD request for a missing adapter hangs until the bootstrap timeout (600s).
Load it on every engine first:
  POST <engine>/load_lora_adapter {"lora_name": "<path>", "lora_path": "<path>"}
and resubmit afterwards.
```

实测：3 秒返回失败（旧行为是 600s 静默）。

**正确流程**：先对**每个**引擎（prefill + decode）显式加载 adapter（或走 smg 控制面部署到全量引擎），
再提交引用它的 job / 续写请求。**不要依赖"请求触发自动加载"。**

---

## 10. 成本与性能（决定值不值得续写）

| 部分 | 是否需要重算/重传 |
|---|---|
| 原 prompt | prefill 端命中 radix / HiCache 前缀缓存 → 基本不重算 |
| **已生成的 token** | **必须重新 prefill**（它们从未进过 prefill 的缓存） |
| KV 传输 | decode 端 `--disable-radix-cache` → **整个上下文的 KV 都要从 prefill 传过去**，传输量 ∝ 上下文长度 |

结论：续写几万 token 的 rollout 很划算；**100k+ 上下文时传输是大头**。
（要"零重算零重传"需 KV 驻留能力，属另一条线，当前不支持。）

---

## 11. 运维备注

| 变量 / 项 | 说明 |
|---|---|
| `SMG_JOBS_ENGINE_URLS` | 逗号分隔的引擎地址；不设则自动取 router 启动参数里的 worker URL（用于 tokenizer 还原 prompt id / adapter 预检） |
| `SMG_JOBS_DIR` | job 持久化目录（默认 `./smg-jobs`），重启后自动恢复（本集群 67 个 job 实测恢复） |
| 保留期 | 结果在磁盘保留 48h，超期后不能作为续写源 |
| 控制键 | `Authorization: Bearer <CONTROL_KEY>`（见部署密钥） |

---

## 附：与 `/v1/completions` 的区别

| | `/v1/control/jobs`（本文档） | 裸 `/v1/completions` |
|---|---|---|
| 续写引用 | ✅ `continue_from`（服务端拼 token） | ❌ 客户端自己拼 |
| token + logprob 三元组 | ✅ 结构化返回 | 需自行解析 |
| 作业语义（提交/轮询/取消/恢复） | ✅ | ❌ |
| 适用 | 训练 / RL rollout | 单次推理

