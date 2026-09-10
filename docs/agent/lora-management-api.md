# LoRA 操作接口文档（smg control-plane + 引擎直连）

> B300 1P1D（glm53 rank-32 MoL）· smg router :31000 / 入口 18888
> 版本：2026-09-07 v2 · 每个接口均含完整 输入/输出 JSON 结构示例

---

## 0. 事故背景（为什么写这份文档）

2026-09-07 PD 双端各残留一个僵尸 LoRA（15:12:34 加载后从未卸载，占 slot）。
根因：调用方用**错误的 name** 调 `DELETE /v1/control/models/{name}`，全部 404 `not_deployed` → smg 从未执行引擎 unload → 引擎残留。

核心规则：
- smg registry 以**部署时注册的 name** 为 key；DELETE 必须传注册名，不是 lora_path / URL / `lora-xxx` 别名
- 引擎侧以 **lora_name（= path）** 为 key

---

## 1. 两条路径总览

| 路径 | 接口 | 用途 | 槽位管理 |
|---|---|---|---|
| A. smg control-plane | `/v1/control/models*` | 平台/训练侧声明式管理 | ✅ 自动 |
| B. 引擎直连 | `/load_lora_adapter` `/unload_lora_adapter` | 排查/清理 | ❌ 需双端手动 |

---

## 2. 路径 A：smg control-plane

> Base: `http://8.213.214.14:18888` ｜ 认证头: `Authorization: Bearer sk-control-pd-2026` ｜ Content-Type: `application/json`

### 2.1 POST /v1/control/models — 部署 LoRA

**请求体（输入）**

```json
{
  "name": "lora-demo-v1",
  "type": "lora",
  "path": "https://tos-mint-ckpt.tos-s3-accelerate.volces.com/adapters-v1/01a07aac-ff70-7e51-9809-4478474d82d7/01a07aac-ff70-7e51-9809-4478474d82d7.tar.zst",
  "strategy": "any"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | ✅ | **注册名（自定义，DELETE 用它）**；非空 |
| `type` | string | ✅ | `"lora"` 或 `"base"` |
| `path` | string | ✅ | 引擎本地路径或 OSS URL（引擎各自本地下载）；非空 |
| `strategy` | string | ❌ | `"any"`（单引擎，默认）/ `"all"`（所有引擎） |

**响应（输出）**

`202 Accepted`（正常异步接收）
```json
{
  "status": "accepted",
  "name": "lora-demo-v1",
  "hint": "poll GET /v1/control/models for the deployment state"
}
```

`409 Conflict`（同名已部署在某引擎）
```json
{
  "error": "already_deployed",
  "model": "lora-demo-v1",
  "engine": "<unit_id>"
}
```

`409 Conflict`（已有部署进行中，串行保护）
```json
{ "error": "deployment_in_progress", "model": "lora-demo-v1" }
```

`400 Bad Request`（name/path 为空）
```json
{ "error": "name and path are required" }
```

`202 Accepted`（同名已在 QUEUED/LOADING，幂等）
```json
{ "status": "LOADING", "name": "lora-demo-v1" }
```

---

### 2.2 GET /v1/control/models — 查询全部模型状态

无请求体。

**响应（输出）**

```json
{
  "models": {
    "glm52-fp8-official": {
      "name": "glm52-fp8-official",
      "type": "base",
      "state": "ACTIVE",
      "engine_count": 2,
      "per_engine": [
        { "engine_id": "decode-b300-2", "inflight": 3 },
        { "engine_id": "prefill-b300-1", "inflight": 3 }
      ]
    },
    "lora-demo-v1": {
      "name": "lora-demo-v1",
      "type": "lora",
      "state": "LOADING",
      "engine_count": 0,
      "per_engine": [],
      "error": null
    },
    "lora-failed-x": {
      "name": "lora-failed-x",
      "type": "lora",
      "state": "FAILED",
      "engine_count": 0,
      "per_engine": [],
      "error": "load failed on decode-b300-2; check weights exist on all cluster nodes"
    }
  },
  "generated_at_secs": 1788090480,
  "ttl_secs": 10
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `models` | object | key = name，value = ModelView |
| `ModelView.name` | string | 注册名 |
| `ModelView.type` | string | `"lora"` \| `"base"` |
| `ModelView.state` | string | `QUEUED` \| `LOADING` \| `ACTIVE` \| `DRAINING` \| `FAILED` 等 |
| `ModelView.engine_count` | int | 已加载引擎数 |
| `ModelView.per_engine` | array | `[{engine_id, inflight}]` |
| `ModelView.error` | string/null | state=FAILED 时的失败原因 |
| `generated_at_secs` | int | 生成时间戳 |
| `ttl_secs` | int | 缓存 TTL |

---

### 2.3 DELETE /v1/control/models/{name} — 删除/卸载 LoRA

**请求**：无 body；`name` 在 URL path，**必须是部署时注册的 name**。

```
DELETE /v1/control/models/lora-demo-v1
```

**响应（输出）**

`200 OK`（成功）
```json
{ "status": "removed", "model": "lora-demo-v1" }
```

`404 Not Found`（registry 无此 name → 常见错误：传了 path/URL/别名）
```json
{ "error": "not_deployed" }
```

`500 Internal Server Error`（drain 超时）
```json
{ "error": "drain_timeout" }
```

**执行语义**：drain（等 inflight=0）→ PD 双端（decode+prefill）unload → 移出 registry。QUEUED/LOADING/FAILED 状态也清理（引擎可能持有 partial load）。

---

## 3. 路径 B：引擎直连（排查 / 清理僵尸）

> ⚠️ 日常不要用这条路径——绕过 smg registry，是僵尸温床。仅排查/清理时用。
> 引擎本地接口地址：decode `http://10.0.0.67:30200`（公网 ssh -p 1022）/ prefill `http://10.0.0.75:30100`（ssh -p 1021）
> 认证头: `Authorization: Bearer sk-glm52-pd`

### 3.1 POST /load_lora_adapter — 引擎侧加载

**请求体（输入）**

```json
{
  "lora_name": "https://tos-mint-ckpt.tos-s3-accelerate.volces.com/adapters-v1/01a07aac-ff70-7e51-9809-4478474d82d7/01a07aac-ff70-7e51-9809-4478474d82d7.tar.zst",
  "lora_path": "https://tos-mint-ckpt.tos-s3-accelerate.volces.com/adapters-v1/01a07aac-ff70-7e51-9809-4478474d82d7/01a07aac-ff70-7e51-9809-4478474d82d7.tar.zst",
  "pinned": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `lora_name` | string | ✅ | 引擎侧索引名（= path，运行时请求带它命中） |
| `lora_path` | string | ✅ | 本地路径或 URL（引擎本地下载） |
| `pinned` | bool | ❌ | 是否固定不参与 LRU 淘汰，默认 false |
| `lora_id` | string/null | ❌ | 可选，自动生成可省略 |

**响应（输出）**

```json
{
  "success": true,
  "error_message": "",
  "loaded_adapters": {
    "https://tos-mint-ckpt.tos-s3-accelerate.volces.com/adapters-v1/01a07aac-ff70-7e51-9809-4478474d82d7/01a07aac-ff70-7e51-9809-4478474d82d7.tar.zst": {
      "lora_id": "51fbb8d14e2f488d8103eaae34548c85",
      "lora_name": "https://...tar.zst",
      "lora_path": "https://...tar.zst",
      "pinned": false
    }
  }
}
```

失败时：`{"success": false, "error_message": "<原因>", "loaded_adapters": null}`

---

### 3.2 POST /unload_lora_adapter — 引擎侧卸载（清理僵尸用）

**请求体（输入）**

```json
{
  "lora_name": "https://tos-mint-ckpt.tos-s3-accelerate.volces.com/adapters-v1/01a07aac-ff70-7e51-9809-4478474d82d7/01a07aac-ff70-7e51-9809-4478474d82d7.tar.zst"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `lora_name` | string | ✅ | 加载时用的完整 name（= URL/path）；查引擎日志 `lora_name=` 字段可得 |
| `lora_id` | string/null | ❌ | 可选，精确到 hash |

**响应（输出）**

```json
{ "success": true, "error_message": "", "loaded_adapters": {} }
```

unload 不存在的 adapter = 无害 `{"success": false, "error_message": "lora not loaded"}`（HTTP 400）。

> ⚠️ **PD 必须双端都执行**：decode（30200）+ prefill（30100）各调一次。

---

## 4. 完整 curl 示例

```bash
# ===== 路径 A：smg control-plane（推荐）=====
KEY=sk-control-pd-2026
BASE=http://8.213.214.14:18888

# 部署
curl -X POST $BASE/v1/control/models \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"name":"lora-demo-v1","type":"lora","path":"https://.../adapters-v1/xxxx/xxxx.tar.zst","strategy":"any"}'

# 查询（拿真实 name 与 state）
curl $BASE/v1/control/models -H "Authorization: Bearer $KEY"

# 删除（name = 注册名）
curl -X DELETE $BASE/v1/control/models/lora-demo-v1 -H "Authorization: Bearer $KEY"

# ===== 路径 B：引擎直连（僵尸清理，PD 双端）=====
EK=sk-glm52-pd
curl -X POST http://10.0.0.67:30200/unload_lora_adapter -H "Authorization: Bearer $EK" \
  -H 'Content-Type: application/json' -d '{"lora_name":"https://.../adapters-v1/xxxx/xxxx.tar.zst"}'
curl -X POST http://10.0.0.75:30100/unload_lora_adapter -H "Authorization: Bearer $EK" \
  -H 'Content-Type: application/json' -d '{"lora_name":"https://.../adapters-v1/xxxx/xxxx.tar.zst"}'
```

---

## 5. name 对照表（为什么之前 404）

| 上下文 | key | 示例 |
|---|---|---|
| smg POST 部署 | 自定义注册名 `name` | `lora-demo-v1` |
| smg DELETE path | **必须 = 注册名** | `lora-demo-v1` |
| smg GET 查询 | name（从 models 里取） | `lora-demo-v1` |
| 引擎 load/unload | `lora_name` = path/URL | `https://tos-...tar.zst` |
| 引擎日志 LoRARef | 自动 lora_id（hash） | `51fbb8d1...` |
| smg 内部别名 | `lora-<hash>-v<hash>` | ❌ 勿用于任何外部调用 |

---

## 6. 运维备注

- smg registry 在内存（router 重启即失）；引擎 loaded adapter 随引擎进程存续（**引擎重启 = 全部 LoRA 清空**，需重新 deploy）
- PD 加载顺序：decode 先（~13s）→ prefill 后（~6s）
- 槽位：`max-loaded-loras 3`（base 不计入，只计 LoRA）
- 引擎注册 key = path；runtime 请求带 `lora_path` 直接命中，decode 对未知 path 报 "never been loaded"
- 代码：`sgl-model-gateway/src/control_plane/deploy.rs`、`server.rs:882`；引擎侧 `python/sglang/srt/managers/io_struct.py`
