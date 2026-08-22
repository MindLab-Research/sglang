# 远程下载 LoRA 启用文档（Remote URL LoRA Hot-Swap）

> 分支：`b300-glm52` | 更新：2026-08-11
> 功能：给最上层 router 提交一个 **远程 URL（含 .tar.gz / .tar.zst 签名链接）**，
> 自动在 PD 集群节点本地下载 → zstd 解压 → adapter 定位 → 两端（prefill+decode）加载 →
> 提供服务。全程无需中间层中转下载、无需 SSH 到子节点。

---

## 1. 架构原理

```
最上层 Router (B)
   │  POST /v1/control/models {name, path=远程URL}
   ▼
中间 Router (A)   ← 递归下钻（forward_to_subrouter）
   │
   ▼
pd_cluster（叶子）
   ├─ prefill 引擎  ──┐
   └─ decode  引擎  ──┼─ 各自【本地】下载 URL（本地带宽、并行、无中转）
                      └─ 解压(zstd) → 定位 adapter 子目录 → 加载
```

**关键设计（区别于中间层中转下载）：**
- **引擎本地下载**：`resolve_lora_local_path`（引擎侧）每个 PD 节点自己下载，
  本地带宽、两端并行、不经过 router 中转、不依赖子节点 SSH。
- **一次下载，缓存复用**：`.done` marker + 单次下载锁（8 个 TP rank 只有一个下载），
  已下载的权重不重复拉取。
- **异步部署**：`POST` 立即返回 `202 Accepted`，下载/解压/加载后台执行，不阻塞、不被 HTTP 超时中断。

---

## 2. 启用步骤（一次性）

### 2.1 引擎代码（两端 prefill + decode 的 sglang 均已部署）

改动文件：`python/sglang/srt/lora/lora_manager.py`
- `resolve_lora_local_path()` — URL 检测 + 本地缓存 + zstd 解压 + adapter 定位
- `_find_adapter_dir()` — 在 checkpoint 嵌套结构里定位 `adapter_config.json` 所在目录
- `_stream_download()` — 用 `curl` 下载（比 requests 流式稳定）

> 若从源码部署：直接 rsync 该文件到两端 site-packages 后**重启 PD**（引擎代码变更）。

### 2.2 Router 代码（`sgl-model-gateway/`）

- `src/control_plane/deploy.rs` — 声明式部署状态机（202 异步 + swap-slot 排水 + URL 透传）
- `src/control_plane/mod.rs` — 控制面板状态/视图/能力发现

### 2.3 前置环境（集群节点）

| 依赖 | 说明 | 检查 |
|---|---|---|
| **zstd 命令** | `tar --zstd` 解压必需 | `which zstd`（两端都要有，`apt-get install -y zstd`） |
| **加速域名** | OSS 签名 `X-Amz-SignedHeaders=host` 绑定域名，**换 host 会 403** | 用与签名一致的域名 |
| 节点到 OSS 网络 | 下载速度关键（实测加速域名 141MB/s+） | `curl -m 10 -o /dev/null -w "%{speed_download}" <url>` |

### 2.4 启动 router（三级）

```bash
# A（中间层）：注册 pd_cluster
./smg launch --host 0.0.0.0 --port 30500 --prometheus-port 29011 &
curl -X POST http://127.0.0.1:30500/v1/control/register \
  -d '{"id":"b300-1p1d","url":"http://127.0.0.1:30000","kind":"pd_cluster",
       "prefill_url":"http://127.0.0.1:30100","decode_url":"http://10.0.0.67:30200",
       "api_key":"sk-glm52-pd","models":[],"capacity":4}'

# B（最上层）：注册 A 为子 router
./smg launch --host 0.0.0.0 --port 30501 --prometheus-port 29012 &
curl -X POST http://127.0.0.1:30501/v1/control/register \
  -d '{"id":"router-a","url":"http://127.0.0.1:30500","kind":"router"}'
```

---

## 3. 使用方式（从最上层提交）

### 3.1 声明式部署（远程 URL 下载）

```bash
curl -X POST http://<最上层router>:30501/v1/control/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sess-step5",
    "type": "lora",
    "path": "https://<oss>/.../sess-xxx.tar.zst?X-Amz-...",
    "strategy": "any"
  }'
```

**响应（立即，202 异步）：**
```json
{"status":"accepted","name":"sess-step5",
 "hint":"poll GET /v1/control/models for the deployment state"}
```

### 3.2 查询部署状态

```bash
curl http://<router>/v1/control/models
# → models.sess-step5.state: QUEUED → LOADING → ACTIVE（或 FAILED + error）
```

### 3.3 参数说明

| 字段 | 说明 |
|---|---|
| `name` | LoRA 名字（也是缓存目录名，重名复用缓存） |
| `type` | `lora` / `base` |
| `path` | **本地目录** 或 **http(s):// URL**（`.tar.gz` / `.tar.zst`，带签名） |
| `strategy` | `any`（选压力最低引擎） / `all`（滚动全部） |

### 3.4 卸载 / 移除

```bash
curl -X DELETE http://<router>/v1/control/models/sess-step5
# 排干（等在途归零）→ 双端 unload → 移除
```

---

## 4. 缓存机制（不重复下载）

| 机制 | 说明 |
|---|---|
| `.done` marker | 下载+解压成功后写 `SGLANG_LORA_CACHE_DIR/<name>/.done`；存在则跳过下载直接复用 |
| 单次下载锁 | `mkdir <name>.lock`（原子）——8 个 TP rank 只有 1 个下载，其余轮询等待 marker |
| adapter 定位 | `_find_adapter_dir` 自动定位 `adapter_config.json` 所在目录（兼容 checkpoint 嵌套 `adapter/` 子目录结构） |
| 缓存位置 | `SGLANG_LORA_CACHE_DIR`（默认 `/root/glm52_local/loras`），两端各自缓存 |

---

## 5. 部署时序与实测

```
POST（202）→ 后台执行：
  drain（选 replacee=最闲 lora → 排干 → unload，60s 超时兜底）
  → LOADING → 两端【并行】load（各节点下载压缩包 ~6GB → zstd 解压 15GB → 加载）
  → ACTIVE → 可服务
```

| 场景 | 实测耗时 |
|---|---|
| 冷部署（含下载，两端并行） | **62 秒**（加速域名，压缩包 ~6GB @ 141MB/s+） |
| 缓存命中（已下载过） | ~1.5 分钟（加载 + 编排） |
| 加载本身（15GB 权重） | ~20 秒/端 |

---

## 6. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `403` 下载 | OSS 签名 `SignedHeaders=host` 绑定域名，换 host 签名失效 | 用与签名一致的域名（加速域名需重新签名） |
| `tar: zstd: Cannot exec` | 节点缺 `zstd` | `apt-get install -y zstd`（两端） |
| `No such file ... adapter_config.json` | checkpoint 嵌套 `adapter/` 子目录，旧代码未定位 | 升级到 `_find_adapter_dir` 版本（缓存+下载分支都定位） |
| `already_deployed` | 引擎已加载该 lora | 先 `DELETE /v1/control/models/<name>` 卸载再提交 |
| `load_lora_adapter ... 400` | 引擎 load 失败（`success=false`） | 查 `error_message`；确认权重在节点本地、adapter 目录完整 |
| `drain timeout` | replacee 有在途请求未归零 | 部署互斥已保证单次；等在途完成或手动处理 |
| 下载慢 | 原域名带宽低（~5MB/s） | 用 OSS **加速域名**（实测 141MB/s，需重新签名） |

---

## 7. 代码位置

| 文件 | 职责 |
|---|---|
| `python/sglang/srt/lora/lora_manager.py` | 引擎侧 `resolve_lora_local_path` / `_find_adapter_dir` / `_stream_download` |
| `sgl-model-gateway/src/control_plane/deploy.rs` | `deploy_model`（202 异步）/ `execute_deploy` / `load_model_on_unit`（并行）/ `forward_to_subrouter` |
| `sgl-model-gateway/src/control_plane/mod.rs` | 控制面板状态、`/v1/control/*` handler、能力发现、指标采集 |
| `sgl-model-gateway/src/server.rs` | `/v1/control/*` 路由挂载 |
| `scripts/sync_lora_weights.sh` | （可选）手动同步本地权重到集群节点 |

---

## 8. 已知边界

1. **下载只发生一次**：同 `name` 复用 `.done` 缓存；换名字会重新下载。
2. **prefill 与 decode 并行下载**：总耗时 ≈ `max(单端)`（62s 实测），若两端带宽竞争会更慢。
3. **PD 两端权重独立**：各节点本地缓存，删除某端缓存需重新下载该端。
4. **warmup 时序**：PD 完全 warmup（日志 `ready to roll` / `capture cuda graph finished`）后才能调
   `load_lora_adapter`，否则触发 NCCL 死锁（本分支已知陷阱）。
