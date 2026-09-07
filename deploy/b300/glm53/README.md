# GLM-5.3 b300 PD 集群 — 最终部署手册（v2）

> 本目录是 b300 GLM-5.3 PD 集群（B300 spot 双节点）的**唯一真相来源**（2026-09-07 v2 更新）。
> v2 核心变化：**最终 image 无需任何 mooncake/代码挂载**——b200 完整 mooncake 包 + HEAD 代码全部内置。

---

## 1. 最终 image（v2，无需挂载）

```
b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v2
digest: sha256:2f97fe6347c1b3dede538994085ab0a59e1991fe2c13f62061e31122695a6003
```

**v2 内置内容（已验证）**：
| 项 | 值 | 验证方式 |
|---|---|---|
| mooncake engine.so | `8ea08f1e`（6MB b200 版） | md5sum = 8ea08f1ee95150fc218b23cb4d91978b |
| mooncake Python 包 | b200 完整 35 文件（8 个 ep_*.so 绑定 + __init__.py） | `from mooncake.engine import TransferEngine` OK |
| 代码 | HEAD `4aa32c29bc`（fix/eagle-seed-contract） | conn.py = e79f3463 |
| DCP 修复 | memory_pool.py 含 `_slot = (loc // (_ps * _dws))` | grep 验证存在 |

**部署时只需挂载数据目录**（models/hicache/deep_gemm/tvm-ffi），无需挂载 mooncake 包或 sglang 代码 overlay。

## 2. 部署拓扑

| 端 | 节点 | 内网 IP | 端口 | SSH |
|---|---|---|---|---|
| decode | B300 spot decode | 172.31.45.101 | 30002 | `ssh -i ~/.ssh/b300-spot.pem ubuntu@15.164.0.39` |
| prefill | B300 spot prefill | 172.31.47.105 | 30100 | `ssh -i ~/.ssh/b300-spot.pem ubuntu@3.37.20.114` |
| router | prefill 节点 | 172.31.47.105 | **31000** | 同 prefill |

**请求端点（对外）：`http://3.37.20.114:31000/v1`（公网 IP）**
⚠️ 从本地测试用公网 IP `3.37.20.114`，不要用内网 `172.31.47.105`（本地 HTTP 代理会劫持内网请求返回假 502；且内网 IP 本地不可达）。

## 3. 文件说明

| 文件 | 说明 |
|---|---|
| `b200_aligned.env` | b200 2p2d 对齐的完整 env（33 条，生产验证） |
| `start_decode.sh` | decode 启动（v2 image，bootstrap 30011，DCP=4+EAGLE） |
| `start_prefill.sh` | prefill 启动（v2 image，bootstrap 30012，CP layersplit+HiCache） |
| `start_router.sh` | router 启动（PD ready 后；api-key 从 env 读） |

## 4. 部署铁律（顺序不可颠倒）

1. **decode + prefill 同时启动**（两端必须配对，engine/mooncake/env 完全一致）：
   - decode 节点: `bash start_decode.sh`
   - prefill 节点: `bash start_prefill.sh`
2. **等 PD 双端 ready**（两端日志都出现 `fired up and ready` + health 200）
   - decode: `curl http://127.0.0.1:30002/health`
   - prefill: `curl http://127.0.0.1:30100/health`
3. **PD ready 前 router 必须 = 0**（铁律）
4. PD 双端 ready 后起 router:
   ```bash
   export GLM53_ROUTER_API_KEY=<真值见 secrets.env>
   bash start_router.sh
   ```
5. 验证: `curl http://3.37.20.114:31000/health` 应 200，然后 e2e:
   ```bash
   curl http://3.37.20.114:31000/v1/chat/completions \
     -H "Authorization: Bearer $GLM53_ROUTER_API_KEY" -H "Content-Type: application/json" \
     -d '{"model":"glm53","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":40}'
   ```

## 5. env 关键差异（b200_aligned.env vs 旧配置）

- **删掉** tcp pool 对: `MC_TCP_ENABLE_CONNECTION_POOL`、`MC_TCP_CONNECTION_IDLE_TIMEOUT`（b200 2p2d 没有）
- **补齐** b200 有而旧配置缺的: `MOONCAKE_PROTOCOL=tcp`、`MC_FORCE_TCP=1`、`MC_TCP_LANES_PER_PEER=16`、`MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384`、`MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=16384`、`MC_TCP_ADMISSION_TIMEOUT_MS=10000` 等
- 完整 33 条见 `b200_aligned.env`

## 6. 关键坑（排障）

- **mooncake Python 包必须与 engine.so 配套**: v2 image 已内置 b200 完整包（engine.so 8ea08f1e + ep_*.so 绑定）。若只挂单文件 engine.so 而不配套 ep 绑定 → `ImportError: Please install mooncake`
- **PD 两端必须配对重启**: 只重启一端会导致 mooncake session/engine 状态不一致 → 传输失败（prefill 报 "Decode instance could be dead"）
- **不要用内网 IP 测试**: 本地 curl 内网 172.31.47.105 会被 HTTP 代理劫持返回假 502；用公网 3.37.20.114
- **router worker 注册是一次性的**: router 启动时 decode 未 ready → discover_metadata 超时 → 永远 "No decode workers" → 必须重启 router（PD ready 后）
- **api-key 不硬编码**: start_router.sh 从 `GLM53_ROUTER_API_KEY` 环境变量读（真值见仓库根 secrets.env）
