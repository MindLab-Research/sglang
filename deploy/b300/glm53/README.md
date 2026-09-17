# GLM-5.3 b300 PD 集群 — 最终部署手册（v2）

> 本目录是 b300 GLM-5.3 PD 集群（B300 spot 双节点）的**唯一真相来源**（2026-09-07 v2 更新）。
> v2 核心变化：**最终 image 无需任何 mooncake/代码挂载**——b200 完整 mooncake 包 + HEAD 代码全部内置。

---

## 1. 最终 image（v3 = v2 + decode-radix leak 修复 4da29b8f87，无需挂载）

```
b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v3
digest: sha256:bef3f085373042e83aedd9e29e0d5050a49607e5417a78f05d898d0937324ae3
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

## 7. 镜像重建（v4 = v3 + 目标 commit；本地构建，**不 push**）

分支上有了新修复（例如 PD 隐式重载守卫、base 不占池槽）之后，不必复刻 mooncake 打包 /
`sgl-kernel` 编译 —— 直接在 v3 上**叠加目标 commit 的代码**即可：

```bash
cd <sglang 仓库>
git fetch origin
bash deploy/b300/glm53/rebuild-image.sh --ref origin/b300-glm52        # 或用具体 sha
# 可选：--with-p2p-ar（同时编译 #24 的 P2P AR 扩展；该功能默认关）
```

产出本地 tag `mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v4-<sha8>`；
脚本**只构建与自检，不做 `docker push`**，结尾打印 `docker save` → `scp` → `docker load` 的具体命令。

**为什么叠加就够了**（2026-09-17 实测）：v3 的 `python/` 树 == commit `05243630`
—— 3544 个文件里 3445 个 sha256 与提交一致，8 个不同（3 个是我们自己的 overlay
文件、5 个是被构建期改写的 `pyproject*.toml`），另有 **91 个构建期产物不在 git 树里**
（`sglang.egg-info/*`、`sglang/_version.py`、`jit_kernel/csrc/**.cuh` …）。
`COPY sglang-src/ /sgl-workspace/sglang/` 只覆盖**同路径**文件，不会删除这些产物。

**自检**（`rebuild-image.sh` 自动执行，任一不过就失败退出）：

| 阶段 | 检查 |
|---|---|
| 构建期 | `sglang` 的 import 路径仍指向 `/sgl-workspace/sglang/python/sglang/`（否则覆盖不生效）；`compileall` 通过；`_version.py` / `sglang.egg-info/PKG-INFO` 仍在；mooncake `engine.so` md5 = `8ea08f1ee95150fc218b23cb4d91978b`；`python/` 下 `.so` 计数 = 0 |
| 构建后 | 镜像里 `python/**` 全量 sha256 清单 == 该 commit 的树（证明 COPY 完整、无旧文件残留）；探针逐项对比基础镜像（`.so` 数、`jit .cuh` 数不减少、`/opt/p2p_ar` 与 `--with-p2p-ar` 一致） |

**上机与回滚**：

```bash
# 两台机器分别 docker load 之后（两端镜像必须一致，PD 铁律）
LMS_ENGINE_IMAGE=mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v4-<sha8>
bash /root/mol-stack.sh restart          # 两端一起
# 回滚：把 LMS_ENGINE_IMAGE 换回 v3 tag（v3 镜像仍在机器上）+ 配对重启
```

**已知边界**：

- v4 只解决**代码**：`p2p_ar` 的 CUDA 扩展必须显式 `--with-p2p-ar` 才编译，启用还需
  `SGLANG_P2P_AR=1` + `SGLANG_P2P_AR_LIB=/opt/p2p_ar/libp2p_ar.so`（默认关、未在机器上验证）；
- `pyproject*.toml` 会被目标 commit 的版本覆盖（仅构建元数据，运行时不读）；
- 若目标 commit 与 v3 的差异包含**本目录的 start 脚本**（`deploy/b300/glm53/start_*.sh`），
  那是给 spot 集群用的版本，AWS 上机实际用的是平台仓 `deploy/aws-mol/start_mol_*.sh`，两者要各自同步。
