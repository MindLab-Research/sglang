# Production Topology

The mol production cluster is a hybrid of two **sglang PD disaggregation**
pairs and one **vllm** single-node worker, fronted by a Rust gateway and a
Python Proxy on deploy-0. All five nodes are production.

## Node / IP map

| Node | Internal IP | Role |
|---|---|---|
| mol-deploy-0 | 10.0.58.3 | vllm worker + **gateway (30001)** + **Proxy (30000)** |
| mol-deploy-1 | 10.0.58.22 | pd-router-2: prefill + router |
| mol-deploy-2 | 10.0.58.21 | pd-router-1: prefill + router |
| mol-deploy-3 | 10.0.58.20 | pd-router-1: decode |
| mol-deploy-4 | 10.0.58.23 | pd-router-2: decode |

⚠️ Re-image can change internal IPs. Always confirm with `hostname -I` before
trusting a stale IP.

## Workers (gateway view, 3 total)

| worker name | URL | runtime | physical composition |
|---|---|---|---|
| mol-deploy-pd-router-1 | http://10.0.58.21:30000 | sglang | deploy-2 prefill + deploy-3 decode |
| mol-deploy-pd-router-2 | http://10.0.58.22:30000 | sglang | deploy-1 prefill + deploy-4 decode |
| mol-deploy-0 | http://10.0.58.3:8000 | vllm | single-node TP8/DCP4 (only pure vllm) |

> Gateway worker labels may read `mol-deploy-pd-router` (router-1) /
> `mol-deploy-pd-router-2` (router-2). The `-1`/no-suffix naming is the same
> pair; resolve by the URL's IP, not the suffix.

## PD port table (identical for both PD pairs)

| role | port | notes |
|---|---|---|
| router (facing gateway) | 30000 | the URL registered to gateway |
| prefill | 30100 | kv_producer |
| decode | 30200 | kv_consumer |
| bootstrap (dist-init handshake) | 8998 | mooncake/NIXL |
| prometheus | 29001 | router metrics |

Each PD pair = prefill node (30100 + 8998 + router 30000) + decode node
(30200 + 8998). The router runs on the **prefill** node.

## Front-end (deploy-0)

| component | bind | notes |
|---|---|---|
| gateway (Rust `bin/smg`) | 127.0.0.1:30001 (also 0.0.0.0) | worker registry, health-check 60s, policy `min_load_then_group` |
| Proxy (Python `mol_harness.proxy`) | 0.0.0.0:30000 | `UPSTREAM=http://127.0.0.1:30001`; public entry `8.222.11.182:18275` → Proxy |

## Model identity

- served-model-name / worker `model_id`: `glm52-fp8-official`
- Proxy-facing model name: `Macaron-V1-Venti` (Proxy maps it to the worker
  model_id + injects the Macaron identity prompt)
- weights: `/root/glm52_local/base` (GLM-5.2-FP8) + 4 LoRA (L0/L1/L2/L3) at
  `/root/glm52_local/loras/L{0,1,2,3}`; L2 = ckpt_step1369

## Auth

- sglang PD routers are started with `--api-key sk-glm52-pd`. Any direct call
  to a router (30000) must carry `Authorization: Bearer sk-glm52-pd` or it 401s.
- The gateway forwards the key automatically: register the worker with
  `"api_key":"sk-glm52-pd"` in `POST /workers`; the gateway adds the header on
  every forwarded request (`WorkerMetadata.api_key` →
  `extract_auth_header`, `sgl-model-gateway/src/core/worker.rs`). The
  `/workers` list does **not** echo the key back.
- Proxy (30000) and gateway (30001) take no auth for `/v1/chat/completions`.

## B300 dev stack (source/reference machine)

B300-1 (`ssh -p 1021 root@8.213.215.2`) runs a **develop-branch** MoL stack for
compatibility testing. It is **separate** from production deploy-0.

| component | bind | notes |
|---|---|---|
| sglang prefill | 30100 | same binary as production |
| sglang PD router | 30000 | `--api-key sk-glm52-pd` |
| smg gateway (develop) | 127.0.0.1:31001 | compiled from `/root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway` |
| Proxy (develop) | 0.0.0.0:31000 | `PYTHONPATH=/root/Mixture-of-LoRA-Harness-alpha` |

- develop smg binary: `/root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway/target/release/smg`
- develop proxy + library: `/root/Mixture-of-LoRA-Harness-alpha/mol_harness/`
- Proxy needs `MOL_UPSTREAM_API_KEY=sk-glm52-pd` (develop smg enforces auth)
- sglang must have `Glm45Detector` continue_final_message fix (see commands.md)
- Start/verify commands in [commands.md](commands.md) § "B300 dev gateway + proxy"

## Node boundary (HARD)

- **All 5 deploy nodes are production.** None may be touched without target
  confirmation; production-side mutations need explicit user confirmation.
- "deploy-2/3 is a debug sandbox" is **obsolete** — they are now
  pd-router-1's prefill/decode and serve live traffic.
- Source/reference machines `ssh -p 1021 root@8.213.215.2` (prefill+router)
  and `ssh -p 1022 root@8.213.215.2` (decode) are **read-only**. Copy from
  them; never modify. ⚠️ their sglang `conn.py` drifted — copy sglang from a
  verified deploy-2 instead (see deploy-notes §8.1).
- The dev box (this host) **cannot** reach 10.0.58.x. Any curl to a worker
  must run **on deploy-0** (or the worker node itself) over ssh.

## Topology drift

The cluster evolved: 5 vllm workers (deploy-0/1/2/3/4) → deploy-2/3 converted
to sglang PD (pd-router-1) → deploy-1/4 converted to sglang PD (pd-router-2) →
deploy-0 kept as vllm. If `/workers` disagrees with this doc, **trust the live
`prod_status.sh` output** and update this file.

## 2P3D cluster (8.222.11.182, SSH ports 1100–1104)

A separate **2-prefill + 3-decode** sglang PD disaggregation cluster on five
L20D VF nodes.  All nodes share the same public IP `8.222.11.182` but use
distinct SSH ports.  Code and weights live under `/opt/sglang-venv/`.

### Node / IP map

| SSH port | hostname | Internal IP | Role |
|---|---|---|---|
| 1100 | dsw-46797-55c447f97b-97jmp | 10.0.58.38 | decode |
| 1101 | dsw-46919-56557f65dc-p9snz | 10.0.58.34 | prefill + router + gateway + proxy |
| 1102 | dsw-47108-85fc7c8464-wrchg | 10.0.58.35 | prefill |
| 1103 | dsw-47110-7fbfb9bf75-l4jq7 | 10.0.58.36 | decode |
| 1104 | dsw-48098-66b7b84f76-p6w46 | 10.0.58.37 | decode |

⚠️ Re-image can change internal IPs.  Always confirm with `hostname -I`.

### Binaries & paths

| item | path / value | notes |
|---|---|---|
| Python (1102/1104) | `/root/sglang_venv/bin/python3` | ✅ 当前生产用（2026-08-19 起，1102 全量同步到 1104，含全部 LoRA/CP 修复）；`/root/start_1p1d_lora.sh` 默认用它 |
| Python (1101, 老 2P3D) | `/opt/sglang-venv/bin/python` | 旧 0.5.15.post1；1102/1104 上**勿用**（sglang-kernel 0.4.4 触发版本断言）；1104 的 `/root/v15_patched` 已被 sglang_venv 取代 |
| sglang code (1102/1104) | `/root/sglang_venv/lib/python3.12/site-packages/sglang/` | rsync target for code patches（同步后清 `__pycache__` + md5 抽查） |
| sglang code (1101) | `/opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/` | 老 2P3D rsync target |
| sglang-router | `/opt/sglang-venv/bin/sglang-router` | **NOT in PATH** — 1102 router :30000 用它启动（override `ROUTER_BIN=...`） |
| smg gateway | `/usr/local/bin/smg` | override `GATEWAY_BIN=/usr/local/bin/smg` in start_pd.sh |
| start script (1P1D) | `/root/start_1p1d_lora.sh` | 1102/1104 当前形态：prefill 1102:30100 (CP interleave+VE, 0.85, 无 HiCache/layersplit) + decode 1104:30200 (DCP=4+EAGLE5+VE, 0.90) |
| start script (老 2P3D) | `/root/start_pd.sh` | prefill/decode/router/gateway/proxy functions；1101 的 HiCache+layersplit 配置会 OOM，勿照抄 |
| alt script | `/root/start_1p4d.sh` | 1P4D layout；当前**不用** |
| node ssh | 1102→1104 免密 | `ssh root@10.0.58.37`（2026-08-19 配好）；venv/权重走节点间 rsync ~580MB/s |
| model weights | `/root/glm52_local/base` | GLM-5.2-FP8 |
| LoRA adapters | `/root/glm52_local/loras/L0–L3` | 1102/1104 用老 L2（无 L2_0818001；老 2P3D 节点用 L2_0818001） |

### Service ports (identical layout to mol-prod)

| role | port | notes |
|---|---|---|
| router (facing gateway) | 30000 | runs on 1101 |
| prefill | 30100 | 1101 + 1102 |
| decode | 30200 | 1100 + 1103 + 1104 |
| smg gateway | 31001 | 127.0.0.1 only, on 1101 |
| proxy (mol_harness) | 31000 | 0.0.0.0, on 1101 |
| bootstrap (mooncake) | 8998 | on every prefill + decode node |
| prometheus (router) | 29001 | on 1101 |

### Public entry

- **Public URL**: `http://8.222.11.182:18777/v1/chat/completions`
- **Port mapping**: 18777 (public) → proxy 31000 (1101) → gateway 31001 → router 30000 → prefill/decode
- **Proxy API key**: `$MOL_API_KEY_2P3D`（真值见仓库根 secrets.env）
- **Router API key**: `sk-glm52-pd` (used by gateway → router)
- **Model name**: `Macaron-V1-Venti` (proxy routes to `glm52-fp8-official` + LoRA)

### Decode configuration (current, DCP=4 + EAGLE 5 steps)

```
--tp 8 --kv-cache-dtype fp8_e4m3 --page-size 128
--mem-fraction-static 0.90 --skip-server-warmup
--cuda-graph-max-bs-decode 64 --max-running-requests 64
--disaggregation-mode decode --dcp-size 4
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998
--disaggregation-ib-device mlx5_0
# 2026-08-05 changed from DCP=2 no-EAGLE → DCP=4 + EAGLE (start_pd.sh decode function)
```

### Prefill configuration (current)

```
--tp 8 --kv-cache-dtype fp8_e4m3 --page-size 128
--mem-fraction-static 0.85 --enable-metrics
--enable-hierarchical-cache --hicache-ratio 1
--hicache-write-policy write_back --hicache-mem-layout page_first
--hicache-storage-backend file --file-storage-path /root/hicache
--enable-prefill-cp --cp-strategy interleave
--enable-dsa-prefill-cp-layersplit
--disaggregation-mode prefill
--disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998
--disaggregation-ib-device mlx5_0
```

### Environment variables (set in start_pd.sh)

```
TVM_FFI_CUDA_ARCH_LIST=10.0a
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G
MOONCAKE_DISABLE_HIP_DMABUF=1
IBV_ACCESS_RELAXED_ORDERING=1
```

### Code patches deployed (2026-08-05, local HEAD `6a00bbde3`)

当前部署 = 本地仓库 `/home/smith/src/sglang-b300-glm52`（分支 b300-glm52）整个 `python/sglang/srt/` 目录 rsync 到 5 节点，git HEAD=`6a00bbde3`。关键 fix（按时间）：

| commit | fix |
|---|---|
| `13181f2f1` | Revert a605ab4e6（全局 seq_len 修复导致 prefill 崩溃，回退） |
| `d4d23041d` | DSA prefill CP gates rank-invariant（`can_dsa_cp_split` 用 seq_len，`can_dsa_prefill_cp_round_robin_split` 用 extend_num_tokens） |
| `4ba456a80` | HiCache `bulk_check_prefetch_progress`（防 cross-collective 死锁） |
| `3df13dd42` | **prefill iteration barrier 移到 pop_bootstrapped 前**（修 py-spy 发现的 pop_bootstrapped all_reduce 死锁） |
| `6a00bbde3` | bulk_check_prefetch_progress finalize path 修正（当前 HEAD） |

### Known ops traps (2P3D-specific)

1. **`sglang-router` not in PATH**: must use `ROUTER_BIN=/opt/sglang-venv/bin/sglang-router bash /root/start_pd.sh router ...` — bare `sglang-router` fails silently.
2. **`PYTHON` wrong default**: start_pd.sh defaults to `/root/sglang_venv/bin/python3` which doesn't exist on 2P3D — must set `PYTHON=/opt/sglang-venv/bin/python`.
3. **Decode restart → router stale**: after restarting any decode node, the router marks it unhealthy and **never auto-recovers** — must restart router + smg gateway.
4. **Router restart → gateway stale**: after restarting router, smg gateway must also be restarted (it caches worker health state).
5. **All 5 nodes must get code patches**: rsync to ports 1100–1104, target `/opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/`; clear `__pycache__` + triton config JSON + deep_gemm cache after.
6. **⛔ 每次重启必须同步最新代码（用户硬规则 2026-08-05）**: 重启任何 2P3D 服务前，先 rsync 本地最新代码（`/home/smith/src/sglang-b300-glm52/python/sglang/srt/`，git HEAD 即当前版本）到全部 5 节点 + 清 cache。绝不带旧代码重启。

### Full cluster restart (2026-08-05 current flow)

⛔ **每次重启第一步：同步最新代码**（用户硬规则）。本地仓库：
`/home/smith/src/sglang-b300-glm52/python/sglang/srt/`（git HEAD 即当前版本，勿用旧仓库 `sglang_b300_decode`——已删）。

```bash
# 1) rsync 最新代码到 5 节点（增量，排除 pycache）
SRC=/home/smith/src/sglang-b300-glm52/python/sglang/srt/
DEST=/opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/
for port in 1100 1101 1102 1103 1104; do
  rsync -avz -e "ssh -p $port -o ConnectTimeout=15" --exclude='__pycache__' --exclude='*.pyc' "$SRC" root@8.222.11.182:"$DEST" &
done; wait

# 2) 清 cache（每节点）
find /opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/ -name '__pycache__' -exec rm -rf {} +
find /opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/ -name '*.json' -path '*/triton_*' -delete
rm -rf /root/.cache/deep_gemm/cache/

# 3) 杀干净（每节点：launch_server + scheduler + router + smg + mol_harness + 端口）
ps aux | grep -E 'launch_server|sglang::scheduler|sglang::router|smg|mol_harness' | grep -v grep | awk '{print $2}' | xargs -r kill -9
lsof -ti :30100 :30200 :30000 :31000 :31001 2>/dev/null | xargs -r kill -9
# ⛔ 验证 0 残留 0 端口后才启动（skill 硬规则）

# 4) 启动 2 prefill（1101/1102）+ 3 decode（1100/1103/1104）
PYTHON=/opt/sglang-venv/bin/python bash /root/start_pd.sh prefill    # on 1101, 1102
PYTHON=/opt/sglang-venv/bin/python bash /root/start_pd.sh decode     # on 1100, 1103, 1104

# 5) 等 5 节点 health=200（模型加载 + CUDA graph capture 5-10 分钟）
#    直接用 curl 检查（勿信嵌套 SSH 轮询的 000，会误判漏检）

# 6) 重启 router（on 1101，重新发现 worker）+ gateway + proxy + otelcol
ROUTER_BIN=/opt/sglang-venv/bin/sglang-router \
  bash /root/start_pd.sh router \
  --prefill 10.0.58.34:30100 --prefill 10.0.58.35:30100 \
  --decode 10.0.58.38:30200 --decode 10.0.58.36:30200 --decode 10.0.58.37:30200
GATEWAY_BIN=/usr/local/bin/smg bash /root/start_pd.sh gateway
MOL_API_KEY="${MOL_API_KEY_2P3D}" PROXY_SRC=/root/mol-stack bash /root/start_pd.sh proxy
bash /root/manage_otelcol.sh start    # ⛔ 重启后必须重启 otelcol，否则看板 nodata

# 7) 确认公网：curl -s -m 60 http://8.222.11.182:18777/v1/chat/completions（期望 200 + 正确输出）
```

### 单独重启命令（router/gateway/decode/prefill）

```bash
# Restart router (on 1101) — must use full path for sglang-router
ROUTER_BIN=/opt/sglang-venv/bin/sglang-router \
  bash /root/start_pd.sh router \
  --prefill 10.0.58.34:30100 --prefill 10.0.58.35:30100 \
  --decode 10.0.58.38:30200 --decode 10.0.58.36:30200 --decode 10.0.58.37:30200

# Restart smg gateway (on 1101)
GATEWAY_BIN=/usr/local/bin/smg bash /root/start_pd.sh gateway

# Restart proxy (on 1101)
MOL_API_KEY="${MOL_API_KEY_2P3D}" PROXY_SRC=/root/mol-stack bash /root/start_pd.sh proxy

# Restart decode (on any decode node, e.g. 1100)
PYTHON=/opt/sglang-venv/bin/python bash /root/start_pd.sh decode

# Restart prefill (on any prefill node, e.g. 1101)
PYTHON=/opt/sglang-venv/bin/python bash /root/start_pd.sh prefill
```

### 死锁诊断（py-spy）

prefill 卡死（health 200 但公网超时、scheduler 无 DCP-XFER 日志）时用 py-spy 拉栈定位：

```bash
# 拉 launch_server 主进程 + 8 个 scheduler 子进程的 Python 栈
ps aux | grep 'launch_server' | grep -v grep | awk '{print $2}'   # 主进程 PID
ps -eo pid,cmd | grep 'sglang::scheduler' | grep -v grep          # scheduler 子进程 PID
/opt/sglang-venv/bin/py-spy dump --pid <pid> > /root/pyspy_<name>.txt
# 已知死锁签名：scheduler 全卡 poll_and_all_reduce_attn_cp_tp_group 的 all_reduce
# （prefill.py pop_bootstrapped，utils.py:138 vs 143 行号不一致 = collective 不匹配）
```

PD deploy/recover-from-reimage details live in `.md/sglang-pd-deploy-notes.md`
and `.md/sglang-pd-recover-after-reimage.md` (not part of this skill).
