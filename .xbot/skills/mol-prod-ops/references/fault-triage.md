# Fault Triage

Start here when a worker is missing, unhealthy, returning 503, or producing
garbled output. Read [topology.md](topology.md) for node/port resolution.

## Step 1 — classify from the gateway view

Run `scripts/prod_status.sh`. Match the symptom:

| Symptom (gateway view) | First move |
|---|---|
| worker missing from `/workers` | engine process died → Step 2 |
| worker listed, `is_healthy=False` | gateway health-check failing on `<worker>/health` → Step 3 |
| worker `is_healthy=True` but requests 503 | gateway-healthy ≠ PD-chain-working → Step 4 |
| output garbled (backticks/runes/repeats) | PD chain desync → Step 5 |

## Step 2 — worker missing (engine died)

1. Confirm readiness the right way: `curl /v1/models` on the worker — **not**
   `/health` (returns empty body, indistinguishable from "still loading").
2. `ps` for the engine process; if gone, `tail` its log.
3. Match the crash signature in the table below.
4. Restart per [ops-playbook.md](ops-playbook.md) ("Restart PD pair" or
   "Restart vllm worker").

## Step 3 — worker unhealthy

Gateway polls `<worker>/health` every 60s; 3 failures → unhealthy. Probe the
worker's `/health` **from deploy-0** (dev box can't reach 10.0.58.x):

```bash
ssh mol-deploy-0 'curl -s -o /dev/null -w "%{http_code}\n" http://10.0.58.21:30000/health'
```

- connection refused → engine process down → Step 2.
- 503 from the router → PD chain broken (prefill down, router unhealthy) → Step 4.
- 200 from worker but gateway still unhealthy → wait one interval; if stuck,
  the gateway registry may be stale → restart gateway (drops workers,
  re-register).

## Step 4 — healthy worker, requests 503 (the trap)

⚠️ **A sglang router returns `/health` 200 even when its prefill is dead.**
The gateway sees healthy and routes traffic, which then 503s inside the PD
pair. `prod_status.sh` will not catch this.

Send a real request **through the router** (on the prefill node, with the
api_key):

```bash
ssh mol-deploy-1 'curl -s -m 60 -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-glm52-pd" \
  -d "{\"model\":\"glm52-fp8-official\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20}"'
```

- `No available prefill workers (all circuits open or unhealthy)` → prefill
  down or router cached it unhealthy. Probe prefill `:30100/v1/models` and
  decode `:30200/v1/models` directly. If prefill is up but the router still
  refuses, **restart the router** (it caches health state; today's incident).
- prefill/decode `/v1/models` both 200 but router 503 → router stale → restart router.
- prefill down → Step 2 + "Restart PD pair" (prefill **and** decode together).

## Step 5 — garbled output

Prefill/decode/router were not restarted together → NIXL handshake mismatch →
corrupted KV transfer. Direct-to-prefill (bypass PD) outputs are correct, which
misleads you into thinking prefill is fine. Restart the **whole chain**:
prefill → decode → router (see [ops-playbook.md](ops-playbook.md) "Restart PD
pair"). Verify with a known-answer request (e.g. "1+1" or 17×24) through the
Proxy.

## Known-fault signature table

Symptom → root cause → fix → memory. Always confirm against the live log
before matching.

| Symptom / log signature | Root cause | Fix | memory |
|---|---|---|---|
| `zmq.error.ZMQError: No space left on device (addr='ipc:///tmp/...')` at vllm startup; `df -h /tmp` shows free space but `df -i /tmp` = 100% | `/tmp/torchinductor_root` piled up ~434k torch.compile cache files, exhausting the 491,520-inode cap on the 30G disk (inode exhaust, not disk full) | `rm -rf /tmp/torchinductor_root` (regenerable) then restart; long-term set `TORCHINDUCTOR_CACHE_DIR` to a large-inode partition | `mol-vllm-torchinductor-inode-exhaust` |
| `scheduler crashed with exit code -7` (SIGBUS) on real prefill; dmesg all 8 GPU Xid 31 + Xid 43, `gpuHandleSanityCheckRegReadError_GH100`; 0 ECC errors | GPU driver/GSP firmware in a bad state (system-level); process restart can't clear it, crash recurs | reboot the **whole machine** (cloud reboot); if it recurs after reboot → physical GPU failure → cloud ticket | `mol-sglang-gpu-xid-sigbus-crash` |
| `kill_process_tree called` after `All pyspy dump attempts failed`; 0 Xid; log has `RuntimeError: Prefill out of memory` + KV `full=yes` | VRAM OOM (software watchdog deadlock): `--mem-fraction-static 0.90` + EAGLE + 4 LoRA + hicache leaves no prefill headroom | lower `--mem-fraction-static` 0.90→0.85; soft-restart sglang only | `mol-sglang-watchdog-deadlock-crash` |
| `No space left on device` while sglang serving; `/tmp/hicache` (30G) full of KV-cache files | sglang writes `/tmp/hicache` despite `--file-storage-path /root/hicache`; the 30G disk fills under load | `ln -s /root/hicache /tmp/hicache`; verify the symlink **before every** sglang restart | `mol-sglang-hicache-tmp-symlink` |
| `tvm.error.InternalError ... NVCC compilation failed` ~1min into sglang startup | image has no `nvcc`; DeepGEMM JIT FP8 kernel needs it; startup script's `rm -rf deep_gemm cache` wipes the JIT cache | copy deep_gemm + tvm-ffi cache from a running same-arch (B300) host; comment out the `rm -rf`; restart | `mol-sglang-deepgemm-nvcc-cache` |
| PD requests 503 `no_available_workers` / garbled output after restarting **only** prefill (or only decode) | prefill restart changes NIXL engine_id; decode/router keep the old handshake → KV transfer corrupted; router registry stale | restart the **whole chain** prefill→decode→router; restart gateway if registry still stale | `mol-vllm-pd-restart-needs-full-chain` |
| gateway ERROR `Upstream stream error ... error decoding response body` (status 200 misleading) → Proxy `ClientPayloadError` → `answer_hop_failed` at ~30min | gateway `--request-timeout-secs` default 1800s = reqwest **total** timeout; long streams get cut (NOT `MOL_HOP_TIMEOUT`) | set gateway `--request-timeout-secs 86400` + Proxy `MOL_HOP_TIMEOUT=86400`; restart both | `mol-restart-gateway-proxy-full` |
| sglang prefill crash: `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0` at `dsa_backend.py set_dsa_prefill_impl max_kv_len = forward_batch.seq_lens_cpu.max().item()` | DSA prefill backend on an **empty batch** (degenerate/aborted request passes `seq_lens_cpu` with 0 elements); sglang self-bug | patch `init_forward_metadata` (guard `batch_size==0`) + `set_dsa_prefill_impl` (guard `numel()==0`); restart prefill+decode | commit `86747d1` on b300 sglang fork |
| sglang prefill: `Reasoning parsing error: Glm45Detector.__init__() got an unexpected keyword argument 'continue_final_message'` | MoL Proxy L0 route hop sends `continue_final_message: true`; sglang `Glm45Detector` doesn't accept the kwarg (sglang issue #30000) | patch `Glm45Detector.__init__` to accept `continue_final_message` + `previous_content`; restart prefill+decode | commit `86747d1` on b300 sglang fork |
| Proxy: `L0 model routing failed: upstream router returned HTTP 401` | develop smg enforces `--api-key` auth; Proxy doesn't inject upstream auth header | start gateway with `--api-key sk-glm52-pd`; patch Proxy `_routing_headers()` to inject `Authorization: Bearer $MOL_UPSTREAM_API_KEY`; set env `MOL_UPSTREAM_API_KEY=sk-glm52-pd` | see commands.md § B300 dev |
| **2P3D**: `Watchdog caught collective operation timeout: WorkNCCL(OpType=_ALLGATHER_BASE, Timeout(ms)=600000) ran for 600039ms` → SIGABRT (exit code -6); py-spy shows all ranks stuck in `cp_layersplit_pool.py broadcast_owner_layer_prefix` | `cp_layersplit_should_broadcast_prefix` gate had per-rank conditions (`extend_prefix_lens_cpu is not None` + `any(...)`); HiCache prefetch is local-only → radix tree diverges across ranks → some ranks enter NCCL broadcast, others skip → permanent call-count mismatch → deadlock | patch gate to only use rank-invariant conditions (`enable_dsa_prefill_cp_layersplit` + `is_extend_without_speculative()`); add None guard in `broadcast_owner_layer_prefix`; deploy to all 5 nodes + clear pycache/triton/deep_gemm | `2p3d-cp-layersplit-gate-deadlock` |
| **2P3D/1P1D**: public returns 503 `answer_hop_failed` / `no_available_workers` but router direct curl = 200 and all node `/health`=200 | smg gateway circuit breaker opened after transient failures (default: 10 failures → open, 60s cooldown); does NOT auto-recover reliably | **permanent fix**: start gateway with `--disable-circuit-breaker` flag (added to start_pd.sh gateway function); restart gateway (`GATEWAY_BIN=/usr/local/bin/smg bash /root/start_pd.sh gateway`) | `smg-circuit-breaker-stuck` |
| **2P3D**: public 18777 returns 503 `answer_hop_failed` but all node `/health`=200 and processes alive | a decode node was restarted; router marked it unhealthy and **never auto-recovers** (AGENTS.md known trap); smg gateway also caches stale health | restart router (`ROUTER_BIN=/opt/sglang-venv/bin/sglang-router bash /root/start_pd.sh router --prefill ... --decode ...`) then restart smg gateway (`GATEWAY_BIN=/usr/local/bin/smg bash /root/start_pd.sh gateway`); proxy (31000) does not need restart | `2p3d-router-unhealthy-no-autorecover` |
| **2P3D**: router starts and immediately exits (health=000, log shows only banner); `ps` shows no `sglang::router` process | `sglang-router` not in default PATH on 2P3D; `start_pd.sh` defaults `ROUTER_BIN=sglang-router` which silently fails | set `ROUTER_BIN=/opt/sglang-venv/bin/sglang-router` env var when launching router | `2p3d-router-not-in-path` |
| **2P3D**: `start_pd.sh decode` fails with `python3: command or file not found` or loads wrong sglang | start_pd.sh defaults `PYTHON=/root/sglang_venv/bin/python3` which does not exist on 2P3D nodes | set `PYTHON=/opt/sglang-venv/bin/python` env var when launching prefill/decode | `2p3d-python-path-mismatch` |
| **2P3D**: public 18777 chat times out (000, 60s) but 5 nodes all `/health`=200 and router `/health`=200; router direct curl = 000 (timeout) | **router 进程活着但转发卡死**（反复复发：23:30/07:30/13:00；进程 CPU 时间在涨但请求超时） | restart router（杀干净 0 残留 + 启动）；router 恢复后必须重启 gateway + proxy（它们缓存 router 状态）；最后确认公网 | `2p3d-router-hang` |
| **2P3D**: public 18777 times out but router direct = 200 and gateway = 200; proxy (31000) direct = 000 | **mol_harness proxy 进程活着但转发卡死**（02:30；重启 proxy 即恢复，重启 gateway 不够） | restart proxy：`MOL_API_KEY=... PROXY_SRC=/root/mol-stack bash /root/start_pd.sh proxy`；等 proxy 转发恢复后确认公网 | `2p3d-proxy-hang` |
| **2P3D**: prefill 卡死（health 200 但无 DCP-XFER 日志、公网超时）；py-spy 显示全部 scheduler 卡 `poll_and_all_reduce_attn_cp_tp_group` 的 all_reduce（prefill.py pop_bootstrapped，utils.py:138 vs 143 行号不一致） | **cross-collective 死锁**：per-rank collective 顺序不匹配（部分 rank 卡 all_reduce，对端已走下一步）→ 永久挂起 | 已由 `3df13dd42`（iteration barrier 移到 pop_bootstrapped 前）修复；若复发：py-spy dump 全部 scheduler 确认卡点，再查代码 | `2p3d-prefill-pop-bootstrapped-deadlock` |
| **2P3D 1101**: prefill 崩溃 `RuntimeError: gloo/tcp/pair.cc:547 Connection closed by peer [22.0.68.78]` + Scheduler hit an exception（07:38/11:28/12:40 三次，同一对端 IP） | 1101 与 22.0.68.78 之间的 **Gloo TCP 网络连接反复断连**（网络层问题，非代码） | 备份日志 → 重启整个集群；**根因待查**：排查 1101 与 22.0.68.78 的网络链路（网卡/防火墙/丢包） | `2p3d-1101-gloo-tcp-drop` |
| **2P3D**: 公网 chat 超时但 prefill/decode 日志**活跃**（DCP-XFER 持续、无 Scheduler exception） | **生产大请求队列积压**（输入 5万+ tokens 的请求在跑，测试小请求排后面超时）——不是故障 | 不重启；等队列消化后重测公网（通常恢复）。观察 prefill 日志最近 1 分钟有无 DCP-XFER 判断 | `2p3d-load-queue` |
| **2P3D**: 重启后 TPOT 从 23ms 暴涨到 46-80ms（Grafana `sglang:inter_token_latency_seconds` P50），decode 日志同时出现 `Using MoE kernel config from .../E=1024,...`（找到 config） | **本地 configs 目录含打包的错误 `E=1024,*L20D*.json`**：rsync 代码时被带到集群，重启时 sglang 找到它 → MoE kernel 按 1024 专家展开 → 性能暴跌。1P1D 无此文件（全部 `Config file not found` → 默认 kernel）→ 正常 7-21ms | `rm -f /opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/*L20D*.json`（5 节点）+ 重启 prefill/decode/router/gateway/proxy；**每次 rsync 后必须删** | `2p3d-triton-l20d-config-tpot` |

## What to report back

- problem class (which Step)
- strongest signal (exact log line)
- current best guess + what was ruled out
- action taken / proposed
- production risk (is traffic affected? which worker?)
