---
name: mol-prod-ops
description: MoL (Mixture-of-LoRA) production ops runbook for the 5-node hybrid sglang-PD + vllm cluster AND the separate 2P3D cluster (8.222.11.182:1100-1104). Use when asked to report production status, add or remove a worker, restart a PD pair / prefill / decode / router, restart a vllm worker, replace the Proxy, restart the gateway or proxy, or triage a worker that is down, crashed, unhealthy, returning 503, or producing garbled output. Covers gateway:30001 + Proxy:30000 on deploy-0, two sglang PD pairs (pd-router-1=deploy-2/3, pd-router-2=deploy-1/4), the deploy-0 vllm worker, and the 2P3D cluster (2 prefill + 3 decode, public entry 8.222.11.182:18777, key $MOL_API_KEY_2P3D, model Macaron-V1-Venti).
---

# MoL Production Ops

Runbook for operating the mol production cluster. This is a **mutation-aware**
skill: status checks are safe; anything that stops/starts a process or
re-registers a worker touches production and needs care.

## Hard constraints (always hold)

These override any default and any tool suggestion:

- **No `pkill` / `killall` / `pgrep -f` for killing.** Use exact-PID `kill
  <pid>`, `kill -9 <pid>` fallback, after `ps -p <pid>` confirms the target.
  Fuzzy matches kill unrelated processes (ssh/bash/system).
- **⛔ RESTART MUST KILL CLEANLY (hard rule, 2026-08-04 incident):** before
  starting any sglang worker, kill by BOTH `lsof -ti :<port> | xargs kill -9`
  AND `ps aux | grep launch_server` (all PIDs), then **verify 0 processes +
  0 port listeners remain**. Never start a new worker while an old one may
  still hold the port (old pid survives → new process fails to bind → old
  process runs until it crashes → worker appears "restarted" then dies).
  After start, confirm the **new PID differs from the old PID** and
  `/v1/models` returns the model list. If unsure, repeat kill+verify until
  clean.
- **No tmux.** Background with `nohup` / `setsid`; record the PID (`echo $!`).
- **Temp files and logs go to `/root/tmp/`, never `/tmp`.**
- **ssh output → redirect to a file, then Read it.** Do not `ssh ... | grep`
  (the PAI banner corrupts stdout).
- **Worker readiness = `/v1/models` (200 + model list), never `/health`.**
  `/health` returns an empty body and is indistinguishable from "still loading".
- **Back up logs before a restart:** `mv foo.log foo.log.<ts>.bak`.
- **All 5 deploy nodes are production.** Confirm the target before acting;
  production-side mutations (stop/start engine, replace Proxy, re-register
  worker) need explicit user confirmation.
- Verify file sync with `ls`/`wc`/`md5sum` after rsync (heredoc/base64 silently
  fail in PAI). The dev box cannot reach 10.0.58.x — run worker curls on
  deploy-0 or the worker node over ssh.

## Standard flow

1. **Get current state** — `bash scripts/prod_status.sh`. This reports
   gateway, Proxy, and every worker (name/url/runtime/healthy/load).
2. **Classify** — if a worker is missing, unhealthy, 503ing, or garbling
   output, read [references/fault-triage.md](references/fault-triage.md) and
   follow the decision tree. The key trap: a sglang router returns `/health`
   200 while its prefill is dead, so the gateway looks healthy but requests 503.
3. **Act** — for a mutation, follow the matching section in
   [references/ops-playbook.md](references/ops-playbook.md). Use the helper
   scripts instead of hand-rolling commands.
4. **Verify** — `bash scripts/verify_3hop.sh` (identity + stream usage +
   per-worker `/v1/models`). Do not declare success on `/health` alone.
5. **Report** — state, what changed, verification result, production risk.

## References (load only what the step needs)

- [references/topology.md](references/topology.md) — nodes, IPs, ports, roles,
  node boundary, auth/api_key.
- [references/ops-playbook.md](references/ops-playbook.md) — add/remove worker,
  replace Proxy, restart PD pair, restart vllm, restart gateway+proxy.
- [references/fault-triage.md](references/fault-triage.md) — symptom → step
  decision tree + known-fault signature table (symptom → cause → fix → memory).
- [references/commands.md](references/commands.md) — script paths, PD env
  blueprint, full Proxy env, gateway flags, verification commands.

## Scripts

- `scripts/prod_status.sh` — one-shot gateway/Proxy/workers status report.
- `scripts/pd_restart.sh <pd-router-1|pd-router-2> [router-only]` — restart a
  PD pair (prefill+decode+router) together, or just the router. **Mutates prod.**
- `scripts/verify_3hop.sh` — end-to-end gate (identity, stream usage,
  per-worker readiness). Read-only.

## 2P3D cluster (8.222.11.182:1100–1104)

A **separate** sglang PD cluster on five L20D VF nodes, distinct from the
mol-prod deploy-0–4 cluster. All topology, paths, restart commands, and known
traps are in
[topology.md § "2P3D cluster"](references/topology.md#2p3d-cluster-822211182-ssh-ports-11001104)
and [fault-triage.md](references/fault-triage.md) (search `2P3D`).

### venv 真相（2026-08-19 实测，勿混淆）

| venv | 节点 | 状态 |
|---|---|---|
| `/root/sglang_venv` | **1102、1104** | ✅ 当前生产用（1102 全量 rsync 到 1104，双端 md5 一致；含全部 LoRA/CP 修复 + sglang-kernel 0.4.5） |
| `/opt/sglang-venv` | 1101（也在 1102/1104 有残留） | 旧 0.5.15.post1；1101 仍用它跑 2P3D prefill；1102/1104 上**勿用**（sglang-kernel 0.4.4 会触发版本断言） |
| `/root/v15_patched` | 1104 | 过渡用，已被 `/root/sglang_venv` 取代 |

节点直连：1102→1104 已配免密（`ssh root@10.0.58.37`），venv/权重同步走节点间 rsync（内网 ~580MB/s）。

### 当前部署形态

- **1102+1104 = 1P1D 4-LoRA 集群**（2026-08-19 起，`/root/start_1p1d_lora.sh` 双端）：
  prefill 1102:30100（TP8 + CP interleave + VE + L0-L3，mem-fraction 0.85，**无 HiCache/layersplit**）；
  decode 1104:30200（TP8 + DCP=4 + EAGLE5 + VE，0.90）；router 1102:30000
  （`/opt/sglang-venv/bin/sglang-router`，cache_aware，key `sk-glm52-pd`）。
  验证过 base+L0-L3 全通、bench L2（8K/1K×16 并发）TPOT p50 17.5ms。
- **1101+1100/1103 = 老 2P3D**（/opt/sglang-venv 旧代码）：1101 的 prefill 配置
  （HiCache ratio=1 + layersplit）在 B300 上**必然 OOM**（需求恒定 ~268GB，与
  mem-fraction 无关）——1101 反复崩的根因；勿照抄其参数。

### 2026-08-19 关键修复（本地 git，已部署 1102/1104）

- `c1946917cb` **CP×LoRA IMA 根修**：csgmv batch_info 覆盖切分前全量 token 而
  dense/attn 层跑 CP shard（~1/cp 行）→ permutation OOB。修复=按 parallel
  runtime 重建 shard 视图（round-robin 模式 metadata 为空对象，勿 keying on it）。
  之前怀疑的 VE kernel/gate GEMM/DSA topk 全是 sticky IMA 浮出点。
- `0eccdf831c` LoRA sync-load（load_stream 竞态损坏首次前向权重）。
- VE flag（`--lora-use-virtual-experts`）**必须带**：缺失走无加固 classic
  `fused_moe_lora` → 首个 LoRA 请求必崩（20h 排查教训：对照前先 diff flags）。
- LoRA L2 变体：2P3D 老节点用 `L2_0818001`；1102/1104 无此目录，用老 `L2`。

### 通用铁律（不变）

- 公网入口 `8.222.11.182:18777`，key `$MOL_API_KEY_2P3D`（真值见仓库根 secrets.env）
- **每次重启必须同步最新代码**（rsync 本地 `python/sglang/` → 双端 venv）+ 清
  `__pycache__` + md5 抽查关键文件
- **⛔ rsync 后必须删除 L20D triton config**（TPOT 23ms→80ms 杀手）：
  `rm -f <venv>/lib/python3.12/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/*L20D*.json`
- 代码版本 = 本地 git HEAD（skill 不再记录易过期 hash；关键修复 commit 见上节）
### 1P2D decode 崩溃链（2026-08-19 全部结案）

1. **flashinfer-cubin 不匹配**（v15_patched 专属）：`system_packages.pth` 暴露系统
   `flashinfer_cubin 0.6.12` 给 venv `flashinfer 0.6.14` → trtllm kernel IMA。
   修复 = 删系统 cubin 目录+dist-info、从 1102 完整同步 flashinfer（含 88M
   `data/` 内嵌 cubin）。判据：`import flashinfer` 报 cubin version mismatch。
2. **transfer 风暴 IMA**（commit `8bfe264118`）：`transform_index_page_table_decode_fast`
   只 mask `topk<0` 缺上界 → 40+ 并发 KV 传输下 indexer 竞态越界 → OOB。
   判据：并发 burst 崩一个 decode，栈在 `_forward_trtllm → transform_index`；
   复现脚本 `/root/bench_case50/repro_storm.py`（40×11K 并发）。

- 已知故障模式：router/proxy 进程活着但转发卡死（重启即恢复）、1101 Gloo TCP
  断连崩溃（22.0.68.78，根因待查）——详见 fault-triage.md

## Out of scope

- Release switch (6e5b010 → develop, dual-gateway) — see the plan file, not here.
- Fresh-image PD deploy / recover-from-reimage — see
  `.md/sglang-pd-deploy-notes.md` and `.md/sglang-pd-recover-after-reimage.md`.
- Benchmarking / capacity planning.
