# Ops Playbook

Mutating production operations. Each assumes the hard constraints in `SKILL.md`
hold: exact-PID kills, no tmux, temp in `/root/tmp/`, `/v1/models` (not
`/health`) for readiness, `mv` logs before restart, user confirmation for
production-side changes.

Read [topology.md](topology.md) for node/IP/port resolution and
[commands.md](commands.md) for full env + script paths.

## Add a worker

Register against the gateway on deploy-0 (`127.0.0.1:30001`):

```bash
# sglang PD router
ssh mol-deploy-0 'curl -s -X POST http://127.0.0.1:30001/workers \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"http://10.0.58.21:30000\",\"model_id\":\"glm52-fp8-official\",\"worker_type\":\"regular\",\"runtime\":\"sglang\",\"priority\":50,\"cost\":1.0,\"api_key\":\"sk-glm52-pd\",\"labels\":{\"deployment\":\"macaron-v1-venti\",\"name\":\"mol-deploy-pd-router-1\"}}"'

# vllm worker (no api_key)
ssh mol-deploy-0 'curl -s -X POST http://127.0.0.1:30001/workers \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"http://10.0.58.3:8000\",\"model_id\":\"glm52-fp8-official\",\"worker_type\":\"regular\",\"runtime\":\"vllm\",\"priority\":50,\"cost\":1.0,\"labels\":{\"deployment\":\"macaron-v1-venti\",\"name\":\"mol-deploy-0\"}}"'
```

- For sglang workers the `api_key` field is **required** (router 401s otherwise).
- Wait for healthy: gateway health-checks each worker's `/health` every 60s;
  usually flips healthy within ~8s, worst case one full interval.

## Remove a worker

`DELETE /workers/{id}` — drains immediately, inflight requests complete:

```bash
# find the worker id
ssh mol-deploy-0 'curl -s http://127.0.0.1:30001/workers' | python3 -c \
  "import sys,json; [print(w['id'], w['metadata'].get('name'), w['url']) for w in json.load(sys.stdin)['workers']]"
# remove
ssh mol-deploy-0 'curl -s -X DELETE http://127.0.0.1:30001/workers/<id>'
```

To drain a node without deregistering (e.g. take it offline), stop its engine
process; the gateway marks it unhealthy after a few failed health checks and
stops routing to it. Removing from `/workers` is for a clean decommission.

## Replace the Proxy (Python-only change)

Pure-Python Proxy changes (new commit, identity prompt, routing) do **not**
require a gateway restart — Proxy and gateway are decoupled (Proxy →
`UPSTREAM=127.0.0.1:30001` → gateway). Restarting the gateway drops all worker
registrations and causes traffic jitter.

1. Build the new release dir on deploy-0 (reuse the old `bin/` + gateway
   binary): `/root/mol-stack-releases/<DATE>-<HASH>-<tag>/`.
2. rsync the code from dev box: `mol_harness/` **and** `lora_library/` (the
   identity routing rules live in `lora_library/mol_glm52/*.md`; forgetting
   `lora_library/` silently breaks identity routing).
3. Stop the old Proxy:
   ```bash
   ssh mol-deploy-0 'pid=$(pgrep -f mol_harness.proxy | head -1); ps -p $pid; kill $pid; sleep 3; kill -9 $pid 2>/dev/null'
   ```
   SIGTERM often does not exit — `kill -9` is the fallback.
4. Relaunch with the new PYTHONPATH + full MOL_* env (see [commands.md](commands.md)):
   ```bash
   ssh mol-deploy-0 'setsid nohup env PYTHONPATH=/root/mol-stack-releases/<NEW> ... python3 -m mol_harness.proxy > /root/tmp/mol_proxy_<hash>.log 2>&1 < /dev/null &'
   ```
5. Verify with `verify_3hop.sh`.

Restart the gateway **only** for Rust/config changes (see "Restart gateway +
proxy" below).

## Restart a PD pair

⚠️ prefill + decode **must restart together** — decode holds a NIXL handshake
tied to the prefill's engine_id; restarting only one corrupts KV transfer
(garbled output + 503). Restarting only the crashed side is a common mistake.

- Default: use `scripts/pd_restart.sh pd-router-1` (or `pd-router-2`). It backs
  up logs, restarts prefill + decode together, polls `/v1/models`, then starts
  the router.
- Start order: **prefill → decode → router**. prefill needs ~1–2min of
  DeepGEMM warmup before decode can connect to bootstrap 8998.
- The **router** runs on the prefill node. Start it with
  `nohup python3 -m sglang_router.launch_router ...` directly — **not** via
  `setsid bash recover_b300_pd.sh router` (the Rust binary panics / fails to
  bind under setsid). Confirm 30000 + 29001 are free first (`lsof -i:30000
  -i:29001`); kill stale PIDs with exact `kill`.
- After a prefill crash, even if decode is still up, restart **both**. If the
  router was running while prefill was down it may have cached prefill as
  unhealthy and keep returning `No available prefill workers` even after
  prefill is back — **restart the router** to clear it (today's incident).
- Reuse `scripts/pd_restart.sh`; manual commands in [commands.md](commands.md).

## Restart a vllm worker (deploy-0)

1. `kill <pid>` the main `vllm.entrypoints` process, `kill -9` fallback.
2. **Also kill the 8 TP worker subprocesses** — killing the main process
   orphans them and they keep holding ~256GB GPU memory:
   ```bash
   ssh mol-deploy-0 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9'
   ```
   Confirm `nvidia-smi` shows all 8 cards at 0 MiB before relaunch.
3. Relaunch with the flock launch script (TP8/DCP4). If restart reports
   `zmq ZMQError: No space left on device`, check `df -i /tmp` — it's inode
   exhaustion, not disk (see [fault-triage.md](fault-triage.md)).

## Restart gateway + proxy

Needed when the Rust gateway binary or its config changes (e.g.
`--request-timeout-secs`), or to clear a stuck worker registry.

- Gateway **restart drops all workers** — after it is back up, re-register every
  worker via `POST /workers` (see "Add a worker"). Budget ~1 health-check
  interval per worker.
- gateway must run with `--request-timeout-secs 86400` (default 1800s cuts
  streams >30min — see fault-triage). Proxy needs `MOL_HOP_TIMEOUT=86400`.
- SIGTERM often won't exit either; `kill -9` after a short wait.

Order: stop Proxy → stop gateway → start gateway (86400) → re-register workers
→ start Proxy (86400) → `verify_3hop.sh`.

### B300 dev gateway (develop branch)

The B300 dev gateway uses a **different binary** and **different ports** from
production deploy-0. See [commands.md](commands.md) § "B300 dev gateway + proxy"
for the full start sequence. Key differences:

- Binary: `/root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway/target/release/smg`
  (compiled from develop branch; production uses `/root/mol-stack-dev/bin/smg`)
- Port: 31001 (production: 30001)
- `--api-key sk-glm52-pd` is **required** (develop smg enforces auth)
- Proxy port: 31000, `UPSTREAM=http://127.0.0.1:31001`
- Proxy needs `MOL_UPSTREAM_API_KEY=sk-glm52-pd` env + one-line patch in
  `_routing_headers()` to inject the upstream auth header

## Verification gate

After any restart/replace, run `scripts/verify_3hop.sh` (identity + stream
usage + per-worker `/v1/models`). Do not declare success on `/health` alone.
