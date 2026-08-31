# Commands & Env

Authoritative paths, env blueprints, and verification commands. These mirror
the live processes (verified 2026-07-29), not memory.

## Paths (on each node, under `/root`)

| what | path |
|---|---|
| PD launch script (prefill/decode/router) | `/root/recover_b300_pd.sh` (arg: `prefill` \| `decode` \| `router`) |
| sglang code (system python) | `/usr/local/lib/python3.12/dist-packages/sglang/srt/` |
| sglang code (venv, if present) | `/opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/` |
| MoL stack (dev/test) | `/root/Mixture-of-LoRA-Harness-alpha/` (develop branch) |
| MoL stack (prod) | `/root/mol-stack-releases/mol-release-411303d/` |
| OTel Collector | `/usr/local/bin/otelcol` + `/root/otel-collector-config.yaml` + `/root/otel-collector.sh` |
| Gateway binary (develop) | `/root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway/target/release/smg` |
| Gateway binary (prod) | `/root/mol-stack-releases/mol-release-411303d/bin/smg` |
| Proxy code | `mol_harness/proxy.py` inside MoL stack dir |

## Startup scripts (in skill assets)

| script | use case |
|---|---|
| `assets/start_pd_docker.sh` | 1P1D Docker deployment |
| `assets/start_1p4d.sh` | 1P4D bare-metal deployment (1 prefill + 4 decode) |
| `assets/otel-collector-config.yaml` | OTel Collector config (scrape prefill:30100 + decode:30200) |
| `assets/otel-collector.sh` | OTel Collector manage script (start/stop/restart/status/logs) |

To deploy scripts to a node:
```bash
cat assets/start_pd_docker.sh | ssh -p <PORT> root@<IP> "cat > /root/start_pd_docker.sh && chmod +x /root/start_pd_docker.sh"
cat assets/otel-collector.sh | ssh -p <PORT> root@<IP> "cat > /root/otel-collector.sh && chmod +x /root/otel-collector.sh"
cat assets/otel-collector-config.yaml | ssh -p <PORT> root@<IP> "cat > /root/otel-collector-config.yaml"
```

⚠️ OTel config has hardcoded IPs — update `10.0.0.75:30100` (prefill) and `10.0.0.67:30200` (decode) to match current node IPs before deploying.

## Building the sglang Docker image

See the `sglang-docker-build` skill for full build/push instructions.
The image includes: sglang 0.5.15.post1 + all B300 patches + MoL gateway + proxy + hicache .so.

```bash
# Quick reference — see sglang-docker-build skill for details
sshpass -p 'hvwjzugsvbbzgA458$s' ssh root@47.87.64.67 \
  "cd /root/docker_build && docker build -t sglang-b300:v0.5.15 . 2>&1 | tail -5"
```

## OTel Collector deployment (one-time per node)

```bash
# Install otelcol binary
cd /tmp && curl -sL -o otelcol.tar.gz \
  'https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.129.1/otelcol-contrib_0.129.1_linux_amd64.tar.gz' \
  && tar xzf otelcol.tar.gz otelcol-contrib \
  && mv otelcol-contrib /usr/local/bin/otelcol && chmod +x /usr/local/bin/otelcol

# Deploy config + script (update IPs first!)
cat assets/otel-collector-config.yaml | ... > /root/otel-collector-config.yaml
cat assets/otel-collector.sh | ... > /root/otel-collector.sh && chmod +x

# Start
bash /root/otel-collector.sh start
```
| PD logs | `/root/tmp/prefill.log`, `/root/tmp/decode.log`, `/root/tmp/router.log` |
| vllm launch script (deploy-0) | `/root/tmp/start_glm52_b300_tp8dcp4_host_all.sh` (flock launch) |
| Proxy launch script (deploy-0) | `/root/tmp/start_proxy_<hash>.sh` |
| Proxy release dir (deploy-0) | `/root/mol-stack-releases/<DATE>-<HASH>-<tag>/` (contains `mol_harness/` + `lora_library/` + `bin/`) |
| weights | `/root/glm52_local/base` + `/root/glm52_local/loras/L{0,1,2,3}` |
| HiCache L3 storage | `/root/hicache` (11T disk); symlinked from `/tmp/hicache` |

## PD env blueprint (exported by `recover_b300_pd.sh`, both ends)

```
TVM_FFI_CUDA_ARCH_LIST=10.0a          # B300/SM103 — set every launch; clear deep_gemm cache
MOONCAKE_DISABLE_HIP_DMABUF=1
IBV_ACCESS_RELAXED_ORDERING=1
MC_IB_PCI_RELAXED_ORDERING=1
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1   # CP8 — without it TP1-7 hang at Bootstrapping
SGLANG_MOE_PADDING=1
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G
```

The actual running flags (prefill/decode/router) live in `.md/sglang-pd-deploy-notes.md`
§2 — trust `/proc/<pid>/cmdline` over the script file (they can drift).

## PD start commands

Start order: **prefill → decode → router**.

```bash
# prefill (on the prefill node, e.g. deploy-2 for pd-router-1)
ssh mol-deploy-2 'cd /root; nohup bash /root/recover_b300_pd.sh prefill > /root/tmp/prefill.log 2>&1 < /dev/null & echo "prefill pid $!"'
# decode (on the decode node, e.g. deploy-3) — after prefill /v1/models is 200
ssh mol-deploy-3 'cd /root; nohup bash /root/recover_b300_pd.sh decode > /root/tmp/decode.log 2>&1 < /dev/null & echo "decode pid $!"'
# router (on the prefill node) — NOT via the script's router branch (panics under setsid)
ssh mol-deploy-2 'lsof -i:30000 -i:29001 2>/dev/null | grep LISTEN || echo "ports free"
nohup python3 -m sglang_router.launch_router \
  --pd-disaggregation --prefill http://10.0.58.21:30100 --decode http://10.0.58.20:30200 \
  --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
  --policy cache_aware --max-concurrent-requests 32 --health-check-timeout-secs 300 \
  --disable-circuit-breaker --request-timeout-secs 600 --log-level info --prometheus-port 29001 \
  > /root/tmp/router.log 2>&1 < /dev/null & echo "router pid $!"'
```

For pd-router-2, swap IPs: prefill=deploy-1 (10.0.58.22), decode=deploy-4
(10.0.58.23).

## Proxy env (full, as run in production)

```bash
NEW=/root/mol-stack-releases/<DATE>-<HASH>-<tag>
export PYTHONPATH="$NEW"
export UPSTREAM="http://127.0.0.1:30001"
export PROXY_PORT=30000
export LIBRARY_DIR="$NEW/lora_library/mol_glm52"
export MOL_METRICS_PORT=8201
export MOL_USE_MODEL_ROUTER=1
export MOL_PURE_MODEL_ROUTE=1
export MOL_HOP_TIMEOUT=86400
export MOL_SSE_KEEPALIVE_INTERVAL_S=10
export MOL_MAX_CONNECTIONS=8192
export MOL_MAX_INFLIGHT_REQUESTS=512
export MOL_MAX_CONVOS=5000
export MOL_MAX_RESPONSES=5000
export MOL_MAX_QUEUED_REQUESTS=1024
export MOL_MAX_PENDING_RESPONSES=512
export MOL_MAX_REQUEST_BYTES=33554432
export MOL_MAX_RESPONSE_STATE_BYTES=2147483648
export MOL_MAX_PENDING_RESPONSE_STATE_BYTES=2147483648
export MOL_QUEUE_TIMEOUT_S=30
export MOL_DRAIN_TIMEOUT_S=600
export MOL_KEEPALIVE_TIMEOUT_S=15
export MOL_STATE_EXECUTOR_WORKERS=8
export MOL_STORE_CLEANUP_INTERVAL_S=5
export MOL_RELEASE_WORKERS=16
export MOL_RELEASE_QUEUE_SIZE=8192
export MOL_RELEASE_BATCH_SIZE=512
export MOL_RELEASE_TIMEOUT_S=10
export MOL_UPSTREAM_MAX_CONNECTIONS=1024
export MOL_UPSTREAM_MAX_KEEPALIVE=512
exec setsid /usr/bin/python3 -m mol_harness.proxy
```

## Gateway launch flags (deploy-0)

```
smg launch --host 0.0.0.0 --port 30001 --prometheus-port 29000 \
  --policy manual --assignment-mode min_load_then_group \
  --max-idle-secs 1800 --request-timeout-secs 86400
```

`--request-timeout-secs 86400` is mandatory (default 1800s cuts >30min streams).

## B300 dev gateway + proxy (develop branch)

The B300 source/reference machine (`ssh -p 1021 root@8.213.214.14`) runs a
**develop-branch** MoL stack at `/root/Mixture-of-LoRA-Harness-alpha/`. This is
used for compatibility testing before promoting to production deploy-0.

### Prerequisites (one-time, on B300-1)

```bash
# Rust toolchain + build deps (for compiling smg from source)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
apt-get install -y pkg-config libssl-dev protobuf-compiler
cd /root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway && cargo build --release
```

### Start gateway (B300-1, port 31001)

```bash
# kill old gateway first (exact PID from ps)
nohup /root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway/target/release/smg launch \
  --host 127.0.0.1 --port 31001 --prometheus-port 29002 \
  --policy manual --assignment-mode min_load_then_group \
  --worker-urls http://127.0.0.1:30000 \
  --api-key sk-glm52-pd \
  --max-idle-secs 1800 --request-timeout-secs 86400 \
  > /root/tmp/gateway.log 2>&1 < /dev/null & echo "gateway pid $!"
```

⚠️ `--api-key sk-glm52-pd` is **required** — the develop-branch smg enforces
auth on all `/v1/chat/completions` requests. Without it, the Proxy's L0 route
hop gets 401.

### Start Proxy (B300-1, port 31000)

The develop-branch `proxy.py` needs a one-line patch to inject
`MOL_UPSTREAM_API_KEY` into forwarded headers (upstream auth for the gateway):

```python
# In mol_harness/proxy.py, after PROXY_PORT definition:
UPSTREAM_API_KEY = os.environ.get("MOL_UPSTREAM_API_KEY", "").strip()

# In _routing_headers(), before return:
if UPSTREAM_API_KEY:
    headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
```

```bash
kill <old-proxy-pid>   # exact PID
export PYTHONPATH=/root/Mixture-of-LoRA-Harness-alpha
export UPSTREAM=http://127.0.0.1:31001
export MOL_API_KEY=${MOL_API_KEY_1P1D}   # 真值见仓库根 secrets.env (gitignored)
export MOL_UPSTREAM_API_KEY=sk-glm52-pd
export PROXY_PORT=31000
export MOL_USE_MODEL_ROUTER=1
export MOL_PURE_MODEL_ROUTE=1
export MOL_HOP_TIMEOUT=86400
export MOL_SSE_KEEPALIVE_INTERVAL_S=10
export MOL_MAX_CONNECTIONS=8192
export MOL_MAX_INFLIGHT_REQUESTS=512
export MOL_UPSTREAM_MAX_CONNECTIONS=1024
export MOL_UPSTREAM_MAX_KEEPALIVE=512
nohup /usr/bin/python3 -m mol_harness.proxy > /root/tmp/proxy.log 2>&1 < /dev/null & echo "proxy pid $!"
```

### Verify (B300-1)

```bash
# gateway readiness
curl -s http://127.0.0.1:31001/readiness   # expect {"status":"ready",...}

# end-to-end through Proxy
curl -s -m 60 http://127.0.0.1:31000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${MOL_API_KEY}" \
  -d '{"model":"Macaron-V1-Venti","messages":[{"role":"user","content":"1+1等于几?只回答数字"}],"max_tokens":64}'
# expect: content "2", route L0, pure_model_route
```

### sglang fixes required on B300-1 (for continue_final_message)

The sglang at `/usr/local/lib/python3.12/dist-packages/sglang/` must have:

1. **`srt/parser/reasoning_parser.py`** — `Glm45Detector.__init__` must accept
   `continue_final_message` and `previous_content` kwargs (sglang issue #30000).
   Without this, the L0 route hop crashes with `TypeError`.
2. **`srt/layers/attention/dsa_backend.py`** — `init_forward_metadata` must
   guard `batch_size == 0` and `set_dsa_prefill_impl` must guard
   `seq_lens_cpu.numel() == 0` (empty tensor `max()` crash).

These are in the b300 sglang fork at `/home/user/src/sglang_b300_decode`
(commit `86747d1`). Sync with:
```bash
scp srt/parser/reasoning_parser.py root@8.213.214.14:/usr/local/lib/python3.12/dist-packages/sglang/srt/parser/
scp srt/layers/attention/dsa_backend.py root@8.213.214.14:/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/
```
After syncing, **restart prefill + decode** for the fix to take effect.

## Verification commands

```bash
# per-worker readiness — /v1/models, NEVER /health
ssh mol-deploy-0 'curl -s -o /dev/null -w "%{http_code}\n" http://10.0.58.3:8000/v1/models'        # vllm
ssh mol-deploy-2 'curl -s -o /dev/null -w "%{http_code}\n" http://10.0.58.21:30100/v1/models'      # pd-router-1 prefill
ssh mol-deploy-3 'curl -s -o /dev/null -w "%{http_code}\n" http://10.0.58.20:30200/v1/models'      # pd-router-1 decode

# PD direct (through router, with api_key) — L0..L3
curl http://10.0.58.21:30000/v1/chat/completions -H "Authorization: Bearer sk-glm52-pd" \
  -d '{"model":"L2","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":32}'
# note: sglang reasoning is in reasoning_content (vllm uses reasoning);
# empty content with small max_tokens is normal (reasoning not finished)

# end-to-end through the Proxy — identity + stream usage
ssh mol-deploy-0 'curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Macaron-V1-Venti\",\"messages\":[{\"role\":\"user\",\"content\":\"你是什么模型?\"}],\"max_tokens\":400}"'
# expect: route L0 → answer "Macaron-V1-Venti 748B"
```

`verify_3hop.sh` automates the Proxy identity + stream-usage + per-worker checks.
