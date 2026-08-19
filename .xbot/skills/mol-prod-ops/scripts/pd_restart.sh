#!/usr/bin/env bash
# pd_restart.sh — restart a sglang PD pair (prefill + decode + router) together.
#
# WHY together: decode holds a NIXL handshake tied to prefill's engine_id.
# Restarting only one side corrupts KV transfer (garbled output + 503).
# See references/ops-playbook.md "Restart a PD pair".
#
# USAGE:
#   bash pd_restart.sh pd-router-1   # deploy-2 prefill + deploy-3 decode
#   bash pd_restart.sh pd-router-2   # deploy-1 prefill + deploy-4 decode
#   bash pd_restart.sh pd-router-1 router-only   # restart just the router
#
# Hard constraints: exact-PID kills only (no pkill), no tmux, logs to /root/tmp,
# readiness via /v1/models (not /health), mv old logs before restart.
#
# ⚠️ THIS MUTATES PRODUCTION. Only run with explicit user confirmation.
set -uo pipefail

ROUTER="${1:-}"
MODE="${2:-full}"

case "$ROUTER" in
  pd-router-1)
    PREFILL_HOST=mol-deploy-2; PREFILL_IP=10.0.58.21
    DECODE_HOST=mol-deploy-3;  DECODE_IP=10.0.58.20
    ;;
  pd-router-2)
    PREFILL_HOST=mol-deploy-1; PREFILL_IP=10.0.58.22
    DECODE_HOST=mol-deploy-4;  DECODE_IP=10.0.58.23
    ;;
  *)
    echo "Usage: $0 pd-router-1|pd-router-2 [router-only]" >&2
    exit 2
    ;;
esac

say() { printf '%s\n' "$*"; }
ts() { date +%Y%m%d-%H%M%S; }

poll_v1models() {  # <host> <port> <label>
  local host="$1" port="$2" label="$3"
  for i in $(seq 1 40); do
    code=$(ssh "$host" "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then say "  ${label} /v1/models -> 200 (ready)"; return 0; fi
    sleep 15
  done
  say "  ${label} /v1/models never reached 200 (last=$code)" >&2
  return 1
}

# Probe the PD chain end-to-end through the router. router /v1/models==200 is NOT
# enough: the router stays up (and reports healthy to the gateway) while prefill
# or decode is down, or while the PD handshake hasn't completed. A real request
# that returns a completion (has "choices") means prefill→decode transfer works.
probe_chain() {  # <host> <label>
  local host="$1" label="$2"
  for i in $(seq 1 20); do
    resp=$(ssh "$host" "curl -s -m 60 -X POST http://127.0.0.1:30000/v1/chat/completions \
      -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-glm52-pd' \
      -d '{\"model\":\"glm52-fp8-official\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'" 2>/dev/null)
    if printf '%s' "$resp" | grep -q '"choices"'; then
      say "  ${label} chain OK (real request returned a completion)"
      return 0
    fi
    sleep 10
  done
  say "  ${label} chain still 503 after retries (router cached prefill/decode unhealthy, or handshake not up)" >&2
  return 1
}

# ---- preflight: confirm ports / partner state ----
say "=== Preflight: $ROUTER (prefill=$PREFILL_HOST decode=$DECODE_HOST) ==="
if [ "$MODE" != "router-only" ]; then
  ssh "$PREFILL_HOST" "pgrep -f 'sglang.launch_server.*disaggregation-mode prefill' >/dev/null && echo '  prefill: RUNNING' || echo '  prefill: NOT running (expected if crashed)'"
  ssh "$DECODE_HOST"  "pgrep -f 'sglang.launch_server.*disaggregation-mode decode'  >/dev/null && echo '  decode: RUNNING'  || echo '  decode: NOT running'"
fi
# router port + prometheus port free?
ssh "$PREFILL_HOST" "lsof -i:30000 -i:29001 2>/dev/null | grep LISTEN || echo '  router ports 30000/29001 free'"

if [ "$MODE" = "router-only" ]; then
  say "=== router-only restart ==="
  # kill stale router holding 30000 — use lsof on the port (NOT pgrep on process
  # name): the running router is a Rust binary `sglang::router`, which
  # `pgrep -f sglang_router.launch_router` does NOT match.
  old_rpids=$(ssh "$PREFILL_HOST" "lsof -ti:30000 -sTCP:LISTEN 2>/dev/null")
  if [ -n "$old_rpids" ]; then
    for p in $old_rpids; do ssh "$PREFILL_HOST" "ps -p $p -o pid,etime,comm --no-headers; kill $p 2>/dev/null"; done
    sleep 3
    for p in $old_rpids; do ssh "$PREFILL_HOST" "kill -9 $p 2>/dev/null"; done
    say "  killed stale router(s): $old_rpids"
  fi
  ssh "$PREFILL_HOST" "mv /root/tmp/router.log /root/tmp/router.log.$(ts).bak 2>/dev/null; echo router.log backed up"
  ssh "$PREFILL_HOST" "tmux new-session -d -s pd-router 'cd /root && python3 -m sglang_router.launch_router \
    --pd-disaggregation --prefill http://${PREFILL_IP}:30100 --decode http://${DECODE_IP}:30200 \
    --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
    --policy cache_aware --max-concurrent-requests 32 --health-check-timeout-secs 300 \
    --disable-circuit-breaker --request-timeout-secs 600 --log-level info --prometheus-port 29001 \
    > /root/tmp/router.log 2>&1'; echo router tmux session started"
  poll_v1models "$PREFILL_HOST" 30000 "router"
  rc=$?
  sleep 15
  if [ $rc -eq 0 ]; then
    probe_chain "$PREFILL_HOST" "$ROUTER"
    rc=$?
  fi
  exit $rc
fi

# ---- full restart: prefill + decode together ----
say "=== Backing up logs ==="
ssh "$PREFILL_HOST" "mv /root/tmp/prefill.log /root/tmp/prefill.log.$(ts).bak 2>/dev/null; echo prefill.log backed up"
ssh "$DECODE_HOST"  "mv /root/tmp/decode.log  /root/tmp/decode.log.$(ts).bak  2>/dev/null; echo decode.log backed up"

# stop both engines (exact PIDs)
say "=== Stopping prefill + decode ==="
for spec in "$PREFILL_HOST prefill" "$DECODE_HOST decode"; do
  set -- $spec; h=$1; mode=$2
  pids=$(ssh "$h" "pgrep -f \"sglang.launch_server.*disaggregation-mode $mode\"")
  for p in $pids; do ssh "$h" "kill $p 2>/dev/null"; done
  sleep 3
  for p in $pids; do ssh "$h" "kill -9 $p 2>/dev/null"; done
  # orphan TP workers hold GPU mem — clear them
  ssh "$h" "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null; echo '  GPU compute-apps cleared'"
done

say "=== Starting prefill (decode waits 60s — let prefill DeepGEMM warmup + bootstrap 8998 first) ==="
ssh "$PREFILL_HOST" "tmux new-session -d -s pd-prefill 'cd /root && bash /root/recover_b300_pd.sh prefill > /root/tmp/prefill.log 2>&1'; echo prefill tmux session started"

say "=== Waiting 60s for prefill bootstrap before starting decode ==="
sleep 60

say "=== Starting decode ==="
ssh "$DECODE_HOST"  "tmux new-session -d -s pd-decode 'cd /root && bash /root/recover_b300_pd.sh decode > /root/tmp/decode.log 2>&1'; echo decode tmux session started"

say "=== Polling readiness (/v1/models) — DeepGEMM warmup ~1-2min ==="
poll_v1models "$PREFILL_HOST" 30100 "prefill"
pf_rc=$?
poll_v1models "$DECODE_HOST"  30200 "decode"
dc_rc=$?
if [ $pf_rc -ne 0 ] || [ $dc_rc -ne 0 ]; then
  say "FAIL: prefill or decode did not become ready" >&2
  exit 1
fi

say "=== Starting router (detached tmux) ==="
# clear any stale router holding 30000 — use lsof on the port, NOT pgrep on the
# process name: the running router is a Rust binary named `sglang::router`, which
# `pgrep -f sglang_router.launch_router` does NOT match (that only matches the
# python launcher). Matching by port catches both forms.
old_rpids=$(ssh "$PREFILL_HOST" "lsof -ti:30000 -sTCP:LISTEN 2>/dev/null")
if [ -n "$old_rpids" ]; then
  for p in $old_rpids; do
    ssh "$PREFILL_HOST" "ps -p $p -o pid,etime,comm --no-headers; kill $p 2>/dev/null"
  done
  sleep 3
  for p in $old_rpids; do ssh "$PREFILL_HOST" "kill -9 $p 2>/dev/null"; done
  say "  killed stale router(s): $old_rpids"
fi
ssh "$PREFILL_HOST" "tmux new-session -d -s pd-router 'cd /root && python3 -m sglang_router.launch_router \
  --pd-disaggregation --prefill http://${PREFILL_IP}:30100 --decode http://${DECODE_IP}:30200 \
  --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
  --policy cache_aware --max-concurrent-requests 32 --health-check-timeout-secs 300 \
  --disable-circuit-breaker --request-timeout-secs 600 --log-level info --prometheus-port 29001 \
  > /root/tmp/router.log 2>&1'; echo router tmux session started"
poll_v1models "$PREFILL_HOST" 30000 "router"
r_rc=$?
# router /v1/models is up — now verify the PD chain actually works (router 200
# does not mean prefill↔decode handshake is up). Wait for the router to settle on
# worker health before probing.
sleep 15
if [ $r_rc -eq 0 ]; then
  probe_chain "$PREFILL_HOST" "$ROUTER"
  r_rc=$?
fi

say ""
if [ $r_rc -eq 0 ]; then
  say "RESULT: $ROUTER restarted — prefill+decode+router up, PD chain verified."
  exit 0
else
  say "RESULT: router up but PD chain not working — check /root/tmp/router.log on $PREFILL_HOST" >&2
  exit 1
fi
