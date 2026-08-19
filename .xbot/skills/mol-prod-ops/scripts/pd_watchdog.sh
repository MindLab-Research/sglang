#!/usr/bin/env bash
# pd_watchdog.sh — auto-heal loop for sglang PD workers.
#
# Every 60s, for each sglang PD worker registered in the gateway, send a real
# inference request through its router. If it 503s / times out (prefill or decode
# crashed while the router stayed up — the gateway still reports healthy), the
# watchdog:
#   1. drains the worker from the gateway (so traffic stops 503ing on it),
#   2. restarts the whole PD pair (prefill+decode+router) via pd_restart.sh,
#   3. verifies the chain, then re-registers it to the gateway.
# Retries indefinitely with backoff (the root cause may recur).
#
# Only sglang PD workers are watched. vllm workers are left to the gateway's own
# health check. PD workers not currently registered (already drained) are skipped
# — they'll be re-registered by a recovery in progress, or manually.
#
# Run detached:  tmux new-session -d -s pd-watchdog 'bash pd_watchdog.sh'
# Stop:          tmux kill-session -t pd-watchdog   (or kill the loop PID)
#
# Env overrides: MOL_PROD_HOST (def mol-deploy-0), MOL_GATEWAY_PORT (30001),
#                 MOL_PD_API_KEY (sk-glm52-pd), PD_WATCH_INTERVAL (60).
set -uo pipefail

HOST="${MOL_PROD_HOST:-mol-deploy-0}"
GW_PORT="${MOL_GATEWAY_PORT:-30001}"
API_KEY="${MOL_PD_API_KEY:-sk-glm52-pd}"
INTERVAL="${PD_WATCH_INTERVAL:-60}"
PROXY_PORT="${MOL_PROXY_PORT:-30000}"

# pd-router name (gateway label) -> pd_restart.sh arg
declare -A PD_NAME_TO_ARG=(
  [mol-deploy-pd-router]=pd-router-1
  [mol-deploy-pd-router-1]=pd-router-1
  [mol-deploy-pd-router-2]=pd-router-2
)

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESTART="$SKILL_DIR/scripts/pd_restart.sh"
DRAIN="$SKILL_DIR/scripts/drain_pd_worker.sh"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# list online sglang workers: "name url" per line (only those registered now)
online_pd_workers() {
  ssh "$HOST" "curl -s -m 10 http://127.0.0.1:${GW_PORT}/workers" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for w in d.get('workers', []):
    if w.get('runtime_type') == 'sglang':
        name = (w.get('metadata') or {}).get('name', '')
        print(name, w.get('url', ''))
" 2>/dev/null
}

# probe one PD: send a real request through its router. 0=healthy, 1=dead.
probe_pd() {  # <name> <url>
  local name="$1" url="$2"
  local body='{"model":"glm52-fp8-official","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
  local resp
  resp=$(ssh "$HOST" "curl -s -m 60 -o - -w '\n__HTTP_%{http_code}' -X POST ${url}/v1/chat/completions \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer ${API_KEY}' -d '${body}'" 2>/dev/null)
  # a real completion has "choices" → healthy; anything else (503/timeout/parse) → dead
  if printf '%s' "$resp" | grep -q '"choices"'; then
    return 0
  else
    return 1
  fi
}

# register a PD back to the gateway by its short name (pd-router-1/2)
register_pd() {  # <pd_restart_arg> <url>
  local arg="$1" url="$2"
  ssh "$HOST" "curl -s -m 10 -X POST http://127.0.0.1:${GW_PORT}/workers -H 'Content-Type: application/json' \
    -d '{\"url\":\"${url}\",\"model_id\":\"glm52-fp8-official\",\"worker_type\":\"regular\",\"runtime\":\"sglang\",\"priority\":50,\"cost\":1.0,\"api_key\":\"${API_KEY}\",\"labels\":{\"deployment\":\"macaron-v1-venti\",\"name\":\"mol-deploy-${arg}\"}}'" 2>/dev/null
}

# per-PD recovery state: name -> consecutive failures (for backoff)
declare -A FAILS

recover_pd() {  # <name> <url>
  local name="$1" url="$2" arg
  arg="${PD_NAME_TO_ARG[$name]:-}"
  if [ -z "$arg" ]; then
    log "  $name: no pd_restart arg mapping, cannot auto-restart; skipping"
    return 1
  fi
  log "  $name ($arg): draining from gateway"
  bash "$DRAIN" "$name" >/dev/null 2>&1
  sleep 5
  log "  $name ($arg): restarting PD pair (prefill+decode+router)"
  if bash "$RESTART" "$arg" >/root/tmp/pd_watchdog_restart_${arg}.log 2>&1; then
    log "  $name ($arg): restart OK, verifying chain"
  else
    log "  $name ($arg): restart script reported failure (see /root/tmp/pd_watchdog_restart_${arg}.log)"
  fi
  # verify chain directly (router may say 200 while prefill still warming/failed)
  if probe_pd "$name" "$url"; then
    log "  $name ($arg): chain healthy, re-registering to gateway"
    register_pd "$arg" "$url" >/dev/null 2>&1
    FAILS[$name]=0
    log "  $name ($arg): RECOVERED ✓"
    return 0
  else
    FAILS[$name]=$(( ${FAILS[$name]:-0} + 1 ))
    log "  $name ($arg): chain still dead after restart (fail #${FAILS[$name]}), will retry next cycle"
    return 1
  fi
}

log "pd_watchdog started: interval=${INTERVAL}s host=$HOST gw=$GW_PORT"
log "watching sglang PD workers; vllm left to gateway health check"

while true; do
  # snapshot online PDs each cycle (a drained one won't appear until re-registered)
  workers=$(online_pd_workers)
  if [ -z "$workers" ]; then
    : # no sglang workers online right now (all drained) — wait
  else
    while read -r name url; do
      [ -z "$name" ] && continue
      if probe_pd "$name" "$url"; then
        if [ "${FAILS[$name]:-0}" -ne 0 ]; then
          log "$name: healthy again (cleared ${FAILS[$name]} prior fail)"
        fi
        FAILS[$name]=0
      else
        log "$name: DEAD (router up but chain 503/timeout) — recovering"
        recover_pd "$name" "$url" || true
      fi
    done <<< "$workers"
  fi
  sleep "$INTERVAL"
done
