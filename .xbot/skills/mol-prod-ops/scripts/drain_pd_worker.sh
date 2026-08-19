#!/usr/bin/env bash
# drain_pd_worker.sh — remove a sglang PD worker from the gateway when it is
# half-dead (router up but prefill/decode crashed → 503).
#
# The gateway keeps a sglang worker "healthy" while its prefill is dead, because
# the router's /health stays 200. Traffic routed to such a worker 503s. This
# script drains it: DELETE /workers/{id} so the gateway stops routing to it.
#
# It does NOT touch the prefill/decode/router processes — use pd_restart.sh for
# that. Drain + restart is the safe order (drain first, restart at leisure).
#
# USAGE:
#   bash drain_pd_worker.sh pd-router-1     # by name
#   bash drain_pd_worker.sh pd-router-2
#   bash drain_pd_worker.sh --all           # drain every sglang PD worker
#   bash drain_pd_worker.sh <worker_id>     # by UUID
#
# Verify first with verify_3hop.sh — it reports chain=DEAD on half-dead PDs.
set -uo pipefail

HOST="${MOL_PROD_HOST:-mol-deploy-0}"
GW_PORT="${MOL_GATEWAY_PORT:-30001}"

say() { printf '%s\n' "$*"; }

# fetch worker list as: "id name runtime url" per line
list_workers() {
  ssh "$HOST" "curl -s -m 10 http://127.0.0.1:${GW_PORT}/workers" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for w in d.get('workers', []):
    name = (w.get('metadata') or {}).get('name') or ''
    print(w.get('id',''), name, w.get('runtime_type',''), w.get('url',''))
"
}

drain_one() {  # <id> <name>
  local id="$1" name="$2"
  say "  draining $name ($id) ..."
  ssh "$HOST" "curl -s -m 10 -X DELETE http://127.0.0.1:${GW_PORT}/workers/$id -w '\nHTTP %{http_code}\n'"
}

target="${1:-}"
if [ -z "$target" ]; then
  say "Usage: $0 <pd-router-1|pd-router-2|--all|worker_id>"
  say "Registered workers:"
  list_workers | awk '{printf "  %s  %s  %s\n", $1, $2, $4}'
  exit 2
fi

# resolve target -> list of "id name" lines
if [ "$target" = "--all" ]; then
  targets=$(list_workers | awk '$3=="sglang"{print $1" "$2}')
else
  # match by name (contains target) or exact id
  targets=$(list_workers | awk -v t="$target" '$2==t || $2~t || $1==t {print $1" "$2}')
fi

if [ -z "$targets" ]; then
  say "no worker matched '$target'"
  say "registered:"
  list_workers | awk '{printf "  %s  %s  %s\n", $1, $2, $4}'
  exit 1
fi

echo "$targets" | while read id name; do
  [ -z "$id" ] && continue
  drain_one "$id" "$name"
done

say ""
say "=== gateway /workers after drain ==="
ssh "$HOST" "curl -s -m 10 http://127.0.0.1:${GW_PORT}/workers" | python3 -c "
import sys, json
ws = json.load(sys.stdin).get('workers', [])
print('worker count:', len(ws))
for w in ws:
    print('  %s healthy=%s url=%s' % ((w.get('metadata') or {}).get('name','?'), w.get('is_healthy'), w.get('url')))
"
