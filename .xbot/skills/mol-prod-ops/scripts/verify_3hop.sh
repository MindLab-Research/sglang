#!/usr/bin/env bash
# verify_3hop.sh — end-to-end gate after any Proxy/worker/gateway change.
# Checks: (1) identity request through Proxy → Macaron-V1-Venti 748B,
#         (2) stream request returns a usage chunk,
#         (3) per-worker readiness — router /v1/models AND (for sglang PD workers)
#             a real inference request through the router. router /v1/models==200
#             alone is NOT enough: a sglang router stays up while its prefill is
#             dead, so the gateway sees healthy but requests 503.
# If check 3 reports a sglang worker as chain=DEAD, that PD is half-dead — drain
# it with drain_pd_worker.sh (this script is read-only, it does NOT remove workers).
# Exits non-zero if any check fails.
#
# Usage: bash verify_3hop.sh
set -uo pipefail

HOST="${MOL_PROD_HOST:-mol-deploy-0}"
PROXY_PORT="${MOL_PROXY_PORT:-30000}"
GW_PORT="${MOL_GATEWAY_PORT:-30001}"
API_KEY="${MOL_PD_API_KEY:-sk-glm52-pd}"

fail=0
say() { printf '%s\n' "$*"; }

# ---- check 1: identity through Proxy (3-hop route→answer→summary) ----
say "=== [1/3] identity request through Proxy ==="
resp=$(ssh "$HOST" "curl -s -m 90 -X POST http://127.0.0.1:${PROXY_PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"Macaron-V1-Venti\",\"messages\":[{\"role\":\"user\",\"content\":\"你是什么模型?\"}],\"max_tokens\":400}'")
content=$(printf '%s' "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']
    txt = (c.get('content') or '') + (c.get('reasoning_content') or '')
    print(txt)
except Exception as e:
    print('__PARSE_FAIL__' + str(e))
")
if printf '%s' "$content" | grep -qi "Macaron-V1-Venti"; then
    say "  OK: identity answered Macaron-V1-Venti"
else
    say "  FAIL: identity answer did not contain Macaron-V1-Venti"
    say "  raw (head): $(printf '%s' "$resp" | head -c 300)"
    fail=1
fi

# ---- check 2: stream usage chunk ----
say "=== [2/3] stream usage chunk ==="
stream=$(ssh "$HOST" "curl -s -m 90 -N -X POST http://127.0.0.1:${PROXY_PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"Macaron-V1-Venti\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20,\"stream\":true}'")
if printf '%s' "$stream" | grep -qi '"usage"'; then
    say "  OK: stream returned a usage chunk"
else
    say "  FAIL: no usage chunk in stream response"
    say "  raw (head): $(printf '%s' "$stream" | head -c 300)"
    fail=1
fi

# ---- check 3: per-worker readiness — router /v1/models + PD chain probe ----
# For sglang PD workers, router /v1/models==200 does NOT mean prefill/decode are alive
# (router stays up while prefill is dead → gateway thinks healthy but requests 503).
# So for sglang workers we ALSO send a real inference request through the router; a
# 503 "No available prefill/decode workers" means the PD chain is half-dead.
say "=== [3/3] per-worker readiness (router /v1/models + PD chain probe) ==="
if ssh "$HOST" "GW_PORT=$GW_PORT API_KEY=$API_KEY python3 -" <<'PYEOF'
import json, os, urllib.request
workers = json.load(urllib.request.urlopen("http://127.0.0.1:%s/workers" % os.environ["GW_PORT"], timeout=10))["workers"]
if not workers:
    print("  FAIL: no workers registered"); raise SystemExit(2)
bad = 0
hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_KEY"]}
body = b'{"model":"glm52-fp8-official","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
for w in workers:
    url = w.get("url", "").rstrip("/")
    name = (w.get("metadata") or {}).get("name") or url
    rt = w.get("runtime_type") or ""
    # 3a: router /v1/models
    try:
        code = urllib.request.urlopen(url + "/v1/models", timeout=8).getcode()
        models_ok = (code == 200)
    except Exception as e:
        print("  %s -> /v1/models ERR %s" % (name, e)); bad += 1; continue
    # 3b: for sglang PD workers, send a real request through the router
    chain_ok = True; detail = ""
    if "sglang" in rt:
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers=hdr, method="POST")
            r = urllib.request.urlopen(req, timeout=60)
            d = json.load(r)
            # a real completion (has choices) = chain works
            chain_ok = bool(d.get("choices"))
            if not chain_ok: detail = " (no choices in response)"
        except urllib.error.HTTPError as e:
            chain_ok = False; detail = " (chain 503: prefill/decode down)" if e.code == 503 else " (HTTP %d)" % e.code
        except Exception as e:
            chain_ok = False; detail = " (chain ERR %s)" % str(e)[:60]
    status = "OK" if (models_ok and chain_ok) else "FAIL"
    extra = "" if ("sglang" not in rt) else (" chain=OK" if chain_ok else " chain=DEAD" + detail)
    print("  %s -> /v1_models=%s%s  [%s]" % (name, code, extra, status))
    if not (models_ok and chain_ok): bad += 1
raise SystemExit(1 if bad else 0)
PYEOF
then
    :
else
    fail=1
fi

say ""
if [ $fail -eq 0 ]; then
    say "RESULT: ALL CHECKS PASSED ✅"
    exit 0
else
    say "RESULT: SOME CHECKS FAILED ❌"
    exit 1
fi
