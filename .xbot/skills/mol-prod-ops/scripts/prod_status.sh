#!/usr/bin/env bash
# 生产环境状态汇报:Gateway / Proxy / Workers
# 用法: ./scripts/prod_status.sh
# 依赖: 能 ssh mol-deploy-0(生产 gateway/proxy 所在机)
set -euo pipefail

HOST="${MOL_PROD_HOST:-mol-deploy-0}"
GW_PORT="${MOL_GATEWAY_PORT:-30001}"
PX_PORT="${MOL_PROXY_PORT:-30000}"

# 远端状态检查脚本(写到远端 /tmp 执行,避免 ssh 单引号转义地狱)
REMOTE_SCRIPT='
import json, urllib.request, subprocess, re, sys

def get(url, t=5):
    try:
        return urllib.request.urlopen(url, timeout=t).read().decode()
    except Exception as e:
        return "ERR: " + str(e)

def pidof(pat):
    r = subprocess.run(["pgrep","-f",pat],capture_output=True,text=True)
    pids = r.stdout.strip().split()
    return pids[0] if pids else "?"

print("===== Gateway =====")
gwpid = pidof("smg launch")
print("PID=" + gwpid)
print("health: " + get("http://127.0.0.1:'"$GW_PORT"'/health"))
try:
    cmd = open("/proc/" + gwpid + "/cmdline").read().replace("\0"," ")
    for k in ["request-timeout-secs","max-idle-secs","prometheus-port","policy","assignment-mode"]:
        m = re.search(k + r" (\S+)", cmd)
        if m: print("  " + k + "=" + m.group(1))
except Exception as e: print("  cmd err:", e)

print("")
print("===== Proxy =====")
pxpid = pidof("mol_harness.proxy")
print("PID=" + pxpid)
print("health: " + get("http://127.0.0.1:'"$PX_PORT"'/health"))
try:
    env = open("/proc/" + pxpid + "/environ").read().split("\0")
    for e in env:
        if e.startswith(("PYTHONPATH=","MOL_HOP_TIMEOUT=","PROXY_PORT=","LIBRARY_DIR=")):
            print("  " + e)
except Exception as e: print("  env err:", e)

print("")
print("===== Workers =====")
w = get("http://127.0.0.1:'"$GW_PORT"'/workers")
try:
    d = json.loads(w)
    ws = d.get("workers",[])
    healthy = sum(1 for x in ws if x.get("is_healthy"))
    print("worker count: " + str(len(ws)) + " (healthy=" + str(healthy) + ")")
    for x in ws:
        name = (x.get("metadata") or {}).get("name") or x.get("url","")
        print("  " + str(name) + " url=" + str(x.get("url")) +
              " runtime=" + str(x.get("runtime_type")) +
              " healthy=" + str(x.get("is_healthy")) +
              " load=" + str(x.get("load")) +
              " model=" + str(x.get("model_id")))
except Exception as e:
    print("parse fail:", e, w[:200])
'

ssh "$HOST" "cat > /root/tmp/_mol_prod_status.py && python3 /root/tmp/_mol_prod_status.py" <<< "$REMOTE_SCRIPT"
