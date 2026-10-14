#!/usr/bin/env bash
# perf 声明前的冒烟清单（只读 curl / 只读查询；无 GPU 操作）
# 用法: bash smoke_checklist.sh [PUBLIC_HOST] [LAN_HOST]
set -uo pipefail
PUB=${1:-8.222.11.182}; LAN=${2:-10.0.58.56}
KEY=${SMG_API_KEY:-}
P_PORT=${P_PORT:-30100}; D_PORT=${D_PORT:-30200}; SMG_PORT=${SMG_PORT:-31000}
pass=0; fail=0
chk(){ local name=$1; shift; if "$@" >/dev/null 2>&1; then echo "  [PASS] $name"; pass=$((pass+1)); else echo "  [FAIL] $name"; fail=$((fail+1)); fi; }
code(){ curl -s -o /dev/null -m 5 -w '%{http_code}' "$1"; }
is200(){ [[ "$(code "$1")" == "200" ]]; }

echo "== 1. 就绪性（判据 = HTTP 200，body 非空判据已证伪）=="
chk "P  :$P_PORT/health    (LAN)"  is200 "http://$LAN:$P_PORT/health"
chk "D  :$D_PORT/health    (LAN)"  is200 "http://$LAN:$D_PORT/health"
chk "smg:$SMG_PORT/health   (公网)" is200 "http://$PUB:$SMG_PORT/health"

echo "== 2. 模型与 adapter 可见性 =="
AUTH=(); [[ -n "$KEY" ]] && AUTH=(-H "Authorization: Bearer $KEY")
chk "GET /v1/models (含 base + dsv41_moe_{rnd,zero,tinker})" \
  bash -c "curl -s -m 10 ${AUTH[*]:-} http://$PUB:$SMG_PORT/v1/models | grep -q dsv41_moe_rnd"
chk "adapter 列表含 dsv41_moe_zero" bash -c "curl -s -m 10 ${AUTH[*]:-} http://$PUB:$SMG_PORT/v1/models | grep -q dsv41_moe_zero"

echo "== 3. 双入口功能冒烟（base / zero / rnd 各一发短请求）=="
for a in none dsv41_moe_zero dsv41_moe_rnd; do
  model=$([[ $a == none ]] && echo default || echo $a)
  body=$(curl -s -m 60 ${AUTH[*]:-} -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话说明1+1=2\"}],\"max_tokens\":32,\"temperature\":0}" \
    "http://$PUB:$SMG_PORT/v1/chat/completions")
  if echo "$body" | grep -q '"content"'; then
    n=$(echo "$body" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["choices"][0]["message"]["content"]))' 2>/dev/null || echo 0)
    echo "  [PASS] $a 有输出 (len=$n)"; pass=$((pass+1))
  else echo "  [FAIL] $a 无输出: ${body:0:200}"; fail=$((fail+1)); fi
done

echo "== 4. 适配器「真的生效」抽查（rank 选路）=="
echo "  手工项：rnd 与 zero 对同一 prompt 的输出必须不同；zero 与 base(无 adapter) 的 token 流必须全等"
echo "          命令：python3 bench_c64.py --adapter dsv41_moe_zero --n 8 --conc 1 --dump-tokens zero.jsonl"
echo "                python3 bench_c64.py --adapter none --n 8 --conc 1 --dump-tokens base.jsonl"
echo "                bash gate_zero_bitexact.sh compare zero.jsonl base.jsonl"

echo "== 5. 污染检查（bench 前必查）=="
echo "  手工项：nvidia-smi 利用率应只来自本 bench；smg 日志中不得有本 key 之外的请求（新 key 已轮换）"

echo; echo "SUMMARY: pass=$pass fail=$fail"; [[ $fail -eq 0 ]] && echo "VERDICT: PASS" || echo "VERDICT: FAIL"
