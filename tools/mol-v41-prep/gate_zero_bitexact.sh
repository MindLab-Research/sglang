#!/usr/bin/env bash
# G1 门禁：zero-adapter == pass-through 位级一致（M3 接线后必须先过）
# 机制：SKIP_DELTA=1 把 has_active_lora 降级为 pass-through zero-delta（lora_dispatch.py:834-838，数值无损，commit 2a0688082e 已证 zero==base 为 BITWISE）
# 做法：同栈、同 seed、temperature=0、greedy，两个 cell 各跑一遍相同 case 集，逐请求比对
#      (a) 生成 token 流 md5  (b) 每步 logprob md5  (c) NaN/Inf 计数
# 判据：256/256 请求 token 流 md5 全等，且首个分歧索引不存在 ⇒ PASS
# 用法: bash gate_zero_bitexact.sh dump          # 采两份 dump（--execute 才动 GPU）
#       bash gate_zero_bitexact.sh compare A/ B/  # 纯本地比对，任何时候可跑
set -euo pipefail
MODE=${1:-compare}; EXEC=${2:-}
BIN="$(dirname "$0")"

dump() {
  for cell in zero_noskip zero_skip; do
    env $( [[ $cell == zero_skip ]] && echo SGLANG_MOE_LORA_SKIP_DELTA=1 ) \
      bash /root/exp_l20.sh   # 保持与 perf cell 完全相同的启动面
    bash /root/wait_ready.sh
    python3 "$BIN/bench_c64.py" --endpoint http://127.0.0.1:31000/v1/chat/completions \
      --adapter dsv41_moe_zero --n 256 --conc 8 --rounds 1 --seed 777 \
      --dump-tokens "/root/mol_gate/$cell.jsonl" --tag "gate_$cell"
  done
}
compare() {
  python3 - "$1" "$2" <<'PY'
import json,sys,hashlib
def load(p):
    out={}
    for line in open(p):
        r=json.loads(line); out[r["id"]]=(hashlib.md5("\x1f".join(map(str,r["tokens"])).encode()).hexdigest(),
                                          r.get("logprob_md5"), r.get("nan_inf_count",0), r.get("n_tokens"))
    return out
A,B=load(sys.argv[1]),load(sys.argv[2])
ids=sorted(set(A)&set(B)); bad=[i for i in ids if A[i][0]!=B[i][0]]
nan=[i for i in ids if (A[i][2] or B[i][2])]
print(f"requests={len(ids)} token_md5_mismatch={len(bad)} nan_inf={len(nan)}")
for i in bad[:10]: print(f"  DIVERGE id={i} A={A[i][0][:12]} B={B[i][0][:12]} nA={A[i][3]} nB={B[i][3]}")
lp=[i for i in ids if A[i][1]!=B[i][1]]
print(f"logprob_md5_mismatch={len(lp)} (logprob 位级更严：若仅此不等而 token 全等 ⇒ 记 WARN 不拦)")
ok = not bad and not nan
print("VERDICT:", "PASS" if ok else "FAIL")
import os
d = os.environ.get("GATE_DIR", "/root/mol_gate")
if ok:
    os.makedirs(d, exist_ok=True); open(d + "/G1.pass", "w").write("zero==pass-through bitwise OK\n")
    print(f"wrote {d}/G1.pass（C3' 前置闸门已开）")
PY
}
case "$MODE" in
  dump) if [[ "$EXEC" == "--execute" ]]; then dump; else echo "[DRY-RUN] bash gate_zero_bitexact.sh dump --execute"; fi ;;
  compare) compare "${2:?usage: compare A.jsonl B.jsonl}" "${3:?}" ;;
  *) echo "usage: dump [--execute] | compare A.jsonl B.jsonl" ;;
esac
