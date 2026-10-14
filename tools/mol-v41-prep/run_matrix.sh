#!/usr/bin/env bash
# MoL-V41 归因矩阵运行器（默认 DRY-RUN，绝不碰 GPU）
# 用法:
#   bash run_matrix.sh plan                 # 只打印矩阵与环境变量（默认）
#   bash run_matrix.sh cell C2 --execute    # 真的要打：重启服务 + 跑 bench
# 依赖（1101 盒子上已有）: /root/exp_l20.sh(含 --lora-target-layers 20-39) /root/wait_ready.sh
set -euo pipefail
ROOT=${ROOT:-/root}
TREE=${TREE:-$ROOT/dsv41_upstream/sglang}
CASES=${CASES:-/home/smith/src/mint-bench/bench/case50/cases_50.json}
SMG=${SMG:-http://127.0.0.1:31000}
OUTDIR=${OUTDIR:-$ROOT/mol_matrix}
# 注意：仅真实执行时才建目录，保证本地 dry-run 零副作用（/root 不可写）

# --- 矩阵定义: cell -> 额外环境变量 / 适配器 / 假设 ---
declare -A ENV_C=(
  [C0p_stockbase]=""
  [C0_base]=""
  [C1_rnd]=""
  [C3p_fuseddelta]="SGLANG_LORA_FUSED_DELTA=1"
  [C2_rnd_skipdelta]="SGLANG_MOE_LORA_SKIP_DELTA=1"
  [C5p_rnd_skipdelta_pdl]="SGLANG_MOE_LORA_SKIP_DELTA=1 SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS=8192"
  [C4_rnd_fusedgemm]="SGLANG_OPT_MOL_FUSED_GEMM=1"
  [C5_rnd_fusedgemm_pdl]="SGLANG_OPT_MOL_FUSED_GEMM=1 SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS=8192"
  [C6_layer_slope]=""
)
declare -A ADAPTER_C=(
  [C0p_stockbase]="__none__"
  [C0_base]="__none__" [C1_rnd]=dsv41_moe_rnd
  [C3p_fuseddelta]=dsv41_moe_rnd [C2_rnd_skipdelta]=dsv41_moe_rnd
  [C5p_rnd_skipdelta_pdl]=dsv41_moe_rnd [C4_rnd_fusedgemm]=dsv41_moe_rnd
  [C5_rnd_fusedgemm_pdl]=dsv41_moe_rnd
  [C6_layer_slope]=dsv41_moe_rnd
)
declare -A HYP_C=(
  [C0p_stockbase]="无 --enable-lora 的真 stock 分母（另起一次启动）"
  [C0_base]="同栈基线(注意: 挂 LoRA 的栈上 base 仍含 piecewise +18% 污染)"
  [C1_rnd]="复现 7.281（若 >7.5 或 <7.0 先停：口径漂移，别做归因）"
  [C3p_fuseddelta]="融合 delta 路径（environ.py:1129 默认 False；消费点 kernels/ops/moe/virtual_experts.py:876 读 :1811-1815）——前置 G1 位级闸门，未过闸不读数"
  [C2_rnd_skipdelta]="只摘 MoE VE delta；预测 6.9~7.2 ⇒ 若≈base 则结构与 GEMM 罚都在 delta 分支内"
  [C5p_rnd_skipdelta_pdl]="PDL 覆盖缺口的边际收益；预测 -0.05~-0.25ms"
  [C4_rnd_fusedgemm]="M3 融合核；预测 6.3~7.0（首版常无收益，看 launch 数与 cubin 家族）"
  [C5_rnd_fusedgemm_pdl]="融核+PDL 叠加；预测 6.2~6.9"
  [C6_layer_slope]="0-39 层 vs 20-39 层；Δ/20=per-layer 摊销，区分固定开销 vs 每层开销"
)

plan() {
  printf '%-26s %-8s %-52s %s\n' CELL ADAPTER EXTRA_ENV HYPOTHESIS
  for c in C0p_stockbase C0_base C1_rnd C3p_fuseddelta C2_rnd_skipdelta C5p_rnd_skipdelta_pdl C4_rnd_fusedgemm C5_rnd_fusedgemm_pdl C6_layer_slope; do
    printf '%-26s %-8s %-52s %s\n' "$c" "${ADAPTER_C[$c]:-（未定义）}" "${ENV_C[$c]:-（无）}" "${HYP_C[$c]:-（未定义）}"
  done
  cat <<'TXT'

执行协议（每个 cell）:
  1) 轮换 smg API key（老 key 追加 /root/smg_api_keys_history.txt）
  2) 停栈 → 全新启动:  ENV... bash /root/exp_l20.sh ; bash /root/wait_ready.sh
  3) 就绪判据: curl -s -o /dev/null -m 5 -w '%{http_code}' $SMG/health  == 200
  4) 确认 P(30100)/D(30200) 两侧 adapter 均已加载，无外部流量(nvidia-smi 利用率 + smg 日志)
  5) bench: python3 bench_c64.py --endpoint $SMG/v1/chat/completions --adapter <A> --tag <cell> --rounds 2
  6) 交叉 A/B 顺序: C0→C1→C0→C1（每轮 ≥2 次），取 client_agg p50
TXT
}

# 短名别名 C0..C5 / C0p
alias_of() {
  case "$1" in
    C0p|stock_base) echo C0p_stockbase ;;
    C0|base)        echo C0_base ;;
    C1|rnd)         echo C1_rnd ;;
    C3p|fuseddelta) echo C3p_fuseddelta ;;
    C2|skip)        echo C2_rnd_skipdelta ;;
    C5p|C3|skip_pdl) echo C5p_rnd_skipdelta_pdl ;;
    C4|fused)       echo C4_rnd_fusedgemm ;;
    C5|fused_pdl)   echo C5_rnd_fusedgemm_pdl ;;
    C6|slope)       echo C6_layer_slope ;;
    *)              echo "$1" ;;
  esac
}

cell() {
  local C; C=$(alias_of "${1:-}"); shift || true
  if [[ -z "${ENV_C[$C]+x}" ]]; then echo "unknown cell: ${1:-}（可用: plan / C0p C0 C1 C3p C2 C5p C4 C5 C6）" >&2; return 2; fi
  if [[ "${1:-}" != "--execute" ]]; then
    echo "[DRY-RUN] 不执行、不建目录、不碰 GPU。cell=$C"
    echo "  ADAPTER : ${ADAPTER_C[$C]:-（未定义）}"
    echo "  EXTRA_ENV: ${ENV_C[$C]:-（无）}"
    echo "  HYPOTHESIS: ${HYP_C[$C]:-（未定义）}"
    echo "  真实执行: 确认 1101 空闲后 bash $0 cell $C --execute"
    return 0
  fi
  GATE_DIR=${GATE_DIR:-$ROOT/mol_gate}
  if [[ "$C" == "C3p_fuseddelta" && ! -f "$GATE_DIR/G1.pass" ]]; then
    echo "REFUSED: $C 的前置 G1 位级闸门未过（缺 $GATE_DIR/G1.pass）。先跑 gate_zero_bitexact.sh dump/compare，过了再读数。" >&2
    return 3
  fi
  mkdir -p "$OUTDIR"
  echo "== $C =="
  # G0 环境核对清单：三项并列导出 + 落盘，供事后核对
  { echo "cell=$C  ts=$(date -Is)"; for v in SGLANG_MOE_LORA_SKIP_DELTA SGLANG_LORA_FUSED_DELTA SGLANG_OPT_MOL_FUSED_GEMM; do
      echo "  $v=${!v:-<unset>}"; done; } | tee "$OUTDIR/$C.env.txt"
  # shellcheck disable=SC2086
  env ${ENV_C[$C]:-} SGLANG_MOL_MATRIX_CELL="$C" bash "$ROOT/exp_l20.sh"
  bash "$ROOT/wait_ready.sh"
  local bench_args=(--endpoint "$SMG/v1/chat/completions" --cases "$CASES" --tag "$C" --rounds 2 --out "$OUTDIR/$C.json")
  [[ "${ADAPTER_C[$C]:-__none__}" != "__none__" ]] && bench_args+=(--adapter "${ADAPTER_C[$C]}")
  python3 "$(dirname "$0")/bench_c64.py" "${bench_args[@]}"
}

case "${1:-plan}" in
  plan) plan ;;
  cell) shift; cell "$@" ;;
  *) plan ;;
esac
