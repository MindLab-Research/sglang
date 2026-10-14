#!/usr/bin/env bash
# decode 步 kernel 级 timeline 抓取（DRY-RUN 默认；--execute 才动 1101）
# 目标：稳态 conc=64 下的 15~20s decode 窗口，含 NVTX 步边界，供 nsys_extract.py 分桶
set -euo pipefail
TAG=${1:-rnd}; EXEC=${2:-}
ROOT=${ROOT:-/root}; OUTDIR=${OUTDIR:-$ROOT/prof}; DECODE_PID_FILE=${DECODE_PID_FILE:-$ROOT/d_decode.pid}
mkdir -p "$OUTDIR"
CMDS=$(cat <<'CMDS'
# 0) 开启 NVTX（sglang 内建 span 发射器，见 python/sglang/srt/utils/nvtx_utils.py）
export SGLANG_ENABLE_NVTX_SCHEDULER=1
export SGLANG_ENABLE_NVTX_OPERATIONS=1
# 1) 找到 decode 侧 pid（PD 分离：18100-ish 那个 sglang::scheduler / TP worker）
PID=$(pgrep -f "sglang.*decode|sglang::scheduler" | head -1); echo "decode pid=$PID"
# 2) 挂 nsys（attach，不重启进程）：cuda + nvtx + osrt，采样窗口 20s
nsys profile --attach-pid "$PID" -t cuda,nvtx,osrt -s none \
  --force-overwrite true -o "$OUTDIR/kern_$TAG" --duration 20 --sample cpu --cpuctxsw none
# 3) 同时刻并发打 bench（保证窗口内有稳态 conc=64 的 decode）
python3 tools/mol-v41-prep/bench_c64.py --endpoint http://127.0.0.1:31000/v1/chat/completions \
  --tag "prof_$TAG" --n 256 --conc 64 --rounds 1 --out "$OUTDIR/bench_prof_$TAG.json"
# 4) 导出 sqlite + CSV（供 nsys_extract.py）
nsys export --type sqlite --force-overwrite true -o "$OUTDIR/kern_$TAG.sqlite" "$OUTDIR/kern_$TAG.nsys-rep"
nsys stats --report cuda_gpu_trace --format csv   "$OUTDIR/kern_$TAG.nsys-rep" > "$OUTDIR/gpu_trace_$TAG.csv"
nsys stats --report nvtx_pushpop_trace --format csv "$OUTDIR/kern_$TAG.nsys-rep" > "$OUTDIR/nvtx_$TAG.csv"
CMDS
)
echo "=== 计划命令 (tag=$TAG) ==="; echo "$CMDS"
if [[ "$EXEC" == "--execute" ]]; then echo "== 执行 =="; bash -c "$CMDS"; else echo "[DRY-RUN] 1101 空闲后再加 --execute"; fi
# 备选（无 nsys 权限时）：torch profiler 路径 —— 服务端 SGLANG_TORCH_PROFILER_DIR=/root/prof_torch 启动，
# 然后 POST /start_profile {"activities":["CUDA"],"num_steps":200} ... POST /stop_profile，得 chrome trace。
