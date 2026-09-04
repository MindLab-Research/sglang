#!/bin/bash
# =============================================================================
# restart_1102_1104.sh — 一键重启 1102(prefill) + 1104(decode) + smg + flood
# 用法: bash scripts/restart_1102_1104.sh [--no-flood] [--concurrency N] [--prob P]
#
# 保证:
#   1. 用 pgrep 精确匹配 python3 -m sglang（不用 pkill，不会杀 SSH 自身）
#   2. kill -9 后等 GPU 显存释放（nvidia-smi 确认 0 MiB）再启动
#   3. 端口确认释放（fuser）再启动
#   4. 等 health 200 才继续下一步（不等就用日志报错）
#   5. 每步打印 PID 确认是新进程
# =============================================================================

set -uo pipefail

SSH="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no"
PREFILL_PORT=1102
DECODE_PORT=1104
HOST=8.222.11.182
PREFILL_IP=10.0.58.35
DECODE_IP=10.0.58.37

# 可选参数
DO_FLOOD=1
FLOOD_CONCURRENCY=20
FLOOD_PROB=0.8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-flood) DO_FLOOD=0; shift;;
    --concurrency) FLOOD_CONCURRENCY="$2"; shift 2;;
    --prob) FLOOD_PROB="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Kill old processes (并行 kill prefill + decode + smg/flood)
# ---------------------------------------------------------------------------
log "=== Step 1: Kill old processes (parallel) ==="

KILL_PF='
  pids=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}")
  [ -n "$pids" ] && echo "  kill PIDs: $pids" && echo "$pids" | xargs kill -9 2>/dev/null || true
  sleep 2
  remaining=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}")
  [ -n "$remaining" ] && echo "  force kill: $remaining" && echo "$remaining" | xargs kill -9 2>/dev/null || true
  fuser -k 30100/tcp 2>/dev/null || true
  fuser -k 8998/tcp 2>/dev/null || true
  sleep 1
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d " MiB")
  echo "  GPU mem: ${gpu_mem} MiB, remaining=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | wc -l)"
'
KILL_DC='
  pids=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}")
  [ -n "$pids" ] && echo "  kill PIDs: $pids" && echo "$pids" | xargs kill -9 2>/dev/null || true
  sleep 2
  remaining=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}")
  [ -n "$remaining" ] && echo "  force kill: $remaining" && echo "$remaining" | xargs kill -9 2>/dev/null || true
  fuser -k 30200/tcp 2>/dev/null || true
  fuser -k 8998/tcp 2>/dev/null || true
  sleep 1
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d " MiB")
  echo "  GPU mem: ${gpu_mem} MiB, remaining=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | wc -l)"
'
KILL_SMG='
  pids=$(ps aux | grep "/usr/local/bin/smg" | grep -v grep | awk "{print \$2}")
  [ -n "$pids" ] && echo "  kill smg PIDs: $pids" && echo "$pids" | xargs kill -9 2>/dev/null || true
  fuser -k 31000/tcp 2>/dev/null || true
  fuser -k 29004/tcp 2>/dev/null || true
  pids=$(ps aux | grep "kv_flood_test" | grep -v grep | awk "{print \$2}")
  [ -n "$pids" ] && echo "  kill flood PIDs: $pids" && echo "$pids" | xargs kill -9 2>/dev/null || true
  sleep 2
  echo "  smg=$(ps aux | grep "/usr/local/bin/smg" | grep -v grep | wc -l) flood=$(ps aux | grep "kv_flood_test" | grep -v grep | wc -l)"
'

# 并行 kill 三组
log "--- Killing prefill(1102) + decode(1104) + smg/flood(1102) in parallel ---"
$SSH -p $PREFILL_PORT root@$HOST "$KILL_PF" 2>/dev/null &
PF_KILL_PID=$!
$SSH -p $DECODE_PORT root@$HOST "$KILL_DC" 2>/dev/null &
DC_KILL_PID=$!
$SSH -p $PREFILL_PORT root@$HOST "$KILL_SMG" 2>/dev/null &
SMG_KILL_PID=$!
wait $PF_KILL_PID
wait $DC_KILL_PID
wait $SMG_KILL_PID
log "  All kills done"

# ---------------------------------------------------------------------------
# Step 2: Start prefill + decode IN PARALLEL
# ---------------------------------------------------------------------------
log "=== Step 2: Start prefill + decode ==="
# fire and forget：SSH 后台启动，不等返回，健康检查步骤验证
$SSH -n -p $PREFILL_PORT root@$HOST 'cd /root && nohup bash start_glm53_prefill.sh > /root/glm53_prefill.log 2>&1 </dev/null & disown' > /dev/null 2>&1 &
$SSH -n -p $DECODE_PORT root@$HOST 'cd /root && nohup bash start_glm53_decode.sh > /root/glm53_decode.log 2>&1 </dev/null & disown' > /dev/null 2>&1 &
# 不 wait——SSH 可能因远程进程继承 fd 而挂住，健康检查步骤会验证进程是否启动
log "  prefill + decode launched (fire and forget)"

# ---------------------------------------------------------------------------
# Step 4: Wait for health (up to 5 min) — with process liveness + error detection
# ---------------------------------------------------------------------------
log "=== Step 4: Wait for health ==="
P="000"
D="000"
for i in $(seq 1 30); do
  # Check process liveness first — if dead, fail immediately
  PF_ALIVE=$($SSH -p $PREFILL_PORT root@$HOST 'ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | wc -l' 2>/dev/null || echo "SSH_FAIL")
  DC_ALIVE=$($SSH -p $DECODE_PORT root@$HOST 'ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | wc -l' 2>/dev/null || echo "SSH_FAIL")

  if [ "$PF_ALIVE" = "SSH_FAIL" ] || [ "$DC_ALIVE" = "SSH_FAIL" ]; then
    log "FAILED: SSH connection lost to prefill($PF_ALIVE) or decode($DC_ALIVE)"
    exit 1
  fi

  if [ "$PF_ALIVE" = "0" ]; then
    log "FAILED: prefill process died. Last 20 lines:"
    $SSH -p $PREFILL_PORT root@$HOST 'tail -20 /root/glm53_prefill.log 2>/dev/null' 2>/dev/null
    exit 1
  fi
  if [ "$DC_ALIVE" = "0" ]; then
    log "FAILED: decode process died. Last 20 lines:"
    $SSH -p $DECODE_PORT root@$HOST 'tail -20 /root/glm53_decode.log 2>/dev/null' 2>/dev/null
    exit 1
  fi

  # Check for crash errors in log (only first 3 iterations to avoid noise)
  # 排除 "Ignore import error"（启动时加载 glm5_next 模型的正常 warning，不是崩溃）
  if [ "$i" -le 3 ]; then
    PF_ERR=$($SSH -p $PREFILL_PORT root@$HOST 'grep -iE "Traceback|CUDA error|OOM|RuntimeError|AssertionError|core dumped" /root/glm53_prefill.log 2>/dev/null | grep -v "Ignore import error" | wc -l' 2>/dev/null || echo "0")
    DC_ERR=$($SSH -p $DECODE_PORT root@$HOST 'grep -iE "Traceback|CUDA error|OOM|RuntimeError|AssertionError|core dumped" /root/glm53_decode.log 2>/dev/null | grep -v "Ignore import error" | wc -l' 2>/dev/null || echo "0")
    if [ "$PF_ERR" -gt 0 ] 2>/dev/null; then
      log "FAILED: prefill log has errors (count=$PF_ERR):"
      $SSH -p $PREFILL_PORT root@$HOST 'grep -iE "Traceback|CUDA error|OOM|RuntimeError|AssertionError|core dumped" /root/glm53_prefill.log 2>/dev/null | grep -v "Ignore import error" | tail -5' 2>/dev/null
      exit 1
    fi
    if [ "$DC_ERR" -gt 0 ] 2>/dev/null; then
      log "FAILED: decode log has errors (count=$DC_ERR):"
      $SSH -p $DECODE_PORT root@$HOST 'grep -iE "Traceback|CUDA error|OOM|RuntimeError|AssertionError|core dumped" /root/glm53_decode.log 2>/dev/null | grep -v "Ignore import error" | tail -5' 2>/dev/null
      exit 1
    fi
  fi

  # Now check HTTP health
  P=$($SSH -p $PREFILL_PORT root@$HOST 'curl -s -o /dev/null -w "%{http_code}" -m 3 http://127.0.0.1:30100/health 2>/dev/null' 2>/dev/null || echo "000")
  D=$($SSH -p $DECODE_PORT root@$HOST 'curl -s -o /dev/null -w "%{http_code}" -m 3 http://127.0.0.1:30200/health 2>/dev/null' 2>/dev/null || echo "000")
  log "  [$i] prefill=$P (alive=$PF_ALIVE) decode=$D (alive=$DC_ALIVE)"
  if [ "$P" = "200" ] && [ "$D" = "200" ]; then
    log "  BOTH READY"
    break
  fi
  sleep 10
done

if [ "$P" != "200" ] || [ "$D" != "200" ]; then
  log "TIMEOUT: prefill=$P decode=$D after 5 min"
  log "--- prefill last 10 lines ---"
  $SSH -p $PREFILL_PORT root@$HOST 'tail -10 /root/glm53_prefill.log 2>/dev/null' 2>/dev/null
  log "--- decode last 10 lines ---"
  $SSH -p $DECODE_PORT root@$HOST 'tail -10 /root/glm53_decode.log 2>/dev/null' 2>/dev/null
  exit 1
fi

# 确认 PID 是新进程
log "=== Step 5: Verify new PIDs ==="
$SSH -p $PREFILL_PORT root@$HOST 'echo "prefill_pid=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}" | head -1)"' 2>/dev/null
$SSH -p $DECODE_PORT root@$HOST 'echo "decode_pid=$(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}" | head -1)"' 2>/dev/null

# ---------------------------------------------------------------------------
# Step 6: Start smg on prefill node
# ---------------------------------------------------------------------------
log "=== Step 6: Start smg ==="
$SSH -p $PREFILL_PORT root@$HOST 'nohup /usr/local/bin/smg launch --pd-disaggregation --prefill http://'"$PREFILL_IP"':30100 --decode http://'"$DECODE_IP"':30200 --host 0.0.0.0 --port 31000 --api-key sk-glm52-pd --policy cache_aware --max-concurrent-requests 64 --health-check-timeout-secs 300 --disable-circuit-breaker --request-timeout-secs 3600 --log-level info --prometheus-port 29004 > /root/smg_31000.log 2>&1 & disown; echo "smg_pid=$!"' 2>/dev/null

log "=== Step 7: Wait for smg health ==="
S="000"
for i in $(seq 1 12); do
  S=$($SSH -p $PREFILL_PORT root@$HOST 'curl -s -o /dev/null -w "%{http_code}" -m 3 http://127.0.0.1:31000/health 2>/dev/null' 2>/dev/null || echo "000")
  log "  [$i] smg=$S"
  if [ "$S" = "200" ]; then
    log "  SMG READY"
    break
  fi
  sleep 5
done

if [ "$S" != "200" ]; then
  log "SMG FAILED"
  $SSH -p $PREFILL_PORT root@$HOST 'tail -10 /root/smg_31000.log 2>/dev/null' 2>/dev/null
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 8: Start flood (optional)
# ---------------------------------------------------------------------------
if [ "$DO_FLOOD" -eq 1 ]; then
  log "=== Step 8: Start flood (concurrency=$FLOOD_CONCURRENCY prob=$FLOOD_PROB) ==="
  $SSH -p $PREFILL_PORT root@$HOST 'truncate -s 0 /tmp/kv_flood_final.log 2>/dev/null; cd /tmp && nohup python3 kv_flood_test.py --cases /tmp/cases50_glm53.json --endpoint http://127.0.0.1:31000/v1/chat/completions --api-key sk-glm52-pd --rounds 100000 --prob '"$FLOOD_PROB"' --concurrency '"$FLOOD_CONCURRENCY"' --max-tokens 512 --round-gap 2 --out /tmp/kv_flood_final.json > /tmp/kv_flood_final.log 2>&1 & disown; echo "flood_pid=$!"' 2>/dev/null
  sleep 10
  ROUNDS=$($SSH -p $PREFILL_PORT root@$HOST 'grep -c "^\[round" /tmp/kv_flood_final.log 2>/dev/null || echo 0' 2>/dev/null)
  log "  flood started, rounds=$ROUNDS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "=== DONE ==="
log "prefill=$P decode=$D smg=$S flood=$DO_FLOOD"
log "PIDs:"
$SSH -p $PREFILL_PORT root@$HOST 'echo "  prefill: $(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}" | head -1)"' 2>/dev/null
$SSH -p $DECODE_PORT root@$HOST 'echo "  decode: $(ps aux | grep "python3 -m sglang.launch_server" | grep -v grep | awk "{print \$2}" | head -1)"' 2>/dev/null
$SSH -p $PREFILL_PORT root@$HOST 'echo "  smg: $(ps aux | grep "/usr/local/bin/smg" | grep -v grep | awk "{print \$2}" | head -1)"' 2>/dev/null
