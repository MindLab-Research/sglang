#!/bin/bash
# preserve_crash.sh — 崩溃现场保全（2026-08-20 纪律：崩溃后严禁先重启，必须先保全现场）
# 用法：nohup bash /root/preserve_crash.sh > /root/preserve_crash.log 2>&1 &
# 检测 launch_server 进程数 N→0（崩溃/被杀），立即备份 decode.log + dmesg + coredump 列表。
# 备份到 /root/crash_preserves/<YYYYmmdd_HHMMSS>/

LOGDIR=/root/crash_preserves
mkdir -p "$LOGDIR"
prev=0
while true; do
  cur=$(ps aux | grep '[l]aunch_server' | wc -l)
  if [ "$cur" -eq 0 ] && [ "$prev" -gt 0 ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    D="$LOGDIR/$TS"
    mkdir -p "$D"
    # decode.log（可能已被重启覆盖——用 tail 抢救 + 复制当前）
    cp /root/decode.log "$D/decode.log" 2>/dev/null
    # dmesg Xid 窗口
    dmesg -T 2>/dev/null | grep -iE 'xid|nvrm' | tail -50 > "$D/dmesg_xid.txt" 2>/dev/null
    # coredump 列表
    ls -la /root/gpucoredump/ 2>/dev/null > "$D/gpucoredump_list.txt"
    # 进程快照
    ps aux | grep -E '[l]aunch_server|[s]cheduler' > "$D/ps_snapshot.txt" 2>/dev/null
    echo "[$(date '+%F %T')] CRASH DETECTED (procs $prev -> 0), preserved to $D" >> /root/preserve_crash.log
    # 等重启完成后备份"崩溃后新日志"对比
    sleep 180
    cp /root/decode.log "$D/decode.log.after_restart" 2>/dev/null
    echo "[$(date '+%F %T')] post-restart log backed up" >> /root/preserve_crash.log
  fi
  prev=$cur
  sleep 5
done
