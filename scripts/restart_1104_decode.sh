#!/bin/bash
# restart_1104_decode.sh — 健壮的 decode 重启脚本
# 彻底解决 SSH 命令导致 decode 启动失败的问题
#
# 问题根因：
#   ssh 'setsid nohup bash script & echo "launched"' 这种命令中，
#   SSH 断开时（超时/连接断），sshd 发 SIGKILL 到整个会话组，
#   setsid 也无法幸免（SIGKILL 不可捕获）。之前多次 decode "启动后
#   被杀" 都是这个问题（bash: line 1: XXXX Killed）。
#
# 解决方案：
#   1. 杀进程 → 等 GPU/端口完全释放
#   2. 用 at now 或 systemd-run 启动（完全脱离 SSH 会话）
#   3. 验证启动成功（进程数 > 0）
#   4. 失败自动重试一次

set -euo pipefail

SSH_CMD="ssh -p 1104 -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@8.222.11.182"
LOG_FILE="/root/glm53_decode.log"

echo "=== Step 1: 杀掉旧 decode 进程 ==="
$SSH_CMD 'for p in $(pgrep -f "python3 -m sglang.launch_serve[r]"); do kill -9 $p 2>/dev/null || true; done; sleep 3; echo "remaining: $(pgrep -fc "python3 -m sglang.launch_serve[r]" || echo 0)"' || true

echo "=== Step 2: 等 GPU/端口释放 ==="
$SSH_CMD '
for i in $(seq 1 10); do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    port_used=$(ss -tlnp 2>/dev/null | grep -c ":30200 " || true)
    if [ "$gpu_procs" -eq 0 ] && [ "$port_used" -eq 0 ]; then
        echo "GPU and port clear (iter $i)"
        break
    fi
    echo "iter $i: gpu_procs=$gpu_procs port_used=$port_used (waiting...)"
    sleep 3
done
fuser -k 30200/tcp 2>/dev/null || true
sleep 1
' || true

echo "=== Step 3: 启动 decode（用 systemd-run 完全脱离 SSH 会话）==="
# 方法 A: systemd-run（最可靠，完全脱离 SSH）
$SSH_CMD 'systemd-run --unit=glm53-decode --collect bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null 2>&1 && echo "systemd-run OK" || {
    echo "systemd-run failed, trying setsid+at..."
    # 方法 B: at now（at 调度器接管进程，完全脱离 SSH）
    echo "bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null" | at now 2>/dev/null && echo "at OK" || {
        echo "at failed, trying setsid..."
        # 方法 C: setsid + 完全脱离（最后手段）
        setsid bash -c "exec bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null" &
        echo "setsid OK"
    }
}'

echo "=== Step 4: 验证启动（等 60 秒检查进程）==="
sleep 5
for i in $(seq 1 12); do
    procs=$($SSH_CMD 'pgrep -fc "python3 -m sglang.launch_serve[r]" 2>/dev/null || echo 0' 2>/dev/null || echo "ssh_fail")
    echo "[$i] decode procs: $procs"
    if [ "$procs" != "0" ] && [ "$procs" != "ssh_fail" ] && [ "$procs" -gt 0 ] 2>/dev/null; then
        echo "=== decode started successfully ==="
        $SSH_CMD "tail -5 $LOG_FILE 2>/dev/null | awk 'length(\$0)<200' | tail -3" || true
        exit 0
    fi
    sleep 5
done

echo "=== WARNING: decode did not start after 60s ==="
echo "=== Checking last log ==="
$SSH_CMD "tail -20 $LOG_FILE 2>/dev/null | awk 'length(\$0)<250' | tail -10" || true

echo "=== Step 5: 重试一次 ==="
$SSH_CMD '
# 检查是否有 GPU 残留
gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$gpu_procs" -gt 0 ]; then
    echo "GPU still occupied, killing..."
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 $p 2>/dev/null || true
    done
    sleep 5
fi
# 清理可能的 systemd 单元残留
systemctl stop glm53-decode 2>/dev/null || true
systemctl reset-failed glm53-decode 2>/dev/null || true
rm -f /run/systemd/transient/glm53-decode.scope 2>/dev/null || true
sleep 2
# 重新启动
systemd-run --unit=glm53-decode --collect bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null 2>&1 || {
    echo "bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null" | at now
}
'

echo "=== Step 6: 再次验证 ==="
sleep 10
for i in $(seq 1 12); do
    procs=$($SSH_CMD 'pgrep -fc "python3 -m sglang.launch_serve[r]" 2>/dev/null || echo 0' 2>/dev/null || echo "ssh_fail")
    echo "[retry-$i] decode procs: $procs"
    if [ "$procs" != "0" ] && [ "$procs" != "ssh_fail" ] && [ "$procs" -gt 0 ] 2>/dev/null; then
        echo "=== decode started successfully (retry) ==="
        exit 0
    fi
    sleep 5
done

echo "=== FAILED: decode did not start after retry ==="
exit 1
