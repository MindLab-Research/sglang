#!/usr/bin/env bash
# ============================================================
# OTel Collector 一键部署脚本 (B300 PD metrics → Grafana)
# 采集: prefill:30100, decode:30200, router:29001
# 推送: otel-i18n.macaron.xin:4317 (OTLP gRPC)
# ============================================================
set -euo pipefail

CONFIG_DIR=/root
CONFIG_FILE=${CONFIG_DIR}/otel-collector-config.yaml
BIN=/usr/local/bin/otelcol
PID_FILE=/var/run/otelcol.pid
LOG_FILE=/var/log/otelcol.log

case "${1:-start}" in
    start)
        echo "=== Starting OTel Collector ==="
        if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "Already running (PID $(cat $PID_FILE))"
            exit 0
        fi
        nohup "$BIN" --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 2
        if kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "✅ Started (PID $(cat $PID_FILE))"
            echo "   Config: $CONFIG_FILE"
            echo "   Log: $LOG_FILE"
        else
            echo "❌ Failed to start, check $LOG_FILE"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        ;;
    stop)
        echo "=== Stopping OTel Collector ==="
        if [ -f "$PID_FILE" ]; then
            kill "$(cat $PID_FILE)" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "✅ Stopped"
        else
            echo "Not running"
        fi
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "✅ Running (PID $(cat $PID_FILE))"
        else
            echo "❌ Not running"
        fi
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
