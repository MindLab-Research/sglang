#!/usr/bin/env bash
# ================================================================
# sync_lora_weights.sh — distribute new LoRA weights to ALL nodes of
# a PD cluster (prefill and decode are separate hosts; the engine's
# /load_lora_adapter expects the weights to exist locally on each node).
#
# Usage:
#   bash scripts/sync_lora_weights.sh <LORA_DIR> <NODE_LIST>
#   bash scripts/sync_lora_weights.sh /root/glm52_local/loras "root@10.0.0.75:1021 root@10.0.0.67:1022"
#
# Notes:
#   - rsync over SSH (uses the same auth as your normal ssh).
#   - Run this BEFORE calling POST /v1/control/models with a remote URL —
#     the router downloads to its own node; the other PD nodes need the
#     weights too (or use shared storage / an OSS mount).
# ================================================================
set -euo pipefail

LORA_DIR="${1:?usage: sync_lora_weights.sh <LORA_DIR> <NODE_LIST>}"
NODE_LIST="${2:?usage: sync_lora_weights.sh <LORA_DIR> <NODE_LIST>}"
LORA_BASE="$(dirname "$LORA_DIR")"
LORA_NAME="$(basename "$LORA_DIR")"

if [ ! -d "$LORA_DIR" ]; then
    echo "ERROR: $LORA_DIR does not exist locally" >&2
    exit 1
fi

echo "Syncing LoRA '$LORA_NAME' to nodes: $NODE_LIST"
for node in $NODE_LIST; do
    # node format: user@host[:ssh_port]
    userhost="${node%:*}"
    port="${node##*:}"
    if [ "$port" = "$node" ]; then
        port="22"
    fi
    echo "  -> $userhost:$port ..."
    rsync -avz -e "ssh -p $port -o BatchMode=yes -o ConnectTimeout=10" \
        "$LORA_DIR/" "$userhost:$LORA_BASE/$LORA_NAME/" \
        --exclude='__pycache__' --exclude='*.pyc'
done

echo "Done. Now you can:"
echo "  curl -X POST <router>/v1/control/models -d '{\"name\":\"$LORA_NAME\",\"type\":\"lora\",\"path\":\"$LORA_DIR\"}'"
