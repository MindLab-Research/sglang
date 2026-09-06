#!/bin/bash
# =====================================================================
# GLM-5.3 b300 PD 集群 - router 端还原脚本
# 最终正确 image: b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final
#
# router 在 prefill 节点运行, 连接两端引擎。必须在 PD 双端 ready 后启动。
# 完整 URL 参数必须按实际环境修改 IP。
# =====================================================================
set -euo pipefail

PREfill_URL="http://172.31.47.105:30100"
PREfill_BOOTSTRAP="30012"
DECODE_URL="http://172.31.45.101:30002"
# api-key 不硬编码, 从环境变量读 (真值放 secrets.env)
ROUTER_API_KEY="${GLM53_ROUTER_API_KEY:?must set GLM53_ROUTER_API_KEY}"

IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final"
CONTAINER_NAME="glm53-router"

echo "=== 停止已存在的 router 容器 ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

echo "=== 启动 router (PD 模式) ==="
docker run -d --name "$CONTAINER_NAME" \
  --network host --runtime nvidia --gpus all --privileged --ipc=host \
  --security-opt label=disable --shm-size 67108864 \
  -w /sgl-workspace/sglang \
  --entrypoint python3 \
  --env-file "$(dirname "$0")/router.env" \
  "$IMAGE" \
  -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "$PREfill_URL" "$PREfill_BOOTSTRAP" \
  --decode "$DECODE_URL" \
  --host 0.0.0.0 --port 31000 \
  --api-key "$ROUTER_API_KEY" \
  --policy cache_aware --max-concurrent-requests 64 \
  --health-check-timeout-secs 300 --request-timeout-secs 600 \
  --log-level info

echo "Router started, container: $CONTAINER_NAME"
echo "健康检查: curl -s http://127.0.0.1:31000/health"
