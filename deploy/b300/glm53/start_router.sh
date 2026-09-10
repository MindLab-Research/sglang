#!/bin/bash
# =====================================================================
# b300 GLM-5.3 PD — router 启动脚本 (最终版 v2 image)
# image: v0.5.15.post1-cuda13-b200-glm53-final-v3 (无需挂载 mooncake/代码)
# 运行节点: prefill 节点 (B300 spot prefill, 内网 172.31.47.105)
#
# ⛔ 启动前提: PD 双端 (decode+prefill) 已 ready (fired up and ready)
#    PD ready 前 router 必须 = 0 (铁律)
#
# api-key 不硬编码: 调用前必须 export GLM53_ROUTER_API_KEY (真值见仓库根 secrets.env)
#    export GLM53_ROUTER_API_KEY=sk-... && bash start_router.sh
# =====================================================================
set -euo pipefail

ROUTER_API_KEY="${GLM53_ROUTER_API_KEY:?must set GLM53_ROUTER_API_KEY (真值见 secrets.env)}"

IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v3"
CONTAINER_NAME="glm53-router"

PREFILL_URL="http://172.31.47.105:30100"
PREFILL_BOOTSTRAP="30012"
DECODE_URL="http://172.31.45.101:30002"

echo "=== 停止已存在的 router ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

echo "=== 启动 router (v2 image) ==="
docker run -d --name "$CONTAINER_NAME" \
  --network host --privileged --ipc=host \
  --security-opt label=disable --shm-size 67108864 \
  -w /sgl-workspace/sglang \
  --entrypoint python3 \
  --env-file "$(dirname "$0")/b200_aligned.env" \
  -v "/nvme/models:/nvme/models" \
  "$IMAGE" \
  -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "$PREFILL_URL" "$PREFILL_BOOTSTRAP" \
  --decode "$DECODE_URL" \
  --host 0.0.0.0 --port 31000 \
  --api-key "$ROUTER_API_KEY" \
  --policy cache_aware --max-concurrent-requests 64 \
  --health-check-timeout-secs 300 --request-timeout-secs 600 \
  --log-level info

echo "Router started: $CONTAINER_NAME"
echo "验证: curl -s http://127.0.0.1:31000/health (应 200)"
