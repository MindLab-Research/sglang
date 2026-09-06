#!/bin/bash
# =====================================================================
# GLM-5.3 b300 PD 集群 - prefill 端还原脚本
# 最终正确 image: b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final
#
# 100% 还原当前运行状态:
#   - image 内置 HEAD 代码 + 正确 engine.so (cf881774)
#   - 不挂 overlay / engine.so; 仅挂数据/缓存目录
#   - prefill 走 CP=8 layersplit + HiCache
#
# 注意: prefill 在 prefill 节点运行
# =====================================================================
set -euo pipefail

BOOTSTRAP_PORT="30012"

IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final"
CONTAINER_NAME="glm53-prefill"

MODEL_PATH="/nvme/models/GLM-5.3"
HICACHE_DIR="/root/hicache"
DEEPGEMM_CACHE="/root/.cache/deep_gemm"
TVMFFI_CACHE="/root/.cache/tvm-ffi"

echo "=== 停止已存在的 prefill 容器 ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

echo "=== 启动 prefill ==="
docker run -d --name "$CONTAINER_NAME" \
  --network host --runtime nvidia --gpus all --privileged --ipc=host \
  --security-opt label=disable --shm-size 67108864 \
  -w /sgl-workspace/sglang \
  --entrypoint python3 \
  --env-file "$(dirname "$0")/prefill.env" \
  -v "$MODEL_PATH:/nvme/models" \
  -v "/nvme/hicache:$HICACHE_DIR" \
  -v "/nvme/sglang-cache/deep_gemm:$DEEPGEMM_CACHE" \
  -v "/nvme/sglang-cache/tvm-ffi:$TVMFFI_CACHE" \
  "$IMAGE" \
  -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name glm53 \
  --host 0.0.0.0 --port 30100 --tp 8 --kv-cache-dtype fp8_e4m3 \
  --enable-cache-report --page-size 64 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
  --model-impl sglang \
  --enable-metrics --trust-remote-code \
  --mem-fraction-static 0.85 \
  --max-total-tokens 5000000 --disable-overlap-schedule \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-write-policy write_back --hicache-mem-layout page_first \
  --hicache-storage-backend file --file-storage-path "$HICACHE_DIR" \
  --enable-prefill-cp --cp-strategy interleave \
  --enable-dsa-prefill-cp-layersplit \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-mode prefill

echo "Prefill started, container: $CONTAINER_NAME"
echo "等待 ready... 检查: docker logs $CONTAINER_NAME 2>&1 | grep 'fired up and ready'"
