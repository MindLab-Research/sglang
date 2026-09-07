#!/bin/bash
# =====================================================================
# b300 GLM-5.3 PD — decode 启动脚本 (最终版 v2 image, 无需挂载 mooncake/代码)
# image: v0.5.15.post1-cuda13-b200-glm53-final-v3
#   - b200 完整 mooncake 包内置 (engine.so 8ea08f1e + ep_*.so 绑定)
#   - HEAD 代码内置 (conn.py=e79f3463, 含 DCP 修复)
#   - 只需挂载数据目录 (models/hicache/cache)
#   - env 用 b200_aligned.env (对齐 b200 2p2d 生产)
# 运行节点: decode 节点 (B300 spot decode, 内网 172.31.45.101)
# =====================================================================
set -euo pipefail

IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v3"
CONTAINER_NAME="sglang-decode"
BOOTSTRAP_PORT="30011"
MODEL_PATH="/nvme/models/GLM-5.3"
HICACHE_DIR="/root/hicache"

echo "=== 停止已存在的 decode ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

echo "=== 启动 decode (v2 image, 无需挂载 mooncake/代码) ==="
docker run -d --name "$CONTAINER_NAME" \
  --network host --runtime nvidia --gpus all --privileged --ipc=host \
  --security-opt label=disable --shm-size 67108864 \
  -w /sgl-workspace/sglang \
  --entrypoint python3 \
  --env-file "$(dirname "$0")/b200_aligned.env" \
  -v "/nvme/models:/nvme/models" \
  -v "/nvme/hicache:$HICACHE_DIR" \
  -v "/nvme/sglang-cache/deep_gemm:/root/.cache/deep_gemm" \
  -v "/nvme/sglang-cache/tvm-ffi:/root/.cache/tvm-ffi" \
  "$IMAGE" \
  -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name glm53 \
  --host 0.0.0.0 --tp 8 --kv-cache-dtype fp8_e4m3 \
  --enable-cache-report --page-size 64 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
  --model-impl sglang \
  --enable-metrics --trust-remote-code \
  --port 30002 --mem-fraction-static 0.90 \
  --skip-server-warmup \
  --cuda-graph-max-bs-decode 64 \
  --max-running-requests 48 \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-mode decode \
  --disaggregation-decode-enable-radix-cache \
  --dcp-size 4 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-write-policy write_back --hicache-mem-layout page_first \
  --hicache-storage-backend file --file-storage-path "$HICACHE_DIR"

echo "Decode started: $CONTAINER_NAME"
echo "验证: docker logs $CONTAINER_NAME 2>&1 | grep 'fired up and ready'"
