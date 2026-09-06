#!/bin/bash
# =====================================================================
# GLM-5.3 b300 PD 集群 - decode 端还原脚本
# 最终正确 image: b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final
#
# 100% 还原当前运行状态:
#   - image 内置 HEAD 代码 (conn.py=e79f3463) + 正确 engine.so (cf881774)
#   - image 自带正确 engine.so, 因此【不再挂载任何 engine.so】
#   - 也不再挂载 overlay (sglang 代码已在 image 内)
#   - 仅挂载数据/缓存目录
#
# 注意: decode 在 decode 节点运行 (本脚本==decode 节点)
# =====================================================================
set -euo pipefail

# 部署节点 IP (按实际环境修改)
DECODE_IP="172.31.45.101"
BOOTSTRAP_PORT="30011"

IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final"
CONTAINER_NAME="sglang-decode"

# 模型/缓存路径
MODEL_PATH="/nvme/models/GLM-5.3"
HICACHE_DIR="/root/hicache"
DEEPGEMM_CACHE="/root/.cache/deep_gemm"
TVMFFI_CACHE="/root/.cache/tvm-ffi"

echo "=== 停止已存在的 decode 容器 ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

echo "=== 启动 decode ==="
docker run -d --name "$CONTAINER_NAME" \
  --network host --runtime nvidia --gpus all --privileged --ipc=host \
  --security-opt label=disable --shm-size 67108864 \
  -w /sgl-workspace/sglang \
  --entrypoint python3 \
  --env-file "$(dirname "$0")/decode.env" \
  -v "$MODEL_PATH:/nvme/models" \
  -v "/nvme/hicache:$HICACHE_DIR" \
  -v "/nvme/sglang-cache/deep_gemm:$DEEPGEMM_CACHE" \
  -v "/nvme/sglang-cache/tvm-ffi:$TVMFFI_CACHE" \
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

echo "Decode started, container: $CONTAINER_NAME"
echo "等待 ready... 检查: docker logs $CONTAINER_NAME 2>&1 | grep 'fired up and ready'"
