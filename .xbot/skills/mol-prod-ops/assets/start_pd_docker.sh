#!/usr/bin/env bash
# ============================================================
# B300 PD 分离部署 — Docker 启动脚本
# 基于 sglang-b300:v0.5.15 镜像
# 用法:
#   bash start_pd_docker.sh prefill    # 在 prefill 节点运行
#   bash start_pd_docker.sh decode     # 在 decode 节点运行
#   bash start_pd_docker.sh router     # 在 router 节点运行 (通常与 prefill 同机)
#
# ⚠️ 关键: SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
#    Prefill 使用 CP=8，8个rank各自有独立ZMQ socket
#    如果 decode 不设此变量，只有 TP0 收到 bootstrap，TP1-7 卡在 Bootstrapping 死循环
#
# ⚠️ 重启时 prefill 和 decode 必须同时重启，否则 RDMA 连接断开后 decode 会假死
# ============================================================
set -euo pipefail

IMAGE="sglang-b300:v0.5.15"

# ============================================================
# 共享环境变量 (prefill + decode 完全一致)
# ============================================================
COMMON_ENV=(
    -e TVM_FFI_CUDA_ARCH_LIST="10.0a"
    -e MOONCAKE_DISABLE_HIP_DMABUF=1
    -e IBV_ACCESS_RELAXED_ORDERING=1
    -e MC_IB_PCI_RELAXED_ORDERING=1
    -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
    -e SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
    -e SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
    -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
    -e SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
    -e SGLANG_MOE_PADDING=1
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
)

# HiCache L3 文件存储 (decode 脚本中有设, prefill 通过 --file-storage-path 参数)
HICACHE_ENV=(
    -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
    -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
    -e SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G
)

MODEL=/root/glm52_local/base
LORA_PATHS="L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3"

COMMON_ARGS="
    --model-path $MODEL \
    --served-model-name glm52-fp8-official \
    --host 0.0.0.0 --tp 8 --kv-cache-dtype fp8_e4m3 \
    --enable-cache-report --page-size 128 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
    --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce --model-impl sglang --enable-lora \
    --lora-paths $LORA_PATHS --max-lora-rank 16 --max-loaded-loras 4 \
    --max-loras-per-batch 4 --lora-use-virtual-experts \
    --enable-metrics
"

case "${1:-}" in
    prefill)
        echo "=== Starting Prefill (sglang-b300) ==="
        docker run -d --name sglang-prefill \
            --gpus all \
            --network host \
            --privileged \
            -v /root/glm52_local:/root/glm52_local \
            -v /root/hicache:/root/hicache \
            -v /root/.cache/deep_gemm:/root/.cache/deep_gemm \
            -v /root/.cache/tvm-ffi:/root/.cache/tvm-ffi \
            "${COMMON_ENV[@]}" \
            "${HICACHE_ENV[@]}" \
            "$IMAGE" \
            python3 -m sglang.launch_server \
                --model-path $MODEL \
                --served-model-name glm52-fp8-official \
                --host 0.0.0.0 --tp 8 --kv-cache-dtype fp8_e4m3 \
                --enable-cache-report --page-size 128 \
                --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
                --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
                --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
                --disable-custom-all-reduce --model-impl sglang --enable-lora \
                --lora-paths $LORA_PATHS --max-lora-rank 16 --max-loaded-loras 4 \
                --max-loras-per-batch 4 --lora-use-virtual-experts \
                --port 30100 --mem-fraction-static 0.85 --enable-metrics \
                --enable-hierarchical-cache --hicache-ratio 1 \
                --hicache-write-policy write_back --hicache-mem-layout page_first \
                --hicache-storage-backend file --file-storage-path /root/hicache \
                --enable-prefill-cp --cp-strategy interleave \
                --enable-dsa-prefill-cp-layersplit \
                --disaggregation-transfer-backend mooncake \
                --disaggregation-bootstrap-port 8998 \
                --disaggregation-ib-device mlx5_0 \
                --disaggregation-mode prefill
        echo "Prefill container started: sglang-prefill"
        echo "Logs: docker logs -f sglang-prefill"
        ;;

    decode)
        echo "=== Starting Decode (sglang-b300) ==="
        docker run -d --name sglang-decode \
            --gpus all \
            --network host \
            --privileged \
            -v /root/glm52_local:/root/glm52_local \
            -v /root/hicache:/root/hicache \
            -v /root/.cache/deep_gemm:/root/.cache/deep_gemm \
            -v /root/.cache/tvm-ffi:/root/.cache/tvm-ffi \
            "${COMMON_ENV[@]}" \
            "${HICACHE_ENV[@]}" \
            "$IMAGE" \
            python3 -m sglang.launch_server \
                --model-path $MODEL \
                --served-model-name glm52-fp8-official \
                --host 0.0.0.0 --tp 8 --kv-cache-dtype fp8_e4m3 \
                --enable-cache-report --page-size 128 \
                --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
                --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
                --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
                --disable-custom-all-reduce --model-impl sglang --enable-lora \
                --lora-paths $LORA_PATHS --max-lora-rank 16 --max-loaded-loras 4 \
                --max-loras-per-batch 4 --lora-use-virtual-experts \
                --port 30200 --mem-fraction-static 0.90 --skip-server-warmup --enable-metrics \
                --cuda-graph-max-bs-decode 16 \
                --speculative-algorithm EAGLE --speculative-num-steps 5 \
                --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
                --max-running-requests 16 \
                --disaggregation-transfer-backend mooncake \
                --disaggregation-bootstrap-port 8998 \
                --disaggregation-ib-device mlx5_0 \
                --disaggregation-mode decode
        echo "Decode container started: sglang-decode"
        echo "Logs: docker logs -f sglang-decode"
        ;;

    router)
        echo "=== Starting Router ==="
        # Router 不需要 GPU，用 --network host 直接访问 prefill/decode
        docker run -d --name sglang-router \
            --network host \
            "$IMAGE" \
            python3 -m sglang_router.launch_router \
                --pd-disaggregation \
                --prefill http://10.0.0.66:30100 \
                --decode http://10.0.0.67:30200 \
                --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
                --policy cache_aware --max-concurrent-requests 32 \
                --health-check-timeout-secs 300 \
                --disable-circuit-breaker \
                --request-timeout-secs 600 \
                --log-level info \
                --prometheus-port 29001
        echo "Router container started: sglang-router"
        echo "Logs: docker logs -f sglang-router"
        ;;

    stop)
        echo "=== Stopping all ==="
        docker stop sglang-prefill sglang-decode sglang-router 2>/dev/null || true
        docker rm sglang-prefill sglang-decode sglang-router 2>/dev/null || true
        echo "Stopped and removed all containers"
        ;;

    *)
        echo "Usage: $0 {prefill|decode|router|stop}"
        echo ""
        echo "  prefill  - Start prefill server (port 30100)"
        echo "  decode   - Start decode server with EAGLE (port 30200)"
        echo "  router   - Start PD disaggregation router (port 30000)"
        echo "  stop     - Stop and remove all containers"
        echo ""
        echo "⚠️  prefill 和 decode 必须同时重启 (mooncake RDMA 连接需要重建)"
        exit 1
        ;;
esac
