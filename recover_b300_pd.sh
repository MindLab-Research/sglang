#!/bin/bash
set -euo pipefail

# ============================================================
# B300 PD 分离启动脚本 (LayerSplit + EAGLE + LoRA + HiCache)
# B300-1 (10.0.0.66): prefill + CP8 + LayerSplit + HiCache
# B300-2 (10.0.0.67): decode + EAGLE + HiCache
# Router: B300-1
#
# 用法:
#   bash recover_b300_pd.sh prefill   # 在 B300-1 上运行
#   bash recover_b300_pd.sh decode    # 在 B300-2 上运行
#   bash recover_b300_pd.sh router    # 在 B300-1 上运行
#   bash recover_b300_pd.sh compile   # 预编译 DeepGEMM
#
# ⚠️ 关键: SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
#    Prefill 使用 CP=8，8个rank各自有独立ZMQ socket
#    如果 decode 不设此变量，只有 TP0 收到 bootstrap，TP1-7 卡在 Bootstrapping 死循环
#
# ⚠️ 重启时 prefill 和 decode 必须同时重启，否则 RDMA 连接断开后 decode 会假死
# ============================================================

export TVM_FFI_CUDA_ARCH_LIST="10.0a"
export MOONCAKE_DISABLE_HIP_DMABUF=1
export IBV_ACCESS_RELAXED_ORDERING=1
export MC_IB_PCI_RELAXED_ORDERING=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_MOE_PADDING=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
#
# HiCache L3 file storage: use 11T disk (/root/hicache), cap 200G, keep 10G free
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G

MODEL=/root/glm52_local/base
LORA_PATHS="L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3"

COMMON_ARGS="
    --model-path $MODEL \
    --served-model-name glm52-fp8-official \
    --host 0.0.0.0 \
    --tp 8 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-cache-report \
    --page-size 64 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --watchdog-timeout 3600 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --moe-runner-backend triton \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --model-impl sglang \
    --enable-lora \
    --lora-paths $LORA_PATHS \
    --max-lora-rank 16 \
    --max-loaded-loras 4 \
    --max-loras-per-batch 4 \
    --lora-use-virtual-experts
"

case "${1:-prefill}" in
    prefill)
        echo "=== Starting Prefill on B300-1 ==="
        python3 -m sglang.launch_server $COMMON_ARGS \
            --port 30100 \
            --mem-fraction-static 0.85 \
            --enable-metrics \
            --enable-hierarchical-cache \
            --hicache-ratio 1 \
            --hicache-write-policy write_back \
            --hicache-mem-layout page_first \
            --hicache-storage-backend file \
            --file-storage-path /root/hicache \
            --enable-prefill-cp --cp-strategy interleave \
            --enable-dsa-prefill-cp-layersplit \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port 8998 \
            --disaggregation-ib-device mlx5_0 \
            --disaggregation-mode prefill
        ;;
    decode)
        echo "=== Starting Decode on B300-2 ==="
        python3 -m sglang.launch_server $COMMON_ARGS \
            --port 30200 \
            --mem-fraction-static 0.85 \
            --skip-server-warmup \
            --enable-metrics \
            --speculative-algorithm EAGLE --speculative-num-steps 5 \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
            --cuda-graph-max-bs-decode 32 --max-running-requests 32 \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port 8998 \
            --disaggregation-ib-device mlx5_0 \
            --disaggregation-mode decode \
            --dcp-size 1
        ;;
    compile)
        echo "=== Pre-compiling DeepGEMM kernels ==="
        python3 -m sglang.compile_deep_gemm \
            --model-path $MODEL --tp 8 \
            --kv-cache-dtype fp8_e4m3 \
            --moe-runner-backend triton \
            --model-impl sglang \
            --load-format dummy
        ;;
    router)
        echo "=== Starting Router ==="
        # IMPORTANT: use lsof to kill old router before starting new one
        lsof -i :30000 -t 2>/dev/null | xargs kill -9 2>/dev/null
        sleep 2
        python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill http://10.0.0.66:30100 \
            --decode http://10.0.0.67:30200 \
            --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
            --policy cache_aware --max-concurrent-requests 64 \
            --health-check-timeout-secs 300 \
            --disable-circuit-breaker \
            --request-timeout-secs 3600 \
            --log-level info \
            --prometheus-port 29001
        ;;
    *)
        echo "Usage: $0 {prefill|decode|router|compile}"
        exit 1
        ;;
esac
