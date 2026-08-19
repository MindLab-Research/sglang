#!/usr/bin/env bash
# ============================================================
# 1P4D PD 分离部署脚本 (5节点: 1 prefill + 4 decode)
# 使用 /opt/sglang-venv (基于 b300-pd-latest 镜像)
# 用法:
#   bash start_1p4d.sh prefill    # 在 1100 上运行
#   bash start_1p4d.sh decode     # 在 1101-1104 上运行
#   bash start_1p4d.sh router     # 在任意节点运行 (通常 1100)
#
# ⚠️ prefill 和所有 decode 必须同时重启 (RDMA 连接重建)
# ============================================================
set -euo pipefail

PYTHON=/opt/sglang-venv/bin/python
MODEL=/root/glm52_local/base
LORA_PATHS="L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3"

# 共享环境变量
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
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G

case "${1:-}" in
    prefill)
        echo "=== Starting Prefill ==="
        exec $PYTHON -m sglang.launch_server \
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
        ;;

    decode)
        echo "=== Starting Decode (EAGLE) ==="
        exec $PYTHON -m sglang.launch_server \
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
        ;;

    router)
        echo "=== Starting Router (1P4D) ==="
        exec $PYTHON -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill http://10.0.58.32:30100 \
            --decode http://10.0.58.31:30200 \
            --decode http://10.0.58.29:30200 \
            --decode http://10.0.58.30:30200 \
            --decode http://10.0.58.28:30200 \
            --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
            --policy cache_aware --max-concurrent-requests 64 \
            --health-check-timeout-secs 300 \
            --disable-circuit-breaker \
            --request-timeout-secs 600 \
            --log-level info \
            --prometheus-port 29001
        ;;

    *)
        echo "Usage: $0 {prefill|decode|router}"
        exit 1
        ;;
esac
