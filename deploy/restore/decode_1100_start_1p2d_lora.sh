#!/bin/bash
# 1P2D 4-LoRA decode @ 1100 — v15_patched venv, 参数=1102/1104 测试集群
export TVM_FFI_CUDA_ARCH_LIST="10.0a"
export MOONCAKE_DISABLE_HIP_DMABUF="1"
export IBV_ACCESS_RELAXED_ORDERING="1"
export MC_IB_PCI_RELAXED_ORDERING="1"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE="1"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="1000"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="600"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_MOE_PADDING="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_DSA_SLOT_OOB_DIAG=1
export SGLANG_DSA_STAGE_SYNC=1
export SGLANG_ENABLE_ASYNC_ASSERT=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE=/root/gpucoredump/core_%h_%p.ncu-coredump
export CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

ps aux | grep -aE "sglang::scheduler|launch_server" | grep -av grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

nohup /root/v15_patched/bin/python3 -m sglang.launch_server \
    --model-path /root/glm52_local/base \
    --served-model-name glm52-fp8-official \
    --host 0.0.0.0 --port 30200 --tp 8 \
    --kv-cache-dtype fp8_e4m3 --enable-cache-report \
    --page-size 64 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
    --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --model-impl sglang \
    --enable-lora \
    --lora-paths L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3 \
    --max-lora-rank 16 --max-loaded-loras 4 --max-loras-per-batch 4 \
    --lora-use-virtual-experts --max-lora-chunk-size 128 \
    --mem-fraction-static 0.90 --skip-server-warmup --enable-metrics \
    --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device mlx5_0 \
    --disaggregation-mode decode --dcp-size 4 \
    --speculative-algorithm EAGLE --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
    > /root/decode.log 2>&1 < /dev/null &
echo "decode_pid=$!"
