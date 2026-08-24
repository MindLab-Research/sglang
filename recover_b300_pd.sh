#!/bin/bash
set -euo pipefail

# ============================================================
# B300 PD 分离启动脚本 —— 生产 1P2D 配置
# 模型: GLM-5.2-FP8 + EAGLE + LoRA 虚拟专家 + HiCache + CP layer-split + DCP
# 拓扑: 1 prefill + 2 decode + smg(router:30000 / gateway:31001)
#
# ⚠️ 本脚本【不含集群 IP/URL】，全部通过环境变量传入（见下方"必需环境变量"）。
#    缺失时直接报错退出，避免误连到错误集群。
#
# 用法:
#   bash recover_b300_pd.sh prefill   # 在 prefill 节点运行
#   bash recover_b300_pd.sh decode    # 在每个 decode 节点运行
#   bash recover_b300_pd.sh router    # 在 router 节点运行 (smg PD router :30000)
#   bash recover_b300_pd.sh gateway   # 在 router 节点运行 (第二层 smg gateway :31001)
#   bash recover_b300_pd.sh compile   # 预编译 DeepGEMM kernels
#
# 必需环境变量 (IP/URL 用 env 传入, 缺失即报错):
#   PREFILL_URL    prefill 引擎 base URL          e.g. http://10.0.58.34:30100
#   DECODE_URLS    空格分隔的 decode base URL 列表  e.g. "http://10.0.58.38:30200 http://10.0.58.36:30200"
#
# 可选环境变量 (带生产默认值):
#   API_KEY        smg 认证 key    (默认 sk-glm52-pd)
#   ROUTER_PORT    30000          GATEWAY_PORT 31001
#   PREFILL_PORT   30100          DECODE_PORT 30200
#   MODEL          /root/glm52_local/base
#   LORA_PATHS     "L0=... L1=... L2=... L3=..."
#   HICACHE_DIR    /root/hicache
# ============================================================

# ---------- 必需环境变量 (无默认, 缺失即退出) ----------
: "${PREFILL_URL:?必须通过环境变量 PREFILL_URL 传入 prefill 引擎 URL (如 http://IP:30100)}"
: "${DECODE_URLS:?必须通过环境变量 DECODE_URLS 传入 decode 引擎 URL 列表 (空格分隔, 如 'http://IP:30200 http://IP2:30200')}"

# ---------- 可选环境变量 (带生产默认值) ----------
MODEL="${MODEL:-/root/glm52_local/base}"
LORA_PATHS="${LORA_PATHS:-L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3}"
API_KEY="${API_KEY:-sk-glm52-pd}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
GATEWAY_PORT="${GATEWAY_PORT:-31001}"
PREFILL_PORT="${PREFILL_PORT:-30100}"
DECODE_PORT="${DECODE_PORT:-30200}"
PREFILL_MEM_FRACTION="${PREFILL_MEM_FRACTION:-0.90}"
DECODE_MEM_FRACTION="${DECODE_MEM_FRACTION:-0.90}"
HICACHE_DIR="${HICACHE_DIR:-/root/hicache}"

# ---------- 共享 env (prefill/decode 两端一致, 生产实际取值) ----------
export_common_env() {
    export TVM_FFI_CUDA_ARCH_LIST="10.0a"
    export MOONCAKE_DISABLE_HIP_DMABUF=1
    export IBV_ACCESS_RELAXED_ORDERING=1
    export MC_IB_PCI_RELAXED_ORDERING=1
    export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
    export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
    export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
    export SGLANG_MOE_PADDING=1
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
    export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
}

# ---------- prefill 额外 env ----------
export_prefill_env() {
    export_common_env
    export SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
    export SGLANG_DISAGGREGATION_QUEUE_SIZE=64
    export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=256
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_DIR"
    export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
    export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G
}

# ---------- decode 额外 env ----------
export_decode_env() {
    export_common_env
    export SGLANG_DSA_SLOT_OOB_DIAG=1
    export SGLANG_DSA_STAGE_SYNC=1
    export SGLANG_ENABLE_ASYNC_ASSERT=1
}

# ---------- 共享启动参数 (不带端口/IP/模式, prefill 与 decode 共用) ----------
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
    --model-impl sglang \
    --enable-lora \
    --lora-paths $LORA_PATHS \
    --max-lora-rank 16 \
    --max-loaded-loras 5 \
    --max-loras-per-batch 5 \
    --lora-use-virtual-experts \
    --max-lora-chunk-size 128 \
    --enable-metrics
"

case "${1:-prefill}" in
    prefill)
        echo "=== Starting Prefill (port $PREFILL_PORT) ==="
        export_prefill_env
        python3 -m sglang.launch_server $COMMON_ARGS \
            --port "$PREFILL_PORT" \
            --mem-fraction-static "$PREFILL_MEM_FRACTION" \
            --max-total-tokens 9000000 \
            --disable-overlap-schedule \
            --enable-hierarchical-cache \
            --hicache-ratio 1 \
            --hicache-write-policy write_back \
            --hicache-mem-layout page_first \
            --hicache-storage-backend file \
            --file-storage-path "$HICACHE_DIR" \
            --enable-prefill-cp --cp-strategy interleave \
            --enable-dsa-prefill-cp-layersplit \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port 8998 \
            --disaggregation-ib-device mlx5_0 \
            --disaggregation-mode prefill
        ;;
    decode)
        echo "=== Starting Decode (port $DECODE_PORT) ==="
        export_decode_env
        python3 -m sglang.launch_server $COMMON_ARGS \
            --port "$DECODE_PORT" \
            --mem-fraction-static "$DECODE_MEM_FRACTION" \
            --disable-custom-all-reduce \
            --skip-server-warmup \
            --cuda-graph-max-bs-decode 64 \
            --max-running-requests 64 \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port 8998 \
            --disaggregation-ib-device mlx5_0 \
            --disaggregation-mode decode \
            --dcp-size 4 \
            --speculative-algorithm EAGLE \
            --speculative-num-steps 5 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 6
        ;;
    router)
        echo "=== Starting smg PD Router (port $ROUTER_PORT) ==="
        # 把空格分隔的 DECODE_URLS 展开为多个 --decode 参数
        read -r -a decode_urls <<< "$DECODE_URLS"
        DECODE_ARGS=()
        for u in "${decode_urls[@]}"; do
            DECODE_ARGS+=(--decode "$u")
        done
        # IMPORTANT: 用 lsof 干净杀掉旧 router (监听 ROUTER_PORT)
        lsof -i :"$ROUTER_PORT" -t 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        sleep 2
        /usr/local/bin/smg launch \
            --pd-disaggregation \
            --prefill "$PREFILL_URL" \
            "${DECODE_ARGS[@]}" \
            --host 0.0.0.0 --port "$ROUTER_PORT" \
            --api-key "$API_KEY" \
            --policy cache_aware --max-concurrent-requests 64 \
            --health-check-timeout-secs 300 \
            --disable-circuit-breaker \
            --request-timeout-secs 3600 \
            --log-level info \
            --prometheus-port 29003
        ;;
    gateway)
        echo "=== Starting smg Gateway (port $GATEWAY_PORT) ==="
        lsof -i :"$GATEWAY_PORT" -t 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        sleep 2
        /usr/local/bin/smg launch \
            --host 127.0.0.1 --port "$GATEWAY_PORT" \
            --prometheus-port 29002 \
            --policy manual --assignment-mode min_load \
            --worker-urls "http://127.0.0.1:$ROUTER_PORT" \
            --api-key "$API_KEY" \
            --max-idle-secs 1800 --request-timeout-secs 86400 \
            --disable-circuit-breaker
        ;;
    compile)
        echo "=== Pre-compiling DeepGEMM kernels ==="
        python3 -m sglang.compile_deep_gemm \
            --model-path "$MODEL" --tp 8 \
            --kv-cache-dtype fp8_e4m3 \
            --moe-runner-backend triton \
            --model-impl sglang \
            --load-format dummy
        ;;
    *)
        echo "Usage: $0 {prefill|decode|router|gateway|compile}"
        echo "必需 env: PREFILL_URL=... DECODE_URLS=\"...\""
        exit 1
        ;;
esac
