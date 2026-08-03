#!/usr/bin/env bash
# ================================================================
# B300 PD Startup Script — unified prefill / decode / router / gateway / proxy
#
# Usage:
#   ./start_pd.sh prefill              # start prefill on this node
#   ./start_pd.sh decode               # start decode on this node
#   ./start_pd.sh router [--prefill IP:PORT]... [--decode IP:PORT]...
#   ./start_pd.sh gateway              # start gateway (alpha smg)
#   ./start_pd.sh proxy               # start proxy (alpha mol_harness)
#   ./start_pd.sh all                  # start everything (prefill+decode+router+gateway+proxy)
#
# Router examples:
#   ./start_pd.sh router                                          # defaults: 10.0.0.75:30100 + 10.0.0.67:30200
#   ./start_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200
#   ./start_pd.sh router --prefill 10.0.0.75:30100 --prefill 10.0.0.76:30100 --decode 10.0.0.67:30200
# ================================================================

set -euo pipefail

# ─── Shared env vars ───
export TVM_FFI_CUDA_ARCH_LIST="10.0a"
export MOONCAKE_DISABLE_HIP_DMABUF="1"
export IBV_ACCESS_RELAXED_ORDERING="1"
export MC_IB_PCI_RELAXED_ORDERING="1"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE="1"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="1000"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="600"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN="1"
export SGLANG_MOE_PADDING="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="1073741824"

# HiCache (prefill only)
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="/root/hicache"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="200G"
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE="10G"

# ─── Paths ───
PYTHON="${PYTHON:-/root/sglang_venv/bin/python3}"
MODEL_PATH="/root/glm52_local/base"
LORA_BASE="/root/glm52_local/loras"
SRV_NAME="glm52-fp8-official"
ROUTER_BIN="sglang-router"

# Gateway & Proxy (alpha)
GATEWAY_BIN="/root/Mixture-of-LoRA-Harness-alpha/sgl-model-gateway/target/release/smg"
PROXY_SRC="/root/Mixture-of-LoRA-Harness-alpha"

# ─── Defaults ───
PREFILL_HOST="${PREFILL_HOST:-10.0.0.75}"
PREFILL_PORT="${PREFILL_PORT:-30100}"
DECODE_HOST="${DECODE_HOST:-10.0.0.67}"
DECODE_PORT="${DECODE_PORT:-30200}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
GATEWAY_PORT="${GATEWAY_PORT:-31001}"
PROXY_PORT="${PROXY_PORT:-31000}"
API_KEY="sk-glm52-pd"
MOL_API_KEY="${MOL_API_KEY:-sk-mol-HUMLtbqXRTNRhDQZoUd1unJ6w6PieionO90XZHVPooQ}"

# ================================================================
# prefill — CP8 + L3 HiCache + LoRA, port 30100
# ================================================================
prefill() {
    echo "=== Starting PREFILL on ${PREFILL_HOST}:${PREFILL_PORT} ==="
    nohup ${PYTHON} -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --served-model-name ${SRV_NAME} \
        --host 0.0.0.0 --tp 8 \
        --kv-cache-dtype fp8_e4m3 --enable-cache-report \
        --page-size 64 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
        --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
        --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
        --model-impl sglang \
        --enable-lora \
        --lora-paths L0=${LORA_BASE}/L0 L1=${LORA_BASE}/L1 L2=${LORA_BASE}/L2 L3=${LORA_BASE}/L3 \
        --max-lora-rank 16 --max-loaded-loras 4 --max-loras-per-batch 4 \
        --lora-use-virtual-experts --max-lora-chunk-size 128 \
        --enable-metrics --port ${PREFILL_PORT} --mem-fraction-static 0.85 \
        --enable-hierarchical-cache --hicache-ratio 1 \
        --hicache-write-policy write_back --hicache-mem-layout page_first \
        --hicache-storage-backend file --file-storage-path /root/hicache \
        --enable-prefill-cp --cp-strategy interleave \
        --enable-dsa-prefill-cp-layersplit \
        --disable-overlap-schedule \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port 8998 \
        --disaggregation-ib-device mlx5_0 \
        --disaggregation-mode prefill \
        --dist-timeout 60 \
        > /root/prefill.log 2>&1 < /dev/null &
    echo "prefill_pid=$!"
}

# ================================================================
# decode — DCP=2 + LoRA, port 30200 (no EAGLE)
# ================================================================
decode() {
    echo "=== Starting DECODE on ${DECODE_HOST}:${DECODE_PORT} ==="
    nohup ${PYTHON} -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --served-model-name ${SRV_NAME} \
        --host 0.0.0.0 --port ${DECODE_PORT} --tp 8 \
        --kv-cache-dtype fp8_e4m3 --enable-cache-report \
        --page-size 128 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
        --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
        --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
        --model-impl sglang \
        --enable-lora \
        --lora-paths L0=${LORA_BASE}/L0 L1=${LORA_BASE}/L1 L2=${LORA_BASE}/L2 L3=${LORA_BASE}/L3 \
        --max-lora-rank 16 --max-loaded-loras 4 --max-loras-per-batch 4 \
        --lora-use-virtual-experts --max-lora-chunk-size 128 \
        --mem-fraction-static 0.90 --skip-server-warmup --enable-metrics \
        --cuda-graph-max-bs-decode 64 --max-running-requests 10 \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port 8998 \
        --disaggregation-ib-device mlx5_0 \
        --disaggregation-mode decode --dcp-size 2 \
        --dist-timeout 60 \
        --speculative-algorithm EAGLE --speculative-num-steps 3 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        > /root/decode.log 2>&1 < /dev/null &
    echo "decode_pid=$!"
}

# ================================================================
# router — supports --prefill IP:PORT --decode IP:PORT (repeatable)
# ================================================================
router() {
    local p_urls=()
    local d_urls=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prefill) p_urls+=("$2"); shift 2 ;;
            --decode)  d_urls+=("$2"); shift 2 ;;
            *) shift ;;
        esac
    done
    # Defaults
    [[ ${#p_urls[@]} -eq 0 ]] && p_urls=("${PREFILL_HOST}:${PREFILL_PORT}")
    [[ ${#d_urls[@]} -eq 0 ]] && d_urls=("${DECODE_HOST}:${DECODE_PORT}")

    local p_args=""; for u in "${p_urls[@]}"; do p_args="$p_args --prefill http://${u}"; done
    local d_args=""; for u in "${d_urls[@]}"; do d_args="$d_args --decode http://${u}"; done

    echo "=== Starting ROUTER on :${ROUTER_PORT} ==="
    echo "  prefill: ${p_urls[*]}"
    echo "  decode:  ${d_urls[*]}"

    nohup ${ROUTER_BIN} launch \
        --pd-disaggregation \
        ${p_args} ${d_args} \
        --host 0.0.0.0 --port ${ROUTER_PORT} --api-key ${API_KEY} \
        --policy cache_aware --max-concurrent-requests 64 \
        --health-check-timeout-secs 300 \
        --disable-circuit-breaker \
        --request-timeout-secs 3600 \
        --log-level info --prometheus-port 29001 \
        > /root/router.log 2>&1 < /dev/null &
    echo "router_pid=$!"
}

# ================================================================
# gateway — alpha smg, port 31001 → router :30000
# ================================================================
gateway() {
    echo "=== Starting GATEWAY on :${GATEWAY_PORT} → router :${ROUTER_PORT} ==="
    nohup ${GATEWAY_BIN} launch \
        --host 127.0.0.1 --port ${GATEWAY_PORT} --prometheus-port 29002 \
        --policy manual --assignment-mode min_load \
        --worker-urls http://127.0.0.1:${ROUTER_PORT} \
        --api-key ${API_KEY} \
        --max-idle-secs 1800 --request-timeout-secs 86400 \
        > /root/tmp/gateway.log 2>&1 < /dev/null &
    echo "gateway_pid=$!"
}

# ================================================================
# proxy — alpha mol_harness, port 31000 → gateway :31001
# ================================================================
proxy() {
    echo "=== Starting PROXY on :${PROXY_PORT} → gateway :${GATEWAY_PORT} ==="
    cd ${PROXY_SRC}
    PYTHONPATH=${PROXY_SRC} \
    UPSTREAM=http://127.0.0.1:${GATEWAY_PORT} \
    MOL_API_KEY=${MOL_API_KEY} \
    MOL_UPSTREAM_API_KEY=${API_KEY} \
    PROXY_PORT=${PROXY_PORT} \
    MOL_USE_MODEL_ROUTER=1 \
    MOL_PURE_MODEL_ROUTE=1 \
    MOL_HOP_TIMEOUT=86400 \
    MOL_SSE_KEEPALIVE_INTERVAL_S=10 \
    MOL_MAX_CONNECTIONS=8192 \
    MOL_MAX_INFLIGHT_REQUESTS=512 \
    MOL_UPSTREAM_MAX_CONNECTIONS=1024 \
    MOL_UPSTREAM_MAX_KEEPALIVE=512 \
    nohup /usr/bin/python3 -m mol_harness.proxy \
        > /root/tmp/proxy.log 2>&1 < /dev/null &
    echo "proxy_pid=$!"
}

# ================================================================
# all — prefill + decode + router + gateway + proxy
# ================================================================
all() {
    prefill
    decode
    sleep 5
    router
    sleep 3
    gateway
    sleep 2
    proxy
    echo "=== All services started ==="
}

# ================================================================
# dispatch
# ================================================================
case "${1:-}" in
    prefill) shift; prefill "$@" ;;
    decode)  shift; decode "$@" ;;
    router)  shift; router "$@" ;;
    gateway) shift; gateway "$@" ;;
    proxy)   shift; proxy "$@" ;;
    all)     shift; all "$@" ;;
    *)
        echo "Usage: $0 {prefill|decode|router|gateway|proxy|all}"
        echo ""
        echo "  prefill              Start prefill (CP8 + L3 HiCache + LoRA, port ${PREFILL_PORT})"
        echo "  decode               Start decode (DCP=1 + LoRA, port ${DECODE_PORT})"
        echo "  router [--prefill IP:PORT --decode IP:PORT]..."
        echo "                       Start router (port ${ROUTER_PORT})"
        echo "                       Defaults: prefill=${PREFILL_HOST}:${PREFILL_PORT} decode=${DECODE_HOST}:${DECODE_PORT}"
        echo "  gateway              Start gateway (alpha smg, port ${GATEWAY_PORT})"
        echo "  proxy                Start proxy (alpha mol_harness, port ${PROXY_PORT})"
        echo "  all                  Start everything"
        echo ""
        echo "Multi-PD router:"
        echo "  $0 router --prefill 10.0.0.75:30100 --prefill 10.0.0.76:30100 --decode 10.0.0.67:30200"
        exit 1
        ;;
esac
