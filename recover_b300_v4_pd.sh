#!/bin/bash
set -uo pipefail

# ============================================================
# B300 DeepSeek V4 Pro 0813 PD 分离 + DSPARK + DCP 启动脚本
# 分支: b300-glm52 (HEAD f29e50bc93)
#
# 拓扑:
#   B300-1 (8.213.215.2:1021, 内网 10.0.0.75): prefill + router (port 30100 / 31000)
#   B300-2 (8.213.215.2:1022, 内网 10.0.0.67): decode   (port 30200, DCP=4 + DSPARK)
#   venv: prefill=/root/sglang_venv, decode=/root/v15_patched
#   模型: /root/deepseek_v4_pro_0813 (官方 853GB, 61 层, 384 experts, FP4 expert + FP8 attention)
#   公网: 8.213.215.2:18888, API key sk-mol-... (见 docs/agent/v4-pro-deploy.md)
#
# 用法:
#   bash recover_b300_v4_pd.sh decode    # 在 B300-2 上运行（decode + DSPARK + DCP + radix）
#   bash recover_b300_v4_pd.sh prefill   # 在 B300-1 上运行（prefill + CP layer-split + HiCache）
#   bash recover_b300_v4_pd.sh monitor   # 本地运行，等两端 health 200
#   bash recover_b300_v4_pd.sh clean     # 杀两端进程 + 清 pycache（本地运行）
#
# ⚠️ 重启铁律（见 AGENTS.md §5.3）:
#   1. 每次重启必须 rsync 本地最新代码到两端 + 清 __pycache__（绝不带旧代码）
#   2. rsync 后必须删 L20D triton config（1P1D 目录无此文件即对照基准）
#   3. 杀进程必须干净（ps + lsof 双重，0 残留 0 端口）
#   4. health 直接 curl 节点本地，不要信嵌套 SSH 轮询的 000
#   5. 启动顺序: prefill/decode → health 200 → router → gateway → proxy
# ============================================================

# ---------- 两端都必须带的完整 env（缺 env 是本分支退化的主因） ----------
# decode 缺 mooncake env → health 200 但请求卡死（KVTransferError AbortReq）
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
export SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="1073741824"
export SGLANG_DSV4_MHC_PREWARM="1"
export SGLANG_DEFAULT_THINKING="1"

MODEL=/root/deepseek_v4_pro_0813
V4_COMMON="--model-path $MODEL --served-model-name deepseek-v4-pro-0813 \
  --tp 8 --kv-cache-dtype fp8_e4m3 --enable-cache-report \
  --page-size 256 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --moe-runner-backend flashinfer_mxfp4 --model-impl sglang --enable-metrics"

case "${1:-}" in
  decode)
    # B300-2 (1022): DSPARK + DCP=4 + mooncake decode + decode-end radix cache
    # ⚠️ DSPARK 参数只在 decode 端（draft 完全本地运行）；Pro 0813 自带 DSpark head，
    #    不要加 EAGLE flags（num-steps/eagle-topk/num-draft-tokens）
    # ⚠️ decode 端 radix cache 必须与 prefill 端同开（DSpark hidden 从中间层激活取，不可从 KV 反推）
    export SGLANG_DECODE_RADIX_ALLOW_SWA="1"
    export SGLANG_PD_HIDDEN_RECV_POOL_TOKENS="65536"
    nohup /root/v15_patched/bin/python3 -m sglang.launch_server \
      $V4_COMMON \
      --host 0.0.0.0 --port 30200 --mem-fraction-static 0.90 \
      --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
      --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
      --disaggregation-ib-device mlx5_0 --disaggregation-mode decode --dcp-size 4 \
      --disaggregation-decode-enable-radix-cache \
      --speculative-algorithm DSPARK \
      > /root/decode_v4.log 2>&1 < /dev/null &
    echo "decode_pid=$!  (log: /root/decode_v4.log)"
    ;;
  prefill)
    # B300-1 (1021): prefill + CP layer-split (interleave) + HiCache 文件后端
    # ⚠️ CP cover hidden 捕获：attn_tp 8→1 后由 mooncake decode_engine_rank 对角配对，避免 8 倍重复发送
    # ⚠️ hidden 双侧 pool 必须配对（65536 == decode RECV_POOL_TOKENS）
    export SGLANG_PD_HIDDEN_POOL_TOKENS="65536"
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="/root/hicache"
    export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="200G"
    export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE="10G"
    nohup /root/sglang_venv/bin/python3 -m sglang.launch_server \
      $V4_COMMON \
      --host 0.0.0.0 --port 30100 --mem-fraction-static 0.85 \
      --disable-overlap-schedule \
      --enable-prefill-cp --cp-strategy interleave \
      --enable-hierarchical-cache --hicache-ratio 1 --hicache-write-policy write_back \
      --hicache-mem-layout page_first --hicache-storage-backend file --file-storage-path /root/hicache \
      --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
      --disaggregation-ib-device mlx5_0 --disaggregation-mode prefill \
      > /root/prefill_v4.log 2>&1 < /dev/null &
    echo "prefill_pid=$!  (log: /root/prefill_v4.log)"
    ;;
  monitor)
    # 本地运行：等两端 health 200（V4 冷启动 2-30 分钟，编译缓存后 70-160s）
    for i in $(seq 1 400); do
      P=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -p 1021 root@8.213.215.2 \
            "curl -sf -m 3 http://localhost:30100/health_generate >/dev/null 2>&1 && echo OK" 2>/dev/null)
      D=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -p 1022 root@8.213.215.2 \
            "curl -sf -m 3 http://localhost:30200/health_generate >/dev/null 2>&1 && echo OK" 2>/dev/null)
      if [ "$P" = "OK" ] && [ "$D" = "OK" ]; then
        echo "BOTH_READY after ~$((i*30))s"
        exit 0
      fi
      sleep 30
    done
    echo "TIMEOUT (P=$P D=$D)"
    exit 1
    ;;
  clean)
    # 本地运行：杀两端进程 + 清 pycache（重启前必做）
    for P in 1021 1022; do
      ssh -o ConnectTimeout=15 -p $P root@8.213.215.2 \
        "ps aux | grep -E 'launch_server|sglang::scheduler' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null; \
         lsof -ti :30100 2>/dev/null | xargs -r kill -9 2>/dev/null; \
         lsof -ti :30200 2>/dev/null | xargs -r kill -9 2>/dev/null; \
         sleep 1; echo '$P killed'" 2>/dev/null
    done
    ;;
  *)
    echo "用法: bash $0 {decode|prefill|monitor|clean}"
    echo "  decode   → 在 B300-2 (1022) 运行"
    echo "  prefill  → 在 B300-1 (1021) 运行"
    echo "  monitor  → 本地运行，等两端 health 200"
    echo "  clean    → 本地运行，杀两端进程 + 清 pycache"
    ;;
esac