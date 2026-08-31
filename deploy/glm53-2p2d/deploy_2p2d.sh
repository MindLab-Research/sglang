#!/usr/bin/env bash
# GLM-5.3 2P2D 部署/恢复脚本（B200 集群 38.255.28.6/.7/.8/.9）
# 拓扑: .6/.7 prefill (30100) + .8/.9 decode (30200) + .6 router (30000, 2P2D)
#       L3 = mooncake store 集群: master .6:61051 + store .6/.7/.8/.9 (TCP, 无 RDMA)
#
# 用法（节点本地执行, 需先放 /tmp/srt_2p2d.tar = repo python/sglang 打包）:
#   bash deploy_2p2d.sh storefix   # 修复本节点已有 L3 store daemon (mkdir offload + start)
#   bash deploy_2p2d.sh storecreate # 新节点创建 L3 store daemon (见下方三个坑)
#   bash deploy_2p2d.sh prefill    # 重建 prefill 引擎 (rm -f + create + tar overlay + start)
#   bash deploy_2p2d.sh decode     # 重建 decode 引擎 (含 HiCache L3 mooncake + radix)
#   bash deploy_2p2d.sh router     # .6 上重建 router (2P2D; api-key 走 ROUTER_API_KEY env)
#
# 代码 overlay: /tmp/srt_2p2d.tar 由 repo 打包:
#   cd <repo>/python/sglang && tar -cf /tmp/srt_2p2d.tar \
#     --exclude='__pycache__' --exclude='*.pyc' srt kernels/ops/attention/dsa/transform_index.py
#   scp /tmp/srt_2p2d.tar root@<node>:/tmp/   (4 节点都要)
#
# ⚠️ 历史坑（2026-08-30 排障实录, 全部踩过）:
#  1. 引擎容器 env 缺 MOONCAKE_PROTOCOL=tcp / MC_FORCE_TCP=1 / MOONCAKE_TE_META_DATA_SERVER
#     → store 复用条件走默认 rdma+P2PHANDSHAKE → InnerTransferEngine cast 崩:
#     "Unable to cast Python instance of type InnerTransferEngine to std::shared_ptr<TransferEngine>"
#  2. store daemon CMD 必须是【单字符串】(bash -lc "exec python3 ... >>log 2>&1");
#     数组形式 ["-lc","exec","python3",...] 会让 bash 只执行 "exec" → 0.16s exit 0 假启动
#  3. store 容器必须 --gpus all (否则 libcuda.so.1 ImportError);
#     镜像里【没有】daemon 脚本, 必须 volume 挂载 /root/tmp/mooncake_ssd_store_daemon.py
#  4. store daemon 启动前必须 mkdir <bind>/offload (否则 FileStorage "storage_filepath does
#     not exist" → ValueError: Invalid FileStorage configuration → 4s exit 1)
#  5. 容器重建后 tar overlay 必须重做 (docker rm 后 cp 的代码全丢); pycache 靠 .py mtime 较新自动重编译
set -euo pipefail

MODEL="/data0/models/GLM-5.3"
SERVED_MODEL="glm-5.3-fp8"
IMAGE_PREFILL="cv-prefill:nccl"      # .6/.7 本地镜像
IMAGE_DECODE="cv-decode:nccl"       # .8/.9 本地镜像
IMAGE_ROUTER="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-roce"
IMAGE_STORE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-roce"
STORE_DAEMON_SCRIPT="/root/tmp/mooncake_ssd_store_daemon.py"   # 镜像没有, volume 挂载
SRT_TAR=/tmp/srt_2p2d.tar
SGL=/sgl-workspace/sglang/python/sglang
NODE_IP=$(hostname -I | awk '{print $1}')

# 引擎容器 env —— 一个都不能少 (缺 MOONCAKE_PROTOCOL=tcp 即 cast 崩, 见坑 1)
common_env=(
  -e MOONCAKE_PROTOCOL=tcp
  -e MC_FORCE_TCP=1
  -e MOONCAKE_TE_META_DATA_SERVER=http://38.255.28.6:61080/metadata
  -e MOONCAKE_MASTER=38.255.28.6:61051
  -e MOONCAKE_MASTER_METRICS_PORT=61053
  -e MC_SLICE_SIZE=1048576
  -e MC_TCP_LANES_PER_PEER=16
  -e MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384
  -e MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=16384
  -e MC_TCP_ADMISSION_TIMEOUT_MS=10000
  -e MOONCAKE_GLOBAL_SEGMENT_SIZE=137438953472
  -e MOONCAKE_DISABLE_HIP_DMABUF=1
  -e IBV_ACCESS_RELAXED_ORDERING=1
  -e MC_IB_PCI_RELAXED_ORDERING=1
  -e TVM_FFI_CUDA_ARCH_LIST=10.0a
  -e NCCL_DEBUG=WARN
  -e SGLANG_MOE_PADDING=1
  -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
  -e SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000
  -e SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
  -e SGLANG_DISAGGREGATION_QUEUE_SIZE=64
  -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=256
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
  -e SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -e SGLANG_ENABLE_FAILED_SESSION_PROBE=1
  -e SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=10
  -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G
  -e SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G
  -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache
)

common_volumes=(
  -v /data0/models:/data0/models:ro
  -v /root/hicache:/root/hicache
  -v /root/.cache/deep_gemm:/root/.cache/deep_gemm
  -v /root/.cache/tvm-ffi:/root/.cache/tvm-ffi
)

base_args=(
  --model-path "$MODEL"
  --served-model-name "$SERVED_MODEL"
  --host 0.0.0.0
  --tp 8
  --kv-cache-dtype fp8_e4m3
  --enable-cache-report
  --page-size 64
  --chunked-prefill-size 16384
  --max-prefill-tokens 16384
  --watchdog-timeout 3600
  --reasoning-parser glm45
  --tool-call-parser glm47
  --moe-runner-backend triton
  --enforce-disable-flashinfer-allreduce-fusion
  --model-impl sglang
  --enable-metrics
  --disaggregation-transfer-backend mooncake_tcp
  --disaggregation-bootstrap-port 8998
)

overlay_code() {
  local cname="$1"
  [ -f "$SRT_TAR" ] || { echo "ERROR: $SRT_TAR missing (repo 打包后 scp 到 /tmp)"; exit 1; }
  cat "$SRT_TAR" | docker cp - "$cname":"$SGL/"
  echo "code overlay done -> $cname"
}

case "${1:-}" in
  storefix)
    # 修复本节点已有 L3 store daemon (Exited 状态: mkdir offload + start)
    for c in $(docker ps -a --format '{{.Names}}' | grep -E '^mc-l3-store'); do
      bind_dir=$(docker inspect "$c" --format '{{range .HostConfig.Binds}}{{.}} {{end}}' | tr ' ' '\n' | grep ':/exp' | head -1 | cut -d: -f1)
      if [ -n "${bind_dir:-}" ]; then mkdir -p "$bind_dir/offload"; fi
      docker start "$c" && echo "started $c (bind=$bind_dir)"
    done
    docker ps --format '{{.Names}} {{.Status}}' | grep mc-l3-store || true
    ;;

  storecreate)
    # 新节点创建 L3 store daemon (daemon 脚本先从已有节点 scp 到 $STORE_DAEMON_SCRIPT)
    node_num=$(echo "$NODE_IP" | cut -d. -f4)
    store_dir=/data0/mooncake-l3/20260828-tcp/store-$node_num
    [ -f "$STORE_DAEMON_SCRIPT" ] || { echo "ERROR: $STORE_DAEMON_SCRIPT 不存在, 先从 .9 scp: /root/tmp/mooncake_ssd_store_daemon.py"; exit 1; }
    docker rm -f "mc-l3-store-${node_num}-20260830" >/dev/null 2>&1 || true
    mkdir -p "$store_dir/offload"
    # 坑 2/3/4: 单字符串 CMD + --gpus all + offload 预建 + daemon 脚本 volume
    docker run -d --name "mc-l3-store-${node_num}-20260830" \
      --gpus all --network host --ulimit memlock=-1 \
      -v "$store_dir":/exp \
      -v "$STORE_DAEMON_SCRIPT":/opt/mooncake_ssd_store_daemon.py:ro \
      -e MOONCAKE_PROTOCOL=tcp -e MC_FORCE_TCP=1 -e MC_MS_AUTO_DISC=0 \
      -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/exp/offload \
      -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=4398046511104 \
      -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
      -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=4398046511104 \
      -e NVIDIA_VISIBLE_DEVICES=all -e IBV_ACCESS_RELAXED_ORDERING=1 \
      --entrypoint bash "$IMAGE_STORE" \
      -lc "exec python3 -u /opt/mooncake_ssd_store_daemon.py --local-hostname=$NODE_IP --metadata-server=http://38.255.28.6:61080/metadata --master-server-address=38.255.28.6:61051 --global-segment-size=137438953472 --local-buffer-size=16777216 --protocol=tcp --ssd-path=/exp/offload >>/exp/store.log 2>&1"
    sleep 6
    docker ps --filter "name=mc-l3-store" --format '{{.Names}} {{.Status}}'
    tail -3 "$store_dir/store.log" | grep -E "READY|rror" || tail -3 "$store_dir/store.log"
    ;;

  prefill)
    docker rm -f glm53-prefill >/dev/null 2>&1 || true
    mkdir -p /root/hicache
    docker create --name glm53-prefill \
      --gpus all --runtime=nvidia --network host --privileged --ipc=host \
      --ulimit memlock=-1 \
      --entrypoint python3 "${common_volumes[@]}" "${common_env[@]}" "$IMAGE_PREFILL" \
      -m sglang.launch_server "${base_args[@]}" \
      --port 30100 --mem-fraction-static 0.85 --max-total-tokens 5000000 \
      --disable-overlap-schedule \
      --enable-hierarchical-cache --hicache-ratio 1 \
      --hicache-write-policy write_back --hicache-mem-layout page_first \
      --hicache-storage-backend mooncake \
      --enable-prefill-cp --cp-strategy interleave \
      --enable-dsa-prefill-cp-layersplit --disaggregation-mode prefill
    overlay_code glm53-prefill
    docker start glm53-prefill
    echo "glm53-prefill started (port 30100)"
    ;;

  decode)
    docker rm -f glm53-decode >/dev/null 2>&1 || true
    mkdir -p /root/hicache
    docker create --name glm53-decode \
      --gpus all --runtime=nvidia --network host --privileged --ipc=host \
      --ulimit memlock=-1 \
      --entrypoint python3 "${common_volumes[@]}" "${common_env[@]}" \
      "$IMAGE_DECODE" \
      -m sglang.launch_server "${base_args[@]}" \
      --port 30200 --mem-fraction-static 0.88 \
      --skip-server-warmup --cuda-graph-max-bs-decode 64 \
      --max-running-requests 64 \
      --speculative-algorithm EAGLE --speculative-num-steps 5 \
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
      --disaggregation-mode decode --dcp-size 8 \
      --disaggregation-decode-enable-radix-cache
    overlay_code glm53-decode
    docker start glm53-decode
    echo "glm53-decode started (port 30200)"
    ;;

  router)
    docker rm -f glm53-router >/dev/null 2>&1 || true
    docker run -d --name glm53-router \
      --network host --privileged --ipc=host --ulimit memlock=-1 --entrypoint python3 \
      -v /data0/models:/data0/models:ro "$IMAGE_ROUTER" \
      -m sglang_router.launch_router \
      --pd-disaggregation \
      --prefill http://38.255.28.6:30100 8998 \
      --prefill http://38.255.28.7:30100 8998 \
      --decode http://38.255.28.8:30200 \
      --decode http://38.255.28.9:30200 \
      --host 0.0.0.0 --port 30000 \
      --api-key "${ROUTER_API_KEY:?set ROUTER_API_KEY (真值见仓库根 secrets.env)}" \
      --policy cache_aware --max-concurrent-requests 64 \
      --health-check-timeout-secs 300 \
      --request-timeout-secs 600 \
      --log-level info
    echo "glm53-router started (port 30000, 2P2D)"
    ;;

  *)
    echo "usage: $0 {prefill|decode|router|storefix|storecreate}" >&2
    exit 2
    ;;
esac
