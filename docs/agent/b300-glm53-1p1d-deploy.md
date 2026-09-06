# B300 spot 三机 GLM-5.3 1P1D 部署（2026-09-05，b300-2/3）

> 背景：1102/1104 B300 集群回收，spot 机器 b300-1/2/3 替代。b300-1（原 prefill+router）宕机中，
> 先用 b300-2/3 组 1P1D。代码 = `fix/eagle-seed-contract` @ `f79bee0347`（含 5085112225
> DSA seed topk 修复 + c23f23fa03 host_lock_ref 修复）。

## 拓扑

| 机 | 公网 | 内网 | 角色 | 容器 | 端口 |
|---|---|---|---|---|---|
| b300-3 | 3.37.20.114 | 172.31.47.105 | **prefill** + router(内部) | glm53-prefill, glm53-router | 30100 / bootstrap 30011 / router 31000 |
| b300-2 | 15.164.0.39 | 172.31.45.101 | **decode** | sglang-decode | 30002 / bootstrap 30011 |
| b300-1 | 13.124.32.173 | 172.31.33.86(旧) | 宕机中（原 router 公网入口 :30000 曾开放） | — | — |

- SSH：`ssh -i ~/.ssh/b300-spot.pem ubuntu@<ip>`（无 root，sudo 可用；`/nvme` → `/opt/dlami/nvme` symlink）
- 镜像：`b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-pd-fix`（含 sglang-router/smg/引擎一体）
- 模型：`/nvme/models/GLM-5.3`（704G，两台都有；GlmMoeDsaForCausalLM，78 层）
- 代码 overlay：`/opt/dlami/nvme/sglang-overlay-v15/sglang`（rsync 本地 `python/sglang/`，:ro 挂载到容器 `/sgl-workspace/sglang/python/sglang`）
- mooncake engine patch：`/tmp/engine-patched.so` :ro 挂载到 `/usr/local/lib/python3.12/dist-packages/mooncake/engine.so`
- 无 IB（AWS EFA）→ `--disaggregation-transfer-backend mooncake_tcp`，bootstrap 30011，L3=file HiCache

## 代码同步（先于一切）

```bash
# 本地 repo @ fix/eagle-seed-contract（f79bee0347），两台都做：
rsync -az --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  -e "ssh -i ~/.ssh/b300-spot.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  python/sglang/ ubuntu@<host>:/opt/dlami/nvme/sglang-overlay-v15/sglang/
# 验证（md5 必须与本地一致）：
md5sum python/sglang/srt/speculative/eagle_worker_v2.py   # f6d7c0a9ba407c7ed8e4809ef689903d
ssh <host> md5sum /opt/dlami/nvme/sglang-overlay-v15/sglang/srt/speculative/eagle_worker_v2.py
```

## prefill（b300-3，glm53-prefill :30100）

docker create 参数（完整，宿主路径经 /nvme symlink）：

```
docker create --name glm53-prefill --gpus all --runtime=nvidia --network host --privileged --ipc=host \
  --entrypoint python3 \
  -v /nvme/models:/nvme/models -v /nvme/hicache:/root/hicache \
  -v /nvme/sglang-cache/deep_gemm:/root/.cache/deep_gemm \
  -v /nvme/sglang-cache/tvm-ffi:/root/.cache/tvm-ffi \
  -v /nvme/sglang-overlay-v15/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /tmp/engine-patched.so:/usr/local/lib/python3.12/dist-packages/mooncake/engine.so:ro \
  -e TVM_FFI_CUDA_ARCH_LIST=10.0a -e NCCL_DEBUG=WARN \
  -e MOONCAKE_DISABLE_HIP_DMABUF=1 -e IBV_ACCESS_RELAXED_ORDERING=1 -e MC_IB_PCI_RELAXED_ORDERING=1 \
  -e MC_TCP_CONNECTION_IDLE_TIMEOUT=10 -e MC_TCP_ENABLE_CONNECTION_POOL=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e SGLANG_MOE_PADDING=1 \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  -e SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=1 -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=1000 -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
  -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 -e SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 \
  -e SGLANG_DISAGGREGATION_QUEUE_SIZE=64 -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=256 \
  -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/hicache -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G \
  -e SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=10G \
  b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-pd-fix \
  -m sglang.launch_server \
  --model-path /nvme/models/GLM-5.3 --served-model-name glm53 --host 0.0.0.0 --port 30100 --tp 8 \
  --kv-cache-dtype fp8_e4m3 --enable-cache-report --page-size 64 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion --model-impl sglang \
  --enable-metrics --trust-remote-code --mem-fraction-static 0.85 --max-total-tokens 5000000 \
  --disable-overlap-schedule \
  --enable-hierarchical-cache --hicache-ratio 1 --hicache-write-policy write_back \
  --hicache-mem-layout page_first --hicache-storage-backend file --file-storage-path /root/hicache \
  --enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp --disaggregation-bootstrap-port 30011
```

参数来源：2p2d deploy 脚本 prefill 分支（deploy/glm53-2p2d/deploy_2p2d.sh）改 file L3 +
1102/1104 测试集群参数；启动 ~5min 到 health 200。

## decode（b300-2，sglang-decode :30002）

与团队 22h 前部署的容器**参数完全一致**（= 1104 等价 + mooncake_tcp），只换了 overlay 代码。
关键参数：`--dcp-size 4 --speculative-num-steps 5 --speculative-eagle-topk 1
--speculative-num-draft-tokens 6 --disaggregation-decode-enable-radix-cache
--cuda-graph-max-bs-decode 64 --max-running-requests 48 --mem-fraction-static 0.90
--skip-server-warmup` + HiCache file L3 + `SGLANG_DECODE_RADIX_ALLOW_SWA=1` +
`SGLANG_METADATA_BUFFER_FREE_DELAY=120`（这两个 env 是 decode 独有）。
GLM-5.3 用 checkpoint 内置 MTP，**无 --speculative-draft-model-path**。启动 ~5min。

## router（b300-3，glm53-router :31000 内部；公网被 SG 拦）

```
docker run -d --name glm53-router --network host --privileged --ipc=host --entrypoint python3 \
  -v /nvme/models:/nvme/models:ro \
  b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-pd-fix \
  -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://172.31.47.105:30100 30011 --decode http://172.31.45.101:30002 \
  --host 0.0.0.0 --port 31000 --api-key <见仓库根 secrets.env MOL_API_KEY_B300_1P1D> \
  --policy cache_aware --max-concurrent-requests 64 \
  --health-check-timeout-secs 300 --request-timeout-secs 600 --log-level info
```

- 实际是 smg 内核（`-m sglang_router.launch_router` 调起 smg::server）。**必须挂 /nvme/models:ro**
  （否则 tokenizer 注册 404 刷屏，smg job 反复失败）。
- `--api-key` 真值在本机 `secrets.env` 的 `MOL_API_KEY_B300_1P1D`。
- b300-1 公网入口方案（待机器回来）：同款命令改 `--port 30000`（b300-1 SG 已开放 30000），
  prefill/decode URL 不变（内网互联）。

## 验证结果（2026-09-05 05:2x UTC）

- prefill health 200（5m07s），decode health 200（4m40s），router /health 200
- E2E（b300-3 本机 curl router 31000）：chat 200，25s 首请求（JIT 预热）/ 1.8s 二次；
  输出质量正常（reasoning 连贯、`1+1等于2。`），无乱码
- **accept len 3.35/3.62，accept rate 0.47/0.53**（健康；1104 修复后基准 0.20-0.38，
  新集群更高）——DSA seed 修复在新集群生效
- KV transfer/bootstrap 正常（decode 侧 68 条 bootstrap/transfer 活动）
- 无 DSA-SEED fallback 告警、无 KVTransferError、无 abort

## 坑（本次实录）

1. **旧 decode 容器 zombie**：prefill 死 16h+ 后 decode init 进程 `Z <defunct>`，`docker stop`
   报 "cannot stop container: PID is zombie"。处理：按 `nvidia-smi --query-compute-apps=pid`
   精确 `kill -9` GPU 进程 → `docker rm -f` → 按原参数 docker create 重建。
2. **`pkill -9 -f sglang` 会自杀 SSH 会话**（远端命令行本身含 "sglang" 字样被匹配）→
   永远用 nvidia-smi PID 列表精确 kill。
3. **本地 curl 走 HTTP 代理劫持 localhost/公网 → 假 502/假不通**：必须 `--noproxy '*'`。
4. **b300-2/3 公网 SG 只开 22**：31000/30100/30002 全被拦（bench 需 b300-1:30000 或开 SG
   或 SSH 隧道）。
5. smg PD 模式 `/v1/completions` 卡死是已知 bug，用 `/v1/chat/completions`。
6. spot 机器重启公网 IP 会变（b300-1 旧入口 http://13.124.32.173:30000/v1 失效，
   内网 172.31.33.86 也不通 = 实例未起或换 IP，需 AWS 控制台确认）。

## 待办

- [ ] b300-1 恢复后：清掉旧容器（原 prefill/router），起 router :30000（公网入口，命令见上），
      bench 打 http://<b300-1新IP>:30000/v1（model=glm53，key 同上）
- [ ] 3h 压测（cases_50，盯 accept rate <0.1 = bug 复现红线）
