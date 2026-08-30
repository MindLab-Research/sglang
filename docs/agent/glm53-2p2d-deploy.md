# GLM-5.3 2P2D 部署与快速恢复手册（38.255.28.6/.7/.8/.9 + 1102/1104）

> 2026-08-30 全集群重部署实录沉淀。覆盖：2P2D 引擎（mooncake TCP + HiCache L3 store + decode radix）、
> L3 store daemon 集群、router、代码 overlay、1102/1104 裸机对、全部踩坑与判据。
> 部署脚本：`deploy/glm53-2p2d/deploy_2p2d.sh`（api-key 走 `ROUTER_API_KEY` env，真值见仓库根 secrets.env）。

## 1. 拓扑

### 1.1 B200 集群（38.255.28.x，容器化）

| 节点 | 角色 | 容器 | 端口 | 镜像 |
|---|---|---|---|---|
| .6 | prefill1 + router + L3 master + L3 store-6 | `glm53-prefill` / `glm53-router` / `mc-l3-master-20260828` / `mc-l3-store-20260828` | 30100 / 30000 / 61051+61080 | `cv-prefill:nccl` / roce |
| .7 | prefill2 + L3 store-7 | `glm53-prefill` / `mc-l3-store-7-20260828` | 30100 | `cv-prefill:nccl` / roce |
| .8 | decode1 + L3 store-8 | `glm53-decode` / `mc-l3-store-8-20260830` | 30200 | `cv-decode:nccl` / roce |
| .9 | decode2 + L3 store-9 | `glm53-decode` / `mc-l3-store-9-20260828` | 30200 | `cv-decode:nccl` / roce |

- 模型：`/data0/models/GLM-5.3`（served name `glm-5.3-fp8`），GLM-5.3-FP8（45 层 = 34 linear attention + 11 DSA，MoE 288 experts topk 8，KV ≈5.6KB/token）
- L3：mooncake store 集群 = master(.6:61051/61080) + 4×store（137GB 段/节点，SSD offload 4TB bucket），**协议全 TCP**（无 RDMA）
- decode：DCP=8 + EAGLE(5 steps, topk1, draft6) + radix + **HiCache L3 mooncake**（mem-fraction 0.88）
- prefill：CP=8 layersplit + HiCache L3 mooncake（write_back page_first，mem-fraction 0.85，max-total-tokens 5M）
- 节点代码路径（容器内）：`/sgl-workspace/sglang/python/sglang/`（tar overlay 目标）

### 1.2 B300 裸机对（1102/1104，8.222.11.182 SSH 端口 = 节点号）

| 节点 | 角色 | 启动脚本 | 日志 |
|---|---|---|---|
| 1102 (10.0.58.35) | prefill (30100) + smg router (31000, key `sk-…pd`（真值见节点 `ps aux` 的 smg 命令或 secrets.env）) | `/root/start_glm53_prefill.sh` | `/root/glm53_prefill.log` |
| 1104 (10.0.58.37) | decode (30200, DCP=4 + EAGLE + HiCache file) | `/root/start_glm53_decode.sh` | `/root/glm53_decode.log` |

- venv：`/root/v15_patched`（代码路径 `lib/python3.12/site-packages/sglang/srt/`），模型 `/root/glm52_local/glm53`
- L3 用 **file backend**（`/root/hicache`，非 mooncake）；transfer backend **mooncake RDMA**（mlx5_0）
- smg：`smg launch --pd-disaggregation --prefill http://10.0.58.35:30100 --decode http://10.0.58.37:30200 --port 31000 --api-key <SMG_KEY 见 secrets.env> --policy round_robin`

## 2. 快速恢复 Runbook（B200 2P2D）

### 2.0 前置：代码 tar 打包（本地 repo，每次部署都要做——容器重建后 overlay 全丢）

```bash
cd <repo>/python/sglang
tar -cf /tmp/srt_2p2d.tar --exclude='__pycache__' --exclude='*.pyc' \
  srt kernels/ops/attention/dsa/transform_index.py
for h in 38.255.28.6 38.255.28.7 38.255.28.8 38.255.28.9; do
  scp /tmp/srt_2p2d.tar root@$h:/tmp/; scp deploy/glm53-2p2d/deploy_2p2d.sh root@$h:/tmp/
done
```

### 2.1 全集群重建（从零到公网可用，~10 分钟）

```bash
# ① L3 master（.6，常驻勿动）+ stores（store daemon 挂了就修）
ssh root@38.255.28.6 'bash /tmp/deploy_2p2d.sh storefix'   # .6
ssh root@38.255.28.7 'bash /tmp/deploy_2p2d.sh storefix'   # .7
ssh root@38.255.28.9 'bash /tmp/deploy_2p2d.sh storefix'   # .9（store 容器 Exited 时）
# .8 新节点建 store（已建过则 storefix）：
#   先 scp store daemon 脚本: ssh .9 'cat /root/tmp/mooncake_ssd_store_daemon.py' | ssh .8 'cat > /root/tmp/mooncake_ssd_store_daemon.py'
#   ssh root@38.255.28.8 'bash /tmp/deploy_2p2d.sh storecreate'

# ② 引擎（每节点一条；脚本内含 tar overlay + 完整 mooncake env）
ssh root@38.255.28.6 'bash /tmp/deploy_2p2d.sh prefill'
ssh root@38.255.28.7 'bash /tmp/deploy_2p2d.sh prefill'
ssh root@38.255.28.8 'bash /tmp/deploy_2p2d.sh decode'
ssh root@38.255.28.9 'bash /tmp/deploy_2p2d.sh decode'

# ③ router（等 4 引擎 health 200 后）
ssh root@38.255.28.6 'ROUTER_API_KEY=<见 secrets.env> bash /tmp/deploy_2p2d.sh router'

# ④ 验证
curl http://38.255.28.6:30000/v1/chat/completions -H "Authorization: Bearer $ROUTER_API_KEY" \
  -d '{"model":"glm-5.3-fp8","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":64}'
```

### 2.2 单节点故障恢复

| 症状 | 动作 |
|---|---|
| 引擎容器 Exited/崩溃 | `bash /tmp/deploy_2p2d.sh {prefill\|decode}`（rm -f + create + overlay + start） |
| store daemon Exited(1) | `bash /tmp/deploy_2p2d.sh storefix`（mkdir offload + start） |
| 新节点加 store | `bash /tmp/deploy_2p2d.sh storecreate`（先 scp daemon 脚本过去） |
| prefill 重启后 decode 卡 | **配对重启 decode**（mooncake 会话失效全报 `Aborted by AbortReq`，AGENTS §swa 文档） |
| L3 行为异常 | 先查 store daemon `tail /data0/mooncake-l3/20260828-tcp/store-N/store.log`（grep READY/error） |

### 2.3 验证判据

- 引擎：`curl -s localhost:{30100,30200}/health` = 200；日志 grep `Unable to cast|Traceback` = 0
- store：`store.log` 末尾 `MOONCAKE_SSD_STORE_READY=1`；master 日志（.6 `docker exec mc-l3-master-20260828 tail /exp/master.log`）有 `mount_segment, segment_name=<ip>:<port>` 且无 E 行
- decode 侧确认 TCP：`disaggregation transfer backend 'mooncake_tcp' -> mooncake with MC_FORCE_TCP=1 (TCP transport, no RDMA)`
- router：`/health` 200 + 冒烟 chat 输出正常（1+1=2）
- HiCache 生效：prefill 日志 `Allocating xx GB host memory for hierarchical KV cache`

## 3. 关键坑（2026-08-30 实录，全部踩过）

### 3.1 ⛔ 引擎 cast 崩 = mooncake env 不全（最高频，一次坑 4 节点）

```
RuntimeError: Unable to cast Python instance of type <class 'mooncake.engine.InnerTransferEngine'>
  to C++ type 'std::shared_ptr<mooncake::TransferEngine>'
```

**根因**：引擎容器 env 缺 `MOONCAKE_PROTOCOL=tcp`（默认 rdma）+ `MOONCAKE_TE_META_DATA_SERVER`（默认 P2PHANDSHAKE）→
`mooncake_store.py::setup()` 的 store 复用条件（device 匹配 + P2PHANDSHAKE + rdma）全真 → 把 PD transfer 的
`InnerTransferEngine` 传给 C++ `Store.setup` → pybind cast 崩。**重建容器 env 必须照抄 `deploy_2p2d.sh` 的
`common_env` 全集**（22 项），一个都不能少。

### 3.2 ⛔ store daemon CMD 必须单字符串

`docker run ... <img> -lc "exec python3 -u ... >>/exp/store.log 2>&1"`（**一个字符串参数**）。
数组形式 `["-lc","exec","python3",...]` 会让 bash 只执行 `-c "exec"` → **0.16s exit 0 假启动**（docker logs 空）。
症状：容器 Exited(0) 且无日志。

### 3.3 ⛔ store 容器必须 --gpus all

镜像有 CUDA 依赖链（libcuda.so.1），`NVIDIA_VISIBLE_DEVICES=none` + 无 `--gpus` → `ImportError: libcuda.so.1`
（store daemon import mooncake.store 时）。store-9 成功运行的 inspect：`DeviceRequests: [gpu]`。

### 3.4 ⛔ store daemon 启动前必须 mkdir offload

bind 目录 `/data0/mooncake-l3/20260828-tcp/store-N/` 挂到容器 `/exp`，**offload 子目录不会自动创建**：
`FileStorageConfig: storage_filepath does not exist: /exp/offload` → `ValueError: Invalid FileStorage configuration`
→ 4s exit(1)。`storefix` 分支已内置 `mkdir -p <bind>/offload`。三个 store 曾在 2026-08-29 05:00 集体死于此。

### 3.5 ⛔ 镜像里没有 store daemon 脚本

`/opt/mooncake_ssd_store_daemon.py` 是 volume 挂载（`/root/tmp/mooncake_ssd_store_daemon.py:ro`），
新节点建 store 前必须先从已有节点 scp。容器重建不会丢（volume），但新节点会。

### 3.6 ⛔ hasattr 3 参数 TypeError = HiCache L2 write-back 驱逐崩溃（1102 实录）

`memory_pool_host.py::backup_from_device_all_layer` 的 `_host_cap` 行曾写
`hasattr(self, "index_k_with_scale_buffer", None)`（Python hasattr 只接受 2 参数）→ L2 池满驱逐时
TypeError → scheduler 崩 → SIGQUIT。**修复 commit `0b589a9b19`**（`getattr(...) is not None`）。
22d8d3763b 的 staging gate 修复漏了这行（同款笔误两处只修一处）。**已部署 6 节点**（B200 tar overlay + 1102/1104 tar 覆盖）。
判据：`grep "hasattr expected 2" <prefill log>` 有输出 = 旧代码还在。

### 3.7 L3 store 在线扩容（.8 实录）

新 store（storecreate）注册 master 即生效（`mount_segment` 幂等，`segment_already_exists` warn 无害），
引擎**无需重启**——master 全局调度 slice，旧连接不动。8/29 旧 store 残留的 `E...SEGMENT_NOT_FOUND`
heartbeat 报错在新 store 接管后停止。

## 4. 1102/1104（B300 裸机）快速恢复

```bash
# 代码（tar 同 2.0 的 /tmp/srt_2p2d.tar）:
for p in 1102 1104; do scp -P $p /tmp/srt_2p2d.tar root@8.222.11.182:/tmp/srt_fix.tar; done
for p in 1102 1104; do ssh -p $p root@8.222.11.182 '
  SGL=/root/v15_patched/lib/python3.12/site-packages/sglang
  tar -xf /tmp/srt_fix.tar -C $SGL/ && find $SGL -name __pycache__ -exec rm -rf {} + 2>/dev/null
  rm -rf /root/.cache/tvm-ffi/sgl_kernel_jit_hicache_*'; done
# 先杀 decode（配对重启铁律）→ prefill → health → decode → health:
ssh -p 1104 root@8.222.11.182 'pkill -9 -f sglang.launch_server; pkill -9 -f sglang::scheduler'
ssh -p 1102 root@8.222.11.182 'cd /root && nohup bash /root/start_glm53_prefill.sh > /root/glm53_prefill.log 2>&1 < /dev/null &'
# 等 1102:30100 health 200 后：
ssh -p 1104 root@8.222.11.182 'cd /root && nohup bash /root/start_glm53_decode.sh > /root/glm53_decode.log 2>&1 < /dev/null &'
# smg router 常驻不动（1102:31000），冒烟：
curl http://8.222.11.182:31000/v1/chat/completions -H "Authorization: Bearer <SMG_KEY 见 secrets.env>" ...
```

1102/1104 启动脚本自带完整 env（`/root/start_glm53_{prefill,decode}.sh`，mooncake RDMA + mlx5_0，无 TCP 那套）。

## 5. 与 2P1D（旧档 `docs/agent/glm53-pd-deploy.md`）的差异

- .8 加入（decode1 + store-8），2P1D→2P2D；decode 新增 `--enable-hierarchical-cache --hicache-storage-backend mooncake`（L3）+ mem-fraction 0.90→0.88
- 引擎重建改 `docker create + tar overlay + docker start` 三段式（overlay 后再 start，无需二次 restart）
- L3 store 从 .6/.7/.9 扩到 4 节点；store 全部 `storefix`/`storecreate` 脚本化管理
- 代码基线 `0b589a9b19`（hasattr 修复）；`22d8d3763b` 及之前的 Xid31/staging gate 修复链全包含
