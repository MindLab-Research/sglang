# GLM-5.3 2P1D PD 部署（38.255.28.x）复盘

## 部署拓扑

| 角色 | 节点 | 关键参数 |
|---|---|---|
| prefill | .6 / .7 | `--enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit --enable-hierarchical-cache --hicache-storage-backend mooncake`，TP=8 |
| decode | .9 | `--dcp-size 8 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --disaggregation-decode-enable-radix-cache`，TP=8 |
| router | .6 | `sglang_router.launch_router --pd-disaggregation --policy cache_aware` |
| L3 store | .6 master + .6/.7/.9 store | mooncake TCP（**无 RDMA**，`MOONCAKE_PROTOCOL=tcp` `MC_FORCE_TCP=1`），137GB 段，SSD `/data0/mooncake-l3/20260828-tcp` |

模型：GLM-5.3-FP8（`/data0/models/GLM-5.3`），tokenizer 从本地路径成功加载（vocab 154820）。

## 关键坑

### 1. 镜像缺 `common/conn.py:_send_multipart`（本次主坑）

- **现象**：decode 全 rank `KVTransferError: Aborted by AbortReq`，请求卡 55s+。
- **根因**：`_send_multipart` 是 2026-08-24 mooncake torn-frame 修复引入的本分支方法（`CommonKVManager`，定义在 `common/conn.py:837`）。但推理镜像 `cv-prefill:nccl` / `cv-decode:nccl` **未打包该修复**（`MooncakeKVManager` 继承 `CommonKVManager`，运行时 `MooncakeKVManager object has no attribute '_send_multipart'` → Transfer thread 崩溃 → AbortReq）。
- **排查确认**：`docker run --rm cv-prefill:nccl python3 -c "..."` → `common has _send_multipart: False`。repo 源码有（`grep -c 'def _send_multipart' common/conn.py` = 1），容器内缺。
- **修复**：`docker cp <repo>/disaggregation/common/conn.py <容器>:<srt>/disaggregation/common/conn.py` + 清 `__pycache__` + `docker restart`。三节点都需 cp（prefill/decode 共用）。
- **教训**：推理镜像必须与 repo 同步；`docker cp` 是代码进运行容器的最可靠方式（rsync 到容器常因 bind-mount / 时机问题静默失败）。

### 2. 容器重建后 `docker cp` 丢代码

- `docker rm` + recreate 后的容器从纯镜像启动，之前 docker cp 的代码**全部丢失**，必须重 cp。`docker restart` 则保留 docker-cp 文件。
- 镜像 ENTRYPOINT 是 `sleep`，`docker run` 的 cmd 参数会被 `sleep` 吞掉——重建必须 `--entrypoint python3`。

### 3. GPU 内存 / zombie 容器

- `.6/.7` 多次出现 `Exited (137)`：`docker restart` 打断 DeepGEMM warmup → 残留请求阻塞 graceful exit → SIGKILL。**避免在 warmup 期间 restart**；先实现逻辑要用 `rm -f` + 重建。

### 4. mooncake 清理（用户要求）

- **无原生 flush API**：master HTTP `/flush /evict /clear` 全 404，TransferEngine Python API 无 flush/evict/clear，store daemon 源码无清理命令。
- 等效清理（防 GLM-5.2 残留 KV 导致乱码）：清 `/root/hicache`(L2) + 重启 L3 store/master daemon（清 137GB 内存段）。磁盘 offload 实际为空（20K）。

## 验证

- 三端 HEALTH-200，早期失败计数 0，重启计数 0。
- 请求 `1+1=?` → `**1 + 1 = 2**` finish=stop，无乱码（thinking 模式下 content 在 reasoning 消费后仍能正常输出，需足够 max_tokens）。
