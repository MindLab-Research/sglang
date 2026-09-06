# GLM-5.3 b300 PD 集群 — 完整还原手册

> 本目录是 b300 GLM-5.3 PD 集群的**唯一真相来源**（2026-09-06 建立）。
> 从此只从这个 image + 本目录脚本还原，不再记忆碎片、不再考古容器。

---

## 1. 最终正确 image

```
b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final
digest: sha256:68616ef1af65a765a6575a7a22fbe7525de2caa597112a0d7251221efcf87409
```

**100% 还原当前运行状态，已验证：**
- `engine.so` = `cf881774d4b4d794c29ed67341dc9499`（正确版，base image 自带，**不覆盖**）
- `conn.py` = `e79f346349b8ddd31e79ea2f0045801a`（HEAD，COPY 覆盖了 base 旧代码）
- `memory_pool.py` 含 DCP 修复（`_slot = (loc // (_ps * _dws)) * _ps + (loc % _ps)`）

**历史教训（为什么这个 tag 才是对的）：**
- base image `v0.5.15.post1-cuda13-b200` **自带正确的 engine.so = cf881774**。
- 错误操作：重建容器时挂载了 `/nvme/mooncake-b200/engine.so`（=8ea08f1e，b200 三件套）覆盖正确版 → `batch_transfer_sync` 返回非0 → 传输失败 500。
- 昨晚正确运行容器（containerd OCI spec 实证）挂的就是自包含 `engine-patched.so`（cf881774），且不挂 libasio/liburing。
- **结论：engine-so 用 base image 自带的正确版即可，绝不挂载任何外部 engine.so。**

---

## 2. 部署拓扑

| 端 | 节点 | 内网 IP | 端口 |
|---|---|---|---|
| decode | B300 spot decode | 172.31.45.101 | 30002 |
| prefill | B300 spot prefill | 172.31.47.105 | 30100 |
| router | prefill 节点 | 172.31.47.105 | 31000 |

- bootstrap 端口：decode=30011, prefill=30012
- `/nvme/models/GLM-5.3` 为模型目录（节点本地显存路径）

---

## 3. 环境文件（已从运行容器完整 dump，含全部 mooncake env）

- `decode.env` — decode 端 env（28 项，含 `MOONCAKE_DISABLE_HIP_DMABUF`/`MC_TCP_*`/`SGLANG_DISAGGREGATION_*`）
- `prefill.env` — prefill 端 env（33 项，含 `MC_FORCE_TCP=1`/`MC_TCP_LANES_PER_PEER=16`/`MC_TCP_MAX_*`）
- `router.env` — router 端 env（19 项）

> decode/prefill 的 mooncake env 有各自专属项（如 prefill 有 `MC_FORCE_TCP`/`MC_TCP_LANES_PER_PEER`，decode 有 `SGLANG_DECODE_RADIX_ALLOW_SWA`），这是**正常设计**，不要求两端完全一致。

---

## 4. 启动脚本

| 脚本 | 跑在 | 说明 |
|---|---|---|
| `start_decode.sh` | decode 节点 | 启动 decode（DCP=4 + EAGLE） |
| `start_prefill.sh` | prefill 节点 | 启动 prefill（CP=8 layersplit + HiCache） |
| `start_router.sh` | prefill 节点 | 启动 router（PD 模式，cache_aware） |

**⛔ 启动铁律（顺序不可颠倒）：**
1. 先跑 `start_decode.sh`（decode 节点） + `start_prefill.sh`（prefill 节点）——两者并行
2. **等 PD 双端 ready**：两端容器日志都出现 `fired up and ready` 且 `health=200`
3. **router 数量在 PD ready 前必须为 0**（铁律）
4. PD 双端 ready 后再跑 `start_router.sh`（第 3 步 + 第 4 步：router ready 前 =0，ready 后起）
5. 最后 `curl http://127.0.0.1:31000/health` 应 200，再发一个 e2e 确认

---

## 5. 完整还原命令（逐条）

### 5.1 还原 image（需先 docker login）
```bash
docker login b200routeraca.azurecr.io -u <user> -p <pass>
docker pull b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final
```

### 5.2 decode（decode 节点）
```bash
bash start_decode.sh
```

### 5.3 prefill（prefill 节点）
```bash
bash start_prefill.sh
```

### 5.4 router（prefill 节点，PD ready 后）
```bash
bash start_router.sh
```

### 5.5 验证
```bash
# 两端 ready
docker logs sglang-decode 2>&1 | grep "fired up and ready"
docker logs glm53-prefill 2>&1 | grep "fired up and ready"
# router 转发
curl -s http://127.0.0.1:31000/health   # 200
export $(grep -E "^MOL_API_KEY" secrets.env | head -1 | xargs)
curl -s http://172.31.47.105:31000/v1/chat/completions \
  -H "Authorization: Bearer $MOL_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm53","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":40,"temperature":0}'
# 应返回 HTTP 200 + content 含 "2"
```

---

## 6. 常见问题

- **传输失败 500**：几乎都是 engine.so 被错误挂载（b200 三件套 8ea08f1e）。确认容器内 `md5sum /usr/local/lib/python3.12/dist-packages/mooncake/engine.so` = `cf881774...`。本目录 image 已内置正确版，不要挂任何 engine.so。
- **router 起不来（`docker run --help` 报错）**：旧 router 容器占名，先 `docker rm -f glm53-router` 再跑脚本。
- **mooncake session not alive / Failed to get kvcache**：通常是 PD 未同时 ready 或 engine.so 版本错，先对照 §5.5 验证。
