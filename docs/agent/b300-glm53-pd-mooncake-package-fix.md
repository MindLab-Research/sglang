# b300 GLM-5.3 PD mooncake 完整包配套修复 + v2 最终 image（2026-09-07）

> **最终正确配置（80RPM replay 验证 transfer 失败=0）**：v2 image = pd-fix base + **b200 完整 mooncake 包内置** + HEAD 代码内置，部署**无需任何 mooncake/代码挂载**。
> 本文记录根因链、v2 image 构建方法、关键坑（防重蹈）。

## 1. 根因链（完整）

### 1.1 mooncake Python 包必须与 engine.so 配套（不是单个文件）

| 尝试 | 结果 |
|---|---|
| 只挂单文件 `/nvme/mooncake-b200/engine.so`（8ea08f1e）覆盖 base image 的 cf881774 | **ImportError: Please install mooncake**——单 6MB engine.so 缺配套 ep_*.so 绑定和 py 文件 |
| 只用 base image 自带 cf881774（4MB 单文件）+ b200 对齐 env | 传输不稳定（decode/prefill 互报 dead） |
| **b200 完整 mooncake 包**（engine.so 8ea08f1e + 8 个 ep_*.so 绑定 + 全部 py，共 35 文件）+ b200 对齐 env | **✅ transfer 失败=0，80RPM replay 稳定** |

**关键**：mooncake 传输正确性取决于**完整包配套**（engine.so + ep_*.so Python 绑定 + py 文件），从 b200 2p2d 容器 `docker cp glm53-decode:/usr/local/lib/python3.12/dist-packages/mooncake /tmp/mooncake_pkg` 打包导出才是正确来源。PyPI 上的 `mooncake==0.0.1` 是 14KB 假包（占位符），不能用。

### 1.2 PD 两端必须配对重启

只重启 decode 不重启 prefill（或反之）→ 两端 mooncake session/engine 状态不一致 → 传输失败（prefill 报 "Decode instance could be dead"、decode 报 "Failed due to an unknown reason from another rank"）。**decode + prefill 必须用完全一致的配置同时重启**。

### 1.3 env 必须对齐 b200 2p2d（删 tcp pool 对 + 补关键项）

- **删**：`MC_TCP_ENABLE_CONNECTION_POOL`、`MC_TCP_CONNECTION_IDLE_TIMEOUT`（b200 2p2d 没有的 tcp pool 对）
- **补**：`MC_TCP_LANES_PER_PEER=16`、`MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384`、`MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=16384`、`MC_TCP_ADMISSION_TIMEOUT_MS=10000`、`MOONCAKE_PROTOCOL=tcp`、`MC_FORCE_TCP=1` 等
- 完整 33 条见 `deploy/b300/glm53/b200_aligned.env`

### 1.4 测试端点：公网 IP + 本地代理劫持坑

**从本地测试必须用公网 IP `http://3.37.20.114:31000/v1`**：
- 本地执行环境设有 HTTP 代理（HTTP_PROXY 等），curl 内网 IP（172.31.47.105）会被代理劫持返回**假 502**（Bad Gateway）
- 内网 IP 从本地本来就不可达（`--noproxy '*'` 时直接 000 超时）
- **教训**：曾在"服务实际健康"时因 curl 内网 IP 得到假 502 而误判服务故障，浪费大量排查。**curl spot 机器：公网 IP 或 ssh 进机器用 127.0.0.1；且注意 `--noproxy '*'`**

## 2. v2 最终 image（无需挂载包含一切）

```
b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v2
digest: sha256:2f97fe6347c1b3dede538994085ab0a59e1991fe2c13f62061e31122695a6003
```

构建（在 spot 节点）：
```dockerfile
FROM b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-pd-fix
COPY mooncake_pkg/ /usr/local/lib/python3.12/dist-packages/mooncake/   # b200 完整包 35 文件
COPY sglangpkg/ /sgl-workspace/sglang/python/sglang/                     # HEAD 代码
```

验证项：engine.so=8ea08f1e（6MB）、`from mooncake.engine import TransferEngine` OK、conn.py=e79f3463、memory_pool.py 含 DCP 修复。

部署（`deploy/b300/glm53/`）：**只挂载数据目录**（models/hicache/deep_gemm/tvm-ffi），env 用 `b200_aligned.env`，无需 mooncake/代码挂载。

## 3. 铁律（防重蹈）

1. **PD 两端配对重启**：decode + prefill 用完全一致配置同时重启；router 在 PD 双端 ready 后才起（PD ready 前 router=0）
2. **mooncake 包必须整包配套**：从 b200 容器 docker cp 打包整目录，不能只换单个 engine.so
3. **测试用公网 IP**：`http://3.37.20.114:31000/v1`；本地 curl 内网 IP 会被 HTTP 代理劫持返回假 502
4. **router worker 注册是一次性的**：router 启动时 decode 未 ready → discover_metadata 超时 → 永远 "No decode workers"，必须重启 router
5. **PD 分离集群的容器日志是排障关键**：重启前先 `docker logs` 备份（`docker rm -f` 会删日志）
