# 2P3D 集群（8.222.11.182）完整配置与运维命令

> 集群别名"1P2D"（当前实际形态 1 prefill + 2 decode）。SSH 端口 1100–1104 映射到同一台公网机
> 8.222.11.182 的不同 VF 节点。本文档记录 2026-08-20 部署/排查后的**实际运行配置**（以实际
> 操作为准，`/root/start_pd.sh` 为老脚本，其中部分路径已过时，见 §7）。

---

## 1. 拓扑与节点

| 节点 | SSH 端口 | 内网 IP | 角色 | venv | 服务端口 |
|---|---|---|---|---|---|
| 1101 | `ssh -p 1101 root@8.222.11.182` | 10.0.58.34 | **prefill + router + gateway + proxy** | `/root/v15_patched` | 30100 / 30000 / 31001 / 31000 |
| 1100 | `ssh -p 1100 root@8.222.11.182` | 10.0.58.38 | **decode** | `/root/v15_patched` | 30200 |
| 1103 | `ssh -p 1103 root@8.222.11.182` | 10.0.58.36 | **decode** | `/root/v15_patched` | 30200 |

测试集群（勿混入线上流量）：
| 1102 | `ssh -p 1102` | 10.0.58.35 | 测试 prefill + 测试 router | `/root/sglang_venv` | 30100 / 30000 |
| 1104 | `ssh -p 1104` | 10.0.58.37 | 测试 decode | `/root/sglang_venv` | 30200 |

**公网入口**：`8.222.11.182:18777` → proxy(31000)。key `MOL_API_KEY_2P3D`。
模型名分两层：**proxy 层 `Macaron-V1-Venti`**（MoL 人格路由），router/引擎层 `glm52-fp8-official`。

**请求链路**：公网 18777 → proxy(31000, mol_harness) → gateway(31001, smg) → router(30000, smg)
→ prefill(1101:30100) + decode(1100/1103:30200)。

## 2. 关键路径速查

| 项 | 路径 |
|---|---|
| Python（线上三台） | `/root/v15_patched/bin/python3` |
| Python（测试 1102/1104） | `/root/sglang_venv/bin/python3` |
| sglang 代码（线上） | `/root/v15_patched/lib/python3.12/site-packages/sglang/`（srt/ 从本地 repo rsync） |
| **smg 二进制（router & gateway）** | `/usr/local/bin/smg`（备份 `smg.bak.0730_v032`） |
| mol_harness（proxy） | `/root/mol-stack/mol_harness`（PYTHONPATH=/root/mol-stack，用 `/usr/bin/python3`） |
| prefill 启动脚本 | 1101 `/root/start_1p2d_lora.sh`（写 prefill 块） |
| decode 启动脚本 | 1100 / 1103 各自 `/root/start_1p2d_lora.sh`（写 decode 块） |
| 模型权重 | `/root/glm52_local/base` |
| LoRA | `/root/glm52_local/loras/{L0,L1,L2,L3}`（L2 当前为 0818001 版本，另有 L2_old/L2.fp8_bak 等历史变体） |
| 日志 | prefill `/root/prefill.log`、decode `/root/decode.log`、router `/root/router.log`、gateway/proxy `/root/tmp/*.log` |

## 3. 服务配置（实际运行参数）

### 3.1 prefill（1101:30100）

```bash
/root/v15_patched/bin/python3 -m sglang.launch_server \
    --model-path /root/glm52_local/base \
    --served-model-name glm52-fp8-official \
    --host 0.0.0.0 --tp 8 \
    --kv-cache-dtype fp8_e4m3 --enable-cache-report \
    --page-size 64 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
    --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion \
    --model-impl sglang \
    --enable-lora \
    --lora-paths L0=/root/glm52_local/loras/L0 L1=/root/glm52_local/loras/L1 \
                 L2=/root/glm52_local/loras/L2 L3=/root/glm52_local/loras/L3 \
    --max-lora-rank 16 --max-loaded-loras 4 --max-loras-per-batch 4 \
    --lora-use-virtual-experts --max-lora-chunk-size 128 \
    --enable-metrics --port 30100 --mem-fraction-static 0.85 \
    --enable-prefill-cp --cp-strategy interleave \
    --disable-overlap-schedule \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device mlx5_0 \
    --disaggregation-mode prefill
```

### 3.2 decode（1100/1103:30200，两台同参）

```bash
/root/v15_patched/bin/python3 -m sglang.launch_server \
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
    --lora-paths L0=... L1=... L2=... L3=...   # 同 prefill
    --max-lora-rank 16 --max-loaded-loras 4 --max-loras-per-batch 4 \
    --lora-use-virtual-experts --max-lora-chunk-size 128 \
    --mem-fraction-static 0.90 --skip-server-warmup --enable-metrics \
    --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device mlx5_0 \
    --disaggregation-mode decode --dcp-size 4 \
    --speculative-algorithm EAGLE --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
```

差异备注：1100 的脚本比 1103 多 3 行 CUDA coredump 诊断 env（`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`
等），其余完全一致。

**公共 env（start_1p2d_lora.sh 内 export，两端都要）**：
```bash
export TVM_FFI_CUDA_ARCH_LIST="10.0a"
export MOONCAKE_DISABLE_HIP_DMABUF="1"
export IBV_ACCESS_RELAXED_ORDERING="1"
export MC_IB_PCI_RELAXED_ORDERING="1"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE="1"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="1000"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="600"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_DISAGGREGATION_QUEUE_SIZE="64"          # prefill 侧
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE="256"   # prefill 侧
export SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN="1"  # prefill 侧，缺则启动即崩
export SGLANG_MOE_PADDING="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_DSA_SLOT_OOB_DIAG=1        # decode 诊断（1100/1103 脚本均有）
export SGLANG_DSA_STAGE_SYNC=1
export SGLANG_ENABLE_ASYNC_ASSERT=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### 3.3 router（1101:30000，smg 二进制）

```bash
nohup /usr/local/bin/smg launch \
    --pd-disaggregation \
    --prefill http://10.0.58.34:30100 \
    --decode http://10.0.58.38:30200 \
    --decode http://10.0.58.36:30200 \
    --host 0.0.0.0 --port 30000 --api-key sk-glm52-pd \
    --policy cache_aware --max-concurrent-requests 64 \
    --health-check-timeout-secs 300 \
    --disable-circuit-breaker \
    --request-timeout-secs 3600 \
    --log-level info --prometheus-port 29003 \
    > /root/router.log 2>&1 < /dev/null &
```

（实验/排查时可只挂单 decode 便于归因；线上标准形态为双 decode。）

### 3.4 gateway（1101:31001，smg 二进制）

```bash
nohup /usr/local/bin/smg launch \
    --host 127.0.0.1 --port 31001 --prometheus-port 29002 \
    --policy manual --assignment-mode min_load \
    --worker-urls http://127.0.0.1:30000 \
    --api-key sk-glm52-pd \
    --max-idle-secs 1800 --request-timeout-secs 86400 --disable-circuit-breaker \
    > /root/tmp/gateway.log 2>&1 < /dev/null &
```

### 3.5 proxy（1101:31000，mol_harness）

```bash
cd /root/mol-stack && \
PYTHONPATH=/root/mol-stack \
UPSTREAM=http://127.0.0.1:31001 \
MOL_UPSTREAM_RUNTIME=sglang \
MOL_API_KEY=MOL_API_KEY_2P3D \
MOL_UPSTREAM_API_KEY=sk-glm52-pd \
PROXY_PORT=31000 \
MOL_USE_MODEL_ROUTER=1 \
MOL_PURE_MODEL_ROUTE=1 \
MOL_HOP_TIMEOUT=86400 \
MOL_SSE_KEEPALIVE_INTERVAL_S=10 \
MOL_MAX_CONNECTIONS=8192 \
MOL_MAX_INFLIGHT_REQUESTS=512 \
MOL_UPSTREAM_MAX_CONNECTIONS=1024 \
MOL_UPSTREAM_MAX_KEEPALIVE=512 \
nohup /usr/bin/python3 -m mol_harness.proxy > /root/tmp/proxy.log 2>&1 < /dev/null &
```

⚠️ `MOL_UPSTREAM_RUNTIME=sglang` **必须显式设置**：mol-stack 新版在 smg RouterManager 后面无法
自动探测 runtime 方言，缺省时启动即失败（"upstream runtime auto-detection could not determine
a safe payload dialect"）。

## 4. 代码部署（本地 → 线上）

```bash
SRC=python/sglang/srt/
DEST=/root/v15_patched/lib/python3.12/site-packages/sglang/srt/
# 三台逐一 rsync（⛔ 严禁目录+文件混合 rsync，会平铺；scp 也禁用）
for port in 1101 1100 1103; do
  rsync -avz -e "ssh -p $port" --exclude='__pycache__' --exclude='*.pyc' $SRC root@8.222.11.182:$DEST
done
# 铁律：清 __pycache__ + 删 L20D triton config
find $DEST \( -name '__pycache__' -o -name '*.pyc' \) | xargs rm -rf
rm -f $DEST/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/*L20D*.json
```

当前代码基线：本地 repo `b300-glm52` HEAD（含 draft pool 修复 `371a991947`）。

## 5. 重启流程（严格按序）

1. **先下 proxy**（再下 gateway）——止血入口；
2. rsync 最新代码 + 清缓存 + 删 L20D（§4）；
3. 杀干净三台进程（`ps aux | grep -aE "sglang::scheduler|launch_server"` 双杀 + 验证 0 残留 0 端口）；
4. 启动 prefill(1101) → decode(1100) + decode(1103)，等三台 health 200；
   ⚠️ **prefill 重启后配对的 decode 必须重启**（mooncake RDMA/bootstrap 配对会断）；
   ⚠️ decode 重启后 DeepGEMM warmup ~1min；若清过 deep_gemm 缓存则全量重编 ~5min；
5. router(1101:30000) → 等 health 200；
6. gateway(1101:31001) → health 200；
7. proxy(1101:31000) → health 200；
8. 公网端到端验证：`curl 8.222.11.182:18777` + mol key + model=Macaron-V1-Venti。

## 6. 验证与诊断命令

```bash
# health（直 curl 节点，勿信嵌套轮询）
ssh -p 1101 root@8.222.11.182 'curl -s -o /dev/null -w "%{http_code}" http://localhost:30100/health'
ssh -p 1100 root@8.222.11.182 'curl -s -o /dev/null -w "%{http_code}" http://localhost:30200/health'

# accept / 崩溃 / draft pool 回归判据
grep 'Decode batch' /root/decode.log | tail -3          # accept len 健康带 2.2-3.2
grep -cE 'Xid|CUDA error|Traceback' /root/decode.log    # 应为 0
grep 'KV Cache is allocated' /root/decode.log           # draft 行 #tokens 必须 = target×4（DCP=4）

# LoRA 管理 API（双端都要操作，保持 PD 同步）
curl -X POST http://10.0.58.34:30100/load_lora_adapter   -d '{"lora_name":"L0","lora_path":"/root/glm52_local/loras/L0"}'
curl -X POST http://10.0.58.38:30200/unload_lora_adapter -d '{"lora_name":"L0"}'

# flush cache（排查 radix 串数据时双端一起）
curl -X POST http://10.0.58.34:30100/flush_cache
curl -X POST http://10.0.58.38:30200/flush_cache
```

## 7. 已知坑（本文档特有）

- **`/root/start_pd.sh` 路径过时**：`ROUTER_BIN=sglang-router` 不在 PATH（用 `/usr/local/bin/smg`）；
  `GATEWAY_BIN/PROXY_SRC=/root/Mixture-of-LoRA-Harness-alpha` 目录不存在（gateway 用
  `/usr/local/bin/smg`，proxy 用 `/root/mol-stack`）。且其 proxy() 缺 `MOL_UPSTREAM_RUNTIME=sglang`。
- **pkill -f 禁令**：曾用 `pkill -9 -f "smg"` 自匹配杀掉 SSH 会话与 gateway——只允许精确 PID kill。
- **1102 测试 router**（sglang_router python CLI）与 1101 smg router 是不同实现；线上链路统一用 smg。
- **模型名两层**：公网请求 model 必须用 `Macaron-V1-Venti`（proxy 层）；直接打 router 用
  `glm52-fp8-official`。LoRA 触发只认请求体 `lora_path` 字段。
- **LoRA 多加载乱码问题（2026-08-20 排查中，未结案）**：第二及以上 LoRA adapter 加载/请求后，
  LoRA 前向持续乱码且 sticky（flush+unload+reload 不恢复）；伴随 radix 跨请求串数据（base 读到
  别人 prompt，双端 flush_cache 可恢复 base）。EAGLE accept 同步崩到 1.0。base 权重本身健康。
  1P1D 测试集群（1102/1104，sglang_venv）未复现——其从未加载第二个 LoRA。排查期间 proxy 保持下线。

## 8. 当前（2026-08-20 23:30 CST）状态快照

- prefill 1101 / decode 1100 / decode 1103：运行（回滚版 lora_manager，max-loras 4/4），health 200
- router：运行，**实验形态（仅挂 1100 单 decode）**——恢复线上需改回双 decode 重启
- gateway / proxy：**停止**（乱码排查期间，proxy 按用户指令下线）
- KV cache（回滚版重启后实测）：prefill 1,505,344 tokens / 77.43 GB；
  decode target ≈1,644K / 84.57 GB + draft ×4（371a991947 修复生效）
