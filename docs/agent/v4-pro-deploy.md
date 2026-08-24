# DeepSeek V4 Pro 0813 通用部署文档

> 目的：给**任意新集群**部署 V4 Pro PD 分离推理的可复用操作手册。本文不含任何具体集群
> IP/端口/密钥（均用 `<占位符>` 标注），但给出**当前生产所用的完整 env 与启动参数**
>（参数本身是通用的，可直接套用）。最后更新：2026-08-24。

## 拓扑（PD 分离，两节点）

| 角色 | 说明 | 端口（约定） |
|---|---|---|
| prefill | 8×TP + CP=8 + 流式 hidden 源侧 | `30100`（engine）/ `8998`（bootstrap） |
| decode | 8×TP + DCP=4 + DSPARK 投机解码 | `30200`（engine） |
| router | PD 分离路由（cache_aware） | `31000`（API）/ `29001`（prometheus） |
| OTel | 指标采集（独立 collector） | `18889`（self） |

公网入口 = `<ROUTER_PUBLIC_HOST>:<ROUTER_PORT>`，API key 部署方自定。

## 模型与 checkpoint

- 模型：`deepseek-ai/DeepSeek-V4-Pro-0813`（官方 FP4 expert + FP8 attention 混合量化，853GB / 66 shards）
- 关键 config 值：hidden_size=7168, q_lora_rank=1536, num_attention_heads=128, head_dim=512, num_hidden_layers=61, n_routed_experts=384, num_experts_per_tok=6, expert_dtype=fp4, moe_intermediate_size=3072, dspark_block_size=5, dspark_target_layer_ids=[58,59,60], index_topk=1024, index_head_dim=128, index_n_heads=64。

### ⛔ 核心教训：第三方重打包 checkpoint

**若拿到的是第三方重打包变体（非官方文件），务必用 HTTP Range 只下载 shard 头几 MB 解析
safetensors JSON header 核对**——官方 vs 坏变体判定表：

| 项目 | 重打包变体（坏） | 官方 |
|---|---|---|
| config.json | Flash 的值（4096/1024/64头/43层/256e） | **7168/1536/128头/61层/384e** + dspark_layers=[58,59,60] |
| expert w1.weight | `[3072, 5376]`（6-bit 两级缩放，1.5×） | `[3072, 3584]`（K/2 packed FP4） |
| expert w1.scale | `[3072, 336]`（3B/64elem） | `[3072, 224]`（标准 per-32 e8m0） |
| shard2 tensors | 1568 | 2337 |

**症状**：load 时 `RuntimeError: size of tensor a (224) must match b (336)`——比值恰好 3:2。
总字节数守恒（831GiB）极具迷惑性。

## 启动参数（两节点均用这些 env；仅 `--host`/`--port`/`--disaggregation-*` 按角色替换）

### prefill 侧

```bash
export HF_HOME=<HF_CACHE_DIR>                     # 共享缓存可加速，否则走本地
export SGLANG_PD_HIDDEN_POOL_TOKENS="393216"       # hidden 源池：覆盖 case50 最大 prompt（365,829）；42KB/row → ~16.9GB
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
# SGLANG_DEBUG_DIAG=1 仅诊断期开（PADDED-AR 集体漂移雷达），生产默认关

python3 -m sglang.launch_server \
  --enable-prefill-cp --cp-strategy interleave \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 --served-model-name deepseek-v4-pro-0813 \
  --host 0.0.0.0 --port 30100 --tp 8 --kv-cache-dtype fp8_e4m3 --enable-cache-report \
  --page-size 256 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --swa-full-tokens-ratio 0.3 \
  --watchdog-timeout 3600 --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --moe-runner-backend flashinfer_mxfp4 --model-impl sglang \
  --enable-metrics --mem-fraction-static 0.85 \
  --disable-overlap-schedule \
  --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device <IB_DEVICE> --disaggregation-mode prefill
```

### decode 侧

```bash
export HF_HOME=<模型缓存目录>
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_DECODE_RADIX_ALLOW_SWA="1"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="0"   # [full] 行平衡本就不可信，降级 warning
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY="0"
export SGLANG_PD_HIDDEN_RECV_POOL_TOKENS="393216"
export SGLANG_PD_HIDDEN_RECV_WINDOW="16384"             # ⛔ 必须 ≥ prefill --chunked-prefill-size(16384)
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
export SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="1073741824"
export SGLANG_DSV4_MHC_PREWARM="1"
export SGLANG_DEFAULT_THINKING="1"
# SGLANG_DEBUG_DIAG=1 生产默认关

python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 --served-model-name deepseek-v4-pro-0813 \
  --host 0.0.0.0 --port 30200 --tp 8 --kv-cache-dtype fp8_e4m3 --enable-cache-report \
  --page-size 256 --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --swa-full-tokens-ratio 0.3 \
  --watchdog-timeout 3600 --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --moe-runner-backend flashinfer_mxfp4 --model-impl sglang \
  --mem-fraction-static 0.90 --enable-metrics \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device <IB_DEVICE> --disaggregation-mode decode --dcp-size 4 \
  --disaggregation-decode-enable-radix-cache --speculative-algorithm DSPARK
```

> DSPark：Pro 0813 自带 DSpark head（config 里 `dspark_target_layer_ids`）——**只加
> `--speculative-algorithm DSPARK`，不要加 EAGLE flags**（num-steps/eagle-topk/num-draft-tokens）。

### router 侧

```bash
python3 -m sglang_router.cli launch \
  --pd-disaggregation \
  --prefill http://<PREFILL_IP>:30100 \
  --decode  http://<DECODE_IP>:30200 \
  --host 0.0.0.0 --port 31000 --api-key <API_KEY> \
  --policy cache_aware --max-concurrent-requests 64 \
  --health-check-timeout-secs 300 \
  --disable-circuit-breaker \
  --request-timeout-secs 3600 \
  --log-level info --prometheus-port 29001
```

## 关键参数差异速查

| 项 | prefill (CP=8) | decode (DCP=4) |
|---|---|---|
| page-size | 256 | 256 |
| chunked-prefill-size | 16384 | 8192 |
| hidden 池 | 源池 `SGLANG_PD_HIDDEN_POOL_TOKENS=393216` | 接收 `RECV_POOL_TOKENS=393216` + `RECV_WINDOW=16384` |
| mem-fraction | 0.85 | 0.90 |
| 投机解码 | 无 | DSPARK |
| radix | `--enable-prefill-cp` | `--disaggregation-decode-enable-radix-cache` + `SGLANG_DECODE_RADIX_ALLOW_SWA=1` |

## OTel 采集（独立 collector，namespace `b300-pd-1p1d-pro`）

参数与模板详见 skill `otel-reporting` §"V4 Pro 1P1D"。要点：
- 两节点各一 collector，抓 prefill/decode/router 三个 `/metrics`，OTLP 推中央 VM
- ⛔ **集群/引擎重启不会自动拉起 otelcol**——重启后必须手动 restart，否则 Grafana nodata
- 若给新集群部署，namespace 可按集群命名（如 `xxx-1p1d-pro`），并同步更新 skill

## 正确部署步骤

1. **下载官方权重**：两端并行 `hf download deepseek-ai/DeepSeek-V4-Pro-0813 --local-dir <out> --max-workers 8`（约 25 min）；多节点可走共享 HF 缓存
2. **验证官方格式**（shard2 头部 w1.weight shape 必须是 `[3072, 3584]`）
3. **部署代码**：rsync 到两端 → **清 `__pycache__`（全 sglang 目录）** → 删 L20D triton config → 杀干净 → 双端同时启动 → 等 health → 启 router → 启 otel
4. **重启前必杀残留 GPU 进程**（`nvidia-smi --query-compute-apps=pid | xargs kill -9`）——旧 scheduler 占显存 OOM；bootstrap 端口 8998 占用致 bind 失败
5. ⛔ **rsync 目录+文件混合会平铺**——逐条 rsync（目录对目录、文件对子目录），部署后 md5 校验关键文件
6. decode 侧 `SGLANG_PD_HIDDEN_RECV_WINDOW` 必须 `≥ prefill --chunked-prefill-size`（当前 16384），否则 src/dst 等长校验失败

## 正确性验收断言（可复用的验收流程）

| 验收项 | 预期 |
|---|---|
| ladder 递增（1 giant → 2 → 4g+10 medium → 全量 case50 42 并发） | 每级 5 探针内容 clean |
| case50 @600rpm | 42/50 + 8 已知 DFlash grammar 400；（内容检查 5/5 clean） |
| 21 请求并发强制中止风暴 | 引擎存活、探针快速恢复 |
| 集合通信奇偶 | PADDED-AR 逐调用插桩全 rank 位点序列一致 |
| 长输出（280K token 单请求） | accept 恒 6.0、无 rebootstrap、步时线性无断崖 |

判据工具：`contam_check.py` / `lc_ladder.py`（内容金丝雀）、`longprobe.py`（长输出剖析）、`padded_seq_diff.py`（奇偶校验）、`abort_storm.py`（中止风暴）。

## 已知行为（非 bug）

- **>1M token 请求返回 400**：agent 侧 token 估算与 tokenizer 计数有 ~8% 偏差，1M 附近偶发超限（非服务 bug）
- 401 = 未带 `Authorization: Bearer <API_KEY>`
- 并发长上下文时步时随 bs 升（5× 批仅 1.5× 步时，良好扩展）；投机预热前 ~200 token accept 从 3 爬升
- DFlash 不支持 grammar-constrained（strict tools/response_format 请求 400 快速失败）

## 关联文档

- `docs/dsv4-dspark-pd-tech-report-2026-08.md`、`docs/agent/pd-hidden-window-design.md`、`docs/agent/dcp-virtual-id-domain-fix.md`、`docs/agent/decode-radix-swa.md`、`docs/agent/dsv4-pro-pd-engineering.md`
- 关键 commit：`35fa068e33`（窗口模式四层修复）、`76639ae1f1`（KV 污染终局）、`98e7a09e9e`（钳制复活）