# AGENTS.md — sglang B300 GLM-5.2 PD 集群专用分支

本仓库是 sglang 的 **B300 GLM-5.2 Prefill/Decode 分离推理集群**专用分支（`b300-glm52`）。
基于上游 `release/v0.5.15`（cherry-pick 至 `0b3bb0cbe`），叠加了大量本地自研修改：
**HiCache + CP layer-split、DCP + PD disaggregation、DSA 稀疏 attention、EAGLE、
MoL（Mixture-of-LoRA）虚拟专家**，以及一系列分布式死锁修复。

> ⚠️ **本分支代码只在 B300 集群部署，改动不向上游提 PR 前需先与本 README/AGENTS 维护者确认。**

---

## 1. 项目身份

| 项 | 值 |
|---|---|
| 分支 | `b300-glm52` |
| 上游 base | `release/v0.5.15`，cherry-pick 到 `0b3bb0cbe`（GLM-5.2 MTP IndexShare） |
| 本地 HEAD | `107716a18`（2026-08-07，见 §3 commit 清单） |
| 模型 | GLM-5.2-FP8（`n_routed_experts=256`，`num_experts_per_tok=8`，`num_hidden_layers=78`，`hidden_size=6144`，FP8 E4M3FNUZ） |
| 推理架构 | PD 分离（prefill/decode 异机）+ DCP（Data/Context Parallel）+ HiCache + DSA + EAGLE |
| 部署 | 1P1D（8.213.215.2）+ 2P3D（8.222.11.182）两个集群，见 §5 |

---

## 2. 核心架构（自研部分）

### 2.1 PD 分离推理（Disaggregated Prefill/Decode）
- prefill 与 decode 运行在不同节点/进程，通过 **mooncake**（RDMA）传输 KVCache。
- decode 用 DCP 模式（`--disaggregation-mode decode --dcp-size N`），prefill 用
  `--disaggregation-mode prefill`。
- bootstrap 端口 `8998`，IB 设备 `mlx5_0`，`mooncake_store.py` 有本地小改动。

### 2.2 DCP（Data/Context Parallel，本分支关键能力）
- decode 端 `--dcp-size 4` + `--tp 8`：8 个 TP rank 按 4 组 DCP 分组，每组负责
  一段 sequence 的 decode（context-parallel decode）。
- prefill 端配合 `--enable-prefill-cp --cp-strategy interleave` +
  `--enable-dsa-prefill-cp-layersplit`（CP layer-split，见 2.4）。
- DCP 相关代码：`model_runner_kv_cache_mixin.py`（+115 行）、`pool_configurator.py`、
  `disaggregation/`、`eagle_worker_v2.py`、`eagle_utils.py`。

### 2.3 HiCache（分层缓存，本分支核心）
- `mem_cache/unified_radix_cache.py` 大改（+699 行）：文件后端（`file-storage-path`、
  `SGLANG_HICACHE_FILE_BACKEND_*`）、write_back 策略、`page_first` 内存布局、
  **local-only prefetch**（每 rank 只预取自己需要的 prefix——这是多处死锁的根源，见 §3）。
- 相关：`memory_pool_host.py`、`mla_buffer.py`、`pool_configurator.py`、
  `model_runner_kv_cache_mixin.py`、`forward_mla.py`（MTP IndexShare）。

### 2.4 DSA prefill CP layer-split（attention 层切分）
- `layers/attention/dsa/`：`dsa_indexer.py`（+161 行，round-robin-split 支持）、
  `dsa_topk_backend.py`、`triton_sparse_mla.py`（FP8 稀疏 MLA）、`quant_k_cache.py`。
- prefill 走 **CP layer-split**：attention 层按 CP 切分，非 attention 层全量；
  prefix broadcast 用 `cp_layersplit_pool.py`（`broadcast_owner_layer_prefix`）。
- 详见 `python/sglang/srt/layers/utils/cp_utils.py` 与
  `python/sglang/srt/mem_cache/cp_layersplit_pool.py`。

### 2.5 EAGLE 投机解码
- decode 端 `--speculative-algorithm EAGLE --speculative-num-steps 5
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`。
- 自研改动集中在 `eagle_worker_v2.py`（+81 行）、`eagle_utils.py`、
  `eagle_draft_extend_cuda_graph_runner.py`（DCP 下 draft 一致性）。

### 2.6 MoL（Mixture-of-LoRA）虚拟专家
- `--enable-lora --lora-paths L0=... L1=... L2=... L3=... --max-lora-rank 16
  --lora-use-virtual-experts`：LoRA 作为"虚拟专家"并入 MoE 专家路径，不依赖 EP。
- 部署层由 `start_pd.sh` + mol_harness proxy（MoL 内容路由）承载。

### 2.7 精度（GLM-5.2-FP8）
- 主模型权重 FP8（deep_gemm，E4M3FNUZ）；KV cache FP8 E4M3（group 128 逐组量化）；
  Attention Q/K/V 激活 FP8，P@KV tl.dot FP8×FP8→FP32 累加；**LoRA 权重保持 BF16**（保微调精度）。

---

## 3. 本地自研 commit 清单（按时间，从旧到新）

> 这是本分支的灵魂。**改任何分布式/缓存/CP 相关代码前先读这些 commit 的动机。**

| commit | 内容 | 关键文件 |
|---|---|---|
| `86a3729e2` | **B300 GLM-5.2 PD 基础功能**：DCP + HiCache + EAGLE + triton config gitignore | dsa_indexer、unified_radix_cache、pool_configurator、fp8_utils、start_pd.sh |
| `7f164f4cd` | fix(nccl)：cp_layersplit gate rank-invariant + prefill barrier + pool None guard（首个死锁修复） | cp_utils、prefill.py、cp_layersplit_pool |
| `a605ab4e6` | fix(nccl)：decision 用 global seq_len（**导致 prefill 崩溃，已 revert**） | dsa/utils.py |
| `13181f2f1` | **Revert a605ab4e6**（回退到 7f164f4cd 稳定版） | — |
| `d4d23041d` | fix(cp)：**DSA prefill CP gates rank-invariant**——`can_dsa_cp_split` 用 `seq_len`，`can_dsa_prefill_cp_round_robin_split` 用 `extend_num_tokens`（batch prep 时算，HiCache 修改前），保留 CP 路径 | dsa/utils.py |
| `4ba456a80` | fix(hicache)：**`bulk_check_prefetch_progress`**——scheduler 每 tick 一次性收集全部请求状态，替代 per-request `check_prefetch_progress`（每个 1-2 个 all_reduce，per-rank 漂移 → cross-collective 死锁） | mem_cache/unified_radix_cache.py |
| `3df13dd42` | fix(prefill)：**iteration barrier 移到 `pop_bootstrapped` 前**——修 py-spy 实锤的 `poll_and_all_reduce_attn_cp_tp_group` all_reduce 死锁（utils.py:138 vs 143 行号不一致 = collective 不匹配） | disaggregation/prefill.py |
| `6a00bbde3` | fix(hicache)：bulk_check_prefetch_progress **finalize path 修正** | mem_cache/unified_radix_cache.py |
| `db3e58904` | fix(cp)：**consensus prefix length across CP ranks**——HiCache local-only prefetch 使 prefix_indices 跨 rank 分歧 → input_ids 长度不同（2048 vs 2045）→ CP split 后 key size 不同 → `cp_all_gather_rerange_output` 死锁（NCCL 证据：Rank0 NumelIn=262144 vs Rank1-5 261760） | （部署前需确认已同步） |
| `3ee5d215` | perf(dcp)：去 decode DCP 路径每步无条件 `seq_lens.tolist()` CPU 同步（只用于 debug 日志）——实测 SSE 间隔 28→25ms | dsa_backend.py |
| `9f22e5ff8` | perf(dcp)：`_localize_page_dcp_metadata_` 的 8+ torch 小 kernel 融合成 1 个 Triton kernel（就地安全：每元素读一次 + 寄存器 cumsum + clamped 输出） | dcp/kernels.py, dsa_backend.py |
| `33c571f6f` | perf(dcp)：cp_lse 复用 `new_output` buffer（省每步 GPU 分配） | dcp/kernels.py, comm.py |
| `101dbe017` | perf(dcp)：`plan_topk_v2` 支持 in-place（省每步 new_empty + copy_） | jit_kernel/dsv4/topk.py, dsa_backend.py |
| `b5a571970` | perf(dcp)：预编译 topk_v2 JIT kernel（`_jit_topk_v2_module()` 移到 backend init，省首请求 tvm_ffi 编译） | dsa_backend.py |
| `7cc8b4b64` | perf(dcp)：Triton `expand_lens_2d`（替代 view+expand+contiguous，省 schedule 中间拷贝） | dcp/kernels.py, dsa_backend.py |
| `107716a18` | **perf(eagle)：移植上游 #30947/#30948**——topk1 draft postprocess 融合 Triton kernel（argmax+positions advance+token store 合一，绕过 `select_top_k_tokens`/per-step list/torch.cat，打 2P3D 的 ~42ms step_time base）+ TP vocab-parallel embedding 融合 kernel（`SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING=0` 可关）；顺带清理 EAGLE-DIAG/POS-DIAG/DCP-TV 诊断日志。**本地无 kernels/ KernelSpec 注册表（v0.5.15 基），kernel 放在 `srt/*/triton_ops/`，未做 namespace 迁移** | eagle_worker_v2.py、speculative/triton_ops/topk1.py、layers/vocab_parallel_embedding.py、layers/triton_ops/vocab_parallel_embedding.py、environ.py |

### 死锁修复演化（读懂这一串 = 理解本分支）
```
per-rank gate 分歧（7f164f4cd）→ global seq_len 失败（a605ab4e6 revert）
→ rank-invariant gates（d4d23041d）
→ per-request check_prefetch all_reduce 漂移（4ba456a80 bulk 化）
→ pop_bootstrapped all_reduce 死锁（3df13dd42 barrier 前移）
→ prefix length 跨 rank 分歧（db3e58904 consensus）
```
**共同根因**：HiCache local-only prefetch 让每 rank 看到的 prefix/请求状态不一致 →
collective（NCCL/gloo）调用数或参数跨 rank 不匹配 → 死锁 → 600s watchdog SIGABRT。
**修复原则：CP 相关 gate 与 collective 必须 rank-invariant**。

---

## 4. 关键代码位置速查

| 关注点 | 路径 |
|---|---|
| DSA CP 决策 gate | `python/sglang/srt/layers/attention/dsa/utils.py`（`can_dsa_cp_split`、`can_dsa_prefill_cp_round_robin_split`） |
| DSA indexer | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` |
| HiCache 缓存 | `python/sglang/srt/mem_cache/unified_radix_cache.py` |
| CP layer-split pool | `python/sglang/srt/mem_cache/cp_layersplit_pool.py` |
| CP 工具 | `python/sglang/srt/layers/utils/cp_utils.py` |
| PD prefill 事件循环 | `python/sglang/srt/disaggregation/prefill.py` |
| DCP KV 传输 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| EAGLE DCP 适配 | `python/sglang/srt/speculative/eagle_worker_v2.py`、`eagle_utils.py` |
| EAGLE topk1 draft postprocess（107716a18 移植） | `python/sglang/srt/speculative/triton_ops/topk1.py`（`draft_topk1_postprocess`，在 `eagle_worker_v2.py::draft_forward` 的 topk=1 chain 路径调用） |
| 融合 TP vocab embedding（107716a18 移植） | `python/sglang/srt/layers/triton_ops/vocab_parallel_embedding.py`（在 `layers/vocab_parallel_embedding.py::_embed_local_shard` 调用，`SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING` 控制） |
| 部署脚本 | `start_pd.sh`、`recover_b300_pd.sh`（仓库根） |

---

## 5. 部署拓扑与运维要点

### 5.1 1P1D（8.213.215.2）
- B300-1 = SSH `1021`：prefill + router(30000) + gateway(31001) + proxy(31000)；
  Python `/root/sglang_venv/bin/python3`；代码 `/root/sglang_venv/.../sglang/srt/`
- B300-2 = SSH `1022`：decode（**DCP=4 + EAGLE 5 steps**）；Python `/root/v15_patched/bin/python3`
- 公网 `8.213.215.2:18888`，model `Macaron-V1-Venti`，key `MOL_API_KEY_1P1D`
- **必须带 `thinking_mode`（reasoning_effort=max）**，否则 GLM-5.2 输出异常

### 5.1.1 1P1D 规范启动配置（2026-08-10 round30-34 二分实验验证）

> ⛔ **2026-08-10 教训：1P1D 重启必须带完整 env**。round29 用"不设 env"启动出现严重退化
> （8/32 correct=4），round30-34 带完整 env 全部正常（21-24/32）。**缺 env 是本分支 1P1D
> 退化的主因之一**（与 L20D config 同级），不是代码问题。

**规范 env（启动前 export，两端都必须带）**：
```bash
export TVM_FFI_CUDA_ARCH_LIST="10.0a"
export MOONCAKE_DISABLE_HIP_DMABUF="1"        # decode 缺 → KV 传输建立失败
export IBV_ACCESS_RELAXED_ORDERING="1"        # decode 缺 → 同上
export MC_IB_PCI_RELAXED_ORDERING="1"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE="1"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE="1000"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="600"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="600"
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1"
export SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN="1"  # prefill 缺 → 启动即崩
export SGLANG_MOE_PADDING="1"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="1073741824"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="/root/hicache"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="200G"
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE="10G"
```

**缺 env 的症状（实测）**：
- prefill 缺 `SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN` → 启动即崩：
  `AssertionError: cp-layersplit requires num_layers (78) % cp_size (8) == 0 unless SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN=True`
- decode 缺 mooncake env（`MOONCAKE_DISABLE_HIP_DMABUF`/`IBV_ACCESS_RELAXED_ORDERING`/
  `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 等）→ **health 200 但请求卡死**：
  prefill/decode 日志出现 `KVTransferError(bootstrap_room=...): Aborted by AbortReq`，
  推理不开始 → 客户端超时。并发负载下表现为失败率高/退化。

**decode 实际启动参数（DCP=4 + EAGLE 5，注意与 start_pd.sh 的过时注释不同）**：
```bash
/root/v15_patched/bin/python3 -m sglang.launch_server \
  --model-path /root/glm52_local/base_l2_merged --served-model-name glm52-fp8-official \
  --host 0.0.0.0 --port 30200 --tp 8 --kv-cache-dtype fp8_e4m3 --enable-cache-report \
  --page-size 128 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion --model-impl sglang \
  --mem-fraction-static 0.90 --skip-server-warmup --enable-metrics \
  --cuda-graph-max-bs-decode 64 --max-running-requests 64 \
  --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device mlx5_0 --disaggregation-mode decode --dcp-size 4 \
  --speculative-algorithm EAGLE --speculative-draft-model-path /root/glm52_local/base \
  --speculative-num-steps 5 --speculative-eagle-topk 1
```
> 注意：**无 `--speculative-num-draft-tokens`、无 `--disable-custom-all-reduce`、page-size 128**
> ——与 2P3D（page-size 64 + draft-tokens 6 + disable-custom-all-reduce）不同，但 round30-34 实测正常。

**prefill 实际启动参数**（page-size 64 + CP layer-split + HiCache）：
```bash
/root/sglang_venv/bin/python3 -m sglang.launch_server \
  --model-path /root/glm52_local/base_l2_merged --served-model-name glm52-fp8-official \
  --host 0.0.0.0 --tp 8 --kv-cache-dtype fp8_e4m3 --enable-cache-report \
  --page-size 64 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --watchdog-timeout 3600 --reasoning-parser glm45 --tool-call-parser glm47 \
  --moe-runner-backend triton --enforce-disable-flashinfer-allreduce-fusion --model-impl sglang \
  --enable-metrics --port 30100 --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-ratio 1 --hicache-write-policy write_back \
  --hicache-mem-layout page_first --hicache-storage-backend file --file-storage-path /root/hicache \
  --enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit \
  --disable-overlap-schedule \
  --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device mlx5_0 --disaggregation-mode prefill
```

**round30-34 二分实验结论（2026-08-10）**——8 个 commit（db3e58904→7cc8b4b64）逐一验证：
| 代码 | CI 结果 | 结论 |
|---|---|---|
| db3e58904（基线） | 24/32 ✅ | 正常 |
| +107716a18（EAGLE topk1 融合 + vocab embedding 融合） | 24/32 ✅ | 排除 |
| +3eeb5d215/+9f22e5ff8/+33c571f6f（DCP perf×3） | 24/32 ✅ | 排除 |
| +101dbe017/+b5a571970（topk_v2 优化×2） | 24/32 ✅ | 排除 |
| +7cc8b4b64（expand_lens_2d，完整版） | 21/32 ⚠️ | **唯一可疑 commit**（-3，偶发） |

**结论**：严重退化（round29 22-34%）主因 = **缺 env**（KV 传输问题），非代码；7cc8b4b64 的
expand_lens_2d 有轻微负影响（24→21），需修复或回退。⚠️ 修复 expand_lens_2d 后建议再跑一轮完整 CI。

**round35-36 性能优化实验（2026-08-10）**——decode 高并发 TPOT 优化尝试：
| 尝试 | 12 并发 TPOT p50 | 结论 |
|---|---|---|
| 基线（默认 kernel，deep_gemm JIT） | **12.8ms** ✅ | 最优 |
| 放回 E=257 fused_moe config（原 L20D 自带） | avg 60ms | ❌ 负优化 |
| 手动 B300 config（Blackwell 参数模式 + K%128 约束） | 35.7ms | ❌ 负优化 |
| `--speculative-num-draft-tokens 6` | 12.8ms | 与默认相同（**topk=1 时 sglang 自动调整 = steps+1 = 6**） |
| `--cuda-graph-max-bs-decode 128` | 12.8ms | 无提升（graph 无 fallback：287/287 True） |

**关键认知**：
1. **GLM-5.2 的 MoE 走 deep_gemm（FP8 E4M3FNUZ）**，`fused_moe_triton` 的 config 文件机制对它**无效**——
   改 `configs/triton_3_6_0/E=257,...` 不会影响性能（deep_gemm 有自己的 JIT 编译，M 覆盖 1-16384 全量）。
2. **B300 的 device_name 伪装为 "NVIDIA L20D"**（275GB 显存）——sglang 按 device_name 匹配 config，
   这就是 L20D config 被加载的原因。**本地 configs 目录的 L20D 文件已全部删除**（gitignore 已覆盖，
   `git archive` 不带它们）。
3. **单请求 TPOT ≈ 4.3ms（235 tok/s）**，12 并发 p50 ≈ 12.8ms（高并发退化 3×）——当前架构上限。
   降一倍需要代码级 profiling（nsys/NCU：MoE vs MLA attention vs DCP 通信），非启动参数可达。
4. round35 最终确认：**默认配置 = 24/32 (75%) 无乱码**。

### 5.2 2P3D（8.222.11.182，SSH 1100–1104）
- 1100/1103/1104 decode（DCP=4 + EAGLE 5 steps），1101 prefill+router+gateway+proxy，1102 prefill
- Python `/opt/sglang-venv/bin/python`；`sglang-router` 在 `/opt/sglang-venv/bin/`（**不在 PATH**）
- 公网 `8.222.11.182:18777`，key `MOL_API_KEY_2P3D`
- 详细运维见 skill：`mol-prod-ops`（拓扑/重启/故障树）、`otel-reporting`（监控）

### 5.3 ⛔ 重启铁律（2026-08-05 确立，2026-08-06 增补）
1. **每次重启必须先 rsync 本地最新代码到全部节点 + 清 cache**（`__pycache__` + triton JSON + deep_gemm），绝不带旧代码重启；
2. **⛔ rsync 后必须删除 L20D triton config**（2026-08-06 事故教训）：`rm -f /opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/*L20D*.json`（1P1D: `/root/v15_patched/...` 同路径）。**本地 configs 目录含打包的错误 E=1024 L20D conf，rsync 会把它带到集群，重启时 sglang 找到它 → MoE kernel 用错配置 → TPOT 从 23ms 暴涨到 80ms**。删除后回退默认 kernel（实测 B300 上 16-27ms 正常）。1P1D 目录无此文件即为对照基准。
3. **杀进程必须干净**：`ps aux | grep -E 'launch_server|sglang::scheduler|sglang::router|smg|mol_harness'` + `lsof -ti :<port>` 双重杀，验证 **0 残留 0 端口** 后才启动；
4. **health 检查直接 curl 节点**，不要信嵌套 SSH 轮询的 000（会误判漏检）；
5. 重启后按序：router → gateway → proxy → **otelcol**（否则 Grafana nodata）；
6. 最后确认公网 chat 请求返回 200 + 正确输出。
7. **⛔ 永远禁止 `scp`，文件传输一律用 `rsync`**（2026-08-11 确立）。scp 在 B300 节点上会挂起/断连（实测 151MB 二进制 scp 7 分钟未完成），rsync 稳定且支持断点/校验。跨节点传文件用：
   ```bash
   # 本地 → 节点
   rsync -avz -e "ssh -p <port> -o ConnectTimeout=15" <src> root@<host>:<dest>
   # 节点 → 节点（B300-1 → B300-2，已有免密）：用 ssh cat 管道
   ssh root@10.0.0.67 "cat > <dest>" < <local_file>
   ```

---

## 6. 开发流程

```bash
# 改代码 → 本地验证 → commit（保持 b300-glm52 分支）
git add <files> && git commit -m "fix(scope): description with root cause"
git push origin b300-glm52   # remote 是 MindLab-Research/sglang 的 b300-glm52

# 部署到集群（每次重启必须同步最新代码）
SRC=python/sglang/srt/; DEST=/opt/sglang-venv/lib/python3.12/site-packages/sglang/srt/
# (1P1D: B300-1 /root/sglang_venv/...、B300-2 /root/v15_patched/...)（2P3D: 5 节点全同步）
rsync -avz --exclude='__pycache__' --exclude='*.pyc' "$SRC" root@<node>:"$DEST"
# 清 cache → 杀干净 → 重启 → 等 health 200 → router/gateway/proxy/otelcol → 确认公网
```

## 7. 已知陷阱速记

- **移植的 Triton 融合 kernel 有 kill-switch**：`SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING=0` 可关闭 TP vocab embedding 融合路径（默认开）；topk1 draft kernel 无开关（topk=1 + CUDA 自动启用，异常时走 `topk1_chain_fits` fallback）。
- **HiCache local-only prefetch 是死锁温床**：任何依赖 per-rank 状态的 collective gate 都会出问题；判断"rank-invariant"再动。
- **`seq_len` 在 CP token split 前是 rank-invariant 的，`extend_seq_lens_cpu` 不是**（HiCache 会改）。
- **a605ab4e6（全局 seq_len 决策）是失败的**：导致 DeepGEMM `attention.hpp:150 cu_seq_len_k_start.size(0) == seq_len` 崩溃，别重新部署。
- **1101 曾反复因 `gloo/tcp/pair.cc:547 Connection closed by peer 22.0.68.78` 崩溃**（网络层，非代码）——重启集群恢复，根因待查。
- **router/proxy 会"进程活着但转发卡死"**（health 200、请求超时）——重启即恢复；router 重启后必须重启 gateway+proxy。
- **decode 是 DCP=4 + DSPARK**（V4 Flash 用 DSPARK；EAGLE 是 GLM-5.2 的旧配置）。
- **⛔ DSV4 的 DCP KV 必须广播（2026-08-14 修复）**：decode 端每 DCP rank 持有**全量池**（`_pre_alloc_fill_len` 不按 dcp_size 缩减，req_to_token 是全量长度）——所以**主 KV（SWA/DSA 经 maybe_send_extra）和压缩 KV（c4/c128 经 transfer_worker）都必须广播**（跳过 DCP token-shard 分片）。若分片传输（每 rank 只收 1/dcp）→ decode 读取范围（req_to_token 0..len）远超写入范围（0..len/dcp）→ **KV 污染 → target logits 错 → 输出乱码（draft 停滞是症状不是根因）**。判断：`_is_dsv4_kv_transfer()`（is_deepseek_v4）；GLM（1/dcp 布局）保持分片。验证：dcp=4 出师表全文背诵（finish=stop）。
- **⛔ rsync 后必须彻底清 `__pycache__`（2026-08-14 教训）**：`find ... -name '__pycache__' -o -name '*.pyc' | xargs rm -rf`——只清目标目录的子目录不够，旧 .pyc 会让进程加载**没有最新修复的代码**（曾导致广播修复"看似没生效"——实际是缓存遮挡）。改 mooncake/conn.py 等热路径后必须全 sglang 目录清缓存再重启。
- **⛔ L20D triton config 是 TPOT 杀手（2026-08-06）**：本地 `configs/triton_3_6_0/` 里有打包的错误 `E=1024,*L20D*.json`，rsync 到集群后 sglang 启动会找到并使用它 → MoE kernel 按 1024 专家展开 → TPOT 23ms→80ms。**每次 rsync 后必须 `rm *L20D*.json`**；找不到 config 回退默认 kernel 才是 B300 上的正确状态（1P1D 无此文件、TPOT 7-21ms 为基准）。
- **TPOT 异常排查**：先拉 Grafana `sglang:inter_token_latency_seconds` P50 时序找突变点（VictoriaMetrics query: `histogram_quantile(0.5, sum(rate(sglang:inter_token_latency_seconds_bucket{cluster=...,engine_type="decode"}[15m])) by (le))`），再对 decode 日志 `Decode batch` 算 `TPOT ≈ #running-req / gen-throughput × 1000ms`。
- **⛔ DSPARK PD 并发崩溃根因（2026-08-15 修复，commit 见 git log）**：4+ 并发 SIGQUIT 全崩的完整链路 = 新 prebuilt batch 的 `build_disagg_draft_input` 返回 None（spec_info.py 缺 DSPARK 分支）→ `merge_batch` 跳过 spec_info merge（None 守卫）→ running batch 持有 stale bs=1 的 draft_input 而 batch 涨到 bs=N → verify ForwardBatch 混用 bs=1 input_ids + bs=N req_pool_indices → cuda graph `fill_from` shape [1] vs [N] 崩。**修复三件套**：#31466 移植（PD hidden state 传输，16 文件）+ `schedule_batch.merge_batch` None 守卫 + `spec_info.py` DSPARK 分支（`make_next_draft_input`）。fake/nixl/mori 的 `send_metadata` 必须带 `spec_metadata` kwarg（签名兼容，否则 warmup 即 TypeError）。
- **⛔ rsync 目录+文件混合会平铺（2026-08-15 重大事故）**：`rsync dir1/ dir2/ file3 root:dest/` 会把 dir1/dir2 的**内容平铺到 dest/ 根**（不是 dest/dir1/）——曾把 `disaggregation/utils.py` 写到 `srt/utils.py`、common/ 平铺到 srt/common/，导致 ImportError + 部署静默丢失。**必须逐条 rsync（目录对目录、文件对子目录），部署后 md5 校验关键文件**。
- **诊断标记（grep 用，保留）**：`FILL-MISMATCH`（cuda_graph_buffer_registry.py，fill_from shape 不匹配时打印 slot+shape）、`DSP-STAGE/DSP-ACCEPT/DSP-WIN/DSP-LOC/DSP-ALLOC`（DSPARK 流程）、`STATE-BCAST/SKIP-FLAGS/DSV4-CHECK`（KV 广播）、`DCP-PD-IDX/DEBUG-INIT/DEBUG-CAN`（decode 请求流）。
- **⛔ DeepGEMM JIT 的 sm_103 必须带 a 后缀（2026-08-15 修复，16 并发 crash 真根因）**：B300 真实 capability=(10,3)=SM103（device_name 伪装 "NVIDIA L20D"），DeepGEMM JIT 拼 `--gpu-architecture=sm_103`（无 a）→ 16384 大 shape 的 `tf32_hc_prenorm_gemm` 生成 tcgen05 指令 → ptxas 报 `Instruction 'tcgen05.fence' not supported on .target 'sm_103'` → prefill crash → decode 表现为 `reconnect to 8998`（**后果非根因**）。且 CUDA 13.2 nvcc 的 `-arch=sm_103a` 短格式在 fatbin 模式下丢 a 后缀（PTX 仍 .target sm_103），**必须用 `-gencode arch=compute_103a,code=sm_103a` 长格式**。修复：**直接替换 pip 包内的 nvcc**（`nvidia/cu13/bin/nvcc` → wrapper 脚本，原文件改名 nvcc.real，拦截 `--gpu-architecture=sm_103` 重写为 gencode 长格式）。注意 `DG_JIT_NVCC_COMPILER` env 无效（DeepGEMM 0.1.4 不读它，实际从 `find_cuda_home()`→`CUDA_HOME/bin/nvcc` 找编译器）；改 CUDA_HOME 会触发 flashinfer JIT 全量重编（崩溃），不能动。失败残留 kernel.cu 在 `/root/.cache/deep_gemm/tmp/`（含模板参数可手动复现编译）。16 并发前最大安全 batch 是 768 token（1536 触发新 shape 编译）。
- **⛔ disagg poll 必须用专用 gloo group（2026-08-15 修复，commit 2dd9d2c168，abort 死锁终极根因）**：并发 abort 洪峰（客户端超时取消 → `KVTransferError: Aborted by AbortReq`）时，per-rank 清理耗时（prepare_abort/stream_output）使 rank 间错开一整轮循环——快 rank 进入下一轮 `recv_requests` 的 **broadcast**、慢 rank 还在上一轮 `pop_transferred` 的 **all_reduce**，两者共用 `attn_tp_cpu_group`，gloo FIFO 把异构 collective 互相匹配 → 8 rank 全卡死（**health 200、请求全超时、0 crash**——py-spy 实证 TP0/TP3 卡 `_padded_all_reduce_min` vs TP5 卡 broadcast）。修复：scheduler 初始化时 `torch.distributed.new_group(backend="gloo")` 创建专用 group，传给 DecodeTransferQueue/DecodePreallocQueue/PrefillBootstrapQueue——poll 序列与 broadcast/barrier 序列隔离，跨 group 时序漂移无害。
- **⛔ poll() 抛异常会永久错位 collective 计数（2026-08-15 终极修复，commit c841e03cf9）**：专用 group 后仍有 7v1 卡死——某 rank 的 `kv_receiver.poll()` **抛异常**（KVTransferError 传播/zmq 错误）在 `_padded_all_reduce_min` **之前**逃逸 → 该 rank 的 per-group collective 计数**永久偏移 1** → 之后该 rank 领先一整轮（卡在下一轮 broadcast 等别人），其余 7 rank 永远等它进 pop gloo（py-spy：TP2 在 broadcast vs TP0/1/3/4/5/6/7 在 all_reduce）。**修复三件套**：① `_poll_with_failure_injection` 每个 poll() 包 try/except → 转 `KVPoll.Failed` + 异常 stash 到 receiver（`_stashed_poll_exception`）——collective 计数对 receiver 异常免疫；② pop_transferred Failed 分支透传 stash 异常（含 is_from_another_rank）；③ 专用 group 加 300s timeout（残余未知分歧 5 分钟响亮崩溃而非静默卡 1h watchdog）+ `_padded_all_reduce_min` 失败时打印进程级调用计数（`PADDED-AR-FAIL` grep 标记，取证用）。
- **⛔ prefill pop_bootstrapped 空队列分支跳过 CP collective（2026-08-15 终极根因，commit 1f0e0cc95a）**：prefill 的 `pop_bootstrapped` 空队列时**直接 return，不调 `poll_and_all_reduce_attn_cp_tp_group`**——bootstrap 队列填充是 per-rank TCP（时序分歧），有请求的 rank 调 collective、空队列 rank 不调 → **attn_cp 组计数永久错位** → 8 个 prefill rank 全卡在 `pop_bootstrapped → _padded_all_reduce_min`（health 200、0 crash、请求零处理；decode 次生卡在 iteration barrier——prefill 卡死 → 8998/HTTP 无响应 → decode 某 rank 卡在 abort 清理的 HTTP 上）。**修复**：空分支在 `attn_cp_size > 1` 时用空 poller 列表参与 collective（与 decode 侧 pop_transferred 空分支补偿同构）。同 commit 系列：`poll_and_all_reduce_attn_cp_tp_group` 改用异常安全的 `_poll_with_failure_injection`（裸 `int(poller.poll())` 抛异常会同时跳过链式两个 `_padded`）。
- **⛔ DSPARK PD 要求两端 radix 策略一致（2026-08-15）**：#31466 的握手检查 `DSpark hidden PD requires matching prefill/decode radix cache policies: prefill=True, decode=False` → 500。DSPARK PD 部署时 **prefill 必须去掉 HiCache/radix**（`--disable-radix-cache`，删 `--enable-hierarchical-cache --hicache-* --file-storage-path`；上游 #31466 自己的测试 `test_pd_prefill_dspark_rejects_hierarchical_cache` 正是这个约束）。prefill 备份在 `/root/start_v4_prefill.sh.bak_hicache`。
- **⛔ tilelang JIT 并发编译崩溃（2026-08-15 修复，prefill 空闲后 16 并发崩的根因）**：tilelang 的 KernelCache 只用 `threading.Lock`（**进程内**），8 个 scheduler 是独立进程——空闲后首个并发请求 burst 让 8 rank 同时 cache-miss → 同时 `tilelang.lower`/`BuildTileLangCUDA` → 共享 staging 目录/disk cache 竞争 → **TVM C 层崩溃**（8 rank 齐崩，非死锁）。**修复**：patch 两端 `tilelang/cache/kernel_cache.py`（备份 `.orig`）——cache-miss 编译段包 `fcntl.flock` 跨进程文件锁（`.compile.flock`）+ 锁内 double-check `_load_kernel_from_disk`（先到的进程编译，后到的直接加载产物）。同时把 `SGLANG_DSV4_MHC_PREWARM` 从 0 改 1（load 时预热 mhc kernel，barrier 同步——锁 patch 后预热并发安全），把 JIT 编译移出 serving 路径。
- **⛔ tilelang `cuda/atomic` 头文件缺失（2026-08-15 修复）**：tilelang 的 `lower.py` 只传 `-I TILELANG_TEMPLATE_PATH` 和 `-I CUTLASS_INCLUDE_DIR`，缺 `-I nvidia/cuda_cccl/include`（`cuda/atomic` 所在路径）。CUDA 13.2 的 nvcc 不自带 CCCL include（旧版自带）。**修复**：patch 两端 `tilelang/engine/lower.py`——在 options 列表里加 `"-I<venv>/lib/python3.12/site-packages/nvidia/cuda_cccl/include"`。三个编译问题都表现为 prefill crash → decode 报 `reconnect to 8998`（断连是后果非根因）。修复后 v39 验证 72/72 全通过（含 16 并发+abort 洪峰+空闲后 16 并发×2）。
