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
| 本地 HEAD | `21b00129fe`（2026-08-22，见 §3 commit 清单） |
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
| `3eeb5d2152` | perf(dcp)：去 decode DCP 路径每步无条件 `seq_lens.tolist()` CPU 同步（只用于 debug 日志）——实测 SSE 间隔 28→25ms | dsa_backend.py |
| `9f22e5ff8` | perf(dcp)：`_localize_page_dcp_metadata_` 的 8+ torch 小 kernel 融合成 1 个 Triton kernel（就地安全：每元素读一次 + 寄存器 cumsum + clamped 输出） | dcp/kernels.py, dsa_backend.py |
| `33c571f6f` | perf(dcp)：cp_lse 复用 `new_output` buffer（省每步 GPU 分配） | dcp/kernels.py, comm.py |
| `101dbe017` | perf(dcp)：`plan_topk_v2` 支持 in-place（省每步 new_empty + copy_） | jit_kernel/dsv4/topk.py, dsa_backend.py |
| `b5a571970` | perf(dcp)：预编译 topk_v2 JIT kernel（`_jit_topk_v2_module()` 移到 backend init，省首请求 tvm_ffi 编译） | dsa_backend.py |
| `7cc8b4b64` | perf(dcp)：Triton `expand_lens_2d`（替代 view+expand+contiguous，省 schedule 中间拷贝） | dcp/kernels.py, dsa_backend.py |
| `107716a18` | **perf(eagle)：移植上游 #30947/#30948**——topk1 draft postprocess 融合 Triton kernel（argmax+positions advance+token store 合一，绕过 `select_top_k_tokens`/per-step list/torch.cat，打 2P3D 的 ~42ms step_time base）+ TP vocab-parallel embedding 融合 kernel（`SGLANG_OPT_USE_TRITON_VOCAB_PARALLEL_EMBEDDING=0` 可关）；顺带清理 EAGLE-DIAG/POS-DIAG/DCP-TV 诊断日志。**本地无 kernels/ KernelSpec 注册表（v0.5.15 基），kernel 放在 `srt/*/triton_ops/`，未做 namespace 迁移** | eagle_worker_v2.py、speculative/triton_ops/topk1.py、layers/vocab_parallel_embedding.py、layers/triton_ops/vocab_parallel_embedding.py、environ.py |
| `5974a4fc56` | **fix(dcp)：decode 概率性乱码双根因终局**——index_k 全域 owner 换算（低水位 target 池写错槽）+ topk `-1` 补齐 lane 重定向本行 lane-0（trtllm 无 mask 真实 attend 无关 KV）。判据 `SGLANG_DSA_SLOT_OOB_DIAG` | dsa_indexer.py、dsa_backend.py |
| `d11d2b5179` | **fix(dsa)：>524K-ctx GPU 楔死**——draft 路径未截断 local seq_lens 超 trtllm max_seq_len clamp 与 lane 表容量（2048×64）→ split-KV completion 失步 → reduce 相位永久自旋（无 Xid）。修复=clamp 到 `_sparse_topk×page_size` + v2 topk 扫描 lens 界定页表宽 | dsa_backend.py、dsa/dsa_topk_backend.py |
| `6504ba9b71` | **fix(prefill)："OOM" 记账分裂**——extend_range.end vs fill_ids vs prefix_indices 三源撕裂 + evict(token域)/alloc(page域) 口径不一 → 按虚高 seq 杀全组。修复=`prepare_for_extend` 不变量强制（`EXTEND-ACCOUNTING-DIVERGENCE` 响亮日志）+ 统一页视角 alloc/evict | managers/schedule_batch.py、mem_cache/allocation.py |
| `21b00129fe` | docs：prefill OOM 记账分裂 postmortem（6504ba9b71 配套，`docs/agent/prefill-oom-accounting.md`）；同族 `04bf47a24e`（DCP 乱码 postmortem）、`0a13d7a435`（正确性战争人类可读复盘）、`23e9666289`（CP 洗冤更正+远程 LoRA 下载文档） | docs/ |

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
- 公网 `8.213.215.2:18888`，model `Macaron-V1-Venti`，key `$MOL_API_KEY_1P1D`（真值见仓库根 secrets.env）
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
- 公网 `8.222.11.182:18777`，key `$MOL_API_KEY_2P3D`（真值见仓库根 secrets.env）
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

### 5.4 DSpark PD 排障铁律（2026-08-24 确立，adm 怒令）

1. **⛔ decode 端 radix 绝对不能关**：PD 分离 + DSpark **必须双端开 radix**
   （prefill `--enable-hierarchical-cache`/radix + decode `--disaggregation-decode-enable-radix-cache`
   + `SGLANG_DECODE_RADIX_ALLOW_SWA=1`）。双端 radix 是 DSpark PD 的**架构性前提**——
   prefill 命中少传 KV，decode 必须命中对齐复用；关 decode radix = prefill 命中后
   hidden/KV 覆盖缺口必炸（`hidden_start != decode_prefix_len` → 500 / out-of-order → abort），
   等于倒退回 2026-08-15 之前的全链路事故。**任何"关 radix 试一下"都是禁止的。**
2. **⛔ 禁止"不改代码、只改参数就重启"**：遇到 bug 必须走**代码级根因修复**
   （读代码 → 定位 → 改代码 → 验证），不许用改启动参数 / env 开关绕过
   （关 radix、关 `--speculative-algorithm`、关 draft graph 之类的参数级"修复"一律无效）。
   需要二分定位时，**以 commit 切换（代码变更）为单位**，不做参数级开关实验。
   例外：仅"恢复被误改的正确生产配置"允许参数级重启（那是纠错，不是修 bug）。

## 6. 安全与密钥铁律（2026-08-24 确立）

任何 API key / access token / 密码 / 私钥 **绝不允许硬编码进 repo**（含 AGENTS.md、docs/agent/、.xbot/skills/、scripts、tools、test）。2026-08-24 已用 `git filter-repo --force` 把历史中的公网 key 整体铲除并强推。铁律：

1. **真值只放仓库根 `secrets.env`**（已 .gitignore，不提交）。repo 内一律用环境变量引用：`$MOL_API_KEY_1P1D`、`$MOL_API_KEY_2P3D`、`$MOL_API_KEY`。脚本用 `${VAR:?msg}` 强制 env 传入（不留硬编码默认值）；文档写"见仓库根 secrets.env"。
2. **新增集群/密钥**：key 一律写进本机 `secrets.env`，绝不进 commit。文件要 key 才能跑就用 `source secrets.env` 或 env 传入。
3. **commit/push 前自查**：`git grep -nE 'sk-mol-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{30,}' HEAD -- .` 应 0 命中；命中即回退改引用，再提交。
4. **若已 push 泄露**（不可逆，务必先备份）：① `git bundle create /tmp/bk.bundle --all`；② `git filter-repo --force --replace-text <file>`（每行 `明文==>替换值`）；③ 重写会移除 origin remote，需 `git remote add origin ...`；④ `git push --force origin b300-glm52`；⑤ **立即轮换/吊销该 key**——历史重写不回收已泄露的凭据，公共网络/日志里可能仍在。
5. **仓库根 remote 带 GitHub token**：`git remote -v` 会显示 `x-access-token:ghp_...`——这只是本地 `.git/config`，不进 commit 历史；但勿把它粘进任何日志/文档/命令输出。

## 7. 开发流程

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

## 8. 已知陷阱速记

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
- **⛔ disagg poll 必须用专用 gloo 组（2026-08-15 修复 2dd9d2c168；2026-08-16 补完 538ee9d0d1——prefill 侧漏改路径致 7 分钟静默悬挂）**：zmq recv 是 per-rank 异步——某 rank 收到控制消息进 broadcast、其余 rank 流入 poll 时，gloo FIFO 会把**不同轮次的异构 collective 互相匹配** → 跨轮错位死锁（health 200、零 crash、请求全超时；低负载下表现为分钟级 TTFT 尖峰，**新请求到达瞬间恢复**是它的签名）。2dd9d2c168 只覆盖了 decode 队列与 PrefillBootstrapQueue 构造参数，**pop_bootstrapped 内部 poll_and_all_reduce 的组参数（attn_cp/attn_tp 原组）与两个 event_loop 的 iteration barrier（tp_cpu_group）漏改**。538ee9d0d1 补完：`pf_barrier_gloo_group / pf_poll_tp_gloo_group / pf_poll_cp_gloo_group`（同成员副本、独立 FIFO、600s 兜底）接管 prefill 循环全部 CPU collective。py-spy 判据：rank 散布在不同轮次的 barrier/poll/dp_gather = 此病；取证 dump 存 `/root/deadlock_dump_*.txt`。

- **⛔ 538ee9d0d1 的漏改比记录的严重得多（2026-08-17 修复 419f592de9）**：该 commit 实际只改了 pop_bootstrapped，还漏了 **6 处** prefill 事件循环的 poll collective——`process_disagg_prefill_inflight_queue`、`resolve_waiting_queue_bootstrap`、`process_batch_result_disagg_prefill`、`get_transferred_rids`、`check_bootstrap`、`get_ready_bootstrapped_rids_for_pp`。它们仍用原组 `attn_cp/tp_cpu_group`，且 `process_disagg_prefill_inflight_queue` 空队列时直接 `return []` 跳过 collective → 某 rank 计数永久错位。后果 = **8 rank 全体卡在同一处 iteration barrier**（prefill.py:1061 `all_reduce`）——与上面"rank 散布"形态互补，是"100% 卡死、health 200、零 crash、请求全超时"的另一种签名。判据：py-spy **并发**抓 8 rank 全停在同一 all_reduce（注意：串行抓会因时间跨度产生"散布"假象）。修复 = 6 处全部改 `getattr(self,"pf_poll_cp/tp_gloo_group",None) or 原组`，空队列用空 poller 参与 collective。
- **⛔ poll() 抛异常会永久错位 collective 计数（2026-08-15 终极修复，commit c841e03cf9）**：专用 group 后仍有 7v1 卡死——某 rank 的 `kv_receiver.poll()` **抛异常**（KVTransferError 传播/zmq 错误）在 `_padded_all_reduce_min` **之前**逃逸 → 该 rank 的 per-group collective 计数**永久偏移 1** → 之后该 rank 领先一整轮（卡在下一轮 broadcast 等别人），其余 7 rank 永远等它进 pop gloo（py-spy：TP2 在 broadcast vs TP0/1/3/4/5/6/7 在 all_reduce）。**修复三件套**：① `_poll_with_failure_injection` 每个 poll() 包 try/except → 转 `KVPoll.Failed` + 异常 stash 到 receiver（`_stashed_poll_exception`）——collective 计数对 receiver 异常免疫；② pop_transferred Failed 分支透传 stash 异常（含 is_from_another_rank）；③ 专用 group 加 300s timeout（残余未知分歧 5 分钟响亮崩溃而非静默卡 1h watchdog）+ `_padded_all_reduce_min` 失败时打印进程级调用计数（`PADDED-AR-FAIL` grep 标记，取证用）。
- **⛔ prefill pop_bootstrapped 空队列分支跳过 CP collective（2026-08-15 终极根因，commit 1f0e0cc95a）**：prefill 的 `pop_bootstrapped` 空队列时**直接 return，不调 `poll_and_all_reduce_attn_cp_tp_group`**——bootstrap 队列填充是 per-rank TCP（时序分歧），有请求的 rank 调 collective、空队列 rank 不调 → **attn_cp 组计数永久错位** → 8 个 prefill rank 全卡在 `pop_bootstrapped → _padded_all_reduce_min`（health 200、0 crash、请求零处理；decode 次生卡在 iteration barrier——prefill 卡死 → 8998/HTTP 无响应 → decode 某 rank 卡在 abort 清理的 HTTP 上）。**修复**：空分支在 `attn_cp_size > 1` 时用空 poller 列表参与 collective（与 decode 侧 pop_transferred 空分支补偿同构）。同 commit 系列：`poll_and_all_reduce_attn_cp_tp_group` 改用异常安全的 `_poll_with_failure_injection`（裸 `int(poller.poll())` 抛异常会同时跳过链式两个 `_padded`）。
- **⛔ DSpark PD 前缀复用必须两端同开 radix（2026-08-15 全链路落地）**：DSpark hidden 吃 target 中间层激活、不可缓存不可从 KV 反推 → prefill 单开命中后 hidden 覆盖缺口必炸（`hidden_start != decode_prefix_len` → 500 / 流式 out-of-order → abort）。现已支持两端同开：decode 加 `--disaggregation-decode-enable-radix-cache` + `SGLANG_DECODE_RADIX_ALLOW_SWA=1`，prefill 去掉 `--disable-radix-cache`。核心设计 = **树不持有 SWA**（`swa_served_from_tree=False`：validator 恒过 / insert 复活跳过防双重 free / 尾窗重算 / finish 时 `free_swa` 补位），共修 4 层 bug（hidden clamp、len-1 match、SWA validator、SWA 池泄漏）。**报 leak 时注意 `[full]` 行平衡也别信——真凶在下一行 `[swa]`**。完整设计与修复链见 `docs/agent/decode-radix-swa.md`。
- **⛔ tilelang JIT 并发编译崩溃（2026-08-15 修复，prefill 空闲后 16 并发崩的根因）**：tilelang 的 KernelCache 只用 `threading.Lock`（**进程内**），8 个 scheduler 是独立进程——空闲后首个并发请求 burst 让 8 rank 同时 cache-miss → 同时 `tilelang.lower`/`BuildTileLangCUDA` → 共享 staging 目录/disk cache 竞争 → **TVM C 层崩溃**（8 rank 齐崩，非死锁）。**修复**：patch 两端 `tilelang/cache/kernel_cache.py`（备份 `.orig`）——cache-miss 编译段包 `fcntl.flock` 跨进程文件锁（`.compile.flock`）+ 锁内 double-check `_load_kernel_from_disk`（先到的进程编译，后到的直接加载产物）。同时把 `SGLANG_DSV4_MHC_PREWARM` 从 0 改 1（load 时预热 mhc kernel，barrier 同步——锁 patch 后预热并发安全），把 JIT 编译移出 serving 路径。
- **⛔ tilelang `cuda/atomic` 头文件缺失（2026-08-15 修复）**：tilelang 的 `lower.py` 只传 `-I TILELANG_TEMPLATE_PATH` 和 `-I CUTLASS_INCLUDE_DIR`，缺 `-I nvidia/cuda_cccl/include`（`cuda/atomic` 所在路径）。CUDA 13.2 的 nvcc 不自带 CCCL include（旧版自带）。**修复**：patch 两端 `tilelang/engine/lower.py`——在 options 列表里加 `"-I<venv>/lib/python3.12/site-packages/nvidia/cuda_cccl/include"`。三个编译问题都表现为 prefill crash → decode 报 `reconnect to 8998`（断连是后果非根因）。修复后 v39 验证 72/72 全通过（含 16 并发+abort 洪峰+空闲后 16 并发×2）。
- **⛔ 乱码/短循环【进行中 2026-08-16 06:4x】已修复崩溃/已对齐/未达成确定性**：①**dup 双 free 已修**（`0c0b127cce`：入口 cap 毒化 cache_unfinished 自 rematch 记账，节点2..9=139520 逐位吻合；cap 属服务策略只能在调用方，自 rematch 必须见全量）——同负载从必崩→零泄漏零崩溃；②**chunk 对齐生效**（STATE-CLEAR-RAN pre_len=131072 实证，快照边界对齐）；③**确定性未达成**（nondet 仍 3-4 分叉）。④**tombstone 探测两次失败**（served_from_tree 仍 True）：`unified_kv_pool` 属性链不通；改用 `swa_kv_pool is None + get_unified_kv 存在` 标记仍不触发（`4e989deec6`）——**下一步先实证本部署的真实池模式**：`is_unified_kv_triton()` 可能 env-gated 返回 False → 实际跑 SPLIT 池（swa_kv_pool 为真对象）→ 若如此，"unified ring 悬垂"理论错位，真凶在 SPLIT 池的 swa 槽复用路径（swa_radix_cache 家族），需换向。判据工具不变：dev `/tmp/nondet.py`（4 连发分叉计数+cached 值）、插桩 `SGLANG_DEBUG_ALLOC=1`。①unified-kv 把 SWA 从内容稳定池槽改为 per-request scratch ring（133 行/槽，按 `rpi*133+pos%133` 寻址），但树 SWA 组件仍按"存槽位 ID=存内容"旧语义工作——insert 存的行 ID 在请求结束后悬垂（内容被下一任 rpi 占用者覆写）→ 命中边界窗读到任意前任尾部内容（同输入 greedy 4 连发 3~4 种输出，插桩实证 rpi=4/5/6 输出各异）；②`full_to_swa` 映射写入时烧录（指向写入者行、永不更新）+ `set_full_to_swa_mapping` 在 unified 下是 no-op 桩 + overlap 恢复把映射置 0（swa_component.py:112-113）——三重自证设计者知道此路不通；③修复快照协议（`e62494cd6a`+`3effe7c3c2`+`ee4798533a`：insert 时抓尾 133 行内容快照挂 host_value、match limit floor 到 chunk 边界保证位置对齐、命中时恢复到本请求行+映射重定向）形式化可证 4/4 一致且逐位等价，**但 HiCache host 匹配不消费 RadixKey.limit——cached=139264 > limit=131072 实证 host_hit 越过对齐边界恢复**，不变量在 host 层被打破（L1-only 下证明闭合）。④**树入口钳位（4aeb0fce58，已 revert 96f6ae5555）实测暴露 prefill insert 双重 free**：边界从 139264 改 131072 后首个 139K 请求完成即 `full_num_used=-139520`（整个请求 545 页双 free）→ invariant_checker 击杀 scheduler——prefill 的 finished-insert/overlap-free 记账在新边界下错配（AGENTS 记载的同类坑家族）。**完整修复 = host 层钳位 + insert 记账边界修复 + 快照协议，三者缺一不可**。判据：dev 47.87.64.67 `/tmp/nondet.py` 的 **cached 值直接指示对齐状态**（131072=对齐✓/139264=host 破坏✗）+ 输出 4/4 一致；插桩 `SGLANG_DEBUG_ALLOC=1`（[ALLOC-DIAG]/[STATE-CLEAR*]）。**CP（d300）已洗清**（CP off 仍复现），修好可恢复。
- **⛔ 乱码/短循环【2026-08-16 17:0x 最新状态：崩溃链已修+恢复链已生效+残余=kernel 级非确定】**：①dup 双 free 已修（`0c0b127cce`，139520 逐位归因）；②快照-恢复协议最终形态=**边界字典**（`0a53594fb2`：`_c128_boundary_snaps[(boundary_len, hash(尾128tok))]`，绕开 SWA horizon 节点分裂与 last_node 语义——树节点挂快照不可达，n_klen=256/p_klen=16128 实证）；`C128RESTORE-OK` 日志实证恢复执行；③**残余非确定的定性**：FORCE_MISS 实验（SGLANG_RADIX_FORCE_MISS=1 同输入全量重算×4）= **4 种输出且零循环** → kernel 级非确定（atomics）实锤——即使完美缓存系统，本部署硬件栈也做不到 4/4 逐字一致；全量重算时输出全是干净变体，命中路径因分布边缘化部分 run 陷循环。**要 100% 确定需 kernel 级确定性**（flashinfer/triton 原子操作路径），属上游深度工程。④生产现状：CP=8+radix 全开、0 崩溃 0 泄漏、accept 0.72-0.78 健康；乱码 prompt 命中仍概率循环（~3/4）。判据：`/tmp/nondet.py`；机制日志 C128SNAP/C128RESTORE（限流 5 条）。
- **⛔ V4 Pro 0813 当前生产模型（2026-08-15）**：集群已切换到 `deepseek-v4-pro-0813`（FP4 expert + FP8 attention，853GB 官方权重）。**本地旧权重是第三方重打包变体**（config 全错 + 6-bit 两级缩放 tensor 1.5× 尺寸），官方权重在 `/root/dsv4_pro_official`（硬链接到运行目录），坏权重备份在 `_repacked_bad`。识别/部署方法见 `docs/agent/v4-pro-deploy.md`。decode 只加 `--speculative-algorithm DSPARK`（Pro 自带 DSpark head，不加 EAGLE flags）；MoE 用 `--moe-runner-backend flashinfer_mxfp4`（SM100 走 trtllm FP4 kernel）。>1M token 请求 400（agent 侧 token 估算偏差所致，非服务 bug）。
- **⛔ PD hidden 长请求 606s 静默楔死（2026-08-18 已修，`1910bb619c`）**：1c9e1c3275 的预留需求是 `min(U, pool.size)`——超长请求（agent 流量 U≫P）要**整个池**，任何并发持有者（1 条 RUNNING/1 个在途 chunk）→ `alloc` 永远 None → 请求静默 park 在 `pending_reqs`（无日志无超时），prefill 侧 Bootstrapping 等 600s 兜底撕裂（TTFT=606s）。次要根因：准入读**本地池可用量**，而 chunk 行按 rank 本地 ACK 时序归还 → 各 rank 准入判定可分歧 → 部分 bootstrap（OB'）。修复 = 窗口化 `w=min(U,W)` + 记账式准入（charge 只随 ADMIT/TRIM/终结等同步事件变化，不读本地池）+ 水位 φ=W 队头豁免。**部署默认 `SGLANG_PD_HIDDEN_RECV_WINDOW=0`（=旧行为，必须显式设 `16384` 才启用窗口）**。⚠️ **窗口硬约束（`8ae56004ba`，16:13 冻结事故）：W 必须 ≥ prefill `--chunked-prefill-size`（当前 16384）**——sender 按 prefill chunk 整块发包且要求 src/dst 等长，W<chunk 会 mismatch；该异常曾裸杀 transfer_worker 线程致双端冻结（现已被捕获转单请求失败）。已实测 68 万 token 大请求 20s 完成、并发小请求不被饿死、判据 `PDH-ADMIT`/`PDH-PARK`/`invariant violated` 全干净。判据：`grep PDH-PARK`（应短暂出现后 PDH-ADMIT）、`invariant violated` 仍应为 0。完整设计与证明见 `docs/agent/pd-hidden-window-design.md`（含 v1.1 冻结事故补遗）。
- **⛔ sgl-model-gateway control plane（`sgl-model-gateway/`，编译产物=smg）**：launch PD 模式**必须注册 children** 否则 `GET /v1/control/models` 返回空（`POST /v1/control/register` 手动注册也行）——已在 server.rs startup 自动注册（commit `c1663ec790`）；`GET /v1/models` 经 RouterManager 转发 single_router（代理引擎，含已加载 lora）；**swap 槽位只数 lora 不算 base**（否则 2 槽 + 1 lora 即显满 → 每次部署强制 swap 连环顶掉旧 adapter，prefill/decode 双端状态不一致 → 请求触发引擎自动 reload 15G + decode abort 卡死）。`v1/completions` 端点在 PD 模式 smg 处理卡死（既有 bug，用 `v1/chat/completions`）。LoRA 触发只认请求体 `lora_path` 字段（按 adapter 名字匹配），`model` 字段引擎不校验。
- **⛔ EAGLE draft loc 三段式崩溃族（2026-08-19/20 结案链，commits `8bfe264118`→`0e5f7704e1`）**：`prepare_for_draft` 从 req_to_token 拷贝的 `out_cache_loc` 是 draft KV 写/读/free 的唯一源头；drain 期竞态（torn kv_start/陈旧 req_to_token）产生 ≥池容量的垃圾 loc。三段演化：①裸奔期=直接 Xid 31 WRITE 崩 TP 组（12 次崩溃）②只修写端（set_mla_kv_buffer 消毒到 scratch 页）=不崩但垃圾仍进 free → 分配器永久中毒 → **accept 3.0→1.0 断崖且新请求也不恢复** ③根治=在源头 clamp（eagle_worker_common.prepare_for_draft，commit `0e5f7704e1`）。判据：`grep DRAFT-LOC-OOB`（SGLANG_DSA_STAGE_SYNC=1 时打印垃圾样本）；accept 断崖+零崩溃日志=free 中毒形态。消毒目标必须用 scratch 页（self.size 起）不能 slot 0（真实槽位）。
- **⛔ DCP 虚拟 id 域（2026-08-20 终极根因，`9db63a6abb`）**：DCP decode allocator = `PagedTokenToKVPoolAllocator(capacity=size×dcp, page_size=64×dcp)`——**req_to_token 存虚拟 id**（虚拟页 256=4×64，物理页 4v+k 属 rank k）。**`3c68f20891` 在 prepare_for_draft 的 global→local 修复 = 双转**（set_mla_kv_buffer 的 DCP 分支本身就转 virtual→local + owner 过滤）→ 写错位/Xid 31/跨 rank 污染/accept 崩；且 set_mla_kv_buffer 的 `≥size→scratch` sanitize 在 DCP 下破坏合法虚拟 id。**修复 = 虚拟 id 全程保留**：①prepare_for_draft **不转**（clamp 只 min=0）②set_mla_kv_buffer **DCP 下跳过 sanitize**（OOB 窗口 size×dcp）③读路径 repair **加 rank 过滤**（`(id//64)%dcp==rank` 才转本地，异 rank→0/-1）④free 天然回到虚拟域。公式 `_page*_dws` 通用 DCP=4/8。判据：`DRAFT-LOC-OOB`/`DRAFT-LOC-FOREIGN`=0。完整见 `docs/agent/dcp-virtual-id-domain-fix.md` + 隔壁崩溃 `docs/agent/decode-crash-2026-08-20-0144.md`。
- **⛔ DCP draft pool 尺寸丢失（2026-08-20 终局，`371a991947`）**：merge v0.5.16 把 draft pool 构造改走 `KVCacheConfigurator._derive_pool_sizes`，**丢了 draft_pool_token_multiplier**（死代码留在 `_apply_memory_pool_config`，draft 不再经过）→ draft pool 从 size×dcp（7.4M 全虚拟空间）缩到 size（1.85M）→ 虚拟 id 越界 → 后续 sanitize/虚拟→本地压缩 workaround 把 foreign-rank 槽位（3/4）挤进一个 scratch 页 → draft 读垃圾 KV → **accept rate 0.07**。修复三件套：①`_derive_pool_sizes` 恢复 multiplier（内存本就由 target cell_size 缩放预留）②prepare_for_draft/extend **移除全部转换**（虚拟 id 是全量 draft pool 的合法直接索引，只留垃圾 clamp）③`move_accept_tokens` 双侧同公式转换（src 现为原始虚拟 id，单侧转换会 OOB）。**accept 健康基准修正：len 2.2-3.2 / rate 0.24-0.43；~5.9 是死循环病态信号不是健康**。判据：draft 行 `#tokens` 必须 = target 行×dcp（7,421,696 vs 1,855,424），相等=multiplier 又丢了；验证 428/428 压测 0 崩溃。完整链条见 `docs/agent/dcp-virtual-id-domain-fix.md` §6。
- **⛔ smg URL 部署 LoRA 三连坑（2026-08-21 结案）**：①引擎注册键必须用 **path**（smg load 时传 `lora_name=path`，否则运行时 `lora_path` 查不到 → "never been loaded" 400）；②engine `resolve_lora_local_path` 的 cache 目录与 archive 文件名都必须用**派生 key**（`basename-sha1[:8]`），`join(cache_root, URL)` 是非法路径 → curl exit 22 → 400；③`LoRAUpdateOutput` 字段是 **`error_message`**（`message` 取值恒 None）。所有 load/unload 400 必须打响应 body。OSS 下载慢=加速端点 GDS 调度新加坡（0.5-2.4MB/s），**北京直连 host 快 10-30 倍**（9.6-22MB/s）。完整链见 `docs/agent/lora-deploy-400-fix.md`。
- **⛔ smg→engine 死连接复用 = SSE 断流+circuit 熔断（2026-08-22 根治 `a5c3672361`，三次事故）**：uvicorn keep-alive 默认 **5s** 关空闲连接，而 smg 各引擎客户端连接池 idle 50s/300s/default90s——池里的"活连接"服务端已死，复用即流中断（`error decoding response body`）。RL burst→空闲>5s→下波并发同秒 12+ 流齐断→circuit open 81s→503。**修复=全部客户端 `pool_idle_timeout(4s)`（<5s，机制性消灭）**+decode 脚本 `SGLANG_TIMEOUT_KEEP_ALIVE=120`（保险）。判据：同秒多条 SSE 错误=复发；引擎 uptime 活着+half_open 试探成功=连接层（2min 自愈），真挂才不会自愈。新增任何引擎客户端必须带 4s。见 `docs/agent/smg-stale-conn-circuit-fix.md`。
- **⛔ DSpark PD 并发巨型乱码 = 死代码 radix 钳制（2026-08-23 终局 `98e7a09e9e`）**：2026-08-20 写的钳制（prepare_for_extend 读 `req.disagg_spec_algorithm`/`req.is_dspark_transfer` 把 radix 命中钳到 decode_prefix_len）**是死代码**——两个属性的 setter 在某次 merge 中丢失（与 371a991947 draft multiplier 同款病），钳制失效 8 天：并发 300K+ 巨型共享 system prefix → 第二个请求的裸匹配复用第一个的缓存 [0,8192) → 这些 token 不经它的 forward → hidden 永不存在 → decode 报 "PD streaming hidden chunk arrived out of order"。单请求时序永不触发（1P1D 生产 1M 满血从没见过）。**修复五件套**：①create_sender 从部署级信号（state_types 含 PD_HIDDEN，prefill 不带 --speculative-algorithm 它是 decode 侧 flag）设 `is_dspark_transfer=True` 复活钳制；②未 finalize 的 DSpark req → key_limit=0（复用零）；③finalize hold：meta 未到不完成 bootstrap（FAKE warmup 豁免——不豁免会卡死 warmup 503）；④admission gate：pending 的 hidden req 不进 batch；⑤flush 直发回退（piggyback 被 KV 游标跳过时 chunk 永久丢失）。判据：`PDH-CLAMP-CHECK`（钳制存活）、`ooo=0`、双巨型 A/B 17.3s/25.8s 通过。插桩：PDH-WRITE/FLUSH/DIRECT/SEND/RECV + CLAMP-CHECK（SGLANG_DEBUG_DIAG=1）。
- **⛔ DSpark PD 水线锁死 + KV-full assert（2026-08-23 修复 `bddcf0fb34`，1102/1104 V4 Pro case50 部署发现）**：①legacy 模式（`SGLANG_PD_HIDDEN_RECV_WINDOW=0`）下 `_pd_hidden_window_cap` 返回 pool.size，非队头准入条件 `free - w >= W` 数学上不可满足（要求 free ≥ 池容量+w）→ **所有非队头请求永久 park**（实证：U=1 请求 free=393215/393216 下 park 1075s，队头阻塞全队列）。修复=水线只在窗口模式生效（legacy watermark=0）。②`_pre_alloc` 的 "KV cache is full" assert 在并发长上下文 decode 耗尽 SWA 池时**杀掉整个 scheduler**——decode-radix 预驱逐释放错池域（实证：驱逐 82944 token 后 available 纹丝不动 320000）。修复=返回 None + 干净 abort 单请求（503 + hidden rows 释放 + receiver.abort），复用同函数 hidden-invariant fail-fast 模式。**配套部署要点**：SWA 池按流量定容（`--swa-full-tokens-ratio 0.3`，多 300K+ 巨型并发时 0.1 的 1.03M 池必炸）、`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`（[full] 行平衡不可信却会 raise 杀引擎）、窗口 streaming 模式（RECV_WINDOW）有跨请求 chunk 乱序 bug 未修（实证：expected_start=0 收到 chunk_start=32768/row_len=2304 → 82K 次 PDH-PARK 楔死）——**DSpark PD 生产部署一律 legacy 模式**。
- **⛔ DSpark PD 巨型流量残余限制（2026-08-23，case50 实测）**：4×300K+ token 巨型并发下 DSpark draft 路径有 `cudaErrorIllegalAddress`（dspark_draft.py:_run_forward，radix OFF 全量计算时触发；radix ON + 缓存命中降低有效 KV 可规避，40/50 后未复现）；另 DFlash 不支持 grammar-constrained（strict tools/response_format 请求 400 快速失败，case50 50 例中 ~7 例确定性失败，非部署缺陷）。DSpark PD + case50 全净通过在本代码版本不可达，最佳 40/50（7 grammar + ~3 容量）。
- **⛔ DCP decode 概率性乱码双根因（2026-08-22 终局修复 `5974a4fc56`，MoL 乱码家族真身）**：Bug A = `dsa_indexer.py::_localize_index_k_cache_locs` 的 `loc >= pool.size` 直通门控——target 池 size=1.85M 而虚拟 id 从低位发放，**低水位（新进程前几个大请求）index_k 全部未换算写错槽** → indexer 打分错 → topk 选错页；水位越过 size 自愈（=重启减轻+尾部自愈+概率性）。Bug B = trtllm 直通路径 topk `-1` 补齐 lane（local 页数 < index_topk=2048，即 ctx < 524K token）被 `clamp(min=0)` 重定向到 slot 0 无关 KV，**trtllm 对每条 lane 真实 attend 无 mask**（空 rank lse=0 实证）。修复=全域 owner 换算 + -1 lane 重定向本行 lane-0 + OOB 计数移到 clamp 前。A/B 判定链：CP/EAGLE/LoRA 全排除，DCP=1 干净。判据：`SGLANG_DSA_SLOT_OOB_DIAG=1` 计数应≈0。完整见 `docs/agent/dcp-virtual-id-domain-fix.md` §7。
- **LoRA URL 下载多线程加速（2026-08-22 零停机部署）**：`/usr/local/bin/curl` = wrapper（`deploy/curl-parallel-wrapper.sh`）——引擎 `subprocess.run(["curl",...])` 每次 spawn 时 PATH 解析拿 wrapper，**无需重启引擎即生效**。规则：单 https URL + `-o dest` + Range 206 + ≥16MB → **16 线程分段并行**（逐段 retry 5、cat 合并、大小精确校验，任何段失败 exit 22 走引擎自身重试）；其余全部透传真 curl（`--version`/健康检查/小文件不受影响）。实测 438MB：B300-2 20.1s（21.8MB/s，单流 2.3×）、B300-1 25.9s；慢加速端点理论 16×（1MB/s×16）。源文件在 repo `deploy/`，新节点按此部署。
- **⛔ MoL 多 LoRA 乱码 = 四层叠加（2026-08-21 全部结案）**：①cuda graph stale mapping + 驱逐（`f070d3d466`，上游 #29468 移植）；②HiCache host 层 **hash 链缺 lora id**（`d56db6bea5`：`get_hash_str` 根部掺 extra_key，L0/L1 同 prompt 文件 key 相同 → 跨 adapter KV 污染）；③prefill-CP pad 行被施加 LoRA delta（`12cda1cd60`，首 token 乱，pad→-1 哨兵）；④**slot 逐出 churn 毒化在途长请求（`37fae98ed7`，生产"中途数字汤+自愈"的真凶）**——prefer-LoRA-over-base 把 base 钉死 slot0 → 4 LoRA 挤 3 slot 连环互逐 → 在途 350k chunked prefill 的 adapter 被中途换血 → 窗口内 chunk KV 毒化 → 中途数字汤~500 字符自愈。**对照实验判决：4-slot 2乱/5逐出 vs 5-slot 0乱/0逐出（唯一变量）**。修复=删 prefer-LoRA-over-base（base 变真 LRU 冷数据被逐，4 LoRA 共存，零显存代价）+ PR #31608 移植（hooks 时禁 TMA down）。⚠️ VE 模式每 slot ~10GB（5-slot 需 prefill 12M→9M）。**乱码排查铁律：双端全新重启后测；"组合才坏"用 SGLANG_LORA_FAMILY 运行时二分；逐层 dump 行级 NaN 分析是金标准；中途乱+自愈先查 LORA-EVICT 时间线与乱码窗口是否重合**。完整链见 `docs/agent/lora-multi-adapter-garbling.md` §8-9。
- **⛔ decode transfer 风暴 IMA = transform_index decode kernel 缺上界 mask（2026-08-19 修复 `8bfe264118`）**：`transform_index_page_table_decode_fast` 只 mask `topk<0`（prefill 孪生 kernel 是双界）；40+ 并发 KV 传输风暴下 indexer 输出竞态越界 → `page_table_ptr+topk` OOB → IMA → NCCL 连锁整组崩。EAGLE draft 走 `dcp_disabled()`（非分片池）正好命中该 else 分支，所以只有 decode 崩。复现判据：40×11K token 并发 burst 必崩一个 decode；修复后 40/40 过。**相关：v15_patched 的 `system_packages.pth` 会暴露系统 flashinfer_cubin 0.6.12 给 venv flashinfer 0.6.14 → trtllm kernel IMA（删系统 cubin + 同步完整 flashinfer 88M data 目录）**。**2026-08-29 补记（1102/1104 GLM-5.3 乱码形态，详见 docs/agent/1102-1104-flashinfer-venv-garble.md）**：同一泄漏的**静默错乱形态**（不 IMA 崩溃）——首 SSE 正常后卡半天输出 token 汤乱码 + accept rate 0.03-0.06（病态，基准 0.24-0.43）+ gen throughput 减半；1102 的 venv flashinfer/data 只有 75M（缺 13M）加剧。修复同法（删系统 flashinfer 0.6.12 + data 补齐 88M + 配对重启），验证 accept 0.32-0.45 / 吞吐 2.5×/无乱码。**乱码+accept<0.1 组合先查 venv 依赖版本错配（pip list 对比系统与 venv 同名包）再查代码**。
- **⛔ BF16 MoL PD LoRA 崩溃 = 脚本 flag 缺失（2026-08-19 结案）**：1021 `/root/start_glm52_bf16_pd.sh` prefill 块曾缺 `--lora-use-virtual-experts` → 走**无加固 classic `fused_moe_lora` kernel**（expert_id 无 clamp）→ 首个 LoRA 请求 CUDA IMA。20h 排查被"单机过/PD 崩"误导——单机测试脚本全带 VE flag 走加固过的 VE 路径，对照从设计上无效。**教训：对照实验前先逐条 diff 两边启动 flags**；空续行 `\` 是 flag 被删的物证。LoRA 正确入口是请求体 `lora_path` 字段（colon 语法 `model:xxx:L0` 不生效）。**单机 CP×LoRA 残余已在 1102 结案（commit `c1946917cb`）：csgmv batch_info 与 CP shard 行数错配（perm 覆盖全量 token、x 只有 1/cp 行 → OOB），layer-0 gate GEMM/DSA topk 都是 sticky IMA 浮出点；修复=按 parallel runtime 重建 shard 视图（round-robin 模式 metadata 为空对象，keying on it 会静默失效）**。完整链见 `docs/agent/cp-lora-crash-investigation.md`。
- **⛔ V4 Pro DSV4 高水位跨请求 KV 污染 = guard 用 SWA 尺寸当 FULL 池 cap（2026-08-23 终局修复）**：`decode.py::_guard_kv_indices` 三处调用点传 `token_to_kv_pool_allocator.size`——hybrid-SWA composite `SWATokenToKVPoolAllocator.size == min(full, swa)` = **SWA 尺寸**（V4 Pro=1,438,464），而 full 池虚拟 id 域是 4,794,880。分配水位一旦越过 SWA 尺寸（≈3 个 300K giant 后），所有 `id ≥ swa_size` 的合法目的地 id 被 guard 判"garbage" **静默洗成 0**（padding sink 页）→ KV 传输写进共享页 0、请求也从页 0 读 → **跨请求互读互写 = 确定性内容污染**（probe 稳定返回别的 case50 内容、accept=1.00 draft/target 同错、[full] leak 608 次）。修复=三处 cap 改 `size_full`（composite 有该属性，基类恒等 `.size`）。**判据**：`grep KV-PRODUCER-OOB decode_v4.log` 应为 0；DCP-PD-IDX 的 page_indices 无 0（修复前 giant 1564 页里 1444 个 0、probe 整段全 0）；`samples=[1438464,1438465,...] cap=1438464`（=SWA 尺寸精确咬合）是本病的指纹。验证：ladder 4 级 20/20 clean + case50×2 @600rpm 42/50+8 grammar、两轮 5/5 clean、泄漏 0、accept 3.15-3.75。**教训：`allocator.size` 在 SWA composite 语义是 min(full,swa)，任何拿它当 full 池 id 上限的守卫都会在高水位静默腐蚀**。
- **✅ DSpark PD 窗口模式（RECV_WINDOW）彻底修复（2026-08-24，四层根因全部结案）**：此前"窗口 streaming 有跨请求 chunk 乱序未修 bug、生产一律 legacy"的结论被推翻——R1 事故是四层叠加：①**start 脚本 export 误置于 launch 之后**（窗口模式从未真正生效，"验证过的 legacy"其实一直在跑）；②**控制面 zmq PUSH 永久黑洞**（common/conn.py `_connect`：`RECONNECT_IVL=-1` 不重连 + monitor 只订 EVENT_DISCONNECTED → 断线轮换后新 socket 连接失败时永不重试永不轮换，L4 并发下 5/8 rank 对集体黑洞，notify 全丢）→ 修复=RECONNECT_IVL=100 + monitor 双事件（CONNECTED|DISCONNECTED）按最后事件判定，已自愈的 socket 不再误轮换；③**notify 丢失无自愈**（LINGER=0 轮换本身丢在途消息）→ 协议层三件套：发送侧 park-for-ack 时 3s 重发 notify（`pd_hidden_renotify_args`+renotify loop）+ 接收侧去重补 ACK（重复 chunk=`end<=next_start` 静默重发 ACK）+ 缺口等待（gap 时 push_back 等 renotify，60s 不愈才响亮失败，不再秒崩）；④**park-behind 唤醒竞态**（inflight 读与 room_waiters 追加非原子，missed-wake 孤儿 chunk）→ `park_chunk_behind_room` 原子 check-and-park + `finish_streaming_chunk` 单锁段 pop+wake。**abort 风暴集体漂移**（6 请求同时 300s 超时中止 → `_padded_all_reduce_min` FIFO 错位楔死，py-spy 取证 `/root/deadlock_dump_win2_*.txt`）修复后未再复现：21 请求同时强制中止实测存活。**验证矩阵**：ladder×3（冷/热树）全 20/20、case50×4 轮全 42/50+8 grammar 5/5 clean、21-abort 风暴存活、**523,099 次集体调用×8 rank 位点序列完美奇偶**（PADDED-AR 插桩，`SGLANG_DEBUG_DIAG=1` 时每调用打点，生产默认关）。**判据**：`grep "PDH-GAP\|gap did not heal\|out of order" decode_v4.log`=0；`/tmp/padded_seq_diff.py` 位点序列全 rank 一致；`grep PADDED-AR` 有输出=插桩存活。**教训：R1"窗口模式有 bug"实为双实验叠加+脚本 env 位置错误的冤案；控制面 socket 的"快速死节点检测"设计（不重连+LINGER=0）本质有损，关键控制消息必须有应用层重发+幂等去重**。

- **⛔ mooncake 控制通道 torn frame = 全集群楔死根因（2026-08-24 结案）**：`common/conn.py::CommonKVManager._connect` 返回**跨 transfer_worker 线程共享的缓存 PUSH socket 但 send 无锁** → 多线程并发 `send_multipart` 帧交错（libzmq 不跨线程串行化 multipart，pyzmq 帧间释 GIL）→ 3 帧通知被劈成畸形 2 帧 → decode 端只启动一次的控制线程 `decode_thread` 解包 `ValueError: expected 3, got 2` 死 → 各 rank 集体调用计数漂移 → gloo 永不完成 → **全集群楔死（health 200、零 crash）**。修复=每 endpoint 锁 + `_send_multipart`（对齐 receiver classmethod `_connect` 既有 `(sock, lock)` 契约，prefill 发送侧漏补）。此前 AGENTS 记为"畸形 2 帧来源未知"即此病。判据：`grep MOONCAKE-RECV-DROP`=0、`PADDED-AR-FAIL`=0。完整见 `docs/agent/pd-stability-fixes-2026-08-24.md` §1。
- **⛔ DSPark decode all_gather 跨 rank batch 分歧 → NCCL 挂起崩溃（2026-08-24 防护）**：`deepseek_v4_dspark.py::_apply_step_logits_sharded` 的 `attn_tp_group.all_gather(dim=-1)` 要求所有 rank 基准 batch 维一致；decode 巨型并发偶发某 rank schedule batch 与同组其他 rank 分歧（某 rank 先完成/移除一请求）→ shape 不匹配 → NCCL collective 挂起 → `Fatal Aborted`。防护=propose 前 `_assert_batch_bs_rank_invariant`（scalar `all_reduce(MAX/MIN)` 让**所有 rank 一致 raise** `DSParkBatchDivergence`→跳过该 batch，分歧时不进 all_gather，NCCL 不挂）。⚠️ sglang `GroupCoordinator.all_reduce` 只收 `(input_)` 不收 `op=` 关键字（固定 SUM），必须用 `torch.distributed.all_reduce(..., op=MAX/MIN, group=gp.device_group)`。判据：`grep DSPARK-BATCH-DIVERGE`=0。完整见 `docs/agent/pd-stability-fixes-2026-08-24.md` §2。
- **⛔ decode `_pre_alloc` SWA 预算/分配口径分裂 → 假性"握手失败"风暴（2026-08-25 终局修复）**：decode radix 命中（prefix_len>0）时旧代码 `uses_swa_tail = ... and prefix_len == 0` 强制走 `alloc_extend`（SWA **全量页**），而准入预算按 SWA **尾窗**算——`--swa-full-tokens-ratio 0.3` 下高并发 SWA 池必耗尽 → `_pre_alloc` None → `prepare_abort`+`kv_receiver.abort()` → **AbortReq → prefill 报 "bootstrap failed ... Aborted by AbortReq"**（症状酷似超时/传输 bug，实为 SWA 分配 bug）。修复=`prefix>0` 也走 `alloc_extend_swa_tail`（传真实 `total_prefix_len`/`delta_len`/`min(swa_tail, delta)` clamp；窗口在序列末尾、部分命中时窗口完全落 delta 内，语义安全）。判据：`PD-PREALLOC-KV-FULL`=0 且 `Aborted by AbortReq`=0。**排障铁律：Aborted by AbortReq 先 grep PD-PREALLOC-KV-FULL/hidden-pool invariant 定位 abort 发起者，不要先怀疑超时**；prefill 重启后 decode 必须配对重启（mooncake 会话失效同样全报 Aborted by AbortReq，曾误诊 2 小时）。完整见 `docs/agent/swa-prealloc-budget-mismatch-fix.md`。
- **⛔ 深测入口与本地代理陷阱（2026-08-25）**：deepseek 1P1D 深度测试**直接起 sglang-router**（`sglang-router launch --pd-disaggregation --prefill ... --decode ... --policy round_robin --port 30000`，api-key 见节点 `/root/start_pd.sh`），**不需要 mol proxy / smg / cache_aware**。本地访问走 SSH 隧道 + `curl --noproxy '*'`（本地 HTTP 代理劫持 localhost → 502 假"不通"）。smg/sglang-router 收 chat 请求 hang 时先查引擎端（PD 配对/mooncake 会话），入口层缓存健康检查 300s 恢复期可用重启 router 清除。

## 9. 知识文件索引（docs/agent/）

| 文件 | 内容 |
|---|---|
| `docs/agent/cp-lora-crash-investigation.md` | **CP×LoRA / PD LoRA 崩溃调查全记录（2026-08-18/19）**：kernel 三层加固（VE 双界钳制/CP gathered 尺寸/SEG-CLAMP 消毒，commit `16a7a569df`）、load_stream 竞态 sync-load 修复、**PD 崩溃最终结案 = prefill 脚本缺 `--lora-use-virtual-experts`（classic 路径无加固）**、单机 CP×VE 残余（另案）、方法论教训（对照实验先 diff flags） |
| `docs/agent/cp-pad-row-garble.md` | **CP round-robin split pad 行 LoRA 乱码结案（2026-08-22，`f1421f2241`）**：非整除 extend 被 pad 后进 CP 分片 → pad 行经 LoRA segment 路径污染真实行（有限值错误 token：参数路径损坏/模板 token 循环）；+1-2 字符消失、全缓存命中也乱、并发混合批次 ~30%、base 免疫；修复=gate 加 `seq_len % cp_size == 0`；含录制代理 + 并发对撞复现方法论与乱码检测启发式 |
| `docs/agent/v4-pro-deploy.md` | **V4 Pro 0813 通用部署文档（2026-08-24 重写）**：面向新集群的可复用操作手册——剔除集群专属 IP/端口/密钥（占位符标注），保留当前生产完整 env + 启动参数（prefill/decode/router 三端）、模型/config 关键值、第三方重打包 checkpoint 识别（HTTP Range 验证法）、OTel 采集要点、部署步骤、验收断言、已知行为 |
| `docs/agent/b300-compile-fixes.md` | 三个编译层修复：DeepGEMM sm_103a nvcc wrapper、tilelang flock 跨进程锁、tilelang CCCL include |
| `docs/agent/dspark-pd-deadlocks.md` | DSPARK PD 五个死锁修复链（专用 gloo group / 异常安全 poll / prefill 空分支补偿 / spec_info DSPARK 分支 / hidden materialize），py-spy 死锁形态识别方法 |
| `docs/agent/dspark-pd-stuck-req-postmortem.md` | **hidden pool 个别请求永久卡死完整技术报告**（2026-08-18，`1c9e1c3275`）：bootstrap 先于 hidden alloc 的时序倒置死锁环 / prefix-free 上界预留先于 bootstrap（背压替代死锁）/ park & drain 非阻塞释放（wait_ack 是 decode 本地 CUDA 事件等待）/ 不变量证明 / 残留边界与验证判据 |
| `docs/agent/decode-radix-swa.md` | DSV4 PD 双端 radix 全链路：为何 DSpark 必须两端同开、`swa_served_from_tree=False` 设计（树不持有 SWA）、四层 bug 修复链（hidden clamp / len-1 match / SWA validator / SWA 池泄漏）、诊断 env 与验证数据 |
| `docs/agent/dsv4-cp-dspark.md` | DSV4 prefill CP + DSpark PD hidden 共存：三层修复链（hook flag 顺序 / aux hidden all-gather 重组 / decode_engine_rank 对角配对防 8 倍重复发送）、flag 语义辨析（V4 用 interleave 非 GLM layersplit）、SGLANG_DEBUG_DIAG 日志 gate、3.5× 长上下文实测数据 |
| `docs/agent/dsv4-pro-pd-engineering.md` | **V4 Pro PD 工程总纲报告**：DSPARK PD 分离支持（#31466 hidden 传输 + spec_info 分支 + merge_batch 守卫）、双端 radix（swa_served_from_tree 协议）、CP=8 共存（3.5×）、死锁三连、KV 广播正确性、乱码战争摘要、编译基础设施、性能汇总、残余项边界 |
| `docs/agent/dsv4-radix-nondet-postmortem.md` | 乱码/循环战争完整 postmortem（2026-08-16，12h）：分层根因（dup 双 free -139520 逐位归因 / SPLIT 池状态污染三件套 / c128 边界字典快照-恢复 / 三个静默失败）/ FORCE_MISS 定音 kernel 级非确定 / 冷热三轮 0 短循环验证 / 可复用判据工具（nondet.py、longloop_scan、ALLOC-DIAG 插桩）/ 经验教训 |
| `docs/agent/pd-hidden-window-design.md` | **PD hidden 接收池窗口化准入设计与形式化证明**（2026-08-18，`1910bb619c`）：606s 长请求楔死事故根因（准入需求=整池 min(U,P)，任何并发持有者→永久静默 park）+ rank 本地 alloc 分歧（OB'部分bootstrap）→ 窗口 W=min(U,W) + 记账式准入（charge 只随同步事件变化）+ 水位 φ=W 队头豁免；定理 OB/K1/S1/S2/L1-L3/C1/C2/R1/B1；env `SGLANG_PD_HIDDEN_RECV_WINDOW`（0=旧行为 kill-switch）；日志判据 [PDH-PARK]/[PDH-ADMIT] |
| `docs/agent/dcp-virtual-id-domain-fix.md` | **DCP 虚拟 id 域 + draft pool 尺寸双重修复（2026-08-20，`9db63a6abb`+`371a991947`）**：§1-5 = 虚拟 id 域（page_size=64×dcp, capacity=size×dcp）下双转/sanitize/无 rank 过滤的修复；**§6 = 终局根因：merge v0.5.16 丢 draft_pool_token_multiplier → draft pool 缩到 1.85M → 虚拟 id 越界 → 压缩 workaround 摧毁 draft 域 → accept 0.07**。修复=draft pool 恢复 size×dcp + 移除全部转换 + move_accept_tokens 双侧转换。**accept 健康基准：len 2.2-3.2；~5.9=死循环病态**。判据：draft #tokens=target×dcp、DRAFT-LOC-OOB=0 |
| `docs/agent/2p3d-cluster-config.md` | **2P3D（1P2D）集群完整配置手册（2026-08-20）**：拓扑/IP/venv/二进制路径（smg=/usr/local/bin/smg、mol-stack proxy）、prefill/decode/router/gateway/proxy 全部实际启动命令与 env、部署与重启流程（prefill 重启后 decode 必须重启配对 RDMA）、LoRA 管理 API、验证判据命令、已知坑（start_pd.sh 过时路径、MOL_UPSTREAM_RUNTIME=sglang 必须显式、pkill -f 禁令、模型名两层） |
| `docs/agent/lora-multi-adapter-garbling.md` | **MoL 多 LoRA 乱码修复结案（2026-08-21，`f070d3d466`）**：第 5 个 uid（base+4 adapter）超 max_loras_per_batch=4 触发 LRU 驱逐 → cuda graph token_lora_mapping 尾部 stale id + rank-0→slot-0 → 乱码 sticky 传染（上游 #29157/#29468 同源）。含复现序列/修复/验证数据/方法论教训（污染状态毁对照实验、"输出≠decode 应用 LoRA"判据、lora_pool_slots_used 验证法）与判据工具 |
| `docs/agent/lora-deploy-400-fix.md` | **LoRA URL 部署 400 双 Bug + 训练任务接口（2026-08-21）**：smg load 无超时/DELETE 30min 挂起/引擎注册键(name≠path)错配；engine 侧 URL 拼 cache 目录与 archive 文件名非法（`join(cache_root, URL)`）→ curl exit 22；`LoRAUpdateOutput` 字段是 `error_message` 非 `message`；FanOutCommunicator 类型过滤加固。含 smg jobs 训练任务 API（token id+logprob 三元组对齐）、OSS 下载慢根因（GDS 调度新加坡双向跨境 0.5-2.4MB/s vs 北京直连 9.6-14.6MB/s）与四端点测速、阿里工单 request id |
| `docs/correctness-war-retro-2026-08.md` | **正确性战争人类可读复盘（2026-08-18~22）**：面向全员（含非引擎同学）的完整叙事——LoRA+CP 四层、draft 崩溃/accept 断崖/虚拟 id 域、DCP 双 bug 终局，含方法论与弯路记录。新人了解系统架构（PD/KV/TP/CP/DCP/DSA/EAGLE/MoL 逐个白话解释）的入口文档 |
| `docs/dsv4-dspark-pd-tech-report-2026-08.md` | **V4 Pro DSpark PD 技术报告（2026-08-23）**：对外展示版——hidden state 流式动态传输算法（搭车+直发回退+ACK 流控+窗口化准入含形式化证明）、双端 radix、CP=8 3.5×、bug 排查方法论（内容金丝雀/判别阶梯/数字指纹）与跨请求 KV 污染终局案例、最终验收数据与残余边界 |
| `docs/agent/prefill-oom-accounting.md` | **Prefill "OOM" 记账分裂（2026-08-22 结案 `6504ba9b71`）**：extend_range.end vs fill_ids 长度 vs prefix_indices 三源撕裂 → alloc 按虚高 seq 算页数杀全组（数学证明+abort 理论否定记录）；evict(token域,free+release) vs alloc(page域,仅free) 口径统一；判据 EXTEND-ACCOUNTING-DIVERGENCE / PREFILL-ALLOC-FORENSICS |
| `docs/agent/pd-stability-fixes-2026-08-24.md` | **PD 稳定性修复双根因（2026-08-24）**：①全集群楔死=mooncake `_connect` 无锁并发 send 帧交错 → torn 2 帧 → 控制线程崩 → 集体分歧（per-endpoint 锁 `_send_multipart` 修复）；②decode 偶发崩溃=DSPark all_gather 跨 NCCL rank batch 分歧（rank-invariant 检测+跳 batch 防护）；附带请求终态保障确认（既有代码，6 轮×50 压测 0 卡死/0 无进度/0 中途失败，24min 双端监控全绿）；铁律见 AGENTS §5.4 |
| `docs/agent/swa-prealloc-budget-mismatch-fix.md` | **decode `_pre_alloc` SWA 预算/分配口径分裂（2026-08-25 终局）**：radix 命中(prefix>0)走 alloc_extend 要求 SWA 全量页 vs 预算按尾窗 → 0.3 ratio 池耗尽 → abort → AbortReq → prefill "bootstrap failed"（假性握手失败风暴）；修复=prefix>0 也走 SWA 尾窗（窗口在 delta 内语义安全）；另含 prefill 重启后 decode 未配对重启的事故复盘（mooncake 会话失效全报 Aborted by AbortReq，曾误诊 collective 死锁 2 小时）、PADDED-AR count 诊断正误用法、深测入口规范（round_robin router + noproxy） |
| `docs/agent/glm52-tpot-decode-optimization.md` | **GLM-5.2 MoL DCP=8 decode TPOT 优化（2026-08-27 结案）**：AllReduce Fusion（去 `--enforce-disable-flashinfer-allreduce-fusion`，-4.6~5.5%）+ tile 对齐 `BLOCK_SIZE_M=64→16`（B200 精确 tune + down TMA，**-8.8%**，decode verify 累计 57.9→50.3ms = **-13.1%**）。含 M 覆盖全维度锁定（18 标准 key 中 BLOCK_M=16 覆盖 1~512，非标准 M 40/72/88/104 fallback 后仍 BLOCK_M=16）、判定不做方向（非标准 M tune 显存不足/上游设计；FP8 AllReduce 负收益已回退；Elementwise 分散 ROI 低）、Chrome trace dur 单位 us 陷阱、`.item()` CPU 算子误判排除。**方法论：同 bs/seq_len 严格 A/B 才可信，grep 混批次均值会误判** |
| `python/sglang/kernels/ops/attention/dsa/fused_dsa_quant_store.py` | **PR#5 合并（2026-08-27, san-tian）**：DSA 专用融合 triton kernel——per-block fp8 量化(k_nope 512) + bf16 保留(k_rope 64) + 混合 layout 直接 paged store，替代 `quantize_k_cache_separate()` + `set_mla_kv_buffer_triton()` 两步（省中间 tensor + 一次 launch）。接入点 `memory_pool.py::_write_mla_kv_buffer` 的 `dsa_kv_cache_store_fp8` 分支（GLM-5.2 走 `GlmMoeDsaForCausalLM`→`DSATokenToKVPool`→use_dsa+fp8_e4m3 触发）。**合并时两处修正**：①量化公式必须与 `quant_k_cache.py::_quantize_k_cache_fast_kernel` **逐行一致**（`y * (1.0/y_s)` 而非 `y / y_s`）——浮点舍入不同则**非字节级无损**（B300+triton 3.6.0 实测 nope diff）；②`test_correctness` 的 `loc` 必须**无碰撞**（`randperm` 而非 `randint`）——128 token→256 slot 随机 `randint` 有 ~27 个碰撞槽，碰撞槽"最后写入者"在 torch 串行 vs triton 并行间不一致 → test 假失败。已验证：unique loc 下 nope/scale/rope 三区域 0-byte diff（无损）；shape 冒烟 view(-1,512)+view(-1,64)+paged store 通过。**收益：博客自承端到端 +0.9%（噪声级），核心增益来自 allreduce fusion 非此 kernel** |
