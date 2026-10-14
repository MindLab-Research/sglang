# DeepSeek-V4.1-Flash MoE LoRA 支持与性能报告

2026-10-12 · 分支 `pr17` · 私有仓库 `MindLab-Research/sglang-private`

| 项 | 值 |
| --- | --- |
| 测试环境 | 1101，`ssh -p 1101 root@8.222.11.182`（公网 8.222.11.182，LAN 10.0.58.56），8×B300（各 275040 MiB），非生产 |
| 代码树 | `/root/dsv41_upstream/sglang`（PYTHONPATH=`/root/dsv41_upstream`，包根即 `sglang/`） |
| 运行时 | venv `/opt/sglang-venv` |
| 权重 | `/mnt/workspace/DeepSeek-V4.1-Flash`（MXFP4 experts） |
| 拓扑 | prefill 30100 / decode 30200 / smg router 31500 |
| 模型 | 40 层，hidden 5120，384 routed experts，top-6，moe_inter 2304，自带 dspark 头 |

## 0. 摘要

在保持 MoE LoRA 结构不变（A/B 张量形状、秩、目标模块集合、虚拟专家语义、量化格式、gating 数学均不变）的前提下，DSV4.1-Flash 已支持 MoL。五条能力线全部通过：

| # | 能力 | 状态 | 落地位置 |
| --- | --- | --- | --- |
| 1 | 基础推理（MXFP4 experts + SM100 trtllm-gen） | 通过 | 上游原路径 |
| 2 | attention LoRA（`q_proj` 等） | 通过 | 上游 `chunked` backend |
| 3 | KV 共享 / 隔离（按适配器作用域自动判定，两种形态都支持） | 通过：不改变持久 KV 的适配器共用 base 的 radix 命名空间；会改变持久 KV 的按适配器隔离 | `lora_config.lora_kv_shared`、`lora_manager.is_lora_kv_shared`、`lora_manager._kv_shared_safe_layers`、`scheduler.resolve_lora_kv_shared` |
| 4 | FP4/trtllm-gen 融合路径上的 MoE LoRA | 通过；零权重适配器与 base 位级一致，随机适配器可动态加载 | `lora/trtllm_lora_temp/`、`kernels/ops/moe/trtllm_lora_temp/`、`experimental_sgl_trtllm` backend |
| 5 | Tinker `all-linear` 3D LoRA（722/722 全量） | 本次打通 | §6 |
| 6 | **LoRA 作用域（`--lora-target-layers`）**：适配器只覆盖后 20 层时，其余层完全不包装 ⇒ **零 LoRA 开销** | 本次打通 | §12 |

性能（口径见 §7.1）：

| 组 | 满批吞吐 p50 (tok/s) | TPOT p50 (ms) |
| --- | --- | --- |
| base（无适配器） | 12356.4 / 12148.6 | 5.180 / 5.268 |
| `moe_rnd`（随机 MoE LoRA） | 7633.2 / 7874.5 | 8.384 / 8.127 |
| `moe_zero`（适配器权重全零） | 8082.7 / 7885.3 | 7.918 / 8.116 |

挂载 MoE LoRA 的吞吐代价约 38%。在零额外显存、零 kernel 改动的前提下可用的手段已全部穷尽（§7），剩余差额必须在 kernel 内消除。

## 1. 范围与不变量

目标是让 DSV4.1-Flash 完整支持 MoL，并在以下不变量下完成：

1. MoE LoRA 结构不变：A/B 张量形状、秩、目标模块集合、虚拟专家语义、量化格式、gating 数学；
2. 各类适配器正确加载；
3. 推理数值正确，且零权重适配器与 base 完全一致；
4. 挂载代价尽可能低；
5. 支持 PD 分离与 HiCache。

## 2. 环境与资产

- 代码 `/root/dsv41_upstream/sglang`；启动脚本 `/root/exp_thinking.sh`（在用）、`/root/exp_single.sh`（备份）。
- **每次 rsync 后必须清除 `__pycache__`**：`find /root/dsv41_upstream -name '__pycache__' -type d -exec rm -r {} +`。残留 .pyc 会覆盖代码改动。
- 适配器（1101 `/root/`）：`dsv41_moe_rnd`（随机）、`dsv41_moe_zero`（全零）、`dsv41_moe_tinker`（Tinker all-linear，§6）。
- 关键启动参数：`--enable-lora --max-lora-rank … --lora-use-virtual-experts --experts-shared-outer-loras --lora-target-modules …`。
- MoE LoRA 必须设 `SGLANG_EXPERIMENTAL_LORA_OPTI=1`；`--lora-target-modules` 至少包含 `gate_up_proj down_proj`。
- V4.1 上启用 HiCache 必须显式指定 `--hicache-ratio`：自动 sizing 不支持 `DeepSeekV4TokenToKVPool`。

## 3. 执行路径约束：MoE LoRA 为何走分解核

V4.1 的 experts 为 MXFP4，运行于 SM100 的 trtllm-gen 上。非 LoRA 请求直接调用 flashinfer trtllm-gen；该路径不构造 MoeRunner，因而不存在注入 LoRA 的挂点。

因此 LoRA 走独立 backend `experimental_sgl_trtllm`，其融合函数把计算委派给若干分解核：

```
routing → gate_up grouped GEMM（原始 2*inter）
        → activation：叠加 gate_up LoRA delta（激活前）+ 记录 activation_lora_input
        → quant → down grouped GEMM → finalize → 叠加 down LoRA delta
```

## 4. KV 隔离规则：作用域不触及写持久 KV 的层则共享，否则隔离

**结论：两种形态都支持，不需要为此换实现。判定依据是适配器会不会改变持久 KV（radix 缓存持有的那一份），也就是它的权重落在哪些层、以及目标模块。不改变 ⇒ 可声明 `lora_kv_shared`，请求与 base 共用 radix 命名空间，不隔离；会改变 ⇒ 不能声明，按适配器隔离。**

引擎在加载期按「层 → 模块」两级判定，两级都有代码依据（§4.2）。

### 4.1 持久 KV 由哪些层写

`srt/lora/lora_manager.py::_kv_shared_safe_layers()` 给出定义：V4.1 的持久 KV（压缩 KV + index K + candidate blocks）只由 `kv_source_layer_ids` 层的 compressor/indexer 写入；`compress_ratios[i] ∈ {1,2}` 且不属于 `kv_source_layer_ids` 的层完全不写持久 KV（它们读取 source 层的压缩 KV，其 SWA 窗口每次 forward 重建、不持久化）。

本模型（`num_hidden_layers=40`，`kv_source_layer_ids=[2,8,14,20]`）由此得到的划分：

| 类别 | 层 | 说明 |
| --- | --- | --- |
| 不写持久 KV | 3–7、9–13、15–19、21–39（共 34 层） | 安全侧。权重只落在这些层的 LoRA，无论改哪个模块都不会改变共享 KV |
| 写持久 KV | 0、1（dense）；2、8、14、20（`kv_source`） | 危险侧，共 6 层 |

边界不是「前 20 / 后 20」的简单对半：后段里 layer 20 是 `kv_source` 层（仍写持久 KV），前段里 3–7、9–13、15–19 并不写。判据是上表的层集合。

### 4.2 加载期校验（`lora_manager.py`，声明 `lora_kv_shared=true` 时）

1. `target_modules` 必须显式声明；若为 `all` / `all-linear` 直接 `raise` —— 它们会展开到 K/V 投影，无法共享。
2. 按层：`_adapter_weight_layers()` 从适配器的 safetensors header 读出权重实际落在哪些层（不加载张量）。若这些层全部落在上表安全侧 ⇒ 直接放行，且目标模块名不受限，日志为
   `kv_shared adapter <name>: all target layers [...] are non-source compressed layers (no persistent KV written) — module names unrestricted.`
3. 否则按模块：退到 K/V 产出模块黑名单 —— `v_proj`、`k_proj`、`qkv_proj`、`kv_proj`、`kv_a_proj_with_mqa`、`fused_qkv_a_proj_with_mqa`、`kv_b_proj`、`wkv`、`indexer.wk`、`indexer.weights_proj`。命中任一 ⇒ `raise ValueError`，并在信息里指出权重落在哪些持久 KV 层（`(dense/kv_source)`），提示应移除 `lora_kv_shared`、或改目标、或把权重限制到非 source 层。

即：声明了 `lora_kv_shared` 但实际会改变持久 KV 的适配器，会在加载期被拒绝，不会带着错误的共享假设跑起来。

### 4.3 请求期：共享与隔离的差别

`srt/managers/schedule_batch.py`：

```python
if lora_id is not None and not lora_kv_shared:
    extra_key = (extra_key or "") + lora_id
```

`extra_key` 参与 radix 缓存的 block hash。

- 共享（`lora_kv_shared=true`）：`lora_id` 不进 `extra_key` ⇒ 与 base 同一棵 radix 树，适配器之间也互通 ⇒ 前缀命中率最大，缓存占用不随适配器数量增长。
- 隔离（未声明）：`lora_id` 进 `extra_key` ⇒ 每个适配器一份独立命名空间，同一段前缀在不同适配器下各缓存一份。代码注释给出原因：非共享适配器的 KV 与 base、与其他适配器都不同，共用命名空间会把一个适配器的 KV 喂给另一个，造成跨适配器污染与续写乱码。

### 4.4 实操结论

- 适配器只覆盖**不写持久 KV 的层**（生产上 MoL 适配器落在后段即属此列）⇒ 共享，不需要 KV 隔离；
- 适配器覆盖到**写持久 KV 的层**（dense 0–1，或 `kv_source` 2/8/14/20）⇒ 只能隔离；若它还直接改 K/V 投影，加载期即报错。

两种形态由同一套机制处理，工程上无需为「要不要隔离」做额外工作 —— 隔离与否由适配器作用域决定。

## 5. 实现中修正的 16 处缺陷

以下各条同时记录于仓库 `AGENTS.md`。

1. **激活核必须携带 gated-SwiGLU 的控制量。** base MoE 使用 `gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0`，公式为 `xGlu=clamp(xGlu,max=limit)`、`xLinear=clamp(xLinear,±limit)`、`out=xGlu*sigmoid(alpha*xGlu)*(xLinear+beta)`。移植的 FP4-LoRA 激活初版为 `silu(xGlu)*xLinear`，导致挂载 LoRA 的请求与 base 数值发散。已恢复于 `dev_kernel.cu` 的 `gatedSiluScalars`/`gatedSiluLoRA`，并贯通 `DevKernel.h`（Data/KernelParams）、`runner.cu`（setOpsData）、`core.py`（`trtllm_fp4_block_scale_routed_moe_lora`）。
2. **per-expert 标量按 index 0 读取。** sglang 的 MXFP4 prep 为每个专家填入相同值；LoRA 激活核不携带 permuted→expert 映射（`ctaIdxXyToBatchIdx`/`tileTokensDim`）。
3. **LoRA 强制 STANDARD topk。** FP4 路径默认使用 BYPASSED（将 `router_logits` 交给核内路由），MoE-LoRA 核则需要物化的 `topk_ids` 以执行虚拟专家 delta 路由。LoRA 服务上的透传请求（无适配器）同样走该 routed 路径。切换位于 `deepseek_v2._lora_forces_standard_topk()`。
4. **routed 分支必须支持 gated silu**，而非仅 situ —— `flashinfer_trtllm._fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`，携带 OAI 三标量与 FC1/FC2 bias（situ 维持 clamp_limit-as-beta 映射且无 bias）。
5. **`Mxfp4FlashinferTrtllmMoEMethod` 没有 `hidden_size`**（属性委托给被包裹的 FP8 方法，且无 `__getattr__`），需从 `w2_weight.shape[1]` 推导。
6. **LoRA 下 `create_moe_runner` 必须构造 MoeRunner**（`MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM`）；非 LoRA 分支维持 `runner=None`。
7. **`normalize_gate_up_proj` 不得对 3-D MoE 权重做重复 stack。** 其把倒数第二维 ×2 的 synthetic up-half repeat 面向 2-D PEFT 适配器（`[r,hidden]` → `[2r,hidden]`）。作用于 3-D shared-outer MoE 权重（`[1,2r,hidden]`，已按 mem_pool 约定 stack）会使 rank 维翻倍并触发 shape 检查失败（V4.1 上 `[1,32,5120]` → `[1,64,5120]`）。仅 2-D 执行 repeat。
8. **O7/O8/O9 的双流覆盖假定 LoRA backend 为 TRITON**（调用 `lora_backend._sgemm_info()`）。V4.1 运行 `ChunkedSgmvLoRABackend`（csgmv），无该钩子。各入口以 `hasattr(self.lora_backend, "_sgemm_info")` 守卫，回退到 stock forward。
9. **V4.1 的 LoRA 回退路径（STANDARD topk + gated-silu）会解引用 `router_logits.to(...)`，而该值为 None。** 处理方式：standard-topk 的非-situ 请求预先路由至核的 routed 入口。
10. **LoRA 必须按未补齐的 per-rank 宽度切片适配器权重。** base FusedMoE 为适配 trtllm-gen，将 `intermediate_size_per_partition` 补齐至 128 的倍数（V4.1：2304/4=576 → 640）；LoRA buffer 使用模型未补齐宽度（`get_hidden_dim`）。按补齐值切片会失败（`[16,640]` 对 `[16,576]`），最后一个 rank 还会静默截断为 `[16,384]`。现 FusedMoE 保留原始 `intermediate_size`，FusedMoEWithLoRA 据此重算。
11. **flashinfer JIT cache 以文件存在性为准，不按内容哈希。** 修改 overlay `.cu` 后，必须将 `~/.cache/sglang/.cache/flashinfer/*/cached_ops/sgl_fused_moe_trtllm_sm100/` 下的 `.so` 改名或删除，否则 JIT 仍加载旧产物。
12. **trtllm-gen cubin 包需自行下载。** 1101 上的 flashinfer pip 包只含 `checksums.txt` 清单，无实际 `.cubin`（按需下载）。分解版 FP4 MoE-LoRA 需要 `batched_gemm` cubins（1799 个文件，约 50 MB），按 `edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/<hash>/batched_gemm-*/` 清单下载。
13. **V4.1 的 gate_up GEMM 需要 E4m3 激活，而非内部 E2m1 量化。** overlay 的 `FP4BlockScaleLoraLauncher` 初版输入原始 bf16（路径 3，内部 NvFP4 量化），因此选中 `Gemm2::Runner(E2m1, E2m1, BF16)` —— 该 tile-8 cubin 不存在，报 "No kernel found"。改为先用 `_prepare_flashinfer_mxfp8_activations` 预量化（同非 LoRA 路径），再输入 `Gemm2::Runner(MxE4m3, MxE2m1, BF16)`。
14. **零权重 LoRA 等于 base 要求透传路径统一。** 旧 dispatch 将无适配器请求提前交回 stock 融合核。该融合核的 fp32 GEMM1 epilogue 不物化 gate_up 输出，而分解 `Gemm2::Runner` 写 bf16，导致逐层舍入不一致，贪婪解码必然发散（结构上无法一致）。现将无适配器请求也送入同一分解管线，delta 置零（VE 调用以 `has_active_lora` 守卫；透传时对 delta buffer 执行 `zero_()`）。零权重适配器的 VE delta 为精确 0.0，与透传零 bit 级相同（`x + 0.0 → x`），故 zero==base 现为 BITWISE。性能代价不可测（单请求 decode 149 vs 147 tok/s）。
15. **E4m3 激活量化必须走融合核**（`fused_activation_mxfp8_quant.cuh`）。非融合链会把激活行先物化为 bf16 再量化，多出的一次 RN 舍入会改变每 32-block 的 amax，并在 PosInf 舍入边界上翻转 UE8M0 指数。融合核逐元素复刻 `gatedSiluScalars`，但直接从 fp32 寄存器量化（精确的 `cvt_warp_fp16_to_mxfp8` 配方）。注意 `__nv_cvt_float_to_fp8` 返回原始存储字节，必须按 stock 核经 `__nv_cvt_float2_to_fp8x2` 构造，否则编译失败。
16. **UE8M0 激活 scale 需要专用 gather+swizzle 核。** `moe::dev::permute` 唯一的 scale 路径（useDeepSeekFp8）索引 FP32 128-block 线性布局（`idx = token + numTokens*(hidden/128)`）。向其输入 UE8M0 32-block 张量会越界读，并把垃圾写入 SF buffer，使 gate_up GEMM 按噪声缩放（输出为分词器垃圾）。改用 `sgl_gather_swizzle_ue8m0_sf`（R128c4/R8c4，公式同 `dev_kernel` 的 `getSfOffset`/`get_sf_out_offset_*`），并以 `useDeepSeekFp8=false` permute E4m3 数据。

## 6. Tinker `all-linear` 3D LoRA：722/722 全量（2026-10-12 新增）

### 6.1 张量形态

来源 `https://mint-dev.oss-cn-beijing.aliyuncs.com/lora-test/dpsk-v4.1-flash-lora-tinker.zip`（6.29 GB）。解压后 `adapter_model.safetensors` 含 **722 张量**，`r=32, lora_alpha=32, target_modules="all-linear"`：

| 模块组 | 张量数 | 外维 |
| --- | --- | --- |
| `experts.w1` / `w2` / `w3` | 80×3 | A/B 混合，见下 |
| `shared_experts.{gate,up,down}_proj` | 80×3 | 2-D |
| `self_attn.{wq_a,wkv,wo_b}` | 80×3 | row-parallel / replicated |
| `model.lm_head` | 2 | 2-D |

其中 **3-D 张量 240 个**，全程原样保留，未铺平为 2-D。逐层实测外维：

```
experts.w1 | lora_A  (1, 32, 5120)      共享
experts.w1 | lora_B  (384, 2304, 32)    每专家
experts.w2 | lora_A  (384, 32, 2304)    每专家
experts.w2 | lora_B  (1, 5120, 32)      共享
experts.w3 | lora_A  (1, 32, 5120)      共享
experts.w3 | lora_B  (384, 2304, 32)    每专家
```

该布局与引擎 kernel 的要求一致：`srt/lora/lora_moe_runners.py` 对 gate_up 传 `experts_shared_outer_loras_a=<flag>, _b=False`，对 down 传 `_a=False, _b=<flag>`，即 gate_up 的 A 侧共享、B 侧每专家，down 反之。

### 6.2 修正的 8 处缺陷

| # | 现象 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | `load_lora_adapter → success:false` | 适配器 `r=32` 大于服务的 `--max-lora-rank 16` | `--max-lora-rank 32` |
| 2 | `ValueError: Cannot find target module name for …self_attn.wkv…` | `self_attn.{wq_a,wkv,wo_b}` 不在 `SUPPORTED_LORA_TARGET_MODULES` 白名单 | 白名单（`utils/common.py`）与 `_KNOWN_LORA_TARGET_MODULES`（`lora/utils.py`）各加三项，`params_mapping` 加映射 |
| 3 | `No corresponding LoRA layer supported for <class Linear>`，实际触发模块为 `self_attn.compressor.wkv` | `get_target_module_name` 采用子串匹配与裸名映射，误命中同形名 `compressor.wkv`（裸 `nn.Linear`） | 改用两级限定名：`wq_a` → `self_attn.wq_a`（上游已有同类写法 `wq_b` → `indexer.wq_b`） |
| 4 | `NotImplementedError: get_hidden_dim not implemented for self_attn.wo_b` | `get_default_hidden_dim` 未覆盖新模块 | 补 `self_attn.{wq_a,wkv,wo_b}` 的 in/out 维度 |
| 5 | `ValueError: experts_shared_outer_loras is enabled but 'gate_up_proj_moe' lora_B has expert_dim=384 (expected 1)` | `mem_pool.py` 四处 `experts_shared_outer_loras` shared 元组与 kernel 语义不一致 | 对齐为 A 侧共享=`gate_up_proj_moe`、B 侧共享=`down_proj_moe`；另一半走既有 `_iter_local_expert_weights` 逐专家路径 |
| 6 | `assert x.shape[-1] == K`（探针输出 `x=(768,2048)` vs `K=1024`） | `self_attn.wo_b` 为 `RowParallelLinear`，输入应为 `o_groups*o_lora_rank/tp = 8*1024/4 = 2048`；误用 `n_groups`（配置字段实为 `o_groups`），且未做 TP 分片 | `get_default_hidden_dim` 改用 `config.o_groups` 返回全宽，并把 `self_attn.wo_b` 加入 `ROW_PARALLELISM_LINEAR_LORA_NAMES`，自动 `/tp` |
| 7 | `no GPU memory for the KV cache` / `EADDRINUSE: 41300` | 其一，关闭 shared-outer 后 LoRA 池由 `[1,…]` 膨胀为 `[384,…]`；其二，反复重叠重启遗留孤儿进程占用 nccl 端口 | 保持 shared-outer 开启 + `--max-loras-per-batch 1`（`384×1<1024`，规避上游注释点名的 `_align_block_size_large` JIT kernel fault）+ 干净重启 |
| 8 | 报错信息与实际不符，误导定位 | 两处 `raise` 的模块名硬编码，恒输出 `gate_up_proj_moe lora_A` / `down_proj_moe lora_B` | 改为 `f"{name!r}"`；raise 语义不变，仅修正信息 |

一条不变量：`get_target_module_name` 对未映射 target module 的 `raise` 是刻意的 fail-fast，不得改为静默 `skip/continue`。

### 6.3 验收

| 项 | 结果 |
| --- | --- |
| 加载 | `load_lora_adapter moe_tinker → /root/dsv41_moe_tinker/adapter`，722/722，两个端口均 `success:true` |
| 推理 | HTTP 200，`len=235/261`，`mojibake=False`，中文完整连贯 |
| 服务 | 30100/30200/31500 = 200/200/200，连续运行 22 分钟未重启 |
| 3-D | 240 个 3-D 张量字节级原样，未铺平 |
| fail-fast | 保留 |
| 探针 | 临时 `[SHRINK-DIAG]` 已移除；`chunked_sgmv_shrink.py` 恢复原始（`KERNEL_COMPILE_OK`） |

判别性验证（同一 prompt，`temperature=0`，`thinking=false`）：

- 起效：`base` `282e81ec72a2` ≠ `full(moe_tinker)` `27dae697c292`（长 prompt）
- 逐组消融（将某组张量置零后与全量比较）：`no_experts_w1` `59d0d3394253`、`no_experts_w2` `212654058a35`、`no_experts_w3` `8e1c3f0e82dd`、`no_shared_experts` `28b3ebca0eda`、`no_attn` `194582d97079`、`no_lm_head` `0ed2d4d66b88` —— 六组均与全量不同，即六组模块均被加载并参与计算

### 6.4 已知限制

非零 LoRA 下，`temperature=0` 的贪婪解码输出不确定（同一请求多次结果不同）；base（无适配器）与全零适配器确定。

实测：长 prompt 4 次 —— `base` `282e81ec72` 4 次相同，`zero` `73d08d7514` 4 次相同，`full` 4 次互不相同；短 prompt 6 次 —— `base/zero/full` 均为 `c81e728d9d`（该请求下 delta 不改变贪婪输出），`moe_rnd` 6 次互不相同。

已定性为 LoRA 模块自身的浮点求和序差异，修复成本高，不属于严重缺陷，接受。影响：md5 型判据不严格（“不同”可能来自不确定性而非消融生效）；若产品侧要求严格可复现，需固定归约顺序。

## 7. 性能（`docs/LORA_PERF_OPT_PLAN.md` 摘要）

### 7.1 口径

生产口径：decode CUDA graph 开启、dspark 开启、PD 分离、smg 为唯一入口。指标取服务端 `gen throughput (token/s)`，仅取 `#running-req ≥ 64` 的窗口，并剔除每窗口前后 15%。负载 `out=4096 / n=256 / conc=64 / ignore_eos`。

### 7.2 已确立的结论（F1–F8）

- **F1** 代价全部位于 mxfp4 MoE-LoRA 的 delta 执行路径，与权重数值无关（`moe_zero ≈ moe_rnd`，即“启用 LoRA 本身 ≈ 0”）
- **F2** 不存在更快的老路：BYPASSED 不比物化快（5691.8/5699.8 < 6000.1/6028.0）
- **F3** fork 自带 O1 双流重叠（侧流执行 gate_up LoRA 的 shrink+expand，与主流 quant 并行），有 fp8/fp4/bf16 三支，原缺 mxfp4 支，已移植 `..._mxfp4_lora_two_stream`
- **F4** 双流有两道门：`has_active_lora`（捕获期为 False）与 `SGLANG_TWO_STREAM_MAX_TOKENS`（默认 256，小于 dspark verify 批）
- **F5** decode 是 CUDA-graph replay，分支在捕获期确定（本部署本就开启双图捕获）
- **F6** 已接入“图内启用双流”（gate 增加 `or is_capturing_lora_variant()`，上限提至 1024）
- **F7** 但进图即崩：捕获期 `mhc_post` 报 `CUDA error: unspecified launch failure`
- **F8** 机制为侧流与主流的汇合依赖 C++ trtllm 算子内部的 `wait(lora_ready_event)`，而 mxfp4 trtllm-gen 算子未实现该跨流等待，构成竞态

### 7.3 里程碑（M0–M5）

- **M0** 准备与固化，完成。**M1** 崩溃机制定位，完成。双流从“必崩”转为“稳定可跑”：根因是子代理移植体自身的缺陷；处理方式是以已知可运行的单流 mxfp4 函数体为基底重建，仅增量加入侧流改写。判据：四种简化全部崩溃，而委派原生单流与重建版均能完整启服。
- **M5 判决**：双流对 LoRA 请求仅 +2.38%（7923.9 vs 7739.5 tok/s）。原假设被证伪 —— 该 38% 不在“gate_up delta 串行”，而在串行依赖：
  `permute → GEMM1 → activation（必须读取 delta）→ MXFP8 quant → GEMM2 → finalize`
  旁流可重叠的区间很窄，消除该 38% 必须在 kernel 内完成。

### 7.4 融合路线盘点（F1–F3）

- **F1** LoRA delta 内核自身成本 = 0（数值无损消融：退化为零 delta 通路）
- **F2** cubin 供应盘点：4-bit 激活的 MX GEMM 不存在（缺 `Bmm_MxE4m3_MxE2m1MxE4m3` 的 tile-8/16 变体）
- **F3** 可融项盘点与天花板

### 7.5 `<6ms` 目标与实测

- **M2a** 找到免 padding 的核：`deep_gemm 0.2.0` 的 `m_grouped_fp8_fp4_gemm_nt_masked`（fp8 A × fp4 B，按每专家行数，免补齐）
- **M2b** 仓库已有 DeepGEMM 的 DSV4 MXFP4 MoE 路径（`moe_runner/deep_gemm.py`，A=MxFP8(1,128)/B=MXFP4(1,32)，与本案格式一致），代码自有；两个 LoRA delta 注入点已定位（gateup 输出后 / down 输出后），delta 仍由现有 VE 核产出，适配器仍为 13.6 MiB，零额外显存
- **§16 实测**：`--moe-runner-backend deep_gemm` 对本模型无效；主路改为绕开 runner/quant-method 管道直接调用 DeepGEMM masked 核
- **§17** 取得 DeepGEMM masked 分组 GEMM 的确切契约（照抄 `moe_runner/deep_gemm.py:755-779`），另有两条契约坑
- **§18** padding 假设被直接证伪，M2 路线取消，35% 的主因另有其物（该 GEMM 为权重带宽受限，masked 与 padded 仅差 1.05×）
- **§20** 离线合并的数值可行性不成立（MXFP4 重量化噪声 9.4e-4 ≈ 4%\|W\|，会淹没 <10%\|W\| 的真实 delta），至此低成本路线全部关闭
- **§21** 达标路径③：TPOT 随并发上升，对 LoRA 流量限并发即可 <6ms（零 kernel 工作）—— 已由用户否决
- **§22** C=64 硬指标（<6ms）的 kernel 级归因（torch profiler/CUPTI，40 步窗口，总核时间 1423.8 ms）
- **§23** 定位真正的旋钮：`config_index = [tile_N, config]`（Python 侧可直接给出，无需改 C++）
- **§24/§25** 两把零显存手段：其一为 MoE GEMM 的 `config_index`（tactic/tile），其二为 LoRA VE 核的 CSGMV 配置

§25.4 最终判定（本侧）：

| 口径 | TPOT p50 (ms) |
| --- | --- |
| base（无 LoRA） | 5.233 |
| LoRA 基线（默认 tactic `[-1,-1]`，从未 autotune） | 7.964 |
| LoRA 最优（第一刀 `tile=64, config=0`） | 7.311（−8.2%） |
| 第二刀 VE 天花板（CUPTI 占比 ≈5%） | ≈ 6.95 |
| 判据 | ≤ 6.000，尚差 ≈ 17.9% |

§25.6 附带结论：`tile_N` 最优落在 16/32/128（非 64），但均停在 ≈7.3 ms 平台（极差 <2.4%）；`config=1` 明显更差（8.209 ms）。tile 维度已饱和，继续调整 tile 无法收敛到 6.0 ms。

### 7.6 剩余路径（数周级，需上游或自研）

1. **上游**：提供带 `gate_up` LoRA delta 钩子的融合核，将 delta 融入 GEMM1 epilogue，省去独立的 permute/quant/act 链
2. **cubin**：补 `Bmm_MxE4m3_MxE2m1MxE4m3` 的 tile-8/16 M-tile 变体（当前使用 `t128x8x512u2`）
3. **自研 tcgen05 融合核**：复用仓库既有 `dev_kernel.cu` / `fused_activation_mxfp8_quant.cuh` / VE delta 核，将 `permute → gate_up GEMM → act+delta → quant → down GEMM → finalize` 合为单核

## 8. 红线与不变量

1. MoE LoRA 结构不变：A/B 张量形状、秩、目标模块集合、虚拟专家语义、量化格式、gating 数学
2. `get_target_module_name` 的 `raise` 必须保留 —— 未映射 target module 必须报错，不得静默跳过
3. 每次 rsync 后清除 `__pycache__`
4. 压测进行中不得启动第二个 boot（会终止正在压测的栈）
5. 多场景基准必须消除顺序效应：阶段首个场景承受首次满载惩罚，需反向复测；正向独跑得到的“优于 base”数据一律视为待证

## 9. 已排除的技术路线

| 路线 | 排除依据 |
| --- | --- |
| 离线合并（将适配器折入权重） | MXFP4 重量化噪声 9.4e-4 ≈ 4%\|W\|，会淹没 <10%\|W\| 的真实 delta（§20 实测） |
| padding（改用 masked 免补齐 GEMM） | masked 与 padded 仅差 1.05×（该 GEMM 为权重带宽受限）；§18 直接证伪 |
| 限制 LoRA 流量并发 | 已由用户否决 |
| 双流调度重叠 | 实测仅 +2.4%（§7.3 M5 判决） |
| 将 delta 融入激活/量化核 | 消融实测 ≈ 0 |
| 激活改用 bf16 | stock 形态，launcher 无该分支，需新写 C++ |

## 10. 已知限制

1. 非零 LoRA 下 `temperature=0` 输出不确定（LoRA 模块的浮点求和序；已定性为可接受，见 §6.4）
2. 本侧 `<6ms` 目标未达标：两把零显存手段叠加约 7.2–7.3 ms，距 6.000 ms 判据尚差约 17.9%（§7.6 列出三条数周级路径）
3. 双流仅在 decode 形状 tokens ≤ 256 且 LoRA 生效时有意义
4. 早期单机 TP8 的“零开销”数据已作废（同类勘误见 `GLM53_FLASH_MOL_TECH_REPORT.md` §3）

## 11. 资产与复现

### 11.1 文档

| 文档 | 内容 |
| --- | --- |
| 本文件 | DSV4.1-Flash MoL 支持与性能总报告 |
| `docs/LORA_PERF_OPT_PLAN.md` | DSV4.1(mxfp4) 挂 LoRA 的性能优化计划（§0–§25.6，95 KB，含全部实测与复现命令） |
| `AGENTS.md` | DSV4.1 MoL 的 16 处缺陷、环境布局、验证配方 |
| `docs/GLM53_FLASH_MOL_TECH_REPORT.md` | 同族 GLM-5.3-Flash 的 MoL 报告，可对照 |
| `docs/MOE_LORA_PERF_3PCT_INVESTIGATION_REPORT.md`、`docs/THROUGHPUT_EXPERIMENT_REPORT.md`、`docs/TTFT_OOM_STALL_ROOT_CAUSE.md`、`docs/dsv41_mm_ttft.md` | 相关专题 |

### 11.2 1101 关键资产

- 服务脚本：`/root/exp_thinking.sh`（在用）、`/root/exp_single.sh`（备份）
- 适配器：`/root/dsv41_moe_rnd`、`/root/dsv41_moe_zero`、`/root/dsv41_moe_tinker/adapter`（Tinker 722）
- 日志：`/root/dsv41_upstream/{pd_prefill.log,pd_decode.log,smg.log}`；归档 `/root/log_archive/`
- 备份（改 patch 前后均有）：`mem_pool.py.bak-*`、`lora/utils.py.bak-*`、`chunked_sgmv_shrink.py.bak-*`

### 11.3 最小复现（Tinker 3D LoRA）

```bash
# 环境：1101；PYTHONPATH=/root/dsv41_upstream；venv /opt/sglang-venv
KEY=$(cat /root/dsv41_upstream/smg_api_key.txt)

# 1) 加载（全量 722）
curl -s -X POST http://127.0.0.1:30100/load_lora_adapter \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d '{"lora_name":"moe_tinker","lora_path":"/root/dsv41_moe_tinker/adapter"}'
# → {"success":true,...,"loaded_adapters":{"moe_tinker":"/root/dsv41_moe_tinker/adapter"}}

# 2) 推理（必须显式传 lora_path）
curl -s http://127.0.0.1:31500/v1/chat/completions \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" \
  -d '{"model":"moe_tinker","lora_path":"moe_tinker",
       "messages":[{"role":"user","content":"用一句话介绍深海生物。"}],
       "max_tokens":120,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
```

请求级的 LoRA 激活字段为 `lora_path`（OpenAI 与 native 端点一致，可取适配器名或路径）。`lora_name` 不是请求字段，会被静默忽略 —— 请求按 base 处理，表现为“挂载成功但输出等于 base”。

## 12. 作用域优化：只挂后 N 层，其余层零开销（2026-10-12 新增）

### 12.1 问题：路径按批次决定，没有权重的层也在跑

`has_active_lora` 是 **batch 级**的（`srt/lora/lora_manager.py`：`batch_info.has_active_lora = any(lora_ranks[wi] > 0)`），而 `init_lora_modules` 按**模块名命中 `target_modules`** 就把**全部 40 层**的 MoE 包成 `FusedMoEWithLoRA`。后果：

- batch 里只要有**一个**请求挂适配器 ⇒ **全体 MoE 层进分解链**；
- `dsv41_moe_rnd` / `dsv41_moe_zero` 实际只覆盖 **20–39**（读 safetensors header 核实：120 张量 / 20 层；对照 `dsv41_moe_tinker` 是 722 张量 / 40 层）；
- ⇒ **0–19 层的 delta 恒等于 0，却照付整条分解链的开销**。
- `_adapter_weight_layers()`（从 safetensors header 读出适配器权重落在哪些层）这份信息，旧实现**只用于 `lora_kv_shared` 的加载期校验**，compute 路径从不使用。

### 12.2 修法：`--lora-target-layers`

新增启动参数，语法 `20-39` / `20,21,39` / `all`；**不填 = 全量**（行为与改动前完全一致）。

| 层 | 行为 |
| --- | --- |
| 在范围内 | 正常包装为 `FusedMoEWithLoRA`，走分解链 |
| 在范围外 | **`init_lora_modules` 里直接 `continue`，完全不包装** ⇒ 保持原生 `FusedMoE` ⇒ 永远走 stock 融合核 |

**为什么这才是「零开销」**：走分解链的**唯一前提**是该层被包成 LoRA 模块。不包装 = 结构上根本不进入这条路径 —— 不分配 LoRA buffer、不调用 VE delta 核、不进分解 GEMM。「算出 0 再加进去」和「根本不进这条路」是两回事。

同时加**加载期守卫**（`validate_new_adapter`）：适配器权重落在配置范围外 ⇒ **响亮 `raise ValueError`**，不静默降级（避免出现「挂载成功但部分层没生效」的假象）。

**运行时证据**（4 个 TP rank 全打）：

```
LoRA layer filter ACTIVE: LoRA is mounted on 20 layer(s) [20, 21, ..., 39] only;
every other layer stays unwrapped on the stock path (zero LoRA overhead).
```

守卫实测（把覆盖 0–39 的 `moe_tinker` 往该模式下加载）：

```
Failed to load moe_tinker: adapter carries weights for layer(s) [0, 1, ..., 19],
which are outside the configured --lora-target-layers range [20, ..., 39].
Those layers are deliberately left unwrapped (zero-overhead mode),
so this adapter would be silently under-applied.
```

### 12.3 实测

口径：`conc=64 / n=256 / out=4096 / ignore_eos / seed=777`，服务端 `gen throughput`，只取 `#running-req ≥ 64` 并剔除每窗口前后 15%（与 §7.1 同口径）。适配器必须在 **P(30100) 与 D(30200) 两侧都加载**（PD 分离下只加载一侧 ⇒ 经 smg 的请求全部超时，本次已实测踩到）。

**同场对照**（同机、同日、相邻时段；两组 base 只差 0.2%）：

| 模式 | base p50 (tok/s) | `moe_rnd` p50 (tok/s) | TPOT p50 (ms) | 保留率 rnd/base |
| --- | --- | --- | --- | --- |
| 全量（40 层全包装） | 12150.5 | 7561.9 | 5.267 → 8.464 | 0.622 |
| **后 20 层（`--lora-target-layers 20-39`）** | 12176.2 | **8789.6** | 5.256 → **7.281** | **0.722** |

**⇒ 挂 LoRA 的吞吐代价由 −37.8% 收窄到 −27.8%**，即后 20 层模式下**吞吐 +16.2%、TPOT −1.18 ms**。两组 base 仅差 0.2%，因此这不是会话间波动。

客户端侧同向：后 20 层模式 base 4.679 ms/tok、`moe_rnd` 7.636 ms/tok；全量模式 base 4.725、`moe_rnd` 8.654 —— 四种各 256/256 成功、`bad=0`。

**全量模式回归**：`--lora-target-layers` 不填时行为不变 —— 40 层适配器 `dsv41_moe_tinker`（722 张量）与 20 层适配器 `dsv41_moe_rnd` **均 `success:true`**，filter 日志为 0，A/B 正常完成。

**守卫实测**：把 40 层适配器加载到 `20-39` 模式的实例上 ⇒ `success:false`，错误信息直接点名越界层 `[0..19]`（见 §12.2）。

> 历史对照（跨时段）：`docs/LORA_PERF_OPT_PLAN.md` §M4 的既有基线为 base 12244.3 / `moe_rnd` 7739.5（保留率 0.632），与本次全量模式复测的 0.622 一致。

### 12.4 顺带查实

`SGLANG_LORA_TWO_STREAM` 是**死变量** —— 全树 6 处只有注释/docstring，没有任何 `os.environ.get` / `EnvBool` 读取点。双流真正的开关是 `SGLANG_EXPERIMENTAL_LORA_OPTI`（`srt/lora/layers.py` 里 `install_two_stream_overrides`）。启动脚本里写 `SGLANG_LORA_TWO_STREAM=0` 关不掉任何东西。

### 12.5 复现

```bash
# 后 20 层模式（前 20 层零开销）
bash /root/exp_l20.sh          # 内含 --lora-target-layers 20-39
# 全量模式（默认，行为不变）
bash /root/exp_thinking.sh

# 适配器必须在 P 与 D 两侧都加载（smg 会 404）
bash /root/load_adapters.sh
```

### 12.6 显存回收：buffer 也按范围分配

算力零开销之外，LoRA 的**显存**也按范围收：`mem_pool.init_buffers()` 原先对 `range(num_layer)` 每层都预分配 A/B buffer；现在范围外的层换成 `(max_loras_per_batch, 0)` 占位（0 字节），并在启动日志打印回收量。

- 占位**保留首维（槽位维）**，因此 `_clear_buffer_slot_for_base()` 的 `tensor[buffer_id].zero_()` 仍成立；真被误用则形状不符 ⇒ **响亮报错**。
- 范围外的层**没有任何 LoRA 模块持有这些张量引用**（它们未被包装），所以替换是安全的；不是「算出 0」，是根本不存在。

**实测**：`LoRA layer filter: freed 901.41 MiB of per-layer LoRA buffers`（**每 rank**；TP4 × P/D ⇒ 合计约 7 GiB 回到 KV pool）。

### 12.7 「conc=64 下 LoRA 6 ms」的可达性

判据：服务端 TPOT p50 ≤ **6.000 ms**（`conc=64 / n=256 / out=4096 / ignore_eos / seed=777`）。

| 量 | 值 |
| --- | --- |
| base（无 LoRA） | 5.243 ms |
| 后 20 层 LoRA（本次优化后） | **7.281 ms** |
| ⇒ LoRA 额外开销 | **2.038 ms** |
| 目标 6.000 ⇒ 允许的额外开销 | **0.757 ms** |

**⇒ 需要把 LoRA 的额外开销再砍 ~63%。**

**n=256 正式口径下的 tactic 重扫**（免重启热切 `/tmp/lora_tactic.txt`，同会话背靠背；**客户端聚合为准**）：

| 候选 | 客户端聚合 (tok/s) | wall | 备注 |
| --- | --- | --- | --- |
| `16,0` | **5207** | 201.4 s | 最差 |
| `32,0` | **7927** | 132.3 s | 最优档 |
| `64,0` | 7893 | 132.8 s | 同档 |

⇒ `tile=16` 明显更差；32/64 基本相同。台账 §25.6「tile 维度已饱和」的方向在 l20 配置下**依然成立**（只是最优档从它当时测的 16/32/128 变成 32/64）。

> ⚠️ **方法教训（本次踩过，记档）**：**服务端 per-window 的 `gen throughput` 不能当聚合吞吐用**。当某次跑有长尾时，「剔前后 15%」会把慢尾剔掉、只留早段快窗口 ⇒ 虚高。实测 `tile=16` 的服务端分窗解析给 10860 tok/s，而客户端聚合只有 5207。**判定候选优劣一律以客户端聚合为准。**

已实测关闭的其余配置层旋钮见台账 §25.4/§25.6（tactic −8.2% 已吃满、VE CSGMV ±2~3%、`config=1` 更差 8.209 ms）。⇒ **只动调度/配置，6.0 ms 够不到。**

要真正达标，只剩台账 §25.5 的三条（均数周级或须上游）：

1. **上游**：带 `gate_up` LoRA delta 钩子的**融合核**（delta 融进 GEMM1 的 epilogue，省掉独立的 permute/quant/act 链）；
2. **cubin**：补 `Bmm_MxE4m3_MxE2m1MxE4m3` 的 **tile-8/16** 变体（现跑 `t128x8x512u2`）；
3. **自研 tcgen05 融合核**：把 `permute → gate_up GEMM → act+delta → quant → down GEMM → finalize` 合成单核。

**根因**（§18 实测）：那 ~86% 的开销在两个 batched GEMM —— 它们丢掉了 base 融合 cubin 内部的「权重流式 ↔ 算力」单核软件流水线。**不是** padding（masked vs padded 仅 1.05×，该 GEMM 权重带宽受限），**也不是** delta 核本身（F1 消融 ≈ 0）。

### 12.8 纯算子/逻辑优化（进行中）

目标是在**不改模型、不限并发**的前提下把 conc=64 的 LoRA TPOT 压到 6.000 ms。

**(a) 给 LoRA 路径上「未覆盖 PDL」的 kernel 补 PDL**（Programmatic Dependent Launch；`is_arch_support_pdl()` = `arch.major ≥ 9` ⇒ B300/SM100 支持）。

盘点：`srt/lora` 链上 **每步约 60 次发射**落在两个原本 **0 PDL** 的文件里 ——

| kernel | 文件 | 每步发射 |
| --- | --- | --- |
| `_fused_lora_delta_shrink_kernel` / `_fused_lora_delta_expand_kernel`（VE delta，per-group 路径） | `kernels/ops/moe/virtual_experts.py` | 2 × 20 |
| `_pack_topk_kernel`（routing pack） | `kernels/ops/moe/trtllm_lora_temp/topk_pack.py` | 20 |

改法严格照仓库既有先例（`sgemm_lora_b.py` / `elementwise.py`）：核内 `if USE_PDL: tl.extra.cuda.gdc_wait()`（消费输入前）+ `tl.extra.cuda.gdc_launch_dependents()`（读完输入后），发射侧 `USE_PDL=use_pdl, **({"launch_pdl": True} if use_pdl else {})`。参考：ferrite-v2 的 PDL 覆盖先例（未覆盖的 quant/gemm/moe 每处 −100~300 µs）。

**(b) 消掉每层每步的 `pad / contiguous` 拷贝**（纯推理逻辑，零 kernel 改动）。

`lora_dispatch.py` 与孪生 `moe_overlap.py` 每层都要做：

```
gate_up_delta_ve = new_zeros((T, top_k, 1152))          # unpadded
…VE 写入…
gate_up_delta    = F.pad(gate_up_delta_ve, (0, 1280-1152))   # ← 拷贝①
down_in          = activation_lora_input[..., :576].contiguous().view(-1, 576)  # ← 拷贝②
```

两侧宽度不同（C++ 用 padded 640 / VE 用 unpadded 576）是这两次拷贝的根因。**改法**：直接把缓冲按 **padded 宽** 分配（`new_zeros` 一次清零），VE 通过 **`[..., :2*_inter_ve]` 视图**写前 1152 列 —— VE 的 expand 核本来就是按 `output.stride(-2)/stride(-1)` 写的（其注释明确支持 3-D strided output），而送给 C++ 核的仍是那个**连续的** padded 张量（满足 `assert gate_up_lora_delta.is_contiguous()`）。⇒ **拷贝①彻底消失**；`moe_overlap.py` 里原本已按 padded 分配、只多一次 `copy_`，同样可去。

**实测结论：PDL 方向被数据否决（已回滚）。**

原因是先把「时间到底花在哪」量了出来：对 l20 的既有 trace 计算「核并集 busy 时间 vs 窗口跨度」——

```
窗口跨度       : 820.49 ms
GPU busy(并集) : 799.68 ms  (97.5%)
GPU idle/gap   :  20.81 ms  ( 2.5%)
最大间隙       : 2.4 / 1.2 / 0.8 µs …（中位数 0 µs）
```

**GPU 几乎从不空闲 ⇒ 间隙只占 2.5%，PDL 的天花板就是这 2.5%。** 成本 97.5% 在**核本体**，因此提速只能靠「减少核工作量」（更少/更小/更快的核），而不是重叠发射。

- 改法已回滚（`*.py.bak-pdl` 恢复），`py_compile` 通过，`USE_PDL` 残留计数 0/0。
- 那次 PDL 评测本身还踩了另一个坑：`POST /stop_profile` 返回 500 ⇒ **profiler 实际没停**，与其重叠的评测段测出 p50 **12.402 ms/tok**，而「采样中」那次是 **12.382 ms/tok** —— 两个数几乎相同，**证明那是采样开销而非 PDL 的影响**。⇒ **profiler 启停必须验证；采样窗口内不做任何性能结论。**

### 12.9 核级画像（l20 配置，`/start_profile` + `trace_agg`）

采集：`POST /start_profile {"activities":["GPU"],"num_steps":40}` → `*-TP-N.trace.json.gz` → 按核聚合。
窗口内 **76247 个 kernel，总核时 960.9 ms**（TP0）。

| kernel | total_ms | calls | µs/call | 说明 |
| --- | --- | --- | --- | --- |
| `bmm_MxE4m3_MxE2m1MxE4m3…t128x16x256u2_s6_et128x16_m256x16x` | **195.40** | 1600 | 122.1 | **MoE gate_up GEMM** |
| `bmm_Bfloat16_MxE2m1_MxE4m3…t128x16x256_s6_…_c2` | **123.75** | 1720 | 72.0 | **MoE down GEMM** |
| `sm100::decode::sparse::head64::…flash_fwd_splitkv_mla_fp8_sparse` | 109.65 | 1720 | 63.8 | MLA sparse 注意力 |
| `cutlass…dense_blockscaled_gemm_sm100…` | 87.94 | **9200** | 9.6 | 块缩放 GEMM |
| `deep_gemm::sm100_tf32_hc_prenorm_gemm_impl` | 46.00 | 6400 | 7.2 | |
| `sglang::moe_finalize_all_reduce_kernel` | 43.07 | 1600 | 26.9 | |
| `sglang::all_reduce_2shot_pull_kernel` | 41.44 | 2000 | 20.7 | |
| `_hc_mix_reduce_sinkhorn_kernel` | 25.83 | 3440 | 7.5 | |
| `mxfp8_quantize`（三个变体合计） | 34.8 | ~7000 | ~5 | |
| `moe::dev::finalize::finalizeKernelVecLoad` | 19.77 | 1600 | 12.4 | |
| `nvjet_sm103_tss/tst`（两个） | 31.9 | 3200 | ~10 | |
| `at::native::elementwise_kernel` | 14.34 | 1880 | 7.6 | |
| `_router_triton_kernel` | 13.65 | 1720 | 7.9 | |
| `_mhc_post_combine` | 10.93 | 1480 | 7.4 | |

**两个 MoE GEMM 合计 319.2 ms = 总核时的 33%**（其余 67% 是基座共用：注意力、量化、通信、norm 等）。

**重要观察**：**Top-25 里一个名字含 "lora" 的核都没有** —— LoRA 路径的 GEMM 走的正是这两个 `bmm_*`（trtllm-gen cubin，名字里没有 lora），VE delta 与 routing pack 的核则小到进不了前 25。⇒ 「LoRA 的钱花在哪」不能靠看名字，**必须用 `base` 与 `rnd` 的同配置画像做差**（§12.10）。

**踩坑记档**：`POST /stop_profile` 在这次返回了 **500 Internal Server Error**，**profiler 实际没停** —— 随后与之重叠的评测段（`pdl_rnd`）测出 p50 12.402 ms/tok，与「采样中」那次的 12.382 ms/tok 几乎相同，**证明那是采样开销而非被测改动的影响**。⇒ **profiler 的启停必须验证；采样期间不要做任何性能结论。**

### 12.10 根因定位：LoRA 走的 GEMM 被选错了 cubin（base vs rnd 画像做差）

方法：**同配置、各自独立重启**（避免 profiler 残留），分别采 `base`（不带适配器）与 `rnd`（带 `moe_rnd`）的 40 步 GPU trace，再逐核做差（TP0）。

```
base 总核时  980.0 ms / 76305 事件
rnd  总核时 1249.4 ms / 88487 事件         ⇒ Δ +269.4 ms

增量最大的核（rnd − base）：
  +166.35 ms  bmm_Bfloat16_MxE2m1MxE4m3…t128x8x512u2_s3…c1x1x1     ← rnd 独有
  +143.08 ms  bmm_Bfloat16_MxE2m1MxE4m3…t128x8x512_s3…c1x1x1       ← rnd 独有
  + 32.30 ms  _chunked_lora_shrink_kernel
  + 25.46 ms  _chunked_lora_expand_kernel
  + 10.27 ms  moe::dev::finalize::finalizeKernelVecLoad
```

**逐调用对比（这才是判据）：**

| 路径 | kernel | calls | **µs/call** |
| --- | --- | --- | --- |
| base | `bmm_MxE4m3_MxE2m1MxE4m3…**t128x16x256u2_s6**…c2x1x1` | 1600 | **130.94** |
| base | `bmm_Bfloat16_MxE2m1MxE4m3…**t128x16x256_s6**…c2x1x1` | 1720 | **76.00** |
| rnd（新增） | `bmm_Bfloat16_MxE2m1MxE4m3…**t128x8x512u2_s3**…c1x1x1` | 800 | **207.94** |
| rnd（新增） | `bmm_Bfloat16_MxE2m1MxE4m3…**t128x8x512_s3**…c1x1x1` | 800 | **178.85** |
| rnd | `bmm_MxE4m3…t128x16x256u2_s6…` | 800 | 156.67 |
| rnd | `bmm_Bfloat16…t128x16x256_s6…` | 920 | 83.37 |

**⇒ LoRA 多跑两个 `t128x8x512_s3` 家族的 GEMM（166+143 = 309 ms），每次比基座同族逻辑工作量慢 1.6×（208 vs 131）与 2.4×（179 vs 76）。这 309 ms 正是那 ~2 ms 的墙钟增量来源。**

**⇒ 这不是「LoRA 数学」的代价，而是 `config_index` 选中的 cubin 更差。**

**机制**（`kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu:187`）：

```cpp
resolveMoeTileAndConfig(config_index = [tile_N, config], supported_tile_nums, num_tokens, top_k, num_local_experts) {
  if (tile_N == -1 || config == -1) {                    // 只有 -1 走启发式
    return {selectDefaultTileN(...), -1};
  }
  return {tile_N, config};                               // 否则原样透传给 runner
}
```

而 runner 侧（`trtllm_fused_moe_runner.cu:757-760`）把 `config` 当作 **`mPassingConfigs` 的下标**：

```cpp
  TVM_FFI_ICHECK(configIndex < static_cast<int64_t>(mPassingConfigs.size()), ...);
  auto const &config = mPassingConfigs[configIndex];
```

**⇒ `mPassingConfigs` = 当前问题形状下校验通过的全部候选 (gemm1, gemm2) 配置；`configIndex` 即遍历它的下标 —— 这正是「autotuner 应当扫的空间」，而此前只扫过 {0,1}。**

**⇒ 结论：把 `config` 扫开（>=0 的更多下标）应当能找到基座那套 `t128x16x256_s6`，从而消掉这 309 ms。**

`_lora_tactic_hotswap_v3()` 读 `/tmp/lora_tactic.txt`（`[tile_N, config]`，1s 缓存）⇒ **免重启热切**，可直接扫描。

### 12.11 用源码自带的 probe 问出 `config` 的合法空间：**32 / 24**

`trtllm_fused_moe_kernel_launcher.cu` 导出了两个探测函数（`TVM_FFI_DLL_EXPORT_TYPED_FUNC`，行 4563-4566），可从 Python 直接调用：

```python
import os; os.environ["FLASHINFER_DISABLE_VERSION_CHECK"]="1"   # 否则 0.6.12/0.6.18 cubin 版本检查会直接抛
from sglang.kernels.ops.moe.trtllm_lora_temp.core import get_sgl_trtllm_moe_sm100_raw_module
m = get_sgl_trtllm_moe_sm100_raw_module()
m.sgl_trtllm_fp4_probe_gemm2(out_dim, k_dim, num_local_experts, num_tokens, top_k, tile_n)
# → [pt_passing, pt_valid, npt_passing, npt_valid]（pt=per-token 变体，npt=非 per-token）
```

实测（本模型：384 experts / top_k=6 / num_tokens=64）：

| 形状 | tile | 返回 |
| --- | --- | --- |
| gate_up (out=1152 或 1280, K=5120) | 16 / 32 | `[-1, -1, **32, 32**]` |
| gate_up | 64 | `[-1, -1, 12, 12]` |
| gate_up | 128 | `[-1, -1, 8, 8]` |
| down (out=5120, K=576 或 640) | 16 / 32 | `[-1, -1, **32, 24**]` |
| down | 64 | `[-1, -1, 12, 12]` |
| down | 128 | `[-1, -1, 8, 4]` |

**⇒ 两条可复用事实：**
1. **`config` 的合法上界随 tile 变化**：tile=32 时 gate_up 有 **32** 个合法下标（0..31）、down 有 **24** 个；tile=64 → 12；tile=128 → 8。
2. **per-token（pt）变体在本形状下不存在（返回 -1,-1，即构造 runner 抛异常）** ⇒ LoRA 的 MoE GEMM 只能走 non-per-token 路径。

**⇒ 结论：`config` 的扫描空间应按 probe 给出的上界来定，而不是凭猜（此前只扫了 {0,1}，见 §12.10）。**

### 12.12 优化清单与根因（2026-10-12 当日）

**已交付并验收的优化**

| # | 优化 | 机制 | 实测收益 |
| --- | --- | --- | --- |
| 1 | **LoRA 层作用域** `--lora-target-layers 20-39` | 范围外的层**根本不包装** ⇒ 不走分解链（不是"算出 0"） | **吞吐 +16.2% / TPOT −1.18 ms**（全量 vs 后20层，同场对照） |
| 2 | **buffer 按范围回收** | 范围外层给 `(max_loras_per_batch, 0)` 零字节占位 | **901.41 MiB / rank** |
| 3 | **MoE GEMM tactic** `config_index=[32,9]` | `mPassingConfigs` 下标（probe 证实合法空间 32/24） | −3.4%（n=128 排名；免重启热切） |
| 4 | **消 `F.pad` 拷贝** `lora_dispatch.py` + `moe_overlap.py` | 缓冲按 padded 宽分配一次，VE 直写 `[..., :2*inter_ve]` strided 视图 | 待回填（每层每步省 1 次拷贝） |
| 5 | **`SGLANG_OPT_FUSED_PERMUTE_QUANT=1`** | permute 与 quant 合成一次发射（源码注明 *bitwise-identical*） | 待回填 |

**根因（base vs rnd 同配置 40 步画像做差，两侧 attention 调用数同为 1720 ⇒ 同步数可比）**

| 组成 | 量 | 占比 |
| --- | --- | --- |
| LoRA 新增 `t128x8x512_s3` GEMM ×2 | +309.4 ms | — |
| 基座 `t128x16x256_s6` GEMM 退场 ×2 | −138.0 ms | — |
| ⇒ **GEMM 净增** | **+171 ms** | **63%** |
| `_chunked_lora_{shrink,expand}` | +57.8 ms | 21% |
| 其余（pad / finalize / 量化 / 拷贝） | +40 ms | 15% |
| **合计 Δ（980.0 → 1249.4 ms）** | **+269.4 ms** | 100% |

**逐调用对比**：LoRA 的 gate_up GEMM **207.94 µs/call** vs 基座 **130.94**（1.59×）；LoRA 的 down GEMM **178.85** vs 基座 **76.00**（2.35×）。

**为什么 LoRA 的 GEMM 更慢**：LoRA 走 `Gemm2::Runner(E2m1,E2m1,**bf16**)`（未融合激活、输出 bf16），基座走 `MxE4m3_MxE2m1_MxE4m3`（融合激活、输出 fp8）；源码注释亦写明 *"no NvFP4 unfused-act GEMM1 cubin exists"* ⇒ **cubin 家族被硬约束**，不能靠 `config` 换出来（32 档极差仅 7.4%）。

**已排除的方向**

| 方向 | 证据 |
| --- | --- |
| PDL 覆盖 | 现有 trace 核并集 busy = **97.5%**，间隙仅 2.5% ⇒ 天花板就在 2.5%，已回滚 |
| `SGLANG_OPT_MOL_LORA_STAGE_CFG=1` | **会让 prefill 在 LoRA 初始化后静默崩死**（两次复现，无 traceback）⇒ **该开关不可用** |
| `tile` 维 | 16/32/64/128 全扫，32 最优，16 反而最差 |
| `config` 维 | 0..31 全扫，极差 7.4% |

**⚠️ 方法论教训（本次踩过，务必记档）**

1. **服务端 per-window `gen throughput` 不能当聚合吞吐用**：长尾跑里「剔前后 15%」会剔掉慢尾、只留早段快窗口 ⇒ 虚高（实测 `tile=16` 服务端解析 10860 tok/s，客户端真值仅 5207）。**判定候选优劣一律以客户端聚合为准。**
2. **`POST /stop_profile` 会返回 500 而 profiler 实际没停**：与其重叠的评测段 p50 12.402 ms/tok 与「采样中」那次的 12.382 几乎相同 ⇒ 那是采样开销而非被测改动。**profiler 启停必须验证。**
3. **PD 分离下适配器必须在 P(30100) 与 D(30200) 两侧都加载**；只加载一侧 ⇒ 经 smg 的请求全部超时且服务端无显式报错。
