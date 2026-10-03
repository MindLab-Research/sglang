# 从 GLM-5.3 到 DeepSeek-V4.1：MoE-LoRA 分解路径的零开销工程

**——把"实验性外挂"改造成"位级正确、-0.3% 开销、149=147 tok/s"的完整技术报告**

> 作者注：本文记录 MindLab MoL 工程线在 `sglang-mol-v41`（分支 `release/dsv41-flash-mol`）上，将 DeepSeek-V4.1-Flash（MXFP4, SM100 trtllm-gen）的 MoE-LoRA 从不可用状态推到零开销生产可用的全部工程决策。所有结论均附 git 提交号、代码位置、实测数字，不引述未经验证的推断。

---

## 摘要

DeepSeek-V4.1-Flash 的原生 MoE 路径是单融合 kernel（permute→GEMM1→激活→量化→GEMM2→finalize 全部合入 NVIDIA 预编译 cubin），LoRA delta 需要在 GEMM1 之后、激活之前注入——融合 kernel 内部无注入点，**该路径天生不支持 LoRA**。可行解只有一条：铺分解路径，在自研 kernel 中携带注入点。我们fork了社区已有的实验性分解骨架（upstream PR #27329），在其上完成六项工程后达到：

- **正确性**：零权重适配器输出与 base **逐位相同**（greedy 文本全等，logprob |Δ| = 0.0000）；
- **性能**：4 路 LoRA 动态混合负载聚合输出 3,391 tok/s vs 无 LoRA 3,400 tok/s（**-0.3%**）；pass-through 统一路径与原生路径 149 vs 147 tok/s 零差距；
- **稳定性**：全链路 0 illegal-memory / 0 no-kernel-found / 0 CUDA error。

作为对照，同一骨架在 GLM-5.3 fp8 线（`sglang-b300-glm52`）上未经此轮工程，MoL 开销显著。本文第 5 节给出两者逐项 diff 的归因。

---

## 1. 引言：问题的结构

### 1.1 V4.1 为什么必须走分解路径

V4.1 每层 MoE 的原生执行是一次融合 launch：

```
MXFP4 fused kernel: permute + GEMM1(MxE2m1×MxE2m1) + gated-SwiGLU
                    + MXFP8量化 + GEMM2 + finalize   ——单 cubin，无中间出口
```

LoRA 的数学要求：在 gate_up 投影结果上叠加 `Δ = (x·A)·B·(α/r)`，再进激活；在 down 投影出口再叠一轮。注入点位于融合 kernel **内部**两处。cubin 是预编译产物不可改——这不是"支持得不好"，是**结构上无门**。

因此唯一解是重构执行序列为分解形态，让每一级中间量暴露出来，在暴露处开注入点。upstream 社区已有此设计（PR #27329, Yanbin Jiang, 2026-06-05，"Experimental fast LoRA path ... for FP8 and NVFP4 models"，Co-authors: fzyzcjy, Chunan Zeng）：`trtllm_lora_temp/` overlay 以 shadow 头文件的方式替换 flashinfer JIT 源，提供 `FP4BlockScaleLoraLauncher` 分解管线与 VE（virtual experts）分组 Triton kernel。**但该骨架只覆盖 FP8 与 NVFP4，且在 V4.1 的 MXFP4 形态上数值全错**（见 §2）。

### 1.2 GLM-5.3 线的现状（对照基线）

`sglang-b300-glm52` 使用 PR #27329 的 fp8 变体跑生产（16k git 历史实证：该目录下团队零提交，全部为 upstream 提交）。其 LoRA 请求路径为：

```python
# lora_dispatch.py:87 —— 无适配器请求早退原生 fp8 融合快路径
if not get_is_capture_mode() and not lora_info.has_active_lora:
    return fused_experts_fp8_sgl(...)          # 快路径
# 有适配器：走分解 op + 外挂 VE
# lora_dispatch.py:151/172 —— 每层 MoE 两个物化
activation_lora_input = torch.empty(...)       # 激活值物化（down-LoRA shrink 输入）
direct_down_output   = torch.empty(...)        # down 输出物化
```

其代价结构（§5 详述）：CUDA graph capture 无法分叉 → **MoL 开启后整个 decode batch（含无适配器请求）跌落出快路径**；分解 op 每层多两次独立 VE launch 与两次 HBM 物化往返；生产叠 EAGLE，draft 步再付一遍。

---

## 2. 基线骨架在 V4.1 上的三层失效

我们接手时的现象：LoRA 请求输出 tokenizer 乱码（`'<｜--PMnarrowaLF141oqDict'`），逐 token logprob 与 base 完全无关。定位出三层独立失效，每层均有明确物证：

**失效一：UE8M0 尺度因子被错误重排。** V4.1 激活侧尺度是 32-元素块、UE8M0 编码的线性布局 `[tokens, hidden/32]`。通用 permute kernel 唯一的 scale 通路（`moe::dev::permute` 的 `useDeepSeekFp8` 分支）按 **FP32、128-元素块**的线性布局索引（`idx = token + numTokens*(hidden/128)`）。喂入 UE8M0 张量后越界读、并把噪声字节散布进 SF 缓冲——GEMM 拿噪声当尺度，输出必然乱码。

**失效二：gated-SwiGLU 标量被丢弃。** V4.1 基路径以 OAI/GPT-OSS 三元组（alpha/beta/clamp_limit）运行激活：`xGlu=clamp(xGlu, max=L)`、`xLin=clamp(xLin, ±L)`、`out=xGlu·σ(α·xGlu)·(xLin+β)`。upstream 移植版 wrapper 对三个标量张量执行了 `(void)` 丢弃，激活退化为裸 `silu(xGlu)·xLin`——每层一步激活即偏离 base。

**失效三：GEMM 通路选错 cubin。** 骨架的 gate_up GEMM 把 bf16 激活直接喂入，走内部 E2m1 量化选择 `Gemm2::Runner(E2m1, E2m1, BF16)`——该组合的 tile-8 cubin 在官方包中**不存在**，运行时报 "No kernel found"。

---

## 3. 方法：六项工程

以下按依赖顺序叙述。全部落在 6 个提交（`92ec87926f` → `3b256ec8e7` → `6dcd29b13b` → `959880fb9d` → `a51fdb1a2d` → `2a0688082e`）。

### 3.1 E4m3 激活输入通路（修复失效三）

V4.1 的非 LoRA 路径在 Python 侧用 `per_token_group_quant(x, group_size=32, scale_ue8m0=True)` 预量化激活——与融合 kernel 内部消费的是**同一份字节**。分解路径必须复用这份字节，否则零适配器 parity 从起点就不可达。

实现：dispatch 侧沿用 `_prepare_flashinfer_mxfp8_activations` 产出 E4m3 + 线性 UE8M0 scale；overlay 对 `is_e4m3_input` 分支选取 `Gemm2::Runner(MxE4m3, MxE2m1, BF16)`，对应 cubin 族 `Bmm_MxE4m3_MxE2m1*` 存在且为官方产物。A 侧量化从 dispatch 进 C++，保证与基路径**逐字节同源**。

### 3.2 UE8M0 专用 gather+swizzle 内核（修复失效一）

数据面（E4m3 字节）继续用通用 permute（`useDeepSeekFp8=false`，不触 scale）；尺度面不再借用任何通用路径，新写 `sgl_gather_swizzle_ue8m0_sf`：

```
对每个 (token, k) → permutedIdx = expanded_idx_to_permuted_idx[...]
R128c4（tile≥128）: offset = sfBlkIdx·512 + ((p%32)·4 + (p%128)/32)·4 + blk%4
R8c4 （tile<128）: offset = sfBlkIdx·32  + (p%8)·4 + blk%4
```

两个公式与 dev_kernel 的 `getSfOffset` 及 `get_sf_out_offset_128x4/8x4` 逐项同源——即 GEMM 消费端自己定义的权威布局。padding 行不写（GEMM 尾部行不进入有效输出，与 bf16 permute 同先例）。

### 3.3 gated-SwiGLU 标量契约（修复失效二）

将三个标量张量经 `FP4BlockScaleLoraLauncher` 构造函数注入 `activation::Data` 的 `gatedActAlphaPtr/BetaPtr/ClampLimitPtr`，kernel 内按 gatedSiluScalars 的原始指令序列消费。V4.1 实际生效的是 clamp（alpha/beta=None 走恒等值），但契约完整保留三元组。

### 3.4 第二代融合洞：注入+激活+量化单内核

至此还有一处**结构级**舍入差：unfused 链先把激活行物化为 bf16（activation kernel 写出、`invokeMxFP8Quantization` 再读入），多一次 RN 舍入——它会扰动每 32 块的 amax，在 `__nv_cvt_float_to_e8m0` 的 **PosInf 边界**上翻转块指数。新内核 `fused_activation_mxfp8_quant.cuh` 复刻 gatedSiluScalars 的逐元素 fp32 数值，但**从 fp32 寄存器直接量化**：

```
SFValue = amax · reciprocal_approximate_ftz(448)
sf.__x  = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf)
scale   = 1/sf；  q = clamp(x·scale, ±448) → __nv_cvt_float2_to_fp8x2(RN, E4M3)
```

即 `cvt_warp_fp16_to_mxfp8` 的原始配方逐条落进自研 kernel（含 `__nv_cvt_float2_to_fp8x2` 的双元素构造——单元素 `__nv_cvt_float_to_fp8` 返回裸存储字节，不匹配类型系统）。LoRA 契约的 `activation_lora_input`（down-LoRA shrink 输入）仍按 bf16 写出——它只进 LoRA 分支，不污染基数值。

### 3.5 pass-through 位级统一（结构决策）

原 dispatch 对无适配器请求早退原生融合 kernel（与 GLM 线同构，见 §1.2）。这使"零权重适配器 == base"在结构上不可达：融合 kernel 的 GEMM1 epilogue 从不物化 gate_up，而分解路径 Gemm2::Runner 写 bf16——每层舍入序列不同，greedy 必然发散（实测：双方各自确定性成立，但互不相同，首分歧在位置 2-19）。

决策：移除早退，无适配器请求走**同一分解管线**，以零填充 delta 缓冲代替 VE 计算。位级依据：零权重 VE 的 delta 恰为 0.0，与 pass-through 的 `zero_()` 字节相同，注入处执行 `x + 0.0 → x`，IEEE-754 下逐位恒等。实测验证：`zero_2139`（21-39 层全零适配器）与 base 的 greedy 文本**全等**，逐位 logprob 差 0.0000。

### 3.6 适配器层契约与 KV 无关性

V4.1 的 persistent KV 仅由 kv_source 层 (2,8,14,20) 的 compressor/indexer 写入；非 source 压缩层的 SWA 每次前向重建、不入 cache。据此把适配器权重约束在 **21-39 层**（最后 source 层之后），则任意 LoRA 不触碰任何缓存 KV——`lora_kv_shared: true` 下 radix/HiCache 跨适配器共享，前置命中实测 4.2×（冷 1.26s→热 0.30s）。

---

## 4. 实验

**环境**：1101 单机 8×B300，P(GPU0-3,TP4)+D(GPU4-7,TP4)+sgl-router；D 端 decode CUDA graph max_bs=128；负载为 `sglang.benchmark.serving` random 数据集（4K in / 2048 out / 96 并发齐发，乱码 prompt 保证输出拉满）。适配器：4 个随机初始化（L0-L3，rank 16/α 32，21-39 层）。

### 4.1 正确性

| 验证项 | 结果 |
|---|---|
| zero 适配器 vs base（贪心 128 token） | 逐字全等；`output_token_logprobs` 逐位 Δ=0.0000 |
| 零增量恒等推理（6×7 类短题） | 与 base 一致（'42' 等） |
| 随机适配器扰动生效 | 首 token logprob -0.167→-0.746，五路输出互异 |
| 服务器错误（全链路） | 0 |

### 4.2 性能矩阵

| 维度 | 聚合输出 tok/s | TPOT p50 | E2E p50 | 备注 |
|---|---:|---:|---:|---|
| base（无 LoRA） | 3,400.5 | 14.46ms | 17.4s | pass-through 统一路径 |
| **L0-L3 uniform 混合** | **3,391.0** | **14.35ms** | 17.6s | **-0.3%，噪声带内** |
| MoL harness 全链路 | 1,809.9 | — | 32.7s | 开销 47%，全部来自 route hop 的额外 4.2K prefill（P 端负载翻倍），非 LoRA 计算本身 |
| base + dSpark | 5,367.9 | 6.49ms | 9.6s | +58%（另文） |
| pass-through vs 原生融合（单流） | 149 vs 147 tok/s | — | — | 分解路径零代价 |

LoRA 增量本身的成本：VE 分组 Triton kernel 相对主 GEMM FLOPs ≈ 1.4%（rank 16），注入融合于 activation/finalize kernel 内零额外 HBM 往返，整段（含 VE）入 decode graph——三项叠加即为 -0.3% 的全部来源。

---

## 5. 与 GLM-5.3 实现的对照归因

同一骨架（PR #27329）在两条产品线上呈现的性能差异，逐项归因如下：

| 维度 | GLM-5.3（glm52，fp8 第一版形态） | DSV4.1（本文） |
|---|---|---|
| delta 注入 | gate_up delta 经 op 参数传入（op 内消费），**down delta 输出后外挂 add** | 两级注入均在自研 kernel 内寄存器闭环 |
| 每层物化 | `activation_lora_input` + `direct_down_output` 各一次 HBM 往返 | activation_lora_input 保留（LoRA 契约），主数值物化为零（§3.4） |
| 无适配器请求 | 早退 fp8 融合快路径；**capture 不能分叉 → 开 MoL 后全体跌落** | 同一管线（§3.5），无跌落概念——MXFP4 本就只有这一条路 |
| 数值契约 | 无 zero==base 要求（无此测试） | 位级 parity 是验收标准，倒逼 §3.1-3.4 全部同源化 |
| spec 叠加 | EAGLE 生产，draft 步再付一遍路径成本 | dSpark 独立验证 +58%，未与 LoRA 联测（后续工作） |

**结论**：GLM-5.3 的开销并非骨架设计错误，而是**第一版（experimental）形态未经本文 §3 的执行深度**，叠加 fp8 存在更强快路径的"跌落"结构。三项可回移：delta 内嵌去物化、finalize 融合 down delta、graph 整段核验；根治跌落则需为 fp8 融合 op 本身开 delta 注入（对应本文在分解路径开洞的思路，工程上需 vendor 该 op 源码）。

---

## 6. 归属声明（git 实证）

| 成果 | 归属 | 证据 |
|---|---|---|
| 分解管线骨架、VE、delta 注入点接口 | upstream PR #27329（Yanbin Jiang；Co-authors fzyzcjy, Chunan Zeng） | `git ls-tree 81f27fb3a7` 确认 `gateUpLoraDeltaPtr` 等 24 处在 base 已存在 |
| V4.1 MXFP4 适配、三项数值修复、融合量化内核、pass-through 统一 | 本文团队（6 commits, mindlab-bot） | base 无 `fused_activation_mxfp8_quant.cuh`（ls-tree 空）；glm52 同目录团队零提交 |

## 7. 结论

零开销不是调参结果，是四条不变式的产物：**激活字节与基路径同源**（§3.1）、**尺度布局与消费端同源**（§3.2）、**激活标量契约完整**（§3.3）、**量化舍入序列唯一**（§3.4）——在此之上，pass-through 统一把"LoRA 开销"归约为一个可位级验证的算术恒等式（x+0.0=x），使全部后续优化获得免费回归测试。149=147 tok/s 与 -0.3% 证明：**分解路径不必然是慢路径；为 LoRA 铺的路，可以修到与无 LoRA 的路完全一样快。**

---

*原始数据：`/root/bench_pd/.tmp/sglangbench_long*.jsonl`、`mol_harness_results.json`；性能汇总另见 `docs/PD_PERF_REPORT_20261003.md`。*
