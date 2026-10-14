# DSV4.1-Flash MoE-LoRA 融合核设计（全部融合）

目标：把 LoRA 路径的 **per-layer 6~8 次发射** 收敛为 **2~3 次**，并消除 **bf16 物化** 与 **慢 cubin 家族**。

## 0. 为什么要融合（实测依据）

base vs rnd 同配置画像做差（两侧 attention 调用数同为 1720 ⇒ 同步数可比）：

| 组成 | 量 | 占 Δ |
| --- | --- | --- |
| LoRA 新增 `t128x8x512_s3` GEMM ×2 | +309.4 ms | — |
| 基座 `t128x16x256_s6` GEMM 退场 ×2 | −138.0 ms | — |
| **GEMM 净增** | **+171 ms** | **63%** |
| `_chunked_lora_{shrink,expand}`（VE delta） | +57.8 ms | 21% |
| 其余（finalize / 量化 / permute / 拷贝） | +40 ms | 15% |
| **合计 Δ** | **+269.4 ms** | 100% |

逐调用：LoRA gate_up GEMM **207.94 µs/call** vs 基座 **130.94**（1.59×）；LoRA down **178.85** vs 基座 **76.00**（2.35×）。

**根因**：基座融合核把 gate_up 累加器**留在寄存器**，直接 act+量化（出 fp8，1 字节）；
LoRA 必须**物化成 bf16**（2 字节）才能在外面加 delta ⇒ **写带宽翻倍**，且被逼进 `t128x8x512_s3` 家族
（源码原话：`"no NvFP4 unfused-act GEMM1 cubin exists"`）。

## 1. 融合后的执行流

```
[permute+quant]                      ← 复用 fused_permute_quant.cuh（F1：SGLANG_OPT_FUSED_PERMUTE_QUANT=1）
[GEMM1 ‖ epilogue]                   ← ★ 新增（本设计）:
      acc(fp32, 寄存器) = W_gate_up(fp4) · x(mxfp8)
      acc += gate_up_lora_delta        （VE 产出，bf16）
      act = gatedSiluLoRA(alpha,beta,limit; acc)
      → mxfp8(E4M3) + 对应 SF(swizzled) 直接写给 GEMM2 用     ← 不再物化 bf16
[GEMM2 ‖ epilogue + finalize]        ← ★ 新增（本设计）:
      acc(fp32) = W_down(fp4) · act(mxfp8)
      acc += down_lora_delta
      → 按 expanded→permuted→token 映射 scatter-reduce 写回输出（bf16）
```

## 2. 逐跳数据契约（与现有实现逐字段对齐）

| 张量 | dtype | 形状 / 布局 | 产出者 |
| --- | --- | --- | --- |
| `hidden_states` | mxfp8 E4M3 + SF | `[num_tokens, hidden=5120]`，SF 128×4 / 8×4 swizzled | 上游投影（已有） |
| `permuted_e4m3` / `permuted_e4m3_sf` | E4M3 + UE8M0 SF | `[max_padded, hidden]` | permute（或 F1 融合核） |
| `w_gate_up` | fp4 E2M1 + SF(1×32) | `[num_local_experts, 2*inter=1280, hidden]`（TRT-LLM 洗牌后） | 权重加载（已有） |
| `gate_up_lora_delta` | bf16 | `[max_padded, 2*inter_ve=1152]`（**unpadded**；尾部 128 列恒 0） | VE gate_up delta（已有） |
| `act_mxfp8` / `act_sf` | E4M3 + UE8M0 | `[max_padded, inter=640]`（padded） | **本设计 GEMM1 epilogue** |
| `w_down` | fp4 E2M1 + SF | `[num_local_experts, hidden=5120, inter=640]` | 权重加载（已有） |
| `down_lora_delta` | bf16 | `[max_padded, hidden]`（unpadded 576 切片的产物） | VE down delta（已有） |
| `output` | bf16 | `[num_tokens, hidden]` | **本设计 GEMM2 epilogue+finalize** |

**数值一致性要求（必须逐条满足，否则与 base 发散）**
1. gated-SwiGLU 用 **OAI 三标量**：`xGlu=clamp(xGlu,max=limit)`、`xLinear=clamp(xLinear,±limit)`、
   `out = xGlu*sigmoid(alpha*xGlu)*(xLinear+beta)`，`alpha=1.702, beta=1.0, limit=7.0`，**per-expert 标量读 index 0**。
2. E4M3 量化必须复刻 `cvt_warp_fp16_to_mxfp8`：`SFValue=amax*reciprocal_approximate_ftz(448)`；
   `E8M0 = __nv_cvt_float_to_e8m0(SFValue, SATFINITE, cudaRoundPosInf)`；
   `outputScale = SFValue!=0 ? reciprocal_approximate_ftz(SFValue) : 0`；
   元素 `__nv_cvt_float2_to_fp8x2(sat(±448), SATFINITE, E4M3)`；SF 用 `get_sf_out_offset_128x4/8x4` 写。
3. gate_up 累加必须在 **fp32 寄存器**里完成（与基座一致），**不得**先落 bf16 再算。
4. down 的 LoRA delta 必须在 **finalize 之前**加到 fp32 累加上。

## 3. 核签名（本设计产出）

```cuda
// GEMM1 + 融合 epilogue
void sgl_moe_lora_gateup_fused(
    int num_tokens, int max_padded, int hidden, int inter, int inter_pad,   // inter=576, inter_pad=640
    uint8_t const* permuted_e4m3, uint8_t const* permuted_e4m3_sf,          // A (mxfp8 + SF)
    uint8_t const* w_gate_up, uint8_t const* w_gate_up_sf,                  // B (fp4 + SF), 洗牌后布局
    __nv_bfloat16 const* gate_up_lora_delta,                               // [max_padded, 2*inter]
    int32_t const* sorted_token_ids, int32_t const* expert_ids, int32_t const* num_tokens_post_padded,
    float const* alpha_ptr, float const* beta_ptr, float const* clamp_limit_ptr,
    uint8_t* act_e4m3, uint8_t* act_sf,                                    // 输出：mxfp8 激活 + SF
    cudaStream_t stream);

// GEMM2 + down delta + finalize
void sgl_moe_lora_down_fused(
    int num_tokens, int max_padded, int hidden, int inter_pad,
    uint8_t const* act_e4m3, uint8_t const* act_sf,                        // A
    uint8_t const* w_down, uint8_t const* w_down_sf,                       // B
    __nv_bfloat16 const* down_lora_delta,                                  // [max_padded, hidden]
    int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_token_idx,
    float const* topk_weights, int top_k,
    __nv_bfloat16* out,                                                    // [num_tokens, hidden]
    cudaStream_t stream);
```

## 4. 实施路径（可独立验收的里程碑）

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| **M1** | GEMM1 融合核（正确性优先，单 tile 版） | 与现有 = `Gemm2::Runner` + act 核 的输出**逐元素比对**（fp8 位级 / 激活 bf16 容差） |
| **M2** | GEMM2 融合核 + 并入 finalize | 端到端输出与现有路径比对 |
| **M3** | 接进 `FP4BlockScaleLoraLauncher`（开关 `SGLANG_OPT_MOL_FUSED_GEMM=1`） | 冒烟 + md5 对比 |
| **M4** | 性能调优（tile / swizzle / TMA / PDL / `tcgen05`） | n=256 客户端聚合 vs 现有 (~7900 tok/s) |

**技术风险（按严重度排序）**
1. **fp4 权重的 TRT-LLM 洗牌布局**：`w_gate_up` 在显存里是洗牌过的（`Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel`），自写 GEMM 必须按同一布局读，否则数值错。
2. **SF 布局契约**：A 侧 SF 与 B 侧 SF 的 swizzle（R128c4/R8c4、1×32）必须与 epilogue 写出的 SF 一致。
3. **`max_padded` 的越界读**：masked/免 padding 方案要保证 padding 行不参与 finalize（现有实现有专门处理）。

## 5. 现有可复用清单（逐行可搬）

| 组件 | 位置 | 用途 |
| --- | --- | --- |
| `gatedSilu` / `gatedSiluLoRA` | `dev_kernel.cu:60-98` | 融合激活 + OAI 三标量 + LoRA delta |
| delta 就地加（标量/矢量） | `dev_kernel.cu:155-162, 472-478` | `x += gateUpLoraDeltaPtr[...]` |
| `activationDeepSeekKernel` | `dev_kernel.cu:375-624` | 参考数据流（含 e4m3 输出） |
| E4M3+E8M0 量化 + SF swizzle | `fused_activation_mxfp8_quant.cuh:186-226` | epilogue 量化段 |
| `permuteKernel` | `dev_kernel.cu:832` | 若不做 F1 融合 |
| `finalizeKernel(VecLoad/DeepSeek)` | `dev_kernel.cu:912/1066/1182` | GEMM2 epilogue 内联 |
| `get_sf_out_offset_128x4/8x4` | `trtllm::gen`（vendored） | SF 写偏移 |
| `is_arch_support_pdl()` | `kernels/jit/utils/arch.py` | PDL（可选，注意：全局 busy 已 97.5%，PDL 收益有限） |

---

## 6. 勘查结论（2026-10-12，两份只读勘查）

### 6.1 三条决定性事实（都改变了原计划）

1. **仓库内没有任何 TileLang MoE GEMM**：全仓 `*_tl.py`/`*_tl.cu` = 0，`tl.gemm` = 0；tilelang 只用于 attention / layernorm。
   ⇒ **原计划「用 TileLang 写 GEMM」在仓库内无料可抄**，需改成 CuTeDSL / CUTLASS / Triton。
2. **仓库内没有自实现的 A=fp8 × B=fp4 分组 GEMM**。唯一精确匹配在**第三方 DeepGEMM**：
   - `srt/layers/quantization/mxfp4.py:386` 原文：*"True W4A8: DeepGEMM fp8_fp4 grouped GEMM (SM100 MXF8F6F4 UMMA). Weights stay MXFP4 (e2m1 + ue8m0 g32, zero requantization); activations are quantized to fp8 per-token-group-128."*
   - `srt/layers/moe/moe_runner/deep_gemm.py:332-333`：`recipe_a=(1,128)`（激活 fp8 g128）、`recipe_b=(1,32)`（权重 fp4 g32）
   - 调用形态：`grouped_gemm_nt_f8f8bf16_masked(lhs=(hidden_fp8, act_scale), rhs=(w13_fp4_as_int8, w13_scale_ue8m0), out, masked_m, expected_m, recipe_a=(1,128), recipe_b=(1,32))`
   - **该分支直接消费 checkpoint 布局（no padding）**（`mxfp4.py:461-465`）
   - 仓库内已有的 CUTLASS 分组核（`es_sm100_mxfp8_blockscaled_moe_group_gemm*`）是 **MXFP8×MXFP8**（`ElementA/B` 都是 `float_e4m3_t`，`traits.cuh:44/49/54`）；要让它吃 fp4 必须把 `ElementB` 换 `cutlass::mx_float4_t` 并改 SF 布局 —— **这就是「从零写」的部分**。
3. **仓库里已经有一颗「分组 GEMM + LoRA shrink/expand + routed weight + add」的 Triton 核**：
   `kernels/ops/moe/fused_moe_lora_kernel.py:50 _fused_moe_lora_kernel`（配套 `:210 _fused_moe_lora_shrink`、`:308 _fused_moe_lora_expand`、`:411 _fused_moe_lora`）。
   ⇒ **融合核的最省力骨架**。

### 6.2 可照抄模板（最终清单）

| 用途 | 抄哪里 | 理由 |
| --- | --- | --- |
| **分组 GEMM + LoRA 全流程（首选骨架）** | `kernels/ops/moe/fused_moe_lora_kernel.py:50` | 已是「分组 GEMM + shrink/expand + routed weight + add」，自带 masked/route 处理 |
| 分组 MoE blockscaled GEMM（CUTLASS） | `kernels/jit/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm.cuh:50` + `traits.cuh:31/68` | 仓库内唯一完整分组 MoE blockscaled 模板；`CustomEVTIdentity` 即 epilogue 挂点 |
| GEMM+epilogue 三段结构（CuTeDSL） | `kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:332/430/633/693/742` | prologue/主体/epilogue 分离，act 就是一个 `if const_expr` 分支 |
| act+quant epilogue（Triton） | `kernels/ops/layernorm/mxfp8_epilogue.py:37`（swizzle 公式 `:52`） | 通用、注释全、含 FlashInfer 128×4 swizzle |
| act+quant epilogue（CUDA） | `kernels/ops/moe/ep_moe_kernels.py:505`；`kernels/ops/moe/dsv4.py:192/219` | silu+mul+per-group fp8+packed UE8M0 一趟 |
| **delta add（VE，可内联）** | `kernels/ops/moe/virtual_experts.py:1024`（rank-chunk 循环 `1097-1140`） | 单 launch「shrink+expand+add」，**纯 register-local、零 CTA 间通信** |
| 权重打包/SF 布局准备 | `srt/layers/quantization/mxfp4.py:621-680`；`srt/layers/moe/mega_moe.py:458-524` | fp4 打包 + sf transform + UTCCP 转置完整前置 |
| 格式定义（A/B recipe） | `srt/layers/moe/moe_runner/deep_gemm.py:290-335` | recipe_a/recipe_b 语义 |

**VE delta 内联可行性（已核实）**：`_fused_lora_delta_kernel`（`:1024`）每 program 处理一个 `(pid_m, pid_n)` tile，两次 `tl.dot`、fp32 acc、`tl.where(has_lora, base+delta, base)` 后 store —— **无跨 CTA 通信**（旧 split-K 版的 atomic 在 `:391`，新核不带）。路由是**前置独立核**（`moe_align_block_size` / `_align_block_size_jit:445`），融合核可直接复用。

**现成挂点**：`srt/lora/lora_moe_runners.py:305 _add_lora_gate_up_delta`、`:400 _add_lora_down_delta`、`:482 build_lora_hooks`（`after_gate_up` `:520` / `after_down` `:540`）。

### 6.3 实现路径（改按代价从低到高）

1. **Triton 融合（首选，最快见效）**：以 `fused_moe_lora_kernel.py:50` 为主体，在 K-loop 后 / store 前插入三段 ——
   (a) `acc → act`（该核内已有 `accumulator += tl.dot(...)`）；(b) `quant`（抄 `mxfp8_epilogue.py:37`）；(c) `delta add`（抄 `virtual_experts.py:1024` 的 `tl.where` + rank-chunk）。
2. **走 DeepGEMM（省自写 GEMM）**：在 `deep_gemm.py:695 _run_masked_gemm` 的 gateup GEMM（`:771`）之后、`_varlen_deep_gemm_silu_mul_quant` 之前内联 after_gate_up 逻辑。
3. **自写 CUDA/CUTLASS**：抄 `es_sm100_mxfp8_blockscaled_moe_group_gemm.cuh:50` + `traits.cuh:68 CustomEVTIdentity`，`ElementB` 换 `mx_float4_t`，EVT 换 act+quant+delta callback。

### 6.4 三条硬风险（按严重度）

1. **格式/布局不匹配才是真正的「从零」**：无自实现 A=fp8×B=fp4 分组核。必须自己搞定 e2m1 打包（2/byte，`view(torch.int8)`）、ue8m0(g32) SF、`packed-UE8M0 TMA-aligned MN-major` swizzle、`transform_sf_into_required_layout(recipe=(1,32))`。⇒ **先做「单 expert 对拍」最小验证，别一上来全 MoE。**
2. **两套路由语义不能混**：base GEMM 路由（deep_gemm 的 `masked_m/src2dst`；triton 的 `sorted_token_ids/expert_ids`）与 VE delta 路由（`virtual_topk_ids` + `align_block_size`）**不同**。融合必须在同一 tile 划分下共享路由；仓库为此踩过多次：`-1`/`2**30` sentinel（`virtual_experts.py:869-876`）、Blackwell 掩码越界 IMA（`:1060-1092`）、`_invoke_fused_lora_delta:1245` 的 OOB 断言。
3. **数值/位级一致性**（zero-LoRA 必须 == base）：
   - act 必须 **fp32 寄存器直接量化**，不得经 bf16 中转（`fused_activation_mxfp8_quant.cuh:3-13` 头注释：bf16 中转会翻 UE8M0 指数）
   - VE rank 中间量有一次 bf16 round（`virtual_experts.py:1097` `intermediate.to(...)`），内联后必须保持同一舍入序
   - e8m0 必须正舍入（`mxfp8_epilogue.py:25` / `deep_gemm.py:96`）

（次级：tile 约束 `256×128×128` + 2SM cluster `(4,1,1)`/`(2,1,1)`；`mega_moe` 路径形状门槛 `align=512` ⇒ `hidden/inter` 需整除 —— 本模型 inter=576/640 **不满足**，故 mega 路径不可用。）
