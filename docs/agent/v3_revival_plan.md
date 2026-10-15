# v3 路径复活 + fusedAct 方案设计（2026-10-15 下 session 第一优先）

## 背景数据（今日统一口径实测）
- base: TPOT 4.961 / acc 3.97 / per-step 19.70
- dzgu_s4（分解链）: TPOT 7.673 / acc 3.51 / per-step 26.88
- **LoRA 真实开销 7.07ms/step（36%）**——唯一大杠杆 = 分解链 vs fused 路径差 ~4.7ms/step

## 方案 v3 复活（最优先——从未真测）
v3 分支（moe_overlap.py:1055，bf16-bmm MoE-LoRA）被关的唯一原因：`dequant_mxfp4`（mxfp4.py:279）
是 AMD-only（quark import 必挂）。v3 的 bmm GEMM 路径 **从未在新进程上测过收益**（旧数据全假）。
若 bmm 比分解 trtllm 快 → 可能直接接近 6ms。

### 实施清单
1. **写 NVIDIA torch 版 dequant**（一次性权重转换，性能无关紧要）：
   - w13_weight: packed e2m1（int8 view，每 byte 2 个 fp4——lo/hi nibble，LUT 16 值查表 + interleave）
   - w13_weight_scale: **e4m3x2 格式**（需考古：moe_overlap.py:1103 `_as_fp8(w13_weight_scale)` 是
     view(float8_e4m3fn)——具体 shape/packing 看 mxfp4.py 的 process_weights 与 kernel 调用）
   - w2_weight_scale: ue8m0（byte → 2^(byte-127)）
   - 参考: mxfp4.py:653-655（DeepGEMM 的 packed 处理）、:830-832（格式注释）
2. moe_overlap.py:1058 的 `from ...mxfp4 import dequant_mxfp4` → torch 版
3. launch: `SGLANG_LORA_DEEPGEMM_V3=1`（当前被注释，v3=死代码）
4. R38: 重启 + bench + 三发（v3 bmm vs 分解 trtllm 首次真对比）
5. **注意**: deep_gemm 已回退 0.2.0（masked/contig 数值 bug——v3 的 bmm 路径不依赖 deep_gemm
   grouped GEMM（v3_gemm_stage 用 bmm）→ 不受 0.2.0 bug 影响）

## fusedAct 恢复方案（已论证死结——不推荐）
- 死结: stock fused kernel（_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen）完全内部化
  （GEMM1+SwiGLU+quant+GEMM2 一颗）——**无中间值可捕获** → down VE 无输入
- dequant 近似也不可行（连 quant 后的值都不物化）
- **结论: 分解链税不可避免，除非 down VE 改数学（见 v3）**

## v3 的结构优势（为何可能快）
- v3_gemm_stage: bf16-bmm（chunked GEMM）+ act + delta + 激活捕获天然可得（物化中间值）
- 与分解 trtllm 的差异: GEMM 引擎不同（bmm vs trtllm-gen grouped）——**bmm 在小 M（decode）下
  的 launch 效率可能显著优于 grouped trtllm**（之前 R21 时代的 bmm 数据是假进程——无效）

## 铁律提醒（AGENTS.md 已固化）
- 输出正确性先于一切数字；per-step 口径（TPOT×acc）；层 20-39 不变；同轮 A/B；
  出错先找根因不盲目重启；报告前自检（输出/成功率/层配置/加载）
