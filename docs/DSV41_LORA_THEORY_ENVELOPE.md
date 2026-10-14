# DSV4.1-Flash MoL LoRA：理论包络与异常定位（理论轮 · 2026-10-13）

> 用户命题：rank=32/16 的 LoRA 数学量极小，实测「LoRA GEMM 占比异常高」一定有问题；
> 优化关键藏在这个高比例里，先弄明白再优化。本文 = 理论预言 + 异常定位 + 可判别签名。

## 1. 理论包络（conc=64, topk=8, A=512, K=5120, I=576/rank, E=384, 20 LoRA 层）

| 项 | 每层 | 20 层合计 | 依据 |
|---|---|---|---|
| LoRA 数学（shrink+expand, rank32） | 1.96e8 MAC = 0.39 GFLOP | 7.8 GFLOP | 张量核 ~0.5µs/层；全 CUDA 核 ~5.6µs/层 |
| LoRA 权重流量（A+B bf16） | ~0.77 MB | ~15 MB | @8TB/s ≈ 2µs 总计 ≈ 0 |
| 中间物化流量（GEMM1 out/act/quant/delta 各 w+r） | ~6.6 MB | ~132 MB | ~17µs 总计 |
| 合理结构税（fused delta 核启用 + 桥消除 + routing 复用后） | 10-20 µs | 0.2-0.4 ms | launch ~1-2µs/个（图内）+ 少量 D2D |

**理论包络：LoRA 应有开销 ≈ 0.35-0.55 ms ⇒ rnd 应为 5.75-5.95ms。**

## 2. 异常定位

- 实测：base 5.414ms / rnd 7.281ms ⇒ 开销 **1.867ms（93µs/层）= 包络的 4-5 倍**。
- 低并发 trace 归因（§12.9/12.10，口径待核）：GEMM 净增 63%（≈1.19ms）/ VE delta 21%（0.40）/ 杂项 15%（0.28）。
- **同一批专家权重、同样的 MAC，挂 LoRA 后 GEMM 链理论上不该变慢——该分量的任何实测回归都是纯异常。**

## 3. 异常假设与预注册签名

| # | 假设 | 机制 | nsys/数据签名 | 修复形态 | 预期收益 |
|---|---|---|---|---|---|
| H1 | 内核选择退化【最强嫌疑】 | lora 变体 GEMM1/2 选了不同 cubin/tile_m ⇒ m=1-8/expert 时 cta 更少 ⇒ 权重流式并行度打折 | lora 侧同名 GEMM duration ≈1.5-2× base；dram bytes 相近；grid/cta 更少 | launcher 选择逻辑/参数（天级） | ~1.0-1.2ms |
| H2 | 权重布局劣化 | 4-D block-shuffled vs flat ⇒ 非合并读 | dram bytes ≫ 理想 1.13GB/层 | 布局对齐 | 视实测 |
| H3 | 图变体/拆批 | lora 图 capture 配置差或 adapter 未就绪拆批 | graph 外 kernel；steps/seq 异常 | 图配置/就绪门 | 视实测 |
| H4 | finalize 重复触碰 | do_finalize 重读权重/激活 | 某 GEMM dram bytes ≈2× 理想 | C++ 参数 | 视实测 |
| H5 | DSPARK 交互 | LoRA 改变 accept length ⇒ 每步 m 失真 | base vs rnd accepted tokens/step 不同 | 无（口径修正） | 口径 |
| H6 | 63% 不迁移 | GEMM 链没变慢，93µs/层全是结构税（VE42+桥13+alloc7+act/quant10+launch10≈92，账自洽） | nsys 显示 GEMM 时长 base≈lora | C3'+P-B+routing 统一 | 0.6-0.9ms |

## 4. 裁决与分叉

- **H1/H2 成立** ⇒ 主刀 = 让 LoRA 路径用回 base 的 GEMM 配置，delta 旁路注入（gate_up_lora_delta 入参已存在，问题在它强制分解变体）——**W2-b 可能从"周级自写 cubin"降级为"天级改选择逻辑"**。
- **H6 成立** ⇒ 结构税自洽 ⇒ C3'（SGLANG_LORA_FUSED_DELTA=1）+ P-B 去桥 + routing 统一 ⇒ 回收 0.6-0.9ms，叠加 PDL 兜底 ⇒ W2-a 路线。
- ⚠️ **C3' 地雷（2026-10-13 主持人自查）**：`SGLANG_LORA_FUSED_DELTA` 默认 OFF 有记载原因（environ.py:1127-1129）——**fused delta 核仅 rank-16 可用；rank 32 会 Triton IMA/Xid 崩 prefill；且其他 rank 更慢**。DSV4.1 MoL 栈 = rank 32 ⇒ C3' 不能直接翻 flag：
  - C3'-r16：rank-16 adapter 探针（验架构收益，不可直接部署 rank 32）；
  - 或新增 **F1'：fused delta 核 rank-32 IMA 修复**（新工作项，修好后 C3' 才成立）。
- 两条路都指向 rnd ≤6ms，施工图完全不同 ⇒ W1 实验矩阵按「H 判别」重排（见 §5）。

## 5. W1 矩阵修订（终版 2026-10-13，证据回填后）

原 W1 的「成分归因」已被盘上 trace 考古（§6.1）基本完成（逐核 µs/call 分解）。修订后 W1 = 确认 + 验收：

| # | 内容 | 判据 |
|---|---|---|
| W1-a | B300 conc=64 现场 nsys（dram bytes 补测）→ 终判 H2 残差 | lora 侧 GEMM1 dram bytes ≈2-4×（写家族证实）vs 相近（另有 stall） |
| W1-b | tactic sweep（零重建，/tmp/lora_tactic.txt）：tile_N×config 全扫 | GEMM2 178.9→≤120µs、GEMM1 207.9→≤190µs ⇒ 烘焙 env |
| W1-c | A1 融合 cubin 后双账本验收 | ① LoRA 净增消失（+171ms@低并发 ⇒ conc=64 复算归零）②共享 gate_up 回到 ~130.9µs/call |
| W1-d | 结构税项：F1'（rank-32 IMA 修复）+ P-B（SGLANG_LORA_DOWN_NOCONTIG，G1 闸后开）+ routing 统一 | G1 zero==pass-through 位级 → G2 M2.5 数值 → e2e A/B |
| P2 结案 | Route C 判死（dot_scaled 仿真 1565µs ≫ bar 29µs），M_gemv 实测即结案 | microbench 已归档 |

前置：R6 已预清（cuda-graph-max-bs-decode=128 ≥ 64）；全部门禁 G0→G1→G2→perf→G3→G4 维持。

## 6. 证据附录

### 6.1 盘上 trace 考古（th-bench，2026-10-13）——【异常机制已钉死】

数据源：`/root/prof2_base`（10-12 16:32）vs `/root/prof2_rnd`（10-12 16:39），L20、TP0、40 步 GPU trace，两侧 attention 调用数均 1720（同步数可比）。

逐核 µs/call（base → rnd）：
| kernel | base | rnd | 判读 |
|---|---|---|---|
| attention flash_mla（1720/1720） | 63.68 | 63.60 | 完全一致 ⇒ **H5 出局**（accept length 无差） |
| gate_up `t128x16x256u2_s6`（量化 GEMM，两路径共用） | 130.94 | 156.67 | **+19.7%，grid 相同 ⇒ 共享基座核被干扰**（L2/带宽，~13% Δ） |
| down `t128x16x256_s6`（共用） | 76.00 | 83.37 | +9.7%（同上） |
| **LoRA 独有 gate_up `t128x8x512u2_s3`（bf16）** | — | **207.94** | **= 1.59× 量化路径 ⇒ 20 个 LoRA 层的 GEMM1 整体降级 bf16！** |
| **LoRA 独有 down `t128x8x512_s3`（bf16）** | — | **178.85** | **= 2.35× 量化路径** |
| shrink / expand | — | 13.24 / 10.44 | LoRA 数学本身极小 ✓（用户判断正确） |
| finalize_all_reduce（1600 call 两侧） | 26.81 | 25.84 | ≈0 ⇒ **H4 出局** |
| dense_gemm | 9.57 | 9.59 | ≈0 |

**根因（机制已实锤）**：LoRA 变体需要 unfused-act GEMM1（delta 注入 pre-SwiGLU 要求物化 GEMM1 输出），而**量化家族（MxE4m3/NvFP4）不存在 unfused-act GEMM1 cubin**（§12.12 config 0..31 极差仅 7.4% + 源码注明）⇒ 20 个 LoRA 层的 MoE GEMM 静默降级到 **bf16 t128x8x512 路径**（per-call 1.59×/2.35× 慢）。

**每步账目（in-kernel Δ ≈ 6.7ms/步，与 ΔTPOT×accept ≈ 6.5ms 吻合 ⇒ ~96% 在核内）**：
- bf16-vs-量化 GEMM 回归（20 LoRA 层）：20×[(207.94+178.85)−(130.94+76.00)] ≈ **+3.6ms/步（主犯）**
- 共享核干扰（另 20 层）：≈ +0.66ms/步
- 其余（act/quant 拆分、delta 链、额外 kernel）：≈ +2.4ms/步

**H1-H6 终判**：H1"真但不可热修"（无 cubin 可选，改 launcher 无效）；H4/H5 出局；H2 待 dram bytes（trace 无该字段）；H3 部分（核外 ~28% 或口径差）；H6 半否（GEMM 确实变慢）。

### 6.2 对路线图的终局推论

- **回收上限重算**：W2-b（unfused-act GEMM1 cubin）≈ 1.0ms TPOT + 结构税修复 ≈ 0.6-0.9ms + 干扰随 W2-b 自消 ⇒ **合计 1.6-1.9ms ≥ 需求 1.28ms ⇒ rnd ≤6ms 可达，但 W2-b 是必要条件**。
- 不做 W2-b 的 C 路上限 ≈ 0.8-1.1ms < 1.28ms ⇒ **不达标**。
- **W2-b 规格精确化**：量化族（1101=B300 走 NvFP4；L20=FP8 E4m3）blockscaled grouped GEMM1 + 物化 epilogue（写 bf16 GEMM1 out 供外部 act+delta）——即 CUTLASS/tcgen05 自写项，工期周级。
- C3' 地雷维持（rank-32 IMA，environ.py:1127-1129）。

[th-cpp 证据（2026-10-13，源码级根因链）：
1. base GEMM1 = `PermuteGemm1::Runner`，options `.fusedAct = !useDeepSeekFp8 && !forceUnfusedAct` = **true**（trtllm_fused_moe_runner.cu:402）⇒ SwiGLU 融合进 epilogue，输出 fp8（1B），permute+act+quant 一体。
2. lora GEMM1 = 裸 `Gemm2::Runner(MxE4m3, MxE2m1, Bfloat16, fusedAct=false, routeAct=false)`（trtllm_fused_moe_kernel_launcher.cu:3264-3297）⇒ 输出 [padded, 2·inter] **bf16（2B）**。
3. 根因源码注释钉死：kernel_launcher.cu:3256「**no NvFP4 unfused-act GEMM1 cubin exists**」——delta 必须 pre-SwiGLU 注入，fused-act cubin 无法容纳 ⇒ 只能退回通用 GEMM 家族（bf16）。
4. **GEMM1 epilogue 写带宽 4×**：base 写 [padded,576]×1B（act 后 fp8）；lora 写 [padded,1152]×2B（act 前 bf16）。
5. H2 证伪：两侧同用 `useShuffledMatrix=true, MajorK` 4-D block-shuffled 布局，无转换（:2913/:3268）。
6. H4 证伪：finalize 两侧各一次，不触权重（runner.cu:924-928 / kernel_launcher.cu:3531-3554）。
7. lora 侧 grid 更大（40,624 vs 10,504）⇒「并行度下降」签名证伪；变慢来自更重的 cubin 家族。
8. lora 侧绕过 autotuner（selectDefaultTileN/getDefaultValidConfigIndex :3647/:3271/:3491），但 config 极差仅 7.4% ⇒ 非主因。
9. **主刀落点**：`FP4BlockScaleLoraLauncher::run` step 5-7（kernel_launcher.cu:3261-3469）= 新 GEMM1 cubin（GEMM1 + gate_up_delta + act + quant 一体，输出 fp8），对应 docs/DSV41_MOL_FUSED_KERNEL_DESIGN.md M1/M3。
10. 验收双账本：① LoRA 净增（+171ms 低并发）应随融合消失；② 共享 gate_up 应回到 ~130.9µs/call（L2 干扰消除的可证伪预言）。
11. 一次 nsys 终判残差：lora 侧该核 dram bytes ≈2-4×（写家族证实）vs 相近（另有 stall）。]
