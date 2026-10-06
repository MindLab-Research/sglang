
---

## DSV4.1 "零开销" 证伪 + GLM 方案移植终态（2026-10-06）

### 性能矩阵（1101，DeepSeek-V4.1-Flash，TP4/GPU4-7，流式 TPOT 精测 5×1024tok）

| 组 | 配置 | TPOT | tps | MoL 损耗 |
|---|---|---|---|---|
| A / A'（无 MoL 基线） | launch.sh 去 --enable-lora（±PER_GROUP_ROUTING） | 3.4ms | 291.2 | — |
| B（MoL 修复前） | launch.sh 原版 | 5.0ms | 200.8 | **+47%** |
| **FIX（移植后）** | launch.sh + 8 commits | **4.0ms** | **249.1** | **+18%** |
| C（lora rand20/39） | dpsk_rand20/rand39 | 4.8ms | 207.5 | — |

历史 "Zero measurable perf cost (149 vs 147 tok/s)" **证伪**：与 +47% 差两个数量级且无可复现记录；149 tps 与 A 组 291 tps 也不吻合——历史数字极可能是分解自比或另一口径，绝非 MoL vs 无 MoL。

### 剩余 18%（0.6ms）根因链（五轮修复完整证据）

- 五次 Python 分流（forward/双 variant/run_moe_core×2/stock 直调）base 恒 4.0ms；trace：分解 GEMM1/2 + attn LoRA shrink 仍录在 nolora graph
- DBG：prepare 正常（cap=lora 192/nolora 236），op_experts 与 run_moe_core **零命中**——decode graph capture 经 piecewise 编译快照（固化 wrap 前模块引用）绕过全部 Python
- FIX5 无效且 lora 变慢（4.8→5.1），已回滚
- 下一步：①piecewise 编译层注入分流 ②decode graph 切非 piecewise backend ③LoRA wrap 提前到编译前
- 判别：RAND20_DIFFERS True；ZERO20 FIDX=1（BYPASSED/STANDARD 浮点序差异，预期）

### GLM 对照（sglang-private pr17 @ dd8443183）：base 5.8ms（0.0%✅）/ lora 8.3-8.7ms。GLM decode graph 不绕过 Python 分流（runner 结构不同）——GLM 达 0% 而 DSV4.1 卡 +18% 的架构根源。
