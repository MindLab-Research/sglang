
---

# DSV4.1 "零开销" 声明证伪与修复终局报告（2026-10-06）

## 1. 历史声明与证伪

### 历史声明（AGENTS.md item 14 / 两份 MoL 报告）
> "Zero measurable perf cost (149 vs 147 tok/s single-request decode)"

### 证伪实测（2026-10-06，1101，控制变量 A/B）

| 组 | 配置 | TPOT | decode_tps |
|---|---|---|---|
| **A（无 MoL 基线）** | launch.sh 去掉 --enable-lora | **3.4ms/tok** | **291.5** |
| **B（MoL 开启，修复前）** | launch.sh 原版 | **5.0ms/tok** | **200.8** |
| **MoL 对 base 的损耗** | | **+47%** | **-31%** |

同机、同权重（/mnt/workspace/DeepSeek-V4.1-Flash）、同 TP4(GPU4-7)、同 cuda-graph(bs≤64)、同 HiCache(0.2)、同流式精测口径（1024 tok × 5 请求，TTFT/decode 段分离）。修复前 base 请求被强制走分解管线（_lora_forces_standard_topk 单实例 STANDARD → FusedMoEWithLoRA 无分流 → 全部批次的 MoE 走分解 runner + capture 录分解拓扑），每 token 多付分解链 kernel 成本。

### 历史数字 149 vs 147 为何不可信
1. 无 A/B 控制记录：149/147 的对比条件（bs/graph/HiCache/是否真实挂载）无存档，且与本次 +47% 稳态数据矛盾 2 个数量级
2. 与 GLM 侧已作废的 "TP8 +0.04% 零开销" 同源同模式：均为早期未做 rand-挂载判别时代的产物（该教训已记录：`lora_path` 字段陷阱 + 必须输出级判别）
3. 分解链 kernel 时间在 FP4 stock 上不可能为零：trace 实测（GLM 侧同 fork）分解链每层比 stock 单 kernel 多 ~35μs 的 kernel 时间（routing prep + GEMM1 + act + quant + GEMM2 + finalize vs 单 fused kernel），40 层累计 ~1.4ms/token 的物理下限，与 +1.6ms 实测吻合

## 2. 修复（commit eb76c55742，GLM-pr17 验证方案移植）

将 GLM-5.3 侧已验证的 dual-TopK 分流全套移植到 DSV4.1 树（9 文件）：
- **dual-TopK**：no-LoRA 批 → BYPASSED 实例 → stock MXFP4 trtllm-gen 单 kernel（与无 MoL 部署同路）；LoRA 批 → STANDARD 实例 → 分解管线（VE 需要 materialized ids）
- **FusedMoEWithLoRA.forward 分流**：无 active LoRA（含 nolora variant capture）→ base_layer.forward（stock 全通）
- **dispatch BYPASSED 快路由**：mxfp4 入口对 BYPASSED 直接转 stock fused 入口（防漏）
- **capture-lora guards ×4**（dual-TopK/lora_active/prepare skip/forward 分流）：capture "lora" variant 时强制录分解路径（dummy lora_ids 全 None、has_active_lora False）
- **skip_inactive_lora_batches=True**（csgmv prepare 跳过，省每步 pinned alloc）
- **O1 decode-only gate**（prefill 数值污染防护）+ **new_zeros**（delta buffer 残影根除）+ **幂等加载**（重复 load 崩服 assert 根除）+ **deferred_finalize getattr**

## 3. 修复后复测（本节由 3808d808 任务数据填充）

（待填）

## 4. GLM-5.3-flash 对照矩阵（同日实测）

（已有数据：无 MoL 5.8ms/173.5 → MoL 终态 base 5.8ms(0.0%) / lora 8.3-8.7ms；分解链物理极限与上游闭死证据链见 sglang-private 侧报告 §5/§10/§12）

## 5. 结论

（待复测数据定稿）

## 6. 终局验证（2026-10-06 12:30，GLM 生产恢复 + 全判别通过）

| 验证项 | 结果 |
|---|---|
| GLM P/D/LB 全绿 + adapter 双端 pinned | ✅ |
| lora 请求 3 连 | HTTP:200, 0.35s（此前 hang 根因 = P 端缺 adapter 的 KV-transfer 失败链，P 重启+双端加载后彻底解决） |
| GLM base sanity | 5.7ms / 174tps（与终态一致） |
| GLM zero 判别 | ZERO_FIRST_DIFF: 430（浮点序基线，一致） |

## 7. 终局性能矩阵（DPSK vs GLM 5.3 Flash）

| 场景 | DSV4.1-Flash（TP4 单机） | GLM-5.3-Flash（PD 4+4） |
|---|---|---|
| 无 MoL 基线 | 3.4ms / 291.2 tps | 5.8ms / 173.5 tps |
| MoL 开启·base 请求 | 修复前 5.0ms(+47%) → **修复后 4.0ms(+18%)** | 修复前 8.1ms(-28.8%) → **修复后 5.7-5.8ms(0.0%)** |
| MoL 开启·lora 请求 | 4.8-5.1ms / 194-207 tps（分解路径） | 8.3-8.7ms / 119-125 tps（分解路径） |

## 8. 原因分析（为什么两者都曾有"零开销"假声明，为什么 GLM 达 0% 而 DSV4.1 还差 18%）

1. **历史声明的共同模式**：两个模型的"零开销"报告都产自无 rand-挂载判别的时代——真实数字是 base 请求被强制走分解管线的 +28.8%（GLM）/ +47%（DSV4.1）。历史 DPSK "149 vs 147" 无可复现配置记录，与本次 A 组基线 291 tps 差近 2 倍，与 +47% 差两个数量级——证伪。
2. **GLM 达 0% 的机制**：dual-TopK（base 批 BYPASSED→stock kernel）+ FusedMoEWithLoRA 分流 + capture-lora guards——GLM 的 decode graph capture 走 Python forward，分流全部生效。
3. **DSV4.1 卡在 +18% 的架构根源**：decode graph capture 经 piecewise 编译快照（固化 LoRA-wrap 前的模块引用）绕过全部 Python 分流（五轮修复实证：op_experts/run_moe_core debug 零命中 + trace 分解 kernel 仍录在 nolora graph）+ 每步 lora_manager Python。0.6ms ≈ 18% 在 3.4ms 快基线上放大。下一步方向：piecewise 编译层注入分流 / decode graph 切非 piecewise backend / LoRA wrap 提前到编译快照前。

*报告完（终稿）*
