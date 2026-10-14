# DSV4.1-Flash MoL 归因/验证包（1101 一空就能打）

口径：client 侧 TPOT p50，conc=64 / n=256 / cases_50.json 长 prompt / out=4096 / ignore_eos / seed=777。
脚本：`tools/mol-v41-prep/`（全部本地写好、未在 1101 执行过任何一条）。

## 1. 最小实验矩阵

| # | cell | 变体 | **证明什么** | 预期 p50 (ms) | 若偏离 ⇒ 结论 |
|---|---|---|---|---|---|
| C0' | stock_base | 无 `--enable-lora` 启动 | 真分母（物理 base） | 3.4–4.0(单流)/5.41(PD·conc64) | — |
| C0 | base_same_stack | 同 LoRA 栈、请求不带 adapter | 同轮参照；含 piecewise 快照污染 | 5.20–5.45 | >5.6 ⇒ 快照污染变重，百分比全部要重算 |
| C1 | rnd | `dsv41_moe_rnd` | 复现 1.87ms 开销 | **7.281** | ∉[6.95,7.60] ⇒ 先查口径（key/bs-decode≥64/外部流量），**禁止做归因** |
| **C3'** | rnd+**FUSED_DELTA** | `SGLANG_LORA_FUSED_DELTA=1`（`environ.py:1129` 默认 False；消费点 `kernels/ops/moe/virtual_experts.py:876` 读 `:1811-1815`；rt-kernel 提案，已核消费点存在） | 融合 delta 路径收益（**前置 G1 位级闸门，未过闸不读数**，`run_matrix.sh` 缺 `G1.pass` 直接 REFUSED） | 待测（预期与 C2 同向，构成交叠） | 与 C2 叠加：两者都减 delta 链 ⇒ **不得同轮混读**，各自独立启动 |
| C2 | rnd+**SKIP_DELTA** | `SGLANG_MOE_LORA_SKIP_DELTA=1`（lora_dispatch.py:834-838，降级为 pass-through zero-delta，数值无损） | **隔离 MoE VE delta 链**（gate_up VE shrink/expand、down VE、2×routing、F.pad 桥、alloc/memset） | 6.90–7.20 | ≈base ⇒ 主因全在 VE/copy 链（P-A 上限更高）；≈C1（无变化）⇒ 开关未生效，先验命中再下结论 |
| **C5'** | C2+PDL | `SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS=8192` 显式 + 缺口补齐 | PDL 缺口边际（ferrite-v2: quant 0/10、gemm 3/17、moe 4/21） | 6.85–7.10 | 无收益 ⇒ PDL 不是杠杆（忙率已 97.5%） |
| C4 | rnd+fused-GEMM(M3) | `SGLANG_OPT_MOL_FUSED_GEMM=1` | P2/M3 融合核 e2e（不看微基准） | 6.30–7.00（首版常无收益） | >7.0 ⇒ 融合核未赢 trtllm-gen，P2 线出局 |
| C5 | C4+PDL | 叠加 | 叠加上限 | 6.20–6.90 | — |
| C6 | **分层斜率** | `--lora-target-layers 0-39` vs `20-39`，同轮 | Δ/(20 层)=per-layer 摊销；区分固定开销 vs 每层开销 | Δ≈1.85–1.90 | 斜率远小于 93µs/层 ⇒ 开销实为固定项（图/launch 面），改打图捕获 |

执行协议：轮换 smg key（老 key 入 `smg_api_keys_history.txt`）→ 停栈重启 → 200 就绪 → **P(30100)/D(30200) 双侧 adapter 均已加载** → 无外部流量 → bench；顺序 C0→C1→C0→C1 交叉 ≥2 轮，取 client_agg p50。
SKIP_DELTA 的作用域只是 MoE trtllm dispatch 的 VE delta（**不覆盖 attention csgmv**）——所以它单独不能解释 1.87ms，必须与 C6 斜率格、nsys 分桶一起读。

## 1.1 C2 先验注册（跑之前钉死，防「读数为准但没人被判对错」）

两个互相判死对方的预测（对方 rt-runtime 已按代码逐行清点 `lora_dispatch.py:833-1016`）：

| 提出方 | C2 预测 p50 | 依据 |
|---|---|---|
| rt-runtime | **6.44–6.60 ms（回收量上界，右收）** | SKIP_DELTA 只摘 delta 的**算**、不摘 delta 的**消费**：gate_up VE(`:871-888`) + down VE(`:994-1016`，含 `:997` contiguous 全量 copy) ≈42µs/层×20 是**回收上界**；**保留** `new_zeros`(`:868`)、`zero_()`(`:892`)、pad 桥(`:893-899`)、`activation_lora_input`(`:901`)、分解 C++ 核(`:951`)、pack_topk(`:928`)、scaling(`:808`)。<br>✅ **rt-measure 逐行复核成立**：`trtllm_fp4_block_scale_routed_moe_lora(` 的 `gate_up_lora_delta=gate_up_delta` / `activation_lora_input=...`（`:944-947`）在 `if has_active_lora` **之外**，全零 delta 仍被 act 核 global load+add ⇒ C2 只能 **≥6.44** |
| rt-measure（我） | **6.90–7.20 ms** | 依 `DSV41_MOL_FUSED_KERNEL_DESIGN.md` §0 做差：GEMM 净增占 63%，若 VE delta 只是 15% 桶的一部分，则摘掉它回收有限 |

**判别映射（二选一，不可能同时成立）**
- 实测 **∈[6.44,6.60]** ⇒ rt-runtime 成立：剩余 ~1.0ms 全在**分解 C++ 核 + 两条 copy 桥 + alloc** ⇒ P-A 必须把 `:893-1016` 连 C++ 核整段换掉才够，P-B 只是铺路石。
- 实测 **≈6.9–7.2** ⇒ 我成立：VE 链比估算轻，大头在 attention csgmv / 快照污染 ⇒ P-A 上限上修到 ~5.6，P-B/P-C 记账重写。
- 实测 **≈C1（无变化）** ⇒ 开关未命中，**两条预测都不算数**，先查命中（R9）。

**P-A 判据修正（采 rt-runtime 意见）**：不看 `gemv_k1_gateup` 单挑 207.9µs，看**整链 Σ**：
`lora_chain = K1 + K2 + finalize` vs `overlay_chain = gemm_base_fused + quant + activation + finalize`（+ 各自的 dispatch gap/share）。
`nsys_extract.py` 已内置 `CHAINS` 聚合与 `--compare base.json rnd.json` 做差表，直接出 per-layer Δ。

## 2. kernel 级标准提取法（nsys）

抓取（`nsys_capture.sh`，attach 不重启）：`SGLANG_ENABLE_NVTX_SCHEDULER/OPERATIONS=1` → `nsys profile --attach-pid <decode> -t cuda,nvtx,osrt -o kern_<tag> --duration 20`，同刻跑 bench_c64 保证窗口内稳态 conc=64 → `nsys export --type sqlite` + `nsys stats --report cuda_gpu_trace/nvtx_pushpop_trace --format csv`。
分桶（`nsys_extract.py`）：按 NVTX 步边界切 decode step，把 kernel 归入
`gemm_base_fused(t128x16x256_s6) / gemm_lora_unfused(t128x8x512_s3) / delta_shrink / delta_expand / gemv_k1_gateup / gemv_k2_down / quant / align_permute / activation / finalize / attn_lora_sgmv / attn / norm_rope / other`，
输出 **total_us / per_layer_us(÷20) / per_call_us / calls / %**，并统计同流相邻 kernel 的间隙（`--gap-threshold-us`，>2µs 计一次）⇒ **dispatch gap 与实际 launch 数**分开给出（发射税 vs 算力税）。
判据：分桶 per-layer 相对抖动 <5%，否则延长窗口或多次采样；基线锚点 = 设计文档 §0（gate_up 207.9µs/call vs 130.9；down 178.9 vs 76.0）。

## 3. 正确性门禁（perf 声明前必须全绿，顺序执行）

| 门 | 内容 | 判据 | 工具 |
|---|---|---|---|
| G0 | 冒烟：就绪 200、双入口、P/D 双侧 adapter、三 adapter 有输出 | 全 PASS | `smoke_checklist.sh` |
| G1 | **zero == pass-through 位级**（zero adapter vs SKIP_DELTA=1，greedy/seed777，n=256） | token 流 md5 全等、无 NaN/Inf | `gate_zero_bitexact.sh` |
| G2 | M2 数值：act median_rel=0（fp8 位级）、**out median_rel ≤2e-2** | PASS | `gate_m2_numeric.py` |
| G3 | M3 后 e2e：同 seed logprob median_rel ≤2e-2（token 流分歧数须列出首个分歧位置） | PASS | 同上（jsonl 模式） |
| G4 | 无回归：base 请求 ≤ C0 +1% | PASS | `bench_c64.py --adapter none` |

## 4. 风险登记（哪条读数证伪哪条路径）

| # | 风险 | 证伪判据 | 证伪后动作 |
|---|---|---|---|
| R1 | 「VE/delta 链是主因」 | C2 ≈ C0（而非 ≈C1） | 主攻 P-B/P-C（去 pad/contiguous 桥、routing 去重），P-A 上限上修 |
| R2 | 「GEMM 罚不可动」 | C2 ≈ C1（delta 链拿掉后仍 ≫base） | 主攻 GEMM cubin 家族 / M3 融合核 |
| R3 | **GEMV K1 权重流量 1.33× 但 L2 命中不足 ⇒ K1 慢于 overlay** | nsys: `gemv_k1_gateup.per_call_us` > 现 gate_up GEMM 207.9µs，或 dram bytes/K1 命中率劣于 | gate_up 回退 trtllm-gen overlay，只留 K2 |
| R4 | PDL × spec decode/DSPARK 不相容 | 需 dspark on/off × PDL on/off 四格；仅 dspark off 才有收益 ⇒ 不可上线 | PDL 降级为「非生产可选」 |
| R5 | 快照污染：base 侧 +18%（piecewise 编译快照绕过 Python 分流） | C0 显著高于 C0'（同栈） | 「相对 base 的百分比」一律改用 C0' 重算 |
| R6【已预清·降级为换栈复核项】 | 口径漂移：`--cuda-graph-max-bs-decode < 64` ⇒ conc=64 落 eager，launch 税裸奔。**仓内实证**：`docs/PD_PERF_REPORT_20261003.md` §3.2 —— max_bs 64 遇 96 并发时 91 流 fallback eager，gen throughput 415 vs 3914 tok/s（**8×**） | 今日 18:46 部署实测 cmdline `--cuda-graph-max-bs-decode 128` ≥64 ⇒ **预清**；换栈/换启动器后复核一次 | 修正后重测，作废该轮全部读数（**整轮有效性的前置条件**） |
| R7 | 微基准外推陷阱（P1 deepgemm 教训：384 组 ×1~8 行/组 ≠ 4 专家×128 行） | 微基准排名与 e2e 排名不一致 | 只认 C4/C6 的 e2e 读数，微基准仅作定位 |
| **R8** | 「rnd 不触发 batch 拆分」只对了一半（`max_loras_per_batch=1`/shared-outer 成立）；若有请求正处 adapter 加载中/未就绪，双图 variant 选择会翻（`decode_cuda_graph_runner.py:1104-1125`） | 跑 C0/C1 时 nsys 里出现**两次 MoE 段**（lora graph + nolora graph 混批） | 该轮读数作废，改为全 adapter 就绪后重测 |
| **R9** | SKIP_DELTA 未生效 / 请求侧没进 LoRA 路 / 走了 stock 分流（三者 p50 同形） | **判别器 `check_ablation_quadrant.py`（不看 p50 单点）**：kernel 计数（delta 类 / 分解链 / stock 融合核）× p50 落点 × `/proc/<pid>/environ`：格1 rt-runtime 先验 / 格2 rt-measure 先验 / 格3 env 未生效 / **格4c stock 分流**（`lora_dispatch.py:780-787` BYPASSED→`fused_experts_none_to_flashinfer_trtllm`；`deepseek_v2.py:112 _lora_forces_standard_topk()`）/ **(d) 桶没命中** / **格5 delta 链≈免费**（两先验双双落空）；另加 **R8 混批前置检测**（窗口内同时出现 stock 与分解链 ⇒ p50 是两图混合值，直接判不可用） | 格4c 需三样齐：请求侧 adapter 命中（smg 日志）+ `/v1/models` 有 `dsv41_moe_rnd` + 拓扑二分 |

## 5. 脚本清单（全部已本地语法自检，未跑 GPU）

| 文件 | 用途 | 说明 |
|---|---|---|
| `run_matrix.sh` | 矩阵驱动 | 默认 DRY-RUN 只打印；`cell C2 --execute` 才重启+测 |
| `bench_c64.py` | client 侧 TPOT p50（conc=64/n=256/长 prompt/多轮） | 与 `bench_ttft_tpot.py` 互校，偏差 >2% 以既有工具为准 |
| `nsys_capture.sh` | attach 抓 decode 窗口 + 导出 sqlite/CSV | NVTX 步边界；备选 torch profiler 路径已注明 |
| `nsys_extract.py` | 分桶 µs 表 + gap 统计 + **整链 Σ** + **`--compare` 做差表** | sqlite 首选，CSV 兜底 |
| `gate_zero_bitexact.sh` | G1 位级一致（dump / compare） | compare 纯本地可跑 |
| `gate_m2_numeric.py` | G2/G3 数值门 | `.pt` 张量或 jsonl logprob |
| `smoke_checklist.sh` | G0 冒烟 | 只读 curl |
| `check_ablation_quadrant.py` | R9 判别（格1/2/3/4c/5/(d) + R8 混批前置） | 七档合成判决已本地验证 |
| `check_p2_budget.py` | **S1 判据守卫**：整核替换口径 + 同形分母强制 | 四档合成案例已验证（可行/判死/缺分母 REFUSED/形状不符 REFUSED） |


## 6. 1101 解冻后 2 小时黄金窗口序列（按信息增益/GPU 分钟重排）

排序原则：**能一举判死一条路径的便宜实验先打**（杀路径 = 最高信息价值）；能一次启动产出两个读数的合并打。

| 顺位 | 步骤 | GPU 分钟 | 决策价值 | 判据 / 分支 |
|---|---|---|---|---|
| **S0** | G0 冒烟 + 启动参数核对（`--cuda-graph-max-bs-decode ≥64`［R6 已预清=128，降级为换栈复核项］、`exp_l20.sh` dump、key 轮换、P/D 双侧 adapter、无外部流量）+ **`/proc/<pid>/environ` 三项并列导出**（`SGLANG_MOE_LORA_SKIP_DELTA`/`SGLANG_LORA_FUSED_DELTA`/`SGLANG_OPT_MOL_FUSED_GEMM`，落 `$OUTDIR/<cell>.env.txt`） | 10 | 前置：无效则整轮读数作废（R6） | 全绿才继续；任一红 ⇒ 修完重来 |
| **S1** | **P2 GEMV 单层 microbench**（K1/K2 单独 µs；**必须真实 decode 形状**：A=384 行、64 experts/rank、1–8 行/专家） | 10–15 | **最高**：直接裁决「ALU 悬崖」A/B，可判死 P-A/P2 | **判据（rt-runtime 补正后）：`M_gemv ≤ M_base_overlay + 29µs`（⇔ ≤ 现状每层 − 64µs）**——P-A 是**整核替换**（K1+K2 连 base GEMM 一起换掉），若只拿 K1+K2+argsort+finalize 对「93/29」= 拿替换后整核比替换前增量，**苹果比橘子**。⇒ 必须**同形状同测 overlay 链基线**（复用 §2 的 `CHAINS`：`gemm_base_fused+quant+activation+finalize`）。守卫脚本 `check_p2_budget.py`：缺 overlay 基线 / 形状不符 ⇒ **REFUSED，不下判死结论** |
| **S2** | **C1 启动 + nsys attach 20s + bench n=256 ×2**（一次启动两产出） | 30–40 | 高：给出 p50 基线 + 真实 per-kernel 分桶（桥/launch/align/quant 实数） | p50 ∈[6.95,7.60] 才作 C2 基准；`--compare` 出 Δ/per-layer |
| **S2.5** | **C3'（FUSED_DELTA）重启 + bench + nsys**（前置 G1；插在 C2 之前） | 25–30 | 中高：融合 delta 单变量 | 与 C2 不得同轮混读；G1 未过 ⇒ REFUSED |
| **S3** | **C2（SKIP_DELTA）重启 + bench + nsys** | 25–30 | 高：delta 链纯量 + R9 四格判决 | `check_ablation_quadrant.py`：格1 ⇒ P-A 须整段替换；格2 ⇒ P-A 上限上修 ~5.6；格3 ⇒ 修 env 重跑；格4 ⇒ 补请求侧证据 |
| **S4** | C6 层斜率（0-39 vs 20-39） | 20–25 | 中：区分固定开销 vs 每层开销 | 仅在 S2/S3 有歧义时打 |
| S5 | P-B 落地后 rnd 重测 | — | — | **依赖代码改动，不在本窗口**：等 P-B 合入再排 |

**串行/并行纪律**：PD 栈占 GPU4-7；S1 的 microbench 理论上可放 GPU0-3，但**必须先证无跨卡功耗/时钟串扰**（同一 p50 窗口内混跑会污染 6.000ms 绝对判据）。稳妥做法：S1 串行前置（它最便宜且能救下后面 100 分钟）；激进做法：并行但 S2/S3 的 p50 只采串扰已证无差异的轮次。
**回退分支**：S2 读数出界 ⇒ 立即停，回 S0 查口径（R6/外部流量/key）；S3 判格3 ⇒ S3 重跑优先于 S4。
