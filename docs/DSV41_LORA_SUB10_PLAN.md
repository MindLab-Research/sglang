# DSV4.1-Flash MoE-LoRA「<10% 开销」系统性优化方案（v2，2026-10-13）

> 口径锁定：1101 PD 栈（P 30100 / D 30200 / smg），conc=64，后 20 层 MoE LoRA
> （`--lora-target-layers 20-39`），客户端聚合 p50 ms/tok，**新 API key**（旧 key 追加
> `/root/smg_api_keys_history.txt`，防外部老 key 流量污染）。
> 基线（10-12 A/B 实测）：base 5.087–5.216 ms/tok，rnd（挂 moe_rnd）7.632 ⇒ **+46%**。
> 目标：rnd ≤ base × 1.10 ⇒ **≤ ~5.6 ms**。

## 1. 差距解剖（kernel 级做差，trace 两侧同步数可比）

| 桶 | Δ（相对 base） | 归属 | 根因 |
| --- | --- | --- | --- |
| MoE GEMM | **+1.5ms（63%）** | LoRA 路 `t128x8x512_s3` GEMM 家族（gate_up 207.9µs/call、down 178.9）vs 基座 `t128x16x256_s6`（130.9/76.0） | delta 必须加在 SwiGLU 之前 ⇒ GEMM1 不得融合 act ⇒ 走 unfused-act 慢 cubin（源码原话：no NvFP4 unfused-act GEMM1 cubin exists） |
| 注意力 LoRA | **+0.5ms（21%）** | `ChunkedSgmvLoRABackend` shrink/expand，20 层 × ~5-6 launch | 小 grid（N=S·r=16~96），kernel 本体延迟受限；**调参已实测零收益（关闭），但融合/PDL 是新杠杆，未动过** |
| 杂项 | **+0.36ms（15%）** | MoE VE delta、gate_up pad 桥、activation_lora_input 捕获、pack_topk、finalize | 分解链的结构性附带 |
| base 侧污染 | （另一本账） | piecewise 编译快照绕过全部 Python 分流 ⇒ nolora graph 仍录分解 kernel | base 请求在挂 LoRA 的服务上 +18%（单机快基线 3.4ms 上 +0.6ms） |

**关键物理账**：conc=64、topk 6 ⇒ P=T·topk≈384 行摊到 64 experts/rank ≈ 6 行/expert —— GEMM 是
**权重读受限**（w13 全量 210MB/层/rank，HBM 峰值下限 ~26µs，实测 130~208µs = 利用率 15-25%）。
写带宽（bf16 gate_up 4×）在 P=384 规模下**不是主因**（0.98MB vs 0.25MB，差 0.1µs 级）——
主因是 **cubin tile 配置 + unfused epilogue 的实现劣势**。这改变了主攻方向的选择依据。

## 2. 候选方案与判决依据

### P1【已出局，2026-10-13 G1 实测】LoRA 层 MoE 换 deep_gemm masked 分组 GEMM
- **判决：失败**。修复全部 sf/布局/激活问题后（K=640 pad、per-128 激活量化、去伪 deinterleave、
  canonical swiglu_limit 分支），conc=64 base e2e：**contig TPOT_p50 7.288ms、masked 7.250ms，
  均 ≫ stock 5.087ms（+42%）**。
- **根因**：decode 形态 = 384 专家 × 每专家 1~8 行。deepgemm 的分组开销（每专家至少一个 tile pass）
  在 384 组下放大 ~2×；contig 的 128 行/专家对齐额外膨胀 ~20×。微基准（4 专家×128 行）掩盖了这一点
  ——**教训：分组 GEMM 的微基准必须按真实专家数/行数分布做**。
- 正确性成果保留：K=640 pad + sf pack 对齐 + canonical 激活分支的修复对本树仍然有效
  （fp8.py/deep_gemm.py 的改动在收口时评估保留/回退）。
- **产出物**：`/root/exp_g1_dgbase.sh`（deepgemm base-only 启动器）、val_gemm1/val_gemm2/val_chain、
  val_oai_kernel 对拍脚本（全部留在盒子上）。

### P2【备选】补完自写 Triton 融合核（fused_moe_lora_all.py）
- 结构上唯一能**消除 unfused-act 惩罚**的路（acc 留寄存器 + delta + gated-SwiGLU + MXFP8 一颗）。
- **现状差距**：M1 只过了 epilogue 数值契约；缺 expert 分组维度（B 单矩阵）、缺 permute/路由对接
  （triton moe_align 路由 vs trtllm permute 两套语义不能混，设计文档 §6.4 风险 2）、缺 gemm1/gemm2
  bias、delta 需 token-major→permuted 重排、M2 端到端对拍 / M3 集成 / M4 调优全未做。
- **风险**：Triton 带 in-kernel fp4 反量化的分组 GEMM vs trtllm-gen cubin，小 m 场景未必赢。
- **定位**：P1 门槛失败时的 B 计划；或 P1 成功后作为 M4 之后的远期替换。

### P3【必做，与 P1/P2 正交】piecewise 快照分流修复
- base 请求在挂 LoRA 服务上回到 stock 快路（+18% → ~0）。三条候选方向
  （①编译层注入分流 ②decode graph 切非 piecewise backend ③LoRA wrap 提前到快照前），
  file:line 机制勘查进行中（子代理）。
- **同时净化测量口径**：修复后 base 才是干净的 stock 基线，<10% 的分母才诚实。

### P4【必做，低风险】LoRA 专用 kernel 的 launch 数减半 + PDL
- 注意力 csgmv：shrink+expand 逐 projection 融合（仓库已有单 launch shrink+expand+add 模板：
  virtual_experts.py:1024 `_fused_lora_delta_kernel`，register-local 无跨 CTA 通信）；
  csgmv 两核加 PDL 链接（本树 attention/gemm 已有 PDL 先例，ferrite 扫描 PREFIX -90µs）。
- MoE VE delta：gate_up/down 各收敛到单 launch；P1 成功则并入 silu_mul_quant/finalize，进一步归零。
- 目标：0.5ms（attn）+ 0.36ms（杂项）→ 合计 **≤ 0.35ms**。

### P5【红线下排除】权重物化 W_eff
用户红线（2026-10-12 下令）：禁止烘权重，delta 必须动态逐请求现算。任何物化提案一律不进方案。

## 3. 量化预算（诚实版）

| 情形 | rnd 构成 | 结论 |
| --- | --- | --- |
| 仅 P3+P4（P1/P2 都不成） | 5.09 + 1.5(GEMM 罚) − 0.5(P4 收益) ≈ 6.1 ⇒ +20% | 远不达标 |
| P1 成 + P3 + P4 | 5.09 + attn 0.25 + MoE VE 0.15 + 杂 0.1 ≈ **5.6 ⇒ +10%** | **压线** |
| P1 成 + P3 + P4 + delta 全融进主 kernel（P1 的深化） | ≈ 5.09 + 0.4~0.45 ⇒ **+8~9%** | 达标（需深化项） |
| P1 败、P2 成 | 5.09 + attn 0.25 + (P2 融合核边际 ~0.1) ≈ 5.45 ⇒ +7% | 达标（工作量大、perf 险） |

⇒ **<10% 可达，但条件苛刻：P1（或 P2）+ P3 + P4 全部落地，且大概率需要「delta 融进主 kernel」的深化项。**
压线风险的兜底：若终态落在 +10~12%，向用户报告并给出选项（接受 / 减 LoRA 层数 / 降 rank / 继续 P2 深化）。

## 4. 执行顺序（每步独立可验收）

1. **Stage 0（并行）**：P3 机制勘查（进行中）→ 定方向；P4 的 csgmv 融合模板准备（virtual_experts:1024 已核实可搬）。
2. **Stage 1【决策门 G1】**：修 deepgemm capture 断言 → base-only deepgemm 启动 + conc=64 bench（新 key）。
   - 通过（base_dg < ~5.0）：进 Stage 2A。
   - 失败：P1 出局，转 Stage 2B（P2 补完 Triton 核，M2 对拍先行）。
3. **Stage 2A**：deepgemm runner 接 LoRA hook（quant info 补口 + delta 插缝 + pass-through 统一），
   数值闸门 zero-adapter==pass-through bitwise（同 kernel 同 delta=0）。
4. **Stage 3**：全链 A/B（新 key、cases_50 长 prompt、多轮取 p50）；deepgemm 化的 MoE VE delta 融合深化。
5. **Stage 4**：收口——盒子热修回灌提交 `release/dsv41-flash-mol`，AGENTS.md/知识文档回填，结论（含失败路径）全记录。

## 5. 测量协议（防污染）

- 每次重启前轮换 smg API key；bench 客户端与新服务端同 key；老 key 立即追加历史文件。
- 就绪判据：`curl -s -o /dev/null -m 5 -w "%{http_code}"` = 200（body 非空判据已证伪）。
- bench 用 cases_50.json 10000+ token 长 prompt；每轮先查 decode GPU 无外部流量（nvidia-smi 利用率 + smg 日志请求归属）。
- A/B 同栈同序（base→rnd→base→rnd 交叉两轮以上），取 client_agg p50，附 raw 数据。

## 6. 本方案的证据链

- 10-12 A/B：base 5.087–5.216 / rnd 7.632（client_agg p50，conc=64）。
- kernel 做差表：设计文档 `docs/DSV41_MOL_FUSED_KERNEL_DESIGN.md` §0（+269.4ms 分解）。
- csgmv 调参零收益（关闭）：AGENTS.md「CSGMV 块大小调参在 conc=64 端到端收益为零」实测结案。
- piecewise 根因五轮实证：`docs/MOL_ZERO_OVERHEAD_RETRACTION_REPORT.md` §8-3。
- csgmv 融合模板可行性：`virtual_experts.py:1024` `_fused_lora_delta_kernel`（单 launch、register-local）。
- deepgemm 插缝点：`deep_gemm.py:771`（GEMM1）/`:813`（silu_mul_quant）；hook 挂点 `lora_moe_runners.py:305/400/482`。
