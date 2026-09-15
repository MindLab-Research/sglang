# MoL LoRA stage 调参（`SGLANG_OPT_MOL_LORA_STAGE_CFG`）

**一句话**：MoL（虚拟专家）的 LoRA shrink/expand 复用 MoE 的 stage config 路径，
fallback 配置 `BLOCK_SIZE_M=64 / BLOCK_SIZE_N=64` 让 launch grid 按 **N-tile 数**爆到
153,984 CTA → 每层 down-delta expand 单次 **160.6 µs（有效 ~85 GB/s ≈ HBM 峰值 1%）**。
本 flag 只加宽 `BLOCK_SIZE_N`（→ N-tiles 4× 更少），不改 `BLOCK_SIZE_K`，**逐位等价**。

---

## 1. 触发（怎么发现的）

decode step 的 launch 归因（B300 decode 节点，bs=23，60 步 profiler 窗口，kernel-only，TP0）：

| annotation | kernel/迭代 | busy ms | MoE launch/迭代 | MoE ms |
|---|---|---|---|---|
| `step[TARGET_VERIFY bs=23]` | 8710 (94.8%) | 79.10 (92.9%) | 372.7 | 30.53 |
| `draft`(5 步)+`EAGLE_VERIFY`+`draft_extend` | 437 | 5.53 | 11.2 | 1.07 |
| 其余 | 36 | 0.53 | 1.0 | 0.15 |

层内 `fused_moe_kernel` **5 次/层**（`_correct_attn_cp_out` 78/迭代 = 78 层，
MoE 段内计数 {5: 4500} = 75 MoE 层 × 60 迭代），执行顺序与 grid：

```
align+count_sort+quant8 → ① grid=1292   175.6µs  ← base gate_up            (fused_moe.py:524)
align+count_sort + LORA_shrink_splitk(43.1) → ② grid=6416 7.5µs ← LoRA gate_up[gate] 半
                                              ③ grid=6416 6.8µs ← LoRA gate_up[up]   半
act_and_mul+quant8 → ④ grid=15504 86.5µs  ← base down (w2)              (fused_moe.py:715)
LORA_shrink_splitk(5.2) → ⑤ grid=153984 160.6µs ← **LoRA down-delta expand**
moe_sum_reduce → AR_bf16
```

每层 MoL LoRA = ②③⑤ + 两个 shrink = **223 µs/层 ≈ 16.7 ms/迭代 ≈ 19.6% 步时**，
其中 ⑤ 一项就 **12.0 ms ≈ 14.1%**。⑤ 的真实工作量是 `M×6144` 输出 + `K=b_rank=32`
的窄 GEMM（有效流量 ~14 MB，应 ≤2 µs），却用 **153,984 个 128-thread CTA ≈ 1040 波**
（148 SM），即 **tile/CTA 开销受限**，不是带宽受限。

## 2. 根因

`virtual_experts.py::_merged_experts_fused_moe_lora_add_impl._get_stage_config()`：

```python
cfg = get_config_func(token_lora_mapping.shape[0])      # 这些形状无 autotune 条目 → ValueError
except ValueError:
    cfg = {"BLOCK_SIZE_M": 64,
           "BLOCK_SIZE_N": min(64, max(16, N_dim)),     # ← 64
           "BLOCK_SIZE_K": ..., "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
```

`_get_routing(...)` 把这个 `BLOCK_SIZE_M` 交给 `moe_align_block_size`，于是
**每个被触及的虚拟专家桶都被 pad 成整 64 行**（虚拟专家数 = `max_loras × n_routed`
= 1024 桶）：

- 真实 (token, 虚拟专家) pair ≈ 1.1k → **padded rows ≈ 1604 blocks × 64 ≈ 1e5（~93× 膨胀）**
- launch 数（host 端，`fused_moe_triton_kernels.py::invoke_fused_moe_kernel`）：
  ```
  grid = cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M) * cdiv(B.shape[1], BLOCK_SIZE_N)
  ```
  其中 `sorted_token_ids` 是最坏情况**分配**缓冲（`_get_routing` 刻意不 trim，见该函数注释）
  ⇒ `sorted_len = numel + virtual_E×(BLOCK_SIZE_M−1)`、`grid_m ≈ virtual_E + numel/BLOCK_SIZE_M`。

**三处 grid 独立对账**（同一窗口实测 vs 公式）：

| launch | N（真实 GEMM 宽） | cdiv(N,64) | blocks×N-tiles | 实测 grid |
|---|---|---|---|---|
| ⑤ down-delta expand | 6144 | 96 | 1604×96 = 153,984 | **153,984** ✓ |
| ②③ gate_up expands | 256 | 4 | 1604×4 = 6,416 | **6,416** ✓ |
| `_moe_lora_shrink_splitk` (gate_up) | rank 2r=64 | 1 | 1604×1 = 1,604 | **1,604** ✓ |

⇒ N 比 6144:256 = **24** 恰好等于 grid 比 153984:6416 = **24**：**代价 ∝ N-tile 数**。

## 3. 修复

`SGLANG_OPT_MOL_LORA_STAGE_CFG`（`environ.py`，默认 **False**）打开后，
`_get_stage_config(weight, stage_top_k, n_dim)` 只改一项：

```python
tuned["BLOCK_SIZE_N"] = min(256, max(16, n_dim))
```

`n_dim` 是**该 stage 真实 GEMM 的 N**（shrink = rank；expand = 输出宽），由调用点显式传入
（shrink 权重视图是 `[E, K, N]`、expand 是 `[E, N, K]`，不能靠 shape 猜）。

**为什么不动 `BLOCK_SIZE_M`**：align 按 `BLOCK_SIZE_M` 对齐，
`padded_rows = blocks × BLOCK_SIZE_M`，因此
`grid_m = cdiv(padded_rows, BLOCK_SIZE_M) ≡ blocks` —— **grid_m 与 `BLOCK_SIZE_M` 无关**，
调小只会把桶切成更多 block（`blocks` 变大 → 更慢）。

**为什么逐位等价**：只改 N 方向分块；`BLOCK_SIZE_K`（≥ rank，expand 只有 1 次 K 迭代）
与 K-loop 不变 ⇒ 每个输出元素仍由**同一 tile 内**完成全部 K 归约，无 split-K、无跨 tile 累加。

**预期**（若确为 CTA 受限）：⑤ 153,984 → **38,496 CTA** ⇒ 160.6 → ~40 µs/层
⇒ 省 ~9-10 ms/迭代 ≈ **11% 步时**；②③ 6416 → 1604 另省 ~10 µs/层。

> ⚠️ 注意：shrink（rank=64/32 ≤ 64）在本 flag 下是 **no-op**（N-tile 本来就是 1）。
> shrink 的 48.3 µs/层（4.3%）是另一件事（小 tile + K=hidden 长 K-loop），
> 走 P1-A（第二 stream 重叠）或换 kernel，不在本 flag 范围内。

## 4. 验收

1. **打开 flag 的 A/B**（同 bs、同 60 步 profiler 窗口、kernel-only、TP0）：
   比 `fused_moe_kernel` 的 ms/迭代与 grid（`args.grid`）。**禁止跨 bs 比均值**。
2. 日志判据：`grep MOL-LORA-STAGE-CFG` 应出现一条 one-shot
   `[MOL-LORA-STAGE-CFG] enabled: n_dim=... BLOCK_SIZE_N 64 -> 256`（证明 flag 到了 kernel 层）。
3. **正确性**：cases_50 全过 + 多 adapter（L0/L2/URL）输出逐位对比（off vs on 期望 bit-identical）；
   改 N-tile 属数值路径相邻改动，必须过正确的回归，不能只看速度。
4. 只在测试集群做；生产 1021/1022 不做影响训练 LoRA 的操作。

## 5. 相关

- 归因证据链：`docs/agent/job-step-profile.md`（step 分解 + 8→32 并发曲线）
- 同一 step 的其它项：base MoE（19.7 ms/迭代，23%）走 P0-A（smem 18 KB vs draft 72 KB 的 K-pipeline 深度）；
  LoRA shrink/注意力 LoRA 走 P1-A（第二 stream）
