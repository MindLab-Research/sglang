# MoL 多 LoRA 乱码修复（2026-08-20/21 结案）

> 症状：DCP=4 + PD 分离 + EAGLE 下，懒加载第 N 个 LoRA 后所有请求乱码（token soup，如
> `!1.4.5.0,0,1,1,1,...`），且 sticky 传染（此前健康的 adapter 也乱码）。单 adapter 一直健康。
> 修复 commit：`f070d3d466`（`lora/backend/base_backend.py`，上游 #29468 移植）。

## 1. 根因（两层）

**触发条件**：`base(None) + N 个 adapter` 的 uid 数超过 `max_loras_per_batch`（4）时，新 adapter
请求触发 `LoRAMemPool` 的 LRU **驱逐**（`get_available_buffer_slot`）→ slot 被重分配给新 adapter。
实测最小复现：串行 `base→L0×2→L1→L0→L2→L3`——L3 首请求（第 4 个 adapter，触发驱逐）乱码，
随后 L0（此前 4 连健康）也乱码 = sticky 传染。

**缺陷本体**（`_compute_moe_lora_info`，base_backend.py）：
1. **CUDA graph 映射 buffer 尾部从不重置**：只在 `mapping_len > num_tokens`（DP-gathered 场景）
   才 `fill_(-1)`；常规 decode/verify（`mapping_len == num_tokens`）时 padding 尾部保留**上一个
   batch 的 stale adapter id**。捕获的 graph 读到 tier 宽度 → 回放时 padding 行被路由到错误
   adapter 的虚拟专家。stale 合法 slot id **不会**被 `>= nslots` 的 SANITIZE-MAP 拦截。
2. **rank-0/base segment 映射到 slot 0 而非 -1**：kernel 与 torch fallback 都无条件写 slot id。
   slot 0 通常被 base（rank 0）占用，但**驱逐后 adapter 可占用 slot 0** → base token 静默应用
   该 adapter 的 delta。

上游对照：issue **#29157**（GLM-5.1-FP8 + 多 LoRA + `--lora-use-virtual-experts` + CUDA graph
乱码，15/16 garbled，关 graph 0/16）+ PR **#29468**（closed 未合并——正是本修复的来源，其验证
环境与我们完全同构）。PR **#34337**（open）进一步说明上游 multi-adapter LoRA + EAGLE 投机解码
本就未完整支持。

## 2. 修复内容（f070d3d466）

1. `_compute_moe_lora_info`：CG buffer（`shape[0] > num_tokens` 时）**全量 `fill_(-1)`** 后再切
   active 前缀——回放时 padding 行永远读到 no-LoRA 哨兵。
2. `_compute_moe_lora_info_kernel`：`map_val = tl.where(lora_rank > 0, lora_id, -1)`。
3. torch fallback：`(wi >= 0) & (lora_ranks[wi.clamp(0)] > 0)` 才保留 slot id，否则 -1。

消费端（`kernels/ops/moe/virtual_experts.py`）本就正确处理 -1 哨兵（`mask = lora_id >= 0`），
无需改动。

## 3. 验证（2026-08-21，1101 prefill + 1100 decode，DCP=4 + PD + EAGLE，代码含修复）

| 序列 | 修复前 | 修复后 |
|---|---|---|
| base → L0×2 → L1 → L0 → L2 | 全健康 | 全健康 |
| **L3 首请求（驱逐触发）** | ❌ 乱码 `!1.4.5.0,0,...` | ✅ OK（114 字符连贯） |
| **L0 复查（sticky）** | ❌ 乱码 `!0.0;1,0, but0` | ✅ OK（134 字符） |
| L1/L2/L3 复查 | — | ✅ 全 OK |
| 16 并发混合（L0-L3 轮转，256 tok） | — | ✅ 16/16 零乱码 |
| decode 异常计数（Xid/CUDA/NaN/Traceback） | — | 0 |
| EAGLE accept len | 崩到 1.0 | 2.54-3.11（健康带） |

判据工具：1101 上 `/root/lora_test.sh`（单请求）、`/root/lora_verify_seq.sh`（复现序列）、
`/root/lora_stress.sh N MAXTOK`（并发混合压测）。

## 4. 排查过程中的方法论教训（重要）

1. **污染状态毁掉对照实验**：所有实验必须在**双端全新重启**后的干净状态上做。曾用未重启的
   prefill（带上一 session 的 sticky 污染）测试，得出"no-DCP 必崩 NaN"的错误结论——实际是
   prefill 污染 KV 传导到 decode 的 assert。引擎重启后 **router 熔断器需一并重启**（否则
   `No available decode workers` 空响应，易误判为请求失败）。
2. **"输出与 base 不同" ≠ "decode 端应用了 LoRA"**：PD 下 prefill 承担全部可观测差异（首
   token + KV）。验证 decode 是否真加载：`curl :30200/metrics | grep lora_pool_slots_used`
   （≥1）+ decode.log 的 "LoRA adapter ... loaded weights" 行数（TP8 应为 8/adapter）。
3. **乱码启发式**：`bangs >= 20`（'!' 计数）或输出超短非 EOS；token soup 样本形如
   `!0.0;1,0,` / `!1.4.5.0,0,1,1`。
4. 温度 0 下不同 batch 组合的输出本身会有风格漂移（batch 数值差异），与乱码（token soup）
   质性不同，勿混淆。

## 5. 遗留与边界

- **乱码主链路已修复**；上游 #34337 指出的 multi-adapter × EAGLE verify 的其余缺陷（seg_lens
  烧录等）在我们 csgmv 路径下未见触发，未移植。
- `--max-loaded-loras 4`（registry 容量）> `max_loras_per_batch`（slot 数）时驱逐是常态——
  若业务需要 4 adapter 全常驻 + base，考虑 `--max-loras-per-batch 5`（需同步评估显存）。
- radix 跨请求串数据的伴随症状（base 读到别人 prompt）在干净状态复现序列中未再出现；此前观察
  可能混合了污染 KV 的影响。`extra_key` 拼接（上游 #25351 的碰撞隐患）仍建议后续做长度前缀
  加固（RADIX-EXTRAKEY-DIAG 日志已在 match/insert 两侧就位，commit `2289765206`）。
- 生产恢复顺序：确认 1103 带修复重启完成后，router 改回双 decode（`--decode 10.0.58.36:30200`）
  → gateway(31001) → proxy(31000, `MOL_UPSTREAM_RUNTIME=sglang`) → 公网 18777 验证。
