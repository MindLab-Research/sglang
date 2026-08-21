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

## 6. 第二 bug：HiCache × LoRA 命名空间污染（2026-08-20 晚 结案）

**症状**：启用三级 HiCache（page_first + write_back）后，LoRA 请求出现**随机性**乱码
（首 token id=0、accept 健康但输出 token soup、上下文完全丢失——正文输出无关内容如
GitHub README）；base 始终干净；权重 checksum 完好；乱码后状态持续（"污染扩散"）。

**触发规律**（实测）：与 KV 分配位置/请求历史相关——同一请求形态在不同 session
随机好坏；chat 加载触发型请求、流式、radix 命中重复均可触发；/generate 短 prompt
（单 segment）几乎不触发。

**根因**：HiCache host 层（write_back/page_first 路径）的前缀匹配/写回**不正确处理
LoRA extra_key 命名空间** → 跨 adapter KV 复用污染（L1 请求读到 L0 的 KV → 前向
上下文错乱 → 乱码）。与上游 #25351（LoRA prefix cache namespace collision）同族，
但发生在 host 层。

**修复**：**生产禁用 HiCache**（从 1101 prefill 启动参数移除
--enable-hierarchical-cache 系列）。禁用后全形态验证套件全绿：加载触发型 chat ×4、
流式、radix 命中重复、新 prompt、16 并发混合、生产形态多轮流式、公网 18777 流式。

**后续**（未做）：修复 host 层 extra_key 隔离（radix host 匹配路径需消费
RadixKey.extra_key 并在写回时保留命名空间），之后可重新启用三级缓存。
判据：重复"加载触发型 chat + 流式 + 多 adapter 混合"套件（本文件 §6 的复现形态）。

**排查中的方法论教训**：
1. bangs 计数启发式不可靠（soup 可能只含 1 个 '!'）——必须人眼看文本
2. 污染状态毁掉一切后续测试——每个假设验证前必须双端全新重启
3. "干净"的验证要覆盖真实流量形态（chat 模板/流式/thinking/多轮）——只测 /generate
   短 prompt 会漏掉整类 bug
4. 随机性好坏 = 怀疑 KV 分配位置/host 层；确定性好坏 = 怀疑代码路径

## 7. 第三 bug（真根因）：prefill-CP pad 行被施加 LoRA delta → pad KV NaN 毒化真实行（2026-08-21 结案，commit `12cda1cd60`）

> §1-2（f070d3d466）只修了驱逐场景；§6 禁 HiCache 后**随机乱码仍在**（filler prompt 复现率
> 75-100%，topic 5-50% 波动）。本节是 CP×LoRA 残留乱码的完整破案链。

**症状矩阵**（定位的路标）：
- base 永远干净（0/32 × 多轮）；LoRA + **CP off** 干净（32/32）；LoRA + **CP on** 乱
  （13-31/32）→ bug 在 CP×LoRA 组合，不在任何一方单独
- 首 token 必乱（`output_ids[0]==0` = NaN logits greedy argmax），logprob=None
- 随机/传染表象 = allocator 复用决定 pad 行激活是否被 delta 打成 NaN + 哪些 rank 带 pad

**证据链**（全部可复现）：
1. 逐层 dump（--debug-tensor-dump）：乱码 pass 中 layer-7 MoE gate **NaN 仅在 gather 后
   32 行的 pad 行 [23,27,31]**（29 真实 token 对齐到 32）；**layer-8 attention 输出 4 真实
   行全 NaN** —— NaN 经 KV pad 槽进入下层 attention 的真实 query。
2. MLA-AUDIT 探针（forward_mla.py，SGLANG_MLA_AUDIT=1）：kv_a 单臂干净跑也有 SWA 层
   （46-77）pad 行 NaN（128 次，无害）；kv_a+o 组合 audit 暴涨至 7821 且 hidden=NaN 从
   layer 1 起。
3. 模块族二分（SGLANG_LORA_FAMILY，lora_manager.py）：kv_a/qb/o/kvb 四单臂全净；两两
   组合中 **kv_a+o = 13/32 乱**（最小犯罪组合），kv_a+qb / kv_a+kvb 净。
4. 离线 kernel 探针（probe1-8）：csgmv / kv_b absorbed（Q/V 修正）/ VE MoE delta 在生产
   形状下全部数值正确、canary 无 OOB —— kernel 数学清白，排除了"写坏邻居"假说。

**根因**：round-robin CP 把 batch pad 到 cp_size 对齐；`_cp_shard_row_request_ids` 把
**pad 行归属给最后一个请求** → shard 视图的 LoRA segments 对 padding 激活施加完整 adapter
delta → pad 行 KV 槽 NaN → 下层真实 query attend 到 NaN pad key → 整个真实 shard NaN →
logits NaN → argmax=0。base 干净（无 delta）、CP off 干净（无 padding）、单臂多为干净
（组合决定 pad 激活数值是否越过 NaN 界）。

**修复（chunked_backend.py）**：pad 行保留 -1 哨兵并映射到 **adapter slot 0（base 槽）**——
`lora_ranks[0]==0` 使所有 LoRA kernel 对其 segments 早退 no-op，slot 0 权重恒零双保险；
**segments 保持覆盖全行**（与验证过的结构一致）。
⚠️ 第一版把 pad 行**排除**出 segments（permutation 短于 x）→ warmup 即 IMA（absorbed
step kernels 未验证过该结构）→ 已回退。教训：**改 segment 结构必须保持 covered==x_rows
不变式**。

**同 commit 的伴随修复**：deepseek_mla_correction.py 的 `_get_state` 改走
`_resolve_batch_info(x_rows)`（absorbed kv_b 修正此前拿 full 视图路由 shard 张量）；
virtual_experts.py shrink kernel 拒绝负 offs_token + routing slack 负值钳哨兵。

**验证**（1101/1100，完整 L0 + CP on + 生产 flags）：电池（12 filler + 20 topic）×3 轮
**0/96 乱码**；MLA-AUDIT 计数 **0**（乱码时 7821）。

**判据工具**：1101 `/root/exp_battery.py`（乱码签名 ids[0]==0）、`/root/analyze_divergence.py`
+ `--debug-tensor-dump-output-folder`（逐层 NaN 定位）、SGLANG_LORA_FAMILY 二分、
SGLANG_MLA_AUDIT。

**方法论**：
1. "组合才坏"的 bug 用**运行时模块族二分**（一个 env，零 adapter 重打包）比读码快一个量级
2. 逐层 dump 的**行级 NaN 分析**（pad 行 vs 真实行）是定位毒化传播的金标准
3. 混沌系统里"净-净也发散"是常态——盯 **NaN/零化的质性签名**，别盯数值幅度

## 8. 第四 bug：slot 逐出 churn 毒化在途长请求（2026-08-21 二次结案，commits `37fae98ed7`+`d56db6bea5`）

> §7 修复上线后生产再次出现**中途数字汤**（18:25 事件：输出进行到一半突变成
> `2.2.0.0.2.2...` 类数字汤 ~500 字符后**自愈**，重试可恢复）。形态与 §7 的
> 首 token 乱不同。

**对照实验判决**（1102/1104 测试对，case50 LoRA 轮转 L0-L3，唯一变量=slot 数）：
| | 4-slot | 5-slot |
|---|---|---|
| 乱码 | **2 例**（数字汤 + `0\n`×97）| **0 例（50/50）** |
| LORA-EVICT | 5+ 次 | **0 次** |

**根因链**：`mem_pool._get_available_buffer_slot` 的 **prefer-LoRA-over-base**
特殊逻辑（952-961 行）在纯 LoRA 流量下把 base 钉死 slot0（chunked prefill 单请求
成批 → cur_uids 窄 → 其他 LoRA 永远是候选）→ 4 LoRA 挤 3 slot **连环 LRU 互逐**
→ 在途 350k chunked prefill（5 分钟）的 adapter 被中途换血 → 换血窗口内计算的
chunk 读到复用 slot 的**另一个 adapter 权重** → KV 毒化 → decode 读到毒化段即出
数字汤，adapter 重载完成后自愈。日志判据：`LORA-EVICT` 风暴 + 乱码时间窗重合。

**修复**：①删 prefer-LoRA-over-base——纯 LoRA 流量下 base（从不被请求）成为真
LRU 冷数据被逐一次，4 LoRA 共存 4 slot，零 churn 零显存代价（**生产主修复**）；
②5-slot 兜底（VE 模式每 slot ~10GB，prefill 12M→9M tokens 才装得下）；
③PR #31608 移植（fused_moe.py：hooks 活跃时禁 TMA down——TMA 的 expert-sorted
block-padded 序与 hook route-major 契约不兼容，B300 默认开 TMA）。

**遗留（defense-in-depth）**：逐出仍不检查在途请求引用（refcount）——长 prefill
不刷新 LRU 时间戳，理论上仍可被逐；生产 4 LoRA 常驻 + 无第 5 者时无触发面。

## 9. HiCache host/file KV 跨 adapter 污染（同日修复，commit `d56db6bea5`）

存储层 hash 链（`get_hash_str`）只掺 token：L0/L1 同 prompt → 文件级 key 完全相同
→ 一个 adapter 的 host/file KV 被另一个 adapter 直接命中（HiCache on 时的乱码源
之一）。修复：hash 链根部掺 `extra_key`（lora 身份）——`sha256("sglang-extra-key:"+
extra_key)` 作 prior_hash，后代页全部按 adapter 分叉；贯通 6 处（utils/radix_cache/
cache_controller×2/hybrid×2）。验证：全新 prompt 下 L0 暖 100% 自命中、L1 跨
adapter **0%**（修复前 100% 污染）、输出全净。
