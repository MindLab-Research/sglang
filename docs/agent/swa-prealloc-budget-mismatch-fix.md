# SWA Prealloc 预算/分配口径分裂 + decode 配对重启事故（2026-08-25 结案）

## 症状

case50（50 并发真实请求）在 prefill 开 CP layer-split + HiCache、decode 开双端 radix + DSPARK 的稳定配置下：

- decode 端 `Decode handshake failed`=40、`KVTransferError`=40
- prefill 端 `Prefill bootstrap failed`=72
- 异常统一为 `KVTransferError(bootstrap_room=...): Aborted by AbortReq`
- TTFT 极慢（>10 分钟跑不完 50 个请求）

## 根因一（核心 bug）：`_pre_alloc` SWA 预算 vs 实际分配口径不一致

**位置**：`disaggregation/decode.py` 的 `alloc_for_decode_prealloc` / `_pre_alloc`

**机制**（有日志实锤）：

```
[PD-PREALLOC-KV-FULL] available=34816, evictable=0, protected=49664,
  required_alloc=46848, delta=46728, fill=48520, prefix=1792, total_prefix=1792, page_size=256
```

1. **准入预算**（`_prealloc_required_tokens` → `_swa_tail_len`）按 **SWA 窗口尾窗** 计算
   （sliding window ~16K ≈ 65 页）→ 判定"够"。
2. **实际分配**：旧代码 `uses_swa_tail = _uses_swa_tail_prealloc() and prefix_len == 0`。
   decode radix 命中（`prefix_len=1792 > 0`）时该门控为 False → 走 `alloc_extend`
   （swa.py:188 `new_pages_available(num_new_pages, num_new_pages)`），**要求 SWA 全量页
   （183 页 = 46848 token）**。SWA 池仅 0.3 ratio（`--swa-full-tokens-ratio 0.3`），高并发
   下可用只剩 136 页 → `alloc_extend` 返回 None。
3. `_pre_alloc` 返回 None → `pop_preallocated` 里 `prepare_abort` + `kv_receiver.abort()`
   → **AbortReq 发给 prefill** → prefill 报 `bootstrap failed ... Aborted by AbortReq`，
   decode 自己报 `handshake failed`。

即：**预算说"SWA 只需尾窗，够"；实际分配说"SWA 要全量，不够"→ abort**。
与 hicache 快慢无关——是 decode 端 `prefix>0` 触发 SWA 全量分配的固有崩溃点。
prefill 开 HiCache 只是通过 radix 命中率的提升提高了 `prefix_len>0` 的触发概率。

**修复**（`alloc_for_decode_prealloc` + `_pre_alloc`）：

- 移除 `and prefix_len == 0` 门控：`prefix>0` 也走 `alloc_extend_swa_tail`。
- 传真实 `prefix_lens=[total_prefix_len]`、`extend_num_tokens=delta_len`（旧代码硬编码
  `[0]`/`fill_len`，只对 prefix==0 正确）。
- `swa_tail_eff = max(0, min(swa_tail_len, delta_len))` clamp 保护
  `assert(0 <= swa_tail_len <= extend_num_tokens)`。

**正确性论证**：decode 端滑动窗口在序列**末尾**；`match_prefix_for_req` 不匹配最后一个
token，故部分命中时 `total_prefix_len < seq_len - window`，窗口完全落在新分配的 delta
区间 `[total_prefix_len, fill_len)` 内，尾窗切分语义与全量等价。SWA 页需求 183→65，
SWA 池充足。数值验证：`get_num_new_pages(48520,256,1792)=183=required_alloc/256` 与日志
逐位吻合。

**判据**：`grep PD-PREALLOC-KV-FULL decode_v4.log`=0；`grep "Aborted by AbortReq"`=0。

## 根因二（运维事故）：prefill 重启后 decode 未配对重启

排障中 prefill 因 NCCL watchdog 崩溃后重启（00:36），decode 未重启。后果：

- decode（旧进程）与旧 prefill 的 mooncake/RDMA 会话全部失效
- decode 日志持续 `Attempting to reconnect to 10.0.58.35:8998...`
- 所有新请求 KV 传输失败 `Aborted by AbortReq`，router 冒烟全 hang
- 曾误诊为 "decode scheduler collective 错位死锁"（PADDED-AR count 各 rank 差 ~8500）——
  实为不同 gloo group 的正常节奏差，decode 心跳一直在推进，**未死锁**

**修复**：重启 decode 配对（AGENTS 铁律 "prefill 重启后 decode 必须重启配对 RDMA" 重申）。

**教训**：排障中任何一端引擎重启后，**另一端必须立即配对重启**，再继续诊断。
在传输层会话失效的状态下调 router/入口层是浪费时间（本次浪费约 2 小时）。

## 排障方法论沉淀

1. **`Aborted by AbortReq` 的第一怀疑对象是对端主动 abort**：先 grep
   `PD-PREALLOC-KV-FULL` / `Could not fetch prefill parallel info` / hidden-pool
   invariant，定位发起 abort 的代码点，不要先怀疑超时。
2. **PADDED-AR count 对比**：各 rank count 差异大 ≠ 死锁。先确认日志时间戳是否仍在
   推进（count 在涨 = 心跳活着）；真死锁是时间戳+count 双停滞。
3. **深测入口**：`sglang-router launch --pd-disaggregation --prefill ... --decode ...
   --policy round_robin --port 30000`（测试不需要 mol proxy / smg / cache_aware）。
   本地访问走 SSH 隧道 + `curl --noproxy '*'`（本地 HTTP 代理会劫持 localhost 返回 502）。
4. **预算口径一致性**：任何"预算判定够、实际分配失败"的组合都是 bug——检查两处是否
   用同一个 token 计算函数。

## 验收（case50 四轮，rpm=200，经 router:30000）

| 轮次 | 时长 | ok/50 | err | 乱码 | cache_ratio | TTFT p50 |
|---|---|---|---|---|---|---|
| R1（冷） | 2m27s | 42 | 8 grammar | 0 | 0.0017 | 40.4s |
| R2（热） | ~2m10s | 42 | 8 grammar | 0 | 0.9947 | 33.6s |
| R3（热） | ~2m | 42 | 8 grammar | 0 | 0.9947 | 33.2s |
| R4（热） | 见 git log/补遗 | 42 | 8 grammar | 0 | ~0.99 | ~33s |

- 8 个 err 全部为 `DFLASH speculative decoding does not support grammar-constrained`
  （case50 固有 ~8 例，模型能力边界，非部署缺陷，与 2026-08-23 结论一致）
- 全程 `Aborted by AbortReq`=0、`PD-PREALLOC-KV-FULL`=0、`bootstrap failed`=0
- 回答抽查有逻辑（技术分析连贯，非胡言乱语）
- HiCache 冷/热分层生效：R1 全 miss（写回）→ R2+ 命中 99.5%
