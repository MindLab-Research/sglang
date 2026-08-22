# Prefill "OOM" 记账分裂事故（2026-08-22 结案，commit `6504ba9b71`）

## 现象

R2 缓存命中轮（50 并发 L2 burst，含 222K/787K 字符巨型 case）跑 ~4 分钟后，测试 prefill
8 个 TP rank 同时 `RuntimeError: Prefill out of memory. Try to lower your batch size.
Try to allocate 16384 tokens. Available full tokens: 1663744 (available=26368 +
evictable=1637376)` → coredump 流程 → 进程树退出。下游级联：2 个已 bootstrap 请求在
decode 侧 `KVPoll.WaitingForInput` 600s 超时（2× 500），router health check 失败 →
circuit open → 26× 503。**不是显存耗尽**（evictable 还有 1.6M token）。

## 静态证明（无需复现）

`alloc_extend` 失败条件：`num_new_pages > len(free_pages)`，其中
`get_num_new_pages = Σ(ceil(seq/64) − ceil(prefix/64))`（utils/common.py:4196，注意是
**ceil−ceil** 不是 ceil−floor）。良构批次上界：

```
num_new_pages ≤ extend_num_tokens/64 + bs   (崩溃时: 256 + ~35 = 291)
available     = 412 页 (free+release, 26368/64)
291 < 412 → 良构批次数学上不可能失败
```

失败发生 ⇒ ∃请求 `extend_range.end > len(full_untruncated_fill_ids)`（差距 ≥7-10K
token）：`input_ids = fill[:end][prefix:]` 受 fill 实际长度限制（数据侧），而
`seq_lens = extend_range.end` 是调度时刻的旧账（声明侧）——**三个独立维护的记账源撕裂**
（Req 契约注释自认 "in-place rewrites would silently corrupt fill_ids"；嫌疑生产者：异步
HiCache loadback 调整 prefix_indices、就地改写 origin_input_ids、CP 共识截断）。
alloc 按虚高的 seq 算页数 → fail-loud 杀全组。

**abort 理论已被日志否定**（崩溃窗口 19:00-19:08 零 abort/disconnect）——不要重复查。

## 两个真实缺陷与修复（`6504ba9b71`）

1. **记账源分裂**（真凶）：`schedule_batch.py::prepare_for_extend` 在三源汇合点强制
   不变量——实际 input_ids 为单一事实源，`seq_lens[i] = prefix_lens[i] +
   len(input_ids[i])`，同步 `set_extend_range`，响亮日志
   `[EXTEND-ACCOUNTING-DIVERGENCE]`（rid/declared/actual/end/prefix/fill_len 全打）。
2. **evict/alloc 口径分裂**（放大器）：evict 门槛用 `available_size()`（token 域，
   free+release），alloc 判定用 `len(free_pages)`（page 域，仅 free）——两个视角永不
   相遇。修复：`allocation.py::alloc_paged_token_slots_extend` 用 allocator 自己的
   `get_num_new_pages` 算精确需求页数，按页缺口驱逐（merge 会把 release 并回 free，
   所以 need vs free+release 是正确比较）。

最终 raise 语义不变（真耗尽仍然 fail-loud）。

## 判据（grep 用）

- `grep EXTEND-ACCOUNTING-DIVERGENCE prefill.log` → 修复后出现 = 抓到生产者实锤
  （fill_len 与 end 的差值就是撕裂量）；持续为 0 = 该路径未再触发
- `grep PREFILL-ALLOC-FORENSICS prefill.log` → 崩溃点取证（extend_arg vs
  extend_actual、free/release 页数与重复数、prefix/seq 前 8 项、inverted 标志）
- `grep "Prefill out of memory" prefill.log` → 应保持 0；再出现即真耗尽（查池容量/并发）

## 相关事实

- `#token: 0` 的 Decode batch 行 = abort 窗口签名（KV 已 free、请求未 filter），
  不是 0-token 请求进 running——排障时勿被误导（pool_stats_observer.py:217）。
- 测试对池 1.78M（CP off 后无 layersplit 分摊）；`--max-running-requests` 默认 2048。
  测试脚本现设 16（等比生产 9M/64 的 1/5）。生产 1101 池 9M 未动。
- R2 命中轮的准入特征：全命中请求几乎不耗计算，调度器瞬时接纳大量请求，radix 保护页
  （matched prefix 全部 in-use）+ 每 chunk 保守预留叠加 → available 低水位，是本 bug
  的最佳触发窗口；R1 冷轮逐 chunk 限流反而难触发。
