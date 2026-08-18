# DSPARK PD 分离：并发死锁修复链（2026-08-15）

五个叠加的 collective 死锁 bug，全部与"per-rank 状态分歧 → gloo/NCCL collective 序列错位"同构。修复后 v39 验证 72/72（含 abort 洪峰 + 空闲后并发）0 crash。

## 修复清单

| commit | bug | 修复 |
|---|---|---|
| `2dd9d2c168` | decode 的 pop all_reduce 与 broadcast 共用 attn group，abort 清理时差 → 异构 collective FIFO 互配 | disagg poll 专用 gloo group（`torch.distributed.new_group(backend="gloo")`），三队列（DecodeTransferQueue/DecodePreallocQueue/PrefillBootstrapQueue）共用 |
| `c841e03cf9` | `poll()` 抛异常在 `_padded_all_reduce_min` **之前**逃逸 → 该 rank collective 计数永久偏移 1 → 领先一整轮互等 | `_poll_with_failure_injection` 每个 poll 包 try/except → 转 `KVPoll.Failed` + 异常 stash 到 receiver；+ 300s group timeout（残余分歧响亮崩溃）+ `PADDED-AR-FAIL` 计数取证 |
| `1f0e0cc95a` | **prefill** `pop_bootstrapped` 空队列分支直接 return 跳过 CP collective（bootstrap 队列 per-rank TCP 时序分歧）→ attn_cp 组永久错位 → 8 rank 全卡 | 空分支在 `attn_cp_size > 1` 时用空 poller 列表参与 collective；同系列把 `poll_and_all_reduce_attn_cp_tp_group` 改用异常安全 poll（裸 `int(poller.poll())` 会跳过链式两个 `_padded`） |
| `0062368b6d` | #31466 部分移植后 4 并发 crash：新 prebuilt batch 的 `build_disagg_draft_input` 返回 None（spec_info.py 缺 DSPARK 分支）→ merge 跳过 → stale bs=1 draft_input vs bs=N batch → cuda graph `fill_from` shape 崩 | spec_info.py 补 DSPARK 分支（`make_next_draft_input`）；merge_batch None 守卫；fake/nixl/mori `send_metadata` 加 `spec_metadata` kwarg |
| `05214bd6b3` | 高并发 pending_bootstrap req 在 inflight queue 路径 hidden 未 materialize 就 send_kv_chunk → RuntimeError 崩 8 rank | `_pd_hidden_payload` 在 src_indices None/written 不完整时返回 `[]`（该轮不发 hidden），decode 端等后续 chunk |

## 上游相关

- #31466（DSPARK PD hidden state 传输）：我们移植了核心（16 文件），后续发现还有 PD hidden pool materialize 时序细节没跟全
- #31513（DSPARK disaggregation decode 三 crash 修复）
- #34343（DCP+DSPARK PD）：draft KV 几何问题，我们广播路径绕过，不适用
- **DSpark PD 要求两端 radix 策略一致**：prefill 必须去掉 HiCache + `--disable-radix-cache`（#31466 握手检查，备份 `.bak_hicache`）

## 死锁形态识别（py-spy）

- **health 200 + 请求全超时 + 0 crash** = collective wedge（非 crash）。py-spy 看 scheduler_TP* 主线程：
  - 部分 rank 在 `_padded_all_reduce_min` vs 部分在 broadcast → 同 group 异构互配
  - 全部 rank 在同一 gloo 点 → 该 group 内调用次数错位（某 rank 多/少调了一次）
- 修复原则（AGENTS.md 已有）：**collective 必须 rank-invariant**——次数、时序对 receiver 异常和 per-rank 队列状态免疫

## 非 collective 型：hidden pool 耗尽 → 个别请求永久卡死（2026-08-18，commit `1c9e1c3275`）

与前五个 collective 死锁不同族：**单个请求永久卡死**，其余正常，health 200、0 crash。

### 死锁环（时序错位，非 collective）

```
decode receiver.init()（bootstrap）先于 hidden rows 分配
→ prefill sender 一进 WaitingForInput（无超时）就等 decode 发 KV indices
→ pool 满时 pop_preallocated 的 alloc()==None 无限 continue（indices 永不发出）
→ sender 永久卡死；且 _release_pd_hidden_rows 的 wait_ack 300s 阻塞跑在调度线程
  → 整个 decode 冻结 300s；超时后 return 不释放 → rows 泄漏 → pool 永久满
```

**关键语义**（hidden_events.py）：`wait_ack_completions` 是 **decode 本地**等待
（等自己的 chunk 注入 CUDA 事件 + drain 循环，正常 µs 级完成，ACK 由 decode 发给
prefill）——不是等 prefill。所以非阻塞探测它完全安全。

### 结构性修复（`1c9e1c3275`，替代 cd15c85d68 的超时兜底；用户红线：零超时依赖）

1. **预留先于 bootstrap**（`_try_prealloc_pd_hidden_rows`，挂在 `add()` fast path
   与 `_resolve_pending_reqs` 的 `init()` 之前）：prefix-free 上界
   `min(_rebootstrap_prefill_len, pool.size)` 预留。pool 满 → 请求留在
   `pending_reqs` 不 bootstrap → **sender 根本不存在，无卡可谈**（背压）。
   pop 时按实际 window_rows 裁剪多余还回。不变量：
   `window_rows = min(hidden_len, pool.size) ≤ min(upper, pool.size) = len(reserved)`
   （hidden_len ≤ upper；radix match 少最后 1 token ⇒ total_prefix ≤ len-1）。
2. **释放非阻塞化**：`wait_ack_completions(room, timeout_s=0.0)` 探测；未完成则
   park 到 `_pending_pd_hidden_releases`，`pop_preallocated` 顶部每 tick drain
   （事件循环常驻 + polling_interval 节奏 ⇒ 空闲也 drain）。永不放弃（不泄漏）、
   永不阻塞调度线程。释放幂等（所有权字段置空），可被 FINISH_ABORT 扫描与
   pop 尾部 failed_reqs 循环重复调用。
3. **alloc-None → fail-fast abort(503) + `kv_receiver.abort()`**：预留机制下
   已 bootstrap 必有 reserved rows，alloc 失败只能是不变量违规（确定性失败，
   非 spin 非超时）。abort() 通知 prefill 释放 sender。
4. **pop 尾部统一释放 failed_reqs 的 hidden rows**：6 个 config-abort 分支
   （pool 布局/streaming/window 检查）在预留存在时 abort，不释放会泄漏。

### 遗留约束

- 预留上界会瞬时占满 pool（大请求 + 同 tick 并发）→ 同 tick 串行 prealloc，
  每 tick 解锁一次（ convoy 有界，ms 级）。
- rank 间 pool 状态可短暂分歧（park drain 时差 1 tick）→ bootstrap 时序分歧。
  安全：prealloc 队列无 collective（local-only poll），旧代码本就有同类分歧
  （且是永久性的），现在只剩 1 tick。
- 判据 grep：`PD hidden ACK still pending at release; parking`（正常少见）、
  `invariant violated`（不应出现，出现=有路径绕过了预alloc）。
