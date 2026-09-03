# PD KV 传输 src/dst 错位根因：decode TP rank radix 命中分歧（2026-09-02 结案）

## 症状（1102/1104 GLM-5.3 PD 对）

flood 并发压测下偶发（~每 30s 一次）transfer_worker 抛
`boolean index did not match index array`（src 37 vs dst 9 页之类）→ 线程死亡 →
shard 队列孤儿化（session_port_sum % N 恒定路由）→ 静默冻结（health 200 但 KV 永不传输）。
prefill 侧 `PD-PFX-MIN` 亲测分歧形态：**8 个 decode TP rank 的 ACK 携带 8 个不同的
`decode_prefix_len`**（如 `[19456×7, 486400]`——7 个 rank 报 19456，1 个报 486400）。

## 根因链（五层，全部代码级确认）

### ① 分歧的产生：`decode_prefix_len = L1 + L2` 是 per-rank 本地状态

`decode.py:1250-1263`（`DecodePreallocQueue._pop_preallocated` 内）：

```python
prefix_match = self._match_prefix_and_lock(decode_req.req)   # per-rank radix 匹配
prefix_len = prefix_match.l1_prefix_len                       # 本 rank L1（显存）命中
total_prefix_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length  # + 本 rank L2（内存树）命中
# 注意：l3 故意排除在承诺之外（decode.py:1259-1262 注释——query_storage_hit_length
# 是乐观值，prefill 传 [l1+l2, fill) 自然覆盖 l3 区间）
```

每个 decode TP rank 是**独立进程**，各有自己的：
- **L1 显存 radix tree**——驱逐由本 rank 的 `available_size()` 水位驱动
  （`decode.py:_pre_alloc` 的 `tree_cache.evict(EvictParams(...))`），各 rank 的
  分配碎片/在途请求/free-list 状态漂移 → **驱逐时间和受害段不同**；
- **L2 host radix tree**（`--hicache-write-policy write_back`）——L1 驱逐段的
  L1→L2 异步写回完成时序 per-rank 不同；L2 满后 L2→L3 逐出也 per-rank；
- L3 查询 5s time-box（`decode_hicache_mixin.py:96-125`，文件 I/O 竞争防护）——
  但 l3 不进 `total_prefix_len`，只影响本 rank 的 restore 路径。

**没有任何跨 rank 同步**：`_match_prefix_and_lock` 是纯本地决策。8 rank 各自报各自
的 L1+L2 前沿。7 个 rank 的 L1 已把 `[19456, 486400)` 驱逐（且该段 L2 写回未完成
或 L2 已逐到 L3），1 个 rank 尚未驱逐 → 前沿差 25 倍。

### ② 分歧的上报：每个 rank 的 ACK 各带各的值

`decode.py:1934-1940`：`kv_receiver.send_metadata(..., decode_prefix_len=total_prefix_len)`
——**per-rank** 调用（每 rank 一个 scheduler 进程、一个 receiver）。prefill 的
bootstrap_thread 收到 8 个 ACK，`transfer_infos[room]` 里 8 个 `TransferInfo`
各有自己的 `decode_prefix_len`。

### ③ 错位的放大：sender 的页预算 vs 各 rank 的 dst 数组

prefill 侧 `req_to_decode_prefix_len[room]` 取 8 个 ACK 中的**一个**值算
`num_kv_indices_to_send`（sender 页预算 = 一个值）；而每个 rank 的
`dst_kv_indices` 长度 = **自己的** `fill_len - 自己的 total_prefix_len` 算出的
分配页数。原实现取 `next(iter(...))`（第一个 ACK）——第一个 ACK 是 19456 时，
sender 按 19456 算 src 页数（多页），报 486400 的 rank dst 只有少量页
（它的 radix 已覆盖 [0, 486400)）→ `src[dst_kv_indices 相关 boolean mask]`
IndexError。

### ④ 崩溃的传导：worker 死 → 队列孤儿 → 静默冻结

transfer_worker 的 except 原实现 `raise RuntimeError` → 线程死亡 → 该 shard 队列
后续 ADD-XFER 永远无人消费 → 600s watchdog SIGABRT / 请求超时堆积 → 系统冻结。

### ⑤ XFER-RANK-OFFSET 不触发的次生 bug（本次新发现）

精确对齐路径（per-rank offset 换算）一直读 `req_to_decode_prefix_len.get(room)`
恒为 None（13,931/13,931 全 None）。根因：**`CommonKVSender.pop_decode_prefix_len`
（common/conn.py:1208）用 `.pop()`**——`finalize_bootstrap`（prefill 事件循环）在
**第一个 chunk 入队之前**就把 entry 删了，transfer_worker 后续读永远是 None。
修复 = `.pop()` → `.get()`（清理职责仍在 transfer_worker Success 路径 +
`clear()`，均已有 pop）。

## 修复层（已部署 + 待部署）

| 层 | 内容 | 状态 |
|---|---|---|
| PD-PFX-MIN | bootstrap_thread 收齐 8 ACK 后取 `min()`（而非 `next()`）作 sender 页预算——覆盖需要最多数据的 rank | **已部署**，亲测 `prefix_lens=[19456×7, 486400] min=19456` 生效 |
| XFER-RADIX-TRUNC | transfer_worker 检测 src>dst 时截断 src 尾到 dst 长（高位 rank 的 radix 已覆盖前段，尾页才是它需要的），不 fail 请求 | **已部署**，等分歧重现验证（TRUNC 计数 + 请求成功） |
| XFER-SRC-DST-MISMATCH | src/dst 长度不匹配兜底：fail 单请求不杀 worker | **已部署** |
| XFER-WORKER-CHUNK-ERROR | worker 线程免疫单 chunk 异常（隔离到单请求失败） | **已部署** |
| XFER-RANK-OFFSET 修复 | `pop_decode_prefix_len` 的 `.pop()` → `.get()`（见 ⑤） | **本地已修**，未部署（不重启纪律；下次窗口随代码同步） |

## 分歧是否 bug？

**分歧本身是 per-rank radix 架构的固有属性，不是 bug**——8 个独立进程的
（L1 驱逐前沿, L2 写回可见性）本来就是各自本地内存管理决策。**bug 是 PD 传输协议
假设了 8 个 ACK 同值**（`next()` 取第一个 + sender 页预算单一值 + dst per-rank）。
修复方向是让协议分歧容忍（min + 截断 + per-rank offset），不是消灭分歧。

## 2026-09-03 02:10 连锁崩溃：L2 恢复失败的单方面 abort → batch 分歧 → NCCL watchdog SIGABRT

同一 per-rank 状态分歧家族的**第二种致命形态**（与上述 KV 传输错位不同路径）。

### 崩溃链（全部实证）

```
TP0 的 L2 host radix 与 TP1-7 分歧（TP0 的 L2 在 match 后 load_back 前丢了 323072 token——
  per-rank L2 驱逐/写回时序，l1=26880 l2=323072 new_indices=0 实证）
→ _try_hicache_queue_load_back 覆盖不足 → hicache_restore_status=FAILED（仅 TP0）
→ HiCacheRestoreGatedKVReceiver.poll 只 gate PENDING（FAILED 直接漏过返回原始 poll）
→ pop_transferred 的本地分支 `or hicache_restore_status == FAILED` 单方面 abort
  （不走集体决议——padded all_reduce min 从未见过这个失败）
→ TP0 队列 19 vs TP1-7 队列 20（[PADDED-AR] len=19 vs 20 铁证 @02:10:12）
→ TP1-7 COMMIT/kv-arrived 把请求放进 running batch，TP0 没有 → batch 成员分歧
→ EAGLE verify 的 eagle_sample broadcast（PG2, NumelIn=18）跨 rank batch 不齐
  （rank0 走到 190399、rank1-7 卡 190393——TP0 的 batch 小一轮跑更快）
→ NCCL 600s watchdog → SIGABRT（exit -6）→ scheduler_1 崩 → SIGQUIT 全树 → decode 死
```

### 关键证据（1104 /root/glm53_decode.log.crash-0903）

- `02:09:28 TP0 HiCache load_back failed for rid=c2bb1b71...: device_indices=26880, new_indices=0, expected decode_prefix_len=349952 (l1=26880, l2=323072, l3=0)`（TP0 单 rank 失败；TP1 同刻 `load_back_end tokens=323072` 成功）
- `02:10:12 TP0 [PADDED-AR] len=19 max_len=20` vs `TP1-7 len=20`（冻结点最后日志）
- `COMMIT rid=c2bb1b71` 只出现在 TP1-TP7（TP0 没有）；CACHE_UNFINISHED 同
- `[rank1-7] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=190393, OpType=BROADCAST, NumelIn=18) ran for 600004ms`（eagle_utils.py:758 eagle_sample 的 TP broadcast；failed collective 栈：run_eagle_verify→eagle_sample→broadcast）
- `[rank0] Watchdog ... SeqNum=190399`（TP0 领先 6 个 op——batch 小跑更快）
- 6 scheduler 主线程卡 `process_batch_result → copy_done.synchronize`（batch_result_processor.py:657）
- `scheduler_1 crashed with exit code -6` → SIGQUIT → kill_process_tree

### 修复（2026-09-03 部署 1104）

三件套（`KVPoll.Failed=0` 是 min 语义的最强传播——Failed 压过 Transferring/Success）：

1. **`decode_hicache_mixin.py::HiCacheRestoreGatedKVReceiver.poll`**：restore FAILED → 直接返回 `KVPoll.Failed`——进 padded all_reduce（min），**同一迭代所有 rank 统一 abort**（不再单方面）
2. **`decode.py::pop_transferred`**：删掉 `or hicache_restore_status == FAILED` 单方面检查（只信集体决议的 `poll == Failed`）；FAILED-restore 时 error_message 追加标记（grep: `uniform abort via gated-poll all_reduce min`）
3. **`decode.py` 异常降级路径**：`_process_hicache_local_restores` 异常时 PENDING→**FAILED**（原 READY 会让 unrestored prefix 带 `hicache_restored_node=None` 走到 COMMIT 的 AttributeError，且同属单方面分歧）

### 判据

- 修复后 L2 恢复失败（~1/77min flood 频率）的表现：**8 个 rank 同刻** `[Decode transfer failed ... uniform abort via gated-poll all_reduce min]`（TP0 loud error + TP1-7 propagated debug）+ `load_back failed` warning（TP0）→ flood 单请求 503（fails+1）但**继续跑**——不再是 batch 分歧崩溃
- `[PADDED-AR]` 的 len 跨 rank 必须恒等（分歧即复发信号）
- 复发崩溃形态（对照）：len 分歧 → COMMIT 只在部分 rank → NCCL watchdog SIGABRT（exit -6）

### 同族判别（为什么不是其它嫌疑）

- **不是** d11d2b5179 的 GPU 楔死家族（reduce 相位自旋）：无 Xid、卡点在 NCCL broadcast 的集合等待不是 kernel 自旋；549K 巨型请求（8a87b50a）在飞但未 kv-arrived，不是触发者——触发者是 350K L2 恢复请求 c2bb1b71
- **不是** 传输 src/dst 错位（PD-PFX-MIN=0/TRUNC=0/MISMATCH=0/WERR=0 全零——那是修复过的好好的）
- **是** DSPark batch 分歧家族（AGENTS 2026-08-24 `DSPARK-BATCH-DIVERGE` 同构）：单 rank 本地状态分歧 → batch 成员跨 rank 不一致 → TP 集体不齐 → NCCL 挂起。当时只给 DSPark 路径加了 `_assert_batch_bs_rank_invariant` 防护；EAGLE 路径的分歧入口是 HiCache restore FAILED 的单方面 abort，本修复从源头（集体决议）解决


## 判据（1102 prefill log，SGLANG_DEBUG_DIAG=1）

- `PD-PFX-MIN`：分歧检测（`prefix_lens=[...] min=`）——分歧重现时非 0
- `XFER-RADIX-TRUNC`：src>dst 截断触发（防护生效 = 请求成功不 500）
- `XFER-SRC-DST-MISMATCH` / `XFER-WORKER-CHUNK-ERROR`：兜底触发（应为 0 或偶发）
- `XFER-OFFSET-DBG`：room_min 值（.get 修复部署后应非 None）
- decode log `[BS-T] prealloc-done`：per-rank `prefix_len`/`total_prefix`（8 行/rid，
  可直接看跨 rank 分歧现场）

## 冷启动注记

decode 重启后 radix 冷（1102 rid × 8 rank 全一致，PD-PFX-MIN=0）——分歧需要
L1 填满 + 驱逐压力积累 + L2 写回时序漂移后才出现（上一次 run 2.5h 内 312 次）。
冷启动"零分歧"不代表修复，只代表树还没漂。
