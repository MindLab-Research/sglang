# PFX-SYNC：跨 rank prefix min 协商——per-rank radix 分歧的源头消灭（2026-09-03，`3812e4febb`）

## 背景：传输层防护（治标）→ 源头消灭（治本）

per-rank radix 分歧（`decode_prefix_len = L1_hit + L2_host_hit` per-rank 本地状态，
L1 驱逐/L2 write-back 时序各 rank 独立）产生 8 个不同的 ACK → prefill 传输层
错位。c9963809ce 在传输层处理（PD-PFX-MIN min-of-8 + XFER-RADIX-TRUNC +
XFER-RANK-OFFSET + MISMATCH 兜底）——**必要但治标**：分歧风暴（~1200 room/h）
下每条处理路径长出残余 bug（offset 下游 IndexError WERR、空dst MISMATCH 503），
~2-4% 请求失败于分歧窗口。根因是**分歧到达了传输层**——本修复让分歧
**根本不出现**：8 rank 在任何 ACK 之前集体协商 min(total_prefix_len)。

## 设计（decode.py DecodePreallocQueue，186 行）

### 1. `_pfx_sync_collective`（集体协议）

每 pop_preallocated tick 一次 all_gather：per-room `(bootstrap_room, cached_total)`
对。**毕业（graduation）**：room 在全部 rank 的 waiting 集且 8 个 cached total 齐
（`pfx_sync_min` 置上，STICKY——不重算）。rank-invariant 协议：
(1) all_reduce MAX 对表长度（恒调）；(2) max>0 时 all_gather（AR 后 max 全员
同值→条件在所有 rank 上一致）。补偿路径（retracted 提前返回、polling else）
调空协议保持 gloo 计数对齐。

### 2. 两阶段 prealloc（每请求 +1 tick）

- **tick N**：radix match + 缓存到 `pfx_sync_match`（inc_lock_ref 持锁跨 tick）+
  延迟（continue，不出队）
- **tick N+1**：毕业（min 齐）→ 钳位 + `_pre_alloc` + ACK（`decode_prefix_len=`
  钳位后的 min，8 rank 同值）

+1 tick/请求延迟 invisible：prefill bootstrap 本就串行最慢的 8 个 ACK。

### 3. 钳位（divergent range 重收）

own_total > min → `prefix_match.prefix_indices` 切到 min（L1 property=
len(prefix_indices) 自动更新）、l2/l3 清零。分歧段 [min, own) 从 prefill 重收
（~40MB/事件，mooncake RDMA 下可忽略）。8 个 ACK 同值 → **传输层（offset/TRUNC/
MISMATCH/state-path）全部变死代码**（保留为安全网）。

### 4. budget 断裂锁保留 + FINISH_ABORT 泄漏修复

- budget break（3 处）：**不再 dec_lock**——毕业请求的 cached match 锁跨
  budget 等待持有（释放=页可驱逐=下 tick 缓存 match 指向已驱逐页=KV 损坏）
- FINISH_ABORT 扫描：释放缓存 match 的锁（延迟中的请求被 abort 时防泄漏）

## 正确性论证

- **8 ACK 同值**：毕业 min 是同 tick 同输入的确定性 all_gather 输出（每 rank
  算出同值）；sticky min 跨 budget skew 存活（A 先 prealloc，B 后 budget 恢复
  后用同值——ACK 一致）
- **无 KV 洞**：钳位段 [min, own) 由 prefill 重传（B rank 的 dst 数组按
  [min, fill) 分配——min 是全 rank 最小，A 的 own>min 部分被切掉重收，
  [0, min) 全 rank 都有 radix 覆盖）
- **无新死锁**：集体 per tick 恒调（空也调）+ 补偿路径对齐；毕业需 8/8
  presence（bootstrap 时序 skew 1-2 tick 内收敛）；budget skew：毕业后的
  sticky min 不需重新协商（无 livelock）
- **锁安全**：cached match 的锁从 match tick 持到 prealloc/abort（budget
  等待不解锁）；FINISH_ABORT 释放

## 判据（1104 部署后）

- `[BS-T] prealloc-done ... pfx_sync(min=X own=Y)`——8 rank 同 min（ACK 均匀）
- `[PFX-SYNC] room=... graduated min=N`（DEBUG_DIAG）——毕业事件
- `[PFX-SYNC] rid=... own_total=N clamped to agreed min=M`——**钳位=分歧被
  源头消化**（deployed 后风暴重现时替代 MISMATCH/WERR 503）
- **prefill 的 `PD-PFX-MIN` 永不再触发**（ACK 均匀）——终极判据
- `MISMATCH/WERR/OFFSET` 归零（传输层死代码不再执行）

## 残余边界

- 分歧段重收的带宽（~40MB/事件，RDMA 下 ~0.1%）
- L2 restore 失败（c78576a835 统一 abort）独立存在——那是 L2 eviction
  TOCTOU，不是 ACK 分歧
- +1 tick/请求 prealloc 延迟（~10-50ms，invisible）
