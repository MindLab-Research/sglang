# PD 稳定性修复（2026-08-24）—— 全集群楔死 + decode 崩溃双根因

> 背景：V4 Pro DSpark 1P1D 集群（8.222.11.182，1102 prefill / 1104 decode，DCP=4）
> 先出现"health 200、请求全挂、零 crash"全集群楔死，修复后压测又揭出 decode 偶发崩溃。
> 本文记录两个根因 + 修复 + 验证判据，供未来会话直达结论。

## 1. 全集群楔死 = mooncake 控制通道 torn 2 帧

### 症状
- 所有请求 hang，`/health` 两边都 200（HTTP 层活着），零 crash，请求全超时。
- decode 日志 `[PADDED-AR-FAIL]`（或 collective count 分歧），`decode_thread` 崩。

### 根因（此前的"畸形 2 帧未知来源"终于定位）
`common/conn.py::CommonKVManager._connect` 返回**跨线程共享的缓存 PUSH socket**，
但 send **完全无锁**。decode `--dcp-size 4` 下最多 8 个 transfer_worker 线程并发
向同一个 endpoint（decode 的 rank_port）发控制消息。

- libzmq **不跨线程串行化 multipart 发送**（pyzmq 在帧之间释放 GIL）。两个线程同时
  `send_multipart` 一条 3 帧消息 `[room, status, prefill_rank]`，帧会交错——A 的消息被 B
  劈开，接收端收到**畸形 2 帧**。
- decode 端 `decode_thread`（只启动一次、永不重启）解包 `bootstrap_room, status,
  prefill_rank = msg` 抛 `ValueError: expected 3, got 2`，线程死。
- 线程死 → 各 rank 的 gloo/gloo 集体调用计数漂移 → `_padded_all_reduce_min` 永不完成 →
  全集群楔死。**HTTP 200 是因为 health 由独立于控制线程的组件服务。**

### 修复（无 hack、无性能损耗）
`common/conn.py`：
1. `__init__` 加 `self._socket_locks: Dict[str, threading.Lock]`。
2. 实例 `_connect` 改为返回 `(sock, lock)`（每 endpoint 一把锁，对齐 receiver 侧
   classmethod `_connect` 早已存在的 `(sock, lock)` + `with lock:` 契约——它一直是对的，
   prefill 发送侧漏了）。
3. 新增 `_send_multipart(endpoint, is_ipv6, frames)`，内部 `with lock: sock.send_multipart(frames)`。
4. `mooncake/conn.py` 6 个发送点（`notify_pd_hidden_chunk_ready`、`ack_pd_hidden_chunk`、
   `_send_chunk_ready`、`_send_aux`、`sync_status_to_decode_endpoint`、ABORT_ACK）全部改走
   `_send_multipart`。

另曾加过 `decode_thread`/`bootstrap_thread` 的 `[MOONCAKE-RECV-DROP]` try/except 兜底
（让未知/畸形帧不杀线程）——那是**止损**，根因修复是本节的锁；兜底仍在，作为纵深防御。

### 判据
- `grep MOONCAKE-RECV-DROP decode_v4.log` 应 ≈0（锁生效后 torn 帧消除）。
- `grep PADDED-AR-FAIL` 应 =0（无 collective 分歧）。

## 2. decode 偶发崩溃 = DSPark all_gather 跨 rank batch 分歧

### 症状
压测偶发 `[rank0] collective timeout → dump signal` → `Fatal Python error: Aborted`。
Stack 落在 `deepseek_v4_dspark.py::_apply_step_logits_sharded → attn_tp_group.all_gather`。

### 根因
`attn_tp_group.all_gather(step_local, dim=-1)` 要求所有 rank 的 leading batch 维完全一致。
decode 端巨型并发下偶发某 rank 的 schedule batch 与同组其他 rank 分歧（某 rank 先完成/移除
一个请求），shape 不匹配 → NCCL collective 挂起 → abort。

两轮实验排除：
- 排除 draft CUDA graph：`SGLANG_DSPARK_PD_DISABLE_DRAFT_CUDA_GRAPH=1`（回退 graph）仍崩。
- 排除我的锁修复：干净 HEAD（无本次改动）同样崩，且 `DROP=0/TT=0/PADDEDFAIL=0` 证明
  mooncake 控制通道正常。

### 修复（rank-invariant 防护，无 hack）
`dspark_worker_v2.py`：
- 新增 `DSParkBatchDivergence(RuntimeError)`。
- `_assert_batch_bs_rank_invariant(bs)`：`attn_tp_group` scalar `all_reduce(MAX/MIN)` 让
  **所有 rank 算出相同全局 batch_size**；不一致则**所有 rank 一致 raise**（rank-invariant，
  不会出现某 rank 继续、某 rank 抛异常的 collective 错位）。
- 在 `_forward_decode` 的 `bs=len(batch.seq_lens)` 之后、`propose`（其内部 all_gather）之前调用。
- `forward_batch_generation` 捕获该异常 → 返回 `_decode_idle_result`（跳过该 batch，请求下
  轮重试）。**分歧时不进 all_gather，NCCL 不挂。**
- 修复过程中踩坑：sglang `GroupCoordinator.all_reduce` 只接受 `(self, input_)`，不接受
  `op=` 关键字（固定 SUM）——必须用 `torch.distributed.all_reduce(..., op=MAX/MIN,
  group=gp.device_group)`。

### 判据
`grep DSPARK-BATCH-DIVERGE decode_v4.log` 应 =0（压测 24min 0 次触发 = 没再分歧）。

## 3. 请求终态保障 —— 既有代码，本次只验证

用户要求"任何请求不卡死/无长时无进度/除网络外不中途失败"。读代码确认这是当前 HEAD
**已有**实现，非本次新增：
- sender `_check_waiting_timeout` → Failed → 同步 decode（common/conn.py）
- receiver `_check_bootstrap_timeout`（common/conn.py）
- PD hidden chunk gap `PDH_GAP_TIMEOUT_S` → record_failure → abort（decode.py）
- KV pool 耗尽 `PD-PREALLOC-KV-FULL` → 503 clean abort（decode.py）
- 预填信息获取失败 `_max_ensure_retries` → receiver.abort（decode.py）

验证：6 轮 × 50 请求（RPM 220 + 随机 abort）终态审计 = 238 成功 + 15 abort + 47 grammar 400，
**0 卡死 / 0 无进度 / 0 除 abort 外中途失败**；24 分钟双端持续采样全绿。

## 4. 铁律（已写入 AGENTS.md §5.4）
1. decode 端 radix 绝对不能关（PD + DSpark 双端 radix 是架构前提）。
2. 禁止"不改代码只改参数就重启"；二分以 commit 为单位。

## 复现方法（供后人）
- 循环波动一个 launch：torn frame 是极偶发并发时序 bug，靠高并发 + 长时压测撞；锁修复后
  直接静态可判。（输出循环 loop 已确认为 DSpark 模型能力问题，不在本群审查范围。）