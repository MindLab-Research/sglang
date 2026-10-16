# PD 分离 + Mooncake KV 传输：超时 / 判死 / 退化根因（2026-10-15 核实）

> 本树 = `/home/smith/src/sglang-mol-v41`（分支 `release/dsv41-flash-mol`）。
> 所有结论均以本树代码 `file:line` 核实，勿用其它树（pr17 / 1021 快照）推断。

## 1. 关键超时一览（本树实际值）

| 超时 | 默认值 | 触发状态 | 读数点 | 后果 |
|---|---|---|---|---|
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | **300s** | PREFILL sender 处于 `Bootstrapping` | `common/conn.py:1675-1692`（`_check_bootstrap_timeout`），由 `mooncake/conn.py:2833-2836` 调用 | `record_failure` + `update_status(Failed)` |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | **300s** | DECODE receiver 处于 `WaitingForInput` | `common/conn.py:1982-2006`（`_check_waiting_timeout`），由 `mooncake/conn.py:3038-3041` 调用 | Failed + `invalidate_cached_bootstrap_infos` + 发 abort 通知给 prefill |
| `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT` | **30s** | abort 后等 prefill drain-ack | `decode.py:2770-2777`（`_defer_release`）/ `decode.py:2794-2846`（`resolve_deferred_releases`） | 超时**强行释放**（`decode.py:2836-2841`，"releasing anyway"） |
| Mooncake 引擎内部 transfer timeout | **30s + total_length ns** | `batch_transfer_sync_write` | `mooncake-integration/transfer_engine/transfer_engine_py.cpp:126-131`（默认 30s）、`:654-686`（超时返回 -1） | `ret != 0` → 触发 session 黑名单 |
| `/health_generate` 健康检查 | **20s** | 任意 | `entrypoints/http_server.py:201` + `:749-777` | 503 + `server_status=UnHealthy` |
| heartbeat 检测 prefill 失联 | 5s × 2 次 | DECODE | `common/conn.py:1419-1461` | `_handle_node_failure` 批量置 Failed |

**缺口（重要）**：
- PREFILL sender 只在 `Bootstrapping` 检查超时；进入 `WaitingForInput`/`Transferring` 后**没有任何超时**（`mooncake/conn.py:2820-2840`）。
- DECODE receiver 只在 `WaitingForInput` 检查超时；`Bootstrapping` 期间无超时（`mooncake/conn.py:3031-3043`）。

**两者都没有自动重连/重试**——超时即 `KVPoll.Failed`，请求层终止。

## 2. Deferred KV release：UAF 风险（`decode.py:2794-2846`）

- 被判失败的 room 若不满足 `is_abort_release_safe`（prefill 所有 rank 的 drain-ack 未到齐，`common/conn.py:655-660`），先挂起，deadline = now+30s。
- `resolve_deferred_releases` 每个 scheduler iteration 跑一次（`decode.py:3169`；PP 路径 `scheduler_pp_mixin.py:1910`）。
- 超过 30s 仍未齐 → **照样释放**（`decode.py:2836-2841`）。代码自己在 `:2823` 注明 "A timeout cannot prove that a remote write has stopped" ⇒ **远端 RDMA 写还在飞就回收了 KV page**，页随后分给新请求 ⇒ 远端写入污染新请求的 KV（或写进已重映射的 VA）。

## 3. Mooncake session 判死 / 恢复（`mooncake/conn.py`）

- `failed_sessions` / `session_failures`：`:248-250`。
- **判死条件**：`_transfer_data`（`:756-763`）返回非 0 → `session_failures[sid] += 1`；**阈值已是可配**：`SGLANG_MOONCAKE_SESSION_FAILURE_THRESHOLD`（`conn.py:2312-2315`，默认 **"10"**，代码注释标 R76：把 1 改成 10，避免 POSTDELTA 下的瞬态 RDMA 失败永久拉黑）。注意：本树**没有** `SGLANG_DISAGGREGATION_SESSION_FAILURE_THRESHOLD`（那是另一个名字，勿混淆），也没有 `hidden_events.mark_session_failed_and_sync`。
  - ⚠️ 但 `maybe_send_extra`（`:2337-2343`）与 `send_aux` 失败分支仍**无条件** `failed_sessions.add(...)`（阈值 1），只统计增量、不比较阈值 → 状态/aux 传输的一次失败即拉黑。
- **拉黑后的短路**：后续任何 chunk 命中 `failed_sessions` → 直接 `conclude_failure("Decode instance could be dead, remote mooncake session ... is not alive")`（`:2159-2168`），**不再尝试发送**。
- **恢复途径**：
  1. 探针 `_run_one_probe_pass`（`:2698-2723`，`engine.send_probe`），但探针线程由 `SGLANG_ENABLE_FAILED_SESSION_PROBE` 控制，**默认 False**（`environ.py:800`），线程默认不启动（`:304-318`）。
  2. decode 重新注册 KVArgs 时清空（`:2542-2546`，`room == "None"` 分支）。
- ⇒ **一次瞬态失败即永久拉黑该 decode session**，直到新注册到来。这是"先正常后突然传输全失败"的机制。

## 4. detokenizer 为什么会"挂"

detokenizer 进程本身不参与 KV；它只在收到 scheduler 的 batch 输出后才转发（`managers/detokenizer_manager.py:179-187`，`sock_recv` 包在 `soft_watchdog.disable()` 里）。

`/health_generate`（`http_server.py:749-777`）判定健康的条件是 **20s 内 `tokenizer_manager.last_receive_tstamp` 前进**（该时间戳在 `managers/tokenizer_manager.py:2344` 收到 detokenizer 任何消息时更新）。
⇒ **scheduler 主循环一停，detokenizer 就没活干，tokenizer_manager 收不到东西 ⇒ 20s 后 503 UnHealthy**。所谓"detokenizer 挂"实为 **scheduler 主循环被卡**。

scheduler 主循环被卡的三个具体机制：
1. **gloo CPU all_reduce 同步屏障**：`poll_and_all_reduce`（`disaggregation/utils.py:191-213`）在 decode `pop_transferred`（`decode.py:2577-2588`）与 prefill `process_disagg_prefill_inflight_queue`（`prefill.py:1017-1021`）里**主线程每轮**执行 `dist.all_reduce`（gloo）。任一 TP rank 慢（传输线程抢 GIL / 卡在 CPU）→ 全体 rank 阻塞 → GPU forward 停。
2. **`resolve_deferred_releases` 的 all_reduce**（`decode.py:2808-2812`，host_receive 开时）。
3. **GIL 争用**：`_send_kvcache_generic` 在 Python 层构建 `transfer_blocks`（`mooncake/conn.py:877-909` 的逐层/逐 index 双重循环、`zip`+`int()` 装箱），在 `process_layers` 路径**整段持 GIL**；加之 pybind 同步路径 `batchTransferSync` 的 `while(!completed)` **无 sleep 自旋**（`transfer_engine_py.cpp:654-686`，对比另一路径 `transfer_engine.cpp:990` 有 `sleep_for(kPollInterval=10µs)`）⇒ 大 KV 量时烧 CPU + 抢 GIL，拖慢主线程。

## 5. 是否有独立 CUDA stream 做 RDMA？没有。

- 走的是 **GPUDirect RDMA**：注册 GPU 显存为 dmabuf（`mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:445-469`，`cuMemGetHandleForAddressRange`），NIC 直接读写显存。
- sglang 侧调用的是**同步** `engine.batch_transfer_sync_write`（`distributed/device_communicators/mooncake_transfer_engine.py:246-275`），跑在 Python 传输 worker 线程里；pybind 层只 `gil_scoped_release`（`transfer_engine_py.cpp:579`）做 CPU 阻塞等待，**不涉及 CUDA stream**。SGLang 全树未使用 `batch_transfer_*_on_cuda(stream_ptr=...)`（grep 0 命中）。
- 于是既没有 stream 级顺序保证（prefill 用 `_early_send_wait_event` 事件补，见 `prefill.py:1318-1322`），也没有 stream 级限流；所谓"阻塞 forward"是 **CPU/GIL + gloo 屏障**层面的，不是 stream 串行。

## 6. 限流 KV 传输密度的现存旋钮（均为间接、非速率限制）

- `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES`（`environ.py:796`，默认 **0 = 不拆批**）——把一次全层批拆成多批，能削峰但不能限速。
- `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`（默认 `min(max(4, cpu/2/8),12)`，`mooncake/conn.py:257-261`）、`SGLANG_DISAGGREGATION_QUEUE_SIZE`（默认 4，`environ.py:690`）。
- `FastQueue` **无界**（`disaggregation/common/utils.py:72-88`，`deque` + `Condition`）⇒ **没有背压**，队列可无限涨。
- 传输超时可调 `MC_TRANSFER_TIMEOUT`（pybind 层，≥5s，默认 30s）。

## 7. 与 POSTDELTA 的关联（density 假设的机制）

POSTDELTA（`lora/trtllm_lora_temp/moe_overlap.py:962` 起的主分支 + `:1007` 的 BYPASSED 合成，环境变量 `SGLANG_LORA_FUSED_POSTDELTA=1`；REAL FIX 段在 `:946-960`）让 decode forward 更快 ⇒ 单位时间完成/新建请求更多 ⇒ decode 侧 `pop_preallocated`/`send_metadata`/abort 密度上升 ⇒ prefill 传输队列（无界）涨得更快。叠加第 1–3 节的固定超时与一次失败即拉黑，**退化是累积的**：worker 被 30s 超时占死 → 队列堆积 → `send()` 返回非 0 → session 拉黑 → 该 session 上所有 room 立刻 Failed → decode abort → 进 deferred-release → 30s 后强行释放仍在被远端写的页 → KV 污染/进程崩溃。

> ⚠️ **2026-10-16 群组核实补充**：本节 density 假设**未被证实且与用户判断冲突**（用户明确「不可能是传输速度」）。已核实两条影响归因的代码事实：
> ① `_lora_forces_standard_topk()` / `_moe_rejects_bypassed_topk()` 返回 False **并不能让 LoRA 请求变成 BYPASSED dispatch** —— 此时 `output_format=None` 会落进 `topk.py:663-681` 的 auto-detect，而它在 `is_experimental_sgl_trtllm()` 分支读的是**服务级** `get_lora().enable_lora`（launch 带 `--enable-lora`）⇒ 仍判 STANDARD。真正的 BYPASSED 只是 `moe_overlap.py:1007` 手工合成的 `BypassedTopKOutput`，只影响那一次融合核调用。
> ② `get_is_capture_mode()`（`runner_utils/capture_mode.py:44-45`）在 decode 全图 capture 期（`decode_cuda_graph_runner.py:489` 的 `model_capture_mode()`）为真 ⇒ capture 期 delta 被跳过；且全树 **`_process_output_after_replay` 无任何 override**（仅 `decode_cuda_graph_runner.py:1526` 调用点 + `:1567` 基类定义）⇒ 不存在「graph 后 eager 补算 delta」的机制。

## 8. 修复方向（代码级）

1. ~~**判死语义**：把 `mooncake/conn.py:2308` 的硬编码 `>= 1` 改成「连续失败」语义~~ **【已落地，2026-10-15 核实】** 现为 `SGLANG_MOONCAKE_SESSION_FAILURE_THRESHOLD`（默认 10，`conn.py:2312-2315`）。**残留缺口**：`maybe_send_extra`（`:2337-2343`）与 `send_aux` 仍是无条件 `add`，未走阈值；且失败计数**成功不复位**。默认开启 `SGLANG_ENABLE_FAILED_SESSION_PROBE`（`environ.py:800`）仍待办。
2. **deferred release 不得超时强放**：`decode.py:2836-2841` 改为「超时后仍挂起 + 告警 + 增大 timeout」，或先向 prefill 发强制 drain 再释放；绝不在无 ack 时回收 page。
3. **补超时缺口**：prefill sender 的 `WaitingForInput`/`Transferring`、decode receiver 的 `Bootstrapping` 增加超时（`mooncake/conn.py:2820-2843` / `:3031-3043`）。
4. **消除主线程 gloo 屏障的放大效应**：`utils.py:191-198` 的 all_reduce 加超时/降频，或改成非阻塞轮询 + 定期一致性检查。**注意**：R79/R80/R89/R91 的 async 尝试全部失败，根因不是「staleness」本身，而是 `_prev_async_state` 只按 `id(group)` 建键 —— 而 `attn_tp_cpu_group` 被 handshake（`decode.py:1041`，只容忍 {0,1,2}）与 transfer（`decode.py:2583/2591`，可到 4）**共用**，于是 transfer 的结果被喂回 handshake → `raise ValueError` at `decode.py:1077`；叠加位置映射（上一轮 tensor 长度/顺序≠当前队列）双重错位。正确做法：**按 (call-site, rid) 双键 + 永不从未确认结果里合成 Success**；collective 调用序列/形状必须跨 rank 一致（gloo 按程序序配对），故建议固定形状（pad 到 max_requests）或至少保证「每轮每站点恰好发一次」。详见本次 explore 结论。
5. **给 FastQueue 加背压**：`common/utils.py:72-88` 增加 `maxsize` + 满时阻塞/降级，替代无界堆积。
6. **削 GIL**：把 `_send_kvcache_generic` 的 block 构建下推成 numpy 向量化（`set_transfer_blocks` 的 Python 双重循环，`mooncake/conn.py:877-909`），避免大 KV 时长时间持 GIL。
7. **pybind 同步路径自旋**：给 `batchTransferSync` 的轮询加 `sleep_for`，或在 sglang 侧改用 `batch_transfer_async_write` + `get_batch_transfer_status`（`transfer_engine_py.cpp:1285` / `:1330`）。

## 9. 共享内存替代 gloo all_reduce：可行性与 R106 实现缺陷（2026-10-15 核实）

> 背景：`_all_reduce_polls` 的 gloo CPU all_reduce 由主线程每轮执行，POSTDELTA 下 KV 传输变密 → 屏障阻塞 → detokenizer 20s 无输出 → 503。
> HEAD = `1ec8a11d59`（R106）已实现「`multiprocessing.shared_memory` + numpy 状态板」，**经核实该实现有三处致命缺陷，不能保留**。

### 9.1 关键结论：gloo all_reduce 同时是**屏障**，不只是 reduction

`_all_reduce_polls`（`utils.py:194`）返回的 MIN 决定 `pop_transferred`（`decode.py:2616`）/ `_update_handshake_waiters`（`decode.py:1022`）里哪些请求被「提交」。TP 各 rank 的 batch 必须**逐 token 一致**（attention all_reduce 的 shape 依赖 num_tokens），所以「提交决策必须跨 rank 同时发生」。
⇒ 单纯去掉屏障（共享内存只写不 rendezvous）会让快 rank 提交、慢 rank 未提交 ⇒ running batch 跨 rank 分叉 ⇒ NCCL shape mismatch/hang。
⇒ **任何「无 barrier 共享内存板」方案都必须在别处补回等价的同步/一致性，否则不成立**。R79–R101 的 stale 失败根因即此（叠加位置映射错位）。

### 9.2 R106 实测缺陷（已用实验证伪）

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| **A** | `utils.py:211` `gid = id(group)` → `:214` `name=f"sglang_poll_{gid}"` | **`id()` 跨进程不一致**（spawn 子进程实测 4 个 id 全不同）。每个 scheduler 进程创建**各自独立**的 shm 段，`create=True` 永不抛 `FileExistsError` | 板不共享：`board[rank]` 只写自己那行，其余行为 `0xFF` → `board.min(axis=0)` == **本地 polls**，reduce 退化为**恒等函数**。「no rank commits ahead of peers」保证**完全丢失** → 跨 rank 分叉 → hang |
| **B** | `utils.py:235` / `:238` | **位置（列）映射**：`board[rank][i]` = 本 rank 队列第 i 项。各 rank 队列内容/顺序/长度可不一致（R98 已证 shape 会漂） | 跨 rank 同一列可能是**不同请求** → MIN 混合无关请求；且请求出队后同列被新请求复用 → 旧 Success 误归属。R94 已确立必须 **rid-keyed（非 position）**，R106 退回 position |
| **C** | `utils.py:210` `SGLANG_MAX_RUNNING_REQUESTS`（默认 256，**无 launcher 设置**） | 板宽写死 256；`max_running_requests` 由配置决定（DSV4 默认 256，但可被改） | 若配置 ≠256：`len(polls) > max_len` 时 `padded` 反而更长 → `board[rank] = np.array(padded)` 形状不符 → **抛异常**；或静默截断 `result[:len(polls)]` |
| D | `utils.py:216-227` | **初始化竞态**：creator `create=True` 后填 `0xFF`，其他 rank 在填完前 `open` 会读到全 0 = `KVPoll.Failed` | 误判传输失败（注：缺陷 A 使此竞态当前不触发，修 A 后必然浮现） |
| E | 全函数 | 无 `unlink()`；名字 `sglang_poll_<id>` 每次运行都变，且不在 `stale_shm_cleanup.py:31` 的 `_ORPHAN_PREFIXES` 内 | SIGKILL 退出（sglang 常规路径）→ `/dev/shm` 段泄漏累积 |
| F | `decode.py:1041` / `:2583/2591` | handshake 与 transfer **两个站点共用同一 group**（`scheduler.py:1531/1548` 均传 `attn_tp_cpu_group`） | R94 的「跨站点污染」根因未解：两个站点 states 语义不同（handshake 只容忍 {0,1,2}，transfer 可到 4）→ `decode.py:1077` `raise ValueError` |

### 9.3 正确做法（若确要用 shm）

1. **稳定的跨进程名字**：用 `port_args.instance_id`（`server_args.py:668/689`，一实例一 uuid，已传播到每个 scheduler；`data_parallel_controller.py:381/670` 会转发）+ 组的全局 rank 元组做 digest。先例：`load_snapshot.py:295 shm_path_for()`（`/dev/shm/...blake2s`）。
2. **一次性 rendezvous**：组内 rank0 `create=True`，其余 `open`；创建后用 **一次** `dist.barrier(group)` 收敛（一次性，非每轮）→ 消除 9.2-D 竞态。
3. **rid-keyed**：列键用 `bootstrap_room`（uint64，P/D 与各 rank 同值，router 生成）或 (room,poll) 对，绝不用队列位置。
4. **站点隔离**：handshake / transfer / staging 各自独立板（或板内 site 维度），延续 R94 结论。
5. **尺寸取真实 `max_running_requests`**（从 `scheduler`/`server_config` 传入，别读 env）。
6. **同机前置检查**：`in_the_same_node_as(group)`（`parallel_state.py:3332`）为真才启用 shm，否则回退 gloo。1101 单机 8 卡 TP=4 满足。
7. **同步语义仍须保留**：若坚持每轮决策跨 rank 一致，需 **epoch/quorum 协议**——每 rank 写完自己行后递增 `written_epoch[rank]`，读者仅在 `min(written_epoch[*]) >= E` 时以 epoch E 的快照决策（非阻塞轮询，仓库已有同类 `PollBasedBarrier`，`utils/poll_based_barrier.py`）。注意这仍是「等最慢者」，只是等待可移出主线程；**不能既无同步又要求 batch 一致**。
8. 若目标仅是「主线程不被卡 20s」：优先修 §4 根因（GIL/无界队列/pybind 自旋）与降频 `--disaggregation-decode-polling-interval`（默认 1，`arg_groups/fields/disagg.py:108`），而不是拿掉屏障。

### 9.4 备选方案评估

- **NCCL non-blocking all_reduce**：仍是 collective，形状/序配对问题不变；poll 向量在 host、量级微小，还引入 device 侧同步。无收益。
- **CUDA IPC（GPU 共享内存）**：poll 状态是 host 侧标量，搬上 GPU 反而增加 D2H/H2D 同步；且不解决「跨 rank 同时决策」。无收益。

## 10. GIL 争用 / pybind 自旋：逐项核实（2026-10-15）

> 源码位置：本树**不含** mooncake C++ 源码；可用副本 `/tmp/Mooncake-review`（= 部署 .so 的同源，version `0.3.12.post1`，HEAD `75142347`；`diff` 与 `/tmp/mooncake-src` 完全一致）。
> 部署形态：pip wheel，**不是**本树源码构建 —— `docker/Dockerfile:17` `ARG MOONCAKE_VERSION=0.3.13`、`:465` `pip install mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}`；spot venv 实装 `mooncake_transfer_engine-0.3.10.post2`（site-packages/mooncake 目录为空壳，engine.so 实为 `/tmp/mc_src/mooncake/engine.so`，build 路径 `/opt/Mooncake`）。

### 10.1 三个机制的真实归属（重要修正）

| 机制 | 位置 | 持 GIL？ | 后果 |
|---|---|---|---|
| transfer_blocks Python 构建 | `conn.py:877-909`（`set_transfer_blocks` 双重循环）+ `:760-762`（`zip(*blocks)` + 3×`list()`） | **是** | 每 chunk ~O(L×B) 次解释器循环 |
| pybind 实参转换 | `transfer_engine_py.cpp:575-578` 形参 `std::vector<uintptr_t>` | **是**（转换发生在 `gil_scoped_release` **之前**） | 3×N 个 Python int → C++ 逐元素转换，N=L×B |
| `batchTransferSync` 自旋 | `transfer_engine_py.cpp:654-686` | **否** —— 函数首行 `:579 gil_scoped_release release` 已释放 | **烧满一个核**；12 线程 × 多进程 → 超订，抢主线程 CPU |

⇒ 任务描述里「自旋 → GIL 争用」不准确：自旋**不持 GIL**，它烧的是 CPU 时间片。
⇒ GIL 争用来自 **Python 侧构建 + pybind 参数转换**；CPU 烧来自 **自旋**。两者最终都表现为「主线程到 gloo 屏障变慢」。

### 10.2 放大系数（本机 `nproc=128`）

- 默认线程池：`conn.py:261` `min(max(4, 0.5*128//8), 12)` = **8** 个 transfer worker/进程，`conn.py:272` 再切成 `8 // QUEUE_SIZE(4)` 个 executor。
- 每个 worker 在 `_await_transfer_futures`（`:1032`）里并发等 N 个 per-layer future → 每 in-flight chunk 可占 8 个忙等/持锁线程。
- 若同机 8 个 rank 进程 → 最多 ~64 个忙等线程 + 64 个正在跑 Python 构建的线程，全部与 scheduler 主线程抢 CPU/缓存。

### 10.3 关键的 API 事实（决定修复可选路径）

- `getBatchTransferStatus`（`:760-819`）**是阻塞的**（内部 `while (!timeout_table.empty())`，`:782`，**同样无 sleep**）——名字有误导性，**不能**在 Python 侧用它做「带 sleep 的非阻塞轮询」。
- `transferCheckStatus`（`:858-874`）确实非阻塞，但只查 `getTransferStatus(batch_id, 0, ...)` 的**第 0 个 slice**，对多 slice 批不能代表整批完成。
- ⇒ 「改用 `batch_transfer_async_write` + `get_batch_transfer_status` 避开自旋」**不成立**：async 的状态查询自己也在自旋。**必须改 C++**。
- 现无任何 sleep/GIL 旋钮：`getenv` 全表只有 `MC_TRANSFER_TIMEOUT`(`:126-131`)、`MC_LEGACY_RPC_PORT_BINDING`、`MC_USE_TENT`/`MC_USE_TEV1`、`MC_FORCE_TCP`。整个 `transfer_engine_py.cpp` `grep sleep` **零命中**。

### 10.4 先例确认

`transfer_engine.cpp:987-993`（`ScatterTransferOperation::Impl::wait`）：`while(!completed_){poll(); if(!completed_) sleep_for(kPollInterval);}`，`kPollInterval = 10µs`（`:1018`）。**此路径与本问题无关**（scatter API，sglang 未用），但证明引擎内部对忙等有既定 10µs 节奏可照抄。

## 11. 1101 盒：mooncake 源码获取 + 重新编译 engine.so（2026-10-16 核实）

> 本节事实全部在 1101 盒（`ssh 1101box` = `8.222.11.182:1101`，主机名 `dsw-46919-7bdfc699d-jpq5v`）实测。

### 11.1 安装形态（纠正 §10 的旧说法）

- pip 包名 = **`mooncake-transfer-engine`（无 `-cuda13` 后缀）**，版本 **`0.3.10.post2`**，装于 `/usr/local/lib/python3.12/dist-packages/`（系统 python3.12 `dist-packages`）。
  - `pip show` Home-page = `https://github.com/kvcache-ai/Mooncake`；`WHEEL` 的 `Build: 0da9dfea3`、`Tag: cp312-cp312-manylinux_2_35_x86_64`。
  - **PyPI 上确有该版本**：`mooncake_transfer_engine-0.3.10.post2-cp312-cp312-manylinux_2_35_x86_64.whl` ⇒ 盒上就是官方 CUDA 版 wheel（非本地构建）。
  - ⚠️ Dockerfile 里的 `mooncake-transfer-engine-cuda13==0.3.13`（`docker/Dockerfile:465`）**与盒上实装无关**；`pip show mooncake-transfer-engine-cuda13` 为空是正常的。
- sglang 跑在 `/opt/sglang-venv/bin/python`，其 `pyvenv.cfg` 为 `include-system-site-packages = true` ⇒ 直接复用系统 `dist-packages` 的 mooncake。**改 `/usr/local/lib/python3.12/dist-packages/mooncake/engine.so` 对运行中的 sglang 立即生效**（需重启进程重载 .so）。
- `/tmp/Mooncake-review` 在盒上**不存在**（§10 说的是另一份副本）；盒上也没有任何 mooncake 源码目录。

### 11.2 源码 revision = tag `v0.3.10.post2`

- repo：`https://github.com/kvcache-ai/Mooncake.git`，tag **`v0.3.10.post2` = commit `e1d6d6f6f49fbbd77b7ee6e5d0c77349f341b3e3`**（"Bump version to 0.3.10.post2"）。
- 目标文件：`mooncake-integration/transfer_engine/transfer_engine_py.cpp`。
- ⚠️ **行号随版本漂移**：§10 记的 `654-686` 是 `0.3.12`；在 `v0.3.10.post2` 里 `batchTransferSync` 的 `while(!completed)` 循环在 **`:528-556`**（函数 `:458` 起）。

### 11.3 编译工具链现状

| 工具 | 状态 |
|---|---|
| `nvcc` | ✅ `/usr/local/cuda-13.0/bin/nvcc`（CUDA 13.0.88，`/usr/local/cuda` 软链到它），有 stubs `/usr/local/cuda/lib64/stubs/libcuda.so` |
| `gcc/g++` | ✅ 11.4.0 |
| `ninja` | ✅ `/usr/local/bin/ninja` |
| `git` | ✅；盒→GitHub 直连 200（`git ls-remote` 正常） |
| `uv` | ✅ `/usr/local/bin/uv`；pip 走阿里云镜像 |
| **`cmake`** | ❌ **缺失** → `uv pip install cmake` 或 `apt-get install -y cmake` |
| dev 库 | ibverbs/numa/openssl/xxhash/zstd/curl 头在，**缺 glog/gflags/jsoncpp/yaml-cpp/uring 的 `-dev`** |

### 11.4 复现 recipe（完整路径）

```bash
# 0) 工具链
apt-get update && apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git libibverbs-dev libgoogle-glog-dev \
  libjsoncpp-dev libunwind-dev libnuma-dev libpython3.12-dev libssl-dev \
  libyaml-cpp-dev libcurl4-openssl-dev liburing-dev libzstd-dev libxxhash-dev \
  libasio-dev pkg-config patchelf

# 1) 源码（含子模块：pybind11 / yalantinglibs 必须！）
cd /root && git clone --depth 1 --branch v0.3.10.post2 \
  --recurse-submodules https://github.com/kvcache-ai/Mooncake.git mc_src
cd mc_src
# 若浅克隆子模块不全：git submodule update --init --recursive

# 2) 依赖（yalantinglibs 是 extern 子模块，transfer_engine 会链它）
bash dependencies.sh -y      # 官方脚本；会 apt 装齐 + 编译装 yalantinglibs（+Go）

# 3) 改代码（见 11.5），然后编译
mkdir -p build && cd build
cmake -G Ninja .. \
  -DUSE_CUDA=ON -DWITH_STORE=OFF -DWITH_EP=OFF -DUSE_ETCD=OFF \
  -DBUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3.12
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
cmake --build . --target engine -j$(nproc)   # 只编引擎目标；产物 build/mooncake-integration/engine.*.so

# 4) 替换（先备份）
D=/usr/local/lib/python3.12/dist-packages/mooncake
cp $D/engine.so /root/engine.so.bak.$(date +%s)
cp build/mooncake-integration/engine.*.so $D/engine.so
ldd $D/engine.so | grep -i "not found" && echo "缺库！" || echo "依赖完整"
```

> 说明：`libasio.so` 等 `$ORIGIN` 依赖已随原 wheel 装在 `mooncake/` 目录，无需重拷（新 `.so` 的 `INSTALL_RPATH=$ORIGIN`）。**最小化**只编 `engine` 目标可避开 store/etcd/EP-PG（省 Go/Torch/etcd 依赖），但 `transfer_engine` 仍会链 `yalantinglibs`，故 `dependencies.sh` 那步不能省。
> 若坚持完全对齐官方 wheel（含 store.so），按 `docker/mooncake.Dockerfile:56-77` 的完整 flags + `scripts/build_wheel.sh` 走 auditwheel 打包。

### 11.5 具体改动：给 `batchTransferSync` 忙等加 10µs sleep

`transfer_engine_py.cpp`（`v0.3.10.post2`）：先补头文件（现有 `#include <numeric>`/`<fstream>` 在 `:18-19`）：

```cpp
#include <chrono>   // 新增
#include <thread>   // 新增
```

再在 `:528-556` 的循环体**末尾**（超时 `return -1` 分支之后、`}` 之前）插入一行：

```cpp
        while (!completed) {
            Status s = engine_->getBatchTransferStatus(batch_id, status);
            // ... COMPLETED/FAILED/TIMEOUT 分支 ...
            if (current_ts - start_ts > timeout) {
                // ... freeBatchID ... return -1;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));  // 新增：让出 CPU，避免整核忙等
        }
```

置于循环**末尾**（而非开头）可保证只在真正空转的迭代睡眠，完成/失败/超时那次不受影响。语义与既有先例 `transfer_engine.cpp:987-993`（`kPollInterval=10µs`）一致。
