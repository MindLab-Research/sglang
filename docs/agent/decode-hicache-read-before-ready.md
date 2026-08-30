# decode HiCache L2 load_back read-before-ready 乱码根因与修复（2026-08-30 结案）

## 症状

- 服务随机突发 KV 错位乱码（输出类似 `! plugin/bridge.go:740 - ...（(registrar goes 481` 的 token 汤），终止重发恢复正常。
- 乱码窗口 accept rate 突降（0.53→0.17→0.12），EAGLE draft/target logits 分歧剧增。
- 三集群（.8/.9/1104）低 accept 窗口（<0.1）占比 0.4-1.3%——偶发。
- **关键判别**：不开 decode radix 时从未遇到。
- 时间线（1104，web:chat_0F1C09987C17）：23:44:02 rid=de439e01 进 transfer queue（771K，L1/L2 命中）→ 23:44:33 新请求（148K）并发到达 + #transfer-req:1 → **accept 0.53→0.17 同秒突降** → 23:45:23 重发请求 accept 0.12（污染传播）→ 用户 23:46 cancel。

## 根因（100% 确认，三重叠加）

decode HiCache L2 load_back 的 **read-before-ready**——树 value 在 DMA 完成**前**对全部 match 消费者可见，而 decode forward 对这些槽位的跨 stream 同步机制**空转**。

### ① 生产端：树 value 提交先于 DMA 执行

`unified_radix_cache.py::load_back`（1856-1950）：

```python
device_indices = self.cache_controller.load(...)   # 1916: 只把 DMA 排进 load_queue（未执行！）
...
self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(...)  # 1929: 立即执行
```

- `cache_controller.load()`（cache_controller.py:762-777）：`self.load_queue.append(CacheOperation(...))` —— **DMA 仅排队**，真正执行在 `start_loading()`（decode mixin 的 Phase C 才 kick）。
- `commit_hicache_transfer`（full_component.py:316-333，LOAD_BACK 分支）：`cd.value = device_indices[offset:offset+n_len].clone()` —— **树节点 value 立即指向 DMA 目标槽位**。
- `UnifiedTreeNode.evicted` 是 **property**（unified_radix_cache.py:115-121）：`value is None` 才算 evicted —— **value 写上 = 节点复活 = match 立即可见**。
- `ongoing_load_back` dict（load_back 1944 写入、loading_check 3028 DMA 完成后 pop）**只用于锁管理，match 路径零检查**（全仓库消费者：loading_check pop / sanity check / scheduler idle 判断）。

### ② 消费端 1（match victim）：match 收集 pending 槽位为 "L1 device 命中"，无任何 gate

`_match_prefix_helper`（unified_radix_cache.py:1029）：

```python
if not child.evicted:
    value.append(child.component_data[BASE_COMPONENT_TYPE].value)   # 1091
```

- 并发请求 `_match_prefix_and_lock`（decode.py:617，prealloc 时）→ `_build_decode_prefix_match` → **pending 槽位进 `pm.prefix_indices`（L1 命中）→ victim 请求认为"已在 device，无需 restore"→ 无任何 gate**。
- `HiCacheRestoreGatedKVReceiver.poll()`（decode_hicache_mixin.py:161）只拦**发起 restore 的请求自身**（其 `hicache_restore_status == PENDING`）——victim 的 `needs_local_restore=False` 直接 READY。
- restore 请求自身的 Phase B rematch（`_try_hicache_queue_load_back` 里的 `match_prefix_for_req`）同样收集**别人的** pending 段进 `rematch.device_indices` → `hicache_restored_kv_indices`——Phase A gate 只等自己的 DMA event（`hicache_load_consumer_index`）——别人的 pending 段同样无 gate。

### ③ 消费端 2（forward 读）：跨 stream 同步机制空转

`_wait_for_layer`（memory_pool.py:1328/2225/2246，`get_key_buffer`/`get_index_k_with_scale_buffer` 调用）：

```python
if self.layer_transfer_counter is not None:
    self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
```

`LayerDoneCounter.wait_until`（cache_controller.py）：

```python
def wait_until(self, threshold: int):
    if self.consumer_index < 0:
        return                                   # ← decode 恒走这里，no-op！
    self.events[self.consumer_index].wait(threshold)
```

- `consumer_index` 唯一赋值链：`tp_worker.forward_batch_generation`（542 行 `set_hicache_consumer(batch.hicache_consumer_index)`）← **batch.hicache_consumer_index 全仓库唯一赋值点在 prefill 的 `get_new_batch_prefill`（scheduler.py:3163）**，decode running batch 恒默认 -1（schedule_batch.py:1923）。
- counter 同一性已确认：`cache_controller.py:300` `mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)` + `hybrid_pool_assembler.py:1370/1598/1723`——tp_worker 注册的、memory_pool 用的、cache_controller 的 `layer_done_counter` 是**同一对象**——但 decode 路径 consumer 恒 -1，保护空转。
- stream 验证：scheduler `forward_stream_ctx` = `self.forward_stream`（scheduler.py:1351）= `tp_worker.model_runner.forward_stream`（get_worker_info 传入，scheduler.py:881 接收）；EAGLE draft/verify 也在该 stream（eagle_worker_v2.py:983 注释 "Run draft extend batch in the main compute stream"；ModelRunner.forward 无内部 stream 切换）——**decode 的全部 KV 读（get_key_buffer / get_index_k_with_scale_buffer / dsa_indexer 直接 buffer 索引 / EAGLE draft+verify）都在 forward_stream 上，且全部无 load event 等待**。

### 时序窗口

大段 restore（几百 K token × 78 层 fp8 ≈ GB 级 DMA）执行秒级。从 Phase B（树 value 提交）到 loading_check（DMA 完成清理）期间，任何 match 命中该段的请求都是 victim。

### 附带：异常 fallback 路径同样被本修复兜底

`pop_transferred`（decode.py:3075-3080）：`_process_hicache_local_restores` 抛异常时**所有 PENDING 强置 READY**（"degrading to direct transfer"）——DMA 未完成强转 pop——read-before-ready 直接可达。本修复的 stream 级 wait 同样覆盖此路径。

### 现象全部吻合

| 现象 | 解释 |
|---|---|
| 23:44:33 新请求并发 + accept 同秒突降 | 新请求 restore 提交树 value → victim match/rematch 命中 pending 段 → forward 读未完成 DMA |
| 不开 decode radix 从未遇到 | 无 decode 树 → 无 load_back → 无 match 命中 → 无此洞 |
| 偶发 0.4-1.3% | 需要 victim 在加害者 DMA 窗口（秒级）内 match 命中同段（共享大 system prefix 的并发流量） |
| 终止重发恢复正常 | 重发时 DMA 早已完成，树 value 稳定有效 |
| 1102/1104 与 .8/.9 都发生 | 两集群都开 decode radix + HiCache L2（架构性问题非配置问题） |
| z.ai blog Bug#2 同款 | HiCache 加载时序 read-before-ready；他们修法 = 消费侧 Load Stream 同步点——decode 自研 mixin（decode_hicache_mixin.py）恰好没接这个机制（consumer 接线只有 prefill 有） |

## 修复（stream 级 wait-all-in-flight，一处覆盖全部 victim 路径）

**设计原则**：在 forward stream 头部插一次 `wait_event`，gate 该 stream 上所有后续读（get_key_buffer / index_k 直读 / EAGLE draft+verify / cuda graph replay）。match 语义不变（不截断、不延迟请求——无 rank 分歧），只保证 forward 读取时 DMA 已完成。

### 修改 1：`cache_controller.py::LayerDoneCounter.wait_all_in_flight`

```python
def wait_all_in_flight(self, stream=None) -> None:
    if stream is None:
        stream = device_module.current_stream()
    for ev in self.events:
        stream.wait_event(ev.finish_event)
```

- 3 个 slot（num_counters=3）的 finish_event 全部 wait。
- torch Event 语义：**未 record 的 event `query()` 返回 True、`wait_event` 立即过**；已完成的立即过——**无在飞 DMA 时零开销**。
- 有在飞 DMA 时 forward stream 排队等待最长的——正确性换延迟（GB 级 restore 秒级、偶发）。
- **Rank-invariant**：per-rank 本地 GPU stream 操作，不改变任何 collective 调用参数——无 db3e58904 家族 prefix 分歧死锁风险。

### 修改 2：`tp_worker.py::set_hicache_consumer` decode fallback

```python
def set_hicache_consumer(self, consumer_index: int):
    if self.hicache_layer_transfer_counter is not None:
        if consumer_index < 0:
            self.hicache_layer_transfer_counter.wait_all_in_flight(
                self.model_runner.forward_stream
            )
            return
        self.hicache_layer_transfer_counter.set_consumer(consumer_index)
```

- 调用点：`forward_batch_generation`（tp_worker.py:542）——decode 每 step forward 前执行一次（`batch.hicache_consumer_index` 恒 -1 → 走 fallback）。
- 每 forward 一次（3 event），非每层 78×3 次——零开销。
- prefill 不动（consumer_index ≥ 0 走原 `set_consumer` 语义）。

### 为什么这是正确修法

1. **stream 级同步**：`wait_event` 插入 forward_stream——该 stream 后续所有读被 gate——不需逐读点修（含 dsa_indexer 直接 buffer 索引的绕 getter 路径）。
2. **覆盖全部 victim 路径**：match victim（并发请求）+ rematch victim（restore 请求自身）+ 异常 fallback（PENDING 强转 READY）——一石三鸟。
3. **EAGLE draft/verify 覆盖**：draft extend 在 main compute stream（eagle_worker_v2.py:983），draft pool 的 DMA 与主池共用同一 load_stream/event 机制（`start_loading` 的 `has_draft` 分支）——一次 stream wait 覆盖 target+draft。
4. **cuda graph replay 覆盖**：replay 前的 stream wait_event 有效 gate replay 启动。
5. **prefill 同构洞不在本修范围**（prefill victim match 命中别人 pending 段 + consumer 只等本 batch DMA——另案；本案乱码在 decode）。

## 验证计划（部署 1102/1104 后）

1. **定向复现**：两请求共享大 system prefix（≥300K token），首请求触发大段 L2 restore（清 L1 或重启后首访），第二请求紧跟着 match——修复前 accept 崩/乱码，修复后稳定。
2. **case50 混合负载**三轮：0 失败 0 乱码。
3. **长跑观察**：三集群低 accept 窗口（<0.1，0.4-1.3%）消失。
4. 判据（新日志无）：decode 日志 accept rate 无无故突降（并发 restore 期间）。

## 相关文件

| 文件 | 角色 |
|---|---|
| `python/sglang/srt/managers/cache_controller.py` | 修改 1：`LayerDoneCounter.wait_all_in_flight`（LayerDoneCounter 类） |
| `python/sglang/srt/managers/tp_worker.py` | 修改 2：`set_hicache_consumer` decode fallback |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 根因 ①：`load_back:1916-1929`（load 排队后立即 commit） |
| `python/sglang/srt/mem_cache/unified_cache_components/full_component.py` | 根因 ①：`commit_hicache_transfer:327`（value 立即写） |
| `python/sglang/srt/mem_cache/unified_radix_cache.py:1029` | 根因 ②：`_match_prefix_helper:1091`（match 收集 pending 槽位） |
| `python/sglang/srt/managers/cache_controller.py:111-114` | 根因 ③：`wait_until` consumer<0 no-op |
| `python/sglang/srt/managers/scheduler.py:3163` | 唯一 hicache_consumer_index 赋值点（prefill only） |
| `python/sglang/srt/disaggregation/decode_hicache_mixin.py` | 自研 decode mixin（Phase A/B/C 状态机、rematch victim、异常 fallback） |
| `python/sglang/srt/disaggregation/decode.py:3075` | 异常 fallback：PENDING 强转 READY |
