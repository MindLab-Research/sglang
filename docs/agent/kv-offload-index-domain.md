# KV host-offload 索引域崩溃：retract 路径用 DCP 虚拟 id 索引物理池（2026-09-16 结案）

**事故**：09-16 08:23:32–37Z（本地 16:23），**PD decode 引擎（1022 / 10.0.0.67）8 个 TP rank 同时 device-side assert → 全部 abort → watchdog `kill_process_tree` → 引擎死亡**。RL 侧表现为「3 个 adapter `state=ACTIVE` 但 `engine_count=0`、`http://10.0.0.67:30200` 探针不可达」（24h 内第二次同症状，但根因不同，见 §5）。
**修复**：`fix/kv-offload-index-domain`（本文件同 PR）：池层 `localize_kv_offload_indices` + `assert_kv_offload_indices_in_range`（`KvOffloadIndexError`）+ `Req.offload_kv_cache/load_kv_cache` 失败降级。

---

## 1. 现场（日志证据）

崩溃栈（8 rank 完全一致，仅 TP 号不同）：

```
scheduler.run_event_loop → dispatch_event_loop → event_loop_overlap_disagg_decode   (decode.py:3549)
 → get_next_disagg_decode_batch_to_run                                              (decode.py:3623)
   → update_running_batch                                                           (scheduler.py:3285)
     → batch.retract_decode                                                         (schedule_batch.py:3054)
       → self.release_req                                                           (schedule_batch.py:3131)
         → release_req                                                              (schedule_batch.py:1801)
           → req.offload_kv_cache                                                    (schedule_batch.py:1629)
             → token_to_kv_pool_allocator.get_cpu_copy                              (allocator/paged.py:312)
               → super().get_cpu_copy                                               (memory_pool.py:4518, DSA)
                 → kv_cpu = self.kv_buffer[layer_id][chunk_indices].to("cpu")       (memory_pool.py:4130)
                   → torch.AcceleratorError: CUDA error: device-side assert triggered
→ Fatal Python error: Aborted（8 rank）→ watchdog: "No live scheduler processes found"
→ kill_process_tree(parent_pid=2614899)
```

- 触发条件：`16:23:30` 最后一条 `Decode batch … #running-req: 61, token usage: 1.00`（**KV 池 100% 满**）→ 1.5 s 后进 retract 路径并崩；此前 16:20–16:22 每分钟 15 条 batch（≈4 s/step，正常）。
- 旁证：`dmesg` **8 张卡同一秒 `Xid 43`**（16:23:01/02，pid 均 python3）= 用户态越界打崩 CUDA context 的伴随信号，**非掉卡/坏卡**。
- 该路径**此前从未成功执行过**：全日志 `offload_kv_cache/load_cpu_copy` 的 32 处命中全部落在本次崩溃栈文本里（8 rank × ~4 处）⇒ 首次真正落地即炸（潜伏型）。

## 2. 根因

```python
# schedule_batch.py::Req.offload_kv_cache
token_indices = req_to_token_pool.req_to_token[self.req_pool_idx, : self.seqlen - 1]
self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(token_indices, ...)
```

`req_to_token` 在 **DCP**（decode `--dcp-size N`）下存的是**虚拟 id**（分配器给出 `size × dcp` 的空间，页 = `page_size × dcp`），而物理 `kv_buffer` 只有 `size + page_size` 行。KV **写路径**（`_write_mla_kv_buffer`）与读侧 repair 都做了 `virtual → local` + owner-rank 过滤（见 `docs/agent/dcp-virtual-id-domain-fix.md`），**唯独这条 retract→host-offload 新路径直接用裸 id 索引物理缓冲** ⇒ `id ≥ size` 时 gather 越界 → device-side assert（CUDA 里越界索引 = assert，不是 Python 异常，**无法被 try/except 捕获**，直接毒化 context 杀全组）。

映射（与写路径逐字一致）：

```
slot  = (id // (page_size * dcp)) * page_size + (id % page_size)
owner = ((id // page_size) % dcp == dcp_rank) and 0 <= id < size * dcp
        → 非 owner / 非法 id 一律路由到 scratch 行 self.size（写路径同款）
```

## 3. 修复

| 层 | 改动 | 作用 |
|---|---|---|
| `MLATokenToKVPool`（memory_pool.py） | **新增** `localize_kv_offload_indices()`：虚拟→本地 + owner 过滤，非 owner/非法 → scratch 行 | 让 offload 索引**构造上在域内**（且与写路径看到同一槽位，恢复语义正确） |
| 同上 | **新增** `assert_kv_offload_indices_in_range()`：`idx_max >= size+page_size or idx_min < 0` → 记 `OFFLOAD-OOB[...]` 并抛 **`KvOffloadIndexError`**（Python 可捕获） | 把"设备态 assert 杀全组"变成"Python 异常跳过一次 offload" |
| `get_cpu_copy` / `load_cpu_copy`（基类 + DSA 子类） | 索引前先 localize + guard（DSA 的 page-indexed `index_k_with_scale_buffer` 同样用 localize 后的 id） | 覆盖全部分配器包装（paged / swa / token） |
| `Req.offload_kv_cache` | 包 `try/except`：失败 → `OFFLOAD-SKIP` 日志 + `kv_cache_cpu = None` | offload 只是"省重算"的优化，**永不带崩引擎** |
| `Req.load_kv_cache` | `kv_cache_cpu is None` → 记 warning 并返回 | resume 不会因 None 崩 |

## 4. 验收

- 单测 `test/srt/test_kv_offload_indices.py`（7 项）：**7/7 pass**（节点 `v15_patched` python3.12 + 真实 torch/triton）：
  - 裸虚拟 id（含 `size*dcp-1`、`size`、`-1`）经 localize 后**全部落在 `[0, size]`**，guard 通过；
  - 非 owner 页 → scratch 行；DCP 关闭时恒等映射；
  - guard 对 `cap`、`cap+4096`、负数**抛 `KvOffloadIndexError`**（不再是设备态 assert）；
  - 源级 pin：`get_cpu_copy/load_cpu_copy`（含 DSA 子类）必须先 localize；`offload_kv_cache` 必须带 `except Exception` + `kv_cache_cpu = None`。
- 现场回归（本次上线）：decode 以 **`--dcp-size 8`** 重启，观察 `token usage` 打满触发 retract 时日志出现 `OFFLOAD-OOB`（或正常 offload）而**引擎存活**。

## 5. 与 09-15 06:43Z 事故的区别（同症状、不同根因）

| 事故 | 时间 | 日志根因 |
|---|---|---|
| 上次 | 09-15 06:43–06:51Z | 8 rank `pool memory leak detected! [full] total=10871296, available=10870784` → **SIGTERM**（"Draining requests and shutting down"）→ detokenizer exit -15 → SIGQUIT → kill_process_tree。**无 device-side assert** |
| **本次** | 09-16 08:23:32–37Z | **retract → offload_kv_cache 越界** → device-side assert（8 rank）→ abort，**无 SIGTERM** |

⇒ RL 侧观测到的"同签名"（`engine_count=0` + 端点不可达）只说明**进程没了**；必须看引擎日志区分。

## 6. 判据（运维）

```bash
# 引擎死亡 vs 存活
ps -eo pid,cmd | grep -c "sglang::scheduler"          # 8 = 健康
# 本次故障指纹
grep -n "device-side assert triggered" decode_glm53mol.log
grep -n "kill_process_tree" decode_glm53mol.log | tail
dmesg -T | grep "Xid.*43" | tail                      # 8 卡同秒 = 用户态越界伴随信号
# 修复生效后的降级指纹（不应再出现 assert）
grep -nE "OFFLOAD-OOB|OFFLOAD-SKIP" decode_glm53mol.log
```

## 7. 遗留

- `OFFLOAD-SKIP` 之后的请求目前"保留新分配的槽位"继续（不会再崩，但也没有恢复到 host 上的旧 KV）。更彻底的做法是把它送回 PD rebootstrap 队列重算（`hold_rebootstrap`）；本次修复以"引擎不崩 + 索引在域内（正常路径不会走到 SKIP）"为界，rebootstrap 化留作后续。
- 池满治理（`max_running_requests=64` + 长上下文 61 并发时 `token usage` 会打到 1.00）属另一条线，见 `docs/agent/swa-prealloc-budget-mismatch-fix.md`。
