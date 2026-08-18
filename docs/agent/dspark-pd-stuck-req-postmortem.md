# DSPARK PD「个别请求永久卡死」技术报告：死锁原理与结构性修复

> **Postmortem / 设计报告**（2026-08-18）
> 现象：V4 Pro 1P1D（1102 prefill + 1104 decode）低负载下**个别请求永久不返回**，health 200、零 crash、日志无新错误。
> 修复：commit `1c9e1c3275`（结构性修复，替代被否决的超时兜底 `cd15c85d68`）。
> 修复原则（用户红线）：**根本不可能卡——零超时依赖；绝不触碰 prefill collective/poll/conn.py**。
> 简版索引见 `dspark-pd-deadlocks.md` 末节；本文是完整原理与设计文档。

---

## 1. 背景：PD 分离下 DSpark 的 hidden 传输协议

### 1.1 三个参与者与状态机

```
┌─────────────────┐    bootstrap(房间号)    ┌─────────────────┐
│  prefill 端      │ ←────────────────────── │  decode 端       │
│                 │                         │                 │
│ KVConnectionSender │   KV indices(元数据)  │ KVConnectionReceiver │
│  状态机:         │ ←────────────────────── │  状态机:         │
│  WaitingForInput │   "你要把 KV/hidden     │  Bootstrapping   │
│      ↓ (无超时!) │      写到这些位置"       │      ↓ (有超时)  │
│  Transferring    │ ──────────────────────→ │  WaitingForInput │
│      ↓           │   KV chunk + hidden chunk│      ↓          │
│  等 ACK          │ ←────────────────────── │  Transferring   │
│                 │   ACK(注入完成)          │   (CUDA注入)     │
└─────────────────┘                         └─────────────────┘
              decode 端固定大小 pd_hidden_pool（行池）
```

DSpark 投机解码的特殊性：draft 模型需要 target 模型**中间层的 hidden states** 才能
bootstrap。这些 hidden 由 prefill 计算并通过 RDMA 写入 decode 端的一个**固定大小行池**
（`metadata_buffers.pd_hidden_pool`，大小由 `SGLANG_PD_HIDDEN_RECV_POOL_TOKENS` 决定）。
每个在途请求从进入 prealloc 到结束，持有 `min(hidden_len, pool.size)` 行。

### 1.2 正常协议时序（死锁的胚胎在这里）

```
1. decode: add() 创建 receiver → kv_receiver.init()  ← 【bootstrap】
           └→ prefill 收到房间号，创建 sender，进入 WaitingForInput
              （此时 sender 开始等 decode 告诉它"写到哪"）

2. decode: pop_preallocated() 循环：
   a. radix 前缀匹配 → total_prefix_len
   b. hidden_len = origin_input_len - total_prefix_len
   c. dspark_pool.alloc(window_rows)   ← 【分配 hidden 行】
   d. 成功 → send_metadata(KV indices + hidden dst rows) 给 prefill
      └→ sender 离开 WaitingForInput，开始传输

3. prefill: 传 KV chunk + hidden chunk → decode 注入（CUDA event）

4. decode: 每个注入完成 → 发 ACK 给 prefill（prefill 释放发送缓冲）

5. decode: 请求结束 → _release_pd_hidden_rows()：
   wait_ack_completions(room)  ← 等【自己这边的注入事件】全部完成
   → pool.free(rows)           ← 行还回池子
```

**注意依赖方向**：bootstrap(步骤1) 在 alloc(步骤2c) **之前**。
sender 的一生都押在"decode 随后会发 indices"上，而 decode 发 indices 的前提
是先拿到 hidden 行——两个前置条件在时间上被倒置了。这是全部问题的根源。

---

## 2. 死锁三环节

### 环节①：prefill sender 的 WaitingForInput 没有超时

sender 进入 WaitingForInput 后唯一的出路是收到 decode 的 KV indices
（`send_metadata`）。没有任何超时机制。

**为什么不能改 prefill 侧**：prefill 事件循环是一台精密调平的 collective 机器
（8 TP rank 的 gloo 组、iteration barrier、padded all_reduce）。历史上 4 次尝试
（`419f592de9`/`d8b28e71a7`/`373041b738`/`7af2c4f8ff`）全部引入更严重的**全局**
collective 死锁，最终全部回退。结论（AGENTS.md 红线）：prefill collective/队列
超时逻辑永不触碰。**修复只能在 decode 端做。**

### 环节②：decode 端 alloc 失败 → 无限 continue

`pop_preallocated` 原代码：

```python
allocated_hidden_indices = dspark_pool.alloc(pd_hidden_window_rows)
if allocated_hidden_indices is None:
    if prefix_len > 0:
        self.tree_cache.dec_lock_ref(...)   # 释放 radix 锁
    # ... 30s 节流的 warning 日志
    continue    # ← 请求留在队列里，下一 tick 重试
```

请求**已经 bootstrap**（sender 已在 WaitingForInput 等着），但 alloc 失败 →
`send_metadata` 永远不执行 → sender 永远等下去。

如果池子只是**暂时**满（并发高峰，运行中请求的 window 之和超出池容量——这是
**设计内的瞬时状态**），这个 continue 会自然恢复；但如果池子**永久**满
（见环节③的泄漏），这就是永久 spin，且该请求的 sender 永久楔死。

### 环节③：`_release_pd_hidden_rows` 的双重缺陷——泄漏 + 阻塞

原代码：

```python
def _release_pd_hidden_rows(self, decode_req):
    if wait_ack_completions is not None and not wait_ack_completions(room):
        # ↑ 默认 timeout 300s，条件变量等待，跑在【调度线程】上！
        logger.error("Timed out waiting for PD hidden ACK...")
        return        # ← 不释放行！直接泄漏！
    ...
    pool.free(indices_by_pp / indices)
```

**缺陷一（泄漏）**：ACK 超时后 `return`，行永远不还 → 池子容量永久减少一个
window。而 ACK 什么时候永远不会完成？——当 prefill 侧的 sender 已经死了/被
abort 了，宣布过的 chunk 永远不会传完，decode 的注入事件永远凑不齐，
`ack_pending_counts[room]` 永远 > 0。

典型触发：**客户端 abort → router 转发 → prefill AbortReq 中途砍掉 hidden 传输**
（生产日志里的 `KVTransferError(bootstrap_room=...): Aborted by AbortReq` 竞态）。
这是合法竞态（用户取消本来就该终止传输），但旧代码把它变成了 decode 端的永久容量损失。

**缺陷二（阻塞）**：`wait_ack_completions` 虽然是 decode **本地**等待（等自己的
CUDA 注入事件，不是等 prefill——见 §4.2 语义分析），但它用 300s 超时的条件变量
**阻塞**——而这个函数从 `pop_preallocated` 的 FINISH_ABORT 扫描（**调度线程**！）调用。
一次超时 = **整个 decode 引擎冻结 300 秒**；多个 abort 请求串行处理时冻结 300s×N。

---

## 3. 成环：三环节如何咬合成「永久卡死」

```
某请求被客户端 abort → prefill 中途砍掉 hidden 传输（合法竞态）
    ↓
decode 端该请求 FINISH_ABORT → _release_pd_hidden_rows
    ↓
ACK 永远等不到 → 【冻结调度线程 300s】→ 超时 return → 【行泄漏】
    ↓
泄漏累积 → pd_hidden_pool 永久满
    ↓
新请求到达 → bootstrap（sender 进 WaitingForInput，无超时）
    ↓
pop_preallocated → alloc() == None → 无限 continue（环节②）
    ↓
send_metadata 永不发出 → 该请求的 sender 永久卡死（环节①）
    ↓
该请求永久卡死：客户端看到的就是"个别请求永远不返回"
                 health 200、零 crash、日志无新错误
```

### 3.1 为什么是「个别请求」而非全局

- 环节②的 `continue` 会跳到队列的**下一个**请求——window 较小的请求可能还
  塞得进剩余空间，所以先卡的是大 window（长 prompt 低命中）请求；
- 泄漏每发生一次吃掉一块容量，逐步恶化——初期只卡"最胖"的请求，池子彻底满后
  所有新 DSpark 请求全卡；
- 300s 冻结是间歇性的（只在 abort 竞态时发生），其余时间系统完全正常。

### 3.2 为什么检查时看不到痕迹

检查时刻日志里 `hidden pool blocked prealloc` 和 `Timed out waiting for PD hidden
ACK` 计数都是 0——**痕迹早已滚出日志窗口**（日志轮转/时间流逝），只剩卡死的
请求和被吃掉的池容量。这也是该问题难以归因的原因：现场证据只在事发瞬间存在。

### 3.3 与 collective 死锁家族的鉴别

| 维度 | collective 死锁（前五案） | 本案（hidden pool wedge） |
|---|---|---|
| 影响范围 | 全部请求 | 个别请求起，逐步扩大 |
| py-spy 形态 | 8 rank 卡同一/错位 collective | 一切正常，请求在队列里 spin |
| 日志 | health 200 + 请求全超时 | health 200 + 个别请求超时 |
| 根因类型 | collective 序列跨 rank 错位 | 资源生命周期 + 时序倒置 |
| 修复域 | gloo 组/poll 异常安全 | 纯资源管理（decode 本地） |

---

## 4. 被否决的方案：cd15c85d68（超时兜底）

### 4.1 方案内容

1. `_release_pd_hidden_rows` 超时后**仍释放**（修泄漏）——方向正确，精神保留；
2. alloc 失败分支：`enqueue_time` 距今 > 600s（`waiting_timeout`）→ abort(503) +
   `kv_receiver.abort()`。

### 4.2 否决原因（用户红线："我不要超时，我要根本不可能卡"）

- 超时方案下请求仍然会卡 600s 才被杀——这是"卡了之后止损"，不是"不可能卡"；
- 300s 阻塞等待仍在调度线程上，冻结问题没有解决；
- `enqueue_time` 度量的是排队时长而非 alloc 阻塞时长，语义不精确；
- 本质是给死锁装烟雾报警器，而不是消除起火条件。

**保留的正确认知**：`abort()` 优于 `clear()`——`clear()` 只清本地状态，
`abort()` 会向 prefill 发终止通知，让它的 sender 离开 WaitingForInput。
这一认知被结构性修复继承。

---

## 5. 结构性修复：`1c9e1c3275`

**核心思想：反转依赖方向。** 既然 sender 楔死的前提是"bootstrap 了但拿不到行"，
那就让"拿到行"成为 bootstrap 的**前置条件**。行不够 → 根本不 bootstrap →
sender 根本不存在 → 无东西可卡。

全部改动在 decode 端纯本地逻辑（`disaggregation/decode.py` 单文件），
**零超时依赖，零 collective 变更，不触碰 conn.py/prefill.py**。

### 5.1 机制 A：预留先于 bootstrap —— `_try_prealloc_pd_hidden_rows`

挂在两个 bootstrap 入口之前（`add()` fast path 和 `_resolve_pending_reqs`）：

```python
def _try_prealloc_pd_hidden_rows(decode_req) -> bool:
    if decode_req.pd_hidden_reserved_indices is not None:
        return True                      # 已预留（重试幂等）
    if 不是 DSpark / 无 PD_HIDDEN / fake transfer:
        return True                      # 不涉及 hidden，无需预留
    upper_bound = _rebootstrap_prefill_len(req)   # = 全长度（prefix-free!）
    pool = metadata_buffers.pd_hidden_pool
    if pool is None or pool.size <= 0:
        return True                      # 配置错误，让 pop 的显式 abort 去报
    reserved = pool.alloc(min(upper_bound, pool.size))
    if reserved is None:
        return False                     # 池满 → 不 bootstrap，留在 pending
    decode_req.pd_hidden_reserved_indices = reserved
    return True
```

调用侧：

```python
# add() fast path / _resolve_pending_reqs：
if not self._try_prealloc_pd_hidden_rows(decode_req):
    self.pending_reqs.append(decode_req)   # 背压：留在 pending，下 tick 重试
    continue
decode_req.kv_receiver.init(prefill_dp_rank)  # 只有拿到行才 bootstrap
```

**效果**：池满时请求安静地留在 `pending_reqs`（事件循环每 tick 重驱
`_resolve_pending_reqs`），prefill 侧没有任何 sender 被创建。这不是死锁，
是**普通的排队背压**——运行中的请求结束、行还回（机制 B 保证一定还），
pending 请求自动续上。背压与死锁的本质区别：背压的解除条件（池子有行）
由系统自身的正常活动保证达成；死锁的解除条件永远不可能达成。

### 5.2 关键简化：预留量不需要精确——prefix-free 上界

精确 window = `min(hidden_len, pool.size)`，其中
`hidden_len = 全长 − 前缀命中`。前缀命中需要 radix 匹配——而 radix 匹配会拿
radix 锁（`inc_lock_ref`），锁的生命周期横跨 bootstrap 边界意味着每一条
abort/失败路径都要补解锁，极易引入 double-dec/泄漏类新 bug
（第一版尝试预匹配即因此放弃）。**放弃精确，改用上界**：

```
radix 匹配永远排除最后一个 token
  （match_tokens = input_ids[:-1]，另有页对齐向下取整）
⇒ total_prefix_len ≤ 全长 − 1 < 全长 = upper_bound
⇒ hidden_len = 全长 − total_prefix_len ≤ upper_bound
⇒ window = min(hidden_len, pool.size)
          ≤ min(upper_bound, pool.size) = len(reserved)
```

**预留恒够用**——pop_preallocated 里做完真实前缀匹配后：

```python
allocated_hidden_indices = decode_req.pd_hidden_reserved_indices[
    :pd_hidden_window_rows
]                                        # 切片，永不为 None
excess = decode_req.pd_hidden_reserved_indices[pd_hidden_window_rows:]
if excess:
    dspark_pool.free(excess)             # 多余的立刻还回，别的请求可以预留
decode_req.pd_hidden_reserved_indices = None
```

多余部分即时归还，避免过度预留人为压紧池子。**全程不碰 radix 锁的生命周期**
——锁的管理代码一个字节都没改，旧代码怎么管锁还怎么管。

### 5.3 机制 B：释放非阻塞化 —— park & drain

**关键语义认知**（读 `hidden_events.py` 得出）：`wait_ack_completions` 等的是
**decode 自己的 CUDA 注入事件**——每个注入起一个 `PDHiddenAckWaiter-{room}` 线程
`event.synchronize()`，完成后入 completion 队列、由 drain 循环递减
`ack_pending_counts[room]` 并发 ACK 给 prefill。**它不是等 prefill**。
正常情况最后一块注入完 µs~ms 级计数归零。所以用 0 超时探测它完全安全：

```python
def _release_pd_hidden_rows(self, decode_req):
    if (有 ACK 跟踪 and 行已宣布给 prefill            # indices_by_pp is not None
        and not wait_ack_completions(room, timeout_s=0.0)):   # 非阻塞探测
        # 未完成 → 不阻塞、不放弃、不泄漏：
        logger.info("PD hidden ACK still pending at release; parking ...")
        self._pending_pd_hidden_releases.append(
            {"room": room, "rid": ..., "indices_by_pp": ...,
             "indices": ..., "reserved_indices": ...})      # park
    else:
        self._do_release_pd_hidden_rows(...)    # 立即释放（正常路径）
    # 所有权字段全部置空 → 函数幂等，重复调用是 no-op
    decode_req.pd_hidden_dst_indices = None
    decode_req.pd_hidden_dst_indices_by_pp = None
    decode_req.pd_hidden_reserved_indices = None
    decode_req.pd_hidden_pp_slices = None
    decode_req.pd_hidden_state.reset()

def drain_pending_pd_hidden_releases(self):     # pop_preallocated 顶部每 tick 调
    if not self._pending_pd_hidden_releases:
        return
    still_pending = []
    for rec in self._pending_pd_hidden_releases:
        if wait_ack_completions(rec["room"], timeout_s=0.0):
            self._do_release_pd_hidden_rows(    # 完成了 → 释放
                rec["room"], rec["indices_by_pp"],
                rec["indices"], rec["reserved_indices"])
        else:
            still_pending.append(rec)           # 还没完成 → 继续 park
    self._pending_pd_hidden_releases = still_pending
```

三个性质同时成立（旧代码最多满足一个）：

| 性质 | 旧代码 | 新代码 |
|---|---|---|
| 调度线程不阻塞 | ❌ 300s 冻结 | ✅ 0 超时探测 |
| 不泄漏 | ❌ 超时 return 丢行 | ✅ 永不放弃，park 到完成为止 |
| 无超时 | ❌ 300s | ✅ 没有任何 deadline |

**为什么 park 不会变成新的"永久滞留"**：park 的记录只会在"该请求自己的注入
事件还在飞"期间存活。事件要么完成（下个 tick drain 释放）——注入事件是
decode 本地 GPU 操作，正常必然完成；要么进程已经因 CUDA 致命错误整个死掉
（任何协议都无从谈起）。事件循环常驻（`polling_interval` 节奏，空闲也转，
`pop_preallocated` 每 tick 被 `event_loop_normal_disagg_decode` 调用），
所以 drain 保证被执行。

**释放的层次**：只有**已宣布给 prefill 的行**（`indices_by_pp is not None`）
才需要等 ACK；纯预留阶段的 abort（prefill 根本不知道这些行存在）直接释放，
无需探测——`_do_release_pd_hidden_rows` 里 `reserved_indices` 分支即此语义。

### 5.4 机制 C：alloc-None → fail-fast（不变量违规处理）

有了机制 A，**到达 alloc 位置的已 bootstrap 请求必然携带 reserved rows**，
切片永不为 None。`else` 分支（fresh alloc）只是防御。万一
`allocated_hidden_indices is None` 仍然发生，说明有路径绕过了预留
（不变量被破坏）——此时 spin 就会重演旧死锁（sender 卡 WaitingForInput），所以：

```python
if allocated_hidden_indices is None:
    logger.error("PD decode hidden pool invariant violated: ...")
    if prefix_len > 0:
        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
    prepare_abort(decode_req.req, message,
                  status_code=HTTPStatus.SERVICE_UNAVAILABLE)
    self.scheduler.output_streamer.stream_output(
        [decode_req.req], decode_req.req.return_logprob)
    if decode_req.kv_receiver is not None:
        # abort()（非 clear()）通知 prefill，让它的 sender 离开 WaitingForInput
        decode_req.kv_receiver.abort()
    decode_req.kv_receiver = None
    self._release_pd_hidden_rows(decode_req)   # 释放残留行（预期没有）
    failed_reqs.append(decode_req)
    indices_to_remove.add(i)
    continue
```

**确定性失败**（立刻 503 + 通知对端），非 spin、非超时。
不变量违规是可枚举的静态条件（配置错误/代码路径遗漏），部署期即可暴露，
不该用运行时超时去兜。

### 5.5 机制 D：failed_reqs 统一释放（堵预留泄漏）

pop 循环里有 ~6 个 config-abort 分支（pool 缺失 / target_layer_ids 空 /
PP slice 布局错 / fixed_pool 不支持 / streaming 不支持 / window≤0），它们发生在
**预留已存在**之后（因为预留提前了）但原本都不释放行——预留提前的代价必须补上。
修法：循环尾部统一释放，幂等性保证已释放过的是 no-op：

```python
self.queue = [
    entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
]

# Config-error aborts inside the loop ... can fire while pre-bootstrap
# reserved hidden rows are still held. Release uniformly here;
# _release_pd_hidden_rows is idempotent (ownership fields are nulled).
for failed_req in failed_reqs:
    self._release_pd_hidden_rows(failed_req)

return preallocated_reqs, failed_reqs
```

### 5.6 所有权生命周期总表

```
reserve（bootstrap 前，pool.alloc(upper)）
   │  pd_hidden_reserved_indices = rows
   ├─ 池满 → 留 pending_reqs，无 sender，下 tick 重试（背压，机制A）
   ▼
bootstrap（kv_receiver.init）
   ▼
pop_preallocated
   ├─ config abort → failed_reqs → 尾部统一释放（机制D）
   ├─ 切片 consume：allocated = reserved[:window]，
   │   excess 立刻 free 回池，reserved 置 None（机制A'）
   ▼
send_metadata → 传输 → 注入 → ACK
   ▼
finish / abort
   ├─ ACK 已齐 → 立即 free（正常路径，µs 级）
   └─ ACK 未齐 → park → 每 tick drain → 完成后 free（机制B，永不泄漏）
```

每条边都有确定的归宿——没有任何路径让行"消失"（泄漏），
也没有任何路径让线程"等死"（阻塞/超时）。

---

## 6. 为什么这从结构上「根本不可能卡」（证明轮廓）

1. **sender 楔死的充要条件**：bootstrap 了但 decode 永不发 KV indices。
2. 由机制 A：**bootstrap ⇒ 行已预留** ⇒ pop 时切片成功（上界不变量，
   §5.2）⇒ metadata 必然发出——或请求走 config-abort 提前失败（那是启动期
   就可见的静态配置错误，非运行时楔死）。
3. 由机制 B：行**必然还回**（park 到注入事件完成为止，永不放弃）
   ⇒ 池子占用 = 活跃请求 + 刚结束请求的 window 之和，单调有界
   ⇒ pending 请求必然等到预留 ⇒ 必然 bootstrap ⇒ 必然前进。
4. 由机制 C：唯一理论漏洞（绕过预留的路径）被 fail-fast 捕获并通知对端，
   不留 spin。
5. 调度线程上**没有任何阻塞等待**（0 超时探测 + park）；**没有任何超时值**
   参与正确性——超时只是性能参数（router 的 3600s 请求超时仍作为业务层
   兜底存在，但正确性不依赖它）。

死锁环的三环节被分别消除：

| 环节 | 消除方式 |
|---|---|
| ① sender 楔死 | 无 sender 可卡（不 bootstrap 就没有 sender） |
| ② alloc 无限 continue | alloc 永不失败（预留切片恒有效） |
| ③ 释放泄漏 + 300s 冻结 | park & drain（永不放弃 + 永不阻塞） |

---

## 7. 残留边界（诚实声明）

1. **config-abort 分支未调 `kv_receiver.abort()`**：这些 abort 发生在 bootstrap
   后，其房间号的 sender 理论上会留在 prefill 等——但这是**改动前就存在的
   行为**（旧代码同样不通知），且属静态配置错误（一旦触发所有请求都触发，
   部署期立刻可见），非本次回归。后续可在这些分支统一补 abort()。
2. **预留上界瞬时压紧池子**：大请求预留 `min(全长, pool.size)` 直到 pop 裁剪，
   同 tick 的后续请求可能因此等一 tick——有界的 ms 级 convoy，非死锁
   （convoy 随池子归还自动消散）。
3. **跨 rank 短暂分歧**：池子是 per-rank 的，park/drain 时差 1 tick 会让各
   rank 的 bootstrap 时序略有差异——安全，因为 prealloc 队列的 poll 本就是
   local-only（无 collective，`pop_preallocated` 内注释明确 "no cross-rank
   all_reduce"），且旧代码的同类分歧是**永久性**的，现在只剩 1 tick。
4. **CUDA 事件永不完**（硬件级故障）：park 记录会一直留着——但那种情况整个
   引擎已经死了（watchdog/驱动会接管），超出任何协议的合同范围。

---

## 8. 验证方案

部署：rsync `python/sglang/srt/disaggregation/decode.py` 到 1102/1104
（decode 是改动生效端，两端同版本最稳）→ 规范重启（清 `__pycache__`）→
case50 压测（含 abort 洪峰 + 空闲后并发）。

**判据 grep**：
- `invariant violated` —— 不应出现（出现 = 有路径绕过了预 alloc，去查该 rid）
- `PD hidden ACK still pending at release; parking` —— 正常极少（仅 abort
  竞态瞬间 park 一次，下 tick drain 掉）
- 旧痕迹 `Timed out waiting for PD hidden ACK` / `hidden pool blocked prealloc`
  —— 已删除，不应再出现

**行为验证**：
1. 人为压小 `SGLANG_PD_HIDDEN_RECV_POOL_TOKENS` 制造池满 → 请求在 pending
   排队（背压，日志可见延迟而非卡死），池子释放后自动恢复；
2. abort 洪峰后 `pd_hidden_pool.available_size()` 恢复满值（无泄漏）；
3. 全程无请求永久挂起、无 300s 级 decode 停顿（Grafana ITL 时序平滑）。

---

## 9. 参考

- 修复 commit：`1c9e1c3275`（decode.py 单文件，+237/−95 行附近）
- 被替代的超时方案：`cd15c85d68`
- ACK 协议实现：`python/sglang/srt/disaggregation/hidden_events.py`
  （`wait_ack_completions` / `drain_ack_completions` / `PDHiddenAckWaiter-{room}`）
- collective 死锁家族（勿动 prefill 侧的原因）：`dspark-pd-deadlocks.md` 前五案
- 运维 runbook：`.xbot/skills/dpsk-pro-ops/SKILL.md` §2
