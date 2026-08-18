# PD Hidden 接收池：窗口化准入设计与形式化证明

- 版本：v1（2026-08-18）
- 改动文件：`python/sglang/srt/disaggregation/decode.py`（记账 + 准入 + TRIM）、`python/sglang/srt/environ.py`（`SGLANG_PD_HIDDEN_RECV_WINDOW`）
- 前置：`1c9e1c3275`（预留先于 bootstrap）。本设计保持其反楔死结构，只改「预留多少」与「何时放行」。
- 事故背景：2026-08-18 14:56 CST，1P1D V4 Pro（1102/1104）单请求 TTFT=606s。根因见 §2（定理 OB）。

---

## 1. 符号与模型

| 符号 | 含义 |
|---|---|
| `P` | 每 rank `pd_hidden_pool` 行数 |
| `U(r)` | 请求 r 的预留上界 = `_rebootstrap_prefill_len(r)`（普通=输入全长；rebootstrap=输入+输出） |
| `H(r)` | pop 时实际隐藏传输长 = `origin_input_len − total_prefix_len`（未命中后缀） |
| `W` | 窗口上限（新 env `SGLANG_PD_HIDDEN_RECV_WINDOW`；0 ⇒ W=P = 旧行为） |
| `w(r)` | 准入需求 `= min(U(r), W)` |
| `w'(r)` | pop 持有 `= min(H(r), W)` |
| `charge(r)` | 记账持有行数（模块级 dict，rid→int） |
| `φ` | 水位（head-of-line 保护额）= W |

请求状态机：`ARRIVE → [CAP-ABORT] → PENDING(park) → ADMITTED(记 charge=w，bootstrap) → POP/TRIM(charge=w') → RUNNING(chunk 流) → COMPLETE/ABORT(charge 清零)`。

**假设**（均有代码出处）：

- A1 chunk 化 + 每 room 单在途 chunk（`hidden_events.inflight_chunks`/`wake_next_room_waiter`）；chunk 注入 ACK 后其行归还全局池（`finish_streaming_chunk → free_chunk_rows → pool.free`）。
- A2 chunk 按位置升序注入（DSpark 顺序重放）。
- A3 sender 只写 decode 在 `chunk_ready(dst_indices)` 中给出的行（`mooncake/conn.py: notify_pd_hidden_chunk_ready`；目的行源自 decode metadata 的 `dst_state_indices`）。
- A4 任何 RUNNING/ADMITTED 请求有限时间内到达 COMPLETE/ABORT（生成结束/客户端 abort/prefill 600s bootstrap 超时/3600s watchdog）；ACK drain 有限时间完成或走失败同步。
- A5 `H(r) ≤ U(r)`（prefix 命中只减不增）。
- A6 请求集合跨 rank 复制（控制面广播）。
- A7 pop 时的 prefix 匹配长度跨 rank 一致（`db3e58904` 家族共识修复）。
- A8 ADMIT / TRIM / COMPLETE / ABORT 等准入相关事件跨 rank 同序（pop 就绪经 `_padded_all_reduce_min` 集合通信对齐；终结由相同 batch result 决定）。
- A9 窗口 < H 的续供路径已存在：现行 pop 已允许 `min(H, P) < H`（H>P 场景），行经 chunk ACK 归还池后复用。

---

## 2. 现行缺陷（定理 OB = 606s 事故）

现行准入（`1c9e1c3275`，decode.py `_try_prealloc_pd_hidden_rows`）：

```
w_old(r) = min(U(r), P)；准入条件 |F| ≥ w_old(r)；失败 → park（静默、无限重试、无日志）
```

**定理 OB（长请求必然饿死）**：设 `U(r*) ≥ P` 且 ∀t≥t₀: 池占用 `≥1`，则 r* 永久 PENDING 且不可诊断。

证：`w_old(r*) = P` 需 `|F| = P` 即池完全空闲，与占用 ≥1 矛盾 ⇒ 永不 ADMIT；park 路径无日志无超时。∎

事故对照：`U ≈ 7×10⁵ ≫ P`；当时有 RUNNING 长请求持有行 ⇒ 占用 ≥1 恒真；prefill 侧 `KVPoll.Bootstrapping` 等 600s 由超时撕裂（15:06:44），router latency 606.56s。

**附带发现（OB'）**：现行 ADMIT 的 `pool.alloc` 依赖本地池可用量，而 RUNNING 请求的行按 chunk 在**本地时序**归还池 ⇒ 各 rank 本地可用量可分歧 ⇒ rank A 准入、rank B park ⇒ 部分 bootstrap ⇒ 楔死（prefill 日志 `rank=0` 报超时与此相符）。本设计以记账消除该分歧（§3 D3、定理 R1）。

---

## 3. 设计

| # | 决策 | 理由 |
|---|---|---|
| D1 | 准入需求 `w = min(U, W)`，`W ≪ P` 可配 | A1 下稳态活跃行 ≈ 1 chunk；整段占位无数据 |
| D2 | 记账持有：ADMIT 置 w，TRIM 调整 w'，终结清零；行物理上仍走既有 chunk 循环（ACK 归还全局池后复用，A9） | 不改流式协议；记账 ≥ 实际持有 ⇒ 本地可用 ≥ 记账可用，alloc 必成功（引理 K1） |
| D3 | 准入判定 = `记账可用 ≥ w` 且（队头或 `记账可用 − w ≥ φ`）；**不读本地池** | 消除 OB' 的 rank 分歧（R1） |
| D4 | 水位 `φ = W` + FIFO 队头豁免 | 无饿死（L3）且保利用率 |
| D5 | park 可观测（限频日志 5s/rid：rid/U/w/记账可用/P/已等待秒数）。显式熔断（park-abort）在实现评审中移除：abort 分支会 bootstrap-then-abort 短暂楔住 sender，违背结构性原则；留作 v2 议题（需先解决 pending 内 FINISH_ABORT 清理路径） | OB 的"静默"不可再现 |
| D6 | `W = 0 ⇒ W = P` 逐字回退旧行为 | 回滚安全（B1） |

### 3.1 准入（伪代码，decode.py）

```python
def _try_prealloc_pd_hidden_rows(decode_req) -> bool:
    if decode_req.pd_hidden_reserved_indices is not None: return True   # 幂等
    if not needs_pd_hidden(decode_req): return True                     # 非 DSpark/fake 直通
    U = _rebootstrap_prefill_len(decode_req.req)
    if U <= 0: return True
    pool = metadata_buffers.pd_hidden_pool
    if pool is None or pool.size <= 0: return True                      # 保留现行：pop 端显式 abort
    W = window_cap()                    # env SGLANG_PD_HIDDEN_RECV_WINDOW，0 ⇒ pool.size
    w = min(U, W)
    free = pool.size - charged_rows()    # 记账，不读 pool.available_size()
    is_head = (not self.pending_reqs) or (decode_req is self.pending_reqs[0])
    ok = free >= w and (is_head or free - w >= W)      # 水位 φ = W
    if ok:
        decode_req.pd_hidden_reserved_indices = pool.alloc(w)   # 由 K1 必成功
        charge_set(rid, w); park_state.pop(rid)
        log_admit(rid, w, free, pool.size)                       # 限频
    else:
        log_park_throttled(rid, U, w, free, pool.size, waited_s) # 限频 5s
    return ok
```

### 3.2 pop/TRIM（decode.py pop_preallocated）

```python
pd_hidden_window_rows = min(pd_hidden_len, W)      # 原 min(pd_hidden_len, pool.size)
# 预留裁剪分支（现有 1403-1420）不变，另加：
charge_set(rid, pd_hidden_window_rows)
```

### 3.3 终结清账

两个 `_release_pd_hidden_rows`（PreallocQueue 版 / TransferQueue 版）入口处 `charge_clear(rid)`。调用点均在终结决策的同步路径上（pop 的 FINISH_ABORT 段；RUNNING 终结处理）。行释放可能被 park 给下一 tick（本地 drain），但记账即时清零——方向保守（见 K1 讨论），无 rank 分歧。

---

## 4. 不变量

- I1 行独占：每行每刻恰属 FREE / (r,reserved) / (r,chunk k)（既有）。
- I2' 记账自足：准入后至终结前，`charge(r)` 恒定（w 或 w'），请求一切全局行获取均被 `charge(r)` 覆盖；本地实际持有 ≤ charge(r)。
- I3 前缀交付（既有，A2）。
- I4' 记账守恒：`记账可用 = P − Σ charge(r)`；本地可用 ≥ 记账可用。
- I5 bootstrap 后置（保持 1c9e1c3275：init 仅在 reserved ≠ None 后）。
- I6 判定确定性：准入判定只依赖 (请求序列, U, H, ADMIT/TRIM/COMPLETE/ABORT 事件) 的复制历史。
- I7 需求有界：`w(r) ≤ W ≤ P`。

## 5. 定理

**K1（alloc 必成功/无 rank 分歧的前提）**：记账门通过 ⇒ 所有 rank 的 `pool.alloc(w)` 成功。
证：I4' 本地可用 ≥ 记账可用 ≥ w。∎

**S1（行安全）**：sender 对行 x 的写仅发生在 `own(x)=(r,chunk k)` 且 k 为该 room 唯一在途 chunk 期间；行未经 ACK/失败同步不被再外借。证：写前提是 `chunk_ready(dst∋x)`（A3），HANDOUT guard 为 x ∈ 请求行集；A1 单在途；释放唯一经 ACK 或失败同步。∎

**S2（交付正确）**：COMPLETE 时 `d_r = H`，每位置恰注入一次且升序；`is_last ⇔ d_r = H`。证：归纳 chunk 序列（A2），COMPLETE guard 即 `d_r = H`。∎

**L1（无 sender 楔死，强化 1c9e1c3275）**：bootstrap 发生 ⇒ WaitingForInput 不因行不可得而永久等待。
证：I5 ⇒ bootstrap 时已持 `w ≥ min(H,W) ≥ 1` 行（A5, I7）；流中行需求 ≤ charge(r)，续供复用本请求 ACK 归还的行（A9）且新准入受记账门约束（不会占用记账外行 ⇒ 不挤占续供余量——记账可用不含在流请求已释放的行，新准入最多用记账份额）；等待剩余仅单在途串行（A1），解除有限（A4）。∎

**L2（死锁自由）**：等待图无环。证：唯一阻塞边 `PENDING → 记账可用`；持有者（ADMITTED/RUNNING）不等待记账可用（准入一次原子完成；流中不申请记账外份额）。∎

**L3（无饿死）**：队头 h 于 t₀ 进入 PENDING，则 h 在 ≤ max(存活者剩余生命期)（A4 有界）内被准入。
证：`w(h) ≤ W = φ`（I7）。不变量 V1：任何非队头准入满足准入后记账可用 ≥ φ。若 t₀ 时记账可用 ≥ w(h)，立即准入；否则 ∀ 非队头 c：`free − w(c) < φ` ⇒ 无任何非队头准入 ⇒ 记账可用单调不减（TRIM 调减/终结清零只增），存活者逐一终结（A4）⇒ 记账可用 → P ≥ φ ≥ w(h)；此后 V1 维持 `≥ w(h)` 直至 h 在下一 tick 准入。∎

**C1（TRIM 正确）**：`w' = min(H,W) ≤ w = min(U,W)`（A5），超额恰释放一次（guard `reserved ≠ None` 后置 None）。∎

**C2（无泄漏/双释放）**：charge 与行的转移 guard 检查源态；终结态（COMPLETE/ABORT/CAP-ABORT/bootstrap-fail/熔断）均含清账+释放子句；漏清只致记账偏保守（多 park，安全方向），不会双重释放（幂等置 None）。∎

**R1（rank 不变）**：所有 rank 对同一 rid 序列产生相同 ADMIT/PARK 判定。
证：判定函数输入 = 复制历史（A6/A7/A8）+ 记账（仅随 ADMIT/TRIM/终结事件变化，皆同步）。本地 ACK 时序不进入判定。由 K1，本地 alloc 结果与判定一致。∎

**B1（回滚安全）**：`W = P` 时 `w = min(U,P) = w_old`，`w' = min(H,P)` 与现行一致；水位 `φ = P` 使非队头条件收紧至与队头相同可达集；行循环、bootstrap 后置均不变。∎

## 6. 失败路径

| 路径 | 行为 |
|---|---|
| `pool.size == 0`/缺配置 | 直通（现行）→ pop 端显式 500 abort，不 park |
| `H = 0` | 上游 last-token 规则已阻断；TRIM 全额归还 |
| bootstrap 后传输失败 | 在途行经失败同步释放（既有）+ 清账 |
| 客户端 abort / watchdog | FINISH_ABORT 段清账+释放（pop 段 1070-1080 / TransferQueue release） |
| rebootstrap | U=输入+输出，同算法；retraction 走 retracted_queue 不 park |

## 7. 可观测性与验收

- 日志：`[PDH-ADMIT] rid U w free/P W`（仅窗口模式）/ `[PDH-PARK] rid U w free/P waited_s`（限频 5s/rid）。
- 验收：
  1. 重放 14:56 型长请求（U≈7e5 + 1 RUNNING）：TTFT 回归正常量级，`[PDH-PARK]` 后数十秒内 `[PDH-ADMIT]`；
  2. 24h 混流：`grep -c "invariant violated" == 0`（不回退既有判据）、park_wait 有界；
  3. 8 rank `[PDH-ADMIT]` 序列同序（R1 实证）；
  4. `W=0` 重启行为与今晨完全一致（B1）。
- 发布：默认 `W=0`（旧行为）；1104 先设 `W=8192` 观察高峰窗口再固化。

---

## 8. v1.1 补遗：2026-08-18 16:13 冻结事故与协议约束（重要）

**事故**：W=8192 首个长请求（U=743401）触发双端事件循环冻结（health 200、零 crash、请求永久卡死）。py-spy：prefill 8 rank 卡在 `pop_bootstrapped → _padded_all_reduce_min`（7v1 行号错位）；decode 7 rank 卡 iteration barrier、TP1 卡 nvtx enter。

**根因（日志实锤）**：
```
Exception in thread Thread-4 (transfer_worker):
  conn.py:1890 → RuntimeError: PD_HIDDEN state index length mismatch:
  prefill=16384, dst=8192
```
1. **协议约束（本设计 v1 遗漏）**：hidden streaming 的 sender 按 prefill chunk（chunked-prefill-size=16384 行）为单位发包，`_send_pd_hidden_packet` 要求 `len(src)==len(dst)`。**decode 窗口 W 必须 ≥ prefill 端 chunked-prefill-size**，否则 mismatch。
2. **既有缺陷**：该 raise 发生在 transfer_worker 线程，异常裸杀线程（Python 默认 handler 打印后吞掉）——无失败同步、无 ACK。后果链：源行池（65536 行，行仅 ACK 后释放）被 4×16384 占满 → `source chunk allocation failed` 刷屏 → 双端集体状态级联冻结。

**修复（v1.1）**：
- `conn.py` transfer_worker：`_send_pd_hidden_packet` 异常捕获 → `ret=-1` → 复用既有 `ret != 0` 失败路径（`_mark_session_failed_and_sync` → decode abort → 源行释放）。任何未来窗口/协议不匹配 = 请求级 500 + `[PDH-SEND-FAIL]` 日志，**绝不冻结集群**。
- 部署：**W 必须 ≥ prefill chunked-prefill-size（当前 16384）**。两端 env：`SGLANG_PD_HIDDEN_RECV_WINDOW=16384`。

**v2 议题**：sender 按 decode 窗口自适应切 sub-chunk（解除 W ≥ chunk 约束，允许更小窗口提高并发）。
