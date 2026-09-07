# Decode radix output-insert KV 污染 — 排查与形式化证明（commit 8cd8707855）

**日期**: 2026-09-07
**对象**: b300 GLM-5.3 PD decode（`--disaggregation-decode-enable-radix-cache` + `--dcp-size 4` + EAGLE + HiCache L2 write_back + L3 file，UnifiedRadixCache FULL-only，DCP 虚拟页 256，80RPM agent replay）
**症状**: 大 batch 多请求同时完成（每个 decode batch 插入 prefix+output）时，剩下在途请求乱码 / accept 塌陷；不插 output 后消失。
**修复**: `8cd8707855`（`release_kv_cache` 把 `kv_len_to_handle` 钳到 `len(origin_input_ids)`）。
**本文目的**: 用户观察到了因果关系但没找到根因——本文给出完整的机制链与形式化论证，并回答"为什么不插 output 就好了"。

---

## 0. 结论（TL;DR）

**输出插入本身不是直接污染源——插入的槽位操作是自洽的（dup-free 只碰自己页、锁覆盖在途引用、EAGLE 未提交段被排除）。真正的机制是"压力源放大 + 风暴窗口"：**

1. **死体积生成（Lemma A）**：agent replay 工作负载下 output 链永不复用，但每次完成都把 ~E[|output|]（GLM-5.3 thinking 输出 2K-8K token）作为**无锁可驱逐叶子**塞进树——纯死重。输入侧只有共享前缀（走既有路径，几乎不产生新节点）。
2. **完成窗口的"免费表债务"（Lemma B）**：钳制前 output 页由树持有而非归还 free list；钳制后完成时归还 → 两种情况下 admission 预算相同（`available + evictable`），但实现成本完全不同：钳制前每次分配都要**驱逐→write_back（L2 host alloc + DMA + 阻塞式 writing_check + evict_host 级联 + L3 写）**才能兑现，钳制后是 free-list pop。
3. **完成窗口三重耦合（Lemma C）**：k 个请求同批完成（观察到的"大 batch"）在一瞬间同时：① k·E[|output|] 死体积进树；② k 次 `dec_lock_ref` 让共享前缀路径锁归零（最后持有者完成 → 可驱逐）；③ 分配需求持续 → `evict()` 触发 → LRU 驱逐死叶 → **write_back 风暴淹没 L2 host 池**。
4. **驱逐越过死体积触达共享前缀 → L1→L2 降级风暴（Lemma C/D）**：驱逐需求超过冷死体积后，波及刚解锁的共享前缀节点 → 降级（device slots 释放、节点 evicted+backuped）→ 后续 bootstrapping 请求匹配到 L2 → **needs_restore 风暴**（同日修复 4889268bbb 的实证锚点："needs_restore bursts 152/min 精确对齐 accept 悬崖"）。
5. **restore 风暴打开竞争窗口 → 在途污染（Theorem）**：L2 restore 是全链路唯一"槽位生死 + 异步 DMA + 锁账本"交汇点，其竞争窗口族（W1-W5）任一命中 → dangling/提前释放槽位或污染内容 → 在途请求 target verify 读到污染 KV → 乱码 + accept 塌陷。修复 8cd8707855 不是修窗口，而是**移除压力源**：死体积=0 → 免费表在完成窗口被 +k·|output| 补充（不再需要驱逐）→ 共享前缀稳留 L1（在途锁 + admission match 的 MRU 刷新）→ restore 率塌缩到基线 → 所有窗口暴露量→0。

一句话：**output 插入把"完成"从 free-list 补充事件变成了 evict/write_back/restore 风暴事件，而 restore 路径的风暴速率正是 KV 污染竞争窗口的触发器（同周四个修复 W1-W4 的实证签名与"在途乱码"完全一致）。**

**附带发现（§6）：修复本身有 output 页释放缺口（leak）——`start_p` 未随 `_insert_len` 一起钳制，`[input, committed)` 的整页 output 区域无人释放。建议一行修复。**

---

## 1. 系统模型（受测部署的形式化）

每 DCP rank 一个独立 scheduler 进程，各自持有：

- 树 `T`（UnifiedRadixCache，FULL 组件，页粒度 256 虚拟 token），节点带 `value`（device 槽位）、`host_value`（L2 host 槽位）、`lock_ref`/`host_lock_ref`；
- 设备 KV 池 `P`（分页分配器，`free(tokens)` 按 `unique(tokens // page_size)` **整页**释放）；
- L2 host 池 `H`（write_back 策略）、L3 file store；
- 每请求 `R` 的槽位视图 `RT(R) = req_to_token[rpi][0, kv_committed_len)`。

请求生命周期（decode 侧 radix 开启时）：

```
admission (pop_preallocated):
  match L1/L2 → _match_prefix_and_lock → inc_lock(匹配路径)      [L2 命中→ load_back DMA + 门控]
  _pre_alloc → 页预算；不够 → tree_cache.evict(required−available)  [驱逐入口 ①]
prebuilt (process_prebuilt):
  cache_unfinished_req(input) → 插入+dedup 改写+inc_lock            [新请求 input 段，output 为空]
decode steps:
  每步 EAGLE verify 储备分配 → free_list 消耗                     [分配压力 ②]
finish (release_kv_cache):
  cache_finished_req(kv_len_to_handle) → 插入 → dec_lock(路径)      [修复点：钳制前=prefix+output]
```

**修复前后唯一差异**：`kv_len_to_handle = effective_kv_committed_len`（= input+output）→ `min(committed, len(origin_input_ids))`（= input）。`release_kv_cache` 其余路径不变。

## 2. 不变量（"正确"的定义）

- **I1（槽位活性）**：∀ 在途 R，∀ s ∈ RT(R)[0, C_R)：s ∉ FreeList(P) ∧ 无其他写者，直至 R 完成。
- **I2（锁覆盖）**：RT(R)[0, prefix_R) 引用的树节点槽位有 `lock_ref ≥ 1`（R 自身或 ongoing restore 持有）。
- **I3（restore 有效性）**：load_back DMA 写入的槽位仅在 DMA 完成后可读，且内容 = 节点 host 副本内容。

症状（在途请求乱码 + accept 塌陷）= I1 或 I3 被违反：在途请求读到被提前释放后**复用给他者**的槽位（I1，跨请求 KV 互读）或恢复进被覆盖的 host 内容（I3）。

## 3. 插入操作本身的槽位自洽性（排除"插入直接污染"）

对完成的 F（`cache_finished_req`，keyed I∪O）逐段核对：

| 区段 | 槽位来源 | 树动作 | 释放动作 | 在途引用？ |
|---|---|---|---|---|
| [0, P_F) 共享前缀 | F 的 admission 匹配改写后的**树共享槽** | 走既有节点，无新叶 | dup-free 由 `prev_prefix_len=cache_protected_len` 守卫跳过（unified `_insert_helper` L1326: `dup_start = max(0, prev_prefix_len − total_prefix_len)`） | 有，但 `value_slice` 不含共享区且 dup_start 守卫 |
| [P_F, match) | F 的 PD 传输自有槽 | 走既有节点 | dup-free 释放（F 无人引用） | 无 |
| (match, floor(⌈I∪O⌉/256)·256) | F 的 decode 生成槽 | **新叶（output 段）**，无锁入树 | 无（树持有） | 无 |
| 尾页 + [C, alloc) | F 的 draft/未提交 | — | 尾页 free + `_release_overallocated` | 无 |

其余核对（详见附录 A）：锁节点不可驱逐（`_is_device_leaf` 查 `lock_ref`；`acquire_component_lock` 从 `evictable_device_leaves` 摘除；`drive_eviction` 出堆后再查成员资格）；split 的锁/value 切分守恒；`_unevict_node_on_insert` 与 restore commit 因单线程调度不交错。**∴ 输出插入不直接违反 I1/I3——污染必经中间机制。**

## 4. 形式化证明：压力源 → 风暴 → 窗口 → 污染

### Lemma A（死体积生成与放大系数）

设工作负载 W（agent replay，harness 重写最后一条消息）：∀ 完成请求 R 的 output 链 O_R，∀ 未来请求 R'：`¬prefix_match(O_R, input(R'))`（output 永不复用为前缀）。

基数插入（radix）只对**未被既有树匹配的后缀**建新叶：`Δmass(R) = |unique_tail(I_R)| + |O_R|`。共享 system prompt 下 `E[|unique_tail|] ≤ 256`（一个 DCP 页），GLM-5.3 thinking 输出 `E[|O|] ∈ [2K, 8K]`。

**∴ 放大系数 κ = E[|O|]/E[|unique_tail|] ≥ 10–30×**；且新叶**无锁**（完成请求不 `inc_lock`）→ 立即进 `evictable_device_leaves`（`_add_new_node` → `_update_evictable_leaf_sets`）。这些叶的 `last_access_time` 之后永不被刷新（W 下无 match 走到它们）→ LRU 优先驱逐对象，但驱逐成本是 write_back 而非直接释放。

### Lemma B（完成窗口的免费表债务）

分页器整页释放。钳制前：F 完成时 output 页区间 `(P(I), P(C))`（P(x)=⌊x/256⌋）转入树持有（新叶 value），**free list 净增 = 尾页 + draft 段 ≈ 0**；钳制后：同区间整页归还，**free list 净增 ≈ |O_R|**。

admission 预算 `B(t) = |FreeList(t)| + evictable(T_t)`（decode.py `_allocatable_token_budgets`：`available_size += tree_cache.evictable_size()`）在两种情况下不变——**但兑现成本从 O(1) pop 变为驱逐链**：

```
evict → drive_eviction → _evict_device_leaf → write_backup(write_back=True)
    → cache_controller.write (host alloc; host 不足 → evict_host 级联)
    → DMA D→H → writing_check(write_back=True) 阻塞 drain
    → _evict_to_host → evict_component(DEVICE) → 设备槽位释放
```

### Lemma C（完成窗口三重耦合——"大 batch"是必要条件）

k 个请求同批完成在一个 scheduler 迭代内同时发生：

(a) **k·E[|O|] 死体积入树**（evictable 跳升）+ k 次共享路径 MRU 刷新（`_insert_helper` 的 `_touch_node`）；
(b) **k 次 `dec_lock_ref`**：若完成队列是共享前缀路径的最后持有者 → 共享前缀节点 `lock_ref→0` → 同刻进入 `evictable_device_leaves`；
(c) 分配需求持续（在途请求 EAGLE verify 储备 + 80RPM 新 admission）→ 下一次 `evict(required − available)` 或 `load_back` 内 pre-evict（L1989）触发。

LRU 驱逐序：先冷死叶（永不刷新）→ **死体积耗尽后波及 (b) 刚解锁的共享前缀** → L1→L2 降级（write_back + 设备槽释放 + 节点 evicted/backuped）。

（与用户观察一致：*"garble/accept dips preceded by a decode batch with many running reqs completing"* —— commit message 的原始观察。）

### Lemma D（restore 风暴）

被降级的共享前缀被后续 bootstrapping 请求匹配 → `needs_local_restore`（`l2_host_hit_length > 0`）→ load_back DMA 状态机（门控 receiver、per-tick rematch、host/device 锁、`ongoing_load_back` 登记）。

**R(t) ∝ 降级率 ∝ 驱逐需求 ∝ 死体积累积率 = λ·E[|O|]**（λ=完成率）。实证锚点：4889268bbb（同日 08:47 修复）—— *"needs_restore bursts (152/min) matched the accept cliff exactly"*（08:09 突发对齐 accept 从 0.26 → 0.00-0.09 的悬崖）。

### Lemma E（restore 路径的竞争窗口族——I1/I3 的违反点）

每个 restore 执行：`inc_host_lock` → `evict(可能)` → host alloc → DMA 入队 → **树 commit（value=新槽位）** → `ongoing_load_back` 登记 → per-tick 门控 → ack → 双 dec 锁。窗口（均为本部署同周实证修复）：

- **W1**（4889268bbb，09-05 08:47 修）：共享节点的并发 restore——第二个 commit 覆盖 `ongoing_load_back[node.id]`（丢第一个请求的锁参数）+ 覆盖树 value → **dangling KV slots → target verify 读污染槽位 → accept 0.26→0.00-0.09 + SSE 断流**。修复=dedup，但风暴率未除。
- **W2**（c23f23fa03，09-04 修）：abort 路径漏放 host 锁 → 陈旧锁定 host 槽位被 force-free → **在途 DMA 读已释放/复用的 host 数据 → 设备 KV 内容污染 → 进行性输出损坏**（同 upstream #33397/#28771 签名）。
- **W3**（58307d33e9，09-03 修）：DCP backup/load 轮换错位 → phantom host 读 → 跨请求 KV 污染。
- **W4**（7da1b4e9b8 等，09-04 修）：read-before-ready（DMA 未启动/未门控 → forward 读未完成槽位）。
- **W5**（残余，未单修）：per-tick rematch 边界漂移（HC-RESTORE-L1-DRIFT）、`_unevict_node_on_insert` 与 restore commit 的 value 交接（顺序无 data race 但有槽位泄漏/账本双记）、restore-range 坐标系错配（HC-RESTORE-RANGE 诊断）。窄窗口，触发概率 ∝ R(t)。

### Theorem（污染）

`P(污染 | [t, t+Δt]) = 1 − Π_i (1 − p_i · R_i(t) · Δt)`，R_i(t) ∝ R(t)（风暴率）。

- 钳制前：R(t) ∝ λ·E[|O|]（Lemma D），k-batch 完成窗口把驱逐需求与解锁时机叠加（Lemma C）→ R 进入突发域（152/min 实证）→ Π 在分钟级收敛到污染（实证：accept 悬崖 + 在途乱码）。
- 钳制后：死体积项=0（Lemma A 无源）→ 完成窗口 free list 被 +k·|O| 补充（Lemma B 反向）→ 驱逐需求消失 → 共享前缀稳留 L1（在途锁 + admission match 的 `_touch_node` MRU 刷新）→ R(t)→容量基线（无降级→无 restore）→ Π→~0。
- **这解释了"同日 W1-W4 已修但 8cd8707855（11:28）仍必要"：W1-W4 关的是具体窗口，风暴（触发率）仍在；钳制关的是源（速率），与 PFX-SYNC（3812e4febb"源头消灭"）同一哲学。** ∎

### 推论（accept 塌陷的具体形态）

污染槽位对在途请求的影响路径：EAGLE target verify 在 draft token 上的 logits 由（可能污染的）前缀 KV 计算 → draft/target 分歧 → `accept_len→1.0`（draft-reject）→ accept 0.26→0.00-0.09；若污染位于前缀更深处，target 自身 token 也错 → 乱码。与 4889268bbb 记录的失败形态逐字吻合。

## 5. 排除法（为什么不是其他嫌疑人）

1. **插入的 dup-free**：只释放 `[prev_prefix_len, match)`——F 自有的 PD 传输槽（§3 表）。restore 恢复区由 `_commit_hicache_local_restore_to_req` 显式置 `cache_protected_len = restore_end` 守卫（代码注释原文：*"prevents cache_finished_req's insert from freeing HiCache indices as duplicates"*）。
2. **锁覆盖**：`acquire_component_lock` 从 `evictable_device_leaves` 摘除 + `drive_eviction` 出堆重查成员资格 + `_evict_device_leaf` assert `_is_device_leaf`（锁检查）→ 被在途锁住的共享前缀不会被驱逐；被驱逐的是解锁后（完成窗口）的——由 Lemma C 的时序耦合，而非锁缺失 bug。
3. **EAGLE 未提交段**：`kv_len_to_handle = effective_kv_committed_len` 只含已接受 token；draft 段由 `_release_overallocated_kv_indices` 释放。
4. **跨 rank**：每 rank 独立树/池/L2，插入与驱逐 per-rank 自洽；跨 rank 只影响 ACK 分歧（PFX-SYNC 已修），不是本 bug 的机制。
5. **`_unevict_node_on_insert`/restore commit**：单线程调度内顺序执行无 data race；其槽位泄漏是账本问题（见 §7 待办 2）。

## 6. 附带发现：修复本身的 output 页释放缺口（leak）

钳制只改了 `cache_finished_req` 的入参，**`release_kv_cache` 的 overallocated 释放起点未同步钳制**（common.py L183）：

```python
_insert_len = min(effective_kv_committed_len, _input_len)   # 钳制了插入
tree_cache.cache_finished_req(..., kv_len_to_handle=_insert_len)
...
start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len  # ← 未钳制
_release_overallocated_kv_indices(req, start_p, end_p, tree_cache)
```

- `cache_finished_req` 内 `free(kv_indices[page_aligned_len:])` 只覆盖 `[⌊I/256⌋·256, I)`（整页释放 → 页 P(I)）；
- `_release_overallocated_kv_indices` 覆盖 `[⌈C/64⌉, alloc)`（EAGLE draft 储备，页 P(C) 起）；
- **缺口 = 页 `(P(I), P(C))` 的 output 整页区间**——树不持有（key 只到 input）、无人释放 → **每完成请求泄漏 ≈ |O|/256 个虚拟页**（80RPM × 3K 输出 ≈ 240K token/min，7.4M 虚拟池 ≈ 30 分钟耗尽 → retract 风暴/"KV full"）。
- 上游 #22373 的 strip_thinking_cache 模式语义印证：其 `effective_kv_committed_len = min(committed, len(input))` 本身就是钳制值，使 output 落入 overallocated 释放区——**本修复改了入参没改释放起点**，与其镜像的语义不一致。

**修复（`fix/decode-radix-release-leak` 分支，2026-09-07 已实施，四件套——比"一行"多两件，原因见下）**：

1. **`start_p` 钳制（主修复，common.py `release_kv_cache`）**：`start_p, end_p = _insert_len, req.kv.kv_allocated_len`（镜像 #22373 strip 语义）——output 区 `[ceil_pg(I), alloc)` 落入 overallocated 释放路径。
2. **对齐粒度改树页（防双 free，common.py `_release_overallocated_kv_indices`）**：`page_size = getattr(tree_cache, "page_size", None) or global_server_args.page_size`（DCP 下树/分配器页=256 vs flag 页=64）。**为什么必要**：朴素 start_p 钳制后，`ceil_align(start_p, 64)` 可落在 256 页中间 → overallocated 释放与树尾 free（`[floor_pg(I), I)` 整页释放 P(I)）**重叠释放 P(I)**。finish 路径有 free_group 包裹（flush 单次 free+`torch.unique` 页去重，安全），但 **retract 路径（`release_req` → is_insert=False）无 free_group** → 立即 free → 同页两次进 free list → 双分配 → KV 污染。对齐到树页后两路径结构化不相交（树分支 floor 页 vs overallocated ceil 页）。
3. **assert 放宽（common.py）**：`assert start_p == end_p` → `assert start_p <= end_p`（非 spec 非 strip 路径）——钳制后 start_p（树声明边界=input 段）< end_p（分配）是合法常态（#22373 strip 模式同语义），只有 start_p > end_p（声明超分配）仍是记账错误。
4. **radix_cache.py disabled 分支配套**：删除 `req.kv.kv_allocated_len = kv_len_to_handle` 钳制——8cd8707855 后该钳制使 disabled 树路径的 overallocated free 恒空 → 同样的 output 泄漏（unified_radix_cache 的 disabled 分支无此行，不受影响）。原双 free 防护（尾页重叠）由件 2 的树页对齐结构性替代。

**验证（三层）**：
- `py_compile` 两文件通过；
- **页覆盖算术仿真**（`/tmp/verify_page_coverage.py`，模型化 paged free 的 `unique(idx//pg)` 语义）：10 场景 + **2,880 组对齐扫描全过**（I%pg × output_len × a_extra × protected 交叉）——FIX 变体在 finish 与 retract 两路径均**无 gap、无双 free**；同参数下 OLD 复现泄漏（3K output→11 页、20K→77 页、1M→93 页）、NAIVE（start_p 钳制但 64 对齐）复现 retract 双 free（P(I) 同页两路径，6/10 场景）——**件 2 的必要性证明**；
- 部署后复测判据：匹配态回归 γ → <2K/min 噪声底（§6.1 同法）、`kv_evictable_tokens` 停止被挤压。

### §6.1 实证验证（2026-09-07 spot 机 b300-2 日志，三层证据）

**① 代码层**：`docker exec sglang-decode grep` 确认跑的镜像带钳制（line 161-167 `[decode-radix fix 2026-09-05]`）且 **line 183 `start_p, end_p = effective_kv_committed_len`（未钳制）**——泄漏路径在生产代码中。

**② 流量画像（metrics + histogram）**：6,552 请求 / 105 min，**96.3% 输出 ≤500 token**（边界页吸收，零泄漏）；>512 长尾 520 请求（含 10-40K monsters ~39 个）承担全部泄漏 → 直方图推算 **~15K token/min**。池：per-rank 2.71M（虚拟域 10.84M），used 3.97M + evictable 6.86M + available 6.9K。

**③ 被动回归实锤（日志 2,681 个 "Decode batch" 样本，01:43-03:27）**：
- 模型 `tok = 40,762×running + 0.332×(pre×total) + γ×t + c`：**γ = +22,052 token/min（block-jackknife SE ~2,062，z≈+10.7）**；
- 非参数匹配组（同 running 桶 × 同 prealloc 桶，早/晚半对照，10 组全部为正）：**+14.2~+26.5K/min，均值 +19.3K/min**；
- 105 min 累积 **+2.31M = 池的 21.3%**（匹配态 used 漂移，排除 running/prealloc 变化）。
- 与直方图推算（15K/min）同量级（残差 ~4-7K/min 归 L2 保护锁累积/pre 指标欠拟合等混杂项，无其他机制可解释匹配态单调漂移——replay 输入平稳、无 session）。

**严重度修正**：此前"80RPM×3K 输出≈30min 耗尽"基于 thinking 长输出假设；实测流量 96% 短输出 → **实测 19K/min → evictable 6.86M ≈ 6 小时耗尽**（慢性：树容量被挤压 → radix 命中率崩 → L2 churn → 最终 OOM/retract）。当前 Up 2h 已耗 21%。若换 thinking 长输出流量（avg>512），泄漏率 ×30+（回到急性）。

**验证判据（部署修复后复测）**：同法回归 γ 应 →~0（<2K/min 噪声底）；监控 `kv_used_tokens` 在匹配态的斜率 / `sglang:generation_tokens_histogram`（>512 人口的 Σ(output−512) = 预期泄漏上界）。数据/脚本：`/tmp/decode_series.tsv`（spot 拉取）+ OLS/匹配组脚本见本节。

## 7. 残余风险与待办

1. **W5 残余窗口**（速率已塌缩但非零）：HC-RESTORE-L1-DRIFT 的边界漂移计数可作健康指标（突发=风暴回归）；建议上监控。
2. `_unevict_node_on_insert` 无 `ongoing_load_back` dedup（load_back 有）：顺序无 race，但 un-evict 后 commit 覆盖会双记 `component_evictable_size_` + 泄漏 F 的槽位——账本级，建议补 dedup/对齐 4889268bbb 的模式。
3. §6 的 leak 修复 + 回归验证（跑 run 长度需 > 池耗尽时间，观察 `[full]` available_size 单调下降或 retract 风暴）。
4. 若需进一步压缩 restore 风暴基线：把死体积的 write_back 阈值与 host 池水位联动（`evict_host` 抢占死叶 host 副本的优先级高于共享前缀副本）——共享前缀 host 副本的 LRU 由 match 刷新（`_touch_node` 更新 `last_access_time`，`drive_host_eviction` 按 `eviction_strategy.get_priority` 出堆），已是 MRU 保护，无需改动。

## 附录 A：核查清单（代码引用）

| 核查项 | 位置 | 结论 |
|---|---|---|
| 插入链与 dup-free 守卫 | unified_radix_cache.py `_insert_helper` L1313-1330、`cache_finished_req` L793-883 | ✓ 自洽 |
| restore 槽位守卫 | decode_hicache_mixin.py `_commit_hicache_local_restore_to_req` L497-502 | ✓ cache_protected_len=restore_end |
| 锁/驱逐互斥 | full_component.py `acquire_component_lock`（discards from evictable set）、unified_radix_cache.py `_is_device_leaf` L1636、`drive_eviction` | ✓ |
| 预算含 evictable | decode.py `_allocatable_token_budgets` L2226-2233、admission evict L2389-2396 | 压力入口 |
| write_back/驱逐链 | unified_radix_cache.py `_evict_device_leaf` L1702-1753、`write_backup` L1775-1850、`evict` L674-713 | L2 风暴机制 |
| restore 状态机 | decode_hicache_mixin.py `_try_hicache_queue_load_back`/`_process_hicache_local_restores`、unified_radix_cache.py `load_back` L1908-2131 | W1-W5 |
| 实证锚点 | 4889268bbb commit message（needs_restore 152/min 对齐 accept 悬崖；COMMIT lock_ref growth A.id=18 21->36 同节点共享） | R(t) 突发域 |
| 分页器整页释放 | paged.py `free` L261-287（`unique(free_index // page_size)`） | §6 leak 依据 |
| #22373 语义镜像 | schedule_batch.py `effective_kv_committed_len` L1116-1121（strip 模式返回 min(committed, input)） | §6 修复建议依据 |

## 附录 B：与同周修复族的关系

| 修复 | 关的窗口 | 本分析中的角色 |
|---|---|---|
| 58307d33e9 (09-03) | W3 DCP 轮换 phantom 读 | 窗口 |
| 7da1b4e9b8/9169c1e184 (09-04) | W4 read-before-ready | 窗口 |
| c23f23fa03 (09-04) | W2 abort 漏放 host 锁 | 窗口 |
| 4889268bbb (09-05 08:47) | W1 同节点并发 restore | 窗口（同日、同症状、needs_restore 风暴实证） |
| **8cd8707855 (09-05 11:28)** | **源（本文）** | **消除 R(t) 的主导项 λ·E[|O|]** |
