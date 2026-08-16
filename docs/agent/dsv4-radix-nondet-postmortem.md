# DSV4 Radix 命中非确定性战争记录（Postmortem）

> 2026-08-15 22:00 → 2026-08-16 09:30，约 12 小时。B300 1P1D 集群，DeepSeek V4 Pro 0813
> PD 分离 + CP=8 + 双端 radix。本文完整记录"乱码/循环"问题的排查历程——包括所有走错的路。
> 走错的路和走对的路一样有价值：它们构成了排除法的外壳。

## TL;DR

**症状**：高并发下输出 token 级循环（`_CTRL_CTRL_...`）、`finish=length` 锁死、
DSPARK accept rate 爬到 0.99；用户 agent（temp=1.0、600K 上下文、高命中）同样中招。

**根因是分层的，不是一个**：

| 层 | 根因 | 修复 |
|---|---|---|
| 崩溃层 | 我自己加的树入口 cap 毒化了 `cache_unfinished` 自 rematch 的 dup 记账 → 整请求双 free（`full_num_used=-139520` 逐位吻合） | cap 回归调用方（`0c0b127cce`） |
| 污染层 | radix 命中请求恢复计算时，从**别的请求写过的状态行**读取：SWA 窗口槽、c4 压缩 carry、c128 压缩链（32K token 记忆）——全都没被初始化/恢复 | 三件套清零 + c128 边界字典快照-恢复（`0a53594fb2`） |
| 静默失败层 | `.get()` 打在 list 上（AttributeError 被 except 吞）、unified-only gate 在实际 SPLIT 池上恒 no-op、字节平面池按 token 槽索引越界 | 逐一修复 |
| 残余层 | **kernel 级数值非确定**（FORCE_MISS 实证：同输入全量重算×4 仍 4 种输出） | 超出本层可修范围，需上游确定性 kernel |

**最终验证**（真实生产 prompt × 600rpm × 冷/热/热三轮）：

- 短循环（KV 污染特征）：**0/150**
- 长循环（模型能力特征）：2/150，集中在同一个内禀退化 prompt
- 崩溃/泄漏：0

---

## 1. 症状与初始误判

### 1.1 现场症状

```
DSP-ACCEPT out_tokens=[[5272,12975,65]×6, [65,5745,79]×6, [223,389]×6 ...]
decode 日志: accept rate 0.99, accept len 5.95（6 token 全接受的锁死）
最终输出: "abort_abort_abort..." / "_CTRL_CTRL_CTRL..."，finish=length
用户 agent: temp=1.0、多轮 600K 上下文、95%+ 命中 → 偶发循环
```

### 1.2 第一个错误结论（被打回，正确地）

最初用 replay 乱码行 + temp=0 复现，得出"greedy × 乱码 prompt 的模型行为"结论，
并往 mint-bench 加了 `replay_temperature` override。**用户当场否决**：他亲眼见到
temp=1.0 的正常 agent 请求循环。回看数据，当时就有一个被我忽略的硬证据：

> 4 个完全相同的请求（同 prompt、greedy、cached 完全一致）→ 3 循环 + 1 正常。
> **greedy 同输入同状态不可能出两种结果——这只能是被读到的外部状态不同。**

教训：当"模型行为"理论与"同输入不同输出"的观测冲突时，永远是状态污染。
hack 回滚（mint-bench `69ab5c3`、sglang `90b7b1f225`），重新开始。

## 2. 排查时间线（含全部失败路线）

### Phase 0：部署卫生（0.5h，排除一个环境变量）

用户提示"之前 GLM 同样症状是删 pycache+config 解决的"。全量执行：md5 三方对齐
（真抓到 1022 上 hook 文件旧版）、L20D config 全盘扫描（无 MoE L20D，只有无害的
LoRA csgmv）、HiCache L3 清空、deep_gemm/triton/tilelang JIT 缓存清剿、双端重启。

**结果：循环依旧。** 部署漂移不是根因——但 md5 逐文件核对从此成为每次部署的标准动作
（本轮后续抓出 3 次"以为部署了其实没生效"）。

### Phase 1：组件二分（3h，三连排除）

| 实验 | 配置 | 结果 |
|---|---|---|
| radix 复用嫌疑 | 双端 radix OFF + HiCache OFF | 8 并发仍 9/50 循环 → 排除 |
| prefill CP 嫌疑 | CP OFF（radix 仍 OFF） | 仍循环 → **CP（d300）洗清** |
| 投机解码嫌疑 | 去 `--speculative-algorithm DSPARK` | decode 起不来（PD bootstrap 依赖 spec_metadata）→ 无法测，死路 |

收获两个运维坑（已记 AGENTS.md）：router `cache_aware` 与 prefill radix OFF 不兼容
（503 卡死，一晚三次）；PD 双端必须一起重启重建 bootstrap。

### Phase 2：单请求最小化（2h，关键转折）

4 并发相同 prompt → 3/4 循环；4 个不同 prompt 并发 → 4/4 循环（含中文语义循环
`是一个错误的输入或者`）。**"并发才触发"假设动摇**。串行同请求 ×10 → 5/10 循环。

**单请求也循环 → 排除 batch 交互，收敛到"每次运行读到的历史状态不同"。**

### Phase 3：分配指纹插桩（1h，实锤"历史依赖"）

`SGLANG_DEBUG_ALLOC=1` 打印每次命中的 `[ALLOC-DIAG] rid/rpi/prefix/loc0/locN`：

```
run2: rpi=5  loc0=36864  → 循环输出A
run3: rpi=6  loc0=72704  → 干净输出
run4: rpi=4  loc0=61952  → 循环输出B
```

三个 run 命中相同（cached=139264/139734 完全一致），但 rpi 和分配槽位不同，输出各异。
**非确定性 = f(rpi 历史, 槽位历史)**。加上全量重算（cached=None）恒干净，
污染锁定在"命中路径读到的、按请求/槽位寻址的状态"。

### Phase 4：状态考古（2h，读码定位读路径）

DSV4（SPLIT 池模式）attention 依赖四族按槽位寻址的状态，全部排查：

| 状态族 | 寻址 | 树是否恢复 | 发现 |
|---|---|---|---|
| c128 压缩链 | `rpi × ring(256)` 行 | ❌ 无恢复代码 | decode bootstrap 有 `clear_c128_req_state`，但 prefill 侧**零清理** |
| c4 压缩 carry | `translate(full→swa)[prefix 末 7 token]` | ❌ | `clear_c128_req_state` 里 `if ratio != 128: continue`——**c4 从未被清** |
| SWA 窗口 KV | full→swa 映射槽 | tombstone 语义 | 命中边界窗读的是上一任槽主的内容 |
| compress state 池与树 | `DEEPSEEK_V4_C4_STATE` 枚举 | **定义了但全仓库零引用** | 树连存取都没接线——设计者写了枚举没写实现 |

### Phase 5：修复迭代（4h，五版三崩溃两静默）

#### 5.1 SWA 窗口钳位（保留至今，正确）

`swa_pd_prefill_reprefill_tail`（`ce98ae3720`）：prefill 树匹配钳到 SWA 窗口外，
命中永不伸入尾窗 → 窗口永远重算+新鲜传输。DCP-PD-INIT 实证 `to_send` 从 ~541
涨到 15K-248K（窗口强制重传）。**正确性修复，无性能损失（窗口本来就 <1% 序列）。**

#### 5.2 树入口钳位（错误，引发崩溃）

把 cap+chunk 对齐下沉到 `match_prefix` 单一漏斗（`4aeb0fce58`）。链路 30 秒后崩：
`pool memory leak detected! full_num_used=-139520`。

**数值破案**：139520 = 8×16384 + 8448 = 节点 2..9 的页数总和。入口 cap 拦截了
`cache_unfinished_req` 内部的**自 rematch**（line 936，无 limit）→
`cache_protected_len` 每块滞后树深一个 chunk → 下一块 insert 的 `dup_start=0`
→ **每块把上一块刚插入的节点整块双 free**。

教训：cap 是服务策略（决定给请求 serve 多长），属于调用方；树的自记账必须见全量。
Revert（`96f6ae5555`）后重新设计 → 最终 `0c0b127cce` 定稿在 `init_next_round_input`。

#### 5.3 状态清零三件套（部分有效）

- `clear_compress_req_state`：全比率（c4+c128）、全族（compress+indexer）清 rpi ring
- `clear_c4_carry_for_prefix`：清命中边界 carry 行。第一版用 HIP 公式（≤7 token），
  后来发现 CUDA C++ planner 按 ring(16)×ratio(4)=64 token 回看 → 扩到全 ring
- `clear_swa_ring_for_req`：清 SWA 窗口槽

效果：4 分叉 → 3 分叉（逐步收敛但未归零）。

#### 5.4 静默失败大扫除（本阶段最大的教训）

排查中发现**三处代码在静默失败**——"部署了"不等于"在执行"：

1. **unified-only gate**：所有 SWA 清理/快照函数 gate 在 `unified_kv_pool` 上，
   而节点实证 `is_unified_kv_triton() = False`——**SPLIT 模式下全部恒 no-op**
   （C4CLEAR 零日志实锤）。理论对象都错了：这不是 unified ring 悬垂，是 SPLIT
   分页池的槽位复用污染。
2. **字节平面寻址**：SPLIT 池 `kv_buffer = [num_pages, bytes_per_page_padded]`
   uint8 字节平面（584B/token，576 对齐 padding），按 token 槽直接 gather →
   越界 → 异步 CUDA assert（栈显示在后续任意 op，极具迷惑性）。
   修成 `_swa_flat_rows()` 字节级索引（`e50c67c47b`）。
3. **`component_data` 是 list**：prefill 侧节点 `component_data` 为 list（decode 侧
   是 dict），`.get(ComponentType.SWA)` 抛 AttributeError 被 `except: pass` 吞掉
   ——**SWA 和 c128 的快照恢复从未执行过**（C128RESTORE-ERR 日志抓到现行）。

教训三连：①hot path 的 `except: pass` 是排查黑洞，至少要限流打日志；
②修复链路必须加执行日志（`C128SNAP/C128RESTORE-OK/MISS/FAIL` 三态）；
③"理论上的调用链"必须用日志实证，本轮靠这个抓出 3 个静默失败。

#### 5.5 c128 快照-恢复（最终正确形态：边界字典）

关键洞察：**c128 压缩链有 32K token 记忆**（ring 256 行 × ratio 128）。命中后只重算
几百 token，永远无法从空 carry 重建正确状态——清零语义对它是错的，必须**恢复真 carry**。

三版迭代：
1. 树节点挂快照（`d30e61488c`）→ 节点诊断实锤 `n_klen=256/p_klen=16128`：
   SWA horizon 分裂把 chunk 节点拆成页级碎片，挂载点不可达
2. parent 回退 → 同样不可达（last_node 语义与预期不符）
3. **边界字典**（`0a53594fb2`，最终形态）：

```python
# insert 侧（cache_unfinished_req，每个 chunk 边界）
self._c128_boundary_snaps[(page_aligned_len, hash(尾128 tokens))] = snapshot(rpi ring)
# 命中侧（schedule_batch）
_key = (pre_len, hash(命中前缀尾128 tokens))
snap = tree_cache._c128_boundary_snaps.get(_key)  # → restore 到本请求 rpi ring
```

完全绕开树节点语义（分裂/last_node/eviction 都不影响），LRU 128 条。
`C128RESTORE-OK` 日志实证执行。

中途还修了 `InsertParams` 无 `req` 字段的 AttributeError 崩溃（`d39186e485`：
dataclass 动态挂载 + getattr 兜底）、撤销了有害的 c128 KV 窗口清零 bisector
（`f41549bef1`：清掉真实长程压缩 KV 不被重算重建 → 上下文塌缩成近窗复制，
循环率反而 2/4→3/4——**bisector 结论是"有害"也要撤**）。

### Phase 6：FORCE_MISS 定音实验（30min，残余定性）

三层清零 + 快照恢复全生效后仍 2-3 分叉。终极对照：

```
SGLANG_RADIX_FORCE_MISS=1（同输入强制全 miss 全量重算）× 4
→ 仍 4 种输出，且全是干净中文回复变体、零循环
```

**残余非确定性在 kernel 层**（GPU atomics 累加顺序等），与缓存完全无关。
这不是新问题——是"这套硬件栈本来就没有 bit 级确定"，只是之前被更响的 KV 污染盖住。

## 3. 最终验证（真实负载）

### 3.1 三轮冷热对比（cases_50，全 temp=0）

| 轮 | 命中率 | 短循环 | 长循环 |
|---|---|---|---|
| va（冷） | 5.8% | 0/42 | 0/42 |
| vb（热） | 99.5% | 0/42 | 1/42（case_50） |
| vc（热） | 99.5% | 0/42 | 1/42（case_50） |

### 3.2 人工逐个读 150 个输出的尾部（重点 finish=length / comp>3000）

- **短循环（KV 特征：跨请求 token 片段、token 级重复）：0/150 绝迹** ✅
- **长循环：2/150，同一个 prompt（case_50，评分细则条款 U1.3/U4.2/U5.3/U7.3
  互相重叠的循环论证）**。判据：循环内容是它自己的语义；冷轮同卡同主题（10000
  token 未结晶）；两热轮重复单元不同（218 vs 114 字符，随机结晶）→ 模型能力问题
- 慢枚举陷阱 1 例（case_37 冷轮：`1e133→1e134→...` 无界枚举，单元不重复），
  热轮正常收尾
- 其余 145 个：Excel 构建脚本/费用表枚举等合法长输出，尾部连贯前进

### 3.3 冷热一致性（temp=0，50 case）

```
cold==hot: 20/50    hot==hot: 20/50    all-identical: 19/50
```

不一致率与"是否命中"无关（冷热和热热一样）——**缓存命中没有引入额外分叉**。
剩余分歧 = kernel 级非确定基线（greedy 下表现为"不同的干净答案"）。

## 4. 生产落地状态

- prefill/decode/router 全 200，0 崩溃 0 泄漏
- **CP=8（interleave）+ 双端 radix 全开**——全部性能特性保留，无任何降级
- accept rate 0.72-0.78 健康；热轮 cache_ratio 0.995
- commit 链 `ce98ae3720 → b3d1790f65`（16 个，含 2 个 revert）

## 5. 工具沉淀（可复用）

| 工具 | 位置 | 用途 |
|---|---|---|
| `/tmp/nondet.py`（dev 47.87.64.67） | 4 连发分叉计数 | **cached 值直接指示对齐状态**（131072=对齐/139264=host 越界） |
| `/tmp/replay_dump.py`（dev） | 复刻 replay 采样 + dump 全文 | 判循环必须 dump 输出，别信正则 |
| `/root/coldhot_cmp.py`（1021） | 三轮 dump 逐 case 一致性 | 冷==热 vs 热==热 分离缓存/内核因子 |
| `/root/longloop_scan.py`（1021） | 长单元（40-500 字符）重复扫描 | 短循环正则（7-39）抓不到长循环——两个都要跑 |
| `SGLANG_DEBUG_ALLOC=1` | [ALLOC-DIAG]/[STATE-CLEAR*]/[SWACLEAR*]/[C128SNAP/C128RESTORE] | 分配指纹 + 修复链三态执行日志 |
| `SGLANG_RADIX_FORCE_MISS=1` | 强制全 miss | 分离"缓存层非确定"与"kernel 层非确定"的定音实验 |

## 6. 经验教训（按伤害排序）

1. **同输入不同输出 = 状态污染，永远**。模型行为论（temp/prompt 质量解释一切）
   是排查的麻醉剂。本用户两次否决（temp=1.0 也循环 / "只有高并发"）都直接命中要害。
2. **`except: pass` 是黑洞**：三个静默失败（list/dict、unified gate、越界）每一个
   都让"已部署的修复"变成空气。修复链必须有执行日志，用日志证明"跑了"，不是"应该跑"。
3. **插桩 > 推理**：ALLOC-DIAG 的 rpi/loc 指纹、节点诊断的 n_klen/p_klen、
   RESTORE-OK/MISS 三态——每个关键转折都来自一针见血的插桩，而不是更深的代码阅读。
4. **bisector 也要撤**：诊断用的清零（c128 KV 窗口）本身有害（清掉不被重算重建的
   长程记忆 → 上下文塌缩）。诊断代码必须显式标注、验证后立即清理。
5. **长记忆链不能清零只能恢复**：c128（32K token 记忆）的教训。清零语义只对
   "重算窗内可自愈"的状态成立（c4：64 token 收敛；SWA：384 token 窗口内重建）。
6. **数值对账可以定罪**：139520 = 8×16384+8448 的逐位吻合，比任何 stack trace 都硬。
7. **诚实记录失败路线**：本记录 60% 篇幅是走错的路。它们是后来者的地图。

## 7. 残余与展望

**Kernel 级非确定**（FORCE_MISS 实证）是本部署做不到 4/4 逐字一致的根本原因。
影响：所有 prompt 的"措辞级微差"；仅当 prompt 自身分布退化（乱码/重复结构/无界枚举）
时放大为循环。**要根除需上游确定性 kernel 工程**（flashinfer/triton 原子操作路径的
确定性模式），属另一个量级的项目。生产建议：正常业务流量无感；agent 侧避免对
退化 prompt 用 temp=0 贪心长生成。

---

*相关文档：`decode-radix-swa.md`（双端 radix 设计）、`dsv4-cp-dspark.md`（CP 共存）、
`AGENTS.md` §7 陷阱速记（本轮新增 6 条）。判据脚本与 dump 在 dev/1021 节点随取。*
