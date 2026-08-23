# DeepSeek V4 Pro DSpark PD 分离推理：技术报告

> **周期**：2026-08-13 → 2026-08-23（10 天连续工程）
> **代码**：sglang `b300-glm52` 分支（基于上游 v0.5.15）
> **集群**：B300 测试对（1102 prefill 8×TP + CP=8 / 1104 decode 8×TP + DCP=4 + DSPARK），
> mooncake RDMA，router PD 分离，公网入口 `8.222.11.182:31000`
> **模型**：`deepseek-v4-pro-0813`（853GB 官方权重，FP4 expert + FP8 attention，61 层，
> DSA 稀疏注意力 + 双层压缩 KV c4/c128 + SWA 滑窗）
> **最终验收**：case50 ×2 @600rpm 全绿（42/50 + 8 例已知 DFlash grammar 限制豁免），
> 内容零污染，单轮 <2.5min，0 崩溃 0 泄漏

---

## 0. 摘要

本工程在 PD（Prefill/Decode 分离）推理架构上完整落地了 **DeepSeek V4 Pro + DSpark
投机解码**，交付的核心资产是一套 **hidden state 流式动态传输算法**及其配套的分布式
正确性体系。DSpark 的 draft 模型消费 target 模型中间层激活，这意味着 PD 分离下
**hidden state 必须作为第四种传输状态**与 KV 一起跨机流动——业界没有现成方案。

| 交付能力 | 核心技术 | 实测结果 |
|---|---|---|
| **Hidden 流式动态传输** | 搭车传输 + 直发回退 + ACK 流控 + 窗口化准入（含形式化证明） | 68 万 token 巨型请求 20s 完成；16 并发 + abort 洪峰 72/72；并发小请求零饿死 |
| **双端 Radix Cache** | `swa_served_from_tree=False` 协议（树只持 FULL，SWA 尾窗重算） | 630K 等同请求 21.8s → **1.7s**（cached=721,920） |
| **Prefill CP=8 共存** | aux hidden CP 重组 + `decode_engine_rank` 对角配对 | 200K 冷前缀 **11K → 38.7K tok/s（3.5×）** |
| **高并发正确性** | 跨请求 KV 污染根因修复 + DCP 虚拟 id 域修复 | 14+ 并发长上下文（4×300K giant + 10 medium）内容 100% 干净 |
| **推理性能** | — | 64 并发 TPOT p50 8.4ms / 峰值 1707 tok/s；DSPARK accept len 2.2-3.75（健康区间） |

排查战役累计定位并根治 **20+ 个深层 bug**（5 类分布式死锁、2 类静默楔死、4 层乱码
叠加、DCP 虚拟 id 域、跨请求 KV 污染等），全部修复走根因路径——无超时兜底、无降级
开关、无性能损耗绕过。

---

## 1. 背景与技术挑战

### 1.1 DSpark 为什么不能直接套用 PD 分离

传统投机解码（EAGLE）的 draft 模型有独立权重，PD 分离下可以让 decode 侧跑一次
draft-prefill 自举。**DSpark 不行**：draft 消费 target 的中间层激活
（`dspark_target_layer_ids=[58,59,60]`），draft 与 target 激活强耦合。PD 分离下这些
hidden 只存在于 prefill 侧，decode 侧的 draft 无从获得。

唯一正确解：**hidden state 本身作为第四种传输状态，从 prefill 流到 decode**——与
KV cache、压缩 KV（c4/c128）、SWA ring 并列。V4 Pro 的四种状态各有不同的生命周期
与共享语义，这是全部工程难点的来源：

| 状态 | 语义 | 生命周期 |
|---|---|---|
| FULL KV | 全局长程注意力 | 可缓存、可跨请求共享 |
| SWA ring（滑窗 128） | 近窗注意力 | per-request 窗口状态，**不可共享** |
| c4 压缩链 | 4:1 压缩 KV 的增量 carry（记忆 ~64 token） | per-request，收敛半径内可清零重建 |
| c128 压缩链 | 128:1 压缩 KV 的增量 carry（**记忆 32K token**） | per-request 长记忆——**只能恢复，不能清零** |
| **DSpark hidden**（新增） | target 中间层激活，draft 自举输入 | per-request，**不可缓存、不可从 KV 反推** |

### 1.2 传输本身的三重难题

1. **时机**：hidden 必须在 prefill 进行中动态流出。若等 prefill 结束整块传，
   decode 侧 draft 无法提前自举，TTFT 退化不可接受。
2. **资源**：decode 侧 hidden 行池固定大小（`pd_hidden_pool`），而请求的 hidden 需求
   上不封顶（agent 流量 68 万 token）。准入策略错一个符号就是永久楔死或互相挤死。
3. **一致性**：8 个 TP rank + 4 组 DCP 的分布式环境里，控制消息（zmq）per-rank 异步、
   chunk ACK 本地时序漂移——任何依赖 per-rank 状态的集合通信都会错位死锁。

---

## 2. 核心创新：Hidden State 流式动态传输算法

### 2.1 协议骨架（三个参与者与状态机）

```
┌─────────────────┐    bootstrap(房间号)    ┌─────────────────┐
│  prefill 端      │ ←────────────────────── │  decode 端       │
│ KVConnectionSender│  KV/hidden 目的地元数据 │ KVConnectionReceiver│
│  WaitingForInput │ ──────────────────────→ │  Bootstrapping  │
│      ↓ (无超时)  │   KV chunk              │      ↓ (有超时)  │
│  Transferring    │   + hidden chunk(搭车)   │  Transferring   │
│      ↓           │ ←────────────────────── │   (CUDA 注入)   │
│  等 ACK          │   ACK(注入完成→行归还)   │                 │
└─────────────────┘                         └─────────────────┘
              decode 端固定大小 pd_hidden_pool（行池）
```

decode bootstrap 时通过元数据告知 prefill：hidden 目的行（`dst_indices`）、流式标志、
窗口行数、target 层号、PP 分片。prefill 按房间号寻址，RDMA 直写 decode 指定行。

### 2.2 动态传输四机制

**（1）搭车传输（piggyback）**——hidden chunk 随同位置的 KV chunk 搭同一传输流，
共享连接调度与游标推进，摊薄每次 RDMA 往返的固定成本。

**（2）直发回退（direct-send fallback）**——KV 传输游标按 chunk 推进，可能跳过
某个尚未生成的 hidden chunk 的 piggyback 窗口。若不处理，该 chunk **永久丢失** →
decode 报 `hidden chunk arrived out of order` → 请求挂死。修复：flush 路径检测被
游标跳过的 chunk，单独直发（`_send_pd_hidden_only_chunk`）。判据 `ooo=0`。

**（3）ACK 式流控与行池回收**——chunk 注入 decode 的 draft 状态（CUDA 内拷贝）完成后
回 ACK，**ACK 到达即行归还全局池**，行池得以在请求进行中滚动复用。释放侧采用
非阻塞 deferred-release（`drain_pending_pd_hidden_releases`）：ACK 未齐的释放挂到
下一 scheduler tick 重试，永不放弃也永不阻塞调度线程——行不会泄漏，调度不死。

**（4）乱序防线**——decode 侧 `expected_start` 逐 chunk 校验位置连续性（DSpark 要求
顺序重放），配合 finalize hold（meta 未到不完成 bootstrap）与 admission gate
（pending hidden 请求不进 batch），把乱序从"静默错数据"变成"响亮报错"。

### 2.3 接收侧窗口化准入（含形式化证明）

行池准入的第一版（上界预留）有一个被形式化证明的必然缺陷：

> **定理 OB（长请求必然饿死）**：预留需求 `w_old = min(U, P)`（U=请求 hidden 需求，
> P=池容量）。当 `U ≥ P` 时需要**整池空闲**才能准入；只要存在任一并发持有者，
> 该请求永久 PENDING——且 park 路径无日志无超时，**不可诊断**。
> 实证：68 万 token 请求 TTFT=606s（600s 是 prefill bootstrap 超时撕裂，不是处理时间）。

修复（commit `1910bb619c`，设计文档含定理 OB/K1/S1/S2/L1-L3/C1/C2/R1/B1 完整证明）：

- **窗口化需求** `w = min(U, W)`（`SGLANG_PD_HIDDEN_RECV_WINDOW`）：长请求只要求
  一个滚动窗口而非整池，行经 chunk ACK 归还后滚动续供；
- **记账式准入**：`charge` 只随 ADMIT/TRIM/终结等**跨 rank 同序的同步事件**变化，
  绝不读本地池可用量——消除 per-rank ACK 时序漂移导致的准入分歧（定理 OB'，
  部分 bootstrap 楔死）；
- **水位 φ=W 队头豁免**：保证队头请求总能拿到窗口，杜绝队头阻塞。

实测：68 万 token 大请求 20s 完成；并发小请求零饿死；`PDH-PARK` 只短暂出现后即
`PDH-ADMIT`。

**工程边界（2026-08-24 更新：窗口模式已彻底修复并通过全量验收）**：窗口硬约束 `W ≥ prefill --chunked-prefill-size`（sender 按
整块 chunk 发包要求 src/dst 等长）。原"窗口 streaming 有跨请求 chunk 乱序未修 bug"结论经四层根因
挖掘被推翻——R1 事故实为：①启动脚本 export 误置于 launch 后（窗口模式从未真正生效）；②控制面
zmq socket 永久黑洞（RECONNECT_IVL=-1 + monitor 只订 DISCONNECTED）；③notify 丢失无自愈；④
park-behind 唤醒竞态。修复后（自愈重连 + 重发/去重/缺口等待协议 + 原子 park），窗口模式通过
ladder×3（冷/热树）20/20、case50×4 轮全绿、**21 请求同时强制中止风暴存活**、523,099 次集体调用
×8 rank 位点序列完美奇偶的验证矩阵。legacy 模式仍为 env 一键可回退项（`SGLANG_PD_HIDDEN_RECV_WINDOW=0`）。

### 2.4 死锁免疫与优雅降级

分布式控制面经历五类死锁，全部根治而非绕过（原则：**根本不可能卡——零超时依赖**）：

| 死锁 | 根因 | 修复 |
|---|---|---|
| 跨轮错位 | disagg poll 与调度共用 gloo 组，异构 collective 被 FIFO 跨轮匹配 | 专用 gloo 组（`pf_poll_cp/tp_gloo_group`，同成员独立 FIFO） |
| 计数漂移 | `poll()` 抛异常在 all_reduce 前逃逸 → per-group 计数永久偏移 1 | 异常安全 poll（try/except → `KVPoll.Failed` + 异常 stash） |
| 空队列分歧 | 空队列分支直接 return 跳过 collective → rank 间计数错位 | 空分支用空 poller 列表参与 collective |
| 漏改残留 | 事件循环 6 处 poll 仍用原组 | 逐一改专用组 + 空队列补偿 |
| 300s 兜底 | 残余未知分歧 | 专用组加超时响亮崩溃（取证用），非依赖项 |

降级路径：接收池耗尽 / KV 池满时，**单请求 503 干净 abort**（hidden 行释放、
receiver.abort、记账清零）替代杀整个 scheduler——引擎存活，其余请求无感。

### 2.5 传输正确性的隐形战线

- **失效钳制复活（`98e7a09e9e`）**：decode radix 命中钳到 `decode_prefix_len` 的
  钳制代码因 merge 丢失 setter 变成死代码 8 天——并发巨型请求复用彼此的 radix 前缀，
  复用段 hidden 永不生成 → 乱序。修复：从部署级信号（`PD_HIDDEN in state_types`）
  复活钳制 + 未 finalize 请求 key_limit=0 + finalize hold + admission gate + flush
  直发回退，五件套缺一不可。判据 `PDH-CLAMP-CHECK` 存活、`ooo=0`。
- **CP=8 对角配对**：aux hidden 在 CP 下 8 个 rank 各发一份 → decode 收 8 份交错
  数据。修复：`decode_engine_rank` 对角配对，恰好一份。

---

## 3. Bug 排查战役：方法论与战果

### 3.1 方法论（按价值排序）

**（1）内容感知金丝雀——HTTP 200 ≠ 正确。** 早期 42/42 "全过" 的轮次全部作废：
bench 只验状态码，不验内容。此后一切验证必须过 `contam_check.py`（已知答案探针
"中国的首都是哪里？"）+ nonce 前缀探针（排除 radix 命中假象）双关。

**（2）判别实验阶梯。** 复现条件从"600rpm 全量"二分收敛到"4 giants + 10 medium"，
再逐级排除：1 giant / 2 giants / 4 giants-only（干净）vs 混合（污染）——一次实验
同时排除一批假设（窗口模式、DCP、EAGLE、CP、LoRA 全部洗清或定罪）。

**（3）数字指纹法。** 终局根因的定位完全由数字咬合驱动：

```
giant 的 page_indices：页 5499..5618 真实，其后 1444 页全 0
第一个零页 = 5619；5619 × 256(page_size) = 1,438,464
SWA 池大小 = 1,438,464  ← 精确咬合
日志指纹：KV-PRODUCER-OOB prealloc.kv_loc: 369491/400211 bad slots,
          samples=[1438464, 1438465, ...] cap=1438464
```

根因：`_guard_kv_indices` 三处调用点用 `allocator.size` 当 full 池虚拟 id 上限——
而 hybrid-SWA composite 的 `.size == min(full, swa)` 是 **SWA 尺寸**。水位越过
SWA 尺寸后，所有合法目的地 id 被静默洗成 0（共享 padding sink 页）→ 跨请求 KV
互读互写。修复=三处 cap 改 `size_full`（commit `76639ae1f1`），**一行比较上限的
差别，零性能损耗**。

**（4）py-spy 死锁形态学。** "8 rank 全停同一 all_reduce" vs "rank 散布不同轮次"
vs "7v1 错位"各自指向不同病根；串行抓取会产生散布假象，必须并发抓。

**（5）插桩判据体系。** 每个修复留下可 grep 的判据标记（`PDH-ADMIT/PARK`、
`KV-PRODUCER-OOB`、`DCP-PD-IDX`、`PDH-CLAMP-CHECK`、`EXTEND-ACCOUNTING-DIVERGENCE`、
`FILL-MISMATCH`…），修复是否存活、是否复发，一条 grep 即知。

### 3.2 战役年表（节选）

| 症状 | 根因 | 修复 |
|---|---|---|
| 16 并发 SIGQUIT 全崩 | spec_info 缺 DSPARK 分支 → stale draft_input 混批 → cuda graph shape 崩 | #31466 移植 + merge_batch None 守卫 + DSPARK 分支 |
| 个别请求永久卡死（health 200） | bootstrap 先于 hidden alloc 的时序倒置死锁环 | 前缀无关上界预留先于 bootstrap（背压替代死锁） |
| TTFT=606s 静默楔死 | 准入需求=整池，定理 OB 必然饿死 | 窗口化 + 记账式准入（§2.3） |
| 非 head 请求 park 1075s | 水位数学在 legacy 模式不可满足 | 水位仅窗口模式生效 |
| KV 满杀整个 scheduler | assert 直接 raise | 单请求 503 干净 abort |
| 并发巨型乱码 | 死代码 radix 钳制失效 → hidden 缺口 | 钳制复活五件套（§2.5） |
| accept 断崖 3.0→1.0 | draft loc 垃圾 → free 列表中毒 | 源头 clamp + scratch 页消毒 |
| accept=0.07 | merge 丢 draft pool multiplier → 虚拟 id 越界 | multiplier 恢复 + 全域转换移除 |
| 概率性乱码（重启减轻） | DCP 虚拟 id 双转 + topk lane 重定向 | 全域 owner 换算 + lane-0 重定向（`5974a4fc56`） |
| 40+ 并发传输风暴 IMA | transform_index kernel 缺上界 mask | 双界 mask（`8bfe264118`） |
| 跨请求 KV 污染（本次） | guard cap 用 SWA 尺寸当 full 域上限 | `size_full`（`76639ae1f1`） |
| 编译层：sm_103 无 a 后缀 ptxas 崩 | CUDA 13.2 fatbin 丢 a | nvcc wrapper gencode 长格式 |
| 编译层：tilelang 8 rank 齐崩 | 进程内锁不跨进程 | flock 跨进程锁 + CCCL include |

### 3.3 终局案例：跨请求 KV 污染（排查全程）

污染形态高度迷惑：**probe 稳定返回完全无关的 case50 内容**（DBT/99公益日/nginx），
且 accept rate = 1.00（draft 与 target 读同一份错数据——错得"自洽"）。被排除的假设
按顺序：接收窗口模式（legacy 也污染）、524K lane 溢出（V4 无此上限）、req_to_token
陈旧行（清零无效）、cuda graph 页表残留（8 处清零无效）、分配器 free 列表污染
（过滤器完好）、radix 树页对齐错位（树与分配器同页大小）。

破局点是把 `[DCP-PD-IDX]` 日志从"看个大概"升级为**逐请求统计**：每个请求的
page_indices 零页数/首零位置/真实页区间。19 条请求的表格让指纹一眼可见——前 15 个
请求页号连续递增（1..5498），第 16 个（第 4 个 giant）在页 5618 后 1444 个零，
随后 3 个 probe 全零。5619×256 与 SWA 尺寸的咬合直接指向 guard，翻代码 30 秒定案。

**教训沉淀**（已写入 AGENTS.md）：`allocator.size` 在 SWA composite 语义是
`min(full, swa)`——任何拿它当 full 池 id 上限的守卫，都会在高水位静默腐蚀。

---

## 4. 最终验证（2026-08-23，1102/1104 测试对）

| 验收项 | 结果 |
|---|---|
| ladder 四级（1g → 2g → 4g+10m → 42 并发全量 case50） | **20/20 probes clean**（含原污染复现场景） |
| case50 ×2 @600rpm | 两轮均 **42/50 ok + 8 已知 DFlash grammar-400**，内容检查 5/5 clean |
| 单轮耗时 | 2m21s / 2m28s（<3min 目标达成） |
| KV-PRODUCER-OOB / 零页 | 0 / 0（200 条 DCP-PD-IDX 全检） |
| 池泄漏（修复前 608 次） | **0** |
| DSPARK accept len | 3.15-3.75（健康区间；~5.9 是死循环病态信号） |
| 引擎状态 | 双端 health 200，0 Traceback / CUDA error |
| 公网冒烟 | `8.222.11.182:31000` → "中国的首都是哪里？" → "北京" ✓ |

性能基线（64 并发 bench）：TPOT p50 8.4ms、峰值 1707 tok/s；630K 等同请求 radix
命中 21.8s → 1.7s；200K 冷前缀 CP=8 加速 3.5×。

## 5. 残余项与边界（2026-08-24 更新）

- **DFlash 不支持 grammar-constrained**：strict tools/response_format 请求 400 快速
  失败（case50 中 8 例确定性失败，非部署缺陷，属上游能力边界）。
- **4×300K+ 巨型并发 + radix OFF 全量重算**时 DSpark draft 路径有
  `cudaErrorIllegalAddress`（radix ON 缓存命中降低有效 KV 可规避，40/50 后未复现）。
- **~~窗口 streaming 模式的跨请求 chunk 乱序 bug 未修~~** → 已于 2026-08-24 彻底
  修复并全量验收（见 §2.3 更新）：四层根因（脚本 env 位置 / socket 黑洞 / notify
  丢失 / park 竞态）全部结案，21-abort 风暴存活 + 523K 集体调用奇偶校验通过。
- **kernel 级数值非确定**：同输入 greedy 4 连发可有 3-4 种干净变体（GPU atomics），
  属硬件栈特性，与缓存系统无关（FORCE_MISS 实验定音）。

## 6. 资产索引

- 设计文档：`docs/agent/pd-hidden-window-design.md`（窗口化准入 + 形式化证明）、
  `docs/agent/dspark-pd-stuck-req-postmortem.md`（死锁原理）、
  `docs/agent/dsv4-pro-pd-engineering.md`（工程总纲）、
  `docs/agent/decode-radix-swa.md`（双端 radix 协议）、
  `docs/agent/dcp-virtual-id-domain-fix.md`（虚拟 id 域）
- 关键 commit：`1910bb619c`（窗口准入）、`98e7a09e9e`（钳制复活五件套）、
  `bddcf0fb34`（水位+优雅 abort）、`76639ae1f1`（污染终局）、
  `9db63a6abb`+`371a991947`（虚拟 id 域）、`5974a4fc56`（DCP 乱码双根因）
- 判据工具：`contam_check.py` / `lc_ladder.py`（内容金丝雀与判别阶梯），
  节点 `/tmp/bench_/`
