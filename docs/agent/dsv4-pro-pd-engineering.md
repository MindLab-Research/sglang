# DeepSeek-V4-Pro PD 分离推理集群工程报告

> **周期**：2026-08-13 → 2026-08-16（72h 连续工程，三段主线）
> **集群**：B300 1P1D（prefill 8×B300 @1021 + decode 8×B300 @1022），mooncake RDMA，
> router PD 分离（cache_aware），公网 `8.213.214.14:18888`
> **模型**：`deepseek-v4-pro-0813`（853GB 官方权重，FP4 expert + FP8 attention，
> 61 层，DSA 稀疏注意力 + 双层压缩 KV c4/c128 + SWA 滑窗）
> **目标**：PD 分离 + DSPARK 投机解码 + 双端 radix cache + prefill CP=8，
> 全特性、无降级、无 hack
>
> 本报告为总纲；乱码/非确定性战役的逐 Phase 细节见
> [`dsv4-radix-nondet-postmortem.md`](dsv4-radix-nondet-postmortem.md)。

---

## 0. 摘要

本工程在 sglang `b300-glm52` 分支上完成了 DeepSeek-V4-Pro 的 PD 分离推理全栈支持，
交付四项核心能力与三组深度修复：

| 能力 | 交付物 | 实测 |
|---|---|---|
| **DSPARK × PD 分离** | target 中间层 hidden 流式传输（#31466 移植 16 文件）+ spec_info DSPARK 分支 + merge_batch None 守卫 | 16 并发 + abort 洪峰 72/72 通过 |
| **双端 Radix Cache** | `swa_served_from_tree=False` 协议（树只持 FULL，SWA 尾窗重算） | 630K 等同请求 21.8s → **1.7s**（cached=721,920） |
| **Prefill CP=8** | aux hidden CP 重组 + `decode_engine_rank` 对角配对 | 200K 冷前缀 prefill **11K → 38.7K tok/s（3.5×）** |
| **确定性修复** | 三层状态清理 + c128 边界字典快照-恢复 | 真实负载冷热三轮 **0/150 短循环**，0 崩溃 0 泄漏 |

最终生产状态：CP=8 + 双端 radix 全开、accept rate 0.72-0.78、64 并发 TPOT p50
8.4ms / 峰值 1707 tok/s。残余项（kernel 级数值非确定）已定性并给出边界。

---

## 1. 系统架构

```
                    ┌─ prefill (1021, 8×TP, CP=8 interleave)
client ── router ───┤     │  KV + 压缩KV(c4/c128) + SWA ring + DSpark hidden
                    └─ decode (1022, 8×TP, DCP=4, DSPARK)
                          │  双端 radix tree（FULL 共享语义）
                          └─ mooncake RDMA (mlx5_0, bootstrap 8998)
```

DSV4-Pro 的注意力栈比常规模型多三层状态，这是全部工程难点的来源：

| 状态 | 语义 | 生命周期 |
|---|---|---|
| FULL KV | 全局长程注意力 | 可缓存、可跨请求共享 |
| SWA ring（滑窗 128） | 近窗注意力 | **per-request 窗口状态，不可共享** |
| c4 压缩链 | 4:1 压缩 KV 的增量 carry（记忆 ~64 token） | per-request，收敛半径内可清零重建 |
| c128 压缩链 | 128:1 压缩 KV 的增量 carry（**记忆 32K token**） | per-request，长记忆——**只能恢复，不能清零** |

---

## 2. DSPARK 投机解码的 PD 分离支持

### 2.1 问题定义

DSPARK 的 draft 模型消费 target 的**中间层激活**（`dspark_target_layer_ids=[58,59,60]`）
作为输入。PD 分离下 prefill 结束时这些激活只存在于 prefill 侧，decode 侧的 draft
无从获得——传统 EAGLE 的独立 draft-prefill 方案不适用（draft 权重与 target 激活
强耦合）。**唯一正确解：hidden state 本身作为第四种传输状态从 prefill 流到 decode。**

### 2.2 并发崩溃链（16 并发 SIGQUIT 全崩）

上线后 4+ 并发稳定崩溃。逐层解剖出一条完整的静默腐蚀链：

```
新请求进 prebuilt batch
  → build_disagg_draft_input() 返回 None        # spec_info.py 无 DSPARK 分支
  → merge_batch() 跳过 spec_info 合并            # None 守卫静默通过
  → running batch 持有 stale bs=1 的 draft_input # batch 已涨到 bs=N
  → verify ForwardBatch 混用 bs=1 input_ids + bs=N req_pool_indices
  → cuda graph fill_from: shape [1] vs [N] → SIGQUIT 全崩
```

关键认知：**崩溃点（cuda graph）离根因（spec_info 分支缺失）隔着四层**。
py-spy 与 `FILL-MISMATCH` 插桩（cuda_graph_buffer_registry 的 shape 不匹配打点）
才把链条接起来。

### 2.3 修复三件套

1. **#31466 上游移植**（16 文件）：PD hidden state 传输协议——prefill 在层循环中
   捕获 target 层激活，经 mooncake 流式分块传输，decode 侧 recv pool 落位后
   供 draft 消费。传输观测点 `PDH-SEND/RECV/ACK*`。
2. **`spec_info.py` 补 DSPARK 分支**：`build_disagg_draft_input` /
   `make_next_draft_input`——prebuilt batch 的 draft 输入构造不再断链。
3. **`schedule_batch.merge_batch` None 守卫**：spec_info 缺失时显式处理，
   杜绝 stale draft_input 混批。

兼容性硬约束：fake/nixl/mori 各 backend 的 `send_metadata` 必须带 `spec_metadata`
kwarg（签名不兼容会在 warmup 即 TypeError）。

### 2.4 Hidden 传输池工程

流式分块传输的两侧窗口必须配对：

- decode `SGLANG_PD_HIDDEN_RECV_POOL_TOKENS` 决定接收窗口（dst_indices）
- prefill `SGLANG_PD_HIDDEN_POOL_TOKENS` 必须装下同尺寸窗口

不配对的故障序列：`DSpark hidden rows exceed prefill hidden pool capacity`
→ 传输失败 → 客户端 abort → decode 侧已分配 window 不归还 →
`PD decode hidden pool blocked prealloc: free_rows=0` → 后续请求全阻塞。
生产配置：**双侧 65536**。

---

## 3. 双端 Radix Cache：DSpark hidden 的硬约束

### 3.1 为什么必须两端同开

DSpark hidden 是 target 中间层激活：**不可缓存、不可从 KV 反推，只能由本请求
实际前向计算产生**。由此产生硬约束：hidden 传输必须精确覆盖
`[decode_prefix_len, N)`。

先分清两类状态的传输语义（传输区间都由 **decode 承诺** `decode_prefix_len` 决定）：

| 状态 | 可缓存？ | 只开 prefill radix（decode 承诺=0） | 两端同开且命中一致 |
|---|---|---|---|
| KV（FULL/压缩/SWA） | ✅ 树槽持有 | **全量传**（从树槽读出即发） | 只传 delta |
| DSpark hidden | ❌ 只能本请求前向产生 | 见下 | 只传 delta |

危险场景是 **prefill 命中超过 decode 承诺且不加钳制**：prefill 命中 139K →
只对 delta 段 `[139K, N)` 做前向 → hidden 只剩 delta 段 → decode 侧 draft 需要
`[0, N)` 覆盖 → `hidden_start != decode_prefix_len` / 流式 out-of-order →
500 / abort（基 1992 实爆即此）。

**解协议**：`init_next_round_input` 将 prefill 的 radix 命中 **clamp 到 decode
承诺**——prefill 实际计算起点恒等于 `decode_prefix_len`，hidden/KV 传输区间
由此保持一致。两端同开的真正意义：只有 decode 侧树也命中同一前缀时，
`decode_prefix_len` 才非零，prefill 才能真正只算 delta、双端只传 delta。
只开 prefill 侧 = clamp 到 0 = 全量重算 + 全量传输，**正确但零缓存收益**。

decode 侧启用参数：`--disaggregation-decode-enable-radix-cache` +
`SGLANG_DECODE_RADIX_ALLOW_SWA=1`。

### 3.2 `swa_served_from_tree = False` 协议

SWA ring 是 per-request 窗口状态（含本请求自己的近窗 K/V），radix 树**从根本上
不可能持有它**。协议设计：

- 树节点只携带 FULL KV 槽位；SWA 组件对树表现为 tombstone
- validator 恒过（不对 SWA 做树校验）
- insert 的 overlap 复活分支跳过（防双重 free）
- **尾窗重算**：命中请求的窗口 SWA 由 prefill 重算后经 PD 新鲜传输
- 请求结束补 `free_swa`（防窗口槽泄漏）

### 3.3 记账教训

`ValueError: pool memory leak detected!` 打印 `[full]` 行完全平衡时**不要信**——
真凶在下一行 `[swa]`。SWA 池按请求生命周期管理，它的泄漏被 FULL 的平衡掩盖。
修复：insert/split 分支补 `free_swa(kv_indices[:page_aligned_len])`，
避免 `free()`/free_group 把 FULL 一起释放。

---

## 4. Prefill Context Parallelism 与 DSPARK 共存

CP=8（`--cp-strategy interleave`）切 DSA indexer 的 O(n²) —— 200K 前缀的收益
是 3.5×，但与 DSpark hidden 捕获存在三处冲突，逐一修复：

### 4.1 Flag 归一化顺序（首启崩溃）

`--enable-prefill-cp` 与 NSA CP 互斥断言崩溃：参数归一化先于 hook 把 MLA 别名
置 True，hook 再置 NSA 别名 → 互斥检查触发。修复：`deepseek_v4_hook.py` 显式
清 MLA 别名（V4 走 interleave，非 GLM 的 layersplit——两个 hook 语义不可混用）。

### 4.2 Aux hidden 的 CP 重组（`NotImplementedError` 硬禁）

v1 对 CP + hidden 捕获直接抛异常。正解：层循环内捕获**本 rank 的 CP 分片行**，
循环尾部 `cp_all_gather_rerange_output` 重组回全序列——与 final hidden 同法，
**下游 PD hidden 管道零改动**。

### 4.3 `decode_engine_rank` 对角配对（8 倍重复发送）

CP 把 prefill 的 attn_tp 从 8 稀释成 1，decode 侧 rank-mapping 退化为广播模式
（`required_dst_info_num=8`）：8 个 prefill rank 向**全部** decode rank 发送同样的
hidden chunk → decode 收到 8 份交错数据 → `hidden chunk arrived out of order` →
全 500。

修复：`TransferInfo` 增 `decode_engine_rank` 字段（metadata 帧 slot 12，旧值 -1
向后兼容），worker 发送循环对角过滤——仅 `decode_engine_rank == prefill_unique_rank`
的请求携带 hidden payload 与 ready 通知。8×8 降为 8 对 1:1，KV 广播语义不变。

### 4.4 实测

| 场景 | CP off | CP=8 |
|---|---|---|
| 200K 冷前缀 prefill drain | 11K tok/s | **38.7K tok/s** |
| 输出质量 | — | garble 零损（出师表全文背诵 finish=stop） |

---

## 5. 分布式死锁修复链

PD 分离引入的异步 poll 循环与集合通信交错，产生一族"health 200 但请求全超时"
的静默死锁。三轮修复，每轮以 py-spy 实证收口：

### 5.1 专用 gloo 组（`2dd9d2c168`）

并发 abort 洪峰（客户端超时取消 → `KVTransferError: Aborted by AbortReq`）下，
per-rank 清理耗时不同：快 rank 进入下一轮 `recv_requests` 的 **broadcast**、
慢 rank 还在 `pop_transferred` 的 **all_reduce**——两者共用 `attn_tp_cpu_group`，
gloo FIFO 把异构 collective 互相匹配 → 8 rank 全卡死。
py-spy：TP0/TP3 卡 `_padded_all_reduce_min`，TP5 卡 broadcast。
修复：`torch.distributed.new_group(backend="gloo")` 专用组给
DecodeTransferQueue/DecodePreallocQueue/PrefillBootstrapQueue——poll 序列与
broadcast/barrier 序列隔离，跨组时序漂移无害。

### 5.2 Poll 异常免疫（`c841e03cf9`）

专用组后仍有 7v1 卡死：某 rank 的 `kv_receiver.poll()` 抛异常，在
`_padded_all_reduce_min` **之前**逃逸 → 该组 collective 计数**永久偏移 1** →
该 rank 领先一整轮、其余 7 个永远等它。修复三件：
① `_poll_with_failure_injection`：每个 poll 包 try/except，异常转 `KVPoll.Failed`
并 stash 到 receiver；② pop_transferred 透传 stash 异常（含 is_from_another_rank）；
③ 专用组加 300s timeout（残余分歧响亮崩溃而非静默卡 watchdog）+
`PADDED-AR-FAIL` 进程级计数取证日志。

### 5.3 Prefill 空队列补偿（`1f0e0cc95a`）

`pop_bootstrapped` 空队列时直接 return、不调 `poll_and_all_reduce_attn_cp_tp_group`
——bootstrap 队列填充是 per-rank TCP（时序天然分歧），有请求的 rank 调 collective、
空队列 rank 不调 → attn_cp 组计数错位 → 8 rank 全卡。修复：空分支在
`attn_cp_size > 1` 时用空 poller 列表参与 collective（与 decode 侧同构）。

**家族共性**：HiCache local-only prefetch 让 per-rank 状态天然分歧——
**任何依赖 per-rank 状态的 collective gate 都是死锁温床**。判断标准只有一条：
collective 的调用与参数必须 rank-invariant。

---

## 6. KV 传输正确性：DSV4 全量池广播

DSV4 decode 的每个 DCP rank 持有**全量 KV 池**（`_pre_alloc_fill_len` 不按
dcp_size 缩减，req_to_token 是全量长度）。因此主 KV（SWA/DSA 经
`maybe_send_extra`）与压缩 KV（c4/c128 经 transfer_worker）都**必须广播**——
若按 DCP 语义分片（每 rank 收 1/dcp）：decode 的读取范围（req_to_token 0..len）
远超写入范围（0..len/dcp）→ **KV 污染 → target logits 错 → 输出乱码**
（draft 停滞是症状不是根因）。判据 `_is_dsv4_kv_transfer()`（is_deepseek_v4）；
GLM（1/dcp 布局）保持分片。验证：dcp=4 出师表全文背诵 finish=stop。

---

## 7. 乱码/非确定性战争（摘要）

> 完整战役记录（六 Phase、全部失败路线、数值破案过程）：
> [`dsv4-radix-nondet-postmortem.md`](dsv4-radix-nondet-postmortem.md)

### 7.1 症状与分层根因

高并发下 token 级循环（`_CTRL_CTRL_...`，复读**其他请求**的 prompt 片段）、
`finish=length` 锁死、DSPARK accept rate 爬到 0.99；temp=1.0 的正常 agent
请求同样中招。根因是分层的：

| 层 | 根因 | 修复 |
|---|---|---|
| 崩溃层 | 树入口 cap 毒化 `cache_unfinished` 自 rematch 的 dup 记账 → 整请求双 free（`full_num_used=-139520` = 8×16384+8448 逐位吻合） | cap 回归调用方（`0c0b127cce`） |
| 污染层 | radix 命中请求恢复计算时读到**前任槽主**的状态：SWA 窗口槽、c4 carry（64 token 记忆）、c128 链（**32K token 记忆**） | SWA 窗口钳位重算（`ce98ae3720`）+ 三族清零 + **c128 边界字典快照-恢复**（`0a53594fb2`） |
| 静默失败层 | `.get()` 打在 list 上被 `except: pass` 吞、unified-only gate 在实际 SPLIT 池恒 no-op、字节平面池按 token 槽索引越界 | 逐一修复 + 执行三态日志（`C128SNAP/C128RESTORE-OK/MISS/FAIL`） |
| 残余层 | **kernel 级数值非确定**：FORCE_MISS 实证——同输入强制全 miss 全量重算 ×4 仍 4 种输出（GPU atomics 累加顺序） | 超出缓存层；需上游确定性 kernel |

### 7.2 关键方法论（按伤害排序）

1. **同输入不同输出 = 状态污染，永远成立**。"模型行为/温度/prompt 质量"解释
   在 greedy 同输入分叉面前一律失效。
2. **`except: pass` 是排查黑洞**——三个静默失败每一个都让"已部署的修复"变成空气。
   修复链必须带执行日志，用日志证明"跑了"。
3. **长记忆链不能清零只能恢复**：c128 的 32K token 记忆决定了命中后几百 token 的
   重算永远无法从空 carry 重建——快照-恢复是唯一正确语义。
4. **数值对账可以定罪**：139520 的逐位分解比任何 stack trace 都硬。
5. **bisector 也要撤**：诊断用的 c128 KV 窗口清零本身有害（清掉不被重算重建的
   长程记忆 → 上下文塌缩成近窗复制，循环率 2/4→3/4）。

### 7.3 最终验证

真实生产 prompt（cases_50，全 temp=0）× 600rpm × 冷/热/热三轮：

| 指标 | 结果 |
|---|---|
| 短循环（KV 污染特征） | **0/150**（修复前同款热轮有） |
| 长循环（模型能力特征） | 2/150，同一内禀退化 prompt（评分细则循环论证；冷轮同卡未结晶） |
| 崩溃 / 泄漏 | 0 / 0 |
| 冷==热 vs 热==热不一致率 | 相同（~60%）→ **缓存命中不再引入额外分叉**；分歧=kernel 级基线 |

---

## 8. 编译与运行时基础设施

### 8.1 DeepGEMM `sm_103a`（16 并发 crash 的编译层根因）

B300 真实 capability=(10,3)=SM103（device_name 伪装 "NVIDIA L20D"）。DeepGEMM JIT
拼 `--gpu-architecture=sm_103`（无 a）→ 大 shape 生成 tcgen05 指令 → ptxas 拒绝
（`'tcgen05.fence' not supported on .target 'sm_103'`）→ prefill crash（decode 表现
为 `reconnect to 8998`——**断连是后果非根因**）。且 CUDA 13.2 nvcc 的
`-arch=sm_103a` 短格式在 fatbin 模式下丢 a 后缀，必须
`-gencode arch=compute_103a,code=sm_103a` 长格式。修复：替换 pip 包内 nvcc 为
wrapper（原文件 `nvcc.real`），拦截重写架构参数。注意 `DG_JIT_NVCC_COMPILER`
env 无效（DeepGEMM 0.1.4 不读），动 CUDA_HOME 会触发 flashinfer 全量重编。

### 8.2 tilelang 并发编译

8 个 scheduler 是独立进程，tilelang KernelCache 只有进程内 `threading.Lock`——
空闲后首个并发 burst 8 rank 同时 cache-miss → 同时 `tilelang.lower` → staging
目录竞争 → **TVM C 层崩溃**（8 rank 齐崩）。修复：`cache/kernel_cache.py` 补
`fcntl.flock` 跨进程文件锁 + 锁内 double-check 磁盘加载；`engine/lower.py` 补
`-I nvidia/cuda_cccl/include`（`cuda/atomic` 头）；`SGLANG_DSV4_MHC_PREWARM=1`
把 JIT 编译移出 serving 路径（barrier 同步预热）。

### 8.3 性能级运行时坑

- **树 sanity check**：`invariant_checker._check_tree_cache` 首次满足
  `is_tree_cache() && is_hybrid_swa` 后，每次 idle 全树 O(N) 检查数秒 →
  **吞吐塌到 0.81 tok/s（health 200 假活）**。`SGLANG_ENABLE_TREE_SANITY_CHECK`
  默认关闭。
- **诊断日志税**：DSP 热路径 12 个观测点全部 gate 在 `SGLANG_DEBUG_DIAG`
  （默认关）——开启前 decode 1281 行/s ≈ 3-5ms/步纯日志开销。
- **L20D triton config**：B300 按 device_name 匹配 fused_moe config，若本地
  `*L20D*.json` 被 rsync 上节点 → MoE kernel 用错配置 → TPOT 23ms→80ms。
  每次 rsync 后必须删（DSV4 Pro 走 flashinfer_mxfp4 不受影响，但 GLM 路径仍在）。
- **部署纪律**：rsync 后全树清 `__pycache__`（旧 .pyc 遮挡新代码，本轮实抓 3 次
  "已部署未生效"）；目录+文件混合 rsync 会平铺（必须逐条对齐）；文件传输禁 scp
  用 rsync（B300 节点 scp 挂起实测）。

---

## 9. 性能与验证汇总

| 指标 | 数值 |
|---|---|
| prefill CP=8 加速（200K 冷前缀） | **3.5×**（11K → 38.7K tok/s） |
| 630K 等同请求（radix 命中） | 21.8s → **1.7s**（cached=721,920） |
| agent 长会话缓存命中 | 95%+（TTFT 恢复正常） |
| 64 并发 bench | 64/64 成功，TPOT p50 8.4ms，峰值 1707 tok/s |
| DSPARK accept rate | 0.72-0.78（健康区间，非锁死值） |
| 冷热三轮短循环 | 0/150 |
| 崩溃 / 泄漏 | 0 / 0 |

---

## 10. 交付物索引

**核心 commit 链**（`b300-glm52`，本工程段）：

```
20856e6d3f  双端 radix：swa_served_from_tree=False + 诊断日志
d300afc750  CP×DSpark 三层修复（hook flag / aux hidden 重组 / decode_engine_rank）
f29e50bc93  hidden pool 双侧配对 65536 + PDH 插桩
ce98ae3720  prefill SWA 窗口钳位（命中永不伸入尾窗）
0c0b127cce  dup 双 free 修复（cap 回归调用方，-139520 逐位归因）
2683f462e4  SPLIT 池适配（unified-only gate 盲点修复）
e50c67c47b  SWA 池字节平面寻址（token 槽索引越界修复）
0a53594fb2  c128 边界字典快照-恢复（最终形态）
e39a1ae7df  component_data list/dict 索引修复（静默失败大扫除）
b3d1790f65  战役收官文档
```

**判据工具**（dev 47.87.64.67 与 1021 节点）：

| 工具 | 用途 |
|---|---|
| `/tmp/nondet.py` | 4 连发分叉计数；**cached 值直接指示对齐状态**（131072=对齐 / 139264=越界） |
| `/tmp/replay_dump.py` | 复刻 replay 采样 + dump 全文（判循环必须看输出，别信正则） |
| `/root/coldhot_cmp.py` | 三轮 dump 逐 case 一致性（分离缓存/内核因子） |
| `/root/longloop_scan.py` | 长单元（40-500 字符）重复扫描——短循环正则抓不到长循环，两者都要跑 |
| `SGLANG_DEBUG_ALLOC=1` | [ALLOC-DIAG]/[STATE-CLEAR*]/[C128SNAP/C128RESTORE] 执行三态 |
| `SGLANG_RADIX_FORCE_MISS=1` | 强制全 miss——缓存层 vs kernel 层非确定的定音分离实验 |

---

## 11. 残余项与边界（如实声明）

1. **Kernel 级数值非确定**（FORCE_MISS 实证）：本部署硬件栈无法做到同输入
   bit 级一致。影响：所有 prompt 存在措辞级微差；仅当 prompt 自身分布退化
   （乱码/重复结构/无界枚举）时放大为循环。根除需上游确定性 kernel 工程
   （flashinfer/triton 原子操作路径的确定性模式）。
2. **退化 prompt 的长循环**：内禀循环论证的 prompt（如评分细则互斥条款）在
   greedy 下仍会结晶为精确长循环——模型能力边界，与缓存无关；agent 侧应避免
   对此类 prompt 用 temp=0 长生成。
3. **router `cache_aware` 与 prefill radix OFF 不兼容**：二分实验关 radix 时
   必须同时换 `--policy round_robin`，否则健康探测失败 503 卡死。

---

*关联文档：`dsv4-radix-nondet-postmortem.md`（战役全记录）、`decode-radix-swa.md`
（双端 radix 设计）、`dsv4-cp-dspark.md`（CP 共存细节）、`dspark-pd-deadlocks.md`
（死锁五连）、`v4-pro-deploy.md`（部署与权重校验）、`b300-compile-fixes.md`
（编译三修复）、`AGENTS.md` §7（陷阱速记，本轮新增 6 条）。*
