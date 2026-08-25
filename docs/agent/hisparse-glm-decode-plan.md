# GLM-5.3 Decode DCP HiSparse 实现计划（hisparse-glm-decode 分支）

> 目标：GLM（GlmMoeDsaForCausalLM，权重 /root/glm52_local/bf16，实为 5.3 架构：
> `indexer_types` 21×full + 57×shared、`index_share_for_mtp_iteration=True`——代码库
> 尚未消费该字段，当前按全层 indexer 跑）decode 端 DCP=8 + hisparse，**无 decode
> radix**，prefill 零改动。上一轮 DSV4 hisparse 实验的死亡根因（prefill forward 无
> hisparse 分支）在本方案中天然不存在。

## Phase 0 实测数据（2026-08-25，B300-2 生产机）

### 基线（生产 decode：tp8/dcp8/page128/EAGLE/fp8_e4m3）
- KV 池：639,488 tokens / 32.89 GB（≈54.5 KB/token/78 层）
- 模型：GLM bf16 TP8；**GPU 空闲显存仅 14.8 GB/rank**（生产占用后实测）

### 门 1：index_k device 驻留开销（host 容量的显存税）
index buffer 结构（`DSATokenToKVPool._create_index_buffers`）：每层一个
`(num_pages, 64×(128+4))` uint8 buffer = **132 B/token/层**。

| 方案 | index 开销 | 10GB 预算下 host 容量 | 收益 vs 原生 640K |
|---|---|---|---|
| 全 78 层存（现状池代码） | 10.3 KB/token | ~1.0 M | 1.56×（不值得） |
| shared indexer（21 full 层） | 2.77 KB/token | ~3.6 M | **5.6×（可行）** |
| 全层 + 降 mem-fraction 腾 26GB | 10.3 KB/token | ~2.5 M | 4.5×（备选，纯配置） |

**结论**：主线按全层 index 走通正确性；shared indexer（`indexer_types` 缓存共享）
是容量跃迁（1.56×→5.6×）的关键，作为 Phase 5；或部署时降 mem-fraction 换容量。

### 门 2：swap-in H2D 带宽（pinned，B300 PCIe gen5 实测）
- **55.5 GB/s**（64/128/256MB 稳定）
- 一步全量换入上限：topk 2048 × 54.5KB ≈ 111MB → **2.0 ms**
- 实际为 plan-then-IO 增量换入（滑动窗口连续步大部分页已驻留），真实增量远小于
  全量；对 GLM bf16 decode TPOT（数十 ms 级）预算占比 <10%。**通过**。

## 工作分解

### Phase 1：池构造 × DCP 虚拟 id 域统一（最核心，2-4 天）——✅ 域映射层已完成（2026-08-25）

**已完成并单测验证**（`test/hisparse/test_hisparse_dcp_allocator.py`，B300-2 上 5/5 PASS）：

- allocator DCP 化：`HiSparseTokenToKVPoolAllocator(dcp_size=D)` 三域定稿
  ——虚拟域 `Paged(size_full×D, page=64×D)`（对齐 9db63a6abb）/ 本地 full 域
  （owner 公式 `slot=(v//(64·D))·64+v%64`，与 `_write_mla_kv_buffer` 锁步对拍
  PASS）/ 本地 hisparse 窗口域 `Paged(size_hisparse, 64)`；mapping 虚拟域 dense 尺寸。
- `translate_loc_to_hisparse_device` DCP 感知（虚拟→local+owner 过滤→mapping；
  foreign→0 哨兵）；`set/get_mla_kv_buffer` 用 `dcp_disabled()` bypass 基类二次
  转换（3c68f20891 双转 bug 修复）。
- owner 分布单测：互斥完备，每 rank 恰 1/D。
- alloc/free 平衡 + 预算口径单测。

**三个关键发现（修正了原计划假设）**：

1. **上游 alloc_decode/alloc_extend 的同步 device 分配在 page>1 下本来就坏**
   （`PagedTokenToKVPoolAllocator.alloc()` 页对齐，bs<64 时发 0 页空 tensor；上游
   仅 page=1 有效——其 `alloc()` 自带 NotImplementedError 佐证）。修正设计：
   DCP/page>1 下 alloc_decode/alloc_extend 走 **logical-only**，device 窗口由
   HiSparseCoordinator demand-paged（`alloc_device_buffer`/`swap_in_selected_pages`
   页级写 mapping）——这才是 hisparse 架构本意（device=滑动窗口）。
2. **预算语义修正**：demand-paged 下窗口是可循环工作集（swap-out 腾页），不是容量
   约束——`available_size()` 在 page>1 下=logical_avail（host 容量），不再
   `min(逻辑, 窗口×dcp)`（SWA 预算口径教训的又一实例）。
3. **slot-0 哨兵歧义（上游遗留）**：`mapping[0]`/`>0` 过滤无法区分"local slot 0"
   与 foreign——本地域 id 0 是真实可分配槽。单测已绕开；若 Phase 3 集成中撞到
   slot-0 corner，把哨兵迁到 -1（mapping 尾部已有 -1 保留位）。

**Phase 1 剩余（未完成）**：坑 B GLM 版（传输页表换算）——依赖 Phase 2 的 host
池页域决策（host 本地页 1/dcp 分片 vs 虚拟页 64×dcp 的换算，与 mooncake DRAM
接收/`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 的 rank 映射耦合），随 Phase 2 一起定稿。
坑 C（guard cap）已改 `size_full`；EAGLE `prepare_for_draft` clamp 经确认**天然兼容**
——draft pool 非 hisparse 且非 sharded（`not is_draft_worker` + `_derive_pool_sizes`
恢复 size×dcp multiplier），虚拟 id 即 draft pool 直接索引，hisparse target 的
req_to_token 存虚拟 id 使其无缝。dcp=1+PD+EAGLE e2e 验证在 Phase 2 后统一执行。

> 部署注：B300-2 site-packages 已带本 patch（原版备份 `/root/hisparse_backup_orig/`）；
> 生产 `enable_hisparse=False` 不激活新代码路径（构造签名向后兼容 dcp_size=1），
> 生产 health 200 无扰验证过。

### Phase 1：池构造 × DCP 虚拟 id 域统一（最核心，2-4 天）
1. **page_size 断言链**：`DSATokenToKVPool` 断言 `page_size == 64`（CUDA 路径）；
   GLM 生产 page_size=128、DCP=8 虚拟域 page=128×8。梳理 configurator→pool 构造
   参数链，hisparse 池在 GLM DCP 下的 page 语义定稿（device 池 page 保持 64 语义
   对齐 index buffer 布局，虚拟域映射在 allocator 层做）。
2. **`HiSparseTokenToKVPoolAllocator`（通用版，allocator/hisparse.py:15）DCP 化**：
   capacity=size×dcp、page_size=64×dcp 虚拟域（对齐 PagedTokenToKVPoolAllocator 的
   DCP 先例 9db63a6abb）；`hisparse_attn_allocator` 的 available/alloc 域。
3. **`set_mla_kv_buffer` 双转修复**：hisparse 包装先 `translate_loc_to_hisparse_device`
   再 super()（基类 DCP 分支 virtual→local）——虚拟 id 输入时顺序错 = 3c68f20891
   同族。translate 感知虚拟域或调整顺序。
4. **坑 B GLM 版**（b752a53446 DSV4 参照）：PD 传输页表 hisparse 压缩页 vs DCP 逻辑
   页换算（GLM: 1024 vs 64），错 16× → KV 写错页。
5. **坑 C GLM 版**：`_guard_kv_indices` cap 用 `size_full`（虚拟域上限），非 hisparse
   压缩域 size。
6. **EAGLE `prepare_for_draft` clamp 域**（371a991947 教训）：draft `out_cache_loc`
   clamp 上限对虚拟域。

判据：dcp=1+PD+EAGLE 单请求 e2e 200、输出与基线逐字一致（nondet.py）、
`KV-PRODUCER-OOB=0`、`DRAFT-LOC-OOB=0`。

### Phase 2：PD 接收写 host 池（2 天）
- mooncake 主 KV（GLM `maybe_send_extra` 路径）dest 指向 `mem_pool_host` 页；
  decode.py:492 现有钩子是 DSV4 c4 路径，GLM 接主 KV。
- GLM 保持 **1/dcp 分片接收**（非 DSV4 广播）+ owner rank 过滤
  （`(id//page)%dcp==rank`）+ 收后登记 `full_to_hisparse_device_index_mapping`。

判据：4×300K 巨型并发传输全成功、host 池占用与 KV 量吻合、index_k 已驻 device。

### Phase 3：swap-in 调度 × EAGLE verify × rank-invariant（2-3 天）
- `HiSparseCoordinator.swap_in_selected_pages`/`naive_load_topk` 与 EAGLE 5-step
  draft+verify 流水线时序（`set_decode_producer_stream` 与 draft graph 流同步）。
- **`hisparse_req_budget` rank-invariant 审查**（decode.py:1203-1234 读本地
  `available_size()` → DCP 组内分歧 → 集体错位楔死；PADDED-AR 家族）。改记账式
  （1910bb619c 方法论）。
- retract/finish 的 host 页释放记账。

判据：accept len 2.2-3.2、TPOT 劣化 ≤10%、PADDED-AR 位点 8 rank 一致。

### Phase 4：cuda graph + 满负载验收（1-2 天）
- bs=64 capture：`padded_buffer_size × max_running_requests` device 窗口预算
  （DSV4 实验的死点，GLM 版数学重推）。
- 四轮负载验收（case50 同标准）+ 12 并发 TPOT 对比基线。

### Phase 5（容量增强，可选）：shared indexer 缓存共享
- `indexer_types` 21×full：池层 index buffer 只为 full 层分配 + layer_id→owning
  full 层映射；模型层 shared 复用（前向共享需确认 checkpoint 权重结构）。
- 收益：host 容量 1.56×→5.6×（门 1 数据）。风险：模型前向改造，独立验证。

## 复用资产
| 资产 | 来源 |
|---|---|
| HiSparse 全套（pool/host/coordinator/sparsity/#34329 IndexShare） | 上游，已在 b300-glm52 |
| 坑 B/C 修复模式 | hisparse-radix-hicache 分支 b752a53446（DSV4 版） |
| DCP 虚拟 id 域方法论 | 9db63a6abb + 371a991947 + dcp-virtual-id-domain-fix.md |
| 记账式准入 | 1910bb619c + pd-hidden-window-design.md |

## 实验环境注意
- GLM 生产 1P1D 在跑（8.213.215.2，B300-1 prefill/B300-2 decode，bf16）；实验重启
  decode 需用户许可或错峰；Phase 0 micro-bench 已在空闲显存完成（无干扰）。
- 实验 flag：decode 端 `--enable-hisparse`（prefill 不加）；env 见
  `start_glm52_bf16_pd.sh`（完整 env 铁律）。
- GLM 生产 page_size=128 注意：`DSATokenToKVPool` CUDA 路径断言 page 64——Phase 1
  第 1 项必查（生产今天怎么过的断言？——查 dcp 下池构造实际 page，可能 configurator
  传的就是 64）。
