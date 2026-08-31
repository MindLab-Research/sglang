# DCP 虚拟 id 域修复：EAGLE accept-cliff + Xid 31 崩溃终极根因（2026-08-20）

> commit `9db63a6abb`（b300-glm52）。解决两个集群的同家族崩溃：
> 2P3D 集群（8.222.11.182，DCP=4 fp8）EAGLE accept-cliff（1.05）+ 虚拟 id 污染；
> B300-2 bf16 集群（8.213.214.14:1022，DCP=8）Xid 31 写越界 + 8 rank 齐崩
> （排查文档 `docs/agent/decode-crash-2026-08-20-0144.md`）。

---

## 1. 背景：崩溃/退化家族时间线

| 阶段 | commit | 现象 |
|---|---|---|
| 裸奔期 | — | draft loc 越界 → Xid 31 WRITE 崩 TP 组（12+ 次） |
| 只修写端 | — | 不崩但垃圾进 free → 分配器永久中毒 → **accept 3.0→1.0 断崖** |
| 源头 clamp | `0e5f7704e1` | draft loc 源头 clamp，防越界 |
| 读/写/4-chokepoint 修复 | `3c68f20891` | 读侧 + write-side + mtp_precompute 4 点 **global→local 转换** |
| **虚拟 id 域修复（本文件）** | **`9db63a6abb`** | **移除 prepare_for_draft 转换（双转）+ DCP sanitize 修复 + 读路径 rank 过滤** |

## 2. 根因（2026-08-20 定位）

### 2.1 DCP 虚拟 id 空间结构

decode 端 allocator（kv_cache_configurator.py L1476-1484，GLM decode 分支）：

```python
PagedTokenToKVPoolAllocator(
    sizes.max_total_num_tokens * dcp_size,   # 虚拟容量 = size×dcp（DCP=4 → ~7.4M）
    page_size=page_size * dcp_size,          # 虚拟页 = 64×dcp = 256（DCP=4）
)
```

- **虚拟 slot id** = 虚拟页×256 + off；**req_to_token 行存的就是虚拟 id**（`_pre_alloc` 写 kv_loc 原样）
- 每个虚拟页 256 slot = 4 个物理页（64 slot 各）；**物理页 4v+k 固定属于 rank k**（页级 round-robin）
- 虚拟 id 大于本地物理池容量（1.85M）是**正常**的（虚拟水位 = 活跃 KV 页数，64 并发×长请求自然超过）

### 2.2 三处映射的一致性（这是"设计"）

| 路径 | 公式 | 位置 |
|---|---|---|
| decode 发页索引 | `kv_to_page_indices(kv, 64)` = `[::64]//64` → `[4v,4v+1,4v+2,4v+3,…]` | common.py L36 |
| prefill 传输分片 | `dst%dcp==rank` 过滤 + `dst//dcp` 落位 | conn.py L2073-2076 |
| decode 写路径 | `(loc//256)*64 + loc%64` + owner 过滤 `(loc//64)%dcp==rank` | memory_pool.py `_write_mla_kv_buffer` |
| decode 读修复 | 同一公式 | dsa_backend.py / dsa_backend_mtp_precompute.py |

**传输、写、读三处公式一致**——2D reshard（prefill CP layer-split × decode DCP）配对起点（decode_prefix_len == total_prefix_len）也一致。**传输没有传错位置。**

### 2.3 真正的 bug：`3c68f20891` 的修复 = 双转

`3c68f20891` 在 `prepare_for_draft` 加了 global→local 转换（`(loc//(64*dcp))*64+loc%64`）——
**但 draft 的 KV 写走 `set_mla_kv_buffer` 的 DCP 分支，该分支本身就做 virtual→local 映射 + owner 过滤**。
两处转换叠加 = **双转**：

- 虚拟 id（≥size）→ prepare_for_draft 转成本地 id → set_mla_kv_buffer 再转 → **写错位置**（本地 id 被当虚拟 id 再转）
- `set_mla_kv_buffer` 的 sanitize（`≥size→scratch`）在 DCP 分支**之前**把合法虚拟 id 打成 scratch，
  scratch 再被 DCP 分支当虚拟 id 转 → **转出的位置可能 ≥ 物理池 → Xid 31 VIRT_WRITE**（隔壁 09:44:05 MMU Fault）
- 读路径修复（dsa_backend / mtp_precompute）**无 rank 归属过滤**：虚拟页 v 属于 rank v%dcp，
  **异 rank 页也被转成本 rank 本地位置** → 读到别的 rank 的物理页 → KV 内容污染 → **accept 0.55→0.01**
- 修复后的本地 id 流入 free() → allocator（期望虚拟 id）按 256 粒度解释 → 归还错误虚拟页 → **分配器中毒**

### 2.4 为什么 1100 中毒而 1103 健康（2P3D）

- 虚拟 id ≥ 本地容量（1.85M）才触发修复 → 虚拟水位高（负载大）的节点触发 → 双转 → 中毒
- 负载低的节点虚拟 id 全 < 容量 → 修复不触发 → 单转 → 健康
- 1103 从 accept 5.95 降到 2.27 = 水位也在涨，即将触发（理论预测，修复后验证）

## 3. 修复（commit `9db63a6abb`，4 文件）

**原则：虚拟 id 是 DCP 下的唯一 id 域，全程保留；写路径单转（set_mla_kv_buffer DCP 分支）、
读路径 rank 过滤、free 用虚拟 id。**

| 文件 | 修改 |
|---|---|
| `speculative/eagle_worker_common.py` | **移除** prepare_for_draft 的 global→local 转换（保持虚拟 id，clamp 只 min=0 不钳 max）；保留 DRAFT-LOC-OOB probe + 新增 DRAFT-LOC-FOREIGN rank 归属审计（`SGLANG_DSA_STAGE_SYNC=1` 时打印） |
| `mem_cache/memory_pool.py` | `set_mla_kv_buffer` **DCP 下跳过 ≥size sanitize**（虚拟 id 合法）；OOB 窗口放宽到 `size×dcp`（非 DCP 保持原 sanitize） |
| `layers/attention/dsa/dsa_backend_mtp_precompute.py` | `_repair_global_kv_slots_` 加 **rank 归属过滤**：`(t//ps)%dws==drank` 才转本地，异 rank 页 → 0；纯 GPU 无同步 |
| `layers/attention/dsa_backend.py` | 读路径修复加 **rank 归属过滤**：本 rank 虚拟页 → 本地，异 rank 页 → -1（clamp 兜底 0） |

公式全部 `_page*_dws` / `%dws` 通用写法——**DCP=4 和 DCP=8 均适用**。

## 4. 验证

### 4.1 2P3D（DCP=4，8.222.11.182，2026-08-20）

部署 `9db63a6abb` 后重启 1100/1103，stress80_mix（80 并发，88% short-think + 12% longctx，600s）：

| 指标 | 修复前（1100 中毒） | 修复后（64 并发满负载） |
|---|---|---|
| DRAFT-LOC-OOB | 59032+ 且每秒 +145 | **0** |
| KV-PRODUCER-OOB | 0 | 0 |
| accept | 1.05（10-14 并发） | **3.19-3.21**（64 并发满负载） |
| DRAFT-LOC-FOREIGN | —（无审计） | 0 |

（验证数据以 task 64f24b3c 最终输出 + drain 后 fresh request 为准，待补）

### 4.2 判据（grep 用）

- `grep -c DRAFT-LOC-OOB /root/decode.log` → **0 = 正常**；>0 增长 = 虚拟水位触发修复（旧代码）或行污染
- `grep -c DRAFT-LOC-FOREIGN /root/decode.log` → **0 = 无跨 rank 污染**；>0 = 行被异 rank 页污染（producer bug）
- `nodecheck.sh` 的 `accept=`：健康单请求 ~5.9；64 并发满负载 ~3.2；<1.5 = 中毒
- 崩溃前兆：accept 健康→0.01 持续 ~30s → Xid 31 → 8 rank 齐崩（见 `decode-crash-2026-08-20-0144.md`）

## 5. 部署备忘

- 代码路径：2P3D decode = `/root/v15_patched/lib/python3.12/site-packages/sglang/...`
- rsync 后必须 `find ... \( -name '__pycache__' -o -name '*.pyc' \) | xargs rm -rf`（AGENTS 铁律）
- 重启脚本：`/root/start_1p2d_lora.sh`（1100/1103 各一份，已备份到本地 `deploy/restore/`）
- 验证链：health 200×2 → router 重启 → warmup → stress80_mix 20min → nodecheck → drain 后 fresh 请求

---

## 6. 终局：draft pool 尺寸丢失才是总根因（commit `371a991947`，2026-08-20 晚）

> 本章是最终结论。第 2-3 章的"虚拟 id 全程保留"修复（9db63a6abb）在 **target 侧**正确且必要，
> 但当时没发现 draft pool 本身的尺寸已经错了——所以 accept 仍停留在 1.4-2.2。

### 6.1 完整因果链

1. **v0.5.16 merge（`31d5960589`）把 draft pool 构造改走 `KVCacheConfigurator.configure() → _derive_pool_sizes()`**，
   而 `draft_pool_token_multiplier`（draft pool = size×dcp）留死在 `_apply_memory_pool_config`
   （draft worker 不再经过 `init_memory_pool`，永远到不了那行代码）。
2. **draft pool 从 7.4M（全虚拟空间）缩到 1.85M（target 本地尺寸）**。而 req_to_token 存虚拟 id
   （可达 7.4M）——draft 的写/读/free 全部用虚拟 id 直接索引 draft pool → 越界。
3. 越界之后的全部"修复"（set_mla_kv_buffer scratch sanitize、prepare_for_draft 虚拟→本地压缩、
   读路径 repair）都是对错误尺寸的补偿：own-rank 虚拟 id 压缩成本地 id（碰撞），**foreign-rank
   虚拟 id（3/4 的 draft 槽位）全部挤进一个 scratch 页互相覆盖** → draft attention 读到垃圾 KV
   → draft 输出近乎随机 → **accept rate 0.07（len 1.38）**。
4. **pre-merge 为什么正常**：merge 前 draft 走 `init_memory_pool → _apply_memory_pool_config`
   （multiplier 生效）→ draft pool = 7.4M，`prepare_for_draft` 无任何转换、`_write_mla_kv_buffer`
   无 else-sanitize——虚拟 id 全程直通 draft pool。merge 后两条 wiring 都变了，两个补偿 bug 叠加。

### 6.2 修复三件套（`371a991947`）

| 文件 | 修改 | 原理 |
|---|---|---|
| `mem_cache/kv_cache_configurator.py` `_derive_pool_sizes` | draft worker（EAGLE/STANDALONE + dcp>1）时 `max_total_num_tokens ×= dcp_size`（hybrid 同步扩 full/swa） | **恢复全量 draft pool**。内存本就预留：target 的 cell_size 已按 `(1 + draft_layers/num_layers × dcp)` 缩放（pool_configurator L147-173） |
| `speculative/eagle_worker_common.py` | `prepare_for_draft` / `prepare_for_draft_extend`：**移除全部虚拟→本地转换**，只保留垃圾 clamp（≥ draft pool size → scratch 页） | 虚拟 id 是全量 draft pool 的**合法直接索引**（draft 不分片）。压缩转换 = id 空间折叠 = 槽位碰撞 |
| `speculative/spec_utils.py` `move_accept_tokens` | `accept_out_cache_loc` 与 `tgt_cache_loc` **双侧同公式转换**（target pool 域：own→local、foreign→scratch） | src 现在是原始虚拟 id（prepare_for_draft 不再预转换），单侧转换会导致 src 以虚拟 id 直接索引 target 池 → OOB 读 |

**读路径无需改动**：`_repair_global_kv_slots_`（mtp_precompute）与 dsa_backend 的读修复都以
各自 backend 的 `token_to_kv_pool.size` 为界——draft pool 恢复 7.4M 后，合法虚拟 id `< size`
全部直通（修复自动变 no-op）；target pool 仍是 1.85M，target 侧转换照常正确。
`_localize_index_k_cache_locs` 是 ContextVar-gated（draft=dcp_disabled 直接 return，原始虚拟 id
直通 draft index_k 缓存 7.4M；target=dcp_enabled 转换）——与 pool 尺寸配套后自然正确。

### 6.3 验证（1102/1104 测试环境，DCP=4，EAGLE topk=1 steps=5）

| 指标 | merge 后（坏） | 本修复后 |
|---|---|---|
| draft pool | 1,857,088 / 1.22 GB | **7,421,696 / 4.89 GB**（=size×4） |
| accept len | 1.38-1.60 | **2.23-3.17 稳定** |
| accept rate | 0.07-0.10 | **0.24-0.43** |
| 10min 8并发压测 | 崩溃/退化 | **428/428 成功，0 崩溃，0 OOB** |
| LoRA L2 8000-token | — | 200、adapter 激活、输出干净无循环 |

### 6.4 认知修正：accept ~5.9 是病态不是健康

- **accept 单请求 ~5.9 ≈ 每个 draft token 全接受 = 极大概率是死循环/复读**（draft 沿着退化
  轨道走，target 全盘接受）。历史"健康单请求 5.9"的判据是错的。
- **健康基准修正：accept len 2.2-3.2 / rate 0.24-0.43**（8 并发实测），与"64 并发 ~3.2"
  的历史数据吻合。**>5 即应怀疑死循环**，配合重复行检测确认。
- （`docs/agent/dsv4-radix-nondet-postmortem.md` 的 FORCE_MISS 结论——kernel 级非确定——
  与此独立：那是输出分布分叉，这是 accept 统计。）

### 6.5 判据更新（grep 用）

- `grep "KV Cache is allocated" decode.log`：draft 行 `#tokens` 必须 = target 行 × dcp_size
  （DCP=4：7,421,696 vs 1,855,424）。**draft 行 #tokens == target 行 = multiplier 又丢了**（merge/重构回归的签名）
- `grep -c 'DRAFT-LOC-OOB\|DSA-SLOT-OOB' decode.log` → 0；>0 = 有 id 越界（查 pool 尺寸）
- accept len 健康带 2.2-3.2；<1.5 = draft 域错；>5 = 疑似死循环（查输出重复）

## 7. 2026-08-22 终局：DCP decode 概率性乱码双根因（commit `5974a4fc56`）

**现象学**（GLM-5.2 1P2D/2P3D 长期 MoL"乱码"家族的真身）：模型"自信地"输出错 token
（logprob rank-0 ≈100%）、正确/错误内容混合、中途开始尾部常自愈、逐请求概率性
（~37-50%）、抄自己上下文更早内容。**A/B 判定链**：prefill CP 关→仍乱（CP 排除）；
EAGLE 关→仍乱（排除）；base 模型→仍乱（LoRA-only 排除）；DCP=1→干净（**根因在 DCP>1**）。

### Bug A（主因）：index_k in-pool 直通错位 — `dsa_indexer.py::_localize_index_k_cache_locs`

- 旧代码 `virtual = loc >= pool.size` 门控：假设 in-pool id 已是本地槽。该假设只对
  EAGLE draft 池（size×dcp）成立；**target 池是 per-rank size（1.85M），而分配器虚拟 id
  从低位顺序发放** → 前几个大请求（水位 < size）的 index_k 全部未换算写错槽
  （应写 `(x//256)*64 + x%64`，实际写到 `x`，越界内静默错位）。
- 后果：indexer 用错位 index_k 打分 → 稀疏 top-k 选错页 → attention 读"真实但错误"的页。
- **解释全部症状**：水位越过 size 后自愈（= "重启 decode 减轻"+"尾部自愈"+概率性）。
- 修复：去门控，全域 owner 判定换算（镜像 `_write_mla_kv_buffer` 的 DCP 分支），
  非 owner → scratch slot（与 MLA 写路径一致）。

### Bug B：topk -1 补齐 lane → slot 0 污染 — `dsa_backend.py`（trtllm 直通路径）

- DCP 下每 rank 从自己的 ~1/4 页选 top-2048（GLM-5.2 `index_topk=2048`）；上下文
  <524K token 的请求 local 页数 <2048 → v2 输出大量 `-1` 补齐 lane。
- 非 DCP 路径的 `transform_index_page_table_decode` 会 mask -1；**DCP 直通路径不 mask**，
  `clamp_(min=0)` 把每条 -1 lane 重定向到 slot 0 = **无关 token 的 KV**。
- 定罪证据：空 rank（seq=0）trtllm 返回 `lse=0` 而非 -inf → **kernel 对每条 lane 真实
  attend，无 per-lane mask**。最多 3/4 lane 指向同一外部 token。
- 修复：-1 lane 重定向到本行 lane-0（自己的 top-1 页）；空行保持 -1（下游 lse=-inf
  掩码）。同时把 `SGLANG_DSA_SLOT_OOB_DIAG` 计数移到 clamp **前**（原位置 post-clamp
  恒 0，永远测不到污染）。

### 验证与部署

- 测试对（1102）：DCP=4+EAGLE，L2 迭代 agent 流量从 ~37-50% 乱码 → 修复后持续干净。
- 生产 1P2D（1100/1103）2026-08-22 部署，公网 18777 base+L2 验证通过。
- 排除项记录：LSE log2/ln 域（恒偏置≠概率性，非主因，留探针）；7cc8b4b64 的
  target_verify indexer 因果性语义（EAGLE-only 路径，次生嫌疑）。

### 7.1 生产部署同日的两个运维坑

- **双 decode 同时"自杀"（17:10:48，新进程跑 8.5min 后双双 multiprocessing 干净退出）**：
  auth 日志零外部会话、无 OOM、无 cron、KillUserProcesses=no → 进程自退出；traceback
  被后续重启的 `> decode.log` 截断丢失。怀疑 SIGKILL 式重启（mooncake/bootstrap 会话
  残留）触发双端 fatal 路径。**重启 decode 前 log 先备份**（cp decode.log decode.log.bak）。
- **stack 脚本 kill_exact 在 1101 上匹配不到端口**（ss 输出格式）→ proxy 杀不死 → 新
  proxy "Address already in use" 启动失败。老 proxy 一直健康，公网 18777 = 云 NAT →
  proxy:31000（本机 curl 127.0.0.1:18777 永远空，必须打 `8.222.11.182:18777`）。
