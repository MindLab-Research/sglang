# DCP 虚拟 id 域修复：EAGLE accept-cliff + Xid 31 崩溃终极根因（2026-08-20）

> commit `9db63a6abb`（b300-glm52）。解决两个集群的同家族崩溃：
> 2P3D 集群（8.222.11.182，DCP=4 fp8）EAGLE accept-cliff（1.05）+ 虚拟 id 污染；
> B300-2 bf16 集群（8.213.215.2:1022，DCP=8）Xid 31 写越界 + 8 rank 齐崩
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
