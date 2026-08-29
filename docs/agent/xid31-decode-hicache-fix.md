# Xid 31 修复报告 — Decode HiCache MLA Staged Write-Back 越界 (2026-08-29)

## 摘要

1104 decode 节点在 case50 大流量下 GPU Xid 31 崩溃（6 GPU 同时 fault，固定偏移 0x7f...ac000）。根因定位为 MLA staged write-back kernel（`hicache_relayout_kernel`）在 DCP 虚拟页（page_size=256）下索引越界。修复后 case50 50/50 通过、0 Xid、0 乱码。

## 环境

| 节点 | 角色 | 配置 |
|---|---|---|
| 1102 (10.0.58.35) | prefill | GLM-5.3 FP8, CP=8 layersplit, HiCache file backend, port 30100 |
| 1104 (10.0.58.37) | decode | GLM-5.3 FP8, DCP=4, EAGLE 5步, decode radix + 三级 HiCache (write_back, file), indexer share off, port 30200 |

## 症状

- case50 @ 200RPM 压测 ~2 分钟后 GPU Xid 31（ACCESS_TYPE_VIRT_READ @ 0x7f...ac000，6 GPU 同偏移 704512B）
- 引擎 crash → CUDA coredump 等待 60s → watchdog SIGQUIT
- 小流量（case0/1/2）不触发——仅大流量（L2 write-back 量大时）才复现

## 排查过程

### 排除项

| 嫌疑 | 排除方法 | 结果 |
|---|---|---|
| DSA indexer 页号越界 | IDX-HOST-DIAG probe（memory_pool_host.py bounds check）| 页号全界内（h 2813/10587, d 12691/42342）|
| MLA 主 KV 页号越界 | MLA-HOST-DIAG probe（pool_host/mla.py bounds check）| 页号全界内（h 12031/169365）|
| staged JIT kernel 本身 | 离线复现（6 组参数：crash 页号/边界页/多 chunk/高页）| 全部 OK，无 Xid |
| srcLocHint 类型 | 读 staged_write_back.cuh 源码 | 是 Device（正确），非 bug |

### 定位

在 `relayout.cuh` 的 GPU kernel 中加入 `[RELAYOUT-OOB]` printf bounds check（越界时打印 layer/page/offset 并跳过），部署后 case50 触发：

```
[RELAYOUT-OOB] layer=37 page_id=62 src_page=16384 tok=18 vec=28 src_off=9448000 dst_off=713927680 elem=576 ps=256 np=64 nl=78
[RELAYOUT-OOB] layer=51 page_id=63 src_page=16640 tok=83 vec=12 src_off=9632640 ... ps=256 np=64 nl=78
```

## Root Cause

**`hicache_relayout_kernel`（relayout.cuh）在 DCP 虚拟页（page_size=256）下 MLA staged write-back 越界。**

调用链：`unified_radix_cache._inc_hit_count → write_backup → hybrid_cache_controller.write → MLATokenToKVPoolHost.backup_from_device_all_layer → jit_transfer_hicache_all_layer_mla_staged_lf_pf → hicache_relayout_kernel`

越界机制：
1. MLA backup 传入 **全局 token 索引**（`src_page=16384` = 第 64 页首 token，DCP 虚拟域）
2. kernel 计算 `src_token = src_page + token_in_page`，再乘 `elem=576B`
3. `src_off = 16384 × 576 + 18 × 576 ≈ 9.4MB`
4. staging buffer 仅 64 页 × 256 × 576 = **9.4MB**——精确越界
5. GPU 读越界 → VIRT_READ fault @ host pinned VA 0x7f...ac000（固定偏移 704512B，6 GPU 同偏移因 staging 布局相同）

**为什么只有大流量触发**：write_back 模式下 L2 backup 只在 `_inc_hit_count ≥ write_through_threshold` 时触发——case50 大流量产生足够多的 radix 命中才调用 backup；小流量不触发。

**为什么 DSA indexer 不触发**：DSA 的 staged write-back（`DSAIndexerPoolHost`）page_size=64（物理页），远小于 MLA 的 256（DCP 虚拟页），不会越界 staging。

## 修复 (commit edccd821c0)

| 文件 | 修改 |
|---|---|
| `pool_host/mla.py` | `page_size ≤ 64` gate：staged JIT 路径只在 ≤64 时走；>64 fallback 到非 JIT `transfer_kv_all_layer_mla_lf_pf` + CUDA tensor device fix |
| `relayout.cuh` | GPU 侧 `[RELAYOUT-OOB]` printf + skip（越界时打印 layer/page/offset 并跳过拷贝，不再静默 MMU fault）|
| `staged_write_back.cuh` | Host 侧 `[BATCHCOPY-OOB]` RuntimeCheck（dst 页号超界时响亮报错）|

诊断工具（保留，env 门控 `SGLANG_INDEX_HOST_DIAG=1`）：
- `memory_pool_host.py`：DSAIndexerPoolHost 的 `_get_indexer_page_indices` bounds probe `[IDX-HOST-DIAG]`
- `pool_host/mla.py`：MLATokenToKVPoolHost 的 `_mla_idx_probe` `[MLA-HOST-DIAG]`

## 验证

| 指标 | 结果 |
|---|---|
| case50 @ 200RPM | **50/50 http=200，0 失败** |
| 乱码检测 | **0/50（无 digit-pipe / doc:N:N / word-loop / 重复）** |
| Xid | **0**（550s 监控全程） |
| 引擎异常 | **0 exception** |
| 引擎健康 | health=200 稳定，decode batches 119+ 持续增长 |
| 总耗时 | ~3 分钟（TTFT p50：32K-128K 10.1s / 4K-32K 19.9s / <4K 36.6s）|

## 排查弯路

1. **write_through vs write_back**：最初改 write_through 以便小流量触发 L2 backup——修复验证后改回 write_back（原始配置）
2. **离线复现失败**：staged JIT kernel 在隔离环境不触发 Xid——因为离线环境的 page_size 用的是物理 64 而非 DCP 虚拟 256
3. **MLA probe 加错位置**：第一版 probe 加在 memory_pool_host.py 的旧类上（未被执行），第二版加到 pool_host/mla.py 正确位置
4. **bench 命令链断裂**：SSH 命令中 `\n` 导致 nohup python3 未执行，bench 从未发出请求——350s 监控 0 batches 是"bench 没跑"而非"decode 死锁"
5. **Python 修复引入两次新 bug**：`kv_buffer.device`（list 无 device 属性）→ `kv_buffer[0].device`（修复后 list.index 兼容）
6. **router AddWorker 超时**：引擎反复重启期间 smg router 的 discover_metadata 10s 超时 → "No decode workers available" → bench 请求 503

## 关键代码位置

```
python/sglang/srt/mem_cache/pool_host/mla.py:429-455   # page_size gate + fallback
python/sglang/jit_kernel/csrc/kvcacheio/relayout.cuh:43-75  # GPU bounds check
python/sglang/jit_kernel/csrc/kvcacheio/staged_write_back.cuh:131-152  # Host bounds check
```
