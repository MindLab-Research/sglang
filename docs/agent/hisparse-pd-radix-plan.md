# hisparse × PD × radix 打通实现计划（2026-08-24）

> 目标：让 DSV4 **HiSparse** 在 **PD 分离 + DSpark 投机解码 + DCP=4 + decode radix + CP** 的既有参数组合下真正跑通，
> **bs=64、case50 四连过、无握手失败**，且**无 hack、不降 bs、不关 radix、不牺牲性能**。
> 本文是**实现计划**（不是已完成的代码），供审查后执行。

---

## 一、核心架构判断（已实测 + 源码溯源确定）

hisparse 的正确内存模型：**完整 KV 常驻 host（`hisparse_coordinator.mem_pool_host`），device 只放压缩 top-k swap 页（`hisparse_attn_allocator`，DSV4 compress_ratio=4）**。
打通的本质 = **让 prefill 与 decode 两端都遵循此模型，且两端几何/索引域严格一致**。

**三个失败表现的根因统一**：
| 现象 | 根因 |
|---|---|
| 两端 hisparse → prefill OOM | `prefill.py` 无 hisparse 分支，KV 走 device 池；host 池（24GB 已建）闲置 |
| decode 单端 + prefill 普通 → 域不匹配 | prefill full 域（page_size=256）vs decode 压缩域（64） |
| bs=64 capture OOM | hisparse device 池 + 权重占用，未给 verify graph 留显存 |

**已确认（Phase 0 实证）**：
- `model_runner.py:781` `maybe_init_hisparse_coordinator` + `scheduler.py:958` `init_hisparse_coordinator` 在 **prefill 端也执行**（非 draft 即创建）。
- prefill 日志实锤：`Allocating 24.26 GB host memory for V4 paged pool 'dsv4_hisparse_c4'` —— **prefill host 池已建（24GB）但 `prefill.py` 不用它**。

---

## 二、分 Phase 实现步骤（含精确改动面）

### Phase 0 — 架构基线（验证 prefill 有 host 池）
- 确认 prefill 端 `hisparse_coordinator.mem_pool_host` 非 None 且容量>0。
- 改动：`prefill.py` 增加获取 `self.scheduler.hisparse_coordinator`（若 None 则响亮报错，给诊断清晰度）。
- 验证：prefill 拿到有效 host 池。

### Phase 1 — prefill 端 host 池分配（**核心缺口，消灭 OOM**）
- `prefill.py` 的 KV 分配路径（`alloc_for_extend` / `alloc_req_slots`，经 `allocation.py:441 alloc_for_extend`）加 hisparse 分支：
  - 完整 KV 分配走 `mem_pool_host.alloc_paged_token_slots`（host 池），非 device 逻辑池。
  - `full_available_size` 在 hisparse 下反映 **host 池可用量**（而非 `min(device 逻辑池, 压缩池×4)`）。
- 关键文件：`prefill.py`、`mem_cache/allocation.py`、`mem_cache/allocator/hisparse.py`（`full_available_size` 381-385）。
- **验证**：prefill 单请求（巨型 42 万 token）不再 OOM（此前 `full_available_size=3584` 但 host 有 1101312 可驱逐）。

### Phase 2 — 两端几何/索引域一致（消灭域不匹配）
- decode 侧坑 B 已修（`decode.py:1887` hisparse 下 page_size=64）。补 prefill 侧：
  - prefill `kv_to_page_indices` 用 `hisparse_page_size`（压缩页 64），与 decode 一致。
  - DCP 拓扑下（dcp=4）确认 `full_loc//256 == compressed_loc//64` 巧合对齐是否保真；不保真则统一两端转换。
- 关键文件：`prefill.py`、`decode.py:1684-1692`（坑 A：`kv_indices = dst_kv_indices[:origin_input_len - prefix_len]` 用 full 域长度切片压缩域数组，radix 命中丢前缀页）。
- **验证**：两端 `[DCP-PD-IDX] page_indices` 数量/值对齐，mooncake 传输不截断。

### Phase 3 — 准入预算纳入 host 池（消灭"假 OOM"）
- hisparse 准入预算（`decode.py:1206` / prefill 对应）纳入 host 池容量。
- `_swa_aware_allocatable_token_budgets` 的 SWA 分支与 host 池统一（explore 根因 B：当前两套账，SWA 池紧张时准入放行但 `new_pages_available` 失败）。
- **验证**：高并发下 `[PD-PREALLOC-KV-FULL]`=0。

### Phase 4 — bs=64 不降级（只缩 device 池，不是降 bs）
- hisparse device 池应只放压缩 topk，**缩小 device 逻辑池**给 bs=64 verify graph 留显存（hisparse 设计本意）。
- 调 `hisparse_config` 的 `device_buffer_size` / KV 池 size 分配（`mem_cache/kv_cache_configurator.py` hisparse 分支），让 device 池小、host 池大。
- **验证**：bs=64 verify graph capture 不 OOM（此前 avail_mem=24GB 不够）。
- ⚠️ 关键：**绝不降低 `--cuda-graph-max-bs-decode`**，而是缩 device KV 池。

### Phase 5 — radix × hisparse 共存（已解除 assert，补 host-only 命中）
- 已解除 `assert disable_radix_cache`（hisparse_hook.py）+ `assert prefix_len==0`（decode.py）。
- 补**完整 host-only 语义**：radix 命中时 device L1 永不命中（`l1_prefix_len=0`），前缀从 host `load_back` 走 sparse swap 而非当 device 完整前缀。
- 关键文件：`mem_cache/unified_radix_cache.py`（`_match_post_processor` 的 `best_match_device_value_len` 在 hisparse 下恒 0）、`mem_cache/unified_cache_components/full_component.py`。
- **验证**：radix 命中的重复请求，前缀正确复用 host KV，不丢页、不乱码。

### Phase 6 — decode 端其余根因（握手/资源释放）
- explore 根因 C（握手超时）：hisparse prefill 启动慢致首批 15s 握手失败——Phase 1 修 prefill 后自然缓解；若仍慢则延长 `_max_ensure_retries`。
- explore 根因 D（abort 不释放 host 页/device buffer，`scheduler.py:4153` TODO）——补 abort 时资源释放 `hisparse_coordinator.request_finished`。
- 关键文件：`scheduler.py`、`decode.py`。

### Phase 7 — 端到端验证（收口）
冷热两轮：
1. e2e 短请求（握手/域通过）。
2. **case50 四连过**（RPM 120，50 请求，观察乱码/崩/超时/握手失败）。
3. **bs=64** 下重复验证。
4. 记录两端 KV 容量（prefill host 池 + decode host 池实际值）。
**判据**：握手失败=0、传输失败=0、乱码=0、case50 到 4/4 clean（对照非 hisparse 基线 ok=42/50，grammar-400 为已知确定性失败）。

---

## 三、风险与诚实标注

| 项 | 把握 | 说明 |
|---|---|---|
| Phase 1（prefill host 池） | 中 | 核心缺口，`prefill.py` 无 hisparse 分支，真正加代码；正确性仅次于"域一致"。 |
| Phase 2（域一致） | 中 | explore 已定位窄化点（坑 A/B），DCP 下巧合对齐是否保真待实证。 |
| Phase 4（bs64 不降） | 中 | hisparse device 池缩小的"正确"配置需实测。 |
| Phase 5（radix host-only） | 中低 | 最复杂（radix tree device/host 双层语义 + host-only 命中路径）。 |
| 整体 | — | 完整多文件工程，须分批 commit + 每 Phase 实测，非数轮可达。 |

---

## 四、执行方式建议
- 每 Phase 独立 commit，标注"实现计划 Phase N"。
- 每 Phase 完成即实测（e2e / case50 / 无握手失败），不做半成品提交。
- 遇"域/几何"问题优先查两端 `[DCP-PD-IDX]` / `DSP-STAGE` 对齐日志（`SGLANG_DEBUG_DIAG=1`）。
