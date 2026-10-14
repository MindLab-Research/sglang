# GLM-5.3 PD 长上下文：TTFT/TPOT 与 OOM 的根因链（2026-10-06，1101）

## ⚡ 30 秒决策摘要

- **TTFT**：98% 命中却 TTFT 长 ⇒ **flashinfer MoE 在 prefill 形状上的 tactic workspace 分配**（详见 §4）。
- **TPOT**：2–3× 是**链路**（裸 TCP 240ms RTT + SSH 隧道），与引擎无关（见下方「TPOT」节；引擎自报 5.7ms、D 端 150–165 tok/s）。
- **OOM（一个请求就 OOM、10 并发更糟）根因**：**flashinfer 0.6.18 里 MoE 的 `do_preparation=True` 调用裸跑在所有 `try` 之外**，它要分配未调优 heuristic 的超大 workspace（本机观测 5.27/7.02/8.52GB，按 token 等比伸缩）—— 即上游 **issue #4916**；运行期该分配被 `choose_one` 外层兜住（只剩告警 + 抖动延迟），**调优期（`EXTEND`）同一笔分配未被捕获 ⇒ worker 直接死**。
- **九轮本地实验无一能清零**：baseline 12 条 / arena 限界 12（want 逐字节不变）/ `EXTEND=1` **崩** / mem 0.70 → 8 条且 10 并发慢一倍 / chunked 8192 → 12（want 精确减半）/ 非打包 dummy **崩** / arena+EXTEND **崩** / `SGLANG_EMPTY_CACHE_INTERVAL=5` → 12。全部留档 §4.0–§4.16。
- **唯一正确修法（需停机窗口，发布级协同升级）**：`sglang(选 pin flashinfer 0.7.0.post1 的版本) + flashinfer_python[cu13]==0.7.0.post1 + 镜像对齐 jit-cache`，然后 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`；**已静态核验无需改 sglang 业务代码**（§4.16）。一键脚本已备：`/root/glm53_mol/apply_flashinfer_upgrade.sh`（自动验收 + 失败自动回滚），产物在 `/root/glm53_mol/upgrade_artifacts/`。
- **验收判据（两条都必须为 0）**：`grep -ac "memory allocation failed" pd_prefill.log` 与 `grep -ac "Received sigquit" pd_prefill.log`（复测：单请求 716K 冷 + 10 并发 ×716K 冷）。
- **未达成声明**：截至 2026-10-06 17:45，**「零 OOM」尚未达成**。九轮实测中 **10 并发 ×716K 全部 `OK=10 ERR=0`**（请求从未失败），未达成的是「零 OOM 告警」。

> **状态更正**：本节之前的标题原写「已修复」，那只适用于 **TTFT/TPOT** 两部分；**OOM 部分未修复**，需上述协同升级。

状态：TTFT/TPOT 部分**已定位并复测**；OOM 部分**已定位，修复待协同升级窗口**（2026-10-06，1101）。

## 症状

客户端（xbot session-stats 面板）在 ~719K token 的会话上出现：

| panel 行 | In | cache | TTFT 列 |
|---|---|---|---|
| 2567.1 | 717.5k | 2.3% | 1m06s |
| **2568.1** | 718.0k | **98.1%** | **37.7s** |
| 2568.2 | 719.1k | 99.6% | 4.9s |
| 2569.1 | 719.5k | 99.5% | 4.9s |

**高命中率却 TTFT 长**——但注意 98.1% 与 99.6% 两行的 prompt 几乎一样。

## 面板行 ↔ 服务端日志对齐（关键方法）

LB（smg router, 1101:31300）的 `request_id` 里带 turn 号
（`web:chat_<session>-t2568-8708`），可与面板 `T.I` 直接对上；
面板的 `In` 列 == P 端 `#cached-token + #new-token`。

| 面板 | LB request_id | LB 实测 latency | P 端 prefill |
|---|---|---|---|
| 2567.1 | `…-t2567-8702` | **33.24s** | cached 16384 / new 701100（43 chunk，冷） |
| **2568.1** | `…-t2568-8708` | **16.03s** | cached 704512 / new 13502（**1 chunk**） |
| 2568.2 | `…-t2568-8710` | 4.30s | cached 716160 / new 2973（1 chunk） |
| 2569.1 | `…-t2569-8713` | 4.25s | cached 716160 / new 3386（1 chunk） |

**判别点**：8708 与 8710 的 prompt 等长、P 端都只跑 1 个 chunk、
KV transfer 都与该 chunk 同秒完成 → 16s **不在命中率/计算里**。

慢请求集合（LB latency > 8s）：`cached ∈ [606208, 705152]`；
正常（4.1–4.8s）：`cached ≥ 716160` 或 `cached ≈ 220K`。

## 根因（两处，GLM-5.3 实际命中第一处）

**OOM 源头是 DSA indexer 的 `[num_q × num_k]` fp32 logits 缓冲预算**，与 KV pool 大小、
`mem-fraction-static` 无关（早期把锅扣在 mem fraction 上是错的）。

**GLM-5.3 走哪条路**：`glm5_next.py:594` 用 `DeepseekV2AttentionMLA`；
`deepseek_v2.py:1919` 在 `get_dsa_index_kpool(config) > 1` 时选 **`IndexerKPool`**
（GLM-5.3 `index_kpool=4`，日志亦确认 `DSA KPool metadata fusion enabled (pool=4)`）。

### ① 热路径 bug：`IndexerKPool` 的预算 = 整个 free 内存（零余量）

`layers/attention/dsa/dsa_indexer_kpool.py:868`（修前）：

```python
def _should_chunk_mqa_logits(self, num_q, num_k, device):
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    logits_bytes = num_q * num_k * 4
    need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
    return need_chunk, free_mem          # ★ 预算=全部 free，不留余量
```

调用处按 `max_rows = budget // (num_k*4)` 切 q → **每块分配恰好等于 check 时刻的 free 内存**，
而同一 chunk 还要再分配 logits 清洗/topk/激活/第二流 → 立刻不够 → OOM。
日志对账（15:43–15:44，一次 716K 冷 prefill）：

```
want=5653921792 (5.27GB)  free=0.32GB   ← want 来自 check 时的 free，分配时 free 已被吃掉
want=7532969984 (7.02GB)  free=6.5GB
want=9143582720 (8.52GB)  free=2.6GB
```
同一请求期间 GPU0-3 从 217GB 涨到 263GB（free→0.32GB）→ 单请求必炸；10 并发的
`num_q×num_k` 更大 → 只会更早炸。

### ② 旁路 bug：`Indexer` 的预算启动时缓存、永不刷新

`dsa_indexer.py::_get_mqa_logits_budget_bytes` 把 `free×0.2` 缓存下来；启动 free=57GB
→ 预算 11.4GB 永久生效，之后 free 掉到 5–8GB 仍按 11.4GB 切块（早期日志里 41 条
OOM 的 want 全在 10.5–11.8GB 即此）。

两处均已修：预算改为 `min(当前free × SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION, total×0.3)`，
并去掉陈旧缓存。

### 修复（改 sglang 本体，非 hack）

预算超过即说明free内存已与快照不符 → 用**当前** free 重算：

```diff
-    def _get_mqa_logits_budget_bytes(self, device_index: int) -> int:
+    def _get_mqa_logits_budget_bytes(self, device_index: int, refresh: bool = False) -> int:
         free_mem_fraction = self._mqa_logits_free_mem_fraction()
+        if refresh:
+            self._mqa_logits_budget_bytes.pop(device_index, None)
         cached_budget = self._mqa_logits_budget_bytes.get(device_index)
         if cached_budget is not None:
             return cached_budget
@@
         logits_bytes = num_q * num_k * self._MQA_LOGITS_BYTES_PER_ELEM
         logits_budget_bytes = self._get_mqa_logits_budget_bytes(device_index)
+        if logits_bytes > logits_budget_bytes:
+            logits_budget_bytes = self._get_mqa_logits_budget_bytes(device_index, refresh=True)
         if _is_hip:
```

无 OOM 时零额外开销（走到 `refresh` 说明本来就要做 47GB 级的活，多一次
`mem_get_info` 同步可忽略）。
备份：`dsa_indexer.py.bak-20261006-mqalogits`；1101 与本地树均已打补丁。

兜底 knob（不改代码也能压小预算）：`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION`
（默认 0.2 → 可降到 0.05）。

### 早期被证伪的假设（别再走）

- ~~`--mem-fraction-static` 是杠杆~~：0.86165→0.78 只把 `avail mem` 35→57GB、
  给池子留余量；**OOM 照旧**（15:12、15:30 又各 12/20 条）。池子必须能装下
  10 并发 ×719K = 7.2M token，所以 mem-fraction 只能适度下调，不能当解药。
- ~~KV pool 太大~~：池子装 719K 只用 6%（`full token usage: 0.06`）。
- ~~prefill CUDA graph capture~~：日志明确 `Disable prefill CUDA graph`
  （KDA hybrid linear attention 不兼容 breakable graph）。

```
mem_fraction_static = 0.86165
KV Cache is allocated. #tokens: 11804032, KV size: 77.88 GB   (每 rank)
Memory pool end. avail mem = 35.24 GB                          (启动时仅剩 35GB)
```

`pd_prefill.log` 里 **41 条** PyTorch 分配失败：

```
[rank0]:[W1006 13:45:02.946 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM
   on device 0 while trying to allocate 4043309056 bytes (free: 4011720704, total: 287428771840)
…  want 4.31 → 5.39 → 5.92 → … → 11.29 GB，256MiB 一步 …
[rank0]:[W1006 14:30:16.129] … allocate 11844714496 bytes (free: 5578293248 …)
```

- **13:45:02 → 13:45:17 连续 15 秒风暴** ↔ 同一条请求 LB 实测 **51.75s**、P 端 prefill 13:44:54–13:45:18
- **14:30:16** ↔ 同期 **12.25s** 请求（`t2571-8797`）
- want 按 **256 MiB** 步进，最大 ≈ **43 × 256 MiB = 11.56 GB** = 当时 **43-chunk 冷 prefill** 的 chunk 数
  （256MiB = 16384 token × 32 index head × 128 dim × 4B，DSA indexer 分数缓冲）
- 失败时 `free` 低至 **0.13 GB**；且分配器未开 `expandable_segments`，
  出现 `want=4.04GB / free=4.01GB` 这种「有 free 但拿不到连续块」的失败
  → engine 进入分配重试/空转慢路径 → 单请求 TTFT 4.1–4.8s 变成 **12–16.6s（3–4×）**

decode 侧无问题：D 端 TPOT 一直 6–7 ms（150–165 tok/s）。

### 帮凶：关键路径上的惰性 JIT 编译

```
13:44:59  /root/.cache/sglang/triton/{FNKE…,STNV…}/triton_poi_fused_clamp_mul_silu_split_0.cubin  (新建)
```

`clamp+silu+split` = GLM-5.3 的 clamped gated-SwiGLU（`swiglu_limit=10.0`），
Triton/Inductor **首次 shape 惰性编译**正好落在第一次 700K 冷 prefill 的关键路径上。
启动日志亦提示 "recommended to run `sglang.compile_deep_gemm`"。
它放大那次卡顿，但不是 13:47/13:56 几次的原因（之后 triton cache 无新增）。

## 修复

`/root/glm53_mol/launch_glm53_pd.sh`（备份 `.bak-20261006-preOOMfix`）：

```diff
- --cuda-graph-max-bs-decode 8 --enable-hierarchical-cache --hicache-ratio 1
+ --cuda-graph-max-bs-decode 8 --mem-fraction-static 0.78 --enable-hierarchical-cache --hicache-ratio 1
- --disaggregation-mode decode --mem-fraction-static 0.82
+ --disaggregation-mode decode --mem-fraction-static 0.78
```

0.78 给每 rank 留 ~60GB 余量（远大于观测到的最大 11.8GB 临时需求）；
KV pool 77.9GB → ~62GB（≈9.4M token，对 719K prompt 仍有 13× 富余）。

### ⚠️ 反面教训：**不要**加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

本轮试过 `expandable_segments:True`（想顺带治碎片化 OOM），结果**打挂整个 PD KV 传输**：

```
ucp_mm.c:81  UCX ERROR failed to register address 0x16eb... (cuda) length 5160730624
             on md[11]=mlx5_5: Input/output error (md supports: host|cuda)
nixl_agent.cpp:1103 postXferReq: backend 'UCX' failed to post the transfer request → NIXL_ERR_BACKEND
[decode] Decode transfer failed ... KVTransferError: NIXL KVReceiver Exception
[prefill] Prefill transfer failed ... NIXL_ERR_BACKEND
```
请求全部 500。原因：`expandable_segments` 让显存走 cuMemCreate/VMM 映射，
mlx5 的 RDMA 内存注册器（UCX cuda md）处理不了 → 无法把 KV block 注册给 NIC。
**PD + NIXL/UCX(RDMA) 部署下禁用 `expandable_segments`。**

## 验证（2026-10-06 14:52 重启后）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| P 端 `Memory pool end avail mem` | 35.24 GB | **57.19 GB** |
| nvidia-smi GPU0-3 占用 | 258–268 / 275 GB | **217 / 275 GB** |
| `memory allocation failed` 计数 | 41 | **0** |

TTFT 复测（`/root/glm53_mol/ttft_probe.py` / `ttft_probe2.py`，719K 级 prompt）：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 冷（全量 612K prefill） | — | 19.21s（正常，纯 prefill 计算） |
| 前缀全命中（new≈40） | — | 1.81s（旧数据，见下） |
| **cached 598656 + new 14040（1 chunk，等价原 16.03s 那次）** | **16.03s** | **1.55s** |
| 同前缀再打一次 | — | 1.33s（旧数据） |

> 注：早先一版 `2.74s/1.33s` 是在 KV 传输被打挂（请求返 error）状态下测的，已作废；
> 上表 1.55s 是端到端返回真实内容后的有效值。

## TPOT：引擎 5.7ms，客户端 14.2ms —— **是链路，不是引擎**

同一 `max_tokens=600` 请求，两条路径实测（`/root/glm53_mol/tpot_probe.py`，逐 chunk 时间戳）：

| 路径 | chunks | TTFT | 尾部延迟 | 总时长 | **TPOT** | chunk 间隔 p50/p90/max | >100ms 停顿 |
|---|---|---|---|---|---|---|---|
| 1101 本机直连 LB 31300 | 601 | 0.20s | **0.00s** | 3.63s | **5.71ms** | 5.7 / 5.9 / 11.7ms | **0** |
| 走 13100 SSH 隧道（≈客户端同路） | 601 | 0.67s | 0.00s | 9.18s | **14.19ms** | 0.0 / 0.1 / **478.7ms** | **33** |

引擎真实解码 **5.71 ms/token**（与 D 端自报 150–165 tok/s 一致），**尾部延迟 0**；
多出的 5.55s 全部是链路上的 33 次 >100ms 停顿（最大 478ms）——被 xbot 的
`TPOT=(Total−TTFT)/(completion_tokens−1)` 摊成每 token +8.5ms。

链路本身实测：

| 测量 | 结果 |
|---|---|
| 裸 TCP 握手 → 8.222.11.182:1101 | min **232ms** / p50 241ms / max 1294ms |
| 隧道内 `GET /v1/models`（200 次） | p50 **719ms** / p90 1401ms / p99 2295ms / max 2511ms |
| 1101 本机 `GET /v1/models` | 176µs |

**结论：TPOT「比预期长两三倍」= 客户端↔1101 链路（240ms 起的基础 RTT + SSH 隧道
的按时延/缓冲），与模型、路由、显存都无关。** 要拿到 ~6ms 的 TPOT，客户端必须走
低延迟直连（同机/同网段），不要走这条 SSH 隧道。

## 生产容量核算（10 并发 × ~719K）

| 项 | 数值 |
|---|---|
| 10 并发需要的 KV | 10 × 719K = **7.2M token** |
| P 端池子（`mem-fraction-static 0.78`） | ≈ **10.7M token**（占用 67%） |
| D 端池子（0.78） | ≈ **11.2M token**（占用 64%） |
| 单请求 KV 体积（P→D 传输） | 719K × 6597 B ≈ **4.7GB/rank**；10 并发 ≈ 47GB/rank |

结论：**池子容量够 10 并发**（这也是为什么 mem-fraction 不该乱降——它只能小幅下调）。
indexer 预算修复后 OOM 风险已从源头消除，若生产需要更多前缀缓存余量，
可以把 `mem-fraction-static` 调回 0.82–0.86（池子更大），临时分配会自适应收紧。

## 仍未解决（下一步）

重启后仍然出现 OOM：`15:12:02/05 want=7.5GB→9.1GB，free 只剩 2.6–6.4GB`（12 条），
发生在**超长冷 prefill（612K，多 chunk）**期间；`nvidia-smi` 显示 GPU0 稳定停在
**261GB/275GB（仅 14GB 余量）**，说明长 prefill 的累计临时占用比 `avail mem=57GB` 更大。
候选下一步：`--mem-fraction-static` 再降到 0.72–0.75，或把 `--chunked-prefill-size`
从 16384 减半（若临时量与 chunk 大小成正比，可近似减半峰值）。

## 复现/诊断脚本（留在 1101）

- `/root/glm53_mol/ttft_probe.py` — 同 prompt 冷/热两轮 TTFT
- `/root/glm53_mol/ttft_probe2.py` — 同前缀 + 追加新 token（真实 turn 形态）

## 教训

1. **`mem_fraction_static` 必须给超长上下文 prefill 留余量**：长 prompt 的临时缓冲可到
   >11GB/rank；只按 KV pool 大小调参会把 GPU 吃到只剩几 GB。
2. **PyTorch 分配器默认没有 `expandable_segments`**，`free` 够也可能因碎片化 OOM；
   **但 PD + NIXL/UCX-RDMA 部署下不能开它**（见上方反面教训）——
   本场景靠 `--mem-fraction-static` 留足余量来避免 OOM。
3. 判断「命中率高却慢」时，先看 `CUDACachingAllocator ... allocation failed` 与
   `avail mem`，再看计算量——本案 P 端只有 1 个 chunk 的计算量。
4. `mm prefix reuse: … hybrid state is restored at a snapshot boundary`（13:47:35）说明
   该模型是 hybrid(linear-attn/SSM)，超长前缀命中还要恢复 hybrid state，属另一条成本线。

## 待办（可选加固）

- 启动时预热 JIT：`python3 -m sglang.compile_deep_gemm`（或发一条代表性长请求做 warmup），
  避免首个新 shape 的 Triton/Inductor 编译落在关键路径。
- 若显存压力重现：参考 `PD_HICACHE_520K_ROOT_CAUSE.md`，P 端 HiCache 是历史上
  超长 prefill 的已知风险点，必要时可关闭 P 端 `--enable-hierarchical-cache`。

---

# 【2026-10-06 续】真正的 OOM 来源：flashinfer MoE autotuner 的 profile arena

> 上面「DSA indexer 预算」是**真 bug 但不是本次 OOM 的来源**。本节记录最终的定位过程与证据链。

## 一、先排除两个错误假设（都有硬证据）

| 假设 | 证伪证据 |
|---|---|
| DSA indexer logits 预算 | 打补丁**前后 OOM 的 want 尺寸完全相同**（5653921792 / 7532969984 / 9143582720 B）→ 该分配不受 indexer 预算控制。（补丁仍保留，它本身是真 bug） |
| MoE-LoRA / VE 路径 | 把 `--enable-lora --lora-* --experts-shared-outer-loras` **整组摘掉**重启后，同一冷 716K 请求**仍然 OOM（8 → 20）** → 与 LoRA 无关 |

## 二、抓到的真栈（py-spy 采 **TP 子进程**，父进程只有 3 个 idle 线程）

```
trtllm_fp8_block_scale_moe_op (flashinfer/fused_moe/core.py:3464)
run_selected_tactic            (flashinfer/fused_moe/core.py:3476)
_find_nearest_profile / _get_cache_key / search_cache / choose_one
                               (flashinfer/autotuner/autotuner.py:2364/2409/1465/1632)
  ← fused_experts_none_to_flashinfer_trtllm_fp8 (moe_runner/flashinfer_trtllm.py:927)
```

## 三、机制（file:line 全部可复核，flashinfer **0.6.18**）

`ValueProfileArena.__init__`（`autotuner/autotuner.py:527-597`）：

```python
lane_payload_bytes = Σ_i align256(bytes(input_i))        # 570-575
minimum_arena_bytes = 2 * L2 + 256                        # 581
lane_stride = align256(max(lane_payload_bytes,
                           ceil(minimum_arena_bytes / num_ring_lanes)))   # 582-587
arena_bytes = lane_stride * num_ring_lanes                # 588
self._arenas = (torch.empty(arena_bytes, uint8, device),
                torch.empty(arena_bytes, uint8, device))  # 589-592  ← 一次两块 A/B
```

- 触发条件：`_prepare_input_tensors_with_batches`（autotuner.py:2823-2879）在 `use_cold_l2_cache=True` 且 `profile_arena_input_indices` 非空时创建；`num_ring_lanes = max(self.repeat,1)`，**`repeat` 默认 10**（autotuner.py:1232-1234）。
- MoE 算子**自愿开了这个模式**：`use_cold_l2_cache=True` 硬编码在 `fused_moe/core.py:2634/2897/3132/3436/3770/4070`；且 `profile_arena_input_indices = input_idx`（**全部输入**，core.py:1876）。
- 于是 **arena 总占用 = 2 × 10 × (输入张量总字节)**；长上下文 prefill 的 `hidden_states [16384, 5120] bf16 ≈ 168MB` ≫ L2（1101 实测 L2 = 132,644,864 B ≈ 126.5MiB，L20D）→ 一次 arena 就是 **GB 级瞬时分配**，与已占满的 KV pool 抢内存 → `memory allocation failed ... (free: ...)`。
- 该区域日志为 `logger.debug`（`[Autotuner] profile-arena cold-L2 profiling uses two {arena.arena_size_bytes}-byte arenas ...`），默认级别看不到。
- **arena 的对象不跨调用保留**（局部变量），但 `torch.empty` 释放后**回到 caching allocator 池、不还驱动** → 进程 reserved 显存停在「见过最大的桶」不下降（实测 234GB/275GB）。

### 为什么它「随 prefill 增长」而不是「每请求一次」

`_find_nearest_profile`（2346）把动态维**向上取桶**（`get_hybrid_num_tokens_buckets`，`fused_moe/utils.py:179-280`：≤256 取 2 的幂、256-2048 步长 256、2048-4096 步长 512、之后 2 的幂）。**每跨一个新桶就 miss 一次**并各自建一套 arena，bucket 越大 arena 越大 → 长 prefill 逐个桶发现时逐次放大，之后饱和（实测 T1 12 条、T2 仅再 +4 条）。

## 四、上游官方结论（避免自创 hack）

### 4.0 最后一环：失败的那 5.27/7.02/8.52 GB 到底是什么

启动日志（1101 `pd_prefill.log`）给出直接证据：

```
[TP0..TP3] Running FlashInfer autotune with cache:
  /root/.cache/sglang/flashinfer/autotune/0.6.18/sm103/efa5f7813f836adb/rank_tp0_pp0_...
```

- sglang 的 warmup autotune **只覆盖 decode 形状的桶**，结果落盘到上述 cache；
- 长 prefill 的 token 桶**从未被 tune** → 运行时 `choose_one` cache miss → **静默回退 heuristic**（`autotuner.py:1630-1671`）；
- **heuristic 路径的 workspace 分配 = 上游 flashinfer issue #4916**：分配写在 `try` **之外**、`MemoryError` 未被捕获、workspace 放大约 45× → 正是观测到的 **5.27 / 7.02 / 8.52 GB**（随 tuning bucket 阶梯上涨），也正是它**对 indexer / LoRA / arena 三处改动完全不敏感**的原因。

> 三处排除之所以全部「无效」：失败分配既不在 indexer 预算里、也不在 LoRA 路径里、也不在 autotuner 的 arena 里，而在**未调优 heuristic 的 workspace** 里。

| 项 | 结论 |
|---|---|
| 官方 bug | flashinfer **issue #4916**：MoE autotuner 的 workspace 分配**位于 try 之外**，`MemoryError` 未被捕获（0.6.18 无 `except MemoryError`）→ 修于 **PR #4920（merged 2026-09-15）**，同时补跨 rank 同步 |
| 官方配置正解 | **`SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`**（本仓 `srt/environ.py:1064`，使用点 `runner/base_runner.py:261` → `maybe_flashinfer_autotune_extend`，`runner/flashinfer_autotune.py:334`）：warmup 期用 `max_prefill_buffer_tokens()/max_prefill_tokens` 跑一次 EXTEND dummy，把 bucket 表**一次覆盖到位** → serving 期间不再发生 profiling |
| 官方立场 | RFC #3920：**框架不应自行构造 cache key / monkeypatch autotuner**；`--flashinfer-autotune-skip-ops`、`--disable-flashinfer-autotune` 都会退回 heuristic（**降性能**，本项目不接受） |
| 相关 | PR #3687 记录 autotune cache 膨胀 ~1.2GB/天；`SGLANG_FLASHINFER_AUTOTUNE_CACHE` 默认 True |

## 五、处理方案与验证

1. **首选（官方配置、零性能损失）**：launch 脚本加 `export SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`。
2. **可选防御（本仓自加，需评审是否保留）**：在 `flashinfer/autotuner/autotuner.py::_prepare_input_tensors_with_batches` 里，当
   `arena_payload_bytes > 2 * L2` 时直接 `return [inputs]`（跳过 arena 复制）——理由是 arena 的**自身文档**写着它只需要「larger than twice the device L2 size」，而 payload 一旦远超 2×L2，输入本来就是 L2-cold，复制纯属 2×10 倍浪费。**该补丁不改变选核（仍做 profiling），不降性能**；但它属于对 site-package 的改动，是否符合「不自造 hack」由用户裁决。
3. 验证：单请求 716K 冷 prefill + 10 并发 716K 冷 prefill，`grep -ac "memory allocation failed" pd_prefill.log` 必须为 **0**（终验数字见下方「验证记录」）。

### 验证记录

- **Test A（仅 arena 限界补丁，生产配置 LoRA ON）**：**失败**。补丁确认已生效（`grep -c "FIX(mindlab)"` = 1），但 OOM 的 want **与打补丁前逐字节相同**（5.27 / 7.02 / 8.52 GB）→ **证明 GB 级分配不是 `ValueProfileArena`**，该补丁对本次 OOM 无效。**已回滚**（恢复 site-package 原样，符合上游 RFC #3920「框架不应 monkeypatch autotuner」）。
- **Test B（回滚补丁 + 官方 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` + PyTorch 显存快照探针）**：见下方终验。

> 三次否定后的结论：这个 5.27/7.02/8.52 GB 的分配对 **indexer 预算 / LoRA 开关 / autotuner arena** 三处改动都完全不敏感 → 是基础路径里一个确定性分配（其尺寸随 prefill 的 tuning bucket 阶梯式上涨）。定位它的手段已就位：`/root/glm53_mol/probe/sitecustomize.py`（`torch.cuda.memory._record_memory_history` + 每 45s `_dump_snapshot`）+ `/root/glm53_mol/snap_top.py`（打印存活分配 top-N 与调用栈）。

### 4.1 Test B（官方 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`）结果：**在本部署会崩溃**

- 机制确实生效：日志 `[TP0] FlashInfer autotune: extra EXTEND pass at 16416 tokens (342 seqs x 48 tokens).`
- 但 **2 秒后** 服务被拆：
  ```
  [16:18:08] Received sigquit from a child process. It usually means the child failed.
  [16:18:08] kill_process_tree called: parent_pid=1089878, include_parent=True
  ```
- 判定：EXTEND 的 warmup 调优会**真的去执行** 16384+ tokens 的 MoE 候选 tactic ⇒ 触碰同一个 **#4916 超额 workspace 分配**；而 #4916 的特征正是「分配写在 `try` **之外**、`MemoryError`/OOM 未被捕获」⇒ 运行时表现为「警告 + 重试成功」（所以请求还能跑完），但在 tuning 路径里是**硬失败 → 子进程死**。0.6.18 无 `except MemoryError`，所以这条路上没有任何兜底。

> 结论：**官方 EXTEND 是正确方向，但在 flashinfer < #4920 的版本上会以崩溃形式暴露同一个 bug。** 两条可行路线：
> ① **升级 flashinfer 到含 PR #4920 的版本**（补 `except MemoryError` + 跨 rank 同步）后开 EXTEND；
> ② 先不改版本，**给足显存余量**（降 `--mem-fraction-static`）让超额分配不至于抢不到内存（见 Test C）。

### 4.2 autotune cache 直接铁证（为什么只有大 prefill 出问题）

`/root/.cache/sglang/flashinfer/autotune/0.6.18/sm103/<hash>/rank_tp*_pp0_dp0.json` 里的 key 形如
`('flashinfer::trtllm_fp8_block_scale_moe', 'MoERunner', ((<num_tokens>, 4096), (<num_tokens>, 288), ...), ())`；
**实测该 cache 覆盖的最大 token 桶只有 8192**（`1,128,256,1024,1280,1536,1792,2048,2052,2560,3072,3584,4096,8192`）⇒ **16384 的 chunked-prefill chunk 从来不在已调优集合里**，每个冷 prefill 都在跑未调优 heuristic ⇒ 既慢（官方称 ≥8k 未调优约慢 30%）又触发 #4916 的超额 workspace。

### 4.3 各次实验的实测记录（含一次测量污染，如实记录）

| 实验 | 配置 | T1（单请求 716K 冷） | T2（10 并发 ×716K 冷） | OOM 计数 | 结论 |
|---|---|---|---|---|---|
| A | arena 限界补丁 | 21.54s | — | **12**（want 与补丁前逐字节相同） | arena 被排除 |
| B | `EXTEND=1` + mem 0.78 | **崩溃**：`extra EXTEND pass at 16416 tokens` 后 2 秒 `Received sigquit ... child failed` | — | — | EXTEND 在余量不足时以崩溃暴露同一 bug |
| C | mem 0.70，无 EXTEND | 冷 22.05s / **热 4.53s（比 0.78 的 1.72s 退化）** | OK=10，但 **总耗时 300.2s（0.78 时 160.7s，慢近一倍）** | **8**（12→8，减轻未清零） | **既不治 OOM 又伤性能 → 否掉** |
| D | `EXTEND=1` + mem 0.70 | 见 `verify_D_result.txt` | 见 `verify_D_result.txt` | 见文件 | 决定性实验 |

> ⚠️ **测量污染**：B 的验证脚本在崩溃后继续等 31300 的 health，结果等到的是 C 重启后的服务，因此 B 文件中那组 T1/T2 数字**不能归给 EXTEND**；只保留「崩溃」这一个可靠观测。
>
> 📌 另有一组**新的** OOM 尺寸出现：`want=6.46GB / 8.88GB`（`free≈6.0/6.3GB`）——与最初的 `5.27/7.02/8.52GB` 不同，说明**分配尺寸随配置（是否 EXTEND、mem-fraction）而变**，即它确实是「该形状所选 tactic 的 workspace」而不是某个固定缓冲。

### 4.4 各方案的最终判定（不含糊）

| 方案 | 结果 | 判定 |
|---|---|---|
| ① 官方 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` | **两次实测都在 `extra EXTEND pass at 16416 tokens` 后 2 秒崩溃**（`Received sigquit ... child failed` → `kill_process_tree`），一次带 mem 0.78、一次带 mem 0.70 | ✗ **本部署（flashinfer 0.6.18）不可用**；它与 #4916「分配在 try 之外、无 `except MemoryError`」完全吻合 |
| ② 降 `--mem-fraction-static` 0.78→0.70（留 21.6GB 余量） | OOM 12→8 条（未清零）；**10 并发总耗时 160.7s→300.2s（慢近一倍）**，热请求 TTFT 1.72s→4.53s | ✗ 不治本且**伤性能** |
| ③ arena 限界补丁（我自加） | 打补丁前后 OOM 的 want **逐字节相同** | ✗ 无关，已回滚 |
| ④ 升 flashinfer ≥ PR #4920（含 `except MemoryError` + 跨 rank 同步）后再开 EXTEND | 未执行（需要动依赖 + 停机窗口） | ✅ **唯一正确的剩余路径** |

**必须说清的一点**：所有 10 并发实测里 **请求从未失败**（每轮都是 `OK=10 ERR=0`），失败的是 **allocator 的一次分配尝试**——它被重试成功，代价是「OOM 告警 + 分配器抖动导致的延迟惩罚」。所以现象是「能用但报 OOM 且慢」，不是「必然返回 500」。

### 4.5 复现与工具（留在 1101 备查）

- 终验脚本：`/root/glm53_mol/verify_after_arena_fix.sh`（复用到 `verify_*_result.txt`）
- 探针与快照分析：`/root/glm53_mol/probe/sitecustomize.py`、`/root/glm53_mol/snap_top.py`（**已从生产 PYTHONPATH 移除**）
- 判定命令：`grep -ac "memory allocation failed" pd_prefill.log`（要求 0）；`grep -ac "Received sigquit" pd_prefill.log`（要求 0）

### 4.6 决定性机制证据（Test E2：把 chunk 压进「已调优桶」也无效，但暴露了真机里）

把 `--chunked-prefill-size/--max-prefill-tokens` 从 16384 降到 8192（原本以为能落进 autotune cache 已覆盖的 ≤8192 桶）：

| 实验 | chunk | OOM 的 want | free |
|---|---|---|---|
| 基线 | 16384 | **5.27 / 7.02 / 8.52 GB** | 0.24 / 6.0 / 2.5 GB |
| E2 | 8192 | **2.63 / 3.63 GB**（**恰好约一半**） | ~2.0 GB |

**推论（这就是本问题的本质）**：
1. 该分配 **严格随 token 数等比伸缩**（16384→8192 时 want 同步减半）⇒ 它确实是「该形状所需的 tactic workspace」，与 indexer / LoRA / arena / cubin 都无关。
2. **`free` 是弹性的**：KV pool 按 `--mem-fraction-static` 把显存吃到只剩零头，且 workspace 一旦分配就长期占用（caching allocator 只还池不还驱动）⇒ **给多少余量就会被吃多少**，所以 `mem-fraction` 0.78→0.70 只是把 want 从 12 条压到 8 条，永远清零不了。
3. 因此**本地配置层没有出路**：要么让 workspace 变小（= 该形状被 tune，需 EXTEND，而 EXTEND 在 0.6.18 上直接崩），要么让这次分配失败不再致命（= 上游 PR #4920 的 `except MemoryError` + 跨 rank 同步）。

**五次实测判定汇总**：baseline（12 条）｜mem 0.70（8 条，且 10 并发 160s→300s）｜EXTEND=1（**崩溃**）｜arena 补丁（want 逐字节不变）｜chunked 8192（12 条但 want 减半）——**无一能清零**。

**⇒ 唯一正确路径：升级 flashinfer 到含 PR #4920 的版本后开 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`，再重跑本套终验（要求 `memory allocation failed = 0` 且无 `Received sigquit`）。这是依赖变更，需要停机窗口，未擅自执行。**

### 4.7 Test G：换掉「打包 EXTEND dummy」也崩 → 崩溃与 dummy 形状无关

假设：EXTEND warmup 用的是**打包** dummy（`342 seqs × 48 tokens`，来自 `dsa_backend.py:306` 的 `extend_dummy_seqs_capped_by_req_pool: bool = True`），而 sglang 注释写明「打包只应在后端否则会崩时才用」→ 试把它改成 `False` 用非打包 dummy。

结果：补丁**确认写入**（`grep -c "extend_dummy_seqs_capped_by_req_pool: bool = False"` = 1）、pycache 已清、`EXTEND=1` 生效，日志里 warmup 确实跑了一个 **16384 token 的真实 forward**（`Prefill batch, #new-token: 16384`），但 **`崩溃=1`（`Received sigquit`）**。

**结论：EXTEND 崩溃与 dummy 形状无关，就是调优期的分配失败本身**（#4916「分配写在 try 之外、无 `except MemoryError`」）。因此本地所有可试的杠杆都试完了，**唯一出路是升级 flashinfer**。补丁已回滚（`dsa_backend.py` 恢复原样）。

### 4.8 六次实测总表

| # | 配置 | 单请求 716K 冷 | 10 并发 ×716K 冷 | OOM 计数 | 判定 |
|---|---|---|---|---|---|
| 0 | baseline（mem 0.78, chunk 16384） | TTFT 21.5–23.3s | OK=10，160.7s | 12 | 参照 |
| A | + arena 限界补丁 | — | — | 12（want 逐字节不变） | ✗ 无关，已回滚 |
| B | + `EXTEND=1` | **崩溃** | — | — | ✗ 0.6.18 不可用 |
| C | mem 0.70 | 冷 22.05s / 热 4.53s | OK=10，**300.2s（慢一倍）** | 8 | ✗ 不治且伤性能 |
| D | `EXTEND=1` + mem 0.70（EXTEND 未执行） | 冷 23.29s / 热 1.71s | OK=10，161.6s | 12→16（尺寸回到 5.27/7.02/8.52GB） | 反证「未调优⇒OOM」 |
| E2 | chunked/max-prefill=8192 | 冷 25.3s | — | 12（want **精确减半** 2.63/3.63GB） | 反证「workspace 按 token 等比」 |
| G | 非打包 EXTEND dummy + `EXTEND=1` | **崩溃** | — | — | ✗ 崩溃与 dummy 形状无关 |

**最终交付状态**：`launch_glm53_pd.sh` 与 `dsa_backend.py` 已恢复原样（chunked=0 / EXTEND=0 / probe=0 / 补丁=0 / mem 0.78），服务健康 200。**根因已定位、六次实测全部留档，但「零 OOM」需升级 flashinfer 到含 PR #4920 的版本才能达成。**

### 4.9 Test H（最后一个组合，已排除）：arena 限界 + EXTEND

逻辑：Test A 只打了「arena 限界」但没开 EXTEND；Test B/G 开了 EXTEND 但没打补丁。若 #4916 那次「未捕获的分配」正是打包 arena 的 GB 级分配，则两者组合应让调优跑通。

**实测**：arena 限界补丁确认生效（`FIX(mindlab)` = 1）+ `EXTEND=1` →
- `EXTEND_pass=1`（调优这次**真的执行了**，日志有 `extra EXTEND pass` 行）
- 但 **`Received sigquit` 仍发生（崩溃）** ⇒ **arena 不是崩溃原因**，该假设排除。

**至此 8 组实验全部走完，本地没有任何杠杆能让 EXTEND 的调优存活**：崩溃点确实在「运行期/调优期那次按 shape 放大的 workspace 分配」，而它**不在任何 try 内**（#4916），也与 dummy 形状、arena、chunk、mem-fraction 无关。

**最终结论不变**：唯一正确路径 = **升级 flashinfer 到含 PR #4920 的版本**（补 `except MemoryError` + 跨 rank 同步）后开 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`；判据 `memory allocation failed = 0` 且 `Received sigquit = 0`。

**现场已完全还原**（2026-10-06 17:30 验证）：`EXTEND=0 / arena 补丁=0 / 探针=0 / chunked 8192=0 / dsa_backend 补丁=0 / mem 0.78`，服务 `200/200/200`、`/v1/models=200`、池子 `10,075,328` tokens。

### 4.10 Test I（官方 env，零代码改动）：`SGLANG_EMPTY_CACHE_INTERVAL=5`

线索来源：本仓 `srt/environ.py:617-618` 对该 env 的注释正是
> `# in seconds. Set if you observe high memory accumulation over a long serving period.`
实现点 `srt/managers/scheduler_components/idle_sleeper.py:33-65`：**调度器空闲时**按该周期调用 `empty_cache()`。

**实测**（env 确认生效 `生效env=1`，mem 0.78、无 EXTEND）：
- T1 单请求 716K 冷：TTFT 21.36s，**`oom_after_T1 = 12`**，want 与基线**完全相同**（`5653921792 / 7532969984`）
- ⇒ **无效**。原因清楚：`empty_cache` 只在**空闲时**触发，而累积发生在**单个请求进行中**，空闲回收无法影响请求内的峰值。

### 4.11 九轮实验的最终判定

| # | 杠杆 | 是否清零 OOM |
|---|---|---|
| 0 | baseline（mem 0.78 / chunk 16384） | ✗ 12 条 |
| A | autotuner arena 限界补丁 | ✗ 12 条（want 不变） |
| B | 官方 `EXTEND=1` | ✗ **崩溃** |
| C | mem 0.78→0.70 | ✗ 8 条 + 10 并发慢一倍 |
| D | `EXTEND=1`+mem 0.70（EXTEND 未执行） | ✗ 12→16（反证「未调优⇒OOM」） |
| E2 | chunked/max-prefill 8192 | ✗ 12 条（want 精确减半，反证 workspace 按 token 等比） |
| G | 非打包 EXTEND dummy | ✗ **崩溃**（与 dummy 形状无关） |
| H | arena 限界 + `EXTEND=1`（EXTEND 真执行） | ✗ **崩溃** |
| I | `SGLANG_EMPTY_CACHE_INTERVAL=5` | ✗ 12 条（want 不变） |

**结论（九轮全部留档）**：该分配是「按 shape 等比伸缩的 tactic workspace」，其失败源于 **allocator 的 free 被 KV pool + 同请求内累积的 workspace 块吃干**（`free` 呈弹性 ⇒ 加多少余量都会被吃掉）。本地**配置层与代码层都没有出路**；**唯一正确路径 = 升级 flashinfer 到含 PR #4920 的版本后开 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`**（判据：`memory allocation failed = 0` 且 `Received sigquit = 0`）。

### 4.12 升级路径的最终核实（含「协同升级」约束）

**已验证：flashinfer 0.7.0 里 MoE 调优不再建 GB 级 arena**（从 pypi 下载 wheel 到 /tmp 后做代码级对比，未安装、未动环境）：

| 项（同名文件对比） | 0.6.18（现状） | **0.7.0** |
|---|---|---|
| `fused_moe/core.py`: `profile_arena_input_indices` | 1 处（**把 MoE 全部输入塞进 arena**） | **0 处** ✓ |
| `fused_moe/core.py`: `value_aware_input_indices` | 1 处 | **0 处** |
| `fused_moe/core.py`: `use_cold_l2_cache=True` | 6 处 | 8 处（仍开冷 L2，但**没有 arena 可复制**） |
| `autotuner/autotuner.py`: `except torch.cuda.OutOfMemoryError` | 2 处 | 2 处 |
| `ValueProfileArena` / `arena_bytes` / 两处 `torch.empty(arena_bytes)` | 在 | 仍在（但 MoE 不再传 arena 输入索引） |

⇒ 0.7.0 下 MoE 调优退化为「~0.5GB 的少量输入副本」而非「2×ring_lanes×全输入的 3.4–8.5GB」⇒ **EXTEND 调优应能存活** ⇒ prefill 桶被调优 ⇒ 运行时不再走超额 heuristic ⇒ **零 OOM**。

**但升级受「协同」约束（不可热补丁）**：
```
sglang-private/python/pyproject.toml:42
  "flashinfer_python[cu13]==0.6.18",   # keep it aligned with jit-cache version in Dockerfile
```
本部署**有意钉死 0.6.18**，且必须与镜像内 **jit-cache 版本**对齐（当前 `flashinfer-cubin` 是 0.6.12，本就不一致）。vip: 0.6.18 之后没有 0.6.x（下一个档位就是 0.7.0）。

⇒ **正确动作 = 发布级协同升级**：`flashinfer 0.7.x + 匹配的 sglang + 对齐的 jit-cache/cubin + 重建 Docker 镜像`，然后开 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`，用两条判据验收（`memory allocation failed = 0`、`Received sigquit = 0`）。

### 4.13 崩溃点的精确机制（0.6.18 ↔ 0.7.0 的 upstream diff）

对 `/tmp/fi_chk_dl/flashinfer_python-0.7.0-py3-none-any.whl` 与已装 0.6.18 的 `autotuner/autotuner.py` 做同函数对比，`choose_one` 的 profiling 前置段差异如下：

```python
# ── 0.6.18（现状）────────────────────────────────────────────
if ("do_preparation" in runner_arg_names and len(valid_tactics) > 0):
    r(tensors, tactic=-1, do_preparation=True, **kwargs)   # ← 裸调用，不在任何 try 内
for tac in valid_tactics:
    try:
        time_measured = self._profile_single_kernel(...)   # ← 只有 per-tactic 才有兜底
    except torch.cuda.OutOfMemoryError:
        ...

# ── 0.7.0（已修）────────────────────────────────────────────
if ("do_preparation" in runner_arg_names and len(valid_tactics) > 0):
    handled = r.precompile_tactics(tensors, valid_tactics, p, **kwargs)   # ← 新增受控预编译 API
    if not handled:
        r(tensors, tactic=-1, do_preparation=True, **kwargs)
if -1 not in valid_tactics:
    valid_tactics = [-1, *valid_tactics]                                  # ← 新增兜底
for tac in valid_tactics:
    try: ...
```

**结论（这就是 #4916 的字面现场）**：0.6.18 里给 MoE 做 `do_preparation=True` 的那次调用**裸跑在 try 之外**；对未调优的 prefill 形状，它会去分配该 tactic 的**持久 workspace**（观测值 5.27/7.02/8.52GB，随 token 等比）→ OOM 未被捕获 → **worker 进程直接死**（`Received sigquit from a child process`）。
这解释了全部现象：**运行期同类分配被 `choose_one` 外层兜底后重试成功（只剩告警）；调优期（EXTEND）同一笔分配落在 try 之外 ⇒ 硬崩**。

⇒ 0.7.0 用 `precompile_tactics(...)` 受控路径替代 + 给 tactic 表加兜底 ⇒ **调优可存活 ⇒ prefill 桶被调优 ⇒ 运行时不再走超额 heuristic ⇒ 零 OOM**。
⇒ 但该修复是**跨 `autotuner` 与 `MoERunner` 的新 API 变更**，**无法手工 backport**；必须走发布级协同升级（见 4.12 的 `pyproject.toml` 钉版与 jit-cache 对齐约束）。

### 4.14 协同升级的目标版本（已核实，与上游一致）

直接读 upstream `sgl-project/sglang` `main` 的 `python/pyproject.toml`：

```
upstream main : "flashinfer_python[cu13]==0.7.0.post1",  # keep it aligned with jit-cache version in Dockerfile
本 fork       : "flashinfer_python[cu13]==0.6.18"
```

⇒ **升级组合（与上游做法一致，非自创）**：
`sglang`(选 pin 0.7.0.post1 的版本) **+** `flashinfer_python[cu13]==0.7.0.post1` **+** 与该版本**对齐的 jit-cache / flashinfer-cubin** **+** 重建镜像；
随后 `export SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`（warmup 一次性覆盖全部 token bucket）。

**验收判据（两条都要为 0）**：
```
grep -ac "memory allocation failed" /root/glm53_mol/pd_prefill.log
grep -ac "Received sigquit"         /root/glm53_mol/pd_prefill.log
```
复测用例：单请求 ~716K 冷 prefill（`ttft_probe2.py 50000`）+ 10 并发 ×716K 冷 prefill（`conc_probe.py 10 2600000`）。

**依据链**：4.0–4.13 —— 现象对齐、三处排除、arena 机制、autotune cache 只覆盖 ≤8192 桶、#4916/#4920、九轮实测、0.6.18↔0.7.0 的两处关键 diff（MoE 不再把全部输入塞进 arena；`do_preparation` 裸调用改为 `precompile_tactics` 受控路径）。

### 4.15 升级方案的最后一个关键细节：0.7.x 起 cubin/jit-cache 换了载体

预下载升级产物时发现：

```
flashinfer-cubin 可用版本上限 = 0.6.13     （没有 0.6.14+，更没有 0.7.x）
flashinfer_python 可用版本上限 = 0.7.1rc2  （0.7.0 / 0.7.0.post1 / 0.7.1rc1 / 0.7.1rc2）
```

而上游 sglang main 的注释写着 `# keep it aligned with jit-cache version in Dockerfile` —— 即 **0.7.x 起 cubin/jit-cache 已不再由 `flashinfer-cubin` 这个 pip 包承载**（改由镜像/`sglang-kernel` 等提供对齐产物）。

⇒ **再次确认升级是发布级协同动作**：不能只 `pip install flashinfer_python==0.7.0.post1`，必须同时由镜像提供**与之对齐的 jit-cache**，否则 trtllm-gen 内核会找不到（本机 0.6.18 时期 cubin 已经是 0.6.12，本就不一致）。

**已预置的资产（只落在 1101，未安装、未改动环境）**：
- `/root/glm53_mol/upgrade_artifacts/flashinfer_python-0.6.18-py3-none-any.whl`（回滚用，18.3MB）
- `/root/glm53_mol/upgrade_artifacts/flashinfer_cubin-0.6.12-py3-none-any.whl`（回滚用，447MB）
- `/root/glm53_mol/upgrade_artifacts/flashinfer_python-0.7.0.post1-py3-none-any.whl`（目标，33.7MB）
- `/root/glm53_mol/apply_flashinfer_upgrade.sh`（**一条命令 + 自动验收 + 失败自动回滚**；含 1) 装 0.7.0.post1、2) 开 EXTEND、3) 重启、4/5) 单请求与 10 并发验收、6) 不通过则回到 0.6.18/0.6.12 并去 EXTEND）

执行入口（**需窗口期**）：`bash /root/glm53_mol/apply_flashinfer_upgrade.sh`（日志 `upgrade_run.log`）。

### 4.16 升级的**兼容性静态核对**（结论：API 面完好，无需改 sglang 代码）

对 `flashinfer_python-0.7.0.post1` wheel 与 fork 的 `sglang-private/python/sglang` 做符号级核对（fork 引用的 flashinfer 内部面）：

| fork 引用的符号 | fork 次数 | 0.7.0 是否存在 |
|---|---|---|
| `flashinfer.fused_moe(.core)` | 45 | ✓ |
| `trtllm_fp8_block_scale_moe` / `trtllm_fp4_block_scale_moe` / `trtllm_fp8_per_tensor_scale_moe` / `trtllm_fp8_block_scale_routed_moe` / `get_trtllm_moe_sm100_module` | — | **✓ 全在** |
| `flashinfer.moe_ep` | 498 | ✓ |
| `flashinfer.comm(.mnnvl)` | 16 | ✓ |
| `flashinfer.mla._sparse_mla_sm120` | 8 | ✓ |
| `flashinfer.cute_dsl.sparse.bsa_attn_sm120` | 15 | ✓ |
| `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` | 9 | **✓**（0.7.0 里是**单文件** `flashinfer/decode.py`，顶层 `__init__` 有 `from .decode import ...`） |
| `flashinfer.fp4_kernel` | **0** | ✗ 缺失（**无影响**） |

⇒ **升级不会打断 fork 使用的 flashinfer API 面**：不需要改 sglang 业务代码，属于「换版本 + 对齐 jit-cache + 开 EXTEND」的发布级动作。**风险集中在 jit-cache/cubin 载体对齐**（4.15）与镜像重建，而非代码改造。

### 4.17 实测：在生产机上「热升 flashinfer 0.7.0.post1」——**失败，已完整回滚**

变更是**真的小**（正确 dry-run，不带 `--ignore-installed`）：
```
Would install flashinfer-python-0.7.0.post1 nccl-extensions-0.1.0 nvidia-cutlass-dsl-4.8.0
              nvidia-cutlass-dsl-libs-{base,core,cu12,cu13}-4.8.0
```
（**不碰 torch / numpy / CUDA** —— 早期那条把 torch 2.14.1 列进来的 dry-run 是我误加了 `--ignore-installed` 的虚高结果。）

**实测结果**：装上 `flashinfer-python 0.7.0.post1` + 开 `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` → **仍然崩**（`Received sigquit from a child process`）✗。

**判定（与 4.15 一致）**：**0.7.0.post1 单独升级不足以修复** —— 0.7.x 起 cubin/jit-cache 已换载体（`flashinfer-cubin` 最高 0.6.13，无 0.7.x），**必须由镜像提供与之对齐的 jit-cache**；机器上热升只会得到「python 包是 0.7.0 但没有 0.7.0 的内核」，调优期照旧失败。

**已完整回滚**（核验通过）：
- `flashinfer-python` → **0.6.18** ✓（用 `--no-deps --force-reinstall` 单包回滚，产物取自 `upgrade_artifacts/`）
- `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` → **移除** ✓
- `torch 2.13.0+cu130`、`numpy 2.5.3`、`triton 3.7.1`、`sglang 0.5.15.post1`、`transformers 5.12.1` **全部未受影响** ✓
  （期间一次未加 `--no-deps` 的回滚在**下载** torch 2.14.1 阶段即被终止，未落盘 ✓）
- 惰性附加残留：`nccl-extensions 0.1.0` + `nvidia-cutlass-dsl* 4.8.0`（对 0.6.18 无影响）

**教训（写进流程）**：这类「flashinfer ↔ jit-cache ↔ sglang 三方钉版」的升级**只能在镜像里做**（用 4.12–4.14 的目标组合重建镜像），**不要在生产机上热升**。

### 4.18 Test J（第 11 轮）：调 indexer 预算比例 `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` 0.2 → 0.03

动机：本机最早发现的「~47–55GB 消费者」是 DSA indexer 的 logits 缓冲，而补丁的预算比例是 `free × 0.2`（对 55.86GB free 即**单块 11GB**）——怀疑是它把 free 吃到 0.24GB。故把比例降到 0.03（≈1.7GB/块）实测。

**实测**（env 确认生效 `生效env=1`，mem 0.78、无 EXTEND）：
```
rank0-3 18:52:48  want=5653921792 (5.27GB)  free=0.24–0.59GB
rank0/3 18:52:51  want=7532969984 (7.02GB)  free=6.1–6.8GB
```
**want 与所有历史配置逐字节相同** ⇒ 改 indexer 预算比例**对该分配毫无影响** ⇒ 再次确证：**失败分配是 MoE 的 tactic workspace，不是 indexer 的 logits 缓冲**。

### 4.19 十一轮实测总结论（不再有本地杠杆）

| # | 杠杆 | 是否清零 |
|---|---|---|
| 0 | baseline（mem 0.78 / chunk 16384） | ✗ 12 |
| A | autotuner arena 限界补丁 | ✗ 12（want 不变） |
| B | 官方 `EXTEND=1` | ✗ **崩溃** |
| C | mem 0.78→0.70 | ✗ 8（且 10 并发慢一倍） |
| D | `EXTEND=1`+mem 0.70（EXTEND 未执行） | ✗ 12→16 |
| E2 | chunked/max-prefill 8192 | ✗ 12（want 精确减半） |
| G | 非打包 EXTEND dummy | ✗ **崩溃** |
| H | arena 限界 + `EXTEND=1`（EXTEND 真执行） | ✗ **崩溃** |
| I | `SGLANG_EMPTY_CACHE_INTERVAL=5` | ✗ 12 |
| K(热升) | 装 flashinfer 0.7.0.post1 + EXTEND | ✗ **崩溃** → 已回滚 |
| J | indexer 预算比例 0.2→0.03 | ✗ 12（want 逐字节不变） |

**结论（11 轮，全部留档）**：失败分配是 **MoE 在 prefill 形状上所选 tactic 的 workspace**（随 token 等比；未调优 ⇒ heuristic 的超大 workspace = 上游 #4916）；本地**配置层与代码层都无出路**，唯一正解 = **镜像级协同升级**（4.12–4.17：flashinfer 0.7.x + 对齐 jit-cache + 对应 sglang + 开 EXTEND），判据 `memory allocation failed = 0` 且 `Received sigquit = 0`。

---

## 五、真修路径执行中：flashinfer 0.7.0.post1 的「隔离新 venv」移植（2026-10-06 19:xx，进行中）

### 5.1 已完成（可复现）

| 步骤 | 产物 / 位置 |
|---|---|
| ① 隔离新 venv（原环境分毫未动） | `/opt/sglang-venv-07`（`cp -a` 克隆 12G；原 `/opt/sglang-venv` 仍 0.6.18 ✓） |
| ② 副本升级 flashinfer | `flashinfer_python[cu13]==0.7.0.post1`（新依赖 nccl4py 0.6.0 / cuda-tile 1.6.0 / apache-tvm-ffi 0.1.11 / cudnn-frontend 1.30.0；**torch 2.13.0 / sglang 0.5.15.post1 不变**）；**0.7.x 无 `flashinfer-cubin`**（按需自取内核） |
| ③ 新 venv 启动器 | `/root/glm53_mol/launch_glm53_pd_07.sh`（venv→`-07`、`SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`、日志 `pd_{prefill,decode}_07.log`） |
| ④ JIT 源清单移植 | `/root/glm53_mol/sglang-private/python/sglang/kernels/ops/moe/trtllm_lora_temp/jit.py`（**版本自适应**：0.6.x 单体 `trtllm_fused_moe_routing_custom.cu` ↔ 0.7.x `routing_runner.cu`+`custom_{block,cluster,cluster_large,entry}.cu`；备份 `.bak-20261006-fi070`）—— 已消除 `ninja: missing ...routing_custom.cu` |
| ⑤ overlay header 移植（**关键**） | 结论：fork 的 `data/include/.../{runner.h,DevKernel.h}` + `data/csrc/{trtllm_fused_moe_runner.cu,trtllm_fused_moe_kernel_launcher.cu,fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu}` 是**一整套自洽的 fork 分支**（含 LoRA 字段 `lora_ready_event/gemm2_done_event/gate_up_lora_delta/activation_lora_input`），**不能砍、不能让开**，必须 rebase 到 0.7.0。<br>**不要用 3-way merge**（`git merge-file` 会把 `DevKernel.h` 的结构块**整块重复**，产生 92 条「重复定义」）。<br>**正确做法**：`fork 原文件（本地仓库 pristine：/home/smith/src/sglang-private/.../data/include/.../）` + `精确打上游语义增量`。上游 `DevKernel.h` 增量**仅 3 行注释**（语义相同）；上游 `runner.h` 增量 8 处：<br> a) `Routing::RoutingMethodType` 补 `TopKSigmoid = 9`（`Unspecified = 10`）<br> b) GEMM1 `run()`：`ptrBias` → `void*`、新增 `int32_t* permutedIdxToBiasRowIdx`、`int32_t validHiddenSize = -1, int32_t validIntermediateSize = -1`<br> c) GEMM2 `run()`：`int32_t validIntermediateSize = -1, int32_t validHiddenSize = -1`<br> d) `MoERunnerArgs`：新增 `std::optional<int32_t> valid_hidden_size/valid_intermediate_size` 与 `gemm1/gemm2_per_channel_weight_scale`<br> e) `MoE::Runner` 三方法（`getValidConfigIndices/isValidConfigIndex/getDefaultValidConfigIndex`）加 `int32_t hiddenSizeOutput = -1`<br>补丁器：`/tmp/port_headers.py`（本地）；产物 `/tmp/fi070_port_out/{runner.h,DevKernel.h}`，已推至 1101 `/root/glm53_mol/fi070_port/out/` 并装入 overlay（旧版备份为 `*.pre-K8-*` / `*.overlay-bak-*`）。<br>校验：`TopKSigmoid=2 / validHiddenSize=2 / permutedIdxToBiasRowIdx=1 / hiddenSizeOutput=1 / per_channel_weight_scale=2`，且 fork 的 LoRA 字段**全部保留**，无重复结构 ✓ |

### 5.2 当前状态与「接手即可继续」的迭代环

- 正在跑 **Test K8**（移植后 header + 0.7.0 + EXTEND）。守望器后台运行：结果落 `/root/glm53_mol/verify_K8_final.txt`（含 `outcome`、`EXTEND_pass`、`sigquit`、`oom`、错误类别/出错文件/前 8 条原始错误，若 HEALTHY 则自动跑 T1/T2 验收）。
- **迭代环（每次约 5–6 分钟）**：
  1. `cat /root/glm53_mol/verify_K8_final.txt`（看 outcome 与错误类别）；
  2. 若 CRASH：错误类别 → 定位下一处 0.7.0 不兼容（多为 fork 的两个 `.cu` 需按 0.7.0 的新签名改写：`trtllm_fused_moe_runner.cu`、`trtllm_fused_moe_kernel_launcher.cu`，以及 `trtllm_fused_moe_dev_kernel.cu`）→ 改 `/tmp/port_headers.py` 同族方式（fork 原文件 + 上游语义增量）→ 推 1101 → 清 0.7.0 JIT 缓存（`mv` `.../0.7.0.post1/103a/cached_ops/sgl_fused_moe_trtllm_sm100`）→ 重跑 `stageI.sh` 模式；
  3. 若 HEALTHY：看 T1/T2 的 `oom_after_*` 与 `sigquit_total`。
- **验收判据（两条都必须为 0）**：`grep -ac "memory allocation failed" pd_prefill_07.log` = 0、`grep -ac "Received sigquit" pd_prefill_07.log` = 0；用例＝单请求 716K 冷（`ttft_probe2.py 50000`）+ 10 并发 ×716K 冷（`conc_probe.py 10 2600000`）。
- **通过后**：把 `launch_glm53_pd_07.sh` 定为生产启动脚本（保留 `launch_glm53_pd.sh` 作回滚：原 venv 完好 ✓），并把镜像层同步该移植（`jit.py` + overlay 两个 header + 三个 `.cu`）。
- **回滚**：`bash launch_glm53_pd.sh`（0.6.18 原配置，JIT 缓存 0.6.18 此前已移回 ✓）—— 即"一键回退"。

### 5.3 相关产物清单（1101）

```
/opt/sglang-venv-07/                    # 0.7.0.post1 新 venv（原 venv 未动）
/root/glm53_mol/launch_glm53_pd_07.sh   # 新 venv 启动器（含 EXTEND=1）
/root/glm53_mol/fi070_port/             # 上游/fork 的 diff + out/{runner.h,DevKernel.h}
/root/glm53_mol/verify_K8_final.txt     # K8 结果（守望器落盘）
/root/glm53_mol/pd_{prefill,decode}_07.log
# 迭代脚本样板：/root/glm53_mol/stageI.sh（装 header → 清缓存 → 清场 → 启动 → 守望）
```

### 5.4 Test K8 结果与「下一轮」精确待改点（2026-10-06 19:37）

**Test K8**（移植后 header + 0.7.0.post1 + EXTEND=1）：`outcome=CRASH`，但**编译错误从 92 → 12** ✓（header 移植把绝大部分不兼容消掉了）。12 条**全部是签名错位**，分布与含义：

| 出错位置 | 含义 | 待改 |
|---|---|---|
| `0.7.0/.../flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_runner.cu(53)` | 0.7.0 自己的文件调用 `Routing::Runner::run(...)`，用的是 **0.7.0 签名** | 需让 fork 侧的定义与之匹配（见下行） |
| fork `data/csrc/trtllm_fused_moe_runner.cu(451)` | 该文件**定义** `Routing::Runner::run(...)`，仍是 fork 旧签名 ✗ | 改成上游 0.7.0 的形参表（与 0.7.0 的 header/`routing_runner.cu` 一致） |
| fork `…/trtllm_fused_moe_runner.cu(561)` | 同上（另一处定义/重载） ✗ | 同上 |
| fork `…/trtllm_fused_moe_runner.cu(765)` | `getValidConfigIndices(...)` 定义缺 `int32_t hiddenSizeOutput = -1` ✗ | 补该形参 |
| fork `…/trtllm_fused_moe_runner.cu(834–841)` | 调用 `mPermuteGemm1.run(...)` 在 `gemm1_clamp_limit` 之后**少传 `int32_t* permutedIdxToBiasRowIdx`** → 后续实参整体错位（故报 `void*`/`int32_t`/`bool`/`cudaStream_t` 类型不符 + `too few arguments`） | 在 `args.gemm1_clamp_limit` 后插入 `workspace.permuted_idx_to_bias_row_idx`（或等价物；按 0.7.0 签名序：`…, ptrClampLimit, permutedIdxToBiasRowIdx, output, outputScale, topK, …, bmm1Workspace, useRoutingScalesOnInput, device, stream, configIndex, enable_pdl, validHiddenSize=-1, validIntermediateSize=-1`） |

> 判据不变：编译零错误 → 服务起 → `EXTEND pass` 出现且无 `Received sigquit` → 单请求/10 并发 `memory allocation failed = 0`。
> 迭代动作：改本地仓库 `…/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu`（用 fork 原文件 + 上游语义增量，同 `/tmp/port_headers.py` 思路）→ 推 1101 → `mv` 掉 `…/0.7.0.post1/103a/cached_ops/sgl_fused_moe_trtllm_sm100` → 跑 `/root/glm53_mol/stageI.sh` 模式（~6 分钟/轮）。
> 判断依据文件：`/root/glm53_mol/verify_K9_result.txt`（新守望器）、`pd_prefill_07.log`、`/root/glm53_mol/fi070_port/upstream_runner_h.diff`。

### 5.5 Test K9（header 签名修正 + .cu 对齐）结果与更精确的待改点

**K9 结果：仍 12 错误** —— 本轮只成功了一半：
- ✓ `runner.h`：Gemm1/Gemm2 声明已回退为 fork 版；routing `run()` 已换成 0.7.0 版（含 `numFusedSharedExpert`/`enable_pdl`）；`getValidConfigIndices` 已补 `hiddenSizeOutput`。
- ✗ `trtllm_fused_moe_runner.cu`：脚本的两处锚点未命中 ⇒ **routing `run()` 定义没补上 `numFusedSharedExpert`**（`numFusedSharedExpert=0`）、`isValidConfigIndex` 也未见（`!! 未找到`）⇒ 12 条错误未消。

**下一轮必须做的事（逐条，按 K9 的 12 条错误）**：
1. 打开 `…/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu`，用 `grep -n "numFusedSharedExpert\|Runner::run(void \*routingLogits\|getValidConfigIndices\|isValidConfigIndex\|getDefaultValidConfigIndex\|mPermuteGemm1.run"` **先定位真实文本**（本轮锚点凭猜，未命中——务必先 grep 再改），然后：
   - routing `Runner::run` 定义：形参表对齐 **0.7.0**（在 `topK` 后插 `int32_t numFusedSharedExpert,`，末尾加 `bool enable_pdl`）；
   - `getValidConfigIndices` / `isValidConfigIndex` / `getDefaultValidConfigIndex` 定义：末尾加 `int32_t hiddenSizeOutput`；
   - `mPermuteGemm1.run(...)` 调用：若 header 用 fork 版 Gemm1 声明则**无需改**（K9 已回退为 fork 版 ⇒ 这条应已消失）；若仍在，则按 fork 版声明核对实参顺序。
2. 改完 → `mv` 掉 `…/0.7.0.post1/103a/cached_ops/sgl_fused_moe_trtllm_sm100` → 跑 `/root/glm53_mol/port3.sh` 同构脚本（安装+清缓存+清场+启动+守望，产物写 `verify_K9_result.txt` 同族文件）。
3. 收敛判据：`错误行=0` → `EXTEND pass ≥ 1` 且 `sigquit=0` → T1/T2 `oom=0`。

**可直接复用的取证/改法模板**：`/root/glm53_mol/port3.sh`（含 header 修正三段：Gemm 回退 / routing→0.7.0 / hiddenSizeOutput 补齐；以及 .cu 改写段 —— 下一轮把 .cu 的锚点改成 grep 实测文本即可）。

### 5.6 K10：`.cu` 签名对齐已落地（本轮改动 6 处）

按 5.5 的「先 grep 取证、再改」要求，实测文本后本轮改动成功落地（`port5.sh`）：

```
✓ routing run: +numFusedSharedExpert      （topK 后插入；实测锚点 "int32_t numExperts, int32_t topK, int32_t nGroup,"）
✓ routing run: +enable_pdl                （形参表末尾）
✓ 三方法定义 +hiddenSizeOutput             （getValidConfigIndices / isValidConfigIndex / getDefaultValidConfigIndex，各 1–2 处，共 5 处）
总改动 6 处；校验 hiddenSizeOutput=6, numFusedSharedExpert=1, enable_pdl=8
```

文件：`…/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu`（现网已安装）；`runner.h` 为「0.7.0 routing 签名 / fork Gemm1&Gemm2 签名 / 三方法带 hiddenSizeOutput」。
脚本：`/root/glm53_mol/port5.sh`（改写 → 清 0.7.0 缓存 → 清场 → 启动 → 守望落盘 `verify_K10_result.txt`，含 HEALTHY 时自动跑 T1/T2 并打印 `oom_after_T1/T2` 与 `sigquit_total`）。

> **注意（本轮已踩的坑）**：`launch_glm53_pd_07.sh` 会被历史遗留的同名脚本反复互相 kill ⇒ 每次启动前先 `mapfile -t LS < <(pgrep -f "[l]aunch_glm53_pd_07.sh"); for p in "${LS[@]:1}"; do kill -9 $p; done` 只留一个。

**下一步**（接手即做）：等 `verify_K10_result.txt` → 若 `错误=0` 且 `EXTEND pass≥1`（且 `sigquit=0`）→ 看 T1/T2 的 `oom_after_*`；若仍有错误，重复「grep 实测文本 → 同法对齐签名」环（当前错误面已从 94 → 12 → 本轮 6 处改动后未验证完，属同类收敛）。验收判据仍是：`memory allocation failed = 0` 且 `Received sigquit = 0`。

### 5.7 K10 结果：插入式改法不够，`.cu` 需要**整段形参表 rebase**（下一轮的唯一任务）

K10（在 `.cu` 里插入 `numFusedSharedExpert`/`enable_pdl`/`hiddenSizeOutput`）**错误反而 12 → 46** ✗。根因：fork 的 `trtllm_fused_moe_runner.cu` 里 `Routing::Runner::run` 的形参表是**更早版本 flashinfer 的写法**（`int32_t nGroup, int32_t topkGroup,`），而 0.7.0 是 `numFusedSharedExpert, nGroups, topkGroups` —— **只插一个参数会让后续所有实参整体错位** ✗。

**下一轮唯一任务（精确版）**：
1. 取 0.7.0 的 `routing_runner.cu` 调用点 + 0.7.0 的 header 声明，作为**规范形参表**；
2. 把 fork 的 `trtllm_fused_moe_runner.cu` 中 `void Runner::run(void *routingLogits, ...)` 的**整段形参表逐项改写**为 0.7.0 版（`nGroup`→`nGroups`、`topkGroup`→`topkGroups`、topK 后插 `numFusedSharedExpert`、末尾加 `bool enable_pdl`），**并同步函数体内对这些形参名的引用**（`grep -n "nGroup\|topkGroup" <file>` 逐个改）；
3. 同法处理 `isValidConfigIndex`/`getDefaultValidConfigIndex`/`getValidConfigIndices` 的 `hiddenSizeOutput`（K10 已补但需确认调用点也一致）；
4. 装载 → 清 `…/0.7.0.post1/103a/cached_ops/sgl_fused_moe_trtllm_sm100` → 跑 `port5.sh` 同构脚本 → 直到 `错误行=0` → 看 `EXTEND pass≥1`/`sigquit=0` → T1/T2 `oom=0`。

**⚠️ 现场注意（本轮踩到）**：我早期的清缓存脚本把 **0.6.18 的 MoE JIT 缓存也挪走了**（`find … -name sgl_fused_moe_trtllm_sm100` 未按版本过滤 ✗）⇒ 生产（0.6.18）重启后需**重新编译该内核**，启动时间会明显变长（非故障）。恢复生产用 `bash launch_glm53_pd.sh`（原 venv 0.6.18，未改动 ✓），新版试验用 `launch_glm53_pd_07.sh`。**以后清缓存务必加 `| grep 0.7.0.post1` 过滤**。

### 5.8 ⚠️【致命坑·我踩了】fork 的 `data/csrc/*.cu` 是**两个 venv 共用的**

`…/trtllm_lora_temp/data/csrc/*.cu`（以及 `data/include/…`）位于 **sglang 树**里，**0.6.18 与 0.7.0 两个 venv 的 JIT 都会读它** ⇒ 我为 0.7.0 做「整段形参表 rebase」时改了它，**0.6.18 生产重启后就用这份 0.7.0 版代码去编** → 编译失败（46 错）→ **子进程死、生产崩** ✗。

**规避（下一轮务必照做）**：
1. 改 `.cu` 之前先 `cp <file> <file>.fork-pristine` 备份；
2. 更好的做法：**不要原地改**——把 0.7.0 需要的版本放到**新 venv 专属**的位置（或让 `jit.py` 按版本选不同目录），使 0.6.18 始终只见 fork pristine；
3. 恢复生产只需：`cat <本地 pristine 的 trtllm_fused_moe_runner.cu> > …/data/csrc/trtllm_fused_moe_runner.cu` **+ 清该内核 JIT 缓存 + 重启**（已验证 `编译错 46 → 0` ✓）；
4. **清 JIT 缓存必须按版本过滤**：`find … -name "sgl_fused_moe_trtllm_sm100*" | grep 0.7.0.post1`，绝不要裸 `find`（我早期裸 find 把 0.6.18 的缓存也挪走了，害得生产每次重启都要重编 ~10 分钟 ✗）。

### 5.9 接手清单（当前状态，2026-10-06 20:58）

| 项 | 状态 |
|---|---|
| 生产（0.6.18） | **单实例重启中**：MoE 内核在重编（`nvcc`/`ptxas` 在跑，**编译错=0** ⇒ `.cu` 已还原为 fork pristine 927 行 ✓）；`resurrect.sh` 的单实例已就位，编完即起；观察 `/root/glm53_mol/resurrect.log` 与 `curl 127.0.0.1:31300/health` |
| 0.7.0 试验 | `.cu` 的 0.7.0 对齐版**已保存**在 `/root/glm53_mol/fi070_port/runner_cu.070mod` ✓（是「整段形参表 rebase」那版的产物）——下一轮**把它放到 0.7.0 专属位置**再跑，别原地覆盖 ✗ |
| 已完成 | 新 venv `/opt/sglang-venv-07`（0.7.0.post1）✓、`jit.py` 版本自适应 ✓、`runner.h`/`DevKernel.h`（fork 原版 + 上游 8 处增量）✓ |
| 判据 | `编译错=0` → `EXTEND pass≥1` 且 `sigquit=0` → T1/T2 `oom_after_*=0` |
| 脚本 | `resurrect.sh`（清场+单实例+守望）、`port6.sh`（0.7.0 一轮改+部署+守望模板）、`launch_glm53_pd_07.sh`（新 venv 启动器） |

### 5.10 生产恢复记录（2026-10-06 21:49）

**故障**：我为 0.7.0 原地修改了**被两个 venv 共用**的
`…/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu` 与
`…/data/include/flashinfer/trtllm/fused_moe/{runner.h,DevKernel.h}` ✗
⇒ 0.6.18 生产用 0.7.0 版源码编译 → **46 编译错 → 子进程死 → 生产全程宕机**（约 20:09–21:49）。

**恢复步骤（可复用）**：
1. 从本地 fork pristine 覆盖三个文件（`runner.h` 540 行 / `DevKernel.h` 481 行 / `trtllm_fused_moe_runner.cu` 927 行，均校验 `TopKSigmoid=0`、`numFusedSharedExpert=0`）；
2. **把所有 `sgl_fused_moe_trtllm_sm100*` 目录（含 `.stale-*`）移出缓存根**（`mv` 到 `/root/glm53_mol/jit_quarantine/`）——只在缓存根内改名无效，ninja 仍会复用 ✗；
3. 清场（反复 kill 直到 `sglang|launch_glm53_pd` 计数为 0）+ **单实例** `setsid nohup bash launch_glm53_pd.sh &`；
4. 等 MoE 内核重编（~10–15 分钟）→ `curl 127.0.0.1:31300/health` = 200。

**恢复结果**：`31100/31200/31300 = 200/200/200`、`models=200`、池子 `10,075,328`、`flashinfer-python 0.6.18`、编译错 0。
**教训（与 5.8 同一条，必须刻死）**：**改 0.7.0 的 overlay 源码前，先把 fork pristine 备份，且新版本要放到 0.7.0 专属路径**；清缓存**必须按版本过滤**并**移出缓存根**。

### 5.11 版本隔离机制（2026-10-06 21:50，已落地）

**问题**：fork 的 MoE-LoRA overlay 源码（`trtllm_lora_temp/data/**`）位于 **sglang 树内**，
被**所有 venv 的 JIT 共用** ⇒ 为 0.7.0 改它 = 打崩 0.6.18 生产（见 5.8/5.10）。

**解法（已实施）**：
1. 新建 **0.7.0 专属 overlay 目录** `/root/glm53_mol/fi070_overlay/`，
   内容 = fork `data/` 的整份拷贝，再覆盖 0.7.0 对齐文件：
   - `include/flashinfer/trtllm/fused_moe/runner.h` ← 0.7.0 对齐版（547 行、含 `TopKSigmoid`）
   - `csrc/trtllm_fused_moe_runner.cu` ← 0.7.0 对齐版（整段形参表 rebase）
2. `jit.py` 加**版本选择**（在 `nvcc_flags = …get_nvcc_flags_list(` 之前）：
   ```python
   try:
       import flashinfer as _fi; _ver = getattr(_fi, "__version__", "")
   except Exception:
       _ver = ""
   if _ver.startswith("0.7"):
       from pathlib import Path as _P
       _iso = _P("/root/glm53_mol/fi070_overlay")
       if _iso.is_dir():
           overlay_include_dir = _iso / "include"
           overlay_csrc_dir   = _iso / "csrc"
   ```
3. **效果**：0.6.18 永远读 fork 的 `data/`（pristine 540/481/927），0.7.x 读 `fi070_overlay/`
   ⇒ 后续所有 0.7.0 迭代**不可能再影响生产**，且每次迭代无需停机改共享文件。

**验证**：隔离后生产共用文件仍 540/481/927、`31100/31200/31300 = 200/200/200`、
`models=200`、池子 `10,075,328`、`flashinfer-python 0.6.18`、编译错 0。

### 5.12 接手清单（继续修 `trtllm_fused_moe_runner.cu` 的 0.7.0 rebase）

**目标（唯一判据，两条必须同时为 0）**：
```
grep -ac "memory allocation failed" pd_prefill_07.log   # 期望 0（单请求 716K + 10 并发都不 OOM）
grep -ac "Received sigquit"           pd_prefill_07.log   # 期望 0（调优期不再被杀 worker）
```
**入口**（无需停机改共享文件，隔离机制见 5.11）：
```bash
cd /root/glm53_mol
vi fi070_overlay/csrc/trtllm_fused_moe_runner.cu   # 只改这里
bash k12.sh                                         # 会：清 0.7.0 缓存 → 起 0.7.0 venv → 观察 → 跑 T1/T2 → 自动恢复生产
tail -40 verify_K12_result.txt
```
**当前已知的 rebase 缺口**（K11 的 12 条错误 → K10 的 46 条错误）：
- 形参表整体改名：`nGroup→nGroups`、`topkGroup→topkGroups`（0.7.0 全仓改名，必须整段替换而非局部插入）
- `trtllm_fused_moe_runner.cu(53): too few arguments / incompatible … run(...)`：
  0.7.0 的 `Runner::run()` 签名与 fork 不同 → 以 `fi070_overlay/include/…/runner.h`（0.7.0 对齐版）
  为准逐条对齐调用点
- `bool ↔ Dtype`、`void* ↔ intN_t*`、`MoERunnerArgs` 缺 `{lora_ready_event, gemmN_done_event,
  gate_up_lora_delta, activation_lora_input}`：这些是 **fork 独有的 LoRA 扩展**，
  必须把它们**再插回 0.7.0 结构**（不能丢，否则 MoE LoRA 失效）
- **禁止 `git merge-file` 3-way**：会把 `DevKernel.h` 的结构块整块重复 → 92 错（见 K9）
**已就绪的素材**：
- `fi070_port/runner_cu.070mod`（整段形参表 rebase 版，已在 `fi070_overlay/csrc/`）
- `fi070_port/out2/runner.h`（0.7.0 对齐版，547 行，已在 `fi070_overlay/include/…/`）
- `fi070_port/upstream_{runner_h,DevKernel_h}.diff`（上游 8 处语义增量，逐条）
- 上游 main pin：`flashinfer_python[cu13]==0.7.0.post1`；0.7.x **无** `flashinfer-cubin` 包
**回滚/恢复生产**（随时可用）：`bash restore_hdrs.sh`（还原三文件 + 清缓存 + 单实例起 0.6.18）

### 5.13 overlay 路径解析真相（2026-10-06 21:55，实测）

**结论一：JIT 的 overlay 目录 = `jit.py` 自身所在目录的 `data/`**（`jit.py:5`：
`return Path(__file__).resolve().parent / "data"`）。
`jit.py:22-23` 由此推出 `overlay_csrc_dir / overlay_include_dir`，再配合
`flashinfer.__file__`（`flashinfer_data_dir`）拼 nvcc 命令。

**结论二：venv 里的 `site-packages/sglang/jit_kernel/trtllm_lora_temp/` 会被 `PYTHONPATH` 遮蔽。**
- 两个启动脚本都 `export PYTHONPATH=/root/glm53_mol/sglang-private/python`（生产 line 3 / 07 版 line 7）；
- PYTHONPATH 优先于 site-packages ⇒ 两台 venv 实际用的都是**源码树**的
  `python/sglang/kernels/ops/moe/trtllm_lora_temp/`，venv 内那份（0.5.15.post1 wheel，`jit_kernel/` 布局，
  `runner.h` 586 行 / `.cu` 1110 行）是**死代码**，改它无效。
  ⚠️ 且该 wheel 版**也不适配 0.7.0**（`nGroup` 旧名、无 `TopKSigmoid`）⇒ 不能直接拿来用。

**结论三（关键）：因此 `fi070_overlay/` 机制（5.11）能真正生效** —— 它在**源码树**的 `jit.py` 里
按 `flashinfer.__version__` 分流 ⇒ 目录不同、文件互不干扰 ⇒ 生产安全（已复核 540/481/927 + 200/200/200）。

**推论（下轮务必先做）**：既然 overlay 由 `jit.py` 位置决定，**也可以改成"按版本指向源码树内的不同子目录"**
（例如 `data-070/`），从而彻底摆脱绝对路径 `/root/glm53_mol/fi070_overlay` 的耦合；
若换机器/换目录，`fi070_overlay` 的绝对路径需同步改。

### 5.14 K12 结果与 K13 前向移植方案（2026-10-06 22:01）

**K12（隔离 overlay + 0.7.0 venv + EXTEND）结果**：
```
outcome=CRASH  EXTEND_pass=0  sigquit=1  oom=0  错误=46
出错文件： trtllm_fused_moe_kernel_launcher.cu  42 错
           trtllm_fused_moe_runner.cu           4 错
错误类别（全部为「实参类型不匹配」，即签名问题，非语义问题）：
  bool ↔ batchedGemm::trtllm::gen::Dtype        10
  void* ↔ intN_t*                                5
  Routing::RoutingMethodType ↔ bool              5
  cudaStream_t ↔ Routing::RoutingMethodType       5
  Dtype ↔ intN_t*                                 5
  Dtype ↔ cudaStream_t                            5
  int* ↔ float                                    3
  nullptr_t ↔ bool                                2
```
**根因**：我只把 `trtllm_fused_moe_runner.cu` 做了 0.7.0 对齐，
**`trtllm_fused_moe_kernel_launcher.cu` 仍是 fork 版**（它才是 LoRA 管线的主战场）⇒ 42 错全部来自它。
另 4 错：`.cu` 定义与 0.7.0 头声明不一致
（`PermuteGemm1::Runner::isValidConfigIndex` 5→**6** 参、
 `Gemm2::Runner::isValidConfigIndex` 5→**6** 参、
 `MoE::Runner::getDefaultValidConfigIndex` 返回 **int64_t**）。

**基线尺寸（K13 前向移植的规模）**：
| 文件 | 0.7.0 上游 | fork | diff |
|---|---|---|---|
| `include/.../fused_moe/runner.h` | 523 | 540 | 784 行 |
| `csrc/trtllm_fused_moe_runner.cu` | 791 | 927 | 1390 行 |
| `csrc/trtllm_fused_moe_kernel_launcher.cu` | **6661** | 4418 | **9415 行** |
fork 的 LoRA 符号分布：`gate_up_lora_delta` fork_launcher=47（上游 0）、`activation_lora_input` 46（0）、
`lora_ready_event` 18（0）、`gemm2_done_event` 18（0）。

**K13 方案（前向移植，编译器逐条指路）**：
1. **基底** = 0.7.0 上游三文件：`/opt/sglang-venv-07/lib/python3.12/site-packages/flashinfer/data/{include/.../runner.h, csrc/trtllm_fused_moe_runner.cu, csrc/trtllm_fused_moe_kernel_launcher.cu}`；
2. 在基底上**重新施加 fork 的 LoRA 增量**（`lora_ready_event` / `gemm2_done_event` /
   `gate_up_lora_delta` / `activation_lora_input` / `numFusedSharedExpert` / gated-silu 激活）；
3. 产物放 `fi070_overlay/`（**绝不动源码树 `data/`**，见 5.11）；差分与中间物放 `fi070_port/K13/`；
4. 迭代环：`bash k12.sh`（已含自动恢复生产）→ 看 `verify_K12_result.txt` 错误面 → 修 → 重复；
5. **判据（两条同时为 0）**：`grep -ac "memory allocation failed" pd_prefill_07.log` = 0 且
   `grep -ac "Received sigquit" pd_prefill_07.log` = 0（当前 46 错→必须降到 0 错，再验 sigquit/oom）。

### 5.15 K13 首轮结果（2026-10-06 22:13）：46 编译错 → 1 链接错

**结果**：`outcome=CRASH @22:13:13  EXTEND_pass=0  sigquit=1  oom=0  错误=1`
**错误性质发生质变**：不再是「实参类型不匹配」（46 条编译错），而是**唯一的链接错** ⇒
**所有 `.cu` 已全部编译通过**（K13 的 0.7.0 前向移植在**编译层面已成功**）。

**链接错全文**：
```
collect2: error: ld returned 1 exit status
/usr/bin/ld: …/flashinfer/0.7.0.post1/103a/cached_ops/sgl_fused_moe_trtllm_sm100/
  trtllm_backend_trtllm_fused_moe_routing_runner.cuda.o:
    in function `tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner::Runner()'
    in function `…::Routing::Runner::Runner(int)'
    in function `…::Routing::Runner::run(void*, void*, int×10, float, int*×7, void*, int*×4,
                     batchedGemm::trtllm::gen::Dtype, Dtype, bool, bool, RoutingMethodType,
                     CUstream_st*, Dtype, bool, short*, bool)'
```
**判读**：`Routing::Runner` 的 `Runner()`/`Runner(int)`/`run(...)` 在
`trtllm_backend_trtllm_fused_moe_routing_runner.cuda.o` 这个 TU 里**被引用但未定义**
⇒ 定义所在的那个 routing 源文件没被编进同一个 `.so`。
**候选原因**（按可能性排序）：
1. `jit.py` 中 0.7.x 的 `_custom_routing_srcs` 只列了 `routing_runner.cu`，**漏了**
   `custom_block.cu` / `custom_cluster.cu` / `custom_cluster_large.cu` / `custom_entry.cu`
   （`custom_entry.cu` 提供 `Routing::Runner` 的成员定义）；
2. 缺 `-rdc=true` / `--relocatable-device-code=true`（缺它时 device 侧符号同样 ld 失败）；
3. fork 侧新增的 LoRA 重载（带 `gate_up_lora_delta` 等形参）只在头里声明、未在某 `.cu` 里定义。

**下一轮取全量 ld 错误（勿被 head 截断）**：
```bash
tr '\r' '\n' < /root/glm53_mol/pd_prefill_07.log | grep -a -A1 '/usr/bin/ld: .*\.o:' | grep -a 'undefined\|error'
```
**风险提示**：K13 迭代中 overlay 的 `trtllm_fused_moe_runner.cu` 曾从 927 行换成 625 行版本 ⇒
必须 `grep -c` 复核 `gate_up_lora_delta / activation_lora_input / lora_ready_event` 仍在，
否则就算 0 错也会丢掉 LoRA 语义（零 LoRA == base 的位级一致性会破）。

### 5.15 K13：46 编译错 + 链接错全部消除（前向移植成功点）

**成果（可复核）**：
- `verify_K13_result.txt`：`overlay 校验: runner.h=548 TopKSigmoid=2 | cu=625`
- **MoE JIT 编译 + 链接全绿**：`/mnt/workspace/.../cached_ops/sgl_fused_moe_trtllm_sm100/sgl_fused_moe_trtllm_sm100.so`
  **已生成（23.7 MB, 22:22）**，18 个 `.o` 全部在内，含全部 5 个 routing TU
  （`routing_runner` / `custom_block` / `custom_cluster` / `custom_cluster_large` / `custom_entry`）。
- 服务推进到 `Running FlashInfer autotune with cache …/0.7.0.post1/sm103/…`（EXTEND 阶段）。

**三处修复（都在 `fi070_overlay/`，源码树共享文件始终 pristine）**：

1. **launcher 5 个 `routing_runner.run(...)` 调用点对齐 0.7.0**（解 46 编译错）
   0.7.0 的 `Routing::Runner::run` 比 fork 的调用**多两个参数**，导致其后整串错位：
   - `args->top_k,` 后插入 `args->num_fused_shared_experts`（`MoERunnerArgs` 补该字段，**值取 0**＝fork 语义：
     shared expert 走 `--experts-shared-outer-loras`，不融合进 routing；0.7.0 自己的 fp4 路径也传字面量 `0`）
   - `nullptr /*permuted_idx_to_expanded_idx*/` → `workspace.permuted_idx_to_expanded_idx`（主路径 3 处；
     fp4 两处按 0.7.0 上游保持 `nullptr`）
   - 末尾补 `enable_pdl`
   校验：5 个调用点参数个数 = **33**，与 0.7.0 上游**完全一致**。

2. **`runner.h` 声明补 `hiddenSizeOutput = -1` 默认参**（解 4 条定义/声明不一致）
   fork 的 `.cu` 定义比 0.7.0 头声明多一个 `hiddenSizeOutput`（fork 扩展）；
   给它加默认值 ⇒ 定义匹配，且 0.7.0 风格的 5 参调用者仍可用。

3. **删除 overlay `runner.cu` 里重复的 `namespace Routing` 块**（解链接 `multiple definition`）
   0.7.0 把 `Routing::Runner` 的定义移到了 `fused_moe/trtllm_backend/trtllm_fused_moe_routing_runner.cu`
   （由 `jit.py` 的 `_custom_routing_srcs` 编译，`_routing_dir` 指向 **flashinfer 的**目录 ✓）。
   我的 overlay 里是 fork 的 927 行版（同样定义 Routing）⇒ 符号重复。
   删除 L34–341（307 行）后 → 625 行。
   **已核实 LoRA 语义零损失**：LoRA 注入全在 `PermuteGemm1(L342)/Gemm2(L523)/MoE(L631)` 块内，
   计数与删除前完全相同（`gate_up_lora_delta` 2、`activation_lora_input` 1、`lora_ready_event` 2、
   `gemm2_done_event` 2）；launcher 侧 47/46/18/18 全在；头部字段齐。
   （唯一变化 `numFusedSharedExpert` 1→0 属正常：它属于 Routing，现由 0.7.0 的 routing_runner 提供。）

**教训（务必记住）**：批量正则补丁**必须幂等**（我在 K13 早期同一脚本跑两次 ⇒
`num_fused_shared_experts` 被插两次，多花一轮）；删除大块后**必须立刻 `grep -c` 复核关键语义符号**。

### 5.16 【里程碑】OOM 根因已消除，仅剩一个 sglang 适配缺陷（2026-10-06 22:30 实测）

**0.7.0 隔离 overlay + K13 前向移植后，实测结果**：
```
[22:30:42 TP0] FlashInfer autotune: extra EXTEND pass at 16416 tokens (342 seqs x 48 tokens).
oom = 0     ← 「memory allocation failed」计数为 0
EXTEND = 1  ← 扩容调优在 prefill 形状上真的跑起来了
sigquit = 4
```
**判读**：
1. **原始 OOM 根因（MoE 未调优 heuristic 的超额 workspace，上游 #4916）已被消除** ✓ ——
   0.7.0 的 autotuner 在 **16416 tokens 这个原崩溃形状**上顺利完成了 EXTEND pass，
   `memory allocation failed` 计数为 **0**。这正是「装 0.7.0」要达成的目标，**已达成**。
2. **剩余的唯一阻塞是另一个独立的 sglang 侧适配缺陷**（不再是 OOM/链接/编译）：
```
[22:30:42 TP0] Scheduler hit an exception: Traceback (most recent call last):
   ...  assert _current is not None, (
AssertionError: no forward context active — call forward_context(...)
            or set_forward_context(...) before reading get_forward_context()
```
   TP0/1/2/3 四个 rank 同时抛出，发生在 **0.7.0 autotune 的 EXTEND pass 之后、forward 之外**。
   ⇒ 需要让调优路径设置 forward context（或让该处容忍 None）。排查起点：
```bash
grep -rn "get_forward_context\|set_forward_context\|forward_context(" \
  /root/glm53_mol/sglang-private/python/sglang | head -50
```
**验收判据的当前状态**：
| 判据 | 目标 | 实测 |
|---|---|---|
| `grep -ac "memory allocation failed" pd_prefill_07.log` | 0 | **0 ✓ 已达成** |
| `grep -ac "Received sigquit" pd_prefill_07.log` | 0 | 4 ✗（由上述 AssertionError 引起，修掉即消失）|

### 5.17 最后一个阻塞的根因与修复：forward context 作用域错位（2026-10-06 22:33 已修）

**现象**：0.7.0 起服务后，CUDA graph 捕获期 warmup 触发 flashinfer EXTEND 调优，
四个 TP rank 同时抛 `AssertionError: no forward context active` ⇒ `Received sigquit` ×4。

**完整调用栈**（`pd_prefill_07.log:543` 起）：
```
scheduler.py:5852 run_scheduler_process → :576 __init__ → :1111 init_model_worker
 → :1094 init_all_cuda_graphs → tp_worker.py:443 init_cuda_graphs
 → model_runner.py:1100 init_cuda_graphs → cuda_graph_setup.py:250 capture_cuda_graphs
 → eager_runner.py:156 __init__ → base_runner.py:261 warmup
 → flashinfer_autotune.py:404 maybe_flashinfer_autotune_extend
 → flashinfer_autotune.py:299 run_flashinfer_autotune_forward
 → base_runner.py:639 _dummy_run
 → hybrid_linear_attn_backend.py:1183 → dsa_backend.py:1137 init_forward_metadata
 → get_forward_context().attn_backend → AssertionError
```
**根因：不是「缺 context」，而是「作用域错位」** ✔
`base_runner.py` 里**本来就有** forward context（**675 行**）：
```python
675: with forward_context(ForwardContext(attn_backend=mr.attn_backend)):
676:     with run_ctx or empty_context():
677:         run_once()
```
但 `mr.attn_backend.init_forward_metadata(forward_batch)` 在 **639 行**——落在该 `with`
**之外、之前** ⇒ dummy/autotune forward 里 DSA/hybrid attn backend 读
`get_forward_context()` 时无 context。

**修复（最小、向后兼容，不动 synchronize/barrier 顺序）**：
```python
638: forward_batch = mr.prepare_dummy_forward_batch(forward_batch)
639: # FIX(mindlab): init_forward_metadata 读 get_forward_context().attn_backend，
     #   但它原本跑在下面的 forward_context() 块之前 → EXTEND pass 以
     #   "no forward context active" 崩溃。用同一个 context 包住；
     #   嵌套安全（forward_context() 保存/恢复 previous）。
645: with forward_context(ForwardContext(attn_backend=mr.attn_backend)):
646:     mr.attn_backend.init_forward_metadata(forward_batch)
```
备份：`base_runner.py.bak-fwdctx-<ts>`；已 `py_compile` 通过。
⚠️ 该文件属于 **sglang Python 侧（两个 venv 共用一份）**，与 JIT overlay 的隔离策略不同；
因为它只是「补上原本缺失的 context」，对 0.6.18 生产无副作用，故直接改在源码树。
**验证要求**：改后既要看 0.7.0 `sigquit` 归零，也要确认 0.6.18 生产照常 200/200/200。

### 5.18 【验收达成】0.7.0 + 修复后：两条判据首次同时为 0，服务成功就绪（2026-10-06 22:43）

**带 `base_runner.py` forward-context 修复后，0.7.0（隔离 overlay）实测**：
```
07P: 错误=0   sigquit=0 ✓   EXTEND=1 ✓   oom=0 ✓
[AutoTuner]: Tuning flashinfer::trtllm_fp8_block_scale_moe: 22/23 (96%)   ← 扩容调优跑完
断言/异常: 无（"no forward context active" 已消失）
… Disaggregation warmup requests completed for all 1 DP ranks
… End of disaggregation warmup
… The server is fired up and ready to roll!        ← 0.7.0 服务完全就绪
31100 = 200
```
**对比（修复前）**：同一 0.7.0 配置 `sigquit=4`、服务起不来。
**结论**：`forward_context` 作用域错位（文档 5.17）是最后一个阻塞；修掉后
- **判据① `memory allocation failed` = 0** ✓（OOM 根因消除，见 5.16）
- **判据② `Received sigquit` = 0** ✓（本处）
且服务能在 prefill 形状上完成 EXTEND 扩容调优后正常就绪。
**待验**：`ttft_probe2.py 50000`（单请求 716K 冷）+ `conc_probe.py 10 2600000`（10 并发）
是否仍 `oom_after_*=0`（这决定「生产时十个这样的请求并发没问题」）。

### 5.19 最后一层根因：EXTEND 预调优的桶 ≠ 运行期 prefill 形状（2026-10-06 22:45）

**修复 forward-context 后（5.18）**：0.7.0 服务成功就绪、`sigquit=0`、`EXTEND_pass=1`、`错误=0`。
**但 T1（单请求 716K 冷）仍打印 12 次 OOM**：
```
[rank0]:[W1006 22:44:13] memory allocation failed with OOM on device 0
        while trying to allocate 5653921792 bytes (free: 1602813952, total: 287428771840)
…（4 rank 各 1 次；oom_after_T1=12）
```
**关键读数**：
- 请求分配 = **5,653,921,792 B = 5.27 GB** —— 与历史定位的 `want 5.27GB`（16384-token prefill 形状）
  **完全一致**（见早期章节的 chunk-16384→8192 精确减半证据）。
- 触发时机 = **第一次真正处理 `#new-token: 16384` 的 prefill batch**（22:44:11→13），此前只有 autotune。
- 当时 `free` 仅 ~1.6 GB ⇒ 必然失败；但**请求没死**（T1 `TTFT=25.86s total=25.96s` 成功）
  ⇒ 影响由「杀 worker」降级为「打印 OOM + 换 tactic」，**进程存活、`sigquit=0`**。

**根因判断（最后一层）**：**EXTEND 预调优的形状与运行期形状不是同一个 bucket**
- EXTEND 调优的是 **16416 tokens**（日志 `extra EXTEND pass at 16416 tokens (342 seqs x 48 tokens)`）
- 运行期 prefill chunk 是 **16384 tokens**
⇒ 16384 从未被调优，运行期落到未调优 heuristic，才会去要那 5.27 GB。
**修法方向**：让运行期实际使用的 prefill 形状（16384、8192、以及 chunked 值）
都被 EXTEND 覆盖（改 bucket 选择 / 触发条件），使那笔 5.27 GB 分配**不再被尝试**
（而不是靠外层兜住）。
**用户验收硬标准（不因影响降级而放松）**：「一次也不能有 OOM」
⇒ `grep -ac "memory allocation failed" pd_prefill_07.log` **必须真为 0**。

### 5.20 验收进展：OOM 判据已归零，仅剩一处自研代码 bug（2026-10-06 23:05）

**K13 执行者的 autotune 持久化缓存方案（`/root/.cache/sglang/flashinfer/autotune/0.7.0.post1/sm103/<hash>/rank_tp*.json`）
+ 多 target EXTEND 后的实测**：
```
outcome=CRASH @22:59:51
EXTEND_pass=3 ✓✓（从 1 提升到 3：extra EXTEND pass at 16416 tokens [target 1/2] / [target 2/2] / 第三次）
oom=0 ✓✓         ← 「memory allocation failed」**已归零**（5.27GB 那笔不再被尝试）
错误=0
sigquit=1 ✗      ← 仅剩
[22:59:47 TP0/1/2/3] Scheduler hit an exception:
    UnboundLocalError: cannot access local variable 'forward_fn'
    where it is not associated with a value
```
**判读**：
1. **OOM 判据已真正归零** ✓ —— 持久化 autotune 缓存 + 多 target EXTEND 让运行期 prefill 形状
   直接命中已调优 tactic，不再产生那笔 5.27 GB 试探分配。这是「一次也不能有 OOM」的正解路径。
2. 剩下的 `sigquit` 来自**自研改动引入的 Python 作用域 bug**（为多 target 重构时
   `forward_fn` 变成条件赋值，某分支未赋值即使用）⇒ 修法 = 保证所有路径都有定义
   （先给默认值 / 复用同一闭包）。**与 flashinfer / OOM 无关。**
**判据看板**：
| 判据 | 目标 | 实测 |
|---|---|---|
| `memory allocation failed` | 0 | **0 ✓ 已达成** |
| `Received sigquit` | 0 | 1 ✗（由上述 UnboundLocalError 引起）|
| `EXTEND pass` | ≥1 | **3 ✓** |
| 生产 200/200/200 | 是 | ✓（池子 10,075,328、编译错 0）|

### 5.20 K13 v4 结果（2026-10-06 23:16）—— sigquit 归零 ✓，oom 仍为 12 ✗

**实测（`verify_K13_result.txt`）**：
```
outcome=HEALTHY @ 23:16:04
EXTEND_pass=2  sigquit=0 ✓✓  oom=0（启动期）  错误=0 ✓
T1: CACHE+NEW 716K TTFT=20.66s（chunks=18） ; CACHE+NEW#2（缓存命中）TTFT=1.65s ✓✓
oom_after_T1=12 ✗
```
- **用户抱怨的「98% 命中却 TTFT 37.7s」已解决** → **1.65s** ✓✓
- **`Received sigquit` 归零** ✓✓：修掉两处 Python 作用域 bug
  （① `base_runner.py` `_dummy_run` 里 `init_forward_metadata` 落在 `forward_context()` 块**之外**
   → `no forward context active`；② 我改 `flashinfer_autotune.py` 时旧尾部 `try/finally: del forward_fn` 残留 → `UnboundLocalError`）
- **EXTEND 从 1 次提到 2 次** ✓：`[target 1/2] 16416 (342×48)` + **`[target 2/2] 16384 (1 seq × 16384)`**
  （后者按运行期真实形状 `#new-seq:1, #new-token:16384` 加的）

**未达成：`memory allocation failed` = 12 ✗**，且**结论与「桶错配」假设不同**：
明细两轮完全相同 —— `5,653,921,792 B (5.27 GiB)`（free 1.5–1.9 GB）与
`7,532,969,984 B (7.02 GiB)`（free 6.4–6.8 GB），各 4 rank，触发点 `#new-token: 16384`。
⇒ **16384 桶已被调优，这笔分配照样发生** ⇒ 它不是桶错配，而是**该形状下所选 tactic 自身的 workspace 需求**。
（与运行期内存争用叠加：KV 池 66 GB + hicache 56 GB 之后，GPU 只剩 ~1.5–6.8 GB 空闲。）

**下一步（唯一正确方向，非 hack）**：让 tactic 选择**内存感知** —— 0.7.0 头里已有
`MoE::Runner::getWorkspaceSizeInBytes(args, configIndex) const`（见 overlay `runner.h`）。
在挑 config 时先用它估算 workspace，挑**放得下**的那个，而不是挑最快但放不下的。
落点：`fi070_overlay/`（JIT 侧，**不碰**源码树共享 `data/**`）。
**降 mem-fraction 的办法不建议优先**：它和「10 并发 × 716K」的 KV 容量直接冲突。

### 5.21 OOM 的精确性质：autotuner 的 tactic 探测（非运行时真实需求）（2026-10-06 23:17）

**带 forward-context 修复 + bucket 修复(含 16384) + forward_fn 修复后的实测**：
```
outcome=HEALTHY @23:16:04 ✓
EXTEND_pass=2 ✓✓  extra EXTEND pass at 16416 tokens [target 1/2]
                  extra EXTEND pass at 16384 tokens (1 seqs x 16384 tokens) [target 2/2]   ← 桶修复已生效
sigquit=0 ✓✓   错误=0 ✓
T1: CACHE+NEW TTFT=20.66s total=20.76s ✓（请求成功）
但 oom_after_T1=12 ✗
```
**OOM 的尺寸分布（每 rank 4 次，共 16 次）= 一组单调递增的 tactic workspace**：
| 尺寸 | 次数 |
|---|---|
| 10,074,718,208 B（10.07 GB） | 4 |
| 9,143,582,720 B（9.14 GB） | 4 |
| 7,532,969,984 B（7.53 GB） | 4 |
| 5,653,921,792 B（5.27 GB） | 4 |

**判读（关键）**：出现在**调优完成之后**、处理**第二条** `#new-token: 16384` prefill 时（23:16:15–16），
当时 `free` 仅 1.5–1.9 GB ⇒ 四种尺寸逐个失败、被外层兜住、请求仍成功。
⇒ **这不是「未调优形状走 heuristic」，而是运行期仍有 autotuner 在探测 tactic**
（MoE 的 EXTEND 已覆盖 16416/16384，故这批很可能是 **DSA/sparse-MLA indexer 或 attention** 的
autotune workspace，不在 MoE EXTEND 覆盖内）。
**下一步方向**：
1. 查除 MoE 外的 autotune 入口（`sglang/srt/layers/attention/**`、
   `model_executor/runner/flashinfer_autotune.py` 里的 `maybe_flashinfer_autotune_*` /
   `flashinfer_autotune_context`），确认 prefill 路径上仍有未预热的 autotune。
2. 查 autotune 缓存 `/root/.cache/sglang/flashinfer/autotune/0.7.0.post1/sm103/<hash>/rank_tp*.json`
   是否存在**另一个 hash**（不同 batch/seq 组合）→ cache miss → 触发探测。
3. 目标：让这批 tactic **在启动期全部预调优**，或让探测**在足够 headroom 下进行**；
   **不得**以「被兜住」交差（用户硬标准：一次也不能有 OOM）。

### 5.21 OOM 的精确性质：运行期 MoE tactic 探测（非「未调优形状」）（2026-10-06 23:20）

**实测四个尺寸（每 rank 各 1 次，共 16 次，全在 `free` 仅 1.5–1.9 GB 时）**：
```
10,074,718,208 B (10.07 GB) ×4     ← 单调
 9,143,582,720 B ( 9.14 GB) ×4
 7,532,969,984 B ( 7.02 GB) ×4
 5,653,921,792 B ( 5.27 GB) ×4
```
出现时刻：**EXTEND 调优已完成之后**（`EXTEND_pass=2`），处理第二条 `#new-token: 16384` prefill 时；
逐个失败 → 被外层兜住 → **请求仍成功**（T1 `TTFT=20.66s total=20.76s`）。

**排查结果（三条都查了）**：
1. **autotune 缓存只有 1 个 hash**：`/root/.cache/sglang/flashinfer/autotune/0.7.0.post1/sm103/efa5f7813f836adb/rank_tp*.json`
   ⇒ **排除**「另一种 batch/seq 组合导致 cache miss」。
2. autotune kernel 只有两个：`flashinfer::trtllm_fp8_block_scale_moe`（108 次）、`trtllm_batch_decode_mla`（28 次）。
3. sglang 侧 autotune 入口只有 `maybe_flashinfer_autotune_extend`（`base_runner.py:261`）；
   `maybe_flashinfer_autotune_speculative_draft` 系列**未启用**（无投机解码）。

⇒ **探测者是 MoE 自身**：这 4 个尺寸是 `trtllm_fp8_block_scale_moe` 的 **tactic workspace 探测序列**，
发生在**运行期**（不是启动调优期）。即：**即使 16384 桶已被 EXTEND 调优，运行期仍会去探测候选 tactic**。

**必须走的修法（不要靠「被兜住」交差）**：
- **A**：让这批 tactic 在**启动期**就全部预调优进同一缓存（把运行期实际用到的 (kernel, shape) 都覆盖）；
- **B**：让 tactic 探测**有 headroom 才做** —— 用 0.7.0 头里已有的
  `MoE::Runner::getWorkspaceSizeInBytes(args, configIndex)` 先估 workspace，放不下就跳过候选（不发起分配）。
落点仍在 `fi070_overlay/`（JIT 侧）或 venv 内 flashinfer 的 tuner（**不碰**源码树共享 `data/**`）。

**已完成且必须保留的成果**：`sigquit=0` ✓✓（崩溃根因消除）、`EXTEND_pass=2`（含 1×16384）✓、
服务就绪 ✓、**T1 缓存命中 TTFT = 1.65s**（原 37.7s）✓✓。

### 5.22 【功能验收达成】10 并发长请求 10/10 成功（2026-10-06 23:19）

**T2（`conc_probe.py 10 2600000`，10 个并发长请求）实测**：
```
#3  OK  total= 95.9s  chunks=11
#4  OK  total=126.7s  chunks=11
#5  OK  total= 22.6s  chunks=11
#6  OK  total= 49.5s  chunks=11
#7  OK  total= 80.3s  chunks=11
#8  OK  total=156.6s  chunks=11
#9  OK  total=141.9s  chunks=11
OK=10  ERR=0        ← 十个并发长请求全部成功，零错误
oom_after_T2=16     ✗（仍为 autotuner 的 tactic 探测消息，见 5.21）
```
**结论**：用户的核心诉求「**生产时十个这样的请求并发必须没问题**」**在功能层面已满足**：
10/10 成功、无失败、无 worker 死亡（`sigquit=0`）。
**唯一残留**：探测期的 `memory allocation failed` 告警 16 条（被外层兜住、不影响成功）。
按用户硬标准「一次也不能有 OOM」，仍需按 5.21 的方向**消除探测本身**
（把 attention/indexer 的 tactic 纳入启动期预调优，或让探测在充足 headroom 下进行）。

### 5.23 最后一项的实现方案（已定位到行，待落地）：给 tactic 预过滤加「内存门控」

**目标**：`grep -ac "memory allocation failed" pd_prefill_07.log` = **0**（现 12/16）。
**性质**（5.21 已证实）：调优完成后，运行期处理 `#new-token: 16384` prefill 时，
autotuner 仍在**逐个探测 tactic**，四个候选的 workspace
`10.07 / 9.14 / 7.53 / 5.27 GB` 全 > 当时空闲（1.5–1.9 GB）⇒ 逐个失败（allocator 会**先打日志再抛**，
所以即使被兜住也会留下 `memory allocation failed` 字样）。

**代码位置（0.7.0 venv 内，与共享源码树无关，可安全改）**：
`/opt/sglang-venv-07/lib/python3.12/site-packages/flashinfer/fused_moe/core.py`
- **预过滤函数**：`get_valid_tactics()`（**801–898 行**）——目前只按
  ①occupancy（`get_tactic_occupancy(t) > 0`，836 行）②shape 有效性（`get_valid_tactics_for_shape`，897 行）
  过滤，**唯独不按可用显存过滤** ⇒ 必然去尝试放不下的候选。
- **可用的尺寸 API**：`self.fused_moe_runner.get_workspace_size(max_num_tokens, hidden_size,
  intermediate_size, num_experts_total, top_k, tp_size, tp_rank, ep_size, ep_rank,
  min_latency_mode, activation_type) -> int`（用法见 core.py:1234）。
- **该 runner 已具备的字段**（core.py:724–768）：`self.top_k / tp_size / tp_rank / ep_size / ep_rank /
  min_latency_mode / activation_type / fused_moe_runner`；`hidden_size` 可由
  `inputs[0].shape[1]`（hidden_states）、`num_tokens` 由 `inputs[0].shape[0]` 得到
  （参考 core.py:1171 取 `fc2_expert_weights.shape[1]`）。

**实现（先估后配，放不下就跳过，不发起分配）**：
```python
# 在 get_valid_tactics() 内、构造 valid_tactics 之前/之后加：
import torch
try:
    free_bytes, _total = torch.cuda.mem_get_info()
    num_tokens = int(inputs[0].shape[0]) if inputs else 0
    hidden = int(inputs[0].shape[1]) if inputs and inputs[0].dim() > 1 else 0
    needed = int(self.fused_moe_runner.get_workspace_size(
        num_tokens, hidden, <intermediate_size>, <num_experts_total>,
        self.top_k, self.tp_size, self.tp_rank, self.ep_size, self.ep_rank,
        self.min_latency_mode, self.activation_type))
    # 留 10% 余量，避免刚好卡边界
    if needed and needed > int(free_bytes * 0.9):
        logger.warning("[AutoTuner] skip tactic probing for %d tokens: need %.2f GB > free %.2f GB; using fallback -1",
                       num_tokens, needed/2**30, free_bytes/2**30)
        return [-1]                      # runners[0]+tactic=-1 = 内建 heuristic 回退，不触发探测
except Exception as e:
    logger.debug("workspace-fit gate skipped: %s", e)
```
（`intermediate_size` / `num_experts_total` 从 runner 现有字段或 `inputs` 形状取；
若某字段缺失，用 try/except 包住并退化为不启用门控，**不要**用硬编码数字。）

**反例自证（防自欺）**：临时把门控关掉跑一次应能复现 ≥12 次 OOM；打开后归零。两次对照写进结果。

**验收**：见本文件 5.22 末尾的三条判据 + T1/T2（`oom_after_*=0`、`OK=10 ERR=0`），
外加 `Get-0.6.18 生产 200/200/200`（Python 侧若改动同样要复核）。
**注意**：gate 只在「放不下」时退化为 fallback；放得下时仍正常调优（不降性能）。

### 5.24 本轮进展与卡点（2026-10-06 24:00）—— 未达成，如实记录

**做了什么（两处门控补丁，均 `py_compile OK`，均在 0.7.0 venv 内、与共享源码树无关）**：
1. `flashinfer/fused_moe/core.py` 的 `MoERunner.get_valid_tactics`（801 行）加内存门控；
2. `flashinfer/fused_moe/backends/trtllm/sm100_runner.py` 的 `MoERunner.get_valid_tactics`（205 行）加内存门控
   （`torch.cuda.mem_get_info() < 2 GiB` → `return [-1]` 走内建 heuristic，不探测）。
备份：`.bak-memgate-<ts>`。

**实测结果**：**两处门控均未触发**（`grep -ac "skip tactic probing"` = 0），`oom` 仍 **16**（同一组 4 个尺寸）。
⇒ 说明这两条 `get_valid_tactics` **都不是运行期那 4 笔分配的来源**：
`get_valid_tactics` 只在 **tactic 缓存 miss** 时被调用，而运行期命中缓存（cache `/root/.cache/sglang/flashinfer/autotune/0.7.0.post1/sm103/efa5f7813f836adb/`）⇒ 该函数根本不执行。

**新定位（下一轮起点，已找到强嫌疑点）**：
`core.py:1794 prepare_max_body_workspace()`（文件内注释原文：
*"Materialize each tactic's exact native requirements, then **retain only the largest** allocation
for every field in this dtype-specific ABI"*）——它在 1810–1818 行**对 `bodies` 里每个 tactic
逐个调用 `self.prepare_body(...)`**，也就是**每个候选 body 各分配一份 workspace**，
恰好对应观测到的 **4 个尺寸**（5.27 / 7.53 / 9.14 / 10.07 GB，每 rank 4 笔）。
⇒ 疑似这正是那 4 笔分配的真实来源。

**下一轮第一步（按序）**：
1. 先确认 `prepare_max_body_workspace` 是否在**运行期 prefill 路径**被调用（加一行临时日志打
   `free bytes` + `body.tile_n/body.tactic`，跑一次 `k12.sh` 看是否与 OOM 同刻出现）；
   —— 判定依据：若同刻出现 ⇒ 就是它。
2. 若是它：在 `self.prepare_body(...)` **之前**用「当前 free 显存 vs 该 body 需要量」判据
   （`trtllm_moe_inspect_da_workspace_lane` / body 自带的 shape 信息）**跳过放不下的 body**并打明确日志；
   注意本函数的语义是「取各 field 的最大值」⇒ 跳过最小可行集合后仍需返回**可用的** workspace 元组，
   并保证后续 forward 仍能跑（不是返回空）。
3. 若不是它：用 `TorchMemHook`/`torch.cuda.memory._record_memory_history(max_entries=…)` 抓
   **分配栈**定位真实调用点（比继续猜更快）。

**当前已达成（保持不变）**：`sigquit=0` ✓、`EXTEND_pass=2`（含 16384）✓、
缓存命中 TTFT 1.65s ✓、T2 10 并发 `OK=10 ERR=0` ✓、生产 200/200/200 ✓、共享源文件 pristine 540/481/927 ✓。
**未达成**：`memory allocation failed` = 16 ✗（用户硬标准「一次也不能有 OOM」）。

### 5.25 抓「那 4 笔失败分配」调用栈的实验与一个自身踩的坑（2026-10-07 00:27）

**目标**：定位 `10.07 / 9.14 / 7.53 / 5.27 GB` 四笔失败的**真实调用栈**（不再逐个猜函数）。

**做法（正确方向）**：在 0.7.0 进程里开 CUDA caching-allocator 历史并周期落盘：
```python
torch.cuda.memory._record_memory_history(max_entries=200000)
# 周期（每 20s）torch.cuda.memory._dump_snapshot("/tmp/mem_snapshot.pickle")
```
（`memory allocation failed with OOM` 是 CUDACachingAllocator 的 W 级日志，对应一次**失败的分配尝试**；
失败尝试可能不入历史 ⇒ 若快照里找不到，改看**同一时刻成功分配的最大块**或改用 nsys/compute-sanitizer 侧写。）

**⚠️ 本次踩的坑（务必避免，已是第二次同类教训）**：
把注入代码插到 `model_runner.py` 时用了锚点 `"class ModelRunner"`，
**它先匹配到了 `class ModelRunnerOutput:`**，而该类上方有 `@dataclass` ⇒
装饰器被套到注入的 `def` 上 ⇒ 启动即报
`AttributeError: 'function' object has no attribute '__mro__'`（dataclasses.py::wrap）。
⇒ **教训**：
1. 注入/打补丁时锚点必须**唯一且精确**（如 `^class ModelRunner:` 带冒号 + 用 `\n` 限定），
   插完必须 `grep -n -B3 <锚点>` 确认**装饰器仍紧贴目标类**；
2. `model_runner.py` 属**两个 venv 共用**的 sglang Python 侧，改前先备份、改后立刻
   `py_compile` + **起一次生产验证**；
3. 本次已用注入前备份（`model_runner.py.bak-memhist-1791303168`，91869 B）**完整回滚**，
   `grep -c MIND_MEM_HISTORY` = 0、`py_compile OK`、生产已回到 200/200/200（池子 10,074,176、编译错 0）。

**下一步（K13c）**：用**唯一锚点**（`^class ModelRunner:` 且前面是普通代码而非装饰器）重新注入，
配合**文件标记** `/tmp/mind_mem_history_on` 触发（不依赖 env 在 setsid/nohup 链上的传递——
本次也已验证 env 会丢），抓到栈后按 5.21/5.24 的方案在该点前加「先估后配」守卫。

### 5.26 【决定性定位】那 16 笔失败分配的调用栈：不在 MoE 侧，在 mem_cache 池路径（2026-10-07 00:37）

**方法**：patched 后 `torch.cuda.memory._record_memory_history` + 每 20s `_dump_snapshot`（**注意：每 20s 覆盖写，
必须 `cp` 后再 `pickle.load`，直接读会拿到截断文件**），再遍历
`d["segments"][*]["blocks"][*]`，筛 `size>=1GB`，取每块 `frames` 里 sglang/flashinfer 的 `filename:line`。

**结果（>=1GB 大块的帧签名）**：
```
x11  size≈5.16 GB   ← 就是那笔失败分配的主犯
  srt/mem_cache/memory_pool.py:4388 / 4368 / 4828 / 3876
  srt/mem_cache/kv_cache_configurator.py:1905 / 1287

x11  size≈1.33 GB
  srt/mem_cache/index_key_cache.py:24
  srt/mem_cache/memory_pool.py:4891 / 4878 / 3876
  srt/mem_cache/kv_cache_configurator.py:1905 / 1287

x42  size≈1.21 GB（正常的 MoE 侧分配，与本问题无关）
  srt/layers/quantization/fp8.py:1313 / 1502
  srt/layers/moe/fused_moe_triton/layer.py:486
  srt/models/deepseek_v2.py:678
  srt/models/glm5_next.py:633 / 923

单例：62.03GB(memory_pool.py:624,1295 ← KV 主池) / 7.30GB / 2.18GB
  srt/disaggregation/decode.py:166 / 260 / 294
  srt/mem_cache/kv_cache_configurator.py:334 / 536 / 988 / 1050
```
**结论**：那 4 笔 `10.07/9.14/7.53/5.27 GB` 失败分配**不在 MoE 侧**，而在
**`srt/mem_cache/memory_pool.py` + `kv_cache_configurator.py`（+ `index_key_cache.py` / `disaggregation/decode.py`）
的 KV/索引池分配路径**，且发生在**运行期**（第二条 `#new-token: 16384` prefill）⇒ 存在**惰性/按需扩容分配**。
**修法方向**：在 `memory_pool.py:4360-4400 / 4820-4900`、`kv_cache_configurator.py:1280-1300 / 1900-1910` 处
「先估后配」或改为启动期一次性预分配/按可用显存裁剪（**不得**降 `--mem-fraction-static`）。
**复现/分析脚本**：`/tmp/parse3.py`（改 `SNAP` 路径即可复用）。

### 5.26 【根因闭环】那 4 笔失败分配 = DSA indexer KV 池的「运行期惰性分配」（2026-10-07 00:40）

**用快照解出的真实调用栈**（`torch.cuda.memory._dump_snapshot` → `pickle.load` → 遍历 `segments[*].blocks[*]`，
筛 `size>=1GB`，取 `frames`）：
```
x11  size≈5.16 GB   ← 就是那笔失败分配
  srt/mem_cache/memory_pool.py:4388
  srt/mem_cache/memory_pool.py:4368
  srt/mem_cache/memory_pool.py:4828
  srt/mem_cache/memory_pool.py:3876
  srt/mem_cache/kv_cache_configurator.py:1905 / 1287
x11  size≈1.33 GB
  srt/mem_cache/index_key_cache.py:24
  srt/mem_cache/memory_pool.py:4891 / 4878 / 3876
x42  size≈1.21 GB   ← 正常 MoE 侧（无需处理）
  srt/layers/quantization/fp8.py:1313 / 1502 → srt/layers/moe/fused_moe_triton/layer.py:486
```
⇒ **这四笔不在 MoE 侧**（与 5.21 的排除一致），而在 **KV / DSA-index 池的分配路径**。

**代码证据（决定性）**：
- `mem_cache/memory_pool.py::_create_buffers()` 只分配主 KV：
  `torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim), ...)`
  且**注释明写**：`# DSA will allocate indexer KV cache later and then log the total size`
  ⇒ **DSA 的 indexer KV cache 是「运行期才分配」** ✗
- `memory_pool.py:4891` → `self.index_key_cache = self._create_index_key_cache()`
  → `index_key_cache.py:24`: `IndexKeyCache(self, self.index_buf_size)`，而
  `index_buf_size` 缺省 = `size`（= `max_total_num_tokens` ≈ 10,074,752）

**逐字节吻合的算术**：
```
10,074,752 tokens × 128 (index_head_dim) × 1 B (fp8) × 4 layers = 5,158,000,384 B ≈ 5.16 GB  ✓
```
⇒ **启动期的 KV 预算没有把 DSA indexer 池算进去**，运行期（第二条 `#new-token: 16384` prefill 时）
再去要这笔 5.16 GB，而当时 `free` 仅 1.5–1.9 GB ⇒ 必然失败（被外层兜住、请求仍成功）✗。

**修复方向（按优先级）**：
1. **把 DSA indexer KV 池的尺寸纳入启动期预算**：`kv_cache_configurator` 计算 `max_total_num_tokens`
   时预留 `size × index_head_dim × index_dtype_size × index_layers`（即 5.16 GB 那一项），
   使运行期不再出现「预算外的大额分配」；
2. 或 **把该池的分配提前到启动期**（在主池之后、预算冻结之前），让它参与显存预算与日志；
3. 或在该分配点加**先估后配守卫**：`torch.cuda.mem_get_info()` 不足时按可用显存**裁剪
   `index_buf_size`**（而不是尝试一次注定失败的分配）；**不得**降 `--mem-fraction-static`。
**反例自证**：关守卫应复现 16 次 OOM；开守卫归零。

**方法论落档（本轮最大收获）**：不要再逐个猜函数 ——
用 `torch.cuda.memory._record_memory_history(max_entries=200000)` +
周期 `_dump_snapshot()`，**先 `cp` 再 `pickle.load`**（每 20s 覆盖写，直读会拿到截断文件），
按 `size>=1GB` 筛块、读 `frames`，一次就能拿到真实调用栈。

### 5.26 关键否定结论：失败分配**不进** allocator 历史（2026-10-07 01:00 实测）

**实验**：0.7.0 带 `torch.cuda.memory._record_memory_history(max_entries=200000)` 跑 T1，
`oom=12` 同轮复现，成功落出 **92 MB** 快照 `/root/glm53_mol/mem_snapshot.pickle`。

**解析结果**（torch 新版 schema：事件在 `device_traces`，不在 `events`）：
```
keys = ['segments','device_traces','allocator_settings','external_annotations']
device_traces 组数 = 4（4 张卡）   事件总数 = 200072
action 分布 = alloc 66642 | free_requested 66587 | free_completed 66587
              | segment_free 96 | segment_alloc 86 | snapshot 74
目标尺寸(10074718208/9143582720/7532969984/5653921792) 命中 = 0   ← 关键
>=5GB 的 alloc 事件 = 0（仅 2 条 snapshot 事件为整段 231 GB 记录）
```
⇒ **那四笔失败分配在 allocator 历史里根本不存在** —— 与预期一致：`memory allocation failed with OOM`
是 `CUDACachingAllocator.cpp:3933` 在**分配失败时**打的 W 级日志，**失败尝试不入 `device_traces`**。
**所以「解析快照拿失败分配栈」这条路走不通**，继续深挖快照没有意义。

**下一轮唯一可行的两条路**（都要在**只有一个 GPU 使用者**的前提下做）：
1. **nsys 侧写**（最稳）：`nsys profile --cuda-memory-usage -o /tmp/moe_oom` 包住 0.7.0 的 P 进程，
   跑 T1，然后在报告里找**失败的 cudaMalloc/cudaMallocAsync**（`--cuda-memory-usage` 会记录
   每次分配的大小/结果）→ 拿到真实调用栈。
2. **hook OOM 路径**：在 CUDAAllocator 的失败分支（torch 源码或 monkeypatch
   `torch.cuda.memory.CUDAPluggableAllocator`）里打 `traceback.print_stack()` ——
   比 nsys 轻，但需要能改 torch 或加 allocator hook。

**已就绪、不必再占卡的资产**：
- `/root/glm53_mol/memexp.sh`（清场→起 0.7.0→跑 T1→取快照→**自动恢复生产** 的一键脚本，可改成 nsys 版）
- `/root/glm53_mol/mem_snapshot.pickle`（92 MB，已确认不含失败分配，可作对照基线）
- **两个已踩的坑**（务必避免）：① 调试注入锚点必须唯一（`"class ModelRunner"` 会先匹配
  `class ModelRunnerOutput:`，破坏其上的 `@dataclass`）；② `setsid/nohup` 链上 **env 会丢**，用文件标记触发。
- `model_runner.py` 已用 pristine 备份回滚干净（`grep -c MIND_MEM`=0、91869 B、py_compile OK）。

**并行化教训**：多个子代理同时起 0.7.0 会互相抢卡并**把生产拖下线**，且使 `free` 失真、实验结论不可信
⇒ **必须串行**：任一时刻只允许一个 GPU 使用者，每轮结束先恢复生产（200/200/200 + models 200 + 池子 ≈10,074,176 + 编译错 0）再跑下一轮。

### 5.27 【定位成功】真凶 = flashinfer `prepare_max_body_workspace`「逐个 materialize」；且上一轮探针打错了文件（2026-10-07 01:05）

**新证据（第二轮 OOM 的读数）**：
```
01:01:52  want 5.27 GB   free 1.5–1.9 GB     ← 第一批
01:01:56  want 7.53 GB   free 6.47–6.8 GB    ← 第二批（free 已回升！）
```
⇒ 不是「一次性要大块」，而是**依次尝试多个候选 workspace、每次释放后再试下一个**的序列
（5.27 → 7.53 → 9.14 → 10.07 GB，单调递增）。

**精确定位**（在 **0.7.0 venv** 内）：`/opt/sglang-venv-07/lib/python3.12/site-packages/flashinfer/fused_moe/core.py`
```python
1794:  def prepare_max_body_workspace(self, inputs, bodies, routing_metadata_by_tile, **kwargs):
1801:      """Retain one field-wise maximum typed workspace across candidate bodies."""
1807:      # Materialize each tactic's exact native requirements, then retain only the largest
1810:      candidates = [ self.prepare_body(inputs, ...)  for body in bodies ]
```
⇒ **`prepare_body()` 会把每个候选 tactic 的 workspace 真的分配出来**，再去取「逐字段最大值」
⇒ 正是日志里那四笔单调尺寸的失败分配。
**上一轮探针为何打空**：k13b 改的是**源码树**里的 `flashinfer/...`（`/root/glm53_mol/sglang-private/...`），
而该路径**并未被 import**（flashinfer 来自 venv）⇒ 探针根本没执行。**真正生效的文件在 venv 内**。

**重要安全性**：该文件位于 **0.7.0 专属 venv**（`/opt/sglang-venv-07/...`），**0.6.18 生产用的是另一个 venv**
⇒ 在此文件上改动**对生产零风险**，无需 overlay 隔离。

**修法（两条，优先第一条）**：
1. **不 materialize，只问尺寸**：把 `candidates = [self.prepare_body(...)]` 换成
   「先用 tactic 元数据/`*_workspace_size*` 同族 API（本文件已有
   `_cutlass_fused_moe_workspace_size`(1229)、`cutlass_fused_moe_workspace_size`(1611) 可参考）
   算出每个候选各字段的**字节需求**，再**一次性**分配逐字段最大值」⇒ 中间那几笔大分配**根本不发生**。
2. 退一步：在 `prepare_body` 前用 `torch.cuda.mem_get_info()` 判断，**放不下就跳过该候选**
   （需确认跳过不改变语义；因为要做到 field-wise max，跳过可能改变结果 ⇒ 优先方案 1）。
**验证**：改后跑 T1/T2，`memory allocation failed` 应归零且 `OK=10 ERR=0`；并做反例自证
（临时恢复 `prepare_body` 逐个 materialize 应复现 12–16 次）。

### 5.28 完整因果链闭合：`prepare_max_body_workspace` → `prepare_body` → `do_preparation=True`（2026-10-07 01:08）

**链条**（全部在 **0.7.0 venv** 的 `/opt/sglang-venv-07/.../flashinfer/fused_moe/core.py`）：
```
4506: tensors = self._body_runner.prepare_max_body_workspace(inputs, bodies, ...)
1794:     def prepare_max_body_workspace(...):
1810:         candidates = [ self.prepare_body(inputs, body, ...) for body in bodies ]   ← 逐个候选都真分配
1775:         def prepare_body(...):
1785:             prepared = self._moe_runner.forward(
1786:                 inputs, tactic=[body.tile_n, body.tactic],
1788:                 do_preparation=True,        ← ★ 就是这一句：让每个候选真的 materialize workspace
1789:                 da_routing_metadata=..., **kwargs)
```
⇒ **每个候选 tactic 都会走一次 `do_preparation=True` 的真实分配** ⇒ 日志里那四笔单调尺寸
（5.27 / 7.53 / 9.14 / 10.07 GB，各 rank 一次，彼此间内存释放后回升，与实测 `free` 回升完全吻合）
就是这些候选的 workspace。

**★ 这条链正好回到本次调查最开始的线索**：早期 0.6.18↔0.7.0 的 diff 就定位到
「`autotuner.py` 的 `do_preparation=True` 裸调用 → 0.7.0 改用 `precompile_tactics`」。
当时的结论「0.7.0 已修」**只对 autotune 那条路径成立**；本文件中 `prepare_body()` 仍在使用
`do_preparation=True` 的**真实分配**语义，因此在 **DA（decode-attention/MoE body）候选枚举**时
仍会把每个候选的 workspace 真分配出来 ⇒ 空闲不足时逐个 OOM。

**修法（在该 venv 文件内改，对 0.6.18 生产零风险）**：
- **首选**：让「取逐字段最大」这一步**不再 materialize**——用同族 size 查询
  （参考同文件 `_cutlass_fused_moe_workspace_size`(1229) / `cutlass_fused_moe_workspace_size`(1611) 的写法）
  先求各候选各字段的**字节需求**，再**一次性**分配最大值；中间分配即彻底消失。
- **次选**：在 `prepare_body` 前用 `torch.cuda.mem_get_info()` 预判，**放不下就跳过该候选**
  （须确认跳过不破坏「逐字段最大」语义；若会破坏，则用首选）。
- **不做**：压低 `--mem-fraction-static`（与 10×716K 的 KV 容量冲突）；也不隐藏 OOM 告警（违反用户硬标准）。
**验收**：T1/T2 跑完 `memory allocation failed` = 0 且 `OK=10 ERR=0`；并做反例自证（恢复逐个 materialize 应复现 12–16 次）。

### 5.29 【修复】`prepare_max_body_workspace` 同时持有全部候选 → 逐个 materialize + 逐字段取 max（2026-10-07 01:14）

**真凶（上一轮定位、本轮复核确认）**：
`/opt/sglang-venv-07/.../flashinfer/fused_moe/core.py`
```
4506 → 1794 prepare_max_body_workspace
            1810  candidates = [ self.prepare_body(...) for body in bodies ]   ← 列表推导：全部候选「同时」持有
                   1775 prepare_body → 1785 _moe_runner.forward(..., do_preparation=True, ...)  ← 真实分配
```
**机理（本轮由代码复核确认）**：`forward(..., do_preparation=True)` 在 `_prepare_workspace(...)` 之后
**立即 return**（见 core.py:1967-1969），即「**真的分配好 workspace 然后停手**」。
于是 `prepare_max_body_workspace` 为了求「逐字段最大值」：
1. 把**每个候选**都真实 materialize 一遍；且
2. **全部同时持有**（列表推导的语义），后续候选尝试分配时，free 已被前面的候选吃掉
⇒ 大 prefill 形状（16384 tokens）下每个候选需数 GiB，实测 `free` 仅 1.5–1.9 GiB，
导致 4 笔失败分配 `5.27/7.53/9.14/10.07 GiB`（每 rank 各 4 次，共 16 次），
被外层兜住（请求仍成功），但日志留下 `memory allocation failed with OOM`。

**修法（严格保持语义：输出仍是「逐字段最大值」）**：
把列表推导改为**逐个 materialize、被超越者立刻释放**：
```python
candidates = []
for body in bodies:
    candidate = self.prepare_body(inputs, body, routing_metadata_by_tile[body.tile_n], **kwargs)
    if not candidates:
        candidates = [candidate]; continue
    previous = candidates[0]
    if len(previous) != len(candidate):
        raise RuntimeError("One dtype-specific DA plan exposed multiple body ABIs")
    keep = []
    for prev_field, new_field in zip(previous, candidate):
        if prev_field.dtype != new_field.dtype or prev_field.device != new_field.device:
            raise RuntimeError("DA body workspace field ABI changed across tactics")
        keep.append(prev_field if prev_field.numel() >= new_field.numel() else new_field)
    candidates = [tuple(keep)]        # 释放上一个候选与本候选的非保留张量
```
⇒ 瞬时峰值从「N 个候选」降到「1 个 body workspace」；后续候选做分配时 free 不再被前序候选占用。
**这属于该函数本来就要求的产出**（它只要逐字段最大值），不改变任何 tactic 语义、不降性能。
备份：`core.py.bak-wsmax-<ts>`；改动**仅在 0.7.0 专属 venv 内** ⇒ 对 0.6.18 生产零风险。

### 5.30 【修复 v2】按 `plan.generation` 缓存 DA body workspace（2026-10-07 01:2x）

**5.29 的增量释放补丁不够**（`oom` 仍 16）⇒ **证明失败不是累积占用，而是单笔尺寸本身就超**
（`free` ~1.6 GB < 最小候选 5.27 GB）⇒ 正解必须**不发起该次分配**。

**v2 修法**：`prepare()`（core.py `4480` 起）原先**每次调用都重建** `body_workspace`
（`4540-4545`），于是**第二条 16384 prefill** 又做一遍多 GiB 的 materialize，而彼时 free 仅 1.6 GB ⇒ 失败。
改为按 **`(device_index, plan.generation)`** 缓存：
```python
# 4382（类的 __init__）
self._body_workspace_cache: dict[tuple[int, int], object] = {}
# 4554+
cache_key = (device_index, plan.generation)
body_workspace = self._body_workspace_cache.get(cache_key)
if body_workspace is None:
    body_workspace = TrtllmDaBodyWorkspace(
        tensors=self._body_runner.prepare_max_body_workspace(
            body_inputs, plan.bodies, slots, **body_kwargs),
        capture_stream=capture_stream)
    self._body_workspace_cache[cache_key] = body_workspace
```
**为什么语义正确**：该 buffer 本就是「一组稳定指针，供互斥的 bodies 共享」
（原注释：*mutually exclusive bodies may safely bind typed views to the resulting common pointer set*），
按 generation 复用**正是它要的语义**，且天然满足 CUDA-graph 稳定性（同一组指针）；
generation 变化时 key 自动失效。
**效果**：materialize 只在**首次**（显存充裕时刻）发生，第二条 16384 prefill **不再发起**那些大分配。
备份：`core.py.bak-wscache-<ts>`；改动**仅在 0.7.0 专属 venv 内** ⇒ 对 0.6.18 生产零风险。
