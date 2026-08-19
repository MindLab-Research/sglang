# CP × LoRA 崩溃调查（2026-08-18/19 通宵）

状态：调查中（compute-sanitizer 运行中）· 生产集群 base 模式已恢复可用

## TL;DR

- **CP（prefill context-parallel）+ LoRA 组合 100% 确定性崩溃**；单机 TP=8 无 CP + LoRA 完全正常（对照实验证实）
- **最小复现**：单机 + `--enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit` + LoRA 请求 → 3 分钟一轮稳定复现
- 第一现场（blocking 模式 ×3 一致）：**`forward_mla.py:1041` MLA attention 的 `torch.bmm(out=非连续view)` → CUBLAS_STATUS_EXECUTION_FAILED**（发生在 LoRA delta kernel 之前的 attention 层）
- 纯 PyTorch bmm 同形状全 OK → 问题在 sglang 传给 bmm 的张量在 CP×LoRA 下的布局

## 最小复现步骤（1021 单机）

```bash
bash /root/start_singlecp.sh          # 无 blocking，健康 ~2.5 分钟
# 或 bash /root/start_singlecp_blk.sh # CUDA_LAUNCH_BLOCKING=1
curl :30333/v1/chat/completions -d '{...,"lora_path":"L0"}'  # 必崩
```

## 证据链（按实验轮次）

| # | 实验 | 结果 |
|---|---|---|
| 1 | 集群 PD+CP=8 + LoRA（两集群 FP8/BF16） | 必崩（8 连崩）|
| 2 | MoE-only 权重（剥离 attention LoRA） | 仍崩 → 排除 MLA target 越界 |
| 3 | fp8 量化 LoRA 权重 | 仍崩 → 排除 dtype |
| 4 | 去掉 --lora-use-virtual-experts | 仍崩（classic 路径崩 align）|
| 5 | align kernel → naive CPU | 仍崩 → 排除 align kernel |
| 6 | mapping kernel → naive torch | 仍崩 → 排除 mapping kernel |
| 7 | -1 尾部 guard / tail-stamp | 排除 |
| 8 | mapping 尺寸 CP 上界修复 | 排除（正确性修复，非崩因）|
| 9 | overlap loader 同步化 | 排除 |
| 10 | canary：加载前后全参数 checksum | 无变化 → 排除加载写坏权重 |
| 11 | POSTLOAD-PROBE ×24 | 加载后 context 干净 → 排除加载 |
| 12 | **单机 TP=8 无 CP + LoRA + VE** | **完全正常** ← 关键对照 |
| 13 | **单机 + CP + LoRA** | **必崩**（7/8 rank）← 最小复现 |
| 14 | 单机 + CP + blocking | MLA bmm 第一现场（×3 与集群一致）|
| 15 | 纯 torch bmm（所有维度组合） | 全 OK |

## 当前假说（待 sanitizer 确认）

LoRA 请求路径的某处（prepare_lora_batch / hook 构建）在 CP gather 布局下改写了 MLA attention 输入张量的 shape/stride 语义，或 attention backend 的 workspace 与 LoRA buffer 产生 allocator 交互。fault 在 attention bmm 显式触发（blocking 证实非 sticky）。

## Sanitizer（进行中）

`/root/scpsan.log` — compute-sanitizer memcheck 包裹的单机+CP+LoRA 实例，等 health 后打 LoRA 请求，第一个 `Invalid __global__ read/write` 即真凶 kernel。

## 白天决策

- **A. 保现状**：生产跑 base（已恢复验证 200），LoRA 流量暂停
- **B. 去 CP 保 LoRA**：prefill 移除 3 个 CP flag（`--enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit`）+ env `SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN`——对照实验 12 证明该组合 LoRA 正常；代价 prefill 吞吐降

## 修复痕迹（1021，全部有 .bak）

- `base_backend.py`：KERNEL-GUARD（-1 越界写）/ CP-fix（mapping 上界）/ TAIL-STAMP / naive-mapping
- `lora_moe_runners.py`：naive-force（align CPU 化）
- `lora_overlap_loader.py`：sync-load（加载 stream 同步化）
- 权重：L0/L2 = MoE-only（原始在 `.full_bak`/`.bf16_bak`）
- 本地仓库：base_backend CP-fix + kernel-guard、lora_overlap_loader sync-fix（未 commit）

## 关键路径

- 集群崩溃日志：1101 `/root/prefill.log.crash*` / `.bak.*`；1021 `/root/scpblk.log`
- 单机实验脚本：1021 `/root/start_singlecp*.sh`

---

## 2026-08-19 上午增补：独立复现 + 三层修复 + 残余

### 方法突破：30 秒级独立 kernel 复现（`1021:/root/mini_repro.py`）

绕过 6 分钟/轮的服务器循环，直接调 `merged_experts_fused_moe_lora_add`，注入各类垃圾值。

### 已修复并独立验证（commit `16a7a569df`）

1. **VE kernel 双界钳制**（`virtual_experts.py`）：
   - `token_lora_mapping` id ≥ max_loras → -1（陈旧 graph 池/padding 垃圾）
   - base topk id ∉ [0, num_experts) → -1（**负向垃圾是致命类**——原生 align 拿 id 做 atomic 偏移，-2e9 → 野指针 → Xid 31 FAULT_PDE）
   - 复现结果：4 类垃圾 × 5 轮全过（修前 negonly/garbage_pad 确定性崩）
2. **CP gathered 尺寸**（`base_backend.get_gathered_moe_num_tokens` v2）：CP 下 MoE 跑 gathered token 集（=Σ各 rank），但 `global_num_tokens_cpu` 只存本 rank 值 → mapping 分配 8、kernel 索引 48 → OOB。实测 DIAG：nt=6 moe_nt=8（错）→ 修复后 64 ✓
3. **mapping/seg/lora_id 消毒**（`_compute_moe_lora_info_kernel` + SANITIZE-MAP）：SEG-CLAMP 钳 seg 到 [0, mapping_len]，lora_id 上界，host 端 masked_fill_

### 服务器级残余（CP×VE 路径，另案未解，已出圈）

> ⚠️ 定性修正（2026-08-19 下午）：本节残余是**单机 CP×VE 路径**的问题，与 PD 集群崩溃（已解，见下节）**不是同一根因**。用户决策 BF16 不开 CP，本残余出圈挂起。

全部消毒生效后（eager、干净 M=6、尺寸正确、权重 checksum 不变），仍崩：**layer-0 gate GEMM 报 `cudaErrorIllegalAddress`（CP0 rank、VE impl 之前、Xid 先于 Python 异常 16s）**。已排除：graph 池垃圾（全部钳制）、尺寸、加载污染（canary+probe 24 次干净）、`.cu` align kernel（独立测干净+naive 绕过仍崩）、torch 分配器冲突（LoRA 池=torch.zeros）、cublas workspace。**指向 CP attention 路径与 LoRA 权重加载的更深层交互**（CP0 是 layersplit broadcast owner）。

### 铁的对照事实

- **无 CP + LoRA（VE 路径）：全链正常**（单机 + PD 集群双验证）——可服务 LoRA 的形态
- CP + base：正常（生产一直跑）
- CP + LoRA（单机 VE 路径）：崩（上述残余，另案）
- PD prefill（无 VE flag，classic 路径）：崩——**即上一晚所有 PD 崩溃的真因，见下节**

---

## 2026-08-19 下午结案：PD 集群 LoRA 崩溃 = 启动脚本 flag 缺失

### 根因

`1021:/root/start_glm52_bf16_pd.sh` 的 **prefill 块缺 `--lora-use-virtual-experts`**（line 46 只到 `--max-loras-per-batch 2`，留有空续行 `\` 残迹——曾有后被删）。无 flag 时 `_add_lora_down_delta` 走 **classic `fused_moe_lora` kernel 路径**（`fused_moe_lora_kernel.py`，`expert_id` 无 clamp 直接索引权重，无任何加固）→ 首个 LoRA 请求 CUDA IMA。

**为什么排查了 20 小时**：所有单机对照测试脚本（`start_singlecp_*.sh`）都带 VE flag 走**加固过的 VE 路径**，全部通过——对照实验从设计上就无效。"单机过、PD 崩"的表象诱导我们排除了 CP/layersplit/HiCache/cuda-graph/VE 垃圾/双挂载等所有真凶之外的方向。崩溃 batch 的 nt=4/nt=23 形态与单机通过的测试完全相同，路径差异才是唯一变量。

### 修复与验证

- 修复 = 脚本补回 `--lora-use-virtual-experts --max-lora-chunk-size 128`（sed 一处，prefill/decode 两块生效；备份 `.bak.nove_*`）。仓库 `start_pd.sh`/`recover_b300_pd.sh` 本来就正确。
- 按 runbook 双端 + smg router 一起重启（只重启 prefill 会导致 decode mooncake 会话过期 → `KVTransferError: Aborted by AbortReq`；router 熔断器记死旧 prefill → 503）。
- 验收：11/11 请求通过（base/L0/L2 混合轮 8/8），prefill/decode 均 0 Scheduler hit，temp=0 下 L0 输出与 base 逐字不同（LoRA 真实生效）。

### 附带发现

- **colon 语法不生效**：`model: "glm52-bf16-pd:L0"` 被原样当作 model 名处理，输出与 base 相同；LoRA 正确入口是请求体 `lora_path: "L0"` 字段（MoL harness 侧注意）。
- `lora_overlap_loader.py` SYNC-LOAD 修复（load_stream 竞态 → 权重损坏 → sticky CUBLAS 失败）已随本结案 commit。

### 教训（方法论）

1. **对照实验前先 diff 两边启动 flags 逐条对比**——"同代码"假设在本案是假的，一个 flag 切换了整条 kernel 路径。
2. 空续行 `\` 残迹是 flag 被删的物证；脚本编辑后 `grep -n` 关键 flag 应成为重启 checklist 项。
3. PD 重启必须双端+router 成套（AGENTS §5.3 铁律的又一实例：health 200 ≠ 可服务）。

### 1021 遗留补丁状态（远端 venv，均有 .bak）

- ve.py `ff62a067`→`de2922a9`(+DIAG-W)：双界钳制 ✓（已 commit 本地）
- base_backend `62296778`：CP-fix v2 + SEG-CLAMP + SANITIZE-MAP + DIAG-PREP（已 commit 本地）
- lora_overlap_loader：SYNC-LOAD 修复 + CANARY 探针（远端版）；**干净版已 commit 本地**
- lora_moe_runners `a2e2aa01`：ALIGN-NATIVE-NAIVE（.cu align 绕过，env `SGLANG_LORA_ALIGN_CUDA=1` 可回开）——未 commit（未证明必要）
- mem_pool/lora_manager：残留 PHASE-LOG 打印（无害，一次性）
