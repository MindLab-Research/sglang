# sglang-mol-v41 — DeepSeek-V4.1-Flash + MoL port

## 实验报告铁律（2026-10-15 用户下令，违反=死罪级事故）
1. **输出正确性先于一切性能数字**：报告任何实验前必须先确认输出**语义正常**（不是垃圾）。输出垃圾 = 正确性有问题 = 代码写错了 = **实验无意义，禁止上报**。拿一眼就是错的东西说"做完了"= 严重事故。
2. **比较口径 = verify step time（per-step）**：TPOT × accept_len = step_time。一切结构/开销对比必须用 per-step 口径（去掉 accept_len 影响）。acc_len 只在最终 6ms 换算时使用（TPOT = step/acc）。
3. **层范围 20-39 永不改**（口径铁律）。任何实验不得减层/加层。
4. **逻辑自洽性检查**：实验结果违反基本逻辑（如"去掉计算反而更慢"且无 acc 变化解释）时，必须先怀疑测量/实验本身错了（跨轮对比/配置漂移/加载失败），不得直接编理论解释。
5. **同轮 A/B 才可比**：跨轮/跨进程的对比不可作为结论依据。对比实验必须同一进程内连续跑。
6. **报告前自检清单**：输出正常？请求成功率？层配置正确？适配器真的加载了？——任何一项不过，数字不许往外报。
7. **出错必须先找根因，禁止盲目重启**（2026-10-15 用户下令）：不允许不分析原因就重启反复撞同一个错误。每次重启前必须能回答「上次为什么挂、这次改动如何避免」。

**仓库铁律：我们用自己的【私有仓库】`MindLab-Research/sglang-private`，不是 fork。**
**分支命名铁律：所有模型的「稳定/发布分支」一律以 `release/` 开头。**
例：`release/glm53flash-*`（GLM-5.3-Flash）、`release/dsv41flash-*`（DeepSeek-V4.1-Flash）。
`feat/`、`fix/` 只用于临时开发分支；不要用 `main` / 模型名裸分支承载发布。

老仓库 `MindLab-Research/sglang` 已废弃，**不要往那边推**。基线 = 上游 `main` 81f27fb3a7，
本地增量**直接提交到私有仓库**（remote `origin` = `https://github.com/MindLab-Research/sglang-private.git`）。

**本文件所在树 = `/home/smith/src/sglang-mol-v41`（DeepSeek-V4.1-Flash + MoL 线），当前分支 `release/dsv41-flash-mol`。**
**1101 盒子的 `/root/dsv41_upstream/sglang` 就是本树 `python/sglang/` rsync 出去的（本地 `python/sglang/srt` → 盒子 `sglang/srt`、本地 `python/sglang/kernels` → 盒子 `sglang/kernels`）。DSV4.1 的一切优化只认【本树 + 1101 盒子】，不要去翻别的树。**

两条模型线的本地树，别搞混（2026-10-13 用户下令写死）：

| 本地树 | 分支 | 模型线 |
| --- | --- | --- |
| `/home/smith/src/sglang-mol-v41`（**本文件所在**） | `release/dsv41-flash-mol` | **DeepSeek-V4.1-Flash + MoL**（跑在 1101 盒子） |
| `/home/smith/src/sglang-private`（pr17 worktree） | `pr17` | GLM-5.3-Flash / MoL / LoRA 线 |

> ⚠️ pr17 树里那份**未跟踪**的 `kernels/ops/moe/trtllm_lora_temp/{fused_moe_lora_all,m1_gate}.py`
> 是 DSV4.1 融合核攻坚的本地镜像（10-12 攻坚的热修改主要在 1101 盒子上，尚未回灌）。
> 见到它**不要**误判「DSV4.1 工作在 pr17」；收口时把盒子热修改回灌到**本树**提交。

> 📖 **PD/Mooncake KV 传输的完整机制（超时表 / session 判死 / deferred-release / detokenizer 挂的根因链）见 `docs/agent/disaggregation-mooncake.md`（2026-10-15 核实）。排查传输类问题先读它。**

**PD/KV 传输铁律（2026-10-10 用户下令写死）——**
**单台机器内的卡间 KV 传输必须走 NCCL。`--disaggregation-transfer-backend` 一律用 `nixl`（该版本的 nixl 后端即 NCCL/UCX 路径），严禁 `mooncake`。**
- mooncake 走 RDMA/IB：本机没指定 IB 设备时日志为 `No IB devices specified for Mooncake backend, falling back to auto discovery` ⇒ P2P handshake 建不起来。
- 症状：请求卡在 `KVPoll.Bootstrapping`，直到 **300s 超时** ⇒ `KVTransferError(bootstrap_room=...): Request ... timed out after 300.0s in KVPoll.Bootstrapping`；请求被 abort/重试（实测表现就是"一笔 500k 请求卡 5 分钟多才成功"）。这是 P0，不是网络问题。
- 附：prefill 与 decode 在**同一台机器**时，禁止共用同一个 `--disaggregation-bootstrap-port`。

> ⚠️ **实测例外（2026-10-10 22:13，血的教训）**：**DSV4.1 + DSpark 的 PD 无法用 `nixl`** —— SGLang 自身硬校验直接拒绝启动：
> ```
> ValueError: DeepSeek-V4.1 DSpark PD requires static verify, Mooncake, DP=1 and CP=1.
> ```
> （`srt/arg_groups/deepseek_v4_hook.py::validate_deepseek_v41_features`）
> ⇒ 该组合下**只能** `Mooncake`；要真正走 NCCL，需另想办法（例如不带 DSpark 跑 nixl，或改用 `--disaggregation-ib-device` 把 mooncake 的 RDMA 路径配通）。
> **教训**：改传输后端前**先跑一次 `--help`/启动 dry-run 验证该 backend 被模型接受**，不要直接重启——否则会把服务打挂（本次 30100/30200 双挂约 22 分钟）。

> ⚠️ **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 会杀死 Mooncake PD（2026-10-15 R50 实锤）**：
> 它让 PyTorch 的 CUDA VA 段可重映射，而 mooncake RDMA 的 MR 注册依赖稳定 VA ⇒ prefill↔decode session
> 双向死亡，日志形态：`KVTransferError: remote mooncake session 10.0.58.56:xxxx is not alive`
> （10.0.58.56 = 1101 自身接口 IP，见 `transfer_engine_impl.cpp:182` 监听行——**不是外部串扰**）。
> 三发全部 500、bench 全 503，但 P/D health 都是 200（进程活着，传输死了）。
> 该参数原本是给 v3 解碎片的——**v3 已死，此参数永久禁用**。

> ⚠️ **`KVTransferError: Aborted by AbortReq` 是结果不是原因（2026-10-15 R51 教训）**：
> bench 客户端超时断连 → 服务端 abort 请求 → KV 传输中止报错。排查 256 长跑雪崩时，
> 248 条 KVTransferError 全是 AbortReq 形态——**先看客户端超时与真实初因（acc_len 崩塌/
> prefill 排队），别把 Abort 流水当传输故障修**。

## 投机解码：模型**自带 MTP / dspark**，永不依赖外部 draft 权重

我们线上用的所有模型都**自带 MTP（nextn）预测层**或 **dspark** 头；投机解码的
draft 能力由**基座权重内建提供**，**绝不依赖外部 draft / EAGLE 权重**。

- 配置直接可见：如 `glm53_flash` 的 `text_config.num_nextn_predict_layers = 1`
  （并含 `index_share_for_mtp_iteration = True`）。
- 仓库内建实现：`srt/models/deepseek_v4_nextn.py`、`deepseek_nextn.py`、
  `deepseek_v4_dspark.py`、`dspark.py`、`glm4_moe_nextn.py`、`exaone_moe_mtp.py` 等。

**硬规矩（红线）**：**不要再提「需要外部 draft 权重」「去找一份 EAGLE draft」**——
这类建议一律无效、已被明确否掉。要开投机解码，只走**基座自带的 MTP/dspark 路径**
（`--speculative-algorithm` 选内建算法如 `NEXTN` / `DSPARK`），先查模型 `config.json`
与仓库内建实现，**不要在外部找权重**。

## What lives here

| Feature | Status | Entry points |
| --- | --- | --- |
| DSV4.1 inference (MXFP4 experts, SM100 trtllm-gen) | verified | stock upstream path |
| attention LoRA (q_proj/…) | verified | upstream `chunked` backend |
| `kv_shared` LoRA (adapter shares the base radix namespace) | verified | `lora_config.lora_kv_shared`, `lora_manager.is_lora_kv_shared`, `scheduler.resolve_lora_kv_shared` |
| MoE LoRA on the FP4/trtllm-gen fused path | verified (zero==base bitwise; random-adapter dynamic load) | `trtllm_lora_temp/`, `experimental_sgl_trtllm` backend |

## 1101 layout

- code `/root/dsv41_upstream/sglang` = **本树 `python/sglang/` 的 rsync 目标**（`srt/`、`kernels/` 同构；盒子是扁平包根，`PYTHONPATH=/root/dsv41_upstream`）
- **rsync 前先对账：盒子上有攻坚热修改（如 10-12 的 `kernels/ops/moe/trtllm_lora_temp/fused_moe_lora_all.py`、`m1_gate.py`、`lora_tuning_config.py` 的 `[sdiag]` 行），盲目 local→box 会把它们抹掉。用 `rsync -rcn -i`（checksum dry-run）先列差异，盒子新文件先拉回本地再动。**
- **after every rsync: `find /root/dsv41_upstream -name '__pycache__' -type d -exec rm -r {} +`** (stale .pyc masks updates)
- **重启测试前必须轮换 smg API key**（2026-10-13 用户下令）：`/root/dsv41_upstream/smg_api_key.txt` 换新 key，旧 key 追加记录到 `/root/smg_api_keys_history.txt`——外部还有人拿老 key 打流量（实测 decode GPU 90%+ 是外部流量），不换 key 测量数据被污染。bench 客户端用同一个新 key。
- launcher: `exp_l20.sh`（标准 LoRA）/ `exp_l20_dg.sh`（deep_gemm 变体，当前启动会失败，见下文）；weights `/mnt/workspace/DeepSeek-V4.1-Flash`
- HiCache on V4.1 needs an explicit `--hicache-ratio` (auto-sizing does not support `DeepSeekV4TokenToKVPool`)
- MoE LoRA needs `SGLANG_EXPERIMENTAL_LORA_OPTI=1` + `gate_up_proj down_proj` in `--lora-target-modules` + `--lora-use-virtual-experts`

## MoE LoRA on MXFP4 / trtllm-gen (the hard part)

The non-LoRA path applies flashinfer trtllm directly (no MoeRunner → no LoRA
hook). LoRA goes through the experimental `experimental_sgl_trtllm` backend,
whose fused func dispatches to the **decomposed** kernels in
`lora/trtllm_lora_temp/lora_dispatch.py` and `kernels/ops/moe/trtllm_lora_temp/`:

```
routing -> gate_up grouped GEMM (raw 2*inter)
        -> activation: += gate_up_lora_delta (pre-activation) + capture activation_lora_input
        -> quant -> down grouped GEMM -> finalize -> += down_lora_delta
```

Non-obvious things that broke (each cost a 30-min reload to find):

1. **The activation must carry the OAI/GPT-OSS gated-SwiGLU controls.**
   V4.1 runs its base MoE with `gemm1_alpha=1.702, gemm1_beta=1.0,
   gemm1_clamp_limit=7.0` (`xGlu=clamp(xGlu,max=limit)`,
   `xLinear=clamp(xLinear,±limit)`, `out=xGlu*sigmoid(alpha*xGlu)*(xLinear+beta)`).
   The ported FP4-LoRA activation had dropped flashinfer's `gatedSilu` and did
   plain `silu(xGlu)*xLinear` → LoRA requests would diverge from base. Restored
   in `dev_kernel.cu` (`gatedSiluScalars`/`gatedSiluLoRA`) and threaded through
   `DevKernel.h` (Data/KernelParams), `runner.cu` (setOpsData) and
   `core.py` (`trtllm_fp4_block_scale_routed_moe_lora`).
2. **The per-expert scalars are read at index 0.** sglang's MXFP4 prep fills
   every expert with the same value, and the LoRA activation kernels do not
   carry the permuted→expert map (`ctaIdxXyToBatchIdx`/`tileTokensDim`).
3. **LoRA forces STANDARD topk.** The FP4 path otherwise uses BYPASSED
   (router_logits; the kernel routes internally), but the MoE-LoRA kernels need
   materialized `topk_ids` for the virtual-experts delta routing — and
   pass-through (no-adapter) requests on a LoRA-enabled server hit the same
   routed path. `deepseek_v2._lora_forces_standard_topk()` switches the format.
4. **The routed branch must accept gated silu**, not just situ
   (`flashinfer_trtllm._fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`), with
   the OAI triple + FC1/FC2 biases (situ keeps its clamp_limit-as-beta mapping
   and no biases).
5. **`Mxfp4FlashinferTrtllmMoEMethod` has no `hidden_size`** (it delegates to
   the wrapped FP8 method, no `__getattr__`) — derive it from
   `w2_weight.shape[1]`.
6. **`create_moe_runner` must build a MoeRunner under LoRA**
   (`MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM`); the non-LoRA branch leaves
   `runner=None`.
7. **`normalize_gate_up_proj` must not repeat-stack 3-D MoE weights.** Its
   synthetic up-half repeat (x2 on the second-to-last dim) exists for 2-D
   PEFT adapters ([r, hidden] -> [2r, hidden]); applied to a 3-D shared-outer
   MoE weight ([1, 2r, hidden], already stacked per the mem_pool contract) it
   doubled the rank dim and failed the buffer shape check ([1, 32, 5120] ->
   [1, 64, 5120] on V4.1). Only 2-D weights get the repeat.
8. **The ported O7/O8/O9 two-stream overrides assume the TRITON LoRA
   backend** (`lora_backend._sgemm_info()`). V4.1 runs ChunkedSgmvLoRABackend
   (csgmv), which has no such hook — guard every entry with
   `hasattr(self.lora_backend, "_sgemm_info")` so they fall back to the stock
   forward on csgmv-style backends (mirrors deepseek_mla_correction.py).
9. **V4.1's LoRA fallback with STANDARD topk + gated-silu dereferenced
   `router_logits.to(...)` (None)** — route standard-topk non-situ requests
   through the kernel's routed entry point up front
   (`flashinfer_trtllm._fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`).
10. **LoRA must slice adapter weights by the UNPADDED per-rank width.** The
   base FusedMoE pads `intermediate_size_per_partition` to a multiple of 128
   for the trtllm-gen kernel (V4.1: 2304/4=576 → 640), but LoRA buffers use
   the model's unpadded width (get_hidden_dim). Slicing by the padded value
   fails the buffer check ([16, 640] vs [16, 576]; the last rank even
   truncates silently to [16, 384]). FusedMoE now keeps the raw
   `intermediate_size`; FusedMoEWithLoRA recomputes from it.
11. **The overlay is JIT-compiled; cache freshness is ninja/mtime-based, not
   content-hash-based.** Verified on flashinfer 0.6.18: `JitSpecNvcc.build_and_load()`
   → `try_load()` returns None for the (non-AOT) JIT path → `build()` →
   `write_ninja()` + `run_ninja()`; no `flashinfer_jit_cache` package / no
   `flashinfer/data/aot` dir ⇒ `is_aot` is False, so there is **no AOT wheel** —
   the `.so` is always built from source at first import. The ninja recipe uses
   `deps = gcc` + depfiles, and `.ninja_deps` DOES track headers: `ninja -t deps`
   lists `.../fused_activation_mxfp8_quant.cuh` as a VALID dep of
   `csrc_trtllm_fused_moe_kernel_launcher.cuda.o` (~1498 deps). So editing a
   `.cuh`/`.cu` with a NEWER mtime recompiles just that object and relinks
   (~50s for the launcher object; the ~471s cold-build figure is a full
   rebuild). Two mtime failure modes remain: (a) an rsync that preserves an
   OLDER source mtime leaves the stale `.so`; (b) a NEWLY added `#include` is
   not yet in `.ninja_deps`. Safe route after any overlay source change: delete
   the module's `.o`/`.so` (or rename the `.so`) in the **LIVE** cache dir
   `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>/cached_ops/
   sgl_fused_moe_trtllm_sm100/` (1101 box:
   `/mnt/workspace/ferrite-env-backup/bbuf-flashinfer/.cache/...`) — **not**
   `~/.cache/sglang/.cache/flashinfer` (that holds a stale tree's build).
12. **The trtllm-gen cubin package must be downloaded.** The 1101 box's
   flashinfer pip package has the checksums.txt manifest but NOT the actual
   .cubin files (only downloaded on demand). The decomposed FP4 MoE-LoRA
   path needs the `batched_gemm` cubins (1799 files, ~50MB total) — download
   them from `edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-
   public-generic-local/<hash>/batched_gemm-*/` per the checksums.txt list.
13. **V4.1's gate_up GEMM needs E4m3 activations, not internal E2m1 quant.**
   The overlay's `FP4BlockScaleLoraLauncher` originally fed raw bf16 (path 3,
   internal NvFP4 quant), which selects `Gemm2::Runner(E2m1, E2m1, BF16)` —
   the E2m1×E2m1 tile-8 cubin does not exist ("No kernel found"). Pre-
   quantize with `_prepare_flashinfer_mxfp8_activations` (same as the
   non-LoRA path) and feed `Gemm2::Runner(MxE4m3, MxE2m1, BF16)` — the
   `Bmm_MxE4m3_MxE2m1*` cubins exist. The overlay now branches on input
   dtype; the pad/unpad intermediate bridge lives in the dispatch.

14. **Zero-LoRA == base requires pass-through unification.** The old
    lora_dispatch early-returned no-adapter requests to the stock fused
    kernel (`_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`) — the fused
    kernel's fp32 GEMM1 epilogue never materializes the gate_up output while
    the decomposed Gemm2::Runner writes bf16, so every layer rounded
    differently and greedy decoding always diverged (structurally impossible
    to pass). The dispatch now routes no-adapter requests through the SAME
    decomposed pipeline with a zero delta (VE calls gated on
    `has_active_lora`; pass-through `zero_()`'s the delta buffer): a
    zero-weight adapter's VE delta (exactly 0.0) and a pass-through zero are
    bit-identical into x + 0.0 -> x, so zero==base is now BITWISE. Zero
    measurable perf cost (149 vs 147 tok/s single-request decode).
15. **The E4m3 activation quant must run through the fused kernel**
    (`fused_activation_mxfp8_quant.cuh`). The unfused chain materialized the
    activated row as bf16 before invokeMxFP8Quantization — one extra RN
    rounding whose perturbation shifts each 32-block amax and flips UE8M0
    exponents at the PosInf rounding boundaries. The fused kernel replicates
    `gatedSiluScalars` element-for-element but quantizes straight from fp32
    registers with the exact `cvt_warp_fp16_to_mxfp8` recipe (fp32 amax,
    `__nv_cvt_float_to_e8m0` PosInf, +/-448 saturate, RN E4m3, swizzled SF
    via `get_sf_out_offset_128x4/8x4`). Beware:
    `__nv_cvt_float_to_fp8` returns a raw storage byte — construct via
    `__nv_cvt_float2_to_fp8x2` like the stock kernel or it won't compile.
16. **UE8M0 activation scales need the dedicated gather+swizzle kernel.**
    `moe::dev::permute`'s only scale path (useDeepSeekFp8) indexes a FP32
    128-block LINEAR layout (idx = token + numTokens*(hidden/128)); feeding
    it the UE8M0 32-block tensor reads OOB and scatters garbage into the SF
    buffer — the gate_up GEMM then scales by noise (output was tokenizer
    garbage). Use `sgl_gather_swizzle_ue8m0_sf` (R128c4/R8c4 — same formulas
    as dev_kernel getSfOffset / get_sf_out_offset_*) and permute the E4m3
    data with useDeepSeekFp8=false.

## LoRA 作用域：`--lora-target-layers`（两种形态都支持）

线上有两种适配器形态，**同一个引擎都要支持**：

- **全量**：适配器覆盖全部 40 层（Tinker `all-linear`，722 张量，层 0–39）。
- **后 20 层**：适配器只覆盖 20–39（`dsv41_moe_rnd` / `dsv41_moe_zero`，120 张量）。

**后 20 层形态下，0–19 层必须零开销。** 用 `--lora-target-layers 20-39` 开启（不填 = 全量，行为不变）：

1. `srt/arg_groups/fields/lora.py`：新增 `lora_target_layers: Optional[str]`。语法 `20-39` / `20,21,39` / `all`；层号与 HF 模块名一致，即适配器 safetensors 的 key 索引。
2. `srt/lora/lora_manager.py`：
   - `_parse_lora_target_layers()` 解析 → `self.lora_layer_ids`，并在启动日志打印 `LoRA layer filter ACTIVE: ...`；
   - **`init_lora_modules` 是唯一的包装入口**，在此做层过滤：层不在范围 ⇒ `continue`，该层保持原生 `FusedMoE`/线性层；
   - **加载期守卫**（`validate_new_adapter`）：适配器权重落在范围外 ⇒ **响亮 `raise ValueError`**，不静默降级。
3. 启动器：`/root/exp_l20.sh`（后 20 层模式）；`/root/exp_thinking.sh`（全量）。

**为什么是「零开销」（机制，不是估计）**：LoRA 走分解链的**唯一前提**是该层被包成 `FusedMoEWithLoRA`。层不在范围内 ⇒ 根本不包装 ⇒ 永远是原生模块 ⇒ 走 stock 融合核：**不分配 LoRA buffer、不做 VE delta、不进分解 GEMM**。不是「算出 0 再加进去」，是结构上不进入这条路径。

**它修掉的浪费**：`has_active_lora` 是 **batch 级**的（`lora_manager.py` 的 `any(lora_ranks[wi] > 0)`）。旧实现把**全部 40 层**都包成 LoRA 模块，于是只要 batch 里有一个请求挂适配器，**没有权重的层也照跑整条分解链**（delta 恒 0）。后 20 层适配器在旧实现下白付 20 层 —— 这正是 `--lora-target-layers` 消除的东西。

**⚠️ 实测坑（PD 分离）**：适配器必须在 **P(30100) 与 D(30200) 两侧都加载**（`/root/load_adapters.sh` 本来就是两侧循环）。只加载一侧 ⇒ 经 smg 的请求会**全部挂到客户端超时**（实测两次：256/256 bad、wall=720s），而服务端日志里**看不到明显报错**，极易误判成"改动把 LoRA 打挂了"。判据：`grep "loading completes" pd_decode.log` 应有记录。

**⚠️ 对比口径铁律（2026-10-14 用户下令）**：任何 LoRA 模块/层隔离实验（`--lora-target-modules` 变体、gate_up-only / down-only 等）必须保持 `--lora-target-layers 20-39` 与历史基线一致——**别拿全量层挂载的新数字跟后 20 层历史数据比**（口径错位 2ms 级，结论全废）。隔离实验只准改 `--lora-target-modules`，`--lora-target-layers` 一律不动。

**⚠️ `SGLANG_LORA_TWO_STREAM` 是死变量**：全树只有 6 处注释提到它，**没有任何 `os.environ.get` / `EnvBool` 读取点**。双流真正的开关是 `SGLANG_EXPERIMENTAL_LORA_OPTI`（`lora/layers.py` 里 `install_two_stream_overrides`）。启动脚本里写 `SGLANG_LORA_TWO_STREAM=0` 关不掉任何东西。

## Verification recipes

- `test/registered/scripts/test_kv_shared_lora.py` (+ `make_kv_shared_adapters.py`)
- `test/registered/scripts/test_kv_hicache.py`, `bench_kv_shared_tpot.py`
- MoE LoRA: `make_zero_moe_lora.py` + `test_moe_lora_v41.py` — **a zero-weight
  adapter must reproduce the base output exactly**; that is the cheap numerical
  parity check for the decomposed path (activation + quant + deltas).

## LoRA 性能攻坚：开销三分、两把死刀、一把活刀（2026-10-12 实测）

### 开销三分（conc=64 decode，客户端聚合口径）
base **5.253 ms/tok** vs LoRA **7.632 ms/tok**（差 2.379ms）。按 kernel 做差（base 980.0ms → rnd 1249.4ms，两侧 attention calls 同为 1720 ⇒ 可比）：

| 桶 | 占比 | 归属 |
| --- | --- | --- |
| `t128x8x512_s3` GEMM 家族 | **63%** | MoE GEMM（LoRA 207.94/178.85 µs/call vs 基座 `t128x16x256_s6` 130.94/76.00） |
| `_chunked_lora_{shrink,expand}` | **21%** | **注意力 LoRA**（`kernels/ops/gemm/chunked_sgmv_{shrink,expand}.py`，`ChunkedSgmvLoRABackend`） |
| 其余（VE delta 等） | 15% | MoE VE + 杂项 |

⇒ 单砍 63% 只到 ~6.12ms；**要 <6.0ms 必须同时动那 21%**。

### 死刀 ①：runner options 不可调
`trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu:377-435` + launcher 实参：`useShuffledMatrix=true`、`weightLayout=MatrixLayout::MajorK`、`usePerChannelScaling=false` **全硬编码**；`usePerTokenScaling = !is_e4m3_input`；`epilogueTileM/routeAct/transposeMmaOutput` 硬编码。
**`fusedAct` 被 delta 的数学位置强制 false**（源码原话：`// FP4 LoRA forces unfused activation so the gate_up LoRA delta can be added pre-SwiGLU.`）。
**物理根因**：`fusedAct=false` ⇒ GEMM 必须落 **bf16 gate_up（宽 2I，2B）**；基座融合路只落 **fp8 激活（宽 I，1B）** ⇒ **epilogue 写带宽 4×**。配置层旋钮已穷尽（tile 16/32/64/128、config 0..31、PDL、stage-cfg、消 pad 全否决）。

### 死刀 ②：cuBLAS 在 V4.1 永不生效
`SGLANG_OPT_LORA_CUBLAS{,_A,_B,_GATE_UP,_QKV,_KV_B}` 读取点全在 `kernels/ops/gemm/trtllm_lora_temp/{sgemm_lora_a,sgemm_lora_b,qkv_lora_b,gate_up_lora_b,kv_b_lora_absorbed}.py`；而 V4.1 的 `ChunkedSgmvLoRABackend.run_lora_{a,b}_sgemm` **只调** `chunked_sgmv_lora_{shrink,expand}_forward` ⇒ 那些 cuBLAS 分支**永不被调用**。同理 O7/O8/O9 双流 overrides 在 csgmv 上被 `hasattr(lora_backend,"_sgemm_info")` 挡掉、退回 stock forward。

### ★ 活刀：CSGMV 调参表缺失 ⇒ 注意力 LoRA 全跑未调优默认
`kernels/ops/gemm/lora_tuning_config.py` 加载路径：
```python
config_dir  = os.environ.get("SGLANG_LORA_CONFIG_DIR", <gem 目录>)
configs_root= os.path.join(config_dir, "csgmv_configs")
version_dir = f"triton_{triton.__version__.replace('.','_')}"
path = os.path.join(configs_root, version_dir, f"lora_{kernel},K={K},R={R},S={S},device={dev}.json")
```
**现状**：`csgmv_configs/triton_3_5_1/` 里**只有 `device=NVIDIA_H200`、`R=64`、K∈{1024..6144}**；本机是 **`device=NVIDIA_L20D`、`triton 3.7.1`、R=16** ⇒ 三重不匹配，服务日志对**全部 4 个形状**都报 `no tuned config, using defaults`：

| 形状 | 默认（未调优） | H200 获奖值（仅参考） |
| --- | --- | --- |
| shrink K=5120/576, R=16 | `BLOCK_N=16, BLOCK_K=256`，无 warps/stages | `BLOCK_N=64, BLOCK_K=64–128, warps 4–8, stages 3–4` |
| expand K=5120/1152, R=16 | `BLOCK_N=64, BLOCK_K=16`，无 warps/stages | `BLOCK_N=64, BLOCK_K=32, warps 4–8, stages 1–3` |

⚠️ **禁止盲抄 H200 表**：H200 是 R=64（shrink 的 `N=S·r`≥128），我们是 R=16（`N=S·r`=16/32/48）⇒ shrink 的 BLOCK_N 必须按我们的小 N 重调。
调参器：`kernels/ops/gemm/tune_csgmv_l20d.py`（自写；仓库自带 `benchmark/kernels/lora_csgmv/tune_lora_csgmv.py` **本树不存在**）。它必须 monkeypatch **模块内绑定名** `chunked_sgmv_shrink.get_lora_shrink_config`（**不是** `lora_tuning_config.get_lora_shrink_config` —— 后者是 `from ... import` 的直接绑定，patch 源模块无效）。
**chunk_size 语义**（`chunked_backend._determine_chunk_size_for_tokens`）：`num_tokens<64→16`、`≥64→32`、`≥256→128`，写入 `batch_info.max_len`（= 查表 key）。**conc=64 decode ⇒ key=32。**

### ⚠️ JIT 缓存操作教训（本次踩两次）
- 新起进程 import overlay 会触发 **flashinfer JIT 串行重建**（`nvcc --threads=1 -O3 -gencode=arch=compute_103a`，多 `.cu`，**实测 ~471s**）。缓存实目录由 `FLASHINFER_CACHE_DIR` 决定（本机曾见 `/mnt/workspace/ferrite-env-backup/bbuf-flashinfer/.cache/flashinfer/0.6.18/103a/cached_ops/sgl_fused_moe_trtllm_sm100/`），**不是** `~/.cache/sglang/.cache/flashinfer`。
- **`timeout N ssh ...` 只杀 ssh 客户端，远端 nvcc 变孤儿继续跑** ⇒ 与后续进程**抢同一缓存目录**，两边都拖死（本次 STAGE-BENCH 300s、M1 闸门 900s 双双 exit 124 且无输出）。解法：`setsid nohup` 把构建与 SSH 解耦，**先跑一次长构建暖缓存**（`/root/warm_build.sh`，实测 471s、rc=0、无孤儿），之后各 testbed 秒级完成。孤儿必须**按精确 PID kill**（`pkill` 被 hook 拦）。

### 融合核 `kernels/ops/moe/trtllm_lora_temp/fused_moe_lora_all.py`
目标：`GEMM1 → +gate_up_delta → gated-SwiGLU(OAI α=1.702/β=1.0/limit=7.0) → MXFP8(E4M3+UE8M0)` 融成一颗，**只在寄存器留 fp32 acc、永不物化 bf16 gate_up**（正消那 4× 写带宽）。
**已修两个硬 bug**（读码发现，非"简化"）：
1. kernel 1 原先**从未量化、从未写 `act_sf`** ⇒ 现在走 `_e4m3_quant_tile`（per-32 UE8M0、PosInf 舍入、±448 饱和）并**同时写数据与 SF**；
2. B 侧 MXFP4 原先**只取低 4 位**（K 维减半，数值必错）⇒ 现在 `_dequant_mxfp4_pair` 取 lo/hi 双 nibble，`join→permute→reshape` 交错成 `[2·BLOCK_K, BLOCK_N]`（**`BLOCK_K` 语义 = 打包字节数**）。

**M1 闸门** `kernels/ops/moe/trtllm_lora_temp/m1_gate.py` 对拍 `M.bench_activation`。布局契约已从源码钉死（`trtllm_fused_moe_dev_kernel.cu::activationKernel`）：`inner_dim = 2*INTER`；gate_up 按 **permutedIdx**、列**交错** `(gate,up,…)`；delta 按 **expandedIdx**、行内 `[up(I) | gate(I)]`；**x1=gu[2j] ← +=delta UPPER half、x2=gu[2j+1] ← +=delta LOWER half**（搞反数值全错但不崩）；`activated` 输出是**直接激活值**（permuted 行、INTER 列），**不是** interleaved。

#### ✅ 实测回填（2026-10-12）
**M1 闸门已过**（T=64/256 两次）：
```
[M1-GATE] T=64   plain   max_abs=5.938e-02  max_rel=3.890e-03  ok=True
[M1-GATE] T=256  plain   max_abs=6.231e-02  max_rel=3.891e-03  ok=True
[M1-GATE] fused kernel imported OK: alpha=1.702 beta=1.0 limit=7.0 SF_BLOCK_A=128 SF_BLOCK_B=32
```
**⚠️ 关键坑（本次踩到，浪费一轮）**：`bench_activation` 的 C++ wrapper **从不设置** `gatedActAlphaPtr/BetaPtr/ClampLimitPtr`（`run()` 也只是原样透传，`dev_kernel:609-610`；结构体注释写着 `// may be null`）⇒ `gatedSiluScalars` 走**早退分支** `return silu(xGlu)*xLinear`。**所以 bench 跑的是「朴素 silu」路，不是 OAI gated 路**。拿生产公式（α=1.702/β=1.0/limit=7.0）去对拍必然 `ok=False`（实测 max_abs≈6.9）。`m1_gate.py` 现在**同时报 plain 与 gated 两个变体**：plain 对 benchmark ✅、gated 是生产契约（靠源码钉死，bench 无法覆盖）。

**CSGMV 调参表已生成**（`csgmv_configs/triton_3_7_1/*NVIDIA_L20D.json`，6 张，JSON 校验通过）。conc=64 ⇒ `chunk_size=32`，实测最优 vs 原默认：

| 形状 | 原默认 | 调参后（chunk=32） |
| --- | --- | --- |
| shrink K=5120 S=3 | `BLOCK_N=16, BLOCK_K=256` | `BLOCK_N=64, BLOCK_K=256, warps4, stages3` |
| shrink K=5120 S=2 | `BLOCK_N=16, BLOCK_K=256` | `BLOCK_N=64, BLOCK_K=64, warps4, stages3` |
| shrink K=576 S=1 | `BLOCK_N=16, BLOCK_K=256` | `BLOCK_N=32, BLOCK_K=64, warps8, stages3` |
| shrink K=1152 S=1 | `BLOCK_N=16, BLOCK_K=256` | `BLOCK_N=32, BLOCK_K=64, warps4, stages3` |
| expand K=5120 S=1 | `BLOCK_N=64, BLOCK_K=16` | `BLOCK_N=32, BLOCK_K=256, warps8, stages3` |
| expand K=1152 S=1 | `BLOCK_N=64, BLOCK_K=16` | `BLOCK_N=64, BLOCK_K=64, warps8, stages3` |

**⚠️ 校验时机坑**：`get_lora_configs` 是 `@functools.lru_cache`，`get_lora_shrink_config` 只在**首次真实 LoRA forward** 时打日志 ⇒ **服务刚就绪时 grep 日志什么都有没有**；必须等第一次带 LoRA 的请求之后再查 `Using LoRA … config from …` vs `no tuned config, using defaults`。

**⚠️ 调参器坑（本次踩到）**：CSGMV 核在 `chunked_sgmv_shrink.py:71` **无条件** `tl.load(permutation + s_offset_logical)` ⇒ 合成 `LoRABatchInfo` 时 `permutation` **不能给 None**（Triton 编译期报错，且报错位置指向注释行，极难读）；给 `torch.arange(s, dtype=int32)` 即可。

**⚠️ 文件名必须覆盖 `S`（num_slices）**：日志行 `LoRA shrink (K=5120, R=16)` **不带 S**，所以只按 (kernel,K,R) 写文件会**全部 miss**（实测：磁盘有文件、日志仍 56 条 `no tuned config`、0 条 `Using … config from`）。**必须写 K×S 全交叉**（本轮 18 张：K∈{576,1152,5120} × S∈{1,2,3} × {shrink,expand}）。误判风险高——**「有文件」不等于「被加载」**，唯一判据是日志出现 `Using LoRA <kernel> config from …`。

**⚠️ `get_lora_configs` 是 `@functools.lru_cache` ⇒ 换表必须重启**（服务一旦跑过 LoRA forward 就把 `None` 缓存住了）。

**⚠️ csgmv expand/shrink 的 `permutation` 在行映射上是「自抵消」的**（2026-10-14 实测，勿再踩）：
`kernels/ops/gemm/chunked_sgmv_expand.py` 里 **x 和 output 都用 `s_offset_physical`** 索引
（:107 读 x / :138 写 out；shrink 同款见 `chunked_sgmv_shrink.py:118`）⇒
`out[perm[i]] = f(x[perm[i]])`，即等价于 **`out += scaling * (x @ W^T)`，没有任何行位移**。
判定实验（随机 perm，`/root/a1/lora_quant/expand_opt/semantics_check.py`）：
按"散布"理解 `out[perm[i]] += x[i]@W^T` 对拍 rel=2.8e-1（错），按"自抵消"理解 rel=1.7e-3（对）。
`permutation` 的真实作用只是**定义哪些行属于哪个 segment**（选 adapter + chunk 切分），不是搬数据。
⇒ 单 adapter（`--max-loras-per-batch 1`）时 M 可任意平铺、且**完全不需要 gather**；
这也解释了为什么该 kernel 的 grid 用 segment 切（`grid=(N/BN, num_slices, num_segments)`、
`BLOCK_M=chunk_size=32` 而每 segment 只有 8 个 expanded row ⇒ M 浪费 75%）。
平铺改造实测（CUDA graph 计时，消 launch 开销）：M=512 仅 1.12×，M=8192 到 2.0–2.2×；
而 decode 侧 expand 实测 ~27µs/shape 里 kernel 只有 ~5µs，**杠杆在 host launch / `torch.zeros` 分配，不在 kernel**。

**⚠️ `/health` 返回 200 但 body 为空** ⇒ 就绪判据必须用 HTTP code：`curl -s -o /dev/null -m 5 -w "%{http_code}"`；用 `[ -n "$(curl -s …/health)" ]` 会**永远判不就绪**（本轮踩到，重启脚本空转 20 分钟）。

**⚠️ `exp_l20.sh` 自带 stop 段**（日志 `[exp_m2] stop current stack` → `ports free after …`）⇒ 重启**不必**手工 kill 旧 P/D/smg；手工 kill 只是冗余。

### 🔴 调参键的真凶是 `R = --max-lora-rank`，**不是适配器的真实 rank**（本次连错两轮）
`get_lora_shrink_config` 的 `R` 来自 `MAX_RANK = weights.shape[1] // num_slices`，而**权重维被 padding 到 `--max-lora-rank`**（本环境 = 32）。所以：
- 启动器写 `--max-lora-rank 32` ⇒ 查表键是 **`R=32`**，哪怕适配器 safetensors 里 `lora_A.weight` 的形状是 `[1, 16, 5120]`（真 rank 16）。
- **日志行 `LoRA shrink (K=5120, R=16)` 是更早一轮 `--max-lora-rank 16` 的旧输出** ⇒ 拿它当依据就会写出整批 `R=16` 文件、**全部 miss**（现象：磁盘有文件、日志全是 `no tuned config`、0 条 `Using … config from`）。

**⇒ 判据只有一条：日志出现 `Using LoRA <kernel> config from …<file>.json`。** 出不来就是键错了。

**⇒ 诊断手段（本次靠它一击定案）**：给 `lora_tuning_config.get_lora_configs` 开头插一行
```python
logger.info("[sdiag] get_lora_configs kernel=%s K=%s R=%s S=%s -> %s", kernel, K, R, S, json_file_name)
```
**重启后第一次 LoRA forward 就会把真实 (kernel,K,R,S) 打出来**（本轮实测键：`shrink {2048,5120,5120S2,576}` + `expand {1152S2,32320,5120}`）。**别再从形状反推。**

### 🔴 CSGMV 块大小调参在 conc=64 **端到端收益为零**（实测结案，勿再走）
`R=32` 表**确认生效**（7/7 `Using LoRA config from`、`no tuned config` 归零）后的 A/B：

| 指标 | 未调优 | 调参生效 | Δ |
| --- | --- | --- | --- |
| base p50 | 5.216 ms | 5.087 ms | 噪声（client_agg 反降 10624→10482） |
| **rnd p50** | **7.632 ms** | **7.636 ms** | **+0.004 ms ⇒ 无改善** |

**⇒ 结论：decode 时 `chunk=32`、各段极短 ⇒ 这些 kernel 是 launch/延迟受限，不是 tile 效率受限**，故 `BLOCK_N/BLOCK_K/num_warps/num_stages` 怎么调都不影响端到端。**该桶实测关闭**（不是推测关闭）。

### 🔬 kernel 微基准协议（2026-10-14 三 lane 实测教训）
给 LoRA/csgmv 类小 kernel 计时，**「flush+sync 再 event」的 raw 协议会把 ~20–30µs 启动延迟算进数字**（空 kernel grid=1 实测 19.65µs），decode T=64 的真实 kernel 时间只有个位数 µs ⇒ raw 协议下「变体加速 2×」常常只是「少了一层 launch」。**判据必须用 CUDA Graph 内计时**（capture 后回放消掉 host launch），或 `graph[flush+kernel] − graph[flush]` 的 cold 口径。另：**权重不是 per-expert 读一次**——shrink 权重按 m-block 重复读（T=64 时 32×），算 issued 流量别按 per-expert；decode T=64 下 MoE shrink 两形状 graph 内合计仅 ~11µs（gate_up 6.59 + down 4.44），**先塞 graph/PDL 再谈 kernel 数学**。

### 🏗️ e2e 收益与 GEMM 时间无关的两个掩盖层（2026-10-15 全闭环实测，勿再按 GEMM 微基准外推 e2e）
1. **MoE GEMM 在 spec decode graph 内被完全掩盖**：GEMM 时间差 4.7ms/step（contiguous 48.66µs vs bmm 283.5µs/层 × 20）在 e2e 里收益=0——MoE GEMM 与 draft 计算重叠，不在关键路径。任何 MoE GEMM 引擎替换（contiguous/bmm/cubin）的 e2e 收益均为 0（实测三路确认）。
2. **A-overlap 与 fused attn shrink 互斥**：两者都消 shrink launch 开销——A-overlap 移 side stream 与 base GEMM 重叠，fused 合并两个模块的 launch。叠加时 fused 收益被 overlap 吸收；独立开也都在噪声带内（shrink launch ~0.17ms < ±0.2ms 噪声带）。全部配置 per-step 26.4-26.6。
3. **推理侧地板 = 7.344ms**（v3 三叠加最优：DEEPGEMM_V3 + MLP_OUT_LORA + A_OVERLAP，acc_len 3.59）。attention LoRA 归属 2.196ms（expand 不可合并——不同 base_output；shrink 被 overlap 吸收）；MoE 归属 0.226ms（GEMM 被掩盖）。**距 6ms 差 1.34ms——剩余全部在训练侧**（输出侧适配器重训 −1.5~2.0 / accept_len draft 质量）。

### ⛔ 红线（2026-10-12 用户下令）：**禁止把 LoRA 烘进权重**
「把 LoRA 直接烘进权重、让 LoRA 请求走基座同一条快路径」——**绝对不行也不允许，必须动态**。
- 即：**不得**物化 `W_eff = W + s·B·A` 去蹭基座的 fused 快路径；LoRA 的 delta 必须**每请求现算**。
- 推论：`fusedAct=false`（delta 必须加在 SwiGLU 之前）这个约束**不能靠改权重绕开** ⇒ 想在 base 那条 `t128x16x256_s6` 快路上跑 LoRA，**只能由我们自己的算子**把 `GEMM1 → +delta → gated-SwiGLU → MXFP8` 融成一颗（即 `fused_moe_lora_all.py`，delta 仍从适配器的 A/B 动态算出）。
- 上述「权重物化」方案**已作废**：即便显存账可行也**不允许**。

### 🟢 63% 桶的门槛实验（未定论，但已具备条件）
- `MoeRunnerBackend.DEEP_GEMM = "deep_gemm"` 合法；`mxfp4.py:386-390` 原文：*"True W4A8: DeepGEMM fp8_fp4 grouped GEMM (SM100 MXF8F6F4 UMMA) … Weights stay MXFP4 … zero requantization"*。
- `deep_gemm 0.2.0` 已装，导出 `m_grouped_fp8_fp4_gemm_nt_masked`（正是 A=MXFP8 × B=MXFP4）。
- 树内有 `srt/layers/moe/moe_runner/deep_gemm.py::DeepGemmRunnerCore`，且 **GEMM 与「SwiGLU+量化」是分开两步**（`:771` GEMM1 → `:813 _varlen_deep_gemm_silu_mul_quant`）⇒ **LoRA delta 天然可插在 `:771` 之后、`:813` 之前**。
- **但 LoRA 只在 `experimental_sgl_trtllm` 一个 backend 里实现**（`srt/lora/trtllm_lora_temp/lora_layer.py` 的 `init_/dispatch_experimental_sgl_trtllm_lora`；`srt/lora/` 全树对 `deep_gemm` 引用数 = 0）⇒ **`--moe-runner-backend deep_gemm` 会完全绕过 LoRA ⇒ 只能先测 base。**
- 变体启动器 `/root/exp_l20_dg.sh`（只插 `--moe-runner-backend deep_gemm`，原 `exp_l20.sh` 不动）；base-only 基准脚本 `/tmp/bench_dg_base.sh`。
- **判据**：base p50 显著优于 **5.087–5.216 ms** ⇒ 63% 可攻；持平/更差 ⇒ **两者都要落 bf16 gate_up（4× 写带宽结构不变）就是墙**。

## ★ R50 方案：融合核 + 输出侧 Δ（用户拍板，2026-10-15 落地）

**方向定论**：decode 带宽受限，**任何 bf16 路线（v3 bmm 的 22.6GB cache）都是反方向**（用户令 + R49a/b/c 三轮 capture OOM 实锤：avail 越大 capture 吸得越多，配置层无解，v3 永久关闭）。

**新方案**：`y = y_fused(x) + Δ(x)`——STANDARD topk 的 LoRA batch 直接进 NVIDIA 闭源融合核的 **routed 入口**
（`_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen`，flashinfer_trtllm.py 的 is_standard+silu 分支——
它天然吃物化 topk、gated-SwiGLU、OAI 三参数、biases），LoRA delta 输出侧相加。

**实现**：`moe_overlap.py::fused_experts_none_to_experimental_sgl_trtllm_mxfp4_lora_two_stream` 的
REAL FIX 段后加 `SGLANG_LORA_FUSED_POSTDELTA=1` 门控分支（R51 三发全过、mini bench 64/64 零错误）。
routed_scaling_factor 已被 REAL FIX 物化进 weights，融合核 TopK-only 分支核内不设 mRouteScale——**无双乘**。

**结构契约（训练侧，已通知）**：
- 放弃 `gate_up_proj` 注入（δ₁ 在 SwiGLU 内——闭源核无挂钩点）
- `down_proj` 改 **parallel adapter（并联适配器）**：`y = FusedMoE(x) + Σ sₖ·B′A′x`
  ——A′ 输入从 SwiGLU 后激活（I=2304）改为**层输入（d=7168）**，权重形状 [r,7168]/[7168,r]；
  建议注入点名 `moe_output`（别与原 down_proj 混淆）；路由二选一（per-expert 带路由 / 纯并联单套）
- 现存 dzgu（A 吃 2304）**不能直接迁移**，需重训；moe_zero 仍可用作推理侧地基基线

**验证状态与已知现象**：
- R51 三发输出正常；zero≠base 一词之差 = STANDARD 入口 vs BYPASSED 入口的**路由舍入差**
  （贪心解码发散的同款现象，非 bug——两条路径分别自洽：zero==rnd 逐字一致）
- **Δ≡0 时是地基测量**（delta 通道未接）——LoRA 严格生效证明（rnd≠base）必须等并联适配器到位
- **口径铁律**：报数必须 TPOT p50 × 稳态 accept_len（per-step），acc 从 bench **中段**抓
  （bench 结束后的残余值 3.8-4.6 不是稳态，mini bench 的 wall/4096 粗口径数字作废）

> ⚠️ **POSTDELTA prefill crash 未解（R51/R54 同模式，2026-10-15）**：
> R54 加了 **decode-only 门控**（`current_forward_is_decode()`，capture_mode.py:121——该 API 的设计用户就是 moe_overlap 的 two-stream gate），
> prefill 走原分解链——**acc 崩塌修了**（R51 的 2.15 → R54 的 3.50-3.89 正常），但 **256 bench 后 prefill 仍 crash**（P=000，和 R51 同模式 16:47:47）。
> ⇒ POSTDELTA 的 **decode 路径本身**对 prefill 有深层副作用（非 prefill 走不走 POSTDELTA 的问题）：
> 嫌疑 = decode 走融合核更快 → mooncake KV 传输模式变化 → prefill 传输线程压力/资源耗尽。
> **POSTDELTA 当前关**（生产恢复 R55），代码保留供专项调查。调查清单：
> ① prefill crash 栈（libc 段错误，无 Python 帧——需 CUDA coredump 或 valgrind）
> ② mooncake session 状态（256 bench 期间的 KVTransferError 形态——是 AbortReq 还是 session-not-alive）
> ③ POSTDELTA decode 路径的 KV 传输时序 vs 分解链（融合核更快是否导致 KV 请求密度变化）
