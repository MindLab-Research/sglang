# sglang-mol-v41 — DeepSeek-V4.1-Flash + MoL port

Base: upstream `main` 81f27fb3a7 (+ fork increments ported in). Branch
`release/dsv41-flash-mol` on `MindLab-Research/sglang`.

## What lives here

| Feature | Status | Entry points |
| --- | --- | --- |
| DSV4.1 inference (MXFP4 experts, SM100 trtllm-gen) | verified | stock upstream path |
| attention LoRA (q_proj/…) | verified | upstream `chunked` backend |
| `kv_shared` LoRA (adapter shares the base radix namespace) | verified | `lora_config.lora_kv_shared`, `lora_manager.is_lora_kv_shared`, `scheduler.resolve_lora_kv_shared` |
| MoE LoRA on the FP4/trtllm-gen fused path | verified (zero==base bitwise; random-adapter dynamic load) | `trtllm_lora_temp/`, `experimental_sgl_trtllm` backend |

## 1101 layout

- code `/root/dsv41_upstream/sglang`, launcher `launch.sh`, weights `/mnt/workspace/DeepSeek-V4.1-Flash`
- **after every rsync: `find /root/dsv41_upstream -name '__pycache__' -type d -exec rm -r {} +`** (stale .pyc masks updates)
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
11. **The flashinfer JIT cache is presence-based, not content-based.** After
   editing the overlay .cu source, the JIT loads the stale .so unless it is
   removed/renamed — it does not hash the source to detect changes. Always
   rename the .so in `~/.cache/sglang/.cache/flashinfer/*/cached_ops/
   sgl_fused_moe_trtllm_sm100/` after rsync'ing overlay source changes.
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

## Verification recipes

- `test/registered/scripts/test_kv_shared_lora.py` (+ `make_kv_shared_adapters.py`)
- `test/registered/scripts/test_kv_hicache.py`, `bench_kv_shared_tpot.py`
- MoE LoRA: `make_zero_moe_lora.py` + `test_moe_lora_v41.py` — **a zero-weight
  adapter must reproduce the base output exactly**; that is the cheap numerical
  parity check for the decomposed path (activation + quant + deltas).
