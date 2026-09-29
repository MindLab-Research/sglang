# sglang-mol-v41 — DeepSeek-V4.1-Flash + MoL port

Base: upstream `main` 81f27fb3a7 (+ fork increments ported in). Branch
`release/dsv41-flash-mol` on `MindLab-Research/sglang`.

## What lives here

| Feature | Status | Entry points |
| --- | --- | --- |
| DSV4.1 inference (MXFP4 experts, SM100 trtllm-gen) | verified | stock upstream path |
| attention LoRA (q_proj/…) | verified | upstream `chunked` backend |
| `kv_shared` LoRA (adapter shares the base radix namespace) | verified | `lora_config.lora_kv_shared`, `lora_manager.is_lora_kv_shared`, `scheduler.resolve_lora_kv_shared` |
| MoE LoRA on the FP4/trtllm-gen fused path | implemented, under test | `trtllm_lora_temp/`, `experimental_sgl_trtllm` backend |

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

## Verification recipes

- `test/registered/scripts/test_kv_shared_lora.py` (+ `make_kv_shared_adapters.py`)
- `test/registered/scripts/test_kv_hicache.py`, `bench_kv_shared_tpot.py`
- MoE LoRA: `make_zero_moe_lora.py` + `test_moe_lora_v41.py` — **a zero-weight
  adapter must reproduce the base output exactly**; that is the cheap numerical
  parity check for the decomposed path (activation + quant + deltas).
