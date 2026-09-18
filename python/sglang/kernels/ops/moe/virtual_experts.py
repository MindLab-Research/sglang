"""
LoRA Virtual Experts Triton Ops.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.moe_align import moe_align_block_size as jit_moe_align_block_size

# ---------------------------------------------------------------------------
# MoL LoRA stage tuning (SGLANG_OPT_MOL_LORA_STAGE_CFG, see environ.py)
# ---------------------------------------------------------------------------
# The virtual-experts LoRA stages (shrink + expand) run through the MoE
# stage-config path. For these shapes the autotuner has no entry, so the
# fallback config is used and _get_routing() hands its BLOCK_SIZE_M to
# moe_align_block_size: every touched virtual-expert bucket is padded to a full
# BLOCK_SIZE_M-row block. With the stock values (BLOCK_SIZE_M=64,
# BLOCK_SIZE_N=64) a decode batch's ~1.1k real (token, virtual-expert) pairs
# become ~1.6k blocks x 64 rows ~= 1e5 padded rows, and every stage's launch
# grid follows from that plus BLOCK_SIZE_N. Measured (B300, bs=23, 60-step
# profiler window): the down-delta expand is a single 153,984-CTA launch taking
# 160.6us/layer at ~85GB/s effective (~1% of HBM roofline) while its four
# sibling launches do the same kind of work in 5-43us -- i.e. all five
# per-layer LoRA launches are tile/CTA-overhead bound, not bandwidth bound.
#
# The fix retunes the knob that actually drives the launch count. The grid is
# built host-side in invoke_fused_moe_kernel as
#     grid = cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M) * cdiv(B.shape[1], BLOCK_SIZE_N)
# and sorted_token_ids is the *worst-case allocated* routing buffer (the caller
# deliberately keeps the untrimmed allocation, see the long note in _get_routing),
# so     sorted_len = numel + virtual_num_experts * (BLOCK_SIZE_M - 1)
#      grid_m    = cdiv(sorted_len, BLOCK_SIZE_M) ~= virtual_num_experts + numel/BLOCK_SIZE_M
# i.e. grid_m only weakly decreases with BLOCK_SIZE_M (a few percent) while each
# tile walks proportionally more padded rows -- so that knob is NOT the lever.
# BLOCK_SIZE_N is: grid_n = cdiv(N, BLOCK_SIZE_N) is a clean 4x cut for the
# down-delta expand (N=6144: 96 -> 24) and for the gate_up halves (N=256: 4 -> 1).
# It is clamped to the stage's real GEMM N so no dead columns are computed.
# BLOCK_SIZE_K and the K-loop stay untouched, so each output element still
# reduces over K inside a single tile -> bit-identical.
_MOL_LORA_TUNED_BLOCK_N = 256
_MOL_LORA_STAGE_CFG_ENV = None


def _mol_lora_stage_cfg_enabled() -> bool:
    """Read SGLANG_OPT_MOL_LORA_STAGE_CFG (2 calls per layer).

    The EnvBool object is cached but its value is read live, so a test/runner can
    still flip it (envs.<X>.override()) after this module was imported.
    """
    global _MOL_LORA_STAGE_CFG_ENV
    if _MOL_LORA_STAGE_CFG_ENV is None:
        from sglang.srt.environ import envs

        _MOL_LORA_STAGE_CFG_ENV = envs.SGLANG_OPT_MOL_LORA_STAGE_CFG
    return bool(_MOL_LORA_STAGE_CFG_ENV.get())


def _apply_mol_lora_stage_cfg(cfg: dict, n_dim: int) -> dict:
    """Widen the MoL LoRA stage N-tile (grid = blocks * cdiv(N, BLOCK_SIZE_N)).

    ``n_dim`` is the stage GEMM's real N (rank for the shrink, output width for
    the expand) -- not the weight tensor's leading dim, since the shrink weight
    is laid out [E, K, N] and the expand weight [E, N, K].
    """
    tuned = dict(cfg)
    tuned["BLOCK_SIZE_N"] = min(_MOL_LORA_TUNED_BLOCK_N, max(16, int(n_dim)))
    return tuned


_MOL_STAGE_PROBE_LOGGED = False


def _probe_mol_stage_cfg(a_cfg: dict, b_cfg: dict) -> None:
    """One-shot diagnostic: proves the MoL stage path ran and shows the tiling
    actually handed to the kernels (grep MOL-PROBE in the engine log).

    This is the log-side counterpart of the profiler check: BLOCK_SIZE_N here is
    what becomes `num_pid_n = cdiv(N, BLOCK_SIZE_N)` in the kernel, i.e. the
    launch-count lever this flag tunes (expand: 64 before, min(256, N) after).
    """
    global _MOL_STAGE_PROBE_LOGGED
    if _MOL_STAGE_PROBE_LOGGED:
        return
    _MOL_STAGE_PROBE_LOGGED = True
    import logging

    logging.getLogger(__name__).info(
        "[MOL-PROBE] flag=%s shrink(BM=%s BN=%s BK=%s) expand(BM=%s BN=%s BK=%s)",
        _mol_lora_stage_cfg_enabled(),
        a_cfg.get("BLOCK_SIZE_M"), a_cfg.get("BLOCK_SIZE_N"), a_cfg.get("BLOCK_SIZE_K"),
        b_cfg.get("BLOCK_SIZE_M"), b_cfg.get("BLOCK_SIZE_N"), b_cfg.get("BLOCK_SIZE_K"),
    )


_MOL_LORA_STAGE_CFG_LOGGED = False


def _log_mol_lora_stage_cfg_once(cfg: dict, tuned: dict, n_dim: int) -> None:
    """One-shot loud marker so a deployment can be verified from the log (grep
    MOL-LORA-STAGE-CFG) -- the decode node's per-layer kernel timing is the
    actual acceptance signal, this just proves the flag reached the kernels."""
    global _MOL_LORA_STAGE_CFG_LOGGED
    if _MOL_LORA_STAGE_CFG_LOGGED:
        return
    _MOL_LORA_STAGE_CFG_LOGGED = True
    import logging

    old_n = int(cfg.get("BLOCK_SIZE_N") or 1)
    new_n = int(tuned.get("BLOCK_SIZE_N") or 1)
    logging.getLogger(__name__).info(
        "[MOL-LORA-STAGE-CFG] enabled: n_dim=%d BLOCK_SIZE_N %d -> %d "
        "(BLOCK_SIZE_M=%s BLOCK_SIZE_K=%s untouched; N-tile count %.1fx smaller)",
        n_dim,
        old_n,
        new_n,
        cfg.get("BLOCK_SIZE_M"),
        cfg.get("BLOCK_SIZE_K"),
        old_n / max(1, new_n),
    )




@triton.jit
def _fused_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    token_lora_mask_ptr,
    num_experts_for_weight: tl.constexpr,
    M,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_loras,
):
    """
    Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.

    For each (m, k):
        lora_id = token_lora_mapping[m]
        mask[m] = (lora_id >= 0)
        safe_lora = max(lora_id, 0)
        if shared_outer:  (handled by num_experts_for_weight == 0 sentinel)
            virtual_topk_ids[m, k] = safe_lora * 1  (= safe_lora)
        else:
            virtual_topk_ids[m, k] = topk_ids[m, k] + safe_lora * num_experts_for_weight
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k
    valid = offs < total

    m = offs // top_k
    # k = offs % top_k  # not needed directly

    lora_id_g = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)
    lora_id_g = tl.where(lora_id_g >= max_loras, -1, lora_id_g)
    lora_id = lora_id_g
    mask_val = lora_id >= 0
    safe_lora = tl.maximum(lora_id, 0)

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    # GARBAGE-GUARD: clamp base topk ids to the expert range. Padding rows on
    # padded/cuda-graph batches hold UNINITIALISED values (huge positives AND
    # negatives); the native moe_align kernel uses ids as atomic offsets, so
    # any out-of-range value produces a wild pointer (Xid 31 FAULT_PDE).
    # Everything outside [0, E) maps to the designed -1 sentinel.
    if num_experts_for_weight > 0:
        base = tl.where(
            (base < 0) | (base >= num_experts_for_weight), -1, base
        )
    shifted = base + safe_lora * num_experts_for_weight
    result = tl.where(base < 0, base, shifted)
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)

    # Write mask once per row (at first k position)
    k = offs % top_k
    is_first_k = k == 0
    tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)


def _fused_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns virtual topk_ids, token_lora_mask, and virtual_num_experts.
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device

    if shared_outer:
        num_experts_for_weight = 1
        # For shared_outer, we need topk_ids to be zeros
        zero_topk = torch.zeros_like(topk_ids)
        input_topk = zero_topk
    else:
        num_experts_for_weight = num_experts
        input_topk = topk_ids

    virtual_topk_ids = torch.empty_like(topk_ids)
    token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)

    BLOCK_SIZE = 1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # GARBAGE-GUARD (host): neutralise stale/padding mapping ids before
    # they can multiply into out-of-range virtual expert ids.
    if max_loras > 0 and token_lora_mapping.numel() > 0:
        token_lora_mapping = torch.where(
            (token_lora_mapping >= 0) & (token_lora_mapping < max_loras),
            token_lora_mapping,
            torch.full_like(token_lora_mapping, -1),
        )

    _fused_virtual_topk_ids_kernel[grid](
        input_topk,
        token_lora_mapping,
        virtual_topk_ids,
        token_lora_mask,
        num_experts_for_weight,
        M,
        top_k,
        BLOCK_SIZE,
    
        max_loras,
    )

    virtual_num_experts = num_experts_for_weight * max_loras
    return virtual_topk_ids, token_lora_mask, virtual_num_experts


@triton.jit
def _fused_sanitize_expert_ids_kernel(
    expert_ids_ptr,
    output_ptr,
    num_virtual_experts,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < N

    eid = tl.load(expert_ids_ptr + offs, mask=valid, other=0)
    result = tl.where(eid < num_virtual_experts, eid, -1)
    tl.store(output_ptr + offs, result, mask=valid)


def fused_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    num_virtual_experts: int,
) -> torch.Tensor:
    """
    Sanitize expert_ids by replacing values >= num_virtual_experts with -1.

    Returns a new tensor with expert_ids >= num_virtual_experts replaced by -1.
    """
    N = expert_ids.numel()
    output = torch.empty_like(expert_ids)

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_sanitize_expert_ids_kernel[grid](
        expert_ids,
        output,
        num_virtual_experts,
        N,
        BLOCK_SIZE,
    )
    return output


@triton.jit
def _moe_lora_shrink_splitk_kernel(
    # Pointers
    a_ptr,  # type: ignore  # [num_tokens, K]
    b_ptr,  # type: ignore  # [num_virtual_experts, N, K]
    c_ptr,  # type: ignore  # [num_tokens * top_k, N]  (pre-zeroed when SPLIT_K > 1)
    sorted_token_ids_ptr,  # type: ignore
    expert_ids_ptr,  # type: ignore
    num_tokens_post_padded_ptr,  # type: ignore
    # Dimensions
    N,  # type: ignore
    K,  # type: ignore
    num_valid_tokens,  # type: ignore
    # Strides
    stride_am,  # type: ignore
    stride_ak,  # type: ignore
    stride_be,  # type: ignore
    stride_bn,  # type: ignore
    stride_bk,  # type: ignore
    stride_cm,  # type: ignore
    stride_cn,  # type: ignore
    # Constexprs
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts."""
    pid = tl.program_id(0)
    pid_sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token routing (same pattern as fused_moe_triton_kernels)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    # The align-block-size buffers keep a worst-case-sized UNINITIALIZED slack
    # past the real tokens; freed-memory reuse can leave negative values there
    # (e.g. -1 fills from token_lora_mapping tails). A negative offs_token
    # passes a plain upper-bound mask and the store below then writes at
    # c_ptr + offs_token*stride — BEFORE the tensor, corrupting the
    # neighbouring allocation (observed as NaN activations / garbled LoRA
    # outputs). Reject the negative range explicitly.
    token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    # Pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        k_mask = offs_k[:, None] < k_remaining
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


def _invoke_moe_lora_shrink_splitk(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, Any],
) -> None:
    """Launch split-K shrink kernel for LoRA A with few virtual experts."""
    N = weight.shape[1]
    K = weight.shape[2]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = min(config.get("BLOCK_SIZE_N", 64), max(16, N))
    BLOCK_SIZE_K = config.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)
    base_grid = num_m_blocks * num_n_blocks
    max_split_k = max(1, K // BLOCK_SIZE_K)
    SPLIT_K = min(max_split_k, max(1, 128 // base_grid)) if base_grid < 128 else 1

    grid = (SPLIT_K * base_grid,)

    _moe_lora_shrink_splitk_kernel[grid](
        hidden_states,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
    )


def _align_block_size_jit(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA JIT align_block_size for num_experts > 1024 (up to 8191).

    Uses the v2 kernel from moe_align_kernel.cu which supports large expert
    counts via per-thread multi-expert processing and a two-level warp scan,
    replacing the previous pure-PyTorch fallback that had excessive CPU overhead
    from 15+ individual kernel launches and torch.argsort.

    The JIT kernel uses a +1 offset convention: topk_ids are shifted by +1 so
    that the EP sentinel value (-1) maps to bucket 0. The kernel internally
    handles histogram, padded prefix-sum, expert_ids assignment, and token
    scattering in just 2–3 CUDA kernel launches.
    """
    assert num_experts <= 8191, (
        f"_align_block_size_jit supports at most 8191 experts "
        f"(num_moe_experts * max_loras), got {num_experts}"
    )

    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1)
    if flat_topk_ids.dtype == torch.int64:
        flat_topk_ids = flat_topk_ids.to(torch.int32)
    num_total_tokens = flat_topk_ids.numel()

    if num_total_tokens == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, torch.zeros(1, dtype=torch.int32, device=device)

    # JIT kernel uses +1 offset convention: -1 -> bucket 0 (sentinel),
    # expert i -> bucket i+1. So pass num_experts + 1 as the bucket count.
    jit_num_experts = num_experts + 1

    if num_total_tokens < jit_num_experts:
        max_num_tokens_padded = num_total_tokens * block_size
    else:
        max_num_tokens_padded = num_total_tokens + jit_num_experts * (block_size - 1)

    # Align every sub-buffer offset to a multiple of 4 (VEC_SIZE). The CUDA
    # kernel fills sorted_token_ids with vectorized int4 writes whose last
    # store can spill up to 3 int32s past the logical end. With a fused
    # allocation the spill would corrupt the adjacent sub-buffer.
    _A4 = lambda n: (n + 3) & ~3  # noqa: E731
    max_num_tokens_padded = _A4(max_num_tokens_padded)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_m_blocks_padded = _A4(max_num_m_blocks)
    num_post_pad_size = _A4(1)  # 1 element, padded to 4
    cumsum_size = _A4(jit_num_experts + 1)

    # Single allocation sliced into 4 views (zero-copy) to avoid
    # per-call Python overhead of 4 separate torch.empty calls.
    total_buf = (
        max_num_tokens_padded
        + max_num_m_blocks_padded
        + num_post_pad_size
        + cumsum_size
    )
    buf = torch.empty(total_buf, dtype=torch.int32, device=device)
    off = 0
    sorted_token_ids = buf[off : off + max_num_tokens_padded]
    off += max_num_tokens_padded
    expert_ids = buf[off : off + max_num_m_blocks]
    off += max_num_m_blocks_padded
    num_tokens_post_padded = buf[off : off + 1]
    off += num_post_pad_size
    cumsum_buffer = buf[off : off + jit_num_experts + 1]

    jit_moe_align_block_size(
        flat_topk_ids,
        jit_num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,  # pad_sorted_token_ids
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


@torch.compile(dynamic=True)
def _align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile.

    Fallback for platforms where the CUDA JIT kernel is unavailable (e.g. AMD/ROCm).

    Out-of-range topk_ids (negative sentinels left by EP dispatch, or virtual-
    expert IDs >= num_experts produced when those sentinels are combined with
    a per-adapter offset) are routed into a dedicated sentinel bucket. Without
    this, indexing ``padded_offsets[sorted_expert_ids]`` would wrap (-1) or
    OOB-read, and the bad expert ids would propagate into the downstream LoRA
    GEMM as real expert slots.
    """
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_total_tokens = flat_topk_ids.numel()

    sentinel = num_experts
    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_experts)
    safe_topk_ids = torch.where(
        valid_mask,
        flat_topk_ids,
        torch.full_like(flat_topk_ids, sentinel),
    )

    bucket_count = num_experts + 1
    max_total_padded_tokens = (
        (num_total_tokens + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,),
        num_total_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_total_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(safe_topk_ids)
    sorted_expert_ids = safe_topk_ids[sorted_order]
    expert_range = torch.arange(bucket_count, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_total_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    real_block_counts = block_counts.clone()
    real_block_counts[sentinel] = 0
    actual_num_blocks = real_block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(real_block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


def _align_block_size_large(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the CUDA JIT kernel when available, otherwise fall back to
    the pure-PyTorch torch.compile path (needed on AMD/ROCm or when the JIT
    module fails to load)."""
    try:
        return _align_block_size_jit(topk_ids, block_size, num_experts)
    except Exception:
        return _align_block_size_torch(topk_ids, block_size, num_experts)


def _merged_experts_fused_moe_lora_add_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    return


def _merged_experts_fused_moe_lora_add_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
) -> None:
    """Fused virtual-experts LoRA delta add.

    ``lora_b`` accepts either a single tensor or a sequence of tensors stacked
    along the output dim. Length-2 is the gate_up case where A has rank ``2*r``
    (gate's A and up's A concatenated along rank) and each B has rank ``r``.
    The shrink runs once over the full ``2*r`` rank; the expand runs once per
    B, each reading its half of the intermediate and writing to its slice of
    ``output``.
    """
    lora_b_list: list[torch.Tensor] = (
        list(lora_b) if isinstance(lora_b, (list, tuple)) else [lora_b]
    )
    n_b = len(lora_b_list)
    assert n_b in (1, 2), f"lora_b must be length 1 or 2, got {n_b}"
    b_rank = lora_b_list[0].shape[3]
    for b in lora_b_list[1:]:
        assert (
            b.shape == lora_b_list[0].shape
        ), f"all lora_b tensors must share shape; got {[tuple(t.shape) for t in lora_b_list]}"

    max_loras, _, max_lora_rank, _ = lora_a.shape
    assert (
        max_lora_rank == n_b * b_rank
    ), f"lora_a rank {max_lora_rank} != n_b ({n_b}) * lora_b rank {b_rank}"
    input_top_k = 1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])

    def _get_stage_config(
        weight: torch.Tensor,
        stage_top_k: int,
        n_dim: int,
    ) -> dict[str, Any]:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:
            cfg = get_config_func(token_lora_mapping.shape[0])
        except ValueError:
            K_dim = weight.shape[2]
            N_dim = weight.shape[1]
            if K_dim >= 1024:
                default_block_k = 256
            elif K_dim >= 64:
                default_block_k = 64
            else:
                default_block_k = max(16, K_dim)
            cfg = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": min(64, max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        # Unified MoL stage tuning (SGLANG_OPT_MOL_LORA_STAGE_CFG).
        #
        # Applied whether or not the autotuner produced a config: measured on the
        # B300 pair, the autotuner returns a *generic* entry for these shapes
        # (BM=16 BN=32 BK=64 -- see MOL-PROBE), there is no tuned-for-MoL config to
        # protect, and this helper is only ever called for the MoL virtual-expert
        # stages. Gating on `used_fallback` (an earlier, over-cautious version)
        # silently turned the flag into a no-op here.
        # Touches only BLOCK_SIZE_N, so the align block size / BLOCK_SIZE_K /
        # K-loop stay as upstream -> bit-identical. Kill-switch: unset the env.
        if _mol_lora_stage_cfg_enabled():
            tuned = _apply_mol_lora_stage_cfg(cfg, n_dim)
            if tuned != cfg:
                _log_mol_lora_stage_cfg_once(cfg, tuned, n_dim)
            cfg = tuned
        return cfg

    def _align_block_size(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The native align kernel consumes num_experts + 1 internally for its
        # sentinel bucket, so the 1024-expert boundary must use the fallback path.
        if num_experts < 1024:
            from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)
        return _align_block_size_large(topk_ids, block_size, num_experts)

    def _get_routing(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        num_experts: int,
        shared_outer: bool,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        cache_key = (num_experts, shared_outer, block_size)
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (
            _fused_virtual_topk_ids(
                topk_ids, token_lora_mapping, num_experts, shared_outer, max_loras
            )
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
            virtual_topk_ids,
            block_size=block_size,
            num_experts=virtual_num_experts,
        )
        # NOTE: do NOT trim sorted_token_ids / expert_ids to a tighter upper bound here.
        # The downstream kernels (_moe_lora_shrink_splitk_kernel, fused_moe_kernel) read
        # sorted_token_ids[pid_m*BLOCK : +BLOCK] and expert_ids[pid_m] WITHOUT a bounds mask
        # for every block up to num_tokens_post_padded (a GPU-side count loaded at run time).
        # num_tokens_post_padded comes from _align_block_size with `virtual_num_experts` buckets
        # and can exceed a tighter `numel + min(numel,virtual_num_experts)*(block-1)` bound
        # (most so for shared-outer, where virtual_num_experts = max_loras is small), so trimming
        # made those unmasked reads land PAST the view. In eager mode the slack still lives inside
        # the same _align_block_size allocation (garbage, masked out downstream) so it worked; under
        # CUDA-graph capture/replay the graph mempool packs tensors tightly and that slack may belong
        # to another pooled tensor / lie past a page -> cudaErrorIllegalInstruction during capture.
        # Keep the full worst-case-allocated buffers so every unmasked read stays in-allocation.
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
        # Defense-in-depth for the uninitialized slack in sorted_token_ids:
        # negative garbage (freed-memory reuse, e.g. -1 fills) would pass the
        # upper-bound-only masks in the downstream GEMM kernels and index
        # BEFORE the destination tensors. Clamp negatives to a large sentinel
        # that every `offs_token < num_valid_tokens` mask rejects.
        sorted_token_ids = torch.where(
            sorted_token_ids < 0,
            sorted_token_ids.new_full((), 2**30),
            sorted_token_ids,
        )
        result = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        )

        if routing_cache is not None:
            routing_cache[cache_key] = result

        return result

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    lora_a_virtual = _merge_lora_expert_weight(lora_a)
    lora_b_virtuals = [_merge_lora_expert_weight(b) for b in lora_b_list]
    num_experts_a = lora_a.shape[1]
    num_experts_b = lora_b_list[0].shape[1]
    half_out = lora_b_list[0].shape[2]

    # The kernels index token_lora_mapping / intermediate by token ids up to
    # topk_ids.shape[0] (the DP-gathered token count under --enable-dp-attention). An
    # under-sized mapping means unmasked OOB reads/writes that surface as a sticky,
    # hard-to-attribute CUDA IMA — fail loudly on the host instead.
    _mapping_numel = token_lora_mapping.numel() if token_lora_mapping is not None else 0
    if _mapping_numel < topk_ids.shape[0]:
        import logging as _logging

        _logging.getLogger(__name__).error(
            "[LORA-MAP-OOB] mapping_numel=%d < topk_rows=%d (deficit=%d) — "
            "VE kernels will read mapping OOB (random wrong-adapter weights "
            "= garbled output); mapping head=%s",
            _mapping_numel,
            topk_ids.shape[0],
            topk_ids.shape[0] - _mapping_numel,
            token_lora_mapping[:8].tolist() if token_lora_mapping is not None else None,
        )
    # --- Fused path: shrink + expand + add in one kernel (no intermediate) ---
    if _fused_delta_enabled():
        b_stage_config = _get_stage_config(lora_b_virtuals[0], 1, half_out)
        _probe_mol_stage_cfg(
            _get_stage_config(lora_a_virtual, input_top_k, lora_a_virtual.shape[1]),
            b_stage_config,
        )
        (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        ) = _get_routing(
            topk_ids,
            token_lora_mapping,
            num_experts_b,
            experts_shared_outer_loras_b,
            b_stage_config["BLOCK_SIZE_M"],
        )
        block_m = b_stage_config["BLOCK_SIZE_M"]
        block_n = b_stage_config.get("BLOCK_SIZE_N", 64)
        block_k = b_stage_config.get("BLOCK_SIZE_K", 64)
        for b_idx, b_virtual in enumerate(lora_b_virtuals):
            if n_b == 1:
                a_half = lora_a_virtual  # [E, rank, K]
                out_arg = output
            else:
                a_half = lora_a_virtual[
                    :, b_idx * b_rank : (b_idx + 1) * b_rank, :
                ]  # [E, b_rank, K]
                out_arg = output[
                    ..., b_idx * half_out : (b_idx + 1) * half_out
                ].contiguous()
            _invoke_fused_lora_delta(
                hidden_states,
                a_half,
                b_virtual,
                out_arg,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                token_lora_mapping,
                topk_ids.shape[1] if input_top_k > 1 else 1,
                b_rank,
                block_size_m=block_m,
                block_size_n=min(block_n, b_virtual.shape[1]),
                block_size_k=block_k,
            )
            if n_b != 1:
                output[
                    ..., b_idx * half_out : (b_idx + 1) * half_out
                ].copy_(out_arg)
        return

    # --- Original path: separate shrink + expand (fallback) ---
    intermediate = torch.zeros(
        [topk_ids.shape[0], topk_ids.shape[1], max_lora_rank],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    a_stage_config = _get_stage_config(
        lora_a_virtual, input_top_k, lora_a_virtual.shape[1]
    )
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_a,
        experts_shared_outer_loras_a,
        a_stage_config["BLOCK_SIZE_M"],
    )

    _invoke_moe_lora_shrink_splitk(
        hidden_states,
        lora_a_virtual,
        intermediate.view(-1, max_lora_rank),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        input_top_k,
        a_stage_config,
    )

    b_stage_config = _get_stage_config(lora_b_virtuals[0], 1, half_out)
    _probe_mol_stage_cfg(a_stage_config, b_stage_config)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    for b_idx, b_virtual in enumerate(lora_b_virtuals):
        if n_b == 1:
            inter_arg = intermediate.view(-1, b_rank)
            out_arg = output
        else:
            inter_arg = (
                intermediate[..., b_idx * b_rank : (b_idx + 1) * b_rank]
                .contiguous()
                .view(-1, b_rank)
            )
            out_arg = output[
                ..., b_idx * half_out : (b_idx + 1) * half_out
            ].contiguous()
        invoke_fused_moe_kernel(
            inter_arg,
            b_virtual,
            None,
            out_arg,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            1,
            b_stage_config,
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
            False,
            False,
            False,
            False,
            False,
            None,
            fuse_add_to_output=True,
            add_output_mask=token_lora_mask,
            router_topk=topk_ids.shape[1],
        )
        if n_b != 1:
            output[..., b_idx * half_out : (b_idx + 1) * half_out].copy_(out_arg)


@triton.jit
def _fused_lora_delta_kernel(
    # Pointers
    hidden_ptr,           # [num_tokens, K] input hidden states
    lora_a_ptr,            # [num_experts, rank, K] shrink weights (this half's A)
    lora_b_ptr,            # [num_experts, N, rank] expand weights (this half's B)
    output_ptr,            # [num_tokens * top_k, N] base output (add delta in-place)
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,  # [num_tokens] -1=no LoRA
    # Dimensions
    N,                    # output dim (this half)
    K,                    # hidden dim
    num_valid_tokens,
    # Strides
    stride_hm, stride_hk,
    stride_ae, stride_ar, stride_ak,
    stride_be, stride_bn, stride_br,
    stride_om, stride_on,
    stride_lm,
    # Constexprs
    top_k: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused LoRA delta kernel: shrink + expand + add in one launch.

    Replaces the 2-kernel sequence (_moe_lora_shrink_splitk + invoke_fused_moe_kernel)
    with a single kernel that keeps the rank=16 intermediate in registers.

    For each (expert_block, N_block):
      1. Load hidden_states for this block's tokens from HBM
      2. Shrink: intermediate = lora_a[expert] × hidden  → [BLOCK_M, RANK] (registers)
      3. Expand: delta = lora_b[expert] × intermediate    → [BLOCK_M, BLOCK_N] (registers)
      4. Read base output, add delta (masked by token_lora_mapping), write back
    """
    pid = tl.program_id(0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    # --- Token routing (same as _moe_lora_shrink_splitk_kernel) ---
    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    # Actual token rows in hidden_states (sorted_token_ids indexes the
    # flattened [num_tokens * top_k] array, so // top_k gives the row).
    token_rows = offs_token // top_k

    # --- 1. Shrink: intermediate = lora_a[expert] × hidden[token] ---
    # lora_a: [num_experts, RANK, K], we load [RANK, BLOCK_K] tiles
    offs_r = tl.arange(0, RANK)  # rank dimension (16, fits in registers)
    intermediate = tl.zeros([BLOCK_M, RANK], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load hidden states: [BLOCK_M, BLOCK_K]
        a = tl.load(
            hidden_ptr + token_rows[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load lora_a weights: [RANK, BLOCK_K]
        b = tl.load(
            lora_a_ptr
            + off_expert * stride_ae
            + offs_r[:, None] * stride_ar
            + offs_k[None, :] * stride_ak,
            mask=k_mask[None, :],
            other=0.0,
        )

        intermediate += tl.dot(a, b.T.to(a.dtype))  # [BLOCK_M, RANK]

    intermediate = intermediate.to(hidden_ptr.dtype.element_ty)

    # --- 2. Expand: delta = lora_b[expert] × intermediate ---
    # lora_b: [num_experts, N, RANK], we load [BLOCK_N, RANK] tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < N

    lora_b = tl.load(
        lora_b_ptr
        + off_expert * stride_be
        + offs_n[:, None] * stride_bn
        + offs_r[None, :] * stride_br,
        mask=n_mask[:, None],
        other=0.0,
    )  # [BLOCK_N, RANK]

    # delta = intermediate @ lora_b^T: [BLOCK_M, RANK] × [RANK, BLOCK_N]
    delta = tl.dot(intermediate, lora_b.T.to(intermediate.dtype))  # [BLOCK_M, BLOCK_N]

    # --- 3. Add delta to base output (masked) ---
    lora_ids = tl.load(
        token_lora_mapping_ptr + token_rows,
        mask=token_mask,
        other=-1,
    )
    has_lora = lora_ids >= 0  # [BLOCK_M]

    base_out = tl.load(
        output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on,
        mask=token_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    result = tl.where(has_lora[:, None], base_out + delta.to(base_out.dtype), base_out)

    tl.store(
        output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on,
        result,
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _invoke_fused_lora_delta(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k: int,
    rank: int,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 64,
) -> None:
    """Launch the fused LoRA delta kernel (shrink + expand + add).

    Args:
        hidden_states: [num_tokens, K] input
        lora_a: [num_experts, rank, K] shrink weights for this half
        lora_b: [num_experts, N, rank] expand weights for this half
        output: [num_tokens * top_k, N] base output (delta added in-place)
        sorted_token_ids, expert_ids, num_tokens_post_padded: routing
        token_lora_mapping: [num_tokens] -1 = no LoRA
        top_k: number of experts per token
        rank: LoRA rank (typically 16)
    """
    N = lora_b.shape[1]
    K = hidden_states.shape[1]

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_size_m)
    num_n_blocks = triton.cdiv(N, block_size_n)
    grid = (num_m_blocks * num_n_blocks,)

    _fused_lora_delta_kernel[grid](
        hidden_states,
        lora_a,
        lora_b,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        sorted_token_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        token_lora_mapping.stride(0),
        top_k=top_k,
        RANK=rank,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_K=block_size_k,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _fused_lora_delta_all_groups_kernel(
    hidden_ptr,
    lora_a_ptr,            # [max_loras, num_experts, rank, K]
    lora_b_ptr,            # [max_loras, num_experts, N, rank]
    output_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    N,
    K,
    num_valid_tokens,
    stride_hm, stride_hk,
    stride_la_l, stride_la_e, stride_la_r, stride_la_k,
    stride_lb_l, stride_lb_e, stride_lb_n, stride_lb_r,
    stride_om, stride_on,
    stride_lm,
    top_k: tl.constexpr,
    RANK: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return
    token_rows = offs_token // top_k

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < N

    lora_ids = tl.load(
        token_lora_mapping_ptr + token_rows,
        mask=token_mask, other=-1,
    )
    result = tl.load(
        output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on,
        mask=token_mask[:, None] & n_mask[None, :], other=0.0,
    )

    for lora_id in tl.static_range(MAX_LORAS):
        group_mask = (lora_ids == lora_id) & token_mask
        offs_r = tl.arange(0, RANK)
        intermediate = tl.zeros([BLOCK_M, RANK], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(
                hidden_ptr + token_rows[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=token_mask[:, None] & k_mask[None, :], other=0.0,
            )
            b = tl.load(
                lora_a_ptr + lora_id * stride_la_l + off_expert * stride_la_e
                + offs_r[:, None] * stride_la_r + offs_k[None, :] * stride_la_k,
                mask=k_mask[None, :], other=0.0,
            )
            intermediate += tl.dot(a, b.T.to(a.dtype))
        intermediate = intermediate.to(hidden_ptr.dtype.element_ty)
        lora_b = tl.load(
            lora_b_ptr + lora_id * stride_lb_l + off_expert * stride_lb_e
            + offs_n[:, None] * stride_lb_n + offs_r[None, :] * stride_lb_r,
            mask=n_mask[:, None], other=0.0,
        )
        delta = tl.dot(intermediate, lora_b.T.to(intermediate.dtype))
        result = tl.where(group_mask[:, None], result + delta.to(result.dtype), result)

    tl.store(
        output_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on,
        result, mask=token_mask[:, None] & n_mask[None, :],
    )


def _invoke_fused_lora_delta_all_groups(
    hidden_states, lora_a, lora_b, output,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    token_lora_mapping, top_k, rank, max_loras,
    block_size_m=64, block_size_n=64, block_size_k=64,
):
    N = lora_b.shape[2]
    K = hidden_states.shape[1]
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_size_m) * triton.cdiv(N, block_size_n),)
    _fused_lora_delta_all_groups_kernel[grid](
        hidden_states, lora_a, lora_b, output,
        sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mapping,
        N, K, sorted_token_ids.numel(),
        hidden_states.stride(0), hidden_states.stride(1),
        lora_a.stride(0), lora_a.stride(1), lora_a.stride(2), lora_a.stride(3),
        lora_b.stride(0), lora_b.stride(1), lora_b.stride(2), lora_b.stride(3),
        output.stride(0), output.stride(1), token_lora_mapping.stride(0),
        top_k=top_k, RANK=rank, MAX_LORAS=max_loras,
        BLOCK_M=block_size_m, BLOCK_N=min(block_size_n, N), BLOCK_K=block_size_k,
        num_warps=4, num_stages=2,
    )


def _per_group_routing_enabled() -> bool:
    """Read SGLANG_LORA_PER_GROUP_ROUTING (2 calls per layer)."""
    global _PER_GROUP_ROUTING_ENV
    if _PER_GROUP_ROUTING_ENV is None:
        from sglang.srt.environ import envs

        _PER_GROUP_ROUTING_ENV = envs.SGLANG_LORA_PER_GROUP_ROUTING
    return _PER_GROUP_ROUTING_ENV


_PER_GROUP_ROUTING_ENV: bool | None = None


def _fused_delta_enabled() -> bool:
    """Read SGLANG_LORA_FUSED_DELTA (2 calls per layer)."""
    global _FUSED_DELTA_ENV
    if _FUSED_DELTA_ENV is None:
        from sglang.srt.environ import envs

        _FUSED_DELTA_ENV = envs.SGLANG_LORA_FUSED_DELTA
    return _FUSED_DELTA_ENV


_FUSED_DELTA_ENV: bool | None = None


def _merged_experts_fused_moe_lora_add_per_group_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
) -> None:
    """Per-group routing: use original num_experts moe_align instead of
    expanding to max_loras × num_experts virtual experts.

    Instead of one moe_align over 1024 virtual expert buckets, run moe_align
    ONCE with the original num_experts (e.g. 256) and loop over active LoRA
    groups. Each group reuses the same 256-bucket routing but indexes its own
    LoRA A/B weights. The add_output_mask ensures only the group's tokens
    receive the LoRA delta.

    For shared_outer=True (LoRA weights shared across experts), the virtual
    expert space is already small (= max_loras), so we fall back to the
    original virtual-experts path which is efficient for that case.
    """
    # shared_outer: LoRA weights are [max_loras, 1, ...] (shared across experts),
    # but expert_ids from 256-bucket moe_align index 0..255 -> OOB.
    # Fall back to original virtual-experts path for either stage.
    if experts_shared_outer_loras_a or experts_shared_outer_loras_b:
        _merged_experts_fused_moe_lora_add_impl(
            output, hidden_states, lora_a, lora_b, topk_ids, topk_weights,
            token_lora_mapping, mul_routed_weight,
            experts_shared_outer_loras_a, experts_shared_outer_loras_b,
            routing_cache,
        )
        return

    # Performance guard: per-group launches max_loras × (shrink+expand) kernels.
    # With 256 buckets each call saves ~25μs moe_align vs 1024, but each extra
    # group adds ~50-70μs of launch+HBM overhead. Net positive only when
    # max_loras is small. Fall back for large max_loras to avoid regression.
    _max_loras = lora_a.shape[0]
    if _max_loras > 3:
        _merged_experts_fused_moe_lora_add_impl(
            output, hidden_states, lora_a, lora_b, topk_ids, topk_weights,
            token_lora_mapping, mul_routed_weight,
            experts_shared_outer_loras_a, experts_shared_outer_loras_b,
            routing_cache,
        )
        return

    lora_b_list: list[torch.Tensor] = (
        list(lora_b) if isinstance(lora_b, (list, tuple)) else [lora_b]
    )
    n_b = len(lora_b_list)
    b_rank = lora_b_list[0].shape[3]
    max_loras, num_experts_a, max_lora_rank, _ = lora_a.shape
    num_experts_b = lora_b_list[0].shape[1]
    half_out = lora_b_list[0].shape[2]
    input_top_k = (
        1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]
    )

    # --- stage configs (same as original, using per-expert weight shape) ---
    def _get_stage_config(
        weight: torch.Tensor,
        stage_top_k: int,
        n_dim: int,
    ) -> dict[str, Any]:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:
            cfg = get_config_func(token_lora_mapping.shape[0])
        except ValueError:
            K_dim = weight.shape[2]
            N_dim = weight.shape[1]
            if K_dim >= 1024:
                default_block_k = 256
            elif K_dim >= 64:
                default_block_k = 64
            else:
                default_block_k = max(16, K_dim)
            cfg = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": min(64, max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        if _mol_lora_stage_cfg_enabled():
            tuned = _apply_mol_lora_stage_cfg(cfg, n_dim)
            if tuned != cfg:
                _log_mol_lora_stage_cfg_once(cfg, tuned, n_dim)
            cfg = tuned
        return cfg

    def _align_block_size_native(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_experts < 1024:
            from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)
        return _align_block_size_large(topk_ids, block_size, num_experts)

    # --- routing: moe_align ONCE with original num_experts (256) ---
    # Use lora_a[0] / lora_b_list[0][0] for config (all have same per-expert shape)
    a_stage_config = _get_stage_config(
        lora_a[0], input_top_k, lora_a[0].shape[1]
    )
    b_stage_config = _get_stage_config(lora_b_list[0][0], 1, half_out)
    _probe_mol_stage_cfg(a_stage_config, b_stage_config)

    block_size_a = a_stage_config["BLOCK_SIZE_M"]
    block_size_b = b_stage_config["BLOCK_SIZE_M"]

    # Cache routing by (num_experts, block_size) — gate_up and down may share
    # if block sizes match.
    def _get_per_group_routing(num_experts, block_size):
        cache_key = (num_experts, block_size)
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _align_block_size_native(topk_ids, block_size, num_experts)
        )
        # Clamp negatives (same defense as original path)
        sorted_token_ids = torch.where(
            sorted_token_ids < 0,
            sorted_token_ids.new_full((), 2**30),
            sorted_token_ids,
        )
        result = (sorted_token_ids, expert_ids, num_tokens_post_padded)
        if routing_cache is not None:
            routing_cache[cache_key] = result
        return result

    sorted_token_ids_a, expert_ids_a, num_tokens_post_padded_a = (
        _get_per_group_routing(num_experts_a, block_size_a)
    )
    if block_size_b == block_size_a and num_experts_b == num_experts_a:
        sorted_token_ids_b = sorted_token_ids_a
        expert_ids_b = expert_ids_a
        num_tokens_post_padded_b = num_tokens_post_padded_a
    else:
        sorted_token_ids_b, expert_ids_b, num_tokens_post_padded_b = (
            _get_per_group_routing(num_experts_b, block_size_b)
        )

    # --- all-groups kernel: single launch, compile-time loop over MAX_LORAS ---
    # Replaces the Python `for lora_id in range(max_loras)` loop with a
    # Triton `tl.static_range(MAX_LORAS)` compile-time unrolled loop.
    # hidden_states is loaded once per K-tile and reused across groups via
    # L2 cache; base output is loaded+written once. No intermediate HBM buffer.
    block_m = b_stage_config["BLOCK_SIZE_M"]
    block_n = b_stage_config.get("BLOCK_SIZE_N", 64)
    block_k = b_stage_config.get("BLOCK_SIZE_K", 64)

    for b_idx in range(n_b):
        if n_b == 1:
            a_half = lora_a  # [max_loras, num_experts, rank, K]
            b_half = lora_b_list[0]  # [max_loras, num_experts, N, rank]
            out_half = output
        else:
            a_half = lora_a[:, :, b_idx * b_rank : (b_idx + 1) * b_rank, :]
            b_half = lora_b_list[b_idx]
            out_half = output[..., b_idx * half_out : (b_idx + 1) * half_out].contiguous()

        _invoke_fused_lora_delta_all_groups(
            hidden_states, a_half, b_half, out_half,
            sorted_token_ids_b, expert_ids_b, num_tokens_post_padded_b,
            token_lora_mapping,
            topk_ids.shape[1] if input_top_k > 1 else 1,
            b_rank, max_loras,
            block_size_m=block_m, block_size_n=min(block_n, b_half.shape[2]),
            block_size_k=block_k,
        )
        if n_b != 1:
            output[..., b_idx * half_out : (b_idx + 1) * half_out].copy_(out_half)


def _merged_experts_fused_moe_lora_add_op(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    if _per_group_routing_enabled():
        _merged_experts_fused_moe_lora_add_per_group_impl(
            output, hidden_states, lora_a, lora_b, topk_ids, topk_weights,
            token_lora_mapping, mul_routed_weight,
            experts_shared_outer_loras_a, experts_shared_outer_loras_b,
        )
    else:
        _merged_experts_fused_moe_lora_add_impl(
            output,
            hidden_states,
            lora_a,
            lora_b,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            mul_routed_weight,
            experts_shared_outer_loras_a,
            experts_shared_outer_loras_b,
        )


from sglang.srt.utils.common import direct_register_custom_op

direct_register_custom_op(
    op_name="merged_experts_fused_moe_lora_add",
    op_func=_merged_experts_fused_moe_lora_add_op,
    mutates_args=["output"],
    fake_impl=_merged_experts_fused_moe_lora_add_fake,
)


def merged_experts_fused_moe_lora_add(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
) -> None:
    """Public API: wraps the registered op with routing_cache support.

    ``lora_b`` accepts a sequence of length 2 for the gate_up case (each B
    holds one half of the stacked output, rank ``r``, with A's rank ``2*r``);
    a single tensor is used for the down case.
    """
    if _per_group_routing_enabled():
        _merged_experts_fused_moe_lora_add_per_group_impl(
            output,
            hidden_states,
            lora_a,
            lora_b,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            mul_routed_weight,
            experts_shared_outer_loras_a,
            experts_shared_outer_loras_b,
            routing_cache,
        )
    else:
        _merged_experts_fused_moe_lora_add_impl(
            output,
            hidden_states,
            lora_a,
            lora_b,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            mul_routed_weight,
            experts_shared_outer_loras_a,
            experts_shared_outer_loras_b,
            routing_cache,
        )
