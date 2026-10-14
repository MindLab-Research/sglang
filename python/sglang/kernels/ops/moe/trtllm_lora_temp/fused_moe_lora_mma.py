# -*- coding: utf-8 -*-
"""DeepSeek-V4.1-Flash MoE LoRA fused Triton kernels — W2-b MMA form (2026-10-13).

Replaces the bf16 unfused-act GEMM1 fallback (207.9µs/call, the theory-round
root cause: no unfused-act GEMM1 cubin exists in the quantized family) with
native tcgen05 MXFP MMA via triton tl.dot_scaled on sm_103a:

  K1: grouped GEMM1 (e4m3 act × e2m1 w13, per-32 e8m0 scales) + gate_up LoRA
      delta + bf16-rounded clamped silu_and_mul + per-32 round-up quant
      -> q [EM, I] fp8 + s [EM, I/32] uint8
  K2: grouped GEMM2 (e4m3 act × e2m1 w2) + down LoRA delta + routed weight
      -> fp32 atomic accumulate

Layout contracts (probe-verified bit-exact, probe_dot_v4 2026-10-13):
- w13 [E, 2I, K/2] uint8: byte p packs logical K {2p=lo nibble, 2p+1=hi};
  dot_scaled rhs tile [BK/2, N] read transposed: flat = n*(K//2) + p.
- w13_scale [E, 2I, K/32] uint8 e8m0 (2^(b-127)); dot_scaled rhs_scale tile
  [N, BK/32], "do NOT transpose" — matches checkpoint layout verbatim.
- A side: e4m3 [M, K] + lhs_scale [M, K/32] — same MXFP8 activation format the
  base path produces (_prepare_flashinfer_mxfp8_activations), so e2e wiring
  reuses it directly.
- Accumulator bf16-rounded before clamp (limit 10.0 bf16-exact).
- M2 GEMV numerics contract carried over (act median_rel=0 out of gate).
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

SF_ACT = 32                        # logical elements per K1-output scale group
SF_ACT_C = tl.constexpr(32)


# =============================================================================
# dequant/quant helpers
# =============================================================================
@triton.jit
def _e4m3_quant_tile_roundup(x, N: tl.constexpr, BLK: tl.constexpr):
    """MXFP8 quant of a [BM, N] fp32 tile -> (e4m3 tile, [BM, N//BLK] bytes).

    Round-UP e8m0, +/-448 saturate (fused_activation_mxfp8_quant.cuh recipe).
    """
    BM: tl.constexpr = x.shape[0]
    xg = tl.reshape(x, (BM, N // BLK, BLK))
    amax = tl.max(tl.abs(xg), axis=2)
    sf_exp = tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1e-30)))
    sf_exp = tl.minimum(tl.maximum(sf_exp, -127.0), 128.0)
    scale = tl.exp2(sf_exp)
    inv = tl.where(sf_exp >= 128.0, 0.0, 1.0 / scale)
    xr = tl.reshape(x, (BM, N // BLK, BLK))
    inv_b = tl.broadcast_to(inv[:, :, None], (BM, N // BLK, BLK))
    q = tl.reshape(xr * inv_b, (BM, N))
    q = tl.maximum(tl.minimum(q, 448.0), -448.0).to(tl.float8e4nv)
    return q, (sf_exp + 127.0).to(tl.uint8)


# =============================================================================
# K1 : grouped GEMM1 (mxfp8 x mxfp4) + LoRA delta + clamped silu + quant
# =============================================================================
@triton.jit
def _fused_gateup_fp4_mma_kernel(
    a_ptr, a_sf_ptr,             # [EM, K] e4m3, [EM, K/32] uint8 (sorted rows)
    b_ptr, b_sf_ptr,             # [E, 2I, K/2] uint8, [E, 2I, K/32] uint8
    d_ptr,                       # [T, topk, 2I] bf16 (token-major, pre-act)
    sorted_token_ids_ptr, expert_ids_ptr,
    q_ptr, s_ptr,                # [EM, I] e4m3, [EM, I/32] uint8
    top_k, limit,
    K, I, EM, num_valid,
    stride_be, stride_bn, stride_bk,      # b:  [E,2I,K/2] strides (elems)
    stride_bse, stride_bsn,               # b_sf: [E,2I,K/32] strides
    stride_d0, stride_d1,                 # delta strides (token, k)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    sid = tl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < EM,
                  other=num_valid)
    live = sid < num_valid
    token = sid // top_k
    kidx = sid % top_k
    expert = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # within [0, I)
    offs_k = tl.arange(0, BLOCK_K)                             # logical K
    offs_pk = tl.arange(0, BLOCK_K // 2)                       # packed bytes
    offs_g = tl.arange(0, BLOCK_K // 32)                       # scale groups

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b_base = b_ptr + expert.to(tl.int64) * stride_be
    sf_base = b_sf_ptr + expert.to(tl.int64) * stride_bse

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k                             # logical K
        pk = k0 * (BLOCK_K // 2) + offs_pk                     # packed bytes
        g = k0 * (BLOCK_K // 32) + offs_g                      # scale groups

        # ---- A: e4m3 activations + per-32 e8m0 (contiguous, pre-gathered)
        a = tl.load(a_ptr + offs_m[:, None] * K + kk[None, :],
                    mask=live[:, None], other=0.0)             # [BM, BK] e4m3
        a_s = tl.load(a_sf_ptr + offs_m[:, None] * (K // 32) + g[None, :],
                      mask=live[:, None], other=0)             # [BM, BK/32]

        # ---- B gate rows: rhs tile [BK/2, BN], flat = n*(K//2) + p (probe v4)
        bg = tl.load(b_base + pk[:, None] + offs_n[None, :] * stride_bn,
                     mask=(pk[:, None] * 2 < K), other=0)
        bg_s = tl.load(sf_base + offs_n[:, None] * stride_bsn + g[None, :])
        acc_g = tl.dot_scaled(a, a_s, "e4m3", bg, bg_s, "e2m1", acc_g)

        # ---- B up rows (I + n)
        bu = tl.load(b_base + pk[:, None] + (I + offs_n[None, :]) * stride_bn,
                     mask=(pk[:, None] * 2 < K), other=0)
        bu_s = tl.load(sf_base + (I + offs_n[:, None]) * stride_bsn + g[None, :])
        acc_u = tl.dot_scaled(a, a_s, "e4m3", bu, bu_s, "e2m1", acc_u)

    # ---- LoRA delta (token-major), pre-activation
    d_base = d_ptr + token.to(tl.int64) * stride_d0 + kidx * stride_d1
    d_g = tl.load(d_base[:, None] + offs_n[None, :], mask=live[:, None],
                  other=0.0)
    d_u = tl.load(d_base[:, None] + I + offs_n[None, :], mask=live[:, None],
                  other=0.0)
    acc_g += d_g.to(tl.float32)
    acc_u += d_u.to(tl.float32)

    # ---- bf16-rounded accumulators, then clamped silu_and_mul
    g16 = acc_g.to(tl.bfloat16).to(tl.float32)
    u16 = acc_u.to(tl.bfloat16).to(tl.float32)
    if HAS_LIMIT:
        g16 = tl.minimum(g16, limit)               # gate: max-clamp only
        u16 = tl.maximum(u16, -limit)              # up: +/- clamp
        u16 = tl.minimum(u16, limit)
    y = g16 * tl.sigmoid(g16) * u16                            # silu(gate) * up

    # ---- per-32 MXFP8 quant (round-up pow2) + store at sorted rows
    q, sf = _e4m3_quant_tile_roundup(y, BLOCK_N, SF_ACT_C)
    offs_sg = pid_n * (BLOCK_N // SF_ACT_C) + tl.arange(0, BLOCK_N // SF_ACT_C)
    tl.store(q_ptr + offs_m[:, None] * I + offs_n[None, :], q,
             mask=live[:, None])
    tl.store(s_ptr + offs_m[:, None] * (I // SF_ACT_C) + offs_sg[None, :], sf,
             mask=live[:, None])


# =============================================================================
# K2 : grouped GEMM2 (mxfp8 x mxfp4) + LoRA delta + routed weight + atomic
# =============================================================================
@triton.jit
def _fused_down_fp4_mma_kernel(
    q_ptr, s_ptr,                # [EM, I] e4m3, [EM, I/32] uint8 (K1 out)
    b_ptr, b_sf_ptr,             # [E, K, I/2] uint8, [E, K, I/32] uint8
    d_ptr,                       # [T, topk, I] bf16 (token-major, pre-weight)
    sorted_token_ids_ptr, expert_ids_ptr,
    topk_weights_ptr,            # [T*topk] float32 (routed-scaled)
    out_ptr,                     # [T, K] float32 (atomic add)
    K, I, EM, num_valid, top_k,
    stride_be, stride_bn, stride_bk,      # b: [E,K,I/2] strides (elems)
    stride_bse, stride_bsn,               # b_sf: [E,K,I/32] strides
    stride_d0, stride_d1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    sid = tl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < EM,
                  other=num_valid)
    live = sid < num_valid
    token = sid // top_k
    kidx = sid % top_k
    expert = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # within [0, K)
    offs_k = tl.arange(0, BLOCK_K)                             # over I
    offs_pk = tl.arange(0, BLOCK_K // 2)
    offs_g = tl.arange(0, BLOCK_K // 32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b_base = b_ptr + expert.to(tl.int64) * stride_be
    sf_base = b_sf_ptr + expert.to(tl.int64) * stride_bse

    for k0 in range(0, tl.cdiv(I, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k                             # over I
        pk = k0 * (BLOCK_K // 2) + offs_pk
        g = k0 * (BLOCK_K // 32) + offs_g

        a = tl.load(q_ptr + offs_m[:, None] * I + kk[None, :],
                    mask=live[:, None], other=0.0)             # [BM, BK] e4m3
        a_s = tl.load(s_ptr + offs_m[:, None] * (I // 32) + g[None, :],
                      mask=live[:, None], other=0)             # [BM, BK/32]

        b = tl.load(b_base + pk[:, None] + offs_n[None, :] * stride_bn,
                    mask=(pk[:, None] * 2 < I), other=0)       # [BK/2, BN]
        b_s = tl.load(sf_base + offs_n[:, None] * stride_bsn + g[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e2m1", acc)

    # ---- down LoRA delta (pre-weight)
    d_base = d_ptr + token.to(tl.int64) * stride_d0 + kidx * stride_d1
    d = tl.load(d_base[:, None] + offs_n[None, :], mask=live[:, None],
                other=0.0)
    acc += d.to(tl.float32)

    # ---- routed weight + atomic-add finalize
    w = tl.load(topk_weights_ptr + sid, mask=live, other=0.0)
    tl.atomic_add(out_ptr + token.to(tl.int64)[:, None] * K + offs_n[None, :],
                  acc * w[:, None], mask=live[:, None], sem="relaxed")


# =============================================================================
# host
# =============================================================================
def _quant_per32_roundup(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-32 UE8M0 (round-up) MXFP8 quant in torch. Returns (fp8, sf bytes)."""
    M, K = x.shape
    nb = K // 32
    amax = x.float().abs().view(M, nb, 32).amax(-1)
    sf_exp = torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-30)))
    sf_exp = sf_exp.clamp(-127.0, 128.0)
    scale = torch.exp2(sf_exp)
    xq = (x.float().view(M, nb, 32) / scale[:, :, None]) \
        .clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return xq.reshape(M, K), (sf_exp + 127.0).to(torch.uint8)


def _align_block_size_torch(topk_ids: torch.Tensor, E: int, block: int):
    """Pure-torch moe_align_block_size (correctness reference)."""
    T, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    dev = topk_ids.device
    counts = torch.bincount(flat, minlength=E)
    blocks_per_expert = (counts + block - 1) // block
    cum_blocks = torch.cumsum(blocks_per_expert, 0)
    npp = int(cum_blocks[-1].item()) * block
    order = torch.argsort(flat, stable=True)
    n_live = int(counts.sum().item())
    starts = (cum_blocks - blocks_per_expert) * block
    e_sorted = flat[order]
    rank = torch.arange(n_live, device=dev) - torch.repeat_interleave(
        (torch.cumsum(counts, 0) - counts), counts)
    targets = starts[e_sorted] + rank
    sorted_ids = torch.full((npp,), T * topk, dtype=torch.int32, device=dev)
    sorted_ids[targets] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(E, device=dev, dtype=torch.int32),
        blocks_per_expert.to(torch.int32))
    return sorted_ids, expert_ids, npp


def fused_moe_lora_mma(
    hidden_states: torch.Tensor,          # [T, K] bf16
    topk_ids: torch.Tensor,               # [T, topk] int32/64
    topk_weights: torch.Tensor,           # [T, topk] float32 (routed-scaled)
    w13: torch.Tensor,                    # [E, 2I, K/2] uint8 (packed fp4)
    w13_scale: torch.Tensor,              # [E, 2I, K/32] uint8 e8m0
    w2: torch.Tensor,                     # [E, K, I/2] uint8
    w2_scale: torch.Tensor,               # [E, K, I/32] uint8
    gate_up_delta: torch.Tensor,          # [T, topk, 2I] bf16 (VE, token-major)
    down_delta: torch.Tensor,             # [T, topk, I] bf16
    swiglu_limit: Optional[float],        # V4.1: 10.0
    block_size: int = 16,
    align_backend: str = "torch",         # "torch" (val) | "cuda" (e2e)
    block_n1: int = 64,
    block_k1: int = 256,
    block_n2: int = 128,
    block_k2: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    return_act: bool = False,
) -> torch.Tensor:
    T, K = hidden_states.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = hidden_states.device
    assert K % 256 == 0 and I % 64 == 0, "K/I must divide BLOCK_K cleanly"

    # ---- activations -> MXFP8 per-32 (e2e reuses the dispatcher's x_quant)
    x_q, x_sf = _quant_per32_roundup(hidden_states)

    # ---- align (assignments -> expert-sorted m-blocks)
    if align_backend == "torch":
        sorted_token_ids, expert_ids, npp = _align_block_size_torch(
            topk_ids, E, block_size)
        EM = max(npp, 1)
    else:
        from sglang.kernels.ops.moe import moe_align_block_size

        EM = E * block_size + num_valid
        sorted_token_ids = torch.full((EM,), num_valid, dtype=torch.int32,
                                      device=dev)
        expert_ids = torch.zeros(((EM + block_size - 1) // block_size,),
                                 dtype=torch.int32, device=dev)
        num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
        cumsum = torch.empty((E + 1,), dtype=torch.int32, device=dev)
        moe_align_block_size(
            topk_ids.to(torch.int32), E, block_size,
            sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum,
            pad_sorted_token_ids=True,
        )

    # ---- pre-gather A rows into sorted order (probe-2026-10-13: contiguous
    # loads required; in-kernel gather miscompiles on sm_103a triton)
    tok = torch.clamp(sorted_token_ids[:EM] // topk, max=T - 1).long()
    x_q_g = x_q[tok].contiguous()
    x_sf_g = x_sf[tok].contiguous()

    q = torch.empty((EM, I), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((EM, I // SF_ACT), dtype=torch.uint8, device=dev)

    grid1 = (triton.cdiv(EM, block_size), triton.cdiv(I, block_n1))
    _fused_gateup_fp4_mma_kernel[grid1](
        x_q_g, x_sf_g,
        w13.view(torch.uint8), w13_scale.view(torch.uint8),
        gate_up_delta,
        sorted_token_ids, expert_ids,
        q, s,
        topk, float(swiglu_limit) if swiglu_limit is not None else 0.0,
        K, I, EM, num_valid,
        w13.stride(0), w13.stride(1), w13.stride(2),
        w13_scale.stride(0), w13_scale.stride(1),
        gate_up_delta.stride(0), gate_up_delta.stride(1),
        BLOCK_M=block_size, BLOCK_N=block_n1, BLOCK_K=block_k1,
        HAS_LIMIT=swiglu_limit is not None,
        num_warps=num_warps, num_stages=num_stages,
    )

    out_f = torch.zeros((T, K), dtype=torch.float32, device=dev)

    grid2 = (triton.cdiv(EM, block_size), triton.cdiv(K, block_n2))
    _fused_down_fp4_mma_kernel[grid2](
        q, s,
        w2.view(torch.uint8), w2_scale.view(torch.uint8),
        down_delta,
        sorted_token_ids, expert_ids,
        topk_weights.reshape(-1).to(torch.float32).contiguous(),
        out_f,
        K, I, EM, num_valid, topk,
        w2.stride(0), w2.stride(1), w2.stride(2),
        w2_scale.stride(0), w2_scale.stride(1),
        down_delta.stride(0), down_delta.stride(1),
        BLOCK_M=block_size, BLOCK_N=block_n2, BLOCK_K=block_k2,
        num_warps=num_warps, num_stages=num_stages,
    )
    out = out_f.to(torch.bfloat16)
    if return_act:
        return out, q, s
    return out
