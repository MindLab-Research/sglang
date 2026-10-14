# -*- coding: utf-8 -*-
"""DeepSeek-V4.1-Flash MoE LoRA fused Triton kernels (GEMV form, decode regime).

Design (2026-10-13, supersedes the block-padded tl.dot version):
- One program per (assignment, N-tile); assignment = (token, topk slot) pair.
- Assignments are expert-sorted on the host (torch argsort) for L2 weight
  reuse; NO moe_align_block_size, NO M-padding, NO tl.dot (a 16-row align
  block would pad ~1.3-token/expert decode batches 12x in compute).
- A side is bf16 straight from hidden_states (no intermediate fp8 quant: the
  in-kernel fp8+UE8M0 dequant chain feeding tl.dot miscompiled on sm_103a and
  the K2 consumes bf16 anyway).
- K1 output: compact bf16 act [A, I] (no per-64 quant; K2's dot input is bf16).
- K2 finalize: fp32 atomic accumulation into out[T, K], single bf16 cast.
- B side: MXFP4 weights unpacked in-kernel (e2m1 nibble by logical-K parity,
  UE8M0 scale per 32 logical K), identical chain validated by offline probes.

Contracts (pinned by source, do not flip without re-validating):
- w13 [E, 2I, K/2] int8: byte k packs logical K {2k=lo nibble, 2k+1=hi nibble}.
- w13_scale [E, 2I, K/32] uint8: UE8M0, value = 2^(b-127).
- gate = w13 rows [0, I) (silu side, max-clamp), up = rows [I, 2I) (+/-clamp).
- Accumulator bf16-rounded BEFORE the clamp (limit 10.0 is bf16-exact).
- Activation: y = silu(clamp_max(g, limit)) * clamp(u, -limit, limit).
"""
from typing import Optional

import torch
import triton
import triton.language as tl

SF_BLOCK_B = 32    # logical K elements per B (weight) scale group
# kernel-visible constexpr (triton cannot read plain int globals in @jit bodies)
SF_BLOCK_B_C = tl.constexpr(32)


# =============================================================================
# dequantisation helpers
# =============================================================================
@triton.jit
def _e2m1_to_fp32(k):
    """Unpack one E2M1 nibble (0..15) to fp32.

        0 -> 0.0 ; 1 -> 0.5 ; 2 -> 1.0 ; 3 -> 1.5
        4 -> 2.0 ; 5 -> 3.0 ; 6 -> 4.0 ; 7 -> 6.0   (8..15 = negative)
    """
    kk = k & 0x7
    sign = tl.where((k & 0x8) != 0, -1.0, 1.0)
    mag = tl.where(
        kk == 0,
        0.0,
        tl.where(
            kk == 1,
            0.5,
            tl.exp2((kk >> 1).to(tl.float32) - 1.0)
            * (1.0 + (kk & 1).to(tl.float32) * 0.5),
        ),
    )
    return sign * mag


@triton.jit
def _ue8m0_to_fp32(sf_byte):
    """UE8M0 scale byte -> fp32 (2^(b-127))."""
    b = sf_byte.to(tl.float32)
    return tl.where(sf_byte == 0, 0.0, tl.exp2(b - 127.0))


# =============================================================================
# K1 : per-assignment gate_up GEMV + LoRA delta + clamped silu_and_mul
# =============================================================================
@triton.jit
def _fused_gateup_lora_kernel(
    x_ptr,                        # [T, K] bf16 (token-major)
    b_ptr, b_sf_ptr,              # [E, 2I, K/2] int8, [E, 2I, K/32] uint8
    d_ptr,                        # [T, topk, 2I] bf16 (token-major, pre-act)
    expert_ptr, token_ptr, kidx_ptr,   # [A] int32 (expert-sorted)
    act_ptr,                      # [A, I] bf16 out (compact, sorted order)
    limit,
    K, I, A,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn,
    stride_d0, stride_d1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    aid = tl.program_id(0)
    pid_n = tl.program_id(1)
    token = tl.load(token_ptr + aid)
    kidx = tl.load(kidx_ptr + aid)
    expert = tl.load(expert_ptr + aid)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # within [0, I)
    offs_k = tl.arange(0, BLOCK_K)                             # logical K

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    b_base = b_ptr + expert.to(tl.int64) * stride_be
    sf_base = b_sf_ptr + expert.to(tl.int64) * stride_bse

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k                             # logical K
        k_mask = kk < K

        x = tl.load(x_ptr + token.to(tl.int64) * K + kk,
                    mask=k_mask, other=0.0).to(tl.float32)     # [BK]

        byte_idx = kk // 2
        nib_hi = (kk & 1) != 0

        bg_u8 = tl.load(
            b_base + offs_n[None, :] * stride_bn + byte_idx[:, None] * stride_bk,
            mask=k_mask[:, None], other=0,
        )
        bg_sf = tl.load(
            sf_base + offs_n[None, :] * stride_bsn + (kk // SF_BLOCK_B_C)[:, None],
            mask=k_mask[:, None], other=0,
        )
        sfv = _ue8m0_to_fp32(bg_sf)
        bg_lo = _e2m1_to_fp32((bg_u8 & 0x0F).to(tl.int32)) * sfv
        bg_hi = _e2m1_to_fp32(((bg_u8 >> 4) & 0x0F).to(tl.int32)) * sfv
        bg = tl.where(nib_hi[:, None], bg_hi, bg_lo)           # [BK, BN] fp32
        acc_g += tl.sum(x[:, None] * bg, 0)

        bu_u8 = tl.load(
            b_base + (I + offs_n[None, :]) * stride_bn
            + byte_idx[:, None] * stride_bk,
            mask=k_mask[:, None], other=0,
        )
        bu_sf = tl.load(
            sf_base + (I + offs_n[None, :]) * stride_bsn
            + (kk // SF_BLOCK_B_C)[:, None],
            mask=k_mask[:, None], other=0,
        )
        sfv_u = _ue8m0_to_fp32(bu_sf)
        bu_lo = _e2m1_to_fp32((bu_u8 & 0x0F).to(tl.int32)) * sfv_u
        bu_hi = _e2m1_to_fp32(((bu_u8 >> 4) & 0x0F).to(tl.int32)) * sfv_u
        bu = tl.where(nib_hi[:, None], bu_hi, bu_lo)           # [BK, BN] fp32
        acc_u += tl.sum(x[:, None] * bu, 0)

    # ---- LoRA delta (token-major), pre-activation
    d_base = d_ptr + token.to(tl.int64) * stride_d0 + kidx * stride_d1
    acc_g += tl.load(d_base + offs_n).to(tl.float32)
    acc_u += tl.load(d_base + I + offs_n).to(tl.float32)

    # ---- bf16-rounded accumulators, then clamped silu_and_mul
    g16 = acc_g.to(tl.bfloat16).to(tl.float32)
    u16 = acc_u.to(tl.bfloat16).to(tl.float32)
    if HAS_LIMIT:
        g16 = tl.minimum(g16, limit)               # gate: max-clamp only
        u16 = tl.maximum(u16, -limit)              # up: +/- clamp
        u16 = tl.minimum(u16, limit)
    y = g16 * tl.sigmoid(g16) * u16                            # silu(gate) * up

    tl.store(act_ptr + aid.to(tl.int64) * I + offs_n, y.to(tl.bfloat16))


# =============================================================================
# K2 : per-assignment down GEMV + LoRA delta + routed weight + atomic finalize
# =============================================================================
@triton.jit
def _fused_down_lora_kernel(
    act_ptr,                      # [A, I] bf16 (K1 out, sorted order)
    b_ptr, b_sf_ptr,              # [E, K, I/2] int8, [E, K, I/32] uint8
    d_ptr,                        # [T, topk, I] bf16 (token-major, pre-weight)
    expert_ptr, token_ptr, kidx_ptr,   # [A] int32 (expert-sorted)
    w_ptr,                        # [A] float32 routed weights
    out_ptr,                      # [T, K] float32 (atomic add)
    K, I, A,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn,
    stride_d0, stride_d1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    aid = tl.program_id(0)
    pid_n = tl.program_id(1)
    token = tl.load(token_ptr + aid)
    kidx = tl.load(kidx_ptr + aid)
    expert = tl.load(expert_ptr + aid)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # within [0, K)
    offs_k = tl.arange(0, BLOCK_K)                             # over I

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    b_base = b_ptr + expert.to(tl.int64) * stride_be
    sf_base = b_sf_ptr + expert.to(tl.int64) * stride_bse

    for k0 in range(0, tl.cdiv(I, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k                             # over I
        k_mask = kk < I

        a = tl.load(act_ptr + aid.to(tl.int64) * I + kk,
                    mask=k_mask, other=0.0).to(tl.float32)     # [BK]

        byte_idx = kk // 2
        nib_hi = (kk & 1) != 0
        bu8 = tl.load(
            b_base + offs_n[None, :] * stride_bn + byte_idx[:, None] * stride_bk,
            mask=k_mask[:, None], other=0,
        )
        bsf = tl.load(
            sf_base + offs_n[None, :] * stride_bsn + (kk // SF_BLOCK_B_C)[:, None],
            mask=k_mask[:, None], other=0,
        )
        sfv = _ue8m0_to_fp32(bsf)
        lo = _e2m1_to_fp32((bu8 & 0x0F).to(tl.int32)) * sfv
        hi = _e2m1_to_fp32(((bu8 >> 4) & 0x0F).to(tl.int32)) * sfv
        b = tl.where(nib_hi[:, None], hi, lo)                  # [BK, BN] fp32
        acc += tl.sum(a[:, None] * b, 0)

    # ---- down LoRA delta (pre-weight)
    d_base = d_ptr + token.to(tl.int64) * stride_d0 + kidx * stride_d1
    acc += tl.load(d_base + offs_n).to(tl.float32)

    # ---- routed weight + atomic-add finalize
    w = tl.load(w_ptr + aid)
    tl.atomic_add(out_ptr + token.to(tl.int64) * K + offs_n, acc * w)


# =============================================================================
# host entry
# =============================================================================
def fused_moe_lora_triton(
    hidden_states: torch.Tensor,          # [T, K] bf16
    topk_ids: torch.Tensor,               # [T, topk] int32/64
    topk_weights: torch.Tensor,           # [T, topk] float32 (routed-scaled)
    w13: torch.Tensor,                    # [E, 2I, K/2] int8 (packed fp4)
    w13_scale: torch.Tensor,              # [E, 2I, K/32] uint8
    w2: torch.Tensor,                     # [E, K, I/2] int8
    w2_scale: torch.Tensor,               # [E, K, I/32] uint8
    gate_up_delta: torch.Tensor,          # [T, topk, 2I] bf16 (VE, token-major)
    down_delta: torch.Tensor,             # [T, topk, I] bf16
    swiglu_limit: Optional[float],        # V4.1: 10.0
    block_size: int = 16,                 # unused (kept for the M3 seam compat)
    align_backend: str = "cuda",          # unused (no align in GEMV form)
    block_n1: int = 64,
    block_k1: int = 128,
    block_n2: int = 128,
    block_k2: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    return_act: bool = False,             # debug: also return the K1 act
) -> torch.Tensor:
    T, K = hidden_states.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    topk = topk_ids.shape[1]
    A = T * topk
    dev = hidden_states.device

    # expert-sorted assignment order (L2 weight reuse across the same-expert
    # programs); the deltas/weights are gathered to match
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    expert_s = flat[order].to(torch.int32).contiguous()
    token_s = (order // topk).to(torch.int32).contiguous()
    kidx_s = (order % topk).to(torch.int32).contiguous()
    w_s = topk_weights.reshape(-1).to(torch.float32)[order].contiguous()

    act = torch.empty((A, I), dtype=torch.bfloat16, device=dev)

    grid1 = (A, triton.cdiv(I, block_n1))
    _fused_gateup_lora_kernel[grid1](
        hidden_states,
        w13.view(torch.int8), w13_scale.view(torch.uint8),
        gate_up_delta,
        expert_s, token_s, kidx_s,
        act,
        float(swiglu_limit) if swiglu_limit is not None else 0.0,
        K, I, A,
        w13.stride(0), w13.stride(1), w13.stride(2),
        w13_scale.stride(0), w13_scale.stride(1),
        gate_up_delta.stride(0), gate_up_delta.stride(1),
        BLOCK_N=block_n1, BLOCK_K=block_k1,
        HAS_LIMIT=swiglu_limit is not None,
        num_warps=num_warps, num_stages=num_stages,
    )

    out_f = torch.zeros((T, K), dtype=torch.float32, device=dev)

    grid2 = (A, triton.cdiv(K, block_n2))
    _fused_down_lora_kernel[grid2](
        act,
        w2.view(torch.int8), w2_scale.view(torch.uint8),
        down_delta,
        expert_s, token_s, kidx_s,
        w_s,
        out_f,
        K, I, A,
        w2.stride(0), w2.stride(1), w2.stride(2),
        w2_scale.stride(0), w2_scale.stride(1),
        down_delta.stride(0), down_delta.stride(1),
        BLOCK_N=block_n2, BLOCK_K=block_k2,
        num_warps=num_warps, num_stages=num_stages,
    )
    out = out_f.to(torch.bfloat16)
    if return_act:
        return out, act
    return out
