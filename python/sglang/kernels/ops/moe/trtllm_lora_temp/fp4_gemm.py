# -*- coding: utf-8 -*-
"""Triton fp4-dequant grouped GEMM: d[g,m,n] = sum_k a[g,m,k] * W4[g,n,k].

A stays bf16; B is MXFP4 (E2M1 packed [G,N,K/2] int8 + f32 SF [G,N,K/32]).
Dequant happens in-kernel (pure-arithmetic E2M1 decode, no LUT gather) so
HBM traffic stays at fp4 width. Replaces the bf16 bmm pair (283->target ~100us).
"""
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_dec(c):
    """E2M1 nibble (0..15) -> float. Pure arithmetic, no gather."""
    sign = tl.where((c & 8) != 0, -1.0, 1.0)
    m = c & 7
    # m: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
    mag = tl.where(
        m == 0, 0.0,
        tl.where(m == 1, 0.5,
                 tl.exp2((m >> 1).to(tl.float32) - 1.0)
                 * (1.0 + (m & 1).to(tl.float32) * 0.5)))
    return sign * mag


@triton.jit
def _k_gemm_fp4(a_ptr, wp_ptr, wsf_ptr, d_ptr,
                M, stride_ag, stride_am,
                stride_wg, stride_wn,
                stride_wsf_g, stride_wsf_n,
                stride_dg, stride_dm,
                N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """grid = (cdiv(M,BM), cdiv(N,BN), G). A bf16 [G,M,K], B fp4 [G,N,K/2]+SF."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = a_ptr + g * stride_ag + offs_m[:, None] * stride_am
    w_base = wp_ptr + g * stride_wg + offs_n[:, None] * stride_wn
    sf_base = wsf_ptr + g * stride_wsf_g + offs_n[:, None] * stride_wsf_n

    # K loop: each iter consumes BLOCK_K fp4 bytes = BLOCK_K*2 k-elems
    for kb in range(0, K // 2, BLOCK_K):
        offs_k2 = kb * 2 + tl.arange(0, BLOCK_K * 2)          # k indices
        a_tile = tl.load(a_base + offs_k2[None, :],
                         mask=mask_m[:, None], other=0.0)     # [BM, BK2] bf16
        wb = tl.load(w_base + kb + tl.arange(0, BLOCK_K)[None, :])  # [BN, BK] uint8
        lo = _e2m1_dec((wb & 0x0F).to(tl.int32))
        hi = _e2m1_dec(((wb >> 4) & 0x0F).to(tl.int32))
        # interleave -> [BN, BK2] dequantized (unscaled)
        wv = tl.interleave(lo, hi)
        # per-32 SF: sf idx = offs_k2 // 32
        sf_idx = offs_k2 // 32
        sfv = tl.load(sf_base + sf_idx[None, :] - (kb * 2) // 32,
                      mask=(sf_idx[None, :] < (K // 32)), other=1.0)
        w_tile = (wv * sfv).to(tl.bfloat16)                   # [BN, BK2]
        acc += tl.dot(a_tile, tl.trans(w_tile))

    d_ptrs = d_ptr + g * stride_dg + offs_m[:, None] * stride_dm + offs_n[None, :]
    tl.store(d_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def fp4_grouped_gemm(a: torch.Tensor, wp: torch.Tensor, wsf: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """a [G, M, K] bf16, wp [G, N, K/2] int8, wsf [G, N, K/32] f32 -> [G, M, N] bf16."""
    G, M, K = a.shape
    N = wp.shape[1]
    d = out if out is not None else torch.empty(
        G, M, N, device=a.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 128, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), G)
    _k_gemm_fp4[grid](a, wp, wsf, d, M,
                      a.stride(0), a.stride(1),
                      wp.stride(0), wp.stride(1),
                      wsf.stride(0), wsf.stride(1),
                      d.stride(0), d.stride(1),
                      N=N, K=K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                      num_warps=8, num_stages=2)
    return d
