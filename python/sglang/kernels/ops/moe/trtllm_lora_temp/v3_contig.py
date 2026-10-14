# -*- coding: utf-8 -*-
"""DeepGEMM contiguous grouped GEMM path for v3 (compact layout).

Replaces the bf16 bmm pair (283.5us/call in-graph) with DeepGEMM
m_grouped_fp8_fp4_gemm_nt_contiguous (~48.7us @ G=96, M=448) — requires:
  - compact row layout (per-expert rows contiguous, M per group aligned)
  - fp8 activations + SF transform (packed UE8M0), fp4 weights + SF transform
Verified numerics on main 2.8.2: G2 per-group rel 0.0267/0.0268 (PASS).
Env: SGLANG_V3_CONTIG=1 (weights cached fp4-side after first call).
"""
import os
from typing import Optional

import torch


def _contig_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


_V3_CONTIG_WCACHE: dict = {}


def _prep_contig_weights(w13_bf: torch.Tensor, w2_bf: torch.Tensor, deep_gemm,
                        E: int) -> tuple:
    """bf16 [E, 2I, K] / [E, K, I] -> fp4 packed + SF transform (cached)."""
    key = (w13_bf.data_ptr(), w2_bf.data_ptr())
    if key in _V3_CONTIG_WCACHE:
        return _V3_CONTIG_WCACHE[key]
    import torch as _t
    NI, K = w13_bf.shape[1], w13_bf.shape[2]
    I = w2_bf.shape[2]
    wp13 = _t.empty(E, NI, K // 2, device=w13_bf.device, dtype=_t.int8)
    wp2 = _t.empty(E, w2_bf.shape[1], I // 2, device=w2_bf.device, dtype=_t.int8)
    sf13_raw = _t.empty(E, NI, K // 32, device=w13_bf.device, dtype=_t.float32)
    sf2_raw = _t.empty(E, w2_bf.shape[1], I // 32, device=w2_bf.device, dtype=_t.float32)
    for g in range(E):
        p, s = deep_gemm.per_token_cast_to_fp4(w13_bf[g], use_ue8m0=True, gran_k=32)
        wp13[g], sf13_raw[g] = p, s
        p2, s2 = deep_gemm.per_token_cast_to_fp4(w2_bf[g], use_ue8m0=True, gran_k=32)
        wp2[g], sf2_raw[g] = p2, s2
    sf13 = deep_gemm.transform_sf_into_required_layout(
        sf13_raw, NI, K, (1, 1, 32), num_groups=E, is_sfa=False)
    sf2 = deep_gemm.transform_sf_into_required_layout(
        sf2_raw, w2_bf.shape[1], I, (1, 1, 32), num_groups=E, is_sfa=False)
    _V3_CONTIG_WCACHE[key] = (wp13, sf13, wp2, sf2)
    return _V3_CONTIG_WCACHE[key]


@torch.no_grad()
def v3_gemm_stage_contig(
    hidden_states: torch.Tensor,          # [T, K] bf16
    topk_ids: torch.Tensor,               # [T, topk] int (local expert ids)
    topk_weights: torch.Tensor,           # [T, topk] float32
    w13_bf: torch.Tensor,                 # [E, 2I, K] bf16 (dequant source)
    w2_bf: torch.Tensor,                  # [E, K, I] bf16
    gate_up_lora_delta: torch.Tensor,     # [T, topk, 2I_padded] bf16
    down_lora_delta: Optional[torch.Tensor],
    swiglu_limit: Optional[float],
    activation_lora_input: Optional[torch.Tensor],
    num_local_experts: int,
    deep_gemm,
    M_ALIGN: int = 224,                   # contiguous M-per-group alignment
    gemm1_alpha: float = 1.702,
    gemm1_beta: float = 1.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    import triton
    from sglang.kernels.ops.moe.trtllm_lora_temp.deepgemm_path import (
        _k_act, _k_fill_a, _k_fin,
    )

    T, K = hidden_states.shape
    topk = topk_ids.shape[1]
    E = num_local_experts
    NI = w13_bf.shape[1]
    I = NI // 2
    dev = hidden_states.device

    wp13, sf13, wp2, sf2 = _prep_contig_weights(w13_bf, w2_bf, deep_gemm, E)

    ids = topk_ids.to(torch.long).reshape(-1)
    wsel = topk_weights.reshape(-1).float()
    nrows = ids.numel()

    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order]
    ones = torch.ones_like(sorted_ids, dtype=torch.long)
    masked_m = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
        0, sorted_ids, ones)
    start = torch.cat([masked_m.new_zeros(1, dtype=torch.long),
                       masked_m[:-1].cumsum(0, dtype=torch.long)])
    tok = order // topk
    exp_of_row = order

    # compact contiguous layout: per-expert rows contiguous, pad to M_ALIGN
    m_pad = masked_m.clamp(min=1)  # every expert at least 1 row (masked rows unused)
    m_aligned = torch.where(m_pad > 0,
                            ((m_pad + M_ALIGN - 1) // M_ALIGN) * M_ALIGN,
                            m_pad)
    row_off = torch.cat([m_aligned.new_zeros(1), m_aligned[:-1].cumsum(0)])
    M_total = int(m_aligned.sum().item())
    gl = torch.repeat_interleave(
        torch.arange(E, device=dev), m_aligned).to(torch.int32)

    # gather rows into compact layout
    a_c = torch.zeros(M_total, K, device=dev, dtype=torch.bfloat16)
    slot_c = torch.arange(nrows, device=dev) - start[sorted_ids]
    dst_c = row_off[sorted_ids] + slot_c
    a_c[dst_c] = hidden_states.index_select(0, tok)

    # fp8 quant + SF transform (A side)
    a_fp8, a_sf_raw = deep_gemm.per_token_cast_to_fp8(
        a_c, use_ue8m0=True, gran_k=32)
    a_sf = deep_gemm.transform_sf_into_required_layout(
        a_sf_raw, M_total, K, (1, 1, 32), num_groups=None, is_sfa=True)

    d1_c = torch.empty(M_total, NI, device=dev, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a_fp8, a_sf), (wp13, sf13), d1_c, gl, recipe=(1, 1, 32))

    # act (+delta1) on valid rows only (compact: valid rows are the first m_g per group)
    d1f = d1_c[dst_c]
    delta1 = gate_up_lora_delta.reshape(nrows, NI)[exp_of_row].float()
    lin = d1f.float()[:, 0::2] + delta1[:, I:]
    glu = d1f.float()[:, 1::2] + delta1[:, :I]
    if swiglu_limit is not None:
        lim = float(swiglu_limit)
        glu = torch.minimum(glu, torch.tensor(lim, device=dev))
        lin = torch.clamp(lin, -lim, lim)
    act = glu / (1.0 + torch.exp(-gemm1_alpha * glu)) * (lin + gemm1_beta)
    act_bf = act.to(torch.bfloat16)

    a2_c = torch.zeros(M_total, I, device=dev, dtype=torch.bfloat16)
    a2_c[dst_c] = act_bf
    a2_fp8, a2_sf_raw = deep_gemm.per_token_cast_to_fp8(
        a2_c, use_ue8m0=True, gran_k=32)
    a2_sf = deep_gemm.transform_sf_into_required_layout(
        a2_sf_raw, M_total, I, (1, 1, 32), num_groups=None, is_sfa=True)

    d2_c = torch.empty(M_total, K, device=dev, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a2_fp8, a2_sf), (wp2, sf2), d2_c, gl, recipe=(1, 1, 32))

    # finalize on valid rows
    d2f = d2_c[dst_c].float()
    out = torch.zeros(T, K, device=dev, dtype=torch.float32)
    delta2 = (down_lora_delta.reshape(nrows, K)[exp_of_row].float()
              if down_lora_delta is not None else 0.0)
    contrib = (d2f + delta2) * wsel[exp_of_row, None]
    out.index_add_(0, tok, contrib)

    ali = None
    if activation_lora_input is not None:
        ali_flat = torch.zeros(nrows, I, device=dev, dtype=torch.bfloat16)
        ali_flat[exp_of_row] = act_bf
        activation_lora_input.copy_(ali_flat.view(T, topk, I))
        ali = activation_lora_input
    return out.to(torch.bfloat16), ali
