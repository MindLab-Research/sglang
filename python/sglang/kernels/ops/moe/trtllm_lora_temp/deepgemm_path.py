# -*- coding: utf-8 -*-
"""bf16-bmm MoE-LoRA path v3.1 (expanded-order, fused glue kernels).

v3 numerics (rel=3.6e-3) + three fused triton kernels replacing the torch
glue op swarm (~15 ops -> 3 kernels):
  K_fill:   a[g, slot] = x[tok]                     (gather+scatter write)
  K_act:    d1f + delta1 -> swiglu -> a2 write      (act epilogue fusion)
  K_fin:    (d2f + delta2) * wsel -> atomic scatter (finalize fusion)
GEMM1/GEMM2 remain cuBLAS bf16 bmm (masked DeepGEMM discarded: wrong content).
delta computed per-request by the caller (red line kept).
"""
from typing import Optional

import torch
import triton
import triton.language as tl


def repack_weights_for_deepgemm(w13: torch.Tensor, w2: torch.Tensor,
                                deep_gemm) -> tuple[torch.Tensor, torch.Tensor]:
    """v3: both GEMMs use bf16 bmm — dequantize w13 fp4->bf16 via deep_gemm cast
    (per-expert loop), return (w13_bf16 [E,2I,K], w2_bf16 [E,K,I])."""
    E, NI, K = w13.shape
    w13_bf = torch.empty(E, NI, K, device=w13.device, dtype=torch.bfloat16)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=w13.device, dtype=torch.float32)
    Kh = K // 2
    for e in range(E):
        p, s = deep_gemm.per_token_cast_to_fp4(w13[e], use_ue8m0=True, gran_k=32)
        lo = (p & 0x0F).to(torch.int64)
        hi = ((p >> 4) & 0x0F).to(torch.int64)
        v = torch.empty(NI, K, device=w13.device, dtype=torch.float32)
        v[:, 0::2] = lut[lo]
        v[:, 1::2] = lut[hi]
        sfe = s.repeat_interleave(32, dim=-1)[..., :K]
        w13_bf[e] = (v * sfe).to(torch.bfloat16)
    return w13_bf, w2


@triton.jit
def _k_fill_a(x_ptr, a_ptr, sorted_ids_ptr, slot_ptr, tok_ptr,
              K: tl.constexpr, M_MAX: tl.constexpr, BLOCK: tl.constexpr):
    """a[g*M+slot, :] = x[tok] — one program per expanded row."""
    i = tl.program_id(0)
    g = tl.load(sorted_ids_ptr + i)
    sl = tl.load(slot_ptr + i)
    t = tl.load(tok_ptr + i)
    dst = a_ptr + (g * M_MAX + sl) * K
    src = x_ptr + t * K
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        m = offs < K
        v = tl.load(src + offs, mask=m)
        tl.store(dst + offs, v, mask=m)


@triton.jit
def _k_act(d1_ptr, gd1_ptr, a2_ptr, valid_pos_ptr, exp_ptr,
           NI: tl.constexpr, I: tl.constexpr, LIM: tl.constexpr,
           ALPHA: tl.constexpr, BETA: tl.constexpr, M_MAX: tl.constexpr,
           BLOCK_I: tl.constexpr):
    """a2[valid_pos] = swiglu(d1[valid_pos] + delta1[exp]) — one program per row."""
    i = tl.program_id(0)
    vp = tl.load(valid_pos_ptr + i)
    ei = tl.load(exp_ptr + i)
    offs = tl.arange(0, BLOCK_I)
    m = offs < I
    d1r = d1_ptr + vp * NI
    gd = gd1_ptr + ei * NI
    lin = tl.load(d1r + 2 * offs, mask=m, other=0.0).to(tl.float32) \
        + tl.load(gd + I + offs, mask=m, other=0.0).to(tl.float32)
    glu = tl.load(d1r + 2 * offs + 1, mask=m, other=0.0).to(tl.float32) \
        + tl.load(gd + offs, mask=m, other=0.0).to(tl.float32)
    glu = tl.minimum(glu, LIM)
    lin = tl.clamp(lin, -LIM, LIM)
    act = glu / (1.0 + tl.exp(-ALPHA * glu)) * (lin + BETA)
    tl.store(a2_ptr + vp * I + offs, act.to(tl.bfloat16), mask=m)


@triton.jit
def _k_fin(d2_ptr, gd2_ptr, out_ptr, valid_pos_ptr, exp_ptr, w_ptr, tok_ptr,
           K: tl.constexpr, M_MAX: tl.constexpr, BLOCK: tl.constexpr):
    """out[tok] += (d2[valid_pos] + delta2[exp]) * w — atomic scatter."""
    i = tl.program_id(0)
    vp = tl.load(valid_pos_ptr + i)
    ei = tl.load(exp_ptr + i)
    t = tl.load(tok_ptr + i)
    w = tl.load(w_ptr + ei)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    d2r = d2_ptr + vp * K
    g2 = gd2_ptr + ei * K
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        m2 = offs < K
        dv = tl.load(d2r + offs, mask=m2, other=0.0).to(tl.float32) \
            + tl.load(g2 + offs, mask=m2, other=0.0).to(tl.float32)
        tl.atomic_add(out_ptr + t * K + offs, dv * w, mask=m2)


@torch.no_grad()
def deepgemm_fp4_moe_lora(
    hidden_states: torch.Tensor,          # [T, K] bf16
    topk_ids: torch.Tensor,               # [T, topk] int
    topk_weights: torch.Tensor,           # [T, topk] float32 (routed-scaled)
    w13_bf: torch.Tensor,                 # [E, 2I, K] bf16 (dequantized)
    w2_bf: torch.Tensor,                  # [E, K, I] bf16 (dequantized)
    gate_up_lora_delta: torch.Tensor,     # [T, topk, 2I] bf16
    down_lora_delta: torch.Tensor,        # [T, topk, K] bf16
    swiglu_limit: Optional[float],        # V4.1 production: 7.0
    deep_gemm=None,
    num_local_experts: int = 96,
    M_MAX: int = 64,
) -> torch.Tensor:
    out, _ = v3_gemm_stage(
        hidden_states, topk_ids, topk_weights, w13_bf, w2_bf,
        gate_up_lora_delta, down_lora_delta, swiglu_limit, None,
        num_local_experts, M_MAX)
    return out


@torch.no_grad()
def v3_gemm_stage(
    hidden_states: torch.Tensor,          # [T, K] bf16
    topk_ids: torch.Tensor,               # [T, topk] int (local expert ids)
    topk_weights: torch.Tensor,           # [T, topk] float32
    w13_bf: torch.Tensor,                 # [E, 2I, K] bf16
    w2_bf: torch.Tensor,                  # [E, K, I] bf16
    gate_up_lora_delta: torch.Tensor,     # [T, topk, 2I_padded] bf16
    down_lora_delta: Optional[torch.Tensor],  # [T, topk, K] bf16 or None
    swiglu_limit: Optional[float],
    activation_lora_input: Optional[torch.Tensor],  # [T, topk, I] out or None
    num_local_experts: int,
    M_MAX: int = 64,
    gemm1_alpha: float = 1.702,
    gemm1_beta: float = 1.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    T, K = hidden_states.shape
    topk = topk_ids.shape[1]
    E = num_local_experts
    NI = w13_bf.shape[1]
    I = NI // 2
    dev = hidden_states.device

    ids = topk_ids.to(torch.long).reshape(-1)
    wsel = topk_weights.reshape(-1).float()
    nrows = ids.numel()

    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order]
    # capture-safe bincount (bincount itself does a CPU sync — illegal in graphs)
    ones = torch.ones_like(sorted_ids, dtype=torch.long)
    masked_m = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
        0, sorted_ids, ones)
    start = torch.cat([masked_m.new_zeros(1, dtype=torch.long),
                       masked_m[:-1].cumsum(0, dtype=torch.long)])
    slot = (torch.arange(nrows, device=dev) - start[sorted_ids]).clamp(max=M_MAX - 1)
    valid_pos = sorted_ids * M_MAX + slot
    exp_of_row = order
    tok = order // topk

    a = torch.zeros(E, M_MAX, K, device=dev, dtype=torch.bfloat16)
    _k_fill_a[(nrows,)](hidden_states, a, sorted_ids, slot, tok,
                        K=K, M_MAX=M_MAX, BLOCK=1024)

    d1 = torch.bmm(a, w13_bf.transpose(1, 2))                    # [E, M_MAX, 2I]
    d1f = d1.view(E * M_MAX, NI)

    BLOCK_I = triton.next_power_of_2(I)
    a2_flat = torch.empty(E * M_MAX, I, device=dev, dtype=torch.bfloat16)
    _k_act[(nrows,)](d1f, gate_up_lora_delta, a2_flat, valid_pos, exp_of_row,
                     NI=NI, I=I, LIM=float(swiglu_limit or 7.0),
                     ALPHA=gemm1_alpha, BETA=gemm1_beta, M_MAX=M_MAX,
                     BLOCK_I=BLOCK_I)

    d2 = torch.bmm(a2_flat.view(E, M_MAX, I), w2_bf.transpose(1, 2))
    d2f = d2.view(E * M_MAX, K)

    out = torch.zeros(T, K, device=dev, dtype=torch.float32)
    BLOCK_K = triton.next_power_of_2(min(1024, K))
    _k_fin[(nrows,)](d2f, down_lora_delta.reshape(nrows, K), out,
                     valid_pos, exp_of_row, wsel, tok,
                     K=K, M_MAX=M_MAX, BLOCK=BLOCK_K)

    ali = None
    if activation_lora_input is not None:
        ali_flat = torch.zeros(nrows, I, device=dev, dtype=torch.bfloat16)
        ali_flat[exp_of_row] = a2_flat[valid_pos]
        activation_lora_input.copy_(ali_flat.view(T, topk, I))
        ali = activation_lora_input
    return out.to(torch.bfloat16), ali
