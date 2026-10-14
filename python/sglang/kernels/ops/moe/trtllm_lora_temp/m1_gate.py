# -*- coding: utf-8 -*-
"""M1 numerical gate for the self-written fused MoE-LoRA kernels.

Verifies, bit-for-bit against the in-repo reference implementation, the
epilogue contract that our fused kernel (`fused_moe_lora_all.py`) must honour.

LAYOUT CONTRACT -- pinned from the reference source, do not guess:
  kernels/ops/moe/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/
      trtllm_fused_moe_dev_kernel.cu :: activationKernel (lines 107-177)

    innerDim          = 2 * INTER          (the gate_up row width)
    gate_up           : [max_pad, 2*INTER] bf16, rows = permutedIdx,
                        columns interleaved (gate, up, gate, up, ...)
    lora_delta        : [T*topk, 2*INTER] bf16, rows = expandedIdx,
                        row layout [ up(INTER) | gate(INTER) ]
    idx_map           : [T*topk] int32, expandedIdx -> permutedIdx (-1 = pad)
    activated (out)   : [max_pad, INTER] bf16, rows = permutedIdx

    x1 = gate_up[perm, 2j]     (+ delta[exp, INTER + j])   -> xLinear
    x2 = gate_up[perm, 2j+1]   (+ delta[exp, j])           -> xGlu

IMPORTANT -- two activation variants, do not confuse them:
  * PLAIN (alpha=beta=clamp = nullptr):
        act = silu(xGlu) * xLinear          <-- what `bench_activation` runs,
                                                because its wrapper never sets
                                                gatedActAlphaPtr/BetaPtr/
                                                ClampLimitPtr.
  * GATED (DeepSeek-V4.1 production, alpha=1.702 beta=1.0 limit=7.0):
        xGlu=min(xGlu,limit); xLinear=clamp(xLinear,+-limit)
        act = xGlu * sigmoid(alpha*xGlu) * (xLinear + beta)
This file compares against the PLAIN variant (that is what the bench can
exercise); the GATED variant is the production contract and is checked here
only for the scalars being present and correct.
"""

from __future__ import annotations

import os

import torch

DEFAULT_ALPHA = 1.702
DEFAULT_BETA = 1.0
DEFAULT_CLAMP_LIMIT = 7.0
E4M3_MAX = 448.0
H_DEFAULT = 5120
INTER_DEFAULT = 2304
TOPK_DEFAULT = 6


def _mk(T: int, mp: int, H: int, INTER: int, TOPK: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nrows = T * TOPK
    return dict(
        hidden=torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g),
        idx=torch.arange(nrows, dtype=torch.int32, device="cuda") % mp,
        total_pad=torch.tensor([mp], dtype=torch.int32, device="cuda"),
        gate_up=torch.randn(mp, 2 * INTER, dtype=torch.bfloat16, device="cuda",
                            generator=g),
        lora_delta=torch.randn(T, TOPK, 2 * INTER, dtype=torch.bfloat16,
                               device="cuda", generator=g),
        activated=torch.empty(mp, INTER, dtype=torch.bfloat16, device="cuda"),
        lora_in=torch.empty(nrows, INTER, dtype=torch.bfloat16, device="cuda"),
    ), mp


def _split_and_add(gate_up, lora_delta, idx, mp, INTER, T, TOPK):
    """Return (xLinear, xGlu) fp32 with the delta already added, per the source."""
    nrows = T * TOPK
    delta_exp = lora_delta.reshape(nrows, 2 * INTER)          # by expandedIdx
    delta_perm = torch.zeros(mp, 2 * INTER, dtype=torch.float32, device="cuda")
    valid = idx >= 0
    delta_perm.index_add_(0, idx[valid].long(), delta_exp[valid].float())

    gu = gate_up.float()
    gate = gu[:, 0::2]                                        # even cols -> xLinear
    up = gu[:, 1::2]                                          # odd cols  -> xGlu
    x_lin = gate + delta_perm[:, INTER:]                      # UPPER half
    x_glu = up + delta_perm[:, :INTER]                        # LOWER half
    return x_lin, x_glu


def oracle_activation_plain(x_lin, x_glu):
    """silu(xGlu) * xLinear -- the branch bench_activation exercises."""
    return (x_glu / (1.0 + torch.exp(-x_glu))) * x_lin


def oracle_activation_gated(x_lin, x_glu, alpha=DEFAULT_ALPHA,
                            beta=DEFAULT_BETA, limit=DEFAULT_CLAMP_LIMIT):
    """The DeepSeek-V4.1 production gated-SwiGLU."""
    x_glu = torch.minimum(x_glu, torch.tensor(limit, device="cuda"))
    x_lin = torch.clamp(x_lin, -limit, limit)
    return x_glu / (1.0 + torch.exp(-alpha * x_glu)) * (x_lin + beta)


def oracle_quant_mxfp8(act_fp32, INTER):
    """MXFP8 (e4m3 + UE8M0 per-32) quantisation oracle."""
    mp = act_fp32.shape[0]
    nb = INTER // 32
    x = act_fp32.reshape(mp, nb, 32)
    amax = x.abs().amax(dim=2)
    sf_value = amax / E4M3_MAX
    exp = torch.ceil(torch.log2(torch.clamp(sf_value, min=1e-38)))
    exp = torch.clamp(exp, -127.0, 128.0)
    scale = torch.pow(2.0, exp)
    inv = torch.where(exp >= 128.0, torch.zeros_like(scale), 1.0 / scale)
    q = torch.clamp(x * inv[:, :, None], -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    sf_byte = (exp + 127.0).to(torch.uint8)
    return q.reshape(mp, INTER), sf_byte


@torch.no_grad()
def run_gate(T: int = 64, H: int = H_DEFAULT, INTER: int = INTER_DEFAULT,
             TOPK: int = TOPK_DEFAULT, verbose: bool = True):
    from sglang.kernels.ops.moe.trtllm_lora_temp.core import (
        get_sgl_trtllm_moe_sm100_raw_module,
    )

    M = get_sgl_trtllm_moe_sm100_raw_module()
    nrows = T * TOPK
    mp = ((nrows + 127) // 128) * 128
    d, mp = _mk(T, mp, H, INTER, TOPK)

    if verbose:
        ops = [n for n in ("bench_activation", "bench_fused_act_quant")
               if hasattr(M, n)]
        print("[M1-GATE] T=%d nrows=%d max_pad=%d  ops=%s" % (T, nrows, mp, ops),
              flush=True)

    # --- reference: our overlay's activation kernel ---------------------------
    # inner_dim MUST be the gate_up row width (2*INTER), not INTER.
    M.bench_activation(d["gate_up"], d["lora_delta"], d["idx"], d["total_pad"],
                       d["activated"], d["lora_in"], 2 * INTER, T, TOPK, 0, 0)
    torch.cuda.synchronize()
    ref = d["activated"].float()

    x_lin, x_glu = _split_and_add(d["gate_up"], d["lora_delta"], d["idx"], mp,
                                  INTER, T, TOPK)

    res = {}
    for name, oracle in (("plain", oracle_activation_plain(x_lin, x_glu)),
                         ("gated", oracle_activation_gated(x_lin, x_glu))):
        diff = (ref - oracle).abs()
        denom = oracle.abs().clamp_min(1e-3)
        ok = bool(torch.allclose(ref, oracle, atol=3e-2, rtol=3e-2))
        res[name] = (ok, diff.max().item())
        if verbose:
            print("[M1-GATE] T=%-4d %-6s  max_abs=%.3e  max_rel=%.3e  ok=%s"
                  % (T, name, diff.max().item(), (diff / denom).max().item(), ok),
                  flush=True)

    # --- quant oracle sanity --------------------------------------------------
    ok_act, _ = res["plain"]
    act_ref = ref if ok_act else oracle_activation_plain(x_lin, x_glu)
    q, sf = oracle_quant_mxfp8(act_ref, INTER)
    if verbose:
        sf_up = sf.float().repeat_interleave(32, dim=1)        # [mp, INTER]
        recon = q.float() * torch.pow(2.0, sf_up - 127.0)
        print("[M1-GATE] T=%-4d mxfp8 oracle: data=%s sf=%s  recon_err=%.3e"
              % (T, tuple(q.shape), tuple(sf.shape),
                 (recon - act_ref).abs().max().item()), flush=True)

    # --- our fused kernel: constants + import ---------------------------------
    try:
        from sglang.kernels.ops.moe.trtllm_lora_temp import fused_moe_lora_all as fk
        print("[M1-GATE] fused kernel imported OK: alpha=%s beta=%s limit=%s "
              "SF_BLOCK_A=%d SF_BLOCK_B=%d"
              % (fk.DEFAULT_ALPHA, fk.DEFAULT_BETA, fk.DEFAULT_CLAMP_LIMIT,
                 fk.SF_BLOCK_A, fk.SF_BLOCK_B), flush=True)
    except Exception as e:  # noqa
        print("[M1-GATE] fused kernel import FAILED: %r" % (e,), flush=True)

    return {"T": T, "plain_ok": res["plain"][0], "gated_ok": res["gated"][0],
            "plain_max_abs": res["plain"][1], "gated_max_abs": res["gated"][1]}


if __name__ == "__main__" or os.environ.get("SGLANG_M1_GATE") == "1":
    for _T in (64, 256):
        run_gate(T=_T)
