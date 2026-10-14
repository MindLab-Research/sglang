# -*- coding: utf-8 -*-
"""Autotune the CSGMV LoRA shrink/expand block sizes for THIS device + shapes.

Why: the service log shows every LoRA dense kernel running the *untuned*
defaults (shrink BLOCK_N=16/BLOCK_K=256; expand BLOCK_N=64/BLOCK_K=16), because
`csgmv_configs/` ships only H200+R64 files (we are NVIDIA_L20D, triton 3.7.1,
R=16).  The measured overhead split puts those chunked-sgmv kernels at ~21% of
the total LoRA penalty.

Writes JSON tables the loader picks up automatically:
    kernels/ops/gemm/csgmv_configs/triton_<ver>/lora_<kernel>,K=..,R=..,S=..,device=<dev>.json

Diagnostic (single config, full traceback):
    TUNE_DIAG=1 python kernels/ops/gemm/tune_csgmv_l20d.py
Full sweep:
    python kernels/ops/gemm/tune_csgmv_l20d.py
"""

import json
import os
import time
import traceback
from types import SimpleNamespace

import torch
import triton

import sglang.kernels.ops.gemm.chunked_sgmv_shrink as sh
import sglang.kernels.ops.gemm.chunked_sgmv_expand as ex

DIAG = os.environ.get("TUNE_DIAG") == "1"

R = 32                       # == --max-lora-rank. The adapter weight dim is PADDED to
                             # this value, so the lookup key is R=32 (NOT the adapter's
                             # true rank). Verified by instrumenting the loader:
                             #   [sdiag] kernel=shrink K=5120 R=32 S=1 ...
NUM_SEGS = 4                 # segments per batch (small at decode)
CHUNKS = [32] if DIAG else [16, 32]
BS = [32] if DIAG else [16, 32, 64]
KS = [64] if DIAG else [64, 128, 256]
WARPS = [4] if DIAG else [4, 8]
STAGES = [3]

# EXACT (K, S) shapes the runtime asks for -- captured from the [sdiag] log lines.
SHRINK = [(5120, 3)] if DIAG else [(2048, 1), (5120, 1), (5120, 2), (576, 1)]
EXPAND = [(5120, 1)] if DIAG else [(1152, 2), (32320, 1), (5120, 1)]


def _bi(chunk):
    s = NUM_SEGS * chunk
    return SimpleNamespace(
        use_cuda_graph=False,
        bs=NUM_SEGS,
        num_segments=NUM_SEGS,
        seg_indptr=torch.arange(
            0, (NUM_SEGS + 1) * chunk, chunk, dtype=torch.int32, device="cuda"
        ),
        weight_indices=torch.zeros(NUM_SEGS, dtype=torch.int32, device="cuda"),
        lora_ranks=torch.full((1,), R, dtype=torch.int32, device="cuda"),
        scalings=torch.ones(1, device="cuda"),
        max_len=chunk,
        seg_lens=torch.full((NUM_SEGS,), chunk, dtype=torch.int32, device="cuda"),
        # MUST be a real tensor: the kernel unconditionally does
        # `tl.load(permutation + s_offset_logical)` (chunked_sgmv_shrink.py:71).
        # Identity is correct for benchmarking.
        permutation=torch.arange(s, dtype=torch.int32, device="cuda"),
    )


def _time(fn, iters=60, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def tune_shrink(K, S, chunk, cfg):
    s = NUM_SEGS * chunk
    x = torch.randn(s, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1, S * R, K, dtype=torch.bfloat16, device="cuda")
    bi = _bi(chunk)
    # NOTE: patch the MODULE-LOCAL name binding (chunked_sgmv_shrink does
    # `from ... import get_lora_shrink_config`), not the source module's attr.
    sh.get_lora_shrink_config = lambda **kw: dict(cfg)
    return _time(lambda: sh.chunked_sgmv_lora_shrink_forward(x, w, bi, num_slices=S))


def tune_expand(K, S, chunk, cfg):
    s = NUM_SEGS * chunk
    x = torch.randn(s, S * R, dtype=torch.bfloat16, device="cuda")
    out_dim = K
    w = torch.randn(1, out_dim, R, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([i * (out_dim // S) for i in range(S + 1)],
                        dtype=torch.int32, device="cuda")
    bi = _bi(chunk)
    ex.get_lora_expand_config = lambda **kw: dict(cfg)
    return _time(
        lambda: ex.chunked_sgmv_lora_expand_forward(
            x, w, bi, offs, out_dim // S, None
        )
    )


def main():
    dev = "NVIDIA_L20D"
    ver = f"triton_{triton.__version__.replace('.', '_')}"
    root = os.path.join(os.path.dirname(os.path.realpath(sh.__file__)),
                        "csgmv_configs", ver)
    os.makedirs(root, exist_ok=True)
    print(f"[tune] DIAG={DIAG} device={dev} dir={root} triton={triton.__version__}",
          flush=True)

    for kernel, shapes, fn in (("shrink", SHRINK, tune_shrink),
                               ("expand", EXPAND, tune_expand)):
        for (K, S) in shapes:
            table, ok_all = {}, True
            for chunk in CHUNKS:
                best, best_cfg = None, None
                for bn in BS:
                    for bk in KS:
                        for w in WARPS:
                            for st in STAGES:
                                cfg = {"BLOCK_N": bn, "BLOCK_K": bk,
                                       "num_warps": w, "num_stages": st}
                                try:
                                    us = fn(K, S, chunk, cfg)
                                except Exception as e:  # noqa
                                    if DIAG:
                                        print(f"  FAIL {cfg} (full traceback):",
                                              flush=True)
                                        traceback.print_exc()
                                        return
                                    print(f"  SKIP {cfg}: {type(e).__name__}: {e}",
                                          flush=True)
                                    continue
                                if best is None or us < best:
                                    best, best_cfg = us, dict(cfg)
                if best_cfg is None:
                    print(f"[tune] NO VALID CONFIG {kernel} K={K} S={S} "
                          f"chunk={chunk}", flush=True)
                    ok_all = False
                    continue
                table[str(chunk)] = best_cfg
                print(f"[tune] {kernel} K={K} S={S} chunk={chunk}: "
                      f"{best:.2f}us  {best_cfg}", flush=True)
            if not ok_all:
                print(f"[tune] {kernel} K={K} S={S}: incomplete, NOT writing",
                      flush=True)
                continue
            name = f"lora_{kernel},K={K},R={R},S={S},device={dev}.json"
            with open(os.path.join(root, name), "w") as f:
                json.dump(table, f, indent=4)
            print(f"[tune] WROTE {name}: {table}", flush=True)
    print("[tune] DONE", flush=True)


if __name__ == "__main__":
    main()
