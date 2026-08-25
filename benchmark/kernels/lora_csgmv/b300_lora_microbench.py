"""Read-only LoRA csgmv config generator for B300 (SM103 / L20D).

Measures the chunked SGMV shrink + expand kernels at GLM-5.2's real
dense-layer dimensions (hidden=6144, rank=16) on one GPU across chunk sizes
16/32/64/128, comparing the hardcoded DEFAULT block sizes against a sweep
that includes SPLIT_K (the Blackwell/SM103 occupancy lever the upstream
tuner omits). Writes the winning config per (kernel, K, S, chunk_size) to
the standard csgmv_configs directory as `device=NVIDIA_L20D.json`, keyed by
chunk_size — exactly the format `lora_tuning_config.get_lora_*_config` loads.

Does NOT touch any running server: allocates only small rank-16 LoRA tensors
and launches Triton kernels directly.

Usage:
    python benchmark/kernels/lora_csgmv/b300_lora_microbench.py [--gpu 6] [--rank 16]
"""

import argparse
import json
import os

import torch
import triton

from sglang.kernels.ops.gemm.chunked_sgmv_expand import _chunked_lora_expand_kernel
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import _chunked_lora_shrink_kernel
from sglang.srt.lora.utils import LoRABatchInfo

HIDDEN = 6144  # GLM-5.2 hidden_size


def make_batch_info(num_tokens, chunk_size, device):
    """Single-adapter batch_info, one segment per chunk (mirrors the upstream
    tuner's build_batch_info)."""
    num_segments = (num_tokens + chunk_size - 1) // chunk_size
    seg_indptr = [0]
    for i in range(num_segments):
        seg_indptr.append(min((i + 1) * chunk_size, num_tokens))
    seg_indptr = torch.tensor(seg_indptr, dtype=torch.int32, device=device)
    weight_indices = torch.ones(num_segments, dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([0, 16], dtype=torch.int32, device=device)
    scalings = torch.ones(2, dtype=torch.float32, device=device)
    permutation = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=num_segments,
        max_len=chunk_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        permutation=permutation,
    )


def timed_ms(fn, warmup=20, trials=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(trials):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / trials


SHRINK_SWEEP = []
for bn in [16, 32, 64]:
    for bk in [64, 128, 256]:
        for sk in [1, 4, 8]:
            SHRINK_SWEEP.append({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk})
DEFAULT_SHRINK = {"BLOCK_N": 16, "BLOCK_K": 256, "SPLIT_K": 1}

EXPAND_SWEEP = []
for bn in [32, 64, 128]:
    for bk in [16, 32]:
        EXPAND_SWEEP.append({"BLOCK_N": bn, "BLOCK_K": bk})
DEFAULT_EXPAND = {"BLOCK_N": 64, "BLOCK_K": 16}


def bench_shrink(K, S, num_tokens, chunk_size, configs, device):
    R = 16
    N = S * R
    x = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16)
    weights = torch.randn(4, N, K, device=device, dtype=torch.bfloat16) * 0.02
    bi = make_batch_info(num_tokens, chunk_size, device)
    out = []
    for cfg in configs:
        output = torch.empty((num_tokens, N), device=device, dtype=torch.bfloat16)
        grid = (triton.cdiv(N, cfg["BLOCK_N"]), bi.num_segments)

        def run(output=output, cfg=cfg):
            _chunked_lora_shrink_kernel[grid](
                x=x, weights=weights, output=output,
                seg_indptr=bi.seg_indptr, weight_indices=bi.weight_indices,
                lora_ranks=bi.lora_ranks, permutation=bi.permutation,
                num_segs=bi.num_segments,
                N=N, K=K, NUM_SLICES=S,
                BLOCK_M=bi.max_len, BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                SPLIT_K=cfg["SPLIT_K"],
            )

        try:
            ms = timed_ms(run)
            out.append((cfg, ms))
        except Exception:
            out.append((cfg, float("inf")))
    return out


def bench_expand(output_dim, S, num_tokens, chunk_size, configs, device):
    R = 16
    x = torch.randn(num_tokens, S * R, device=device, dtype=torch.bfloat16)
    weights = torch.randn(4, output_dim, R, device=device, dtype=torch.bfloat16) * 0.02
    bi = make_batch_info(num_tokens, chunk_size, device)
    slice_offsets = torch.tensor(
        list(range(0, output_dim + 1, output_dim // S)), dtype=torch.int32, device=device
    )
    max_slice = output_dim // S
    out = []
    for cfg in configs:
        grid = (triton.cdiv(max_slice, cfg["BLOCK_N"]), S, bi.num_segments)

        def run(cfg=cfg):
            output = torch.zeros(num_tokens, output_dim, device=device, dtype=torch.bfloat16)
            _chunked_lora_expand_kernel[grid](
                x=x, weights=weights, output=output,
                output_stride_0=output.stride(0), output_stride_1=output.stride(1),
                seg_indptr=bi.seg_indptr, weight_indices=bi.weight_indices,
                lora_ranks=bi.lora_ranks, permutation=bi.permutation,
                num_segs=bi.num_segments, scalings=bi.scalings,
                slice_offsets=slice_offsets,
                NUM_SLICES=S, OUTPUT_DIM=output_dim, MAX_RANK=R,
                BLOCK_M=bi.max_len, BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            )

        try:
            ms = timed_ms(run)
            out.append((cfg, ms))
        except Exception:
            out.append((cfg, float("inf")))
    return out


def save_config(kernel, major_dim, S, per_chunk_best, device_name, triton_version):
    """Write a csgmv config JSON keyed by chunk_size, next to the loader."""
    fname = f"lora_{kernel},K={major_dim},R=16,S={S},device={device_name.replace(' ', '_')}.json"
    config_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "..", "python", "sglang", "kernels", "ops", "gemm",
            "csgmv_configs", f"triton_{triton_version.replace('.', '_')}",
        )
    )
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, fname)
    # Strip SPLIT_K from expand (expand kernel has no SPLIT_K constexpr).
    out = {}
    for chunk, cfg in per_chunk_best.items():
        c = dict(cfg)
        if kernel == "expand":
            c.pop("SPLIT_K", None)
        out[chunk] = c
    with open(path, "w") as f:
        json.dump(out, f, indent=4)
        f.write("\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--total-tokens", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true", help="measure only, write no files")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    import triton as _t
    device_name = "NVIDIA L20D"
    triton_version = _t.__version__
    print(f"LoRA csgmv config generation on {device_name} (GPU {args.gpu}), "
          f"triton {triton_version}, rank {args.rank}, tokens {args.total_tokens}")

    chunk_sizes = [16, 32, 64, 128]
    # (label, kernel, major_dim, S): the four GLM-5.2 dense LoRA layer families.
    layers = [
        ("qkv",      "shrink", HIDDEN,        3),
        ("o_proj",    "shrink", 2 * HIDDEN,    1),
        ("gate_up",  "shrink", HIDDEN,        2),
        ("down_proj", "shrink", 3 * HIDDEN,   1),
        ("qkv",      "expand", 4 * HIDDEN,    3),
        ("o_proj",    "expand", HIDDEN,       1),
        ("gate_up",  "expand", 6 * HIDDEN,    2),
        ("down_proj", "expand", HIDDEN,       1),
    ]

    saved = []
    for label, kernel, major_dim, S in layers:
        per_chunk_best = {}
        default_cfg = DEFAULT_SHRINK if kernel == "shrink" else DEFAULT_EXPAND
        sweep = SHRINK_SWEEP if kernel == "shrink" else EXPAND_SWEEP
        for chunk in chunk_sizes:
            if kernel == "shrink":
                res = bench_shrink(major_dim, S, args.total_tokens, chunk, sweep + [default_cfg], device)
            else:
                res = bench_expand(major_dim, S, args.total_tokens, chunk, sweep + [default_cfg], device)
            best_cfg, best_ms = min(res, key=lambda r: r[1])
            per_chunk_best[chunk] = best_cfg
            base_ms = next(ms for c, ms in res if c == default_cfg)
            print(f"  {kernel} {label} S={S} chunk={chunk:3d}: "
                  f"base={base_ms:.4f}ms best={best_ms:.4f}ms "
                  f"({base_ms/best_ms:.2f}x) {best_cfg}")
        if not args.dry_run:
            path = save_config(kernel, major_dim, S, per_chunk_best, device_name, triton_version)
            saved.append(path)
            print(f"  -> saved {path}")
        print()

    if saved:
        print("Saved config files:")
        for p in saved:
            print(f"  {p}")


if __name__ == "__main__":
    main()
