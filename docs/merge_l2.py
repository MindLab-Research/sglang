#!/usr/bin/env python3
"""Merge L2 LoRA into GLM-5.2 FP8 base model.

base:  FP8 E4M3 + per-block weight_scale_inv (FP8 storage, scale_inv [48,96])
lora:  PEFT LoRA r=16, alpha=32 (scale=2), BF16 A/B
out:   merged model, re-quantized back to FP8 E4M3 + per-block scale_inv

Usage:
  python3 merge_l2.py --validate   # validate dequant/requant on layers.0 only
  python3 merge_l2.py              # full merge (all 141 shards)
"""
import json
import os
import sys
import time
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/root/glm52_local/base"
LORA = "/root/glm52_local/loras/L2"
OUT = "/root/glm52_local/base_l2_merged"

LORA_ALPHA = 32
LORA_R = 16
SCALE = LORA_ALPHA / LORA_R  # 2.0

FP8_MAX = 448.0  # E4M3 max value
BLOCK = 128      # per-block quantization group size


def lora_base_key(k: str) -> str:
    """Convert L2 adapter key to base weight key.
    'base_model.model.model.layers.0.mlp.down_proj.lora_A.weight'
    → 'model.layers.0.mlp.down_proj.weight'
    """
    return k.replace("base_model.model.", "").replace(".lora_A.weight", "")


def split_blocks(dim: int, bsize: int = 128):
    """Block sizes summing to dim: [bsize]*k + [remainder], count = ceil(dim/bsize).
    Matches the quant grid used by the model exporter (e.g. 576 = 4*128 + 64)."""
    nblocks = (dim + bsize - 1) // bsize
    if nblocks == 1:
        return [dim]
    return [bsize] * (nblocks - 1) + [dim - bsize * (nblocks - 1)]


def _block_ids(sizes):
    """Per-element block id: [0]*s0 + [1]*s1 + ..."""
    return torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])


def dequant(w_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 + scale_inv → BF16, per-block (non-uniform grid, 128/remainder).
    Vectorized via block-id index expansion."""
    w = w_fp8.float()
    m, n = w.shape
    sm, sn = scale_inv.shape
    row_sizes = split_blocks(m)
    col_sizes = split_blocks(n)
    if len(row_sizes) != sm or len(col_sizes) != sn:
        raise ValueError(f"grid mismatch: w={w.shape} si={scale_inv.shape} rows={len(row_sizes)} cols={len(col_sizes)}")
    rids = _block_ids(row_sizes)  # [m]
    cids = _block_ids(col_sizes)  # [n]
    si_full = scale_inv[rids][:, cids]  # [m, n]
    return (w / si_full).to(torch.bfloat16)


def requant(w_bf16: torch.Tensor, grid: tuple):
    """BF16 → FP8 E4M3 + scale_inv, preserving original scale grid (sm, sn).
    Vectorized via 128-padding."""
    w = w_bf16.float()
    m, n = w.shape
    sm, sn = grid
    row_sizes = split_blocks(m)
    col_sizes = split_blocks(n)
    assert len(row_sizes) == sm and len(col_sizes) == sn
    m_pad, n_pad = sm * 128, sn * 128
    w_pad = torch.zeros(m_pad, n_pad, dtype=w.dtype)
    w_pad[:m, :n] = w
    w_b = w_pad.reshape(sm, 128, sn, 128)
    scale = w_b.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (w_b / scale).round().clamp(-FP8_MAX, FP8_MAX)
    w_q = q.reshape(m_pad, n_pad)[:m, :n].to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scale).reshape(sm, sn).to(torch.float32)
    return w_q, scale_inv


def main():
    validate_only = "--validate" in sys.argv

    print(f"[merge] LORA={LORA} BASE={BASE} OUT={OUT}")
    print(f"[merge] lora_alpha={LORA_ALPHA} r={LORA_R} scale={SCALE}")

    # 1. Load L2 adapter (BF16 A/B)
    adapter = {}
    with safe_open(f"{LORA}/adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    print(f"[merge] loaded L2 adapter: {len(adapter)} tensors")

    # 2. Build set of base weight names that have LoRA
    lora_weights = {}  # base_key → (A, B)
    for k, v in adapter.items():
        if k.endswith(".lora_A.weight"):
            base_key = lora_base_key(k)
            b_key = k.replace(".lora_A.weight", ".lora_B.weight")
            if b_key in adapter:
                lora_weights[base_key] = (v, adapter[b_key])
    print(f"[merge] LoRA modules: {len(lora_weights)} (e.g. {list(lora_weights)[:3]})")

    if validate_only:
        # Only validate on layers.0
        lora_weights = {k: v for k, v in lora_weights.items() if ".0." in k}
        print(f"[validate] reduced to layers.0: {len(lora_weights)} modules")

    # 3. Load base weight_map
    idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    shard_files = sorted(set(wm.values()))
    print(f"[merge] base shards: {len(shard_files)}")

    # 4. Process shard by shard
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for shard_idx, shard in enumerate(shard_files):
        src = f"{BASE}/{shard}"
        # load all tensors in this shard
        data = {}
        with safe_open(src, framework="pt") as f:
            for k in f.keys():
                data[k] = f.get_tensor(k)

        # merge LoRA weights found in this shard
        for base_key, (A, B) in lora_weights.items():
            w_key = base_key + ".weight"
            si_key = base_key + ".weight_scale_inv"
            if w_key not in data:
                continue
            w_fp8 = data[w_key]
            scale_inv = data.get(si_key)
            if scale_inv is None:
                print(f"[merge] WARN: no scale_inv for {w_key}, skip")
                continue
            # dequant → add LoRA delta → requant
            w_bf16 = dequant(w_fp8, scale_inv)
            delta = (B.float() @ A.float()) * SCALE
            w_merged = (w_bf16.float() + delta).to(torch.bfloat16)
            del w_bf16, delta
            data[w_key], data[si_key] = requant(w_merged, scale_inv.shape)
            del w_merged
            print(f"[merge] {w_key} merged ({data[w_key].shape})")

        # write out shard
        out_path = f"{OUT}/{shard}"
        save_file(data, out_path)
        del data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            el = time.time() - t0
            print(f"[merge] {shard_idx+1}/{len(shard_files)} shards done, elapsed {el:.0f}s")

    # 5. Copy non-weight files
    for fn in ["config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "chat_template.jinja", "LICENSE", "README.md"]:
        src = f"{BASE}/{fn}"
        if os.path.exists(src):
            shutil.copy2(src, f"{OUT}/{fn}")

    # 6. Copy index.json (weight_map unchanged — same shard layout)
    shutil.copy2(f"{BASE}/model.safetensors.index.json", f"{OUT}/model.safetensors.index.json")

    print(f"[merge] DONE. Output: {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
