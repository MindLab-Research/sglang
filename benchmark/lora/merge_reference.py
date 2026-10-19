"""Export a BF16 reference using FP32 W + alpha/r * B@A, one shard at a time.

This deliberately supports plain LoRA only, rejects unsupported PEFT options,
and records every mapping and rounding error. It never changes the source model.
Qwen3.5/3.8 fused qkvz adapters are split into the checkpoint's qkv/z tensors.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge(base, adapter, output, device):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    for key in (
        "use_dora",
        "use_rslora",
        "fan_in_fan_out",
        "lora_bias",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "layer_replication",
        "target_parameters",
        "alora_invocation_tokens",
        "trainable_token_indices",
    ):
        if cfg.get(key):
            raise ValueError(f"Unsupported PEFT option: {key}={cfg[key]}")
    if cfg.get("bias", "none") != "none":
        raise ValueError("Only bias=none is supported")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shapes = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(base / shard, framework="pt", device="cpu") as f:
            keys = f.keys()
            shapes.update({k: tuple(f.get_slice(k).get_shape()) for k in keys})
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    pairs = {}
    consumed = set()
    for name, a in weights.items():
        if ".lora_A." not in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        b = weights[b_name]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != cfg["r"]
            or b.shape[1] != cfg["r"]
        ):
            raise ValueError(f"Invalid plain LoRA factors: {name}, {b_name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Non-finite LoRA factors: {name}")
        key = name.removeprefix("base_model.model.").replace(
            ".lora_A.weight", ".weight"
        )
        destinations = [(key, b)]
        if key not in shapes and ".in_proj_qkvz." in key:
            qkv_key = key.replace(".in_proj_qkvz.", ".in_proj_qkv.")
            z_key = key.replace(".in_proj_qkvz.", ".in_proj_z.")
            qkv_n, z_n = shapes[qkv_key][0], shapes[z_key][0]
            if b.shape[0] != qkv_n + z_n:
                raise ValueError(f"Invalid fused qkvz output width: {b.shape}")
            destinations = [(qkv_key, b[:qkv_n]), (z_key, b[qkv_n:])]
        for destination, factor_b in destinations:
            if destination not in shapes or shapes[destination] != (
                factor_b.shape[0],
                a.shape[1],
            ):
                raise ValueError(f"Adapter / base shape mismatch: {destination}")
            if destination in pairs:
                raise ValueError(f"Multiple updates to {destination}")
            pairs[destination] = (a, factor_b, name)
        consumed.update((name, b_name))
    if consumed != set(weights) or not pairs:
        raise ValueError(f"Unmapped adapter tensors: {set(weights) - consumed}")

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output.mkdir(parents=True)
    manifest = {
        "formula": "BF16(FP32(W) + (alpha/r) * FP32(B) @ FP32(A))",
        "base": str(base),
        "adapter": str(adapter),
        "config": cfg,
        "base_index_sha256": sha256(base / "model.safetensors.index.json"),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "torch": torch.__version__,
        "device": device,
        "tf32": False,
        "tensors": [],
        "shards": {},
    }
    scale = cfg["lora_alpha"] / cfg["r"]
    for shard in sorted(set(weight_map.values())):
        changed = sorted(k for k in pairs if weight_map[k] == shard)
        src, dst = base / shard, output / shard
        if not changed:
            os.link(src, dst)
        else:
            tensors = load_file(str(src))
            for key in changed:
                a, b, source = pairs[key]
                w = tensors[key].to(device=device, dtype=torch.float32)
                delta = (
                    b.to(device=device, dtype=torch.float32)
                    @ a.to(device=device, dtype=torch.float32)
                ) * scale
                exact = w + delta
                rounded = exact.to(torch.bfloat16)
                error = rounded.float() - exact
                manifest["tensors"].append(
                    {
                        "key": key,
                        "source": source,
                        "shape": list(w.shape),
                        "delta_abs_max": delta.abs().max().item(),
                        "rounding_abs_max": error.abs().max().item(),
                        "rounding_abs_mean": error.abs().mean().item(),
                        "nonzero_delta_rounded_to_base_fraction": (
                            (delta != 0) & (rounded.float() == w)
                        )
                        .float()
                        .mean()
                        .item(),
                    }
                )
                tensors[key] = rounded.cpu().contiguous()
                del w, delta, exact, rounded, error
            save_file(tensors, str(dst), metadata={"format": "pt"})
            del tensors
        manifest["shards"][shard] = {
            "base_sha256": sha256(src),
            "merged_sha256": sha256(dst),
            "changed_tensors": changed,
        }
        print(json.dumps({"shard": shard, "changed": len(changed)}), flush=True)
    for src in base.iterdir():
        if src.is_file() and src.suffix != ".safetensors":
            os.link(src, output / src.name)
    (output / "merge-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "MERGE_COMPLETE").write_text("FP32 merge, BF16 export\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    merge(args.base, args.adapter, args.output, args.device)
