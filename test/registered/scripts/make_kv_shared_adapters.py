#!/usr/bin/env python3
"""Generate test adapters for kv_shared LoRA (DSV4.1-Flash, 40 layers).

Produces three adapters in /tmp/lora_adapters:
  kv_shared_a / kv_shared_b — q_proj-only, layers 20-39 only (last 20 layers),
      adapter_config.json declares "lora_kv_shared": true. Distinct random
      weights so their OUTPUTS differ (proving the LoRA actually applies) while
      their KV (base-computed) is identical.
  normal_isolated           — q_proj, all layers, NO kv_shared flag. Isolation
      control: same prompt must NOT hit the shared prefix.
  bad_kv_vproj               — declares kv_shared but targets v_proj: must be
      REJECTED at load time (fail-safe validation).

Run on 1101 (needs the model config + torch + safetensors).
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

MODEL_PATH = "/mnt/workspace/DeepSeek-V4.1-Flash"
OUT_BASE = "/tmp/lora_adapters"
RANK = 16
ALPHA = 32
FIRST_KV_LAYER = 20  # last 20 layers of 40


def hidden_size() -> int:
    cfg = json.load(open(os.path.join(MODEL_PATH, "config.json")))
    return cfg["text_config"]["hidden_size"]


def make_adapter(name, layers, target_modules, kv_shared, seed):
    hidden = hidden_size()
    torch.manual_seed(seed)
    tensors = {}
    for layer in layers:
        p = f"model.layers.{layer}.self_attn.{target_modules[0]}"
        tensors[f"{p}.lora_A.weight"] = torch.randn(RANK, hidden) * 0.01
        tensors[f"{p}.lora_B.weight"] = torch.randn(hidden, RANK) * 0.01
    out = os.path.join(OUT_BASE, name)
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "adapter_model.safetensors"))
    cfg = {
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": list(target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    if kv_shared:
        cfg["lora_kv_shared"] = True
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    print(f"{name}: {len(tensors)} tensors, layers={layers[0]}..{layers[-1]}, "
          f"targets={target_modules}, kv_shared={kv_shared}")


def main():
    all_layers = list(range(40))
    last20 = list(range(FIRST_KV_LAYER, 40))

    make_adapter("kv_shared_a", last20, ["q_proj"], True, seed=101)
    make_adapter("kv_shared_b", last20, ["q_proj"], True, seed=202)
    make_adapter("normal_isolated", all_layers, ["q_proj"], False, seed=303)
    make_adapter("bad_kv_vproj", last20, ["v_proj"], True, seed=404)


if __name__ == "__main__":
    main()
