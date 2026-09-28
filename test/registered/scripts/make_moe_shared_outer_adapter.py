#!/usr/bin/env python3
"""Generate a MoE shared-outer (MoL-style) test adapter for DeepSeek-V4.1-Flash.

Weight layout (see srt/lora/mem_pool.py, experts_shared_outer_loras path):
  gate_up_proj_moe lora_A: (1, 2*r, hidden)   # gate+up stacked on dim1, expert_dim=1
  gate_up_proj_moe lora_B: (1, 2*inter, r)
  down_proj_moe     lora_A: (1, r, inter)
  down_proj_moe     lora_B: (1, hidden, r)

Adapter safetensors names use the HF module path:
  model.layers.{N}.mlp.experts.gate_up_proj.lora_A.weight  (3D, shared)
  model.layers.{N}.mlp.experts.gate_up_proj.lora_B.weight
  model.layers.{N}.mlp.experts.down_proj.lora_A.weight
  model.layers.{N}.mlp.experts.down_proj.lora_B.weight

Run on 1101 (needs the model config + torch).
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

MODEL_PATH = "/mnt/workspace/DeepSeek-V4.1-Flash"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lora_adapters/test_moe_lora"
RANK = 16
ALPHA = 32
SCALE = ALPHA / RANK
SEED = 1234

cfg = json.load(open(os.path.join(MODEL_PATH, "config.json")))
hf = cfg.get("hf_text_config", cfg)
hidden = hf["hidden_size"]
n_layers = hf["num_hidden_layers"]
inter = hf.get("moe_intermediate_size", 1536)
first_k_dense = hf.get("first_k_dense_replace", 1)
n_routed = hf.get("n_routed_experts", 256)

print(f"model: hidden={hidden} layers={n_layers} inter={inter} "
      f"first_dense={first_k_dense} routed_experts={n_routed}")

torch.manual_seed(SEED)
tensors = {}
moe_layers = 0
for layer in range(n_layers):
    if layer < first_k_dense:
        continue  # dense MLP layers have no routed experts
    moe_layers += 1
    p = f"model.layers.{layer}.mlp.experts"
    # LoRA A: (1, 2r, in) ; B: (1, out, r)  — small values for numerical sanity
    tensors[f"{p}.gate_up_proj.lora_A.weight"] = (
        torch.randn(1, 2 * RANK, hidden) * 0.01
    )
    tensors[f"{p}.gate_up_proj.lora_B.weight"] = (
        torch.randn(1, 2 * inter, RANK) * 0.01
    )
    tensors[f"{p}.down_proj.lora_A.weight"] = (
        torch.randn(1, RANK, inter) * 0.01
    )
    tensors[f"{p}.down_proj.lora_B.weight"] = (
        torch.randn(1, hidden, RANK) * 0.01
    )

os.makedirs(OUT_DIR, exist_ok=True)
save_file(tensors, os.path.join(OUT_DIR, "adapter_model.safetensors"))
json.dump(
    {
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": ["gate_up_proj", "down_proj"],
        "peft_type": "LORA",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    open(os.path.join(OUT_DIR, "adapter_config.json"), "w"),
    indent=2,
)
print(f"wrote {len(tensors)} tensors for {moe_layers} MoE layers -> {OUT_DIR}")
for k in list(tensors)[:4]:
    print(f"  {k}: {tuple(tensors[k].shape)}")
