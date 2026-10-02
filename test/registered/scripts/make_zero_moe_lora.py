#!/usr/bin/env python3
"""Build a zero-weight MoE shared-outer LoRA adapter from test_moe_lora.

Loading a zero adapter through the LoRA path must reproduce the base model's
output bit-for-bit: it proves the LoRA dispatch (gated SwiGLU activation with
the OAI controls, the decomposed gate_up/down pipeline, the quant steps) agrees
with the non-LoRA trtllm-gen path. Any numerical divergence shows up here
before real LoRA weights can hide it.

Run on 1101.
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/tmp/lora_adapters/test_moe_lora"
DST = "/tmp/lora_adapters/zero_moe_lora"


def main():
    tensors = {}
    with safe_open(os.path.join(SRC, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensors[key] = torch.zeros(
                f.get_slice(key).get_shape(), dtype=torch.bfloat16
            )
    os.makedirs(DST, exist_ok=True)
    save_file(tensors, os.path.join(DST, "adapter_model.safetensors"))
    with open(os.path.join(SRC, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(DST, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"zero adapter: {len(tensors)} tensors -> {DST}")
    for k in list(tensors)[:4]:
        print(f"  {k}: {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
