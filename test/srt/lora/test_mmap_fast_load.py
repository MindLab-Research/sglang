# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the LoRA mmap fast-load path.

Runs with plain python3 (no GPU) or pytest:
    python3 test/srt/lora/test_mmap_fast_load.py
    pytest test/srt/lora/test_mmap_fast_load.py -v

Invariants pinned:
1. MmapSafetensorsReader returns tensors BYTE-IDENTICAL to the safetensors
   reference loader for every supported dtype + shape.
2. _stage_expert_slab_a/b produce staging slabs BIT-IDENTICAL to the old
   per-expert loop writing into the same slab layout via slicing.
3. extract_zstd_argv uses multi-threaded zstd (âT0), never tar âzstd.
4. open_adapter rejects multi-shard / missing dirs → falls back gracefully.
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np

torch = None


def _import_torch():
    global torch
    if torch is None:
        import torch as _t
        torch = _t


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))


def test_reader_byte_equivalence(tmpdir=None):
    """Write a synthetic safetensors with every dtype, then read with both paths."""
    _import_torch()
    from safetensors.torch import save_file, safe_open
    from sglang.srt.lora.mmap_weights import MmapSafetensorsReader, list_safetensors

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="lora_mmap_test_")

    # Build adapters dict: multi-dim, odd sizes, every dtype we support
    tensors = {
        "w_bf16": torch.randn(3, 7, 2).to(torch.bfloat16),
        "w_f16": torch.randn(5, 1, 128).half(),
        "w_f32": torch.randn(7, 3),
        "w_f64": torch.randn(4, 2).double(),
        "w_i64": torch.arange(24, dtype=torch.int64).reshape(2, 3, 4),
        "w_i32": torch.arange(18, dtype=torch.int32).reshape(9, 2),
        "w_bool": torch.arange(37).reshape(37, 1) % 2 == 0,
        "w_u8": torch.arange(19, dtype=torch.uint8).reshape(19),
    }
    # per-expert style keys (the real 116k-tensor case has this naming)
    for eid in range(12):
        tensors[f"layers.0.mlp.experts.{eid}.down_proj.lora_A"] = torch.randn(16, 128).to(torch.bfloat16)
    for eid in range(12):
        tensors[f"layers.0.mlp.experts.{eid}.down_proj.lora_B"] = torch.randn(64, 16).to(torch.bfloat16)

    path = os.path.join(tmpdir, "adapter_model.safetensors")
    if os.path.exists(path):
        os.remove(path)
    save_file(tensors, path)

    files = list_safetensors(tmpdir)
    assert len(files) == 1, f"expected 1 file, got {len(files)}"
    reader = MmapSafetensorsReader(files[0])

    n_checked = 0
    for key in reader.keys():
        ref_result = _safe_open_tensor(path, key)
        fast = reader.get(key)
        assert fast.dtype == ref_result.dtype, f"{key}: dtype {fast.dtype} != {ref_result.dtype}"
        assert fast.shape == ref_result.shape, f"{key}: shape {fast.shape} != {ref_result.shape}"
        flat_fast = fast.flatten()
        flat_ref = ref_result.flatten()
        assert flat_fast.equal(flat_ref), f"{key}: values differ (first 3 fast={flat_fast[:3]}, ref={flat_ref[:3]})"
        n_checked += 1

    assert n_checked == len(tensors), f"checked {n_checked}/{len(tensors)}"
    print(f"  PASS  reader byte-equality: {n_checked} tensors all equal")


def test_reader_metadata():
    """Reader discards __metadata__, exposes header size."""
    _import_torch()
    from sglang.srt.lora.mmap_weights import MmapSafetensorsReader

    d = tempfile.mkdtemp(prefix="lora_mmap_meta_")
    from safetensors.torch import save_file
    path = os.path.join(d, "adapter_model.safetensors")
    save_file({"x": torch.zeros(4)}, path, metadata={"epoch": "3"})
    reader = MmapSafetensorsReader(path)
    assert len(reader) == 1
    assert "__metadata__" not in reader.header
    assert reader.payload_bytes > 0
    print("  PASS  reader metadata stripping")


def test_open_adapter_fallbacks():
    """open_adapter returns None for multi-shard and empty dirs."""
    from sglang.srt.lora.mmap_weights import open_adapter

    # Empty dir
    d = tempfile.mkdtemp(prefix="lora_mmap_empty_")
    assert open_adapter(d) is None
    # Two safetensors
    d2 = tempfile.mkdtemp(prefix="lora_mmap_multi_")
    for i in range(2):
        import numpy as _np
        from safetensors.torch import save_file as _sf
        _sf({"x": torch.zeros(1)}, os.path.join(d2, f"shard_{i}.safetensors"))
    r = open_adapter(d2)
    assert r is None, f"expected None for multi-shard, got {r}"
    # No dir
    assert open_adapter("/nonexistent/path") is None
    print("  PASS  open_adapter fallbacks")


def test_stage_expert_slab_a_equivalence():
    """_stage_expert_slab_a fills staging == old per-expert loop semantics."""
    _import_torch()
    from sglang.srt.lora.mem_pool import _stage_expert_slab_a

    E, C, R, max_r, D = 247, 2, 16, 32, 128
    staging = torch.zeros(E, C * max_r, D, dtype=torch.bfloat16)
    reference = torch.zeros(E, C * max_r, D, dtype=torch.bfloat16)

    weights = {eid: torch.randn(C * R, D) for eid in range(E)}
    for eid in range(E):
        for ci in range(C):
            reference[eid, ci * max_r : ci * max_r + R, :].copy_(
                weights[eid][ci * R : (ci + 1) * R, :]
            )
            _stage_expert_slab_a(staging, eid, weights[eid], R, max_r, ci)

    assert staging.equal(reference), "staging != reference for slab A"
    assert staging.equal(reference.clone())
    print(f"  PASS  _stage_expert_slab_a equivalence (E={E}, C={C}, D={D})")


def test_stage_expert_slab_b_equivalence():
    """_stage_expert_slab_b fills staging == old per-expert loop (with scaling)."""
    _import_torch()
    from sglang.srt.lora.mem_pool import _stage_expert_slab_b

    E, R, max_r, D, scaling = 256, 16, 32, 6144, 0.25
    staging = torch.zeros(E, D, max_r, dtype=torch.bfloat16)
    reference = torch.zeros(E, D, max_r, dtype=torch.bfloat16)

    weights = {eid: torch.randn(D, R) for eid in range(E)}
    for eid in range(E):
        view_ref = reference[eid, :, :R]
        if weights[eid] is not None:
            view_ref.copy_(weights[eid] * scaling)
        _stage_expert_slab_b(staging, eid, weights[eid], R, scaling)

    assert staging.equal(reference), "staging != reference for slab B"
    print(f"  PASS  _stage_expert_slab_b equivalence (E={E}, D={D})")


def test_extract_zstd_argv():
    """extract_zstd_argv uses multi-threaded zstd, never tar --zstd."""
    from sglang.srt.lora.mmap_weights import extract_zstd_argv

    argv = extract_zstd_argv("/tmp/a.tar.zst", "/tmp/out")
    assert argv[0] == "tar", f"expected tar, got {argv[0]}"
    joined = " ".join(argv)
    assert "zstd -d -T" in joined, f"expected zstd -d -T, got {joined}"
    assert "--zstd" not in joined.replace("--use-compress-program", "").replace("--use-compress-program=zstd", ""), \
        f"double --zstd collision: {joined}"

    # Env override
    os.environ["SGLANG_LORA_EXTRACT_ZSTD_THREADS"] = "8"
    argv8 = extract_zstd_argv("/tmp/a.tar.zst", "/tmp/out")
    assert "-T8" in " ".join(argv8)
    del os.environ["SGLANG_LORA_EXTRACT_ZSTD_THREADS"]
    print("  PASS  extract_zstd_argv")


def _safe_open_tensor(path, key):
    from safetensors.torch import safe_open
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)


if __name__ == "__main__":
    print("=== LoRA mmap fast-load tests ===")
    test_reader_byte_equivalence()
    test_reader_metadata()
    test_open_adapter_fallbacks()
    test_stage_expert_slab_a_equivalence()
    test_stage_expert_slab_b_equivalence()
    test_extract_zstd_argv()
    print("\n6/6 passed")
