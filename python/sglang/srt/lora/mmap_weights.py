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
"""Zero-copy mmap reader for single-file safetensors LoRA adapters.

Why
---
A GLM-5.2/5.3 MoE LoRA adapter on this cluster is a **single**
``adapter_model.safetensors`` of ~15.27 GB BF16 holding **116,136 tensors**
(57,834 per-expert ``lora_A`` + 57,834 ``lora_B`` pairs). The legacy path

    safe_open(path, framework="pt") -> f.get_tensor(key)   # COPIES each tensor

materialised every tensor into anonymous RAM: 116k reads into 15.27 GB of
fresh allocations per rank (and the same file is read by all 8 TP ranks).
Tensors from an ``np.memmap`` are pure zero-copy views of the page cache —
the freshly extracted file is always fully resident on our 3.6 TB RAM nodes —
so the "deserialize" stage becomes page-fault-only with zero user-space copies.

Format (safetensors v1): ``u64 LE header_size | json header | tensor bytes``.
The header maps name -> {"dtype", "shape", "data_offsets": [begin, end]} with
offsets relative to the START of the data region (8 + header_size).

Correctness
-----------
- Views share the mapped pages: nothing is transformed, so a "load" is
  byte-identical by construction.
- Degenerate layouts (odd byte offsets for 2-byte dtypes) raise in ``_view``
  and transparently fall back to the safetensors reference loader **per
  tensor**, so a pathological writer can never corrupt a load.
- Multi-file / sharded / .bin adapters are rejected and the caller keeps its
  existing ``DefaultModelLoader`` path.

The mapping uses ``np.memmap(mode='c')`` (copy-on-write): reads hit the page
cache, any downstream in-place mutation goes to private pages and can never
write the shared cache file.
"""

import json
import logging
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# safetensors dtype -> (numpy view dtype, torch target dtype).
# numpy has no bfloat16, so BF16/F16 are viewed as raw uint16 payloads and
# reinterpreted on the torch side. uint16 view requires a 2-byte-aligned byte
# offset, which safetensors writers guarantee by packing 2-byte dtypes densely.
_DTYPES: Dict[str, Tuple["np.dtype", torch.dtype]] = {
    "BF16": (np.uint16, torch.bfloat16),
    "F16": (np.uint16, torch.float16),
    "F32": (np.float32, torch.float32),
    "F64": (np.float64, torch.float64),
    "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8),
    "I16": (np.int16, torch.int16),
    "I32": (np.int32, torch.int32),
    "I64": (np.int64, torch.int64),
    "BOOL": (np.uint8, torch.bool),
}


def fast_load_enabled() -> bool:
    """Kill-switch for the mmap load path (reader + grouped H2D)."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_LORA_FAST_LOAD.get())


def list_safetensors(model_path: str) -> List[str]:
    try:
        return sorted(
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        )
    except OSError:
        return []


def extract_zstd_argv(
    archive: str, target: str, threads: Optional[str] = None
) -> List[str]:
    """tar + multi-threaded zstd argv. ``threads``: zstd -T value, 0 = all cores.

    ``tar --zstd`` invokes a single-threaded zstd; a 15.27 GB adapter then
    spends ~20 s decompressing. Multi-threaded zstd on the same file takes
    ~2-4 s. Override via ``SGLANG_LORA_EXTRACT_ZSTD_THREADS`` (e.g. "8").
    """
    if threads is None:
        threads = os.environ.get("SGLANG_LORA_EXTRACT_ZSTD_THREADS", "0")
    return [
        "tar",
        "--use-compress-program",
        f"zstd -d -T{threads}",
        "-xf",
        archive,
        "-C",
        target,
    ]


class MmapSafetensorsReader:
    """Zero-copy tensor views over ONE single .safetensors file."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        file_size = os.path.getsize(self.path)
        with open(self.path, "rb") as fh:
            (hdr_len,) = struct.unpack("<Q", fh.read(8))
            if hdr_len == 0 or hdr_len > file_size - 8:
                raise ValueError(
                    f"invalid safetensors header size {hdr_len} for {self.path} "
                    f"(file size {file_size})"
                )
            header = json.loads(fh.read(hdr_len).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors header in {self.path}")
        self._metadata = header.pop("__metadata__", None)
        self.header = header
        self.data_start = 8 + hdr_len
        self.payload_bytes = sum(
            int(e["data_offsets"][1]) - int(e["data_offsets"][0])
            for e in header.values()
        )
        # Whole-file byte mapping; slicing yields zero-copy views.
        # mode='c' = copy-on-write: reads from page cache, private pages on
        # write — an in-place op downstream can never mutate the cache file.
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="c")

    # -- basics --------------------------------------------------------------
    def keys(self) -> List[str]:
        return list(self.header.keys())

    def __len__(self) -> int:
        return len(self.header)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def get(self, name: str) -> torch.Tensor:
        """Return a zero-copy view of ``name``.

        Falls back to the safetensors reference loader (which copies) when the
        tensor's bytes cannot be viewed in place — misaligned offset, exotic
        dtype, or a header/shape inconsistency.
        """
        entry = self.header[name]
        try:
            return self._view(name, entry)
        except Exception as exc:  # noqa: BLE001 — fall back, never fail a load
            logger.debug("lora mmap view fallback for %s: %s", name, exc)
            return self._reference_load(name)

    def _view(self, name: str, entry: Dict) -> torch.Tensor:
        dtype_name = entry["dtype"]
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported safetensors dtype: {dtype_name}")
        np_dtype, torch_dtype = _DTYPES[dtype_name]
        shape = tuple(int(x) for x in entry["shape"])
        beg, end = (int(x) for x in entry["data_offsets"])

        raw = self._mm[self.data_start + beg : self.data_start + end]
        if raw.dtype != np_dtype:
            view = raw.view(np_dtype)  # raises on odd-offset 2-byte tensors
        else:
            view = raw
        t = torch.from_numpy(view)
        if t.dtype != torch_dtype:
            t = t.view(torch_dtype)
        return t.reshape(shape)

    def _reference_load(self, name: str) -> torch.Tensor:
        """safetensors reference loader — always correct, always a copy."""
        from safetensors.torch import safe_open

        with safe_open(self.path, framework="pt") as f:
            return f.get_tensor(name)


def open_adapter(model_path: str) -> Optional[MmapSafetensorsReader]:
    """Open ``model_path`` for zero-copy reads; ``None`` when unsupported.

    Only a directory with exactly one ``.safetensors`` file takes the fast
    path (the cluster's adapters are single-file). Everything else — shards,
    .bin checkpoints, missing dirs — returns ``None`` and the caller falls
    back to ``DefaultModelLoader`` unchanged.
    """
    files = list_safetensors(model_path)
    if len(files) != 1:
        if len(files) > 1:
            logger.info(
                "lora mmap fast path skipped for %s: %d safetensors shards "
                "(single-file only)",
                model_path,
                len(files),
            )
        return None
    try:
        return MmapSafetensorsReader(files[0])
    except Exception as exc:  # noqa: BLE001 — any parse failure falls back
        logger.warning(
            "lora mmap fast path unavailable for %s (%s); using the standard loader",
            files[0],
            exc,
        )
        return None
