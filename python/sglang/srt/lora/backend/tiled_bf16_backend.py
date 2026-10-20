"""Experimental merge-rounding semantics with tile-local weight reconstruction."""

from itertools import pairwise

import torch
import torch.nn.functional as F
from sglang.kernels.ops.lora.dense.tiled_bf16 import (
    grouped_tiled_bf16_linear,
    tiled_bf16_linear,
)
from sglang.srt.lora.backend.merged_bf16_backend import MergedBF16LoRABackend


class TiledBF16LoRABackend(MergedBF16LoRABackend):
    """Reuse eager routing/FP32 factors; never materialize/cache merged weights.

    This backend deliberately matches weight-rounding semantics rather than
    standard output-additive LoRA. It is opt-in, BF16-only, and inherits the
    deterministic eager/DP1/PP1 restrictions of the merge accuracy control.
    """

    name = "tiled_bf16"

    def __init__(self, max_loras_per_batch, device, server_args, **kwargs):
        if getattr(server_args, "lora_merged_bf16_cache_size_mb", 0):
            raise ValueError("tiled_bf16 has no merged-weight cache")
        super().__init__(max_loras_per_batch, device, server_args, **kwargs)
        self._tiled_route_cache = {}

    def prepare_lora_batch(self, *args, **kwargs):
        super().prepare_lora_batch(*args, **kwargs)
        self._tiled_route_cache = {}

    def reset_batch_state(self):
        super().reset_batch_state()
        self._tiled_route_cache = {}

    def set_layer_adapter_presence(self, layer, slot, present):
        super().set_layer_adapter_presence(layer, slot, present)
        layer.tiled_bf16_adapter_present = torch.tensor(
            layer.merged_bf16_adapter_present, dtype=torch.int32, device=self.device
        )

    def _tile_routing(self, routing):
        cache = getattr(self, "_tiled_route_cache", None)
        if cache is None:
            cache = self._tiled_route_cache = {}
        if id(routing) not in cache:
            tiles, scales, offset = [], [], 0
            for slot, rank, scale, rows in routing:
                for start in range(0, rows.numel(), 16):
                    tiles.append(
                        [slot, rank, offset + start, min(16, rows.numel() - start)]
                    )
                    scales.append(scale)
                offset += rows.numel()
            device = routing[0][3].device
            cache[id(routing)] = (
                # Retain the list so that its id cannot be reused this batch.
                routing,
                torch.cat([rows for _, _, _, rows in routing]),
                torch.tensor(tiles, dtype=torch.int32, device=device),
                torch.tensor(scales, dtype=torch.float32, device=device),
            )
        return cache[id(routing)][1:]

    def run_merged_linear(self, layer, x, bias):
        a, b = getattr(layer, "A_buffer", None), getattr(layer, "B_buffer", None)
        if a is None:
            a, b = layer.A_buffer_qkv, layer.B_buffer_qkv
        weight = layer.base_layer.weight
        present = getattr(layer, "merged_bf16_adapter_present", None)
        routing = self.merge_groups
        global_info = getattr(self, "global_batch_info", None)
        if global_info is not None and self.get_batch_info() is global_info:
            routing = self.global_merge_groups
        groups = [
            (slot, rank if present is None or present[slot] else 0, scale, rows)
            for slot, rank, scale, rows in routing
        ]
        if not any(rank for _, rank, _, _ in groups):
            return F.linear(x, weight, bias)
        offsets = layer.output_offset_cpu.tolist()
        output = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        # Grouping reduces launch overhead for decode/small batches. For larger
        # groups the scalar schedule reuses weight tiles better (B300 profile).
        if len(groups) > 1 and max(rows.numel() for _, _, _, rows in groups) <= 32:
            rows, tiles, scales = self._tile_routing(routing)
            presence = getattr(layer, "tiled_bf16_adapter_present", None)
            if presence is None and present is not None:
                # Direct unit callers can bypass the memory-pool load hook.
                presence = layer.tiled_bf16_adapter_present = torch.tensor(
                    present, dtype=torch.int32, device=x.device
                )
            for i, (start, end) in enumerate(pairwise(offsets)):
                grouped_tiled_bf16_linear(
                    x.contiguous(),
                    weight[start:end],
                    a,
                    b,
                    output,
                    rows,
                    tiles,
                    scales,
                    i,
                    start,
                    None if bias is None else bias[start:end],
                    presence,
                )
            if offsets[-1] < weight.shape[0]:
                output[:, offsets[-1] :] = F.linear(
                    x,
                    weight[offsets[-1] :],
                    None if bias is None else bias[offsets[-1] :],
                )
            return output
        for slot, rank, scale, rows in groups:
            xx = x if len(groups) == 1 else x.index_select(0, rows)
            if rank:
                values = torch.empty(
                    (xx.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
                )
                for i, (start, end) in enumerate(pairwise(offsets)):
                    # A slices are packed at active rank, B retains pool stride.
                    values[:, start:end] = tiled_bf16_linear(
                        xx.contiguous(),
                        weight[start:end],
                        a[slot, i * rank : (i + 1) * rank],
                        b[slot, start:end],
                        rank,
                        scale,
                        None if bias is None else bias[start:end],
                    )
                if offsets[-1] < weight.shape[0]:
                    values[:, offsets[-1] :] = F.linear(
                        xx,
                        weight[offsets[-1] :],
                        None if bias is None else bias[offsets[-1] :],
                    )
            else:
                values = F.linear(xx, weight, bias)
            if len(groups) == 1:
                output.copy_(values)
            else:
                output.index_copy_(0, rows, values)
        return output
