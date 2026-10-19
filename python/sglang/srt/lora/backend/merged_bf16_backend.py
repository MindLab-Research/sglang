"""On-demand BF16 weight rounding for dense, eager multi-LoRA inference.

Only the current projection's merged weight is transiently materialized. Base
weights are immutable and adapters remain FP32 low-rank factors in the pool.
This trades extra weight traffic for the rounding semantics of a merged model.
"""

from itertools import pairwise
from math import isfinite

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled


@triton.jit
def _round_merged_weight(delta, base, output, size, scale, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < size
    update = tl.load(delta + index, mask, other=0)
    weight = tl.load(base + index, mask, other=0).to(tl.float32)
    # The launch disables contraction: preserve separate FP32 mul/add rounding.
    value = update * scale + weight
    tl.store(output + index, value, mask)


def merge_bf16_weight(weight, a, b, rank, scaling, offsets):
    """Round FP32(W) + FP32(B@A)*scale once, respecting packed A slices.

    Dense memory-pool A slices are packed at the adapter's active rank;
    B uses the pool rank as its physical row stride. These differ whenever
    an adapter's rank is smaller than the configured maximum.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("merged_bf16 requires IEEE FP32 matmul (TF32 disabled)")
    if weight.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("merged_bf16 requires two-dimensional projection weights")
    if (
        weight.dtype != torch.bfloat16
        or a.dtype != torch.float32
        or b.dtype != torch.float32
    ):
        raise ValueError("merged_bf16 requires BF16 base and FP32 factor buffers")
    if not weight.is_contiguous() or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("merged_bf16 requires contiguous projection weights")
    if not 0 < rank <= b.shape[-1] <= 64:
        raise ValueError("merged_bf16 requires rank <= pool rank <= 64")
    if not isfinite(scaling):
        raise ValueError("merged_bf16 requires finite scaling")
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != b.shape[0]
        or offsets[-1] > weight.shape[0]
        or any(end <= start for start, end in pairwise(offsets))
    ):
        raise ValueError("Invalid packed projection offsets")
    if a.shape != ((len(offsets) - 1) * b.shape[-1], weight.shape[1]):
        raise ValueError("Invalid packed A shape")
    result = (
        torch.empty_like(weight) if offsets[-1] == weight.shape[0] else weight.clone()
    )
    for i, (start, end) in enumerate(pairwise(offsets)):
        # mm.out retains the native cuBLAS implementation when deterministic
        # inference overrides aten::mm. Weight shapes do not depend on token
        # batch size. Match the exporter's contiguous, IEEE FP32 GEMM before
        # the separate scale/add and single BF16 rounding.
        delta = torch.empty(
            (end - start, weight.shape[1]), dtype=torch.float32, device=weight.device
        )
        torch.mm(
            b[start:end, :rank].contiguous(),
            a[i * rank : (i + 1) * rank],
            out=delta,
        )
        if weight.is_cuda:
            _round_merged_weight[(triton.cdiv(delta.numel(), 4096),)](
                delta,
                weight[start:end],
                result[start:end],
                delta.numel(),
                scaling,
                BLOCK=4096,
                enable_fp_fusion=False,
            )
        else:
            delta.mul_(scaling)
            delta.add_(weight[start:end])
            result[start:end].copy_(delta)
        del delta
    return result


class MergedBF16LoRABackend(TritonLoRABackend):
    name = "merged_bf16"
    merges_weights = True
    factor_dtype = torch.float32
    supports_dp_attention = True
    supports_prefill_cuda_graph = False
    skip_inactive_lora_batches = True

    def __init__(self, max_loras_per_batch, device, server_args, **kwargs):
        if (
            server_args.dp_size != 1
            or server_args.pp_size != 1
            or not cuda_graph_fully_disabled()
            or torch.backends.cuda.matmul.allow_tf32
            or not server_args.enable_deterministic_inference
            or server_args.quantization is not None
            or server_args.speculative_algorithm is not None
            or server_args.enable_lora_overlap_loading
            or server_args.max_lora_rank is None
            or not 0 < server_args.max_lora_rank <= 64
        ):
            raise ValueError(
                "merged_bf16 requires DP1/PP1, eager deterministic inference, "
                "unquantized weights, IEEE FP32 matmul, explicit max-lora-rank <=64, "
                "no speculation or overlap loading"
            )
        super().__init__(max_loras_per_batch, device, **kwargs)
        self.merge_groups = []
        self.global_merge_groups = []
        self._merge_ranks = []
        self._merge_scalings = []

    def set_layer_adapter_presence(self, layer, slot, present):
        # Updated by the pool on every slot load, including slot reuse. No
        # device-side reductions or tensor-value inspection in the forward.
        if not hasattr(layer, "merged_bf16_adapter_present"):
            layer.merged_bf16_adapter_present = [True] * self.max_loras_per_batch
        layer.merged_bf16_adapter_present[slot] = present

    def reset_batch_state(self):
        super().reset_batch_state()
        self.merge_groups = []
        self.global_merge_groups = []
        self._merge_ranks = []
        self._merge_scalings = []

    def validate_lora_targets(self, base_model, target_modules):
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        allowed = {
            "gate_up_proj",
            "down_proj",
            "qkv_proj",
            "o_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
        }
        if target_modules - allowed:
            raise ValueError(
                f"merged_bf16 supports only dense targets {sorted(allowed)}"
            )
        for name, module in base_model.named_modules():
            if name.split(".")[-1] not in target_modules:
                continue
            if (
                not isinstance(module, (ColumnParallelLinear, RowParallelLinear))
                or not isinstance(module.quant_method, UnquantizedLinearMethod)
                or module.weight.dtype != torch.bfloat16
            ):
                raise ValueError(f"merged_bf16 unsupported target: {name}")

    def prepare_lora_batch(
        self,
        forward_batch,
        weight_indices,
        lora_ranks,
        scalings,
        use_cuda_graph,
        use_prefill_cuda_graph=False,
    ):
        if use_cuda_graph or use_prefill_cuda_graph:
            raise ValueError("merged_bf16 does not support CUDA graphs")
        super().prepare_lora_batch(
            forward_batch, weight_indices, lora_ranks, scalings, False, False
        )
        lengths = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.forward_mode.is_extend()
            else [1] * forward_batch.batch_size
        )
        if len(weight_indices) != len(lengths):
            raise ValueError("merged_bf16 routing length mismatch")
        self._merge_ranks = list(lora_ranks)
        self._merge_scalings = list(scalings)
        self.global_merge_groups = []
        self.merge_groups = self._make_merge_groups(
            weight_indices, lengths, self.batch_info.expected_tokens
        )

    def _make_merge_groups(self, weight_indices, lengths, expected_tokens):
        groups = {}
        start = 0
        for slot, length in zip(weight_indices, lengths):
            groups.setdefault(slot, []).extend(range(start, start + length))
            start += length
        if start != expected_tokens:
            raise ValueError("merged_bf16 routing does not cover all input tokens")
        return [
            (
                slot,
                self._merge_ranks[slot],
                self._merge_scalings[slot],
                torch.tensor(rows, dtype=torch.long, device=self.device),
            )
            for slot, rows in groups.items()
            if rows
        ]

    def prepare_global_lora_batch(self, forward_batch):
        previous_local = self.batch_info
        super().prepare_global_lora_batch(forward_batch)
        # Global eager routing is token ordered, including padding and tokens
        # from otherwise idle ranks. Copy once per batch, never once per layer.
        if self.batch_info is not previous_local:
            indices = self.batch_info.weight_indices.tolist()
            self.merge_groups = self._make_merge_groups(
                indices, [1] * len(indices), self.batch_info.expected_tokens
            )
        if self.global_batch_info is not None:
            indices = self.global_batch_info.weight_indices.tolist()
            self.global_merge_groups = self._make_merge_groups(
                indices, [1] * len(indices), self.global_batch_info.expected_tokens
            )

    def run_merged_linear(self, layer, x, bias):
        a = getattr(layer, "A_buffer", None)
        b = getattr(layer, "B_buffer", None)
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
        if len(groups) == 1:
            slot, rank, scale, _ = groups[0]
            merged = merge_bf16_weight(weight, a[slot], b[slot], rank, scale, offsets)
            return F.linear(x, merged, bias)
        output = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        for slot, rank, scale, rows in groups:
            merged = (
                merge_bf16_weight(weight, a[slot], b[slot], rank, scale, offsets)
                if rank
                else weight
            )
            values = F.linear(x.index_select(0, rows), merged, bias)
            output.index_copy_(0, rows, values)
            del merged
        return output
