"""Exercise the real row wrapper and Triton kernels with a boundary-owned sum.

Only collective transport is simulated: each rank's input and A are checked
against independent dense dot products before the complete FP32 A is returned.
This catches rounding-before-sum, duplicate B updates, and premature base sums.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.ops.lora.dense.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.kernels.ops.lora.dense.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora import layers
from sglang.srt.lora.utils import LoRABatchInfo


class KernelBackend:
    supports_fp32_lora_a_output = True
    supports_lora_a_overlap = False
    skip_inactive_lora_batches = False

    def __init__(self, info):
        self.info = info
        self.b_calls = 0

    def get_batch_info(self):
        return self.info

    def run_lora_a_sgemm(self, x, weights, **kwargs):
        return sgemm_lora_a_fwd(x, weights, self.info, **kwargs)

    def run_lora_b_sgemm(self, x, weights, base_output, **kwargs):
        self.b_calls += 1
        return sgemm_lora_b_fwd(x, weights, self.info, base_output)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeferredRowPrecisionTest(unittest.TestCase):
    def test_boundary_owned_sum(self):
        for dtype in (torch.bfloat16, torch.float16):
            for sorted_tokens in (False, True):
                for lengths, slots in (([17], [1]), ([4, 2, 17], [1, 0, 2])):
                    for tp_size in (2, 4):
                        for dp_group in (False, True):
                            with self.subTest(dtype=dtype, sorted=sorted_tokens,
                                              lengths=lengths, tp=tp_size, dp=dp_group):
                                self.check_case(dtype, sorted_tokens, lengths, slots,
                                                tp_size, dp_group)

    def test_skip_and_default_guards(self):
        for enabled, skip, skip_mlp in ((False, False, False),
                                        (True, True, False), (True, False, True)):
            with self.subTest(enabled=enabled, skip=skip, skip_mlp=skip_mlp):
                self.check_case(torch.bfloat16, False, [4, 2, 17], [1, 0, 2],
                                2, False, enabled, skip, skip_mlp)

    def check_case(self, dtype, sorted_tokens, lengths, slots, tp_size, dp_group,
                   enabled=True, skip=False, skip_mlp=False):
        total, width, rank, output_size = sum(lengths), 257, 16, 67
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        permutation = torch.arange(total - 1, -1, -1, device="cuda", dtype=torch.int32) if sorted_tokens else None
        ranks, scales = [0, 16, 8], [0., 1., 2.]
        info = LoRABatchInfo(
            use_cuda_graph=False, bs=len(slots), num_segments=len(slots),
            seg_indptr=torch.tensor(boundaries, device="cuda", dtype=torch.int32),
            weight_indices=torch.tensor(slots, device="cuda", dtype=torch.int32),
            lora_ranks=torch.tensor(ranks, device="cuda", dtype=torch.int32),
            scalings=torch.tensor(scales, device="cuda"), max_len=max(lengths),
            seg_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            permutation=permutation, expected_tokens=total,
        )
        xs = [torch.zeros(total, width, device="cuda", dtype=dtype) for _ in range(tp_size)]
        weights = [torch.zeros(3, rank, width, device="cuda", dtype=dtype) for _ in range(tp_size)]
        xs[0][:, 0], weights[0][:, 0, 0] = .70703125, .70703125
        xs[1][:, -1], weights[1][:, 0, -1] = 1., -.5
        b = torch.zeros(3, output_size, rank, device="cuda", dtype=dtype)
        b[:, :, 0] = 1.
        partials = [torch.zeros(total, rank, device="cuda") for _ in xs]
        bases = [torch.zeros(total, output_size, device="cuda", dtype=dtype) for _ in xs]
        expected = torch.zeros_like(bases[0])
        active_rows = []
        for segment, slot in enumerate(slots):
            lo, hi = boundaries[segment:segment + 2]
            rows = permutation[lo:hi].long() if permutation is not None else torch.arange(lo, hi, device="cuda")
            if not ranks[slot]:
                for ordinal, base in enumerate(bases):
                    base[rows] = ordinal + 1
                expected[rows] = sum(range(1, tp_size + 1))
                continue
            active_rows.append(rows)
            for partial, x, a in zip(partials, xs, weights):
                partial[rows, :ranks[slot]] = x[rows].float() @ a[slot, :ranks[slot]].float().T
            full_x = torch.cat(xs, dim=1).float()
            full_a = torch.cat(weights, dim=2).float()
            ax = (full_x[rows] @ full_a[slot, :ranks[slot]].T).to(dtype)
            expected[rows] = (ax.float() @ b[slot, :, :ranks[slot]].float().T * scales[slot]).to(dtype)
        full_a = sum(partials)
        actual, calls, b_calls = [], [], 0
        for ordinal in range(tp_size):
            def reduce(value):
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(value.shape, (total, rank))  # never the base output
                # Packed inactive/padded ranks are intentionally unspecified.
                for segment, slot in enumerate(slots):
                    if not ranks[slot]:
                        continue
                    lo, hi = boundaries[segment:segment + 2]
                    rows = permutation[lo:hi].long() if permutation is not None else torch.arange(lo, hi, device="cuda")
                    self.assertTrue(torch.equal(value[rows, :ranks[slot]],
                                                partials[ordinal][rows, :ranks[slot]]))
                calls.append(ordinal)
                return full_a.clone()

            base = nn.Module()
            base.input_is_parallel, base.reduce_results = True, False
            base.skip_bias_add, base.bias = False, None
            base.tp_group, base.use_dp_attention_reduce = "local-row-group", dp_group
            base.quant_method = SimpleNamespace(apply=lambda *args, **kwargs: bases[ordinal].clone())
            backend = KernelBackend(info)
            with patch.object(layers.envs.SGLANG_LORA_DEFERRED_ROW_FP32_A, "get", return_value=enabled):
                wrapper = layers.RowParallelLinearWithLoRA(base, backend)
            wrapper.set_lora = True
            wrapper.A_buffer, wrapper.B_buffer = weights[ordinal], b
            wrapper.output_offset = torch.tensor([0, output_size], device="cuda", dtype=torch.int32)
            wrapper.output_offset_cpu = torch.tensor([0, output_size], dtype=torch.int32)
            def wrong_group(value):
                self.fail("Global TP reduction used for an attention-TP row layer")
            parallel = SimpleNamespace(attn_tp_group=SimpleNamespace(all_reduce=reduce))
            with patch.object(layers, "get_group_rank_size", return_value=(ordinal, tp_size)), \
                 patch.object(layers, "tensor_model_parallel_all_reduce", side_effect=wrong_group if dp_group else reduce), \
                 patch.object(layers, "get_parallel", return_value=parallel), \
                 patch.object(layers, "should_skip_mlp_all_reduce", return_value=skip_mlp):
                value, bias = wrapper(xs[ordinal], skip_all_reduce=skip)
            self.assertIsNone(bias)
            if ordinal > 0 and enabled and not skip and not skip_mlp:
                self.assertTrue(torch.equal(value, bases[ordinal]))
            actual.append(value)
            b_calls += backend.b_calls
        complete = torch.stack(actual).float().sum(0).to(dtype)
        if enabled and not skip and not skip_mlp:
            self.assertEqual(calls, list(range(tp_size)))
            self.assertEqual(b_calls, 1)
            self.assertTrue(torch.equal(complete, expected))
            self.assertTrue(torch.all(complete[torch.cat(active_rows)] != 0))
        else:
            self.assertFalse(calls)
            self.assertEqual(b_calls, tp_size)
            self.assertTrue(torch.all(complete[torch.cat(active_rows)] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
