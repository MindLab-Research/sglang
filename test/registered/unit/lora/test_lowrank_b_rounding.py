"""GPU regressions for one rounding after low-rank delta plus base output."""

import unittest

import torch

from sglang.kernels.ops.lora.dense.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.kernels.ops.lora.dense.qkv_lora_b import qkv_lora_b_fwd
from sglang.kernels.ops.lora.dense.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LowRankBRoundingTest(unittest.TestCase):
    def check_kernel(self, kind):
        sizes = {"linear": [67], "gate_up": [67, 67], "qkv": [65, 17, 17]}[kind]
        layouts = [([1], [1], False), ([4, 2, 17], [1, 0, 2], False),
                   ([4, 2, 17], [1, 0, 2], True), ([4, 4, 4], [2, 1, 0], False)]
        for dtype in [torch.bfloat16, torch.float16]:
            for lengths, slots, sorted_tokens in layouts:
                for has_base in [False, True]:
                    with self.subTest(kind=kind, dtype=dtype, lengths=lengths,
                                      sorted_tokens=sorted_tokens, has_base=has_base):
                        self.check_case(kind, sizes, lengths, slots, sorted_tokens, has_base, dtype)

    def check_case(self, kind, sizes, lengths, slots, sorted_tokens, has_base, dtype):
        device, max_rank = "cuda", 16
        count, total = len(slots), sum(lengths)
        ranks = [0, 16, 8]
        scales = [0.0, 1.0, 2.0]
        indptr = [0]
        for length in lengths:
            indptr.append(indptr[-1] + length)
        permutation = torch.arange(total - 1, -1, -1, device=device, dtype=torch.int32) if sorted_tokens else None
        info = LoRABatchInfo(
            use_cuda_graph=False, bs=count, num_segments=count,
            seg_indptr=torch.tensor(indptr, device=device, dtype=torch.int32),
            weight_indices=torch.tensor(slots, device=device, dtype=torch.int32),
            lora_ranks=torch.tensor(ranks, device=device, dtype=torch.int32),
            scalings=torch.tensor(scales, device=device), max_len=max(lengths),
            seg_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            permutation=permutation, expected_tokens=total,
        )
        x = torch.zeros((total, len(sizes) * max_rank), device=device, dtype=dtype)
        weights = torch.zeros((len(ranks), sum(sizes), max_rank), device=device, dtype=dtype)
        weights[:, :, 0] = 0.70703125
        base = torch.full((total, sum(sizes)), -3.0, device=device, dtype=dtype)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        row_groups = []
        for segment, slot in enumerate(slots):
            lo, hi = indptr[segment:segment + 2]
            rows = permutation[lo:hi].long() if permutation is not None else torch.arange(lo, hi, device=device)
            row_groups.append(rows)
            if ranks[slot]:
                # A outputs are packed with the active rank in each token row.
                for part in range(len(sizes)):
                    x[rows, part * ranks[slot]] = 0.70703125
                base[rows] = -0.5 * scales[slot]
        expected = base.clone() if has_base else torch.zeros_like(base)
        for slot, rows in zip(slots, row_groups):
            rank = ranks[slot]
            if not rank:
                continue
            for part, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
                delta = x[rows, part * rank:(part + 1) * rank].float() @ weights[slot, lo:hi, :rank].float().T
                expected[rows, lo:hi] = (expected[rows, lo:hi].float() + delta * scales[slot]).to(dtype)
        supplied = base.clone() if has_base else None
        if kind == "linear":
            actual = sgemm_lora_b_fwd(x, weights, info, supplied)
        elif kind == "gate_up":
            actual = gate_up_lora_b_fwd(x, weights, info, sizes[0], supplied)
        else:
            actual = qkv_lora_b_fwd(x, weights, info,
                                  torch.tensor(offsets, device=device, dtype=torch.int32),
                                  max(sizes), supplied)
        if has_base:
            # BF16(delta) is 0.5, so the old early cast incorrectly produced 0.
            active_rows = torch.cat([rows for slot, rows in zip(slots, row_groups) if ranks[slot]])
            self.assertTrue(torch.all(expected[active_rows] != 0).item())
        self.assertTrue(torch.equal(actual, expected),
                        f"{kind}: max abs error {(actual.float() - expected.float()).abs().max().item()}")
        if has_base:
            self.assertEqual(actual.data_ptr(), supplied.data_ptr())

    def test_linear(self):
        self.check_kernel("linear")

    def test_gate_up(self):
        self.check_kernel("gate_up")

    def test_qkv(self):
        self.check_kernel("qkv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
