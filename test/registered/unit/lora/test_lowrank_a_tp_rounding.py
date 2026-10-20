"""GPU regression: sum row-sharded A dot products before low precision storage."""

import unittest

import torch

from sglang.kernels.ops.lora.dense.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.utils import LoRABatchInfo


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LowRankATPPrecisionTest(unittest.TestCase):
    def test_shard_cancellation(self):
        for dtype in (torch.bfloat16, torch.float16):
            for sorted_tokens in (False, True):
                for lengths, slots in (([17], [1]), ([4, 2, 17], [1, 0, 2])):
                    with self.subTest(dtype=dtype, sorted=sorted_tokens, lengths=lengths):
                        self.check_shards(dtype, sorted_tokens, lengths, slots)

    def check_shards(self, dtype, sorted_tokens, lengths, slots):
        total, rank, width = sum(lengths), 16, 257
        boundaries = [0]
        for n in lengths:
            boundaries.append(boundaries[-1] + n)
        permutation = torch.arange(total - 1, -1, -1, device="cuda", dtype=torch.int32) if sorted_tokens else None
        ranks = [0, 16, 8]
        info = LoRABatchInfo(
            use_cuda_graph=False, bs=len(slots), num_segments=len(slots),
            seg_indptr=torch.tensor(boundaries, device="cuda", dtype=torch.int32),
            weight_indices=torch.tensor(slots, device="cuda", dtype=torch.int32),
            lora_ranks=torch.tensor(ranks, device="cuda", dtype=torch.int32),
            scalings=torch.ones(3, device="cuda"), max_len=max(lengths),
            seg_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            permutation=permutation, expected_tokens=total,
        )
        xs = [torch.zeros((total, width), device="cuda", dtype=dtype) for _ in range(2)]
        weights = [torch.zeros((3, rank, width), device="cuda", dtype=dtype) for _ in range(2)]
        xs[0][:, 0], weights[0][:, :, 0] = 0.70703125, 0.70703125
        xs[1][:, -1], weights[1][:, :, -1] = 1.0, -0.5
        precise = [sgemm_lora_a_fwd(x, w, info, output_dtype=torch.float32) for x, w in zip(xs, weights)]
        legacy = [sgemm_lora_a_fwd(x, w, info) for x, w in zip(xs, weights)]
        self.assertTrue(all(x.dtype == torch.float32 for x in precise))
        actual = (precise[0] + precise[1]).to(dtype)
        rounded_partials = legacy[0] + legacy[1]
        full_x = torch.cat(xs, dim=1).float()
        full_w = torch.cat(weights, dim=2).float()
        for segment, slot in enumerate(slots):
            if not ranks[slot]:
                continue
            lo, hi = boundaries[segment:segment + 2]
            rows = permutation[lo:hi].long() if permutation is not None else torch.arange(lo, hi, device="cuda")
            expected = (full_x[rows] @ full_w[slot, :ranks[slot]].T).to(dtype)
            self.assertTrue(torch.all(expected != 0).item())
            self.assertTrue(torch.equal(actual[rows, :ranks[slot]], expected))
            if dtype == torch.bfloat16:
                self.assertTrue(torch.all(rounded_partials[rows, :ranks[slot]] == 0).item())


if __name__ == "__main__":
    unittest.main()
