"""GDN scheduler boundaries must preserve the prefill recurrence tiles."""

import unittest
from itertools import pairwise

import torch
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGDNPrefillPartitions(unittest.TestCase):
    def test_aligned_partitions_preserve_outputs_and_fp32_state(self):
        torch.manual_seed(7)
        length, dim, qk_heads, v_heads = 321, 128, 2, 4
        q, k = [
            torch.randn(1, length, qk_heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, length, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, length, v_heads, device="cuda") * 0.15
        beta = torch.randn(1, length, v_heads, device="cuda").sigmoid().bfloat16()
        slots = torch.tensor([0], device="cuda", dtype=torch.int32)

        def run(boundaries, initial_state):
            state = initial_state.clone()
            outputs = []
            for start, end in pairwise(boundaries):
                result = chunk_gated_delta_rule(
                    *(x[:, start:end].contiguous() for x in (q, k, v, g, beta)),
                    initial_state=state,
                    initial_state_indices=slots,
                    cu_seqlens=torch.tensor(
                        [0, end - start], device="cuda", dtype=torch.int32
                    ),
                    use_qk_l2norm_in_kernel=True,
                )[0]
                outputs.append(result)
            return torch.cat(outputs, dim=1), state

        for nonzero_initial_state in (False, True):
            initial = torch.randn(1, v_heads, dim, dim, device="cuda")
            if not nonzero_initial_state:
                initial.zero_()
            expected, expected_state = run([0, length], initial)
            for boundaries in ([0, 64, length], [0, 128, 256, length]):
                with self.subTest(
                    initial_state_nonzero=nonzero_initial_state,
                    boundaries=boundaries,
                ):
                    actual, state = run(boundaries, initial)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
