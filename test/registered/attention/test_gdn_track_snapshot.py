"""GDN prefix caches must preserve the in-kernel FP32 recurrence state."""

import unittest

import torch
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGDNTrackSnapshot(unittest.TestCase):
    @torch.inference_mode()
    def test_fp32_snapshot_matches_independent_prefix_and_resume(self):
        torch.manual_seed(31)
        length, heads, dim = 164, 2, 128
        q, k, v = [
            torch.randn(1, length, heads, dim, device="cuda").bfloat16()
            for _ in range(3)
        ]
        g = -torch.rand(1, length, heads, device="cuda") * 0.15
        beta = torch.randn(1, length, heads, device="cuda").sigmoid().bfloat16()
        slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 100, 164], device="cuda", dtype=torch.int32)
        kernel = TritonGDNKernel()
        self.assertTrue(kernel.supports_track_state_snapshot)
        for nonzero in (False, True):
            with self.subTest(nonzero_initial_state=nonzero):
                initial = torch.randn(3, heads, dim, dim, device="cuda")
                if not nonzero:
                    initial.zero_()
                plain_state, tracked_state = initial.clone(), initial.clone()
                common = {
                    "q": q,
                    "k": k,
                    "v": v,
                    "g": g,
                    "beta": beta,
                    "cache_indices": slots,
                    "query_start_loc": cu,
                }
                plain, _, plain_h = kernel.extend(ssm_states=plain_state, **common)
                snapshot = torch.full((2, heads, dim, dim), float("nan"), device="cuda")
                tracked, _, h = kernel.extend(
                    ssm_states=tracked_state,
                    **common,
                    track_state=snapshot,
                    track_chunk_idx=torch.tensor(
                        [1, -1], device="cuda", dtype=torch.int32
                    ),
                )
                # Snapshotting must not change any existing forward arithmetic.
                torch.testing.assert_close(tracked, plain, atol=0, rtol=0)
                torch.testing.assert_close(tracked_state, plain_state, atol=0, rtol=0)
                torch.testing.assert_close(h, plain_h, atol=0, rtol=0)
                self.assertTrue(torch.isnan(snapshot[1]).all())
                self.assertTrue(torch.isfinite(snapshot[0]).all())
                prefix_state = initial.clone()

                def run(start, end, state):
                    return chunk_gated_delta_rule(
                        *(x[:, start:end].contiguous() for x in (q, k, v, g, beta)),
                        initial_state=state,
                        initial_state_indices=slots[:1],
                        cu_seqlens=torch.tensor(
                            [0, end - start], device="cuda", dtype=torch.int32
                        ),
                        use_qk_l2norm_in_kernel=True,
                    )[0]

                run(0, 64, prefix_state)
                torch.testing.assert_close(snapshot[0], prefix_state[2], atol=0, rtol=0)
                torch.testing.assert_close(
                    h[0, 1], snapshot[0].bfloat16(), atol=0, rtol=0
                )
                self.assertFalse(
                    torch.equal(snapshot[0], snapshot[0].bfloat16().float())
                )
                restored = initial.clone()
                restored[2].copy_(snapshot[0])
                suffix = run(64, 100, restored)
                torch.testing.assert_close(suffix, plain[:, 64:100], atol=0, rtol=0)
                torch.testing.assert_close(restored[2], plain_state[2], atol=0, rtol=0)

    def test_snapshot_arguments_must_be_paired(self):
        q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
        gate = torch.zeros(1, 1, 1, device="cuda")
        with self.assertRaisesRegex(AssertionError, "must be passed together"):
            chunk_gated_delta_rule(
                q, q, q, gate, gate, track_chunk_idx=torch.tensor([0], device="cuda")
            )


if __name__ == "__main__":
    unittest.main()
