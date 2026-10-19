"""Deterministic GEMMs must honor the server's TF32 precision setting."""

import unittest

import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    addmm_batch_invariant,
    bmm_batch_invariant,
    mm_batch_invariant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMatmulPrecision(unittest.TestCase):
    def setUp(self):
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(17)

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32

    def test_fp32_accuracy_and_batch_invariance(self):
        for m, n, k in [(20, 512, 5120), (128, 512, 5120), (16, 512, 16)]:
            a = torch.randn(m, k, device="cuda")
            b = torch.randn(n, k, device="cuda").T
            bias = torch.randn(n, device="cuda")
            expected = (a.double() @ b.double()).float()
            for name, actual, reference in [
                ("mm", mm_batch_invariant(a, b), expected),
                ("addmm", addmm_batch_invariant(bias, a, b), expected + bias),
                (
                    "bmm",
                    bmm_batch_invariant(a[None], b[None])[0],
                    expected,
                ),
            ]:
                with self.subTest(shape=(m, n, k), operation=name):
                    # Sequential FP32 accumulation over K=5120 has a larger
                    # absolute error near cancellation than cuBLAS reductions.
                    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=1e-5)
                    relative_l2 = (actual - reference).norm() / reference.norm()
                    self.assertLess(relative_l2.item(), 2e-6)
            torch.testing.assert_close(
                mm_batch_invariant(a[:1], b),
                mm_batch_invariant(a, b)[:1],
                atol=0,
                rtol=0,
            )

    def test_tf32_remains_an_explicit_opt_in(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("TF32 requires compute capability >= 8")
        a = torch.randn(20, 5120, device="cuda")
        b = torch.randn(5120, 512, device="cuda")
        for operation, args in [
            (mm_batch_invariant, (a, b)),
            (bmm_batch_invariant, (a[None], b[None])),
        ]:
            with self.subTest(operation=operation.__name__):
                torch.backends.cuda.matmul.allow_tf32 = False
                precise = operation(*args)
                torch.backends.cuda.matmul.allow_tf32 = True
                fast = operation(*args)
                self.assertGreater((fast - precise).abs().max().item(), 0.01)

    def test_bf16_and_fp16_are_independent_of_tf32_flag(self):
        for dtype in (torch.bfloat16, torch.float16):
            a = torch.randn(17, 128, dtype=dtype, device="cuda")
            b = torch.randn(128, 64, dtype=dtype, device="cuda")
            for operation, args in [
                (mm_batch_invariant, (a, b)),
                (bmm_batch_invariant, (a[None], b[None])),
            ]:
                with self.subTest(dtype=dtype, operation=operation.__name__):
                    torch.backends.cuda.matmul.allow_tf32 = False
                    precise = operation(*args)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    fast = operation(*args)
                    torch.testing.assert_close(precise, fast, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
