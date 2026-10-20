"""GPU controls for tile-local BF16 merge rounding; no model acceptance claim."""

import importlib.util
import json
import os
import sys
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import torch
import torch.nn.functional as F

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/lora/dense/tiled_bf16.py"
)
spec = importlib.util.spec_from_file_location("tested_tiled_bf16", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
tiled_bf16_linear = module.tiled_bf16_linear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTiledBF16Merge(unittest.TestCase):
    results: ClassVar[list] = []

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7701)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_per_process_memory_fraction(0.02)

    @classmethod
    def tearDownClass(cls):
        report = {
            "subcases": cls.results,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "scope": "Single-projection GPU tests, not end-to-end model acceptance",
        }
        if os.environ.get("TEST_REPORT_PATH"):
            with open(os.environ["TEST_REPORT_PATH"], "x") as f:
                json.dump(report, f, indent=2)

    def test_matches_independent_cpu_merge(self):
        for m, n, k, rank, pool in [
            (1, 65, 129, 7, 16),
            (17, 128, 256, 16, 32),
            (33, 257, 512, 32, 64),
            (64, 512, 1024, 16, 16),
        ]:
            for precision in ("ieee", "tf32x3"):
                with self.subTest(shape=(m, n, k, rank, pool), precision=precision):
                    x = (torch.randn(m, k) * 0.25).bfloat16().cuda()
                    w = (torch.randn(n, k) * 0.02).bfloat16().cuda()
                    a = (torch.randn(rank, k) * 0.005).float().cuda()
                    b = (torch.randn(n, pool) * 0.005).float().cuda()
                    bias = (torch.randn(n) * 0.001).bfloat16().cuda()
                    merged = (
                        (w.cpu().float() + (b[:, :rank].cpu() @ a.cpu()) * 2)
                        .bfloat16()
                        .cuda()
                    )
                    # Independent CPU merge followed by FP32 accumulation and
                    # FP32 bias, then one BF16 store: the documented kernel
                    # semantics. Native CUDA BF16 addmm uses another bias
                    # rounding path on some odd N shapes; retain its error
                    # below rather than silently treating it as this oracle.
                    expected = (
                        F.linear(
                            x.cpu().float(), merged.cpu().float(), bias.cpu().float()
                        )
                        .bfloat16()
                        .cuda()
                    )
                    native_gpu = F.linear(x, merged, bias)
                    original_version = [t._version for t in (x, w, a, b, bias)]
                    # Warm compilation before measuring Torch tensor allocations.
                    actual = tiled_bf16_linear(
                        x, w, a, b, rank, 2.0, bias, delta_precision=precision
                    )
                    del actual
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    actual = tiled_bf16_linear(
                        x, w, a, b, rank, 2.0, bias, delta_precision=precision
                    )
                    torch.cuda.synchronize()
                    extra = torch.cuda.max_memory_allocated() - before
                    self.assertLessEqual(
                        extra, actual.numel() * actual.element_size() + 512
                    )
                    self.assertEqual(
                        original_version, [t._version for t in (x, w, a, b, bias)]
                    )
                    error = (actual.float() - expected.float()).abs()
                    self.results.append(
                        {
                            "shape": [m, n, k, rank, pool],
                            "delta_precision": precision,
                            "mean_abs": error.mean().item(),
                            "max_abs": error.max().item(),
                            "bitwise_equal": int((actual == expected).sum()),
                            "elements": actual.numel(),
                            "native_GPU_BF16_bias_mean_abs": (
                                actual.float() - native_gpu.float()
                            )
                            .abs()
                            .mean()
                            .item(),
                            "peak_extra_torch_bytes": extra,
                            "output_bytes": actual.numel() * actual.element_size(),
                            "weight_bytes_not_materialized": w.numel()
                            * w.element_size(),
                        }
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.0001)
                    self.assertLess(error.mean().item(), 1e-6)
                    self.assertGreaterEqual(
                        (actual == expected).float().mean().item(), 0.999
                    )

    def test_matches_triton_deterministic_gemm(self):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            _matmul_persistent_triton,
        )

        for n, k, rank in [(257, 512, 32), (512, 1024, 16)]:
            x = (torch.randn(33, k) * 0.25).bfloat16().cuda()
            w = (torch.randn(n, k) * 0.02).bfloat16().cuda()
            a, b = (
                torch.randn(rank, k, device="cuda") * 0.005,
                torch.randn(n, rank, device="cuda") * 0.005,
            )
            bias = (torch.randn(n, device="cuda") * 0.001).bfloat16()
            merged = (w.cpu().float() + (b.cpu() @ a.cpu()) * 2).bfloat16().cuda()
            expected = _matmul_persistent_triton(x, merged.T, torch.bfloat16, bias)
            actual = tiled_bf16_linear(x, w, a, b, rank, 2.0, bias)
            self.assertTrue(torch.equal(actual, expected))
            self.results.append(
                {
                    "SGLang_triton_deterministic_GEMM": [33, n, k, rank],
                    "bitwise_equal": actual.numel(),
                    "elements": actual.numel(),
                }
            )

    def test_rounding_is_applied_before_activation_gemm(self):
        x = torch.ones(1, 2, dtype=torch.bfloat16, device="cuda")
        w = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16, device="cuda")
        a = torch.full((1, 2), 2**-8, device="cuda")
        b = torch.ones(1, 1, device="cuda")
        merged = (w.cpu().float() + b.cpu() @ a.cpu()).bfloat16().cuda()
        expected = F.linear(x, merged)
        actual = tiled_bf16_linear(x, w, a, b, 1, 1.0)
        additive = (
            F.linear(x, w).float() + F.linear(F.linear(x.float(), a), b)
        ).bfloat16()
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, additive))
        self.results.append(
            {
                "adversarial_rounding_control": True,
                "tile_result": actual.item(),
                "CPU_merged_result": expected.item(),
                "standard_additive_result": additive.item(),
            }
        )

    def test_zero_delta_matches_base(self):
        x = torch.randn(5, 128, device="cuda").bfloat16()
        w = torch.randn(128, 128, device="cuda").bfloat16()
        a, b = torch.zeros(16, 128, device="cuda"), torch.randn(128, 32, device="cuda")
        actual = tiled_bf16_linear(x, w, a, b, 16, 2.0)
        torch.testing.assert_close(actual, F.linear(x, w), rtol=0.01, atol=0.0001)

    def test_packed_slices_and_mixed_slots(self):
        from sglang.srt.batch_invariant_ops import batch_invariant_ops as invariant
        from sglang.srt.lora.backend.tiled_bf16_backend import TiledBF16LoRABackend

        for offsets in ([0, 65], [0, 65, 130], [0, 65, 82, 99]):
            width, count, n = 128, len(offsets) - 1, offsets[-1] + 5
            w = (torch.randint(-16, 16, (n, width), device="cuda") / 1024).bfloat16()
            x = (torch.randint(-8, 8, (19, width), device="cuda") / 16).bfloat16()
            a = (
                torch.randint(-16, 16, (3, count * 16, width), device="cuda").float()
                / 4096
            )
            b = torch.randint(-16, 16, (3, n, 16), device="cuda").float() / 4096
            layer = SimpleNamespace(
                base_layer=SimpleNamespace(weight=w),
                A_buffer=a,
                B_buffer=b,
                output_offset_cpu=torch.tensor(offsets),
                merged_bf16_adapter_present=[False, True, True],
            )
            backend = object.__new__(TiledBF16LoRABackend)
            backend.global_batch_info = None
            backend.merge_groups = [
                (1, 8, 2.0, torch.tensor([0, 4, 7, 12, 18], device="cuda")),
                (0, 0, 0.0, torch.tensor([1, 5, 8, 13], device="cuda")),
                (
                    2,
                    16,
                    0.5,
                    torch.tensor([2, 3, 6, 9, 10, 11, 14, 15, 16, 17], device="cuda"),
                ),
            ]
            expected = torch.empty(19, n, dtype=torch.bfloat16, device="cuda")
            merged_by_slot = {}
            for slot, rank, scale, rows in backend.merge_groups:
                merged = w.cpu().clone()
                if rank:
                    for i, (lo, hi) in enumerate(pairwise(offsets)):
                        delta = (
                            b[slot, lo:hi, :rank].cpu()
                            @ a[slot, i * rank : (i + 1) * rank].cpu()
                        )
                        merged[lo:hi] = (
                            merged[lo:hi].float() + delta * scale
                        ).bfloat16()
                merged_by_slot[slot] = merged.cuda()
                expected[rows] = F.linear(x[rows], merged_by_slot[slot])
            actual = backend.run_merged_linear(layer, x, None)
            native_error = (actual.float() - expected.float()).abs()
            # This eager backend requires deterministic inference. Exercise its
            # actual Triton dispatch, including unmodified trailing projections.
            # Keep the original ordinary-cuBLAS discrepancy as a diagnostic.
            with (
                patch.object(invariant, "_ENABLE_MM_DEEPGEMM", False),
                invariant.set_batch_invariant_mode(),
            ):
                for slot, _, _, rows in backend.merge_groups:
                    expected[rows] = F.linear(x[rows], merged_by_slot[slot])
                actual = backend.run_merged_linear(layer, x, None)
            error = (actual.float() - expected.float()).abs()
            self.results.append(
                {
                    "packed_offsets": list(offsets),
                    "mixed_slots": 3,
                    "original_native_GPU_max_abs": native_error.max().item(),
                    "deterministic_max_abs": error.max().item(),
                    "ranks": [0, 8, 16],
                    "bitwise_equal": int((actual == expected).sum()),
                    "elements": actual.numel(),
                    "mismatch_examples": (actual != expected)
                    .nonzero()
                    .cpu()
                    .tolist()[:8],
                }
            )
            self.assertTrue(torch.equal(actual, expected), self.results[-1])

    def test_rejects_cast_factors(self):
        x = torch.ones(1, 128, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "original FP32"):
            tiled_bf16_linear(x, x, x, x, 1, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
