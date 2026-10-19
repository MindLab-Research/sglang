"""Independent merged-weight and mixed-adapter reference checks."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.lora.backend.merged_bf16_backend import (
    MergedBF16LoRABackend,
    merge_bf16_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


class TestMergedBF16Configuration(unittest.TestCase):
    def test_cli_accepts_packed_and_checkpoint_gdn_targets(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        targets = [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_qkvz",
            "in_proj_b",
            "in_proj_a",
            "in_proj_ba",
            "out_proj",
        ]
        args = parser.parse_args(
            [
                "--model-path",
                "/unused-model",
                "--lora-backend",
                "merged_bf16",
                "--lora-target-modules",
                *targets,
            ]
        )
        self.assertEqual(args.lora_target_modules, targets)
        self.assertEqual(args.lora_backend, "merged_bf16")

    def args(self, **overrides):
        values = {
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            # The raw args do not contain the resolved graph configuration.
            "cuda_graph_config": None,
            "enable_deterministic_inference": True,
            "quantization": None,
            "speculative_algorithm": None,
            "enable_lora_overlap_loading": False,
            "max_lora_rank": 32,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_uses_resolved_graph_configuration(self):
        with patch(
            "sglang.srt.lora.backend.merged_bf16_backend.cuda_graph_fully_disabled",
            return_value=True,
        ):
            backend = MergedBF16LoRABackend(4, torch.device("cpu"), self.args())
            backend.merge_groups = ["stale"]
            backend.reset_batch_state()
            self.assertEqual(backend.merge_groups, [])
            self.assertIsNone(backend.batch_info)

    def test_rejects_unsupported_execution_modes(self):
        cases = [
            {"tp_size": 2},
            {"dp_size": 2},
            {"pp_size": 2},
            {"enable_deterministic_inference": False},
            {"quantization": "fp8"},
            {"speculative_algorithm": "EAGLE"},
            {"enable_lora_overlap_loading": True},
            {"max_lora_rank": None},
            {"max_lora_rank": 0},
            {"max_lora_rank": 65},
        ]
        for overrides in cases:
            with (
                self.subTest(overrides=overrides),
                patch(
                    "sglang.srt.lora.backend.merged_bf16_backend.cuda_graph_fully_disabled",
                    return_value=True,
                ),
                self.assertRaisesRegex(ValueError, "requires TP1"),
            ):
                MergedBF16LoRABackend(4, torch.device("cpu"), self.args(**overrides))
        with (
            patch(
                "sglang.srt.lora.backend.merged_bf16_backend.cuda_graph_fully_disabled",
                return_value=False,
            ),
            self.assertRaisesRegex(ValueError, "eager deterministic"),
        ):
            MergedBF16LoRABackend(4, torch.device("cpu"), self.args())

    def test_rejects_unsupported_targets(self):
        backend = object.__new__(MergedBF16LoRABackend)
        for target in ("lm_head", "embed_tokens", "in_proj_qz", "experts"):
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(ValueError, "only dense targets"),
            ):
                backend.validate_lora_targets(torch.nn.Module(), {target})

    def test_validates_dense_gdn_targets_and_base_precision(self):
        from sglang.srt.layers.linear import ColumnParallelLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        model = torch.nn.Module()
        targets = {"in_proj_qkvz", "in_proj_ba", "out_proj"}
        for target in targets:
            layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
            torch.nn.Module.__init__(layer)
            layer.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
            layer.quant_method = UnquantizedLinearMethod()
            model.add_module(target, layer)
        backend = object.__new__(MergedBF16LoRABackend)
        backend.validate_lora_targets(model, targets)
        model.in_proj_qkvz.weight = torch.nn.Parameter(torch.zeros(4, 4))
        with self.assertRaisesRegex(ValueError, "unsupported target: in_proj_qkvz"):
            backend.validate_lora_targets(model, targets)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMergedBF16(unittest.TestCase):
    def test_native_merge_matches_export_under_batch_invariant_mode(self):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            disable_batch_invariant_mode,
            enable_batch_invariant_mode,
            is_batch_invariant_mode_enabled,
        )

        torch.manual_seed(13)
        w = (torch.randn(73, 131, device="cuda") * 0.02).bfloat16()
        a = (torch.randn(16, 131, device="cuda") * 0.01).bfloat16().float()
        b = (torch.randn(73, 16, device="cuda") * 0.01).bfloat16().float()
        was_enabled = is_batch_invariant_mode_enabled()
        disable_batch_invariant_mode()
        expected = (w.float() + (b @ a) * 1.7).bfloat16()
        enable_batch_invariant_mode()
        try:
            actual = merge_bf16_weight(w, a, b, 16, 1.7, [0, 73])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            if not was_enabled:
                disable_batch_invariant_mode()

    def test_packed_factors_padding_scaling_and_unadapted_suffix(self):
        for rank in (8, 16, 32, 64):
            for scale in (2.0, 1.7):
                with self.subTest(rank=rank, scale=scale):
                    torch.manual_seed(17)
                    w = (torch.randn(71, 131) * 0.02).bfloat16()
                    pool_rank = max(32, rank)
                    a = torch.randn(2 * pool_rank, 131) * 0.01
                    b = torch.randn(53, pool_rank) * 0.01
                    expected = w.clone()
                    for i, (start, end) in enumerate([(0, 19), (19, 53)]):
                        expected[start:end] = (
                            w[start:end].float()
                            + (b[start:end, :rank] @ a[i * rank : (i + 1) * rank])
                            * scale
                        ).bfloat16()
                    actual = merge_bf16_weight(
                        w.cuda(), a.cuda(), b.cuda(), rank, scale, [0, 19, 53]
                    )
                    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_mixed_base_zero_and_real_preserve_order_and_base_weights(self):
        torch.manual_seed(23)
        w = (torch.randn(64, 128) * 0.02).bfloat16()
        a = torch.randn(3, 16, 128) * 0.01
        b = torch.randn(3, 64, 16) * 0.01
        b[2].zero_()
        x = torch.randn(5, 128).bfloat16()
        merged = (w.float() + (b[1, :, :8] @ a[1, :8]) * 1.7).bfloat16()
        expected = torch.stack(
            [
                torch.nn.functional.linear(x[i], merged if i in (1, 4) else w)
                for i in range(5)
            ]
        )
        backend = object.__new__(MergedBF16LoRABackend)
        backend.merge_groups = [
            (0, 0, 0, torch.tensor([0, 3], device="cuda")),
            (1, 8, 1.7, torch.tensor([1, 4], device="cuda")),
            (2, 8, 1.7, torch.tensor([2], device="cuda")),
        ]
        layer = SimpleNamespace(
            base_layer=SimpleNamespace(weight=w.cuda()),
            A_buffer=a.cuda(),
            B_buffer=b.cuda(),
            output_offset_cpu=torch.tensor([0, 64]),
        )
        actual = backend.run_merged_linear(layer, x.cuda(), None)
        # CPU and GPU BF16 GEMMs need not use the same accumulation tree.
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(layer.base_layer.weight.cpu(), w, atol=0, rtol=0)

    def test_invalid_factor_precision_and_rank_rejected(self):
        w = torch.zeros(32, 64, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(16, 64, device="cuda")
        b = torch.zeros(32, 16, device="cuda")
        with self.assertRaisesRegex(ValueError, "FP32 factor"):
            merge_bf16_weight(w, a.bfloat16(), b, 8, 2.0, [0, 32])
        with self.assertRaisesRegex(ValueError, "rank"):
            merge_bf16_weight(w, a, b, 17, 2.0, [0, 32])

    def test_rejects_malformed_offsets_and_nonfinite_scaling(self):
        w = torch.zeros(32, 64, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(16, 64, device="cuda")
        b = torch.zeros(32, 16, device="cuda")
        for offsets in ([], [0], [1, 32], [0, 32, 32], [0, 33]):
            with (
                self.subTest(offsets=offsets),
                self.assertRaisesRegex(ValueError, "offsets"),
            ):
                merge_bf16_weight(w, a, b, 8, 2.0, offsets)
        for scale in (float("inf"), float("nan")):
            with (
                self.subTest(scale=scale),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                merge_bf16_weight(w, a, b, 8, scale, [0, 32])


if __name__ == "__main__":
    unittest.main()
