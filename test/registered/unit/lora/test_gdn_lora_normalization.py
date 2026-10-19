"""Qwen GDN packing preserves independently trained low-rank factors."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import get_normalized_target_modules, get_stacked_multiply
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestGDNPacking(unittest.TestCase):
    def adapter(self, rank):
        adapter = LoRAAdapter.__new__(LoRAAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.uid = "test"
        adapter.config = SimpleNamespace(r=rank)
        adapter.base_hf_config = SimpleNamespace(
            linear_key_head_dim=2,
            linear_num_key_heads=1,
            linear_value_head_dim=4,
            linear_num_value_heads=1,
        )
        return adapter

    def factors(self, modules, widths, rank):
        generator = torch.Generator().manual_seed(42)
        weights = {}
        for module, width in zip(modules, widths):
            prefix = f"model.layers.0.linear_attn.{module}"
            weights[prefix + ".lora_A.weight"] = torch.randn(
                rank, 5, generator=generator, dtype=torch.float64
            )
            weights[prefix + ".lora_B.weight"] = torch.randn(
                width, rank, generator=generator, dtype=torch.float64
            )
        return weights

    def projection(self, weights, module, x):
        prefix = f"model.layers.0.linear_attn.{module}"
        return (
            x
            @ weights[prefix + ".lora_A.weight"].T
            @ weights[prefix + ".lora_B.weight"].T
        )

    def packed_projection(self, weights, module, widths, rank, x):
        prefix = f"model.layers.0.linear_attn.{module}"
        a = weights[prefix + ".lora_A.weight"].split(rank, 0)
        b = weights[prefix + ".lora_B.weight"].split(widths, 0)
        return torch.cat([x @ aa.T @ bb.T for aa, bb in zip(a, b)], dim=-1)

    def test_independent_qkv_and_z_preserve_rank_and_scaling(self):
        x = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 17
        for rank in (8, 16, 32):
            for scale in (0.5, 1.0, 2.0):
                with self.subTest(rank=rank, scale=scale):
                    weights = self.factors(["in_proj_qkv", "in_proj_z"], [8, 4], rank)
                    expected = (
                        torch.cat(
                            [
                                self.projection(weights, name, x)
                                for name in ("in_proj_qkv", "in_proj_z")
                            ],
                            dim=-1,
                        )
                        * scale
                    )
                    self.adapter(rank)._normalize_in_proj_qkvz(weights)
                    actual = (
                        self.packed_projection(
                            weights, "in_proj_qkvz", (2, 2, 4, 4), rank, x
                        )
                        * scale
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assertEqual(len(weights), 2)

    def test_qkv_only_and_z_only_zero_missing_delta(self):
        x = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 17
        for module, width in (("in_proj_qkv", 8), ("in_proj_z", 4)):
            with self.subTest(module=module):
                weights = self.factors([module], [width], 8)
                delta = self.projection(weights, module, x)
                expected = (
                    torch.cat([delta, torch.zeros(3, 4)], -1)
                    if width == 8
                    else torch.cat([torch.zeros(3, 8), delta], -1)
                )
                self.adapter(8)._normalize_in_proj_qkvz(weights)
                actual = self.packed_projection(
                    weights, "in_proj_qkvz", (2, 2, 4, 4), 8, x
                )
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_independent_b_a_and_partial_adapters(self):
        x = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 17
        for rank in (8, 16, 32):
            for modules in (("in_proj_b",), ("in_proj_a",), ("in_proj_b", "in_proj_a")):
                with self.subTest(rank=rank, modules=modules):
                    weights = self.factors(modules, [1] * len(modules), rank)
                    expected = torch.cat(
                        [
                            self.projection(weights, name, x)
                            if name in modules
                            else torch.zeros(3, 1)
                            for name in ("in_proj_b", "in_proj_a")
                        ],
                        dim=-1,
                    )
                    self.adapter(rank)._normalize_in_proj_ba(weights)
                    actual = self.packed_projection(
                        weights, "in_proj_ba", (1, 1), rank, x
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fused_factor_retains_shared_a(self):
        x = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 17
        for module, width, slices in (
            ("in_proj_qkvz", 12, (2, 2, 4, 4)),
            ("in_proj_ba", 2, (1, 1)),
        ):
            weights = self.factors([module], [width], 8)
            expected = self.projection(weights, module, x)
            method = (
                self.adapter(8)._normalize_in_proj_qkvz
                if module.endswith("qkvz")
                else self.adapter(8)._normalize_in_proj_ba
            )
            method(weights)
            torch.testing.assert_close(
                self.packed_projection(weights, module, slices, 8, x),
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_overlapping_or_mixed_fused_split_factors_are_rejected(self):
        for modules, widths in [
            (("in_proj_qkv", "in_proj_q"), (8, 2)),
            (("in_proj_qkvz", "in_proj_z"), (12, 4)),
        ]:
            with self.subTest(modules=modules):
                weights = self.factors(modules, widths, 8)
                with self.assertRaisesRegex(
                    ValueError, "overlapping|mixed fused/split"
                ):
                    self.adapter(8)._normalize_in_proj_qkvz(weights)

    def test_target_names_and_ba_dimensions(self):
        self.assertEqual(
            get_normalized_target_modules(
                ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            ),
            {"in_proj_qkvz", "in_proj_ba"},
        )
        self.assertEqual(get_stacked_multiply("in_proj_ba"), 2)
        model = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=5,
                head_dim=1,
                num_attention_heads=1,
                linear_num_value_heads=3,
            )
        )
        self.assertEqual(
            Qwen3_5ForCausalLM.get_hidden_dim(model, "in_proj_ba", 0), (5, 6)
        )
        self.assertIn("in_proj_ba", Qwen3_5ForCausalLM.supported_lora_modules)


if __name__ == "__main__":
    unittest.main()
