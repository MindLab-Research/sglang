"""Projection-cache admission must preserve exact weights and its reserved budget."""

import argparse
import io
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.lora.backend.merged_bf16_backend import (
    MergedBF16LoRABackend,
    merge_bf16_weight,
)
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestProjectionCacheAdmission(unittest.TestCase):
    def backend(self, policy="active", limit=96):
        b = object.__new__(MergedBF16LoRABackend)
        b._weight_cache = OrderedDict()
        b._weight_cache_limit = limit
        b._weight_cache_bytes = 0
        b._cache_policy = policy
        b._cache_transient_reserve = 32
        b._cache_active_slots = {0, 1}
        b._cache_stats = dict.fromkeys(
            [
                "hits",
                "misses",
                "hit_bytes",
                "merged_bytes",
                "admissions",
                "bypasses",
                "evictions",
                "invalidations",
                "peak_retained_bytes",
            ],
            0,
        )
        b.max_loras_per_batch = 3
        b.device = torch.device("cpu")
        return b

    def weights(self):
        layers = [SimpleNamespace() for _ in range(4)]
        weights = [torch.full((4, 4), 0.1 * i).bfloat16() for i in range(4)]
        a = torch.arange(24).reshape(3, 2, 4).float() * 0.01
        b = torch.arange(24).reshape(3, 4, 2).float() * 0.02
        return layers, weights, a, b

    def test_cli_keeps_lru_default_and_accepts_explicit_active_policy(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        minimal = ["--model-path", "/unused-model"]
        self.assertEqual(
            parser.parse_args(minimal).lora_merged_bf16_cache_policy, "lru"
        )
        selected = parser.parse_args(
            minimal + ["--lora-merged-bf16-cache-policy", "active"]
        )
        self.assertEqual(selected.lora_merged_bf16_cache_policy, "active")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(minimal + ["--lora-merged-bf16-cache-policy", "unknown"])

    def test_largest_local_target_is_reserved_before_first_forward(self):
        from sglang.srt.layers.linear import ColumnParallelLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        backend = self.backend(limit=128)
        model = torch.nn.Module()
        for target, rows in [("qkv_proj", 4), ("gate_up_proj", 12)]:
            layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
            torch.nn.Module.__init__(layer)
            layer.weight = torch.nn.Parameter(
                torch.zeros(rows, 4, dtype=torch.bfloat16)
            )
            layer.quant_method = UnquantizedLinearMethod()
            model.add_module(target, layer)
        backend.validate_lora_targets(model, {"qkv_proj", "gate_up_proj"})
        self.assertEqual(backend._cache_transient_reserve, 96)
        layers, weights, a, b = self.weights()
        for slot in (0, 1):
            backend._merged_weight(layers[0], weights[0], a, b, slot, 1, 0.5, [0, 4])
        self.assertEqual(backend._weight_cache_bytes, 32)
        self.assertEqual(backend._cache_stats["bypasses"], 1)

    def test_cyclic_working_set_reuses_active_weights_instead_of_thrashing(self):
        layers, weights, a, b = self.weights()
        for policy in ("lru", "active"):
            backend = self.backend(policy)

            def checked_merge(weight, x, y, rank, scale, offsets, selected=backend):
                self.assertLessEqual(
                    selected._weight_cache_bytes + weight.numel() * 2,
                    selected._weight_cache_limit,
                )
                return merge_bf16_weight(weight, x, y, rank, scale, offsets)

            path = "sglang.srt.lora.backend.merged_bf16_backend.merge_bf16_weight"
            with patch(path, side_effect=checked_merge) as merge:
                for _ in range(5):
                    for layer, weight in zip(layers, weights):
                        for slot in (0, 1):
                            actual = backend._merged_weight(
                                layer, weight, a, b, slot, 1, 0.5, [0, 4]
                            )
                            independent = (
                                weight.float() + (b[slot, :, :1] @ a[slot, :1]) * 0.5
                            ).bfloat16()
                            torch.testing.assert_close(
                                actual, independent, atol=0, rtol=0
                            )
                self.assertEqual(merge.call_count, 40 if policy == "lru" else 32)
                self.assertEqual(
                    backend._cache_stats["hits"], 0 if policy == "lru" else 8
                )
                if policy == "active":
                    self.assertEqual(backend._cache_stats["bypasses"], 30)
                    self.assertLessEqual(backend._weight_cache_bytes + 32, 96)

    def test_departed_adapters_make_room_for_new_active_adapter(self):
        backend = self.backend()
        layers, weights, a, b = self.weights()
        for slot in (0, 1):
            backend._merged_weight(layers[0], weights[0], a, b, slot, 1, 0.5, [0, 4])
        backend._cache_active_slots = {2}
        result = backend._merged_weight(layers[0], weights[0], a, b, 2, 1, 0.5, [0, 4])
        self.assertEqual({key[1] for key in backend._weight_cache}, {2})
        self.assertIs(
            result,
            backend._merged_weight(layers[0], weights[0], a, b, 2, 1, 0.5, [0, 4]),
        )
        self.assertEqual(backend._cache_stats["evictions"], 2)

    def test_slot_reuse_invalidates_cached_projection(self):
        backend = self.backend()
        layers, weights, a, b = self.weights()
        first = backend._merged_weight(layers[0], weights[0], a, b, 1, 1, 0.5, [0, 4])
        backend.set_layer_adapter_presence(layers[0], 1, True)
        a[1].mul_(5)
        second = backend._merged_weight(layers[0], weights[0], a, b, 1, 1, 0.5, [0, 4])
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(backend._cache_stats["invalidations"], 1)

    def test_oversized_projection_cannot_coexist_with_retained_entries(self):
        backend = self.backend()
        layers, weights, a, b = self.weights()
        backend._merged_weight(layers[0], weights[0], a, b, 0, 1, 0.5, [0, 4])
        big = torch.zeros(4, 20).bfloat16()
        result = backend._merged_weight(
            layers[1], big, torch.ones(3, 2, 20), b, 1, 1, 0.5, [0, 4]
        )
        self.assertEqual(result.shape, big.shape)
        self.assertFalse(backend._weight_cache)
        self.assertEqual(backend._weight_cache_bytes, 0)
        self.assertEqual(backend._cache_stats["bypasses"], 1)

    def test_global_dp_adapters_stay_active_on_local_rank(self):
        backend = self.backend()
        backend.batch_info = SimpleNamespace()
        backend.global_batch_info = SimpleNamespace(
            weight_indices=torch.tensor([0, 2]), expected_tokens=2
        )
        backend.merge_groups = [(1, 16, 0.5, torch.tensor([0]))]
        backend.global_merge_groups = []
        backend._merge_ranks = [0, 16, 8]
        backend._merge_scalings = [0, 0.5, 2.0]
        with patch.object(TritonLoRABackend, "prepare_global_lora_batch"):
            backend.prepare_global_lora_batch(SimpleNamespace())
        self.assertEqual(backend._cache_active_slots, {1, 2})
        self.assertEqual([x[0] for x in backend.merge_groups], [1])

    def test_other_layer_presence_does_not_evict_active_attention_adapter(self):
        backend = self.backend()
        layers, weights, a, b = self.weights()
        backend._merged_weight(layers[0], weights[0], a, b, 1, 1, 0.5, [0, 4])
        layer = SimpleNamespace(
            base_layer=SimpleNamespace(weight=weights[1]),
            A_buffer=a,
            B_buffer=b,
            merged_bf16_adapter_present=[True, False, False],
            output_offset_cpu=torch.tensor([0, 4]),
        )
        backend.merge_groups = [
            (0, 1, 0.5, torch.tensor([0])),
            (1, 1, 0.5, torch.tensor([1])),
        ]
        output = backend.run_merged_linear(layer, torch.ones(2, 4).bfloat16(), None)
        self.assertEqual(output.shape, (2, 4))
        self.assertIn(
            (id(layers[0]), 1, 1, 0.5, weights[0].data_ptr(), weights[0]._version),
            backend._weight_cache,
        )


if __name__ == "__main__":
    unittest.main()
