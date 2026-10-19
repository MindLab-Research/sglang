"""Malformed adapters must be rejected before changing resident weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.lora import lora_manager as manager_module
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class IdentitySlices:
    def slice_lora_a_weights(self, weight):
        return weight

    def slice_lora_b_weights(self, weight):
        return weight


class TestLoRALoadValidation(unittest.TestCase):
    def make_pool_and_adapter(self):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.target_modules = {"out_proj", "embed_tokens"}
        pool.num_layer = 2
        pool.tp_rank = 0
        pool.tp_size = 1
        pool.max_lora_rank = 2
        pool.base_model = None
        pool.strict_loading = True
        pool.experts_shared_outer_loras = False
        pool.lora_added_tokens_size = 0
        pool.base_hf_config = SimpleNamespace(vocab_size=3)
        pool.A_buffer = {"out_proj": [torch.full((2, 2, 3), 11.0) for _ in range(2)]}
        pool.B_buffer = {"out_proj": [torch.full((2, 4, 2), 13.0) for _ in range(2)]}
        pool.embedding_A_buffer = {"embed_tokens": torch.full((2, 2, 3), 17.0)}
        pool.embedding_B_buffer = {"embed_tokens": torch.full((2, 4, 2), 19.0)}
        pool.lm_head_A_buffer = {}
        pool.lm_head_B_buffer = {}
        pool.new_embeddings_buffer = {}
        pool.uid_to_buffer_id = {"resident": 0}
        pool.buffer_id_to_uid = ["resident", "other"]
        pool._get_maybe_cached_weight_for_transfer = Mock(
            side_effect=AssertionError("preflight must not pin or cache weights")
        )
        layers = []
        for i in range(2):
            prefix = f"model.layers.{i}.out_proj"
            layers.append(
                SimpleNamespace(
                    weights={
                        prefix + ".lora_A.weight": torch.ones(2, 3),
                        prefix + ".lora_B.weight": torch.ones(4, 2),
                    },
                    pinned_weights={},
                )
            )
        adapter = SimpleNamespace(
            uid="candidate",
            layers=layers,
            config=SimpleNamespace(r=2, lora_added_tokens_size=0),
            scaling=1,
            embedding_layers={},
            pinned_embedding_layers={},
            pinned_added_tokens_embeddings={},
            added_tokens_embeddings={},
        )
        modules = [{"out_proj": IdentitySlices()} for _ in range(2)]
        return pool, adapter, modules

    def check_preflight(self, mutate=None, error=None):
        pool, adapter, modules = self.make_pool_and_adapter()
        if mutate:
            mutate(pool, adapter)
        buffers = [
            *pool.A_buffer["out_proj"],
            *pool.B_buffer["out_proj"],
            *pool.embedding_A_buffer.values(),
            *pool.embedding_B_buffer.values(),
        ]
        before = [x.clone() for x in buffers]
        with patch("sglang.srt.lora.mem_pool.copy_weight_into_buffer") as copy:
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    pool.validate_lora_weights(
                        "candidate", adapter, modules, None, None
                    )
            else:
                pool.validate_lora_weights("candidate", adapter, modules, None, None)
            copy.assert_not_called()
        for old, current in zip(before, buffers):
            self.assertTrue(torch.equal(old, current))
        self.assertEqual(pool.uid_to_buffer_id, {"resident": 0})
        self.assertEqual(pool.buffer_id_to_uid, ["resident", "other"])
        self.assertTrue(all(not layer.pinned_weights for layer in adapter.layers))
        pool._get_maybe_cached_weight_for_transfer.assert_not_called()

    def test_valid_preflight_is_read_only(self):
        self.check_preflight()

    def test_late_bad_shape_leaves_all_buffers_unchanged(self):
        def mutate(pool, adapter):
            adapter.layers[1].weights["model.layers.1.out_proj.lora_B.weight"] = (
                torch.ones(3, 2)
            )

        self.check_preflight(mutate, r"layer 1.*buffer shape \(4, 2\).*\(3, 2\)")

    def test_bad_embedding_does_not_clear_dense_buffers(self):
        def mutate(pool, adapter):
            adapter.embedding_layers = {
                "model.embed_tokens.lora_embedding_A": torch.ones(2, 2),
                "model.embed_tokens.lora_embedding_B": torch.ones(4, 2),
            }

        self.check_preflight(mutate, "buffer shape")

    def test_missing_module_preflight_leaves_buffers_unchanged(self):
        def mutate(pool, adapter):
            adapter.layers[1].weights["model.layers.1.unknown.lora_A.weight"] = (
                torch.ones(2, 3)
            )

        self.check_preflight(mutate, "did not match any target module")

    def test_weight_pair_rank_dtype_and_finiteness(self):
        _, adapter, _ = self.make_pool_and_adapter()
        key = "model.layers.1.out_proj.lora_B.weight"
        for bad, message in [
            (torch.ones(4, 1), "rank 1"),
            (torch.ones(4, 2, dtype=torch.int32), "floating"),
            (torch.full((4, 2), float("nan")), "non-finite"),
            (torch.full((4, 2), float("inf")), "non-finite"),
        ]:
            with self.subTest(message=message):
                adapter.layers[1].weights[key] = bad
                with self.assertRaisesRegex(ValueError, message):
                    LoRAAdapter._validate_weight_pairs(adapter)
        del adapter.layers[1].weights[key]
        with self.assertRaisesRegex(ValueError, "missing paired tensor"):
            LoRAAdapter._validate_weight_pairs(adapter)

    def test_failed_file_and_tensor_load_roll_back_metadata(self):
        for from_tensors in (False, True):
            with self.subTest(from_tensors=from_tensors):
                manager = LoRAManager.__new__(LoRAManager)
                resident = SimpleNamespace(lora_name="resident", lora_path="/resident")
                manager.configs = {"resident-id": object()}
                manager.loras = {"resident-id": object()}
                manager.lora_refs = {"resident-id": resident}
                manager.num_pinned_loras = 1
                manager.base_hf_config = SimpleNamespace(vocab_size=3)
                manager.validate_new_adapter = Mock()
                manager.load_lora_weights = Mock(side_effect=ValueError("bad tensor"))
                manager.load_lora_weights_from_tensors = Mock(
                    side_effect=ValueError("bad tensor")
                )
                ref = SimpleNamespace(
                    lora_name="bad", lora_path="/bad", lora_id="new-id", pinned=True
                )
                with patch.object(manager_module, "LoRAConfig"):
                    if from_tensors:
                        result = manager._load_lora_adapter_from_tensors(ref, {}, {})
                    else:
                        result = manager._load_lora_adapter(ref)
                self.assertFalse(result.success)
                self.assertIn("bad tensor", result.error_message)
                self.assertEqual(set(manager.configs), {"resident-id"})
                self.assertEqual(set(manager.loras), {"resident-id"})
                self.assertEqual(manager.lora_refs, {"resident-id": resident})
                self.assertEqual(manager.num_pinned_loras, 1)

    def test_unsupported_peft_math_is_rejected_before_registration(self):
        manager = LoRAManager.__new__(LoRAManager)
        manager.attn_dp_enabled = False
        ref = SimpleNamespace(lora_name="unsupported")
        cases = {
            "use_rslora": True,
            "rank_pattern": {"q_proj": 4},
            "alpha_pattern": {"q_proj": 16},
            "fan_in_fan_out": True,
            "lora_bias": True,
            "modules_to_save": ["lm_head"],
            "bias": "all",
        }
        for feature, value in cases.items():
            with self.subTest(feature=feature):
                config = SimpleNamespace(
                    lora_added_tokens_size=0, use_dora=False, hf_config={feature: value}
                )
                with self.assertRaisesRegex(ValueError, feature):
                    manager.validate_new_adapter(config, ref)


if __name__ == "__main__":
    unittest.main()
