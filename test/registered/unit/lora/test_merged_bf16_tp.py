"""Merged row projections must apply each TP shard's adapter exactly once."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from sglang.srt.lora.backend.merged_bf16_backend import MergedBF16LoRABackend
from sglang.srt.lora.layers import RowParallelLinearWithLoRA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMergedBF16TensorParallel(unittest.TestCase):
    def test_row_projection_reduces_merged_shards_once(self):
        torch.manual_seed(59)
        x = torch.randn(5, 96).bfloat16()
        weight = (torch.randn(64, 96) * 0.02).bfloat16()
        a = torch.randn(4, 96) * 0.01
        b = torch.randn(64, 4) * 0.01
        bias = torch.randn(64).bfloat16()
        for tp_size in (2, 4, 8):
            width = x.shape[1] // tp_size
            local_outputs = []
            for rank in range(tp_size):
                sl = slice(rank * width, (rank + 1) * width)
                # Independent local export, followed by the base TP projection.
                merged = (weight[:, sl].float() + (b @ a[:, sl]) * 0.5).bfloat16()
                local_outputs.append(
                    F.linear(x[:, sl], merged, bias if rank == 0 else None)
                )
            expected = torch.stack(local_outputs).sum(0)
            for attention_group in (False, True):
                for rank in range(tp_size):
                    with self.subTest(
                        tp=tp_size, rank=rank, attention_group=attention_group
                    ):
                        sl = slice(rank * width, (rank + 1) * width)
                        backend = object.__new__(MergedBF16LoRABackend)
                        backend.merge_groups = [(0, 4, 0.5, torch.arange(x.shape[0]))]
                        backend.get_batch_info = Mock(
                            return_value=SimpleNamespace(has_active_lora=True)
                        )
                        backend.run_lora_a_sgemm = Mock(
                            side_effect=AssertionError(
                                "factorized A would double-count merge"
                            )
                        )
                        backend.run_lora_b_sgemm = Mock(
                            side_effect=AssertionError(
                                "factorized B would double-count merge"
                            )
                        )
                        base = SimpleNamespace(
                            weight=weight[:, sl].contiguous(),
                            bias=bias,
                            skip_bias_add=False,
                            input_is_parallel=True,
                            reduce_results=True,
                            tp_group=object(),
                            use_dp_attention_reduce=attention_group,
                        )
                        layer = RowParallelLinearWithLoRA.__new__(
                            RowParallelLinearWithLoRA
                        )
                        torch.nn.Module.__init__(layer)
                        layer.base_layer = base
                        layer.lora_backend = backend
                        layer.set_lora = True
                        layer.A_buffer = a[:, sl].contiguous().unsqueeze(0)
                        layer.B_buffer = b.unsqueeze(0)
                        layer.output_offset_cpu = torch.tensor([0, 64])
                        reduce = Mock(return_value=expected)
                        unused = Mock(
                            side_effect=AssertionError("wrong TP reduction group")
                        )
                        with (
                            patch(
                                "sglang.srt.lora.layers.get_group_rank_size",
                                return_value=(rank, tp_size),
                            ),
                            patch(
                                "sglang.srt.lora.layers.should_skip_mlp_all_reduce",
                                return_value=False,
                            ),
                            patch(
                                "sglang.srt.lora.layers.tensor_model_parallel_all_reduce",
                                reduce if not attention_group else unused,
                            ),
                            patch(
                                "sglang.srt.lora.layers.get_parallel",
                                return_value=SimpleNamespace(
                                    attn_tp_group=SimpleNamespace(
                                        all_reduce=reduce if attention_group else unused
                                    )
                                ),
                            ),
                        ):
                            actual, output_bias = layer.forward(x[:, sl].contiguous())
                        self.assertIsNone(output_bias)
                        self.assertEqual(reduce.call_count, 1)
                        torch.testing.assert_close(
                            reduce.call_args.args[0],
                            local_outputs[rank],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        backend.run_lora_a_sgemm.assert_not_called()
                        backend.run_lora_b_sgemm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
