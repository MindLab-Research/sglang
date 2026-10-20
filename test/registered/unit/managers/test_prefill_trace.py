"""Trace spans must be correct without GPU synchronization or token reads."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

source = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/prefill_trace.py"
)
spec = importlib.util.spec_from_file_location("tested_prefill_trace", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GPUReadForbidden:
    def __len__(self):
        raise AssertionError("must not inspect GPU metadata")

    def __iter__(self):
        raise AssertionError("must not inspect GPU metadata")

    def __int__(self):
        raise AssertionError("must not synchronize a GPU scalar")


class TokenValuesForbidden(list):
    def __iter__(self):
        raise AssertionError("must not read prompt token values")


class TestPrefillTrace(unittest.TestCase):
    def batch(self):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True, name="EXTEND"),
            forward_iter=7,
            prefix_lens=[8192, 0],
            extend_lens=[8169, 23],
            extend_num_tokens=8192,
            reqs=[
                SimpleNamespace(
                    rid="long",
                    lora_id="slot-31",
                    origin_input_ids=TokenValuesForbidden([None] * 32768),
                ),
                SimpleNamespace(
                    rid="short",
                    lora_id=None,
                    origin_input_ids=TokenValuesForbidden([None] * 23),
                ),
            ],
        )

    def test_exact_request_spans_without_token_contents(self):
        batch = self.batch()
        record = module.prefill_batch_trace_record(batch)
        self.assertTrue(record["cpu_metadata_available"])
        self.assertEqual(record["span_tokens_sum"], record["extend_num_tokens"])
        self.assertEqual(
            record["requests"][0],
            {
                "rid": "long",
                "lora_id": "slot-31",
                "prompt_tokens": 32768,
                "prefix_tokens": 8192,
                "extend_tokens": 8169,
                "extend_end": 16361,
            },
        )
        self.assertEqual(record["requests"][1]["extend_end"], 23)
        self.assertEqual(batch.prefix_lens, [8192, 0])
        self.assertEqual(batch.extend_lens, [8169, 23])

    def test_gpu_metadata_is_never_read(self):
        batch = self.batch()
        batch.prefix_lens = GPUReadForbidden()
        self.assertEqual(
            module.prefill_batch_trace_record(batch),
            {
                "forward_iter": 7,
                "mode": "EXTEND",
                "cpu_metadata_available": False,
            },
        )
        batch.prefix_lens, batch.extend_lens = [GPUReadForbidden(), 0], [1, 1]
        self.assertFalse(
            module.prefill_batch_trace_record(batch)["cpu_metadata_available"]
        )

    def test_mismatched_spans_are_unavailable(self):
        batch = self.batch()
        batch.extend_lens = [8192]
        self.assertFalse(
            module.prefill_batch_trace_record(batch)["cpu_metadata_available"]
        )

    def test_decode_skips_all_metadata(self):
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: False))
        self.assertIsNone(module.prefill_batch_trace_record(batch))


if __name__ == "__main__":
    unittest.main()
