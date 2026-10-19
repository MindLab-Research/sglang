"""A per-rank chunk shorter than the deterministic prefill alignment must fail at
startup: it truncates every chunk to zero tokens and the prompt waits forever."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler"


def _check(align, chunk, attn_dp_size=1, mode=DisaggregationMode.NULL):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.truncation_align_size = align
    scheduler.chunked_prefill_size = chunk
    parallel = SimpleNamespace(attn_dp_size=attn_dp_size)
    with patch(f"{_MOD}.get_parallel", return_value=parallel):
        scheduler.check_truncation_align_fits_chunk()


class TestDeterministicChunkAlign(CustomTestCase):
    def _init_alignment(self, deterministic, gdn, backend="triton", tile=4096):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model_config = object()
        exec_config = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )
        with (
            patch(f"{_MOD}.get_exec", return_value=exec_config),
            patch(f"{_MOD}.attention_backends", return_value=(backend, backend)),
            patch(f"{_MOD}.get_int_env_var", return_value=tile),
            patch(f"{_MOD}.hybrid_gdn_config", return_value=object() if gdn else None),
        ):
            scheduler.init_deterministic_inference_config()
        return scheduler.truncation_align_size

    def test_deterministic_gdn_aligns_prefill_to_recurrence_tiles(self):
        self.assertEqual(self._init_alignment(True, True), 64)
        self.assertIsNone(self._init_alignment(True, False))
        self.assertIsNone(self._init_alignment(False, True))

    def test_gdn_alignment_preserves_attention_tile_constraint(self):
        self.assertEqual(self._init_alignment(True, True, "flashinfer"), 4096)
        self.assertEqual(self._init_alignment(True, True, "flashinfer", 96), 192)

    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)
        with self.assertRaisesRegex(ValueError, "at least 64"):
            _check(align=64, chunk=32)

    def test_chunk_covering_alignment_starts(self):
        _check(align=4096, chunk=4096)
        _check(align=4096, chunk=2048, mode=DisaggregationMode.DECODE)


if __name__ == "__main__":
    unittest.main()
