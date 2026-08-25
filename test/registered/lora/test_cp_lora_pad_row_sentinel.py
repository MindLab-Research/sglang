"""Regression test for the prefill-CP × LoRA pad-row NaN poisoning bug.

Bug (commit ``a4850acf8d``, rebased from ``12cda1cd60``): under prefill
context-parallel round-robin split the batch is padded up to ``cp_size``
alignment. The CP-shard view of the LoRA ``LoRABatchInfo`` attributed those
pad rows to the last request, so the csgmv segments applied a full LoRA
adapter delta to padding activations. The delta produced NaN in the pad
rows' KV slots; real queries in the same attention then attended those NaN
pad keys and the entire real shard got NaN'd (logits NaN -> argmax=0 ->
``!``+digit-soup garbling).

Fix: pad rows keep the ``-1`` sentinel from ``_cp_shard_row_request_ids`` and
emit ``weight_indices = -1`` for their segments; every consuming kernel guards
``if w_index < 0: return`` (``[PAD-NO-DELTA]``) so padding receives zero delta
regardless of which adapter occupies which slot. The earlier slot-0 mapping
was abandoned because it breaks once base is LRU-evicted and a real adapter
takes slot 0.

These tests pin the pure-Python shard-rebuild logic (no GPU needed):
``_cp_shard_row_request_ids`` + ``_cp_shard_batch_info_or_none`` +
``_resolve_batch_info``. A regression that re-attributes pad rows to a real
adapter slot (the original bug), drops the ``-1`` sentinel, or skips the
shard view entirely (the round-robin ``attn_cp_metadata`` is EMPTY, so the
old code that keyed on it silently disabled the shard view) turns these
cases red.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

# A CP "extend" mode: is_extend() True (so is_context_parallel_extend() True),
# is_decode() False. ForwardMode.EXTEND satisfies both.
_CP_EXTEND_MODE = ForwardMode.EXTEND


def _mock_forward_batch(seq_lengths, device, padded_rows):
    """Build a minimal ForwardBatch for the csgmv backend's CP path.

    ``padded_rows`` is the per-batch token count AFTER round-robin CP padding
    (``ceil_align(real, cp_size)``) — the value ``input_ids.shape[0]`` takes
    on this rank. The shard-rebuild logic iterates ``range(cp_rank, rows,
    cp_size)`` over this count and flags positions ``>= real`` as pad rows.

    The shard-rebuild logic reads only: ``forward_mode`` (is_extend /
    is_context_parallel_extend / is_decode), ``batch_size``,
    ``extend_seq_lens``, ``extend_seq_lens_cpu``, ``extend_num_tokens``,
    ``input_ids`` (shape only, for the shard row count) and ``return_logprob``
    (lm_head pruning gate). ``attn_cp_metadata`` is intentionally left EMPTY
    for the round-robin layout — this is the trap that disabled the first fix.
    """
    num_real_rows = sum(seq_lengths)
    return SimpleNamespace(
        batch_size=len(seq_lengths),
        forward_mode=_CP_EXTEND_MODE,
        extend_seq_lens=torch.tensor(seq_lengths, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=list(seq_lengths),
        extend_num_tokens=num_real_rows,
        input_ids=torch.empty((padded_rows, 1), dtype=torch.int32, device=device),
        return_logprob=False,
        # round-robin layout attaches an EMPTY metadata object — keying on it
        # silently disabled the shard view (the original bug). Leave it None.
        attn_cp_metadata=None,
    )


_CP_SERVER_FIELDS = dict(
    max_lora_chunk_size=16,
    enable_prefill_cp=True,
    enable_dsa_prefill_context_parallel=True,
    dsa_prefill_cp_mode="round-robin-split",
    prefill_cp_mode="round-robin-split",
)


def _build_backend(device):
    # The backend constructor only reads max_lora_chunk_size; the CP shard
    # path reads the rest off the published RuntimeContext server_args.
    return ChunkedSgmvLoRABackend(
        max_loras_per_batch=8,
        device=device,
        server_args=SimpleNamespace(max_lora_chunk_size=16),
    )


class TestCPLoraPadRowSentinel(unittest.TestCase):
    """Pad rows under prefill-CP must carry the -1 sentinel (zero delta)."""

    def setUp(self):
        self.device = torch.device("cpu")

    def _prepare_and_resolve(
        self, seq_lengths, req_weight_indices, lora_ranks, cp_size, cp_rank
    ):
        """Drive the real prepare_lora_batch -> _resolve_batch_info path under
        a forced CP topology, return the shard-aligned LoRABatchInfo."""
        # prepare_lora_batch copies lora_ranks into a preallocated
        # max_loras_per_batch-wide buffer, so the source must be that wide.
        max_loras = 8
        lora_ranks = list(lora_ranks) + [0] * (max_loras - len(lora_ranks))
        scalings = [1.0] * len(lora_ranks)
        num_real_rows = sum(seq_lengths)
        # round-robin pads the batch up to a cp_size multiple; the padded total
        # is what input_ids.shape[0] holds on every rank (cal_padded_tokens).
        padded_total = -(-num_real_rows // cp_size) * cp_size
        forward_batch = _mock_forward_batch(seq_lengths, self.device, padded_total)

        backend = _build_backend(self.device)
        # lora_ranks[0] == 0 is the base slot; nonzero entries are real adapters.
        with get_context().override_server_args(**_CP_SERVER_FIELDS):
            with get_parallel().override(
                attn_cp_size=cp_size, attn_cp_rank=cp_rank
            ):
                backend.prepare_lora_batch(
                    forward_batch=forward_batch,
                    weight_indices=list(req_weight_indices),
                    lora_ranks=list(lora_ranks),
                    scalings=scalings,
                    use_cuda_graph=False,
                )
                # This rank's shard row count mirrors the code's own split:
                # len(range(cp_rank, padded_total, cp_size)).
                shard_rows = len(range(cp_rank, padded_total, cp_size))
                shard_info = backend._resolve_batch_info(None, shard_rows)
        return backend, shard_info, shard_rows

    def test_pad_rows_carry_negative_one_weight_index(self):
        """The minimal criminal: a non-divisible extend under cp_size=8 produces
        pad rows; every pad-row segment must have weight_indices == -1 so the
        kernels early-return (zero delta). A regression that re-maps pad rows to
        a real adapter slot (the original bug attributed them to the last
        request) makes this assertion fail."""
        # 29 real tokens, cp_size=8 -> 3 pad rows (pad to 32). Two requests:
        # req0 (adapter 1, 20 tokens) + req1 (adapter 2, 9 tokens). Round-robin
        # assigns positions p%8==rank: pad positions 29/30/31 fall on ranks
        # 5/6/7, so those three ranks must own a -1 pad segment.
        seq_lengths = [20, 9]
        req_weight_indices = [1, 2]
        lora_ranks = [0, 16, 16]  # slot 0 = base (rank 0), slots 1/2 = adapters

        pad_ranks = set()
        for cp_rank in range(8):
            _, shard_info, _ = self._prepare_and_resolve(
                seq_lengths, req_weight_indices, lora_ranks, cp_size=8, cp_rank=cp_rank
            )
            seg_wi = shard_info.weight_indices[: shard_info.num_segments].tolist()
            # Every segment weight index is either a real adapter slot (>=0) or
            # the -1 pad sentinel. No pad row may carry a real adapter slot.
            for wi in seg_wi:
                self.assertIn(
                    wi, (-1, 1, 2), f"rank {cp_rank}: unexpected weight_index {wi}"
                )
            if -1 in seg_wi:
                pad_ranks.add(cp_rank)
        # The 3 pad positions (29,30,31) land on ranks 5,6,7 — those must carry
        # the -1 sentinel. A fix that drops the sentinel or re-maps pad rows to
        # the last request's adapter leaves pad_ranks empty.
        self.assertEqual(
            pad_ranks,
            {5, 6, 7},
            f"pad rows should land on ranks 5,6,7 as -1 sentinels, got {pad_ranks}",
        )

    def test_pad_rows_get_zero_delta_not_real_adapter(self):
        """The whole point of the fix: pad segments map to -1 (zero delta), NOT
        to a real adapter. Distinguishes the live fix from the reverted slot-0
        mapping AND from the original bug (pad attributed to last request).

        real=5, cp_size=4 -> padded=8. Rank 0 owns positions {0,4} (both real);
        ranks 1/2/3 own {1,5}/{2,6}/{3,7} -> each has one real row (positions
        1/2/3) plus one pad row (positions 5/6/7 >= 5). So pad rows exist and
        must surface as -1 segments while real rows keep the real adapter.
        """
        seq_lengths = [5]
        req_weight_indices = [1]
        lora_ranks = [0, 16]
        cp_size = 4

        saw_pad_sentinel = False
        for cp_rank in range(cp_size):
            _, shard_info, shard_rows = self._prepare_and_resolve(
                seq_lengths, req_weight_indices, lora_ranks, cp_size, cp_rank
            )
            seg_wi = shard_info.weight_indices[: shard_info.num_segments].tolist()
            # Every segment is either the real adapter (1) or the -1 pad sentinel.
            for wi in seg_wi:
                self.assertIn(wi, (-1, 1), f"rank {cp_rank}: unexpected {wi}")
            if -1 in seg_wi:
                saw_pad_sentinel = True
        # At least one rank must carry a pad row, and it must be sentinelized.
        self.assertTrue(
            saw_pad_sentinel,
            "no -1 pad sentinel seen across ranks — pad rows not isolated",
        )

    def test_shard_covered_matches_x_rows_no_oob(self):
        """The shard view's covered count must equal this rank's shard row
        count, otherwise _resolve_batch_info falls through to the full view
        (permutation covers >x_rows -> kernel OOB, the original IMA crash)."""
        seq_lengths = [5, 5, 5]  # 15 real, cp_size=4 -> pad to 16
        req_weight_indices = [1, 2, 1]
        lora_ranks = [0, 16, 16]
        cp_size = 4
        for cp_rank in range(cp_size):
            backend, shard_info, shard_rows = self._prepare_and_resolve(
                seq_lengths, req_weight_indices, lora_ranks, cp_size, cp_rank
            )
            self.assertEqual(
                backend._cp_shard_covered,
                shard_rows,
                f"rank {cp_rank}: shard covered {backend._cp_shard_covered} != x_rows {shard_rows}",
            )
            # The picked permutation must stay in-bounds for a shard-sized x.
            perm = shard_info.permutation[: shard_rows]
            self.assertTrue(
                (perm < shard_rows).all(),
                f"rank {cp_rank}: permutation OOB for shard of {shard_rows} rows",
            )

    def test_no_pad_when_divisible_no_sentinel_segments(self):
        """When the batch divides evenly into cp_size there are no pad rows, so
        no -1 segments should appear — guards against the fix over-eagerly
        injecting sentinels on the clean (divisible) fast path."""
        seq_lengths = [8]  # 8 real, cp_size=4 -> no padding
        req_weight_indices = [1]
        lora_ranks = [0, 16]
        cp_size = 4
        for cp_rank in range(cp_size):
            _, shard_info, _ = self._prepare_and_resolve(
                seq_lengths, req_weight_indices, lora_ranks, cp_size, cp_rank
            )
            seg_wi = shard_info.weight_indices[: shard_info.num_segments].tolist()
            self.assertFalse(
                any(wi == -1 for wi in seg_wi),
                f"rank {cp_rank}: no pad rows expected but got -1 segments {seg_wi}",
            )

    def test_cp_off_returns_none_shard_view(self):
        """With cp_size=1 the shard view is disabled (no CP). Guards against the
        shard path activating outside prefill-CP and corrupting the full view."""
        seq_lengths = [10, 6]
        num_real_rows = sum(seq_lengths)
        forward_batch = _mock_forward_batch(seq_lengths, self.device, num_real_rows)
        backend = _build_backend(self.device)
        with get_context().override_server_args(**_CP_SERVER_FIELDS):
            with get_parallel().override(attn_cp_size=1, attn_cp_rank=0):
                # Build a full-view batch_info via prepare_lora_batch so
                # _cp_prepare_ctx is set, then confirm the shard view is None.
                lora_ranks = [0, 16, 16] + [0] * 5
                backend.prepare_lora_batch(
                    forward_batch=forward_batch,
                    weight_indices=[1, 2],
                    lora_ranks=lora_ranks,
                    scalings=[1.0] * 8,
                    use_cuda_graph=False,
                )
                self.assertIsNone(backend._cp_shard_batch_info_or_none())


register_cpu_ci(est_time=10, suite="unit")


if __name__ == "__main__":
    unittest.main()
