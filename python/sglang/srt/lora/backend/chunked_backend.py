import dataclasses
import logging
import os
from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.gemm.chunked_embedding_lora_a import (
    chunked_embedding_lora_a_forward,
)
from sglang.kernels.ops.gemm.chunked_sgmv_expand import chunked_sgmv_lora_expand_forward
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    generate_sequence_lengths,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 16

# [LORA-CP-DBG] runtime probe: logs every _resolve_batch_info decision
# (view picked, row counts, segment shapes) — set SGLANG_LORA_CP_DEBUG=1.
_LORA_CP_DEBUG = bool(os.environ.get("SGLANG_LORA_CP_DEBUG", ""))


class ChunkedSgmvLoRABackend(BaseLoRABackend):
    """
    Chunked LoRA backend using segmented matrix-vector multiplication.

    This backend is largely based on the SGMV (Segmented Gather Matrix-Vector multiplication) algorithm
    introduced in the Punica paper (https://arxiv.org/pdf/2310.18547). One main variation made here is to
    segment the input sequences into fixed-size chunks, which reduces excessive kernel launches especially
    when the LoRA distribution is skewed.
    """

    name = "csgmv"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        server_args: ServerArgs,
    ):
        super().__init__(max_loras_per_batch, device)
        self.max_chunk_size = server_args.max_lora_chunk_size
        # CP-shard context: (forward_batch, req_weight_indices, chunk_size) of the
        # latest prepare_lora_batch, plus a cache for the rebuilt shard batch info.
        self._cp_prepare_ctx: Optional[Tuple[ForwardBatch, List[int], int]] = None
        self._cp_shard_cache_key: Optional[int] = None
        self._cp_shard_batch_info: Optional[LoRABatchInfo] = None
        self._cp_shard_covered: int = -1
        self._cp_full_covered: int = -1

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert (
            extra_embeddings is None
        ), "Extra embeddings for lora a is not supported yet in chunked backend"
        return chunked_embedding_lora_a_forward(
            input_ids=input_ids,
            weights=weights,
            batch_info=self._resolve_batch_info(None, input_ids.shape[0]),
            vocab_size=vocab_size,
        )

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        pruned_batch_info: LoRABatchInfo = None,
        stack_num: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_info = self._resolve_batch_info(pruned_batch_info, x.shape[0])
        return chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=stack_num,
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        pruned_batch_info: LoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # For simple lora B, we use slice offsets [0, output_dim]
        output_dim = weights.shape[-2]
        max_slice_size = output_dim
        batch_info = self._resolve_batch_info(pruned_batch_info, x.shape[0])
        return chunked_sgmv_lora_expand_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_slice_size,
            base_output=base_output,
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        n_slices: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, n_slices * r, input_dim)
        # qkv_lora_b: (num_lora, total_output_dim, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        batch_info = self._resolve_batch_info(None, x.shape[0])
        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=qkv_lora_a,
            batch_info=batch_info,
            num_slices=n_slices,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=qkv_lora_b,
            batch_info=batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_qkv_out_dim,
            base_output=base_output,
        )
        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        assert isinstance(gate_up_lora_b, torch.Tensor)
        output_dim = gate_up_lora_b.shape[-2] // 2

        batch_info = self._resolve_batch_info(None, x.shape[0])
        # lora_a_output: (s, 2 * r)
        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=gate_up_lora_a,
            batch_info=batch_info,
            num_slices=2,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=gate_up_lora_b,
            batch_info=batch_info,
            slice_offsets=output_offset,
            max_slice_size=output_dim,
            base_output=base_output,
        )
        return lora_output

    def _determine_chunk_size(self, forward_batch: ForwardBatch) -> int:
        """
        Heuristically determine the chunk size based on token token number in a batch.

        Args:
            forward_batch (ForwardBatch): The batch information containing sequence lengths.

        Returns:
            The determined chunk size
        """
        num_tokens = (
            forward_batch.extend_num_tokens
            if forward_batch.forward_mode.is_extend()
            else forward_batch.batch_size
        )
        if num_tokens is None:
            # EAGLE target-verify capture batches may have extend_num_tokens unset.
            num_tokens = forward_batch.batch_size or 0
        return self._determine_chunk_size_for_tokens(num_tokens)

    def _determine_chunk_size_for_tokens(self, num_tokens: int) -> int:
        """Determine chunk size given a token count directly."""
        if self.max_chunk_size <= MIN_CHUNK_SIZE:
            return MIN_CHUNK_SIZE

        if num_tokens >= 256:
            chunk_size = 128
        elif num_tokens >= 64:
            chunk_size = 32
        else:  # num_tokens < 64
            chunk_size = 16
        return min(self.max_chunk_size, chunk_size)

    # ------------------------------------------------------------------
    # Prefill-CP support for dense-layer csgmv calls.
    #
    # ``prepare_lora_batch`` runs BEFORE the model forward builds
    # ``forward_batch.attn_cp_metadata``, so its segments/permutation cover the
    # full (pre-split) local tokens. Under prefill-CP the dense MLP layers run
    # on THIS rank's zigzag shard of the batch, whose row count is ~1/cp of the
    # pre-split count — the csgmv kernels index ``x`` via ``permutation``, so
    # any perm value >= x rows is an OOB read/write (CUDA IMA on the first
    # LoRA forward). By the time the layers call run_*_lora the metadata is
    # ready, so we rebuild a shard-aligned batch info lazily here.
    # ------------------------------------------------------------------

    def _cp_shard_row_request_ids(
        self, forward_batch: ForwardBatch
    ) -> Optional[Tuple[List[int], int]]:
        """Per-request attribution of THIS rank's CP-shard rows (prefill-CP).

        Mirrors the model's CP token split — ``dsa_cp_round_robin_split_data``
        (round-robin: rank owns positions p % cp_size == cp_rank, ascending)
        and ``cp_split_and_rebuild_data`` (in-seq zigzag: rank owns block r
        plus the mirror block of every sequence, in [all-prev, all-mirror]
        order). Returns ``(row_req_ids, total_rows)`` or None when CP is not
        active for this batch. Never reads ``attn_cp_metadata`` for the
        round-robin layout: that mode attaches an EMPTY metadata object, so
        keying on it silently disables the shard view (the original bug).
        """
        import bisect

        from sglang.srt.runtime_context import get_parallel, get_server_args

        parallel = get_parallel()
        cp_size = getattr(parallel, "attn_cp_size", None) or 1
        cp_rank = getattr(parallel, "attn_cp_rank", None)
        if cp_size <= 1 or cp_rank is None:
            return None
        if not forward_batch.forward_mode.is_context_parallel_extend():
            return None
        server_args = get_server_args()
        if not getattr(server_args, "enable_prefill_cp", False):
            return None

        extend = forward_batch.extend_seq_lens_cpu
        if extend is None or len(extend) == 0:
            return None
        extend_list = [int(x) for x in extend]
        bs = len(extend_list)
        input_ids = getattr(forward_batch, "input_ids", None)
        rows = (
            int(input_ids.shape[0]) if input_ids is not None else sum(extend_list)
        )

        dsa_cp = getattr(server_args, "enable_dsa_prefill_context_parallel", False)
        mode = getattr(server_args, "dsa_prefill_cp_mode", None) or getattr(
            server_args, "prefill_cp_mode", ""
        )

        row_req: Optional[List[int]] = None
        real = None
        if dsa_cp and mode == "round-robin-split":
            cum = [0]
            for L in extend_list:
                cum.append(cum[-1] + L)
            real = cum[-1]
            row_req = []
            for p in range(cp_rank, rows, cp_size):
                if p >= real:
                    # Padding tail: LoRA must NOT apply a delta here. Pad-row
                    # activations are garbage-prone; a delta on them can
                    # produce NaN which then flows into the KV cache pad
                    # slots — real queries attending those slots get NaN'd
                    # (observed: layer-N gate NaN on pads only, layer-N+1
                    # attention output NaN on ALL real rows). Sentinel -1
                    # excludes the row from every LoRA segment.
                    row_req.append(-1)
                else:
                    row_req.append(bisect.bisect_right(cum, p) - 1)
        else:
            metadata = getattr(forward_batch, "attn_cp_metadata", None)
            split_list = getattr(metadata, "split_list", None)
            zigzag = getattr(metadata, "zigzag_index", None)
            if not split_list or not zigzag:
                return None
            seg_n = max(len(split_list) // max(bs, 1), 1)
            row_req = []
            for j in zigzag:
                row_req.extend([j // seg_n] * split_list[j])

        if not row_req:
            return None
        return row_req, len(row_req)

    def _cp_shard_batch_info_or_none(self) -> Optional[LoRABatchInfo]:
        if self._cp_prepare_ctx is None:
            return None
        forward_batch, req_weight_indices, chunk_size = self._cp_prepare_ctx

        cache_key = id(forward_batch)
        if self._cp_shard_cache_key == cache_key:
            return self._cp_shard_batch_info

        layout = self._cp_shard_row_request_ids(forward_batch)
        if layout is None:
            return None
        row_req, total = layout
        if total > sum(int(x) for x in (forward_batch.extend_seq_lens_cpu or [0])):
            # Guard against stale metadata: shard rows can never exceed the
            # pre-split token count.
            return None

        # Mirror _get_permutation: stable sort by adapter id so tokens group
        # by adapter, then chunk each group (mirror _get_segments_info).
        # Pad rows carry sentinel -1 (see _cp_shard_row_request_ids). Map them
        # to adapter slot 0 — the BASE slot whose lora_ranks[0] == 0, so every
        # LoRA kernel early-returns on their segments and padding activations
        # receive NO delta. (Applying a delta to pad rows NaN'd their KV slots
        # and real queries attending those slots got poisoned — the CP-LoRA
        # garbling root cause.) Keeping the segments covering ALL shard rows
        # preserves the exact segment structure the kernels are validated
        # against; excluding rows instead (permutation shorter than x) hit an
        # IMA in the absorbed step kernels (2026-08-21 warmup crash).
        row_wi = [req_weight_indices[s] if s >= 0 else 0 for s in row_req]
        order = sorted(range(total), key=lambda i: row_wi[i])
        weights_reordered = [row_wi[i] for i in order]

        seg_wi: List[int] = []
        seg_lens: List[int] = []
        i = 0
        while i < total:
            w = weights_reordered[i]
            j = i
            while j < total and weights_reordered[j] == w:
                j += 1
            group_len = j - i
            num_segs = (group_len + chunk_size - 1) // chunk_size
            seg_wi.extend([w] * num_segs)
            seg_lens.extend([chunk_size] * (num_segs - 1))
            seg_lens.append(group_len - (num_segs - 1) * chunk_size)
            i = j

        num_segments = len(seg_wi)
        seg_indptr_cpu = torch.zeros(
            (num_segments + 1,), dtype=torch.int32, pin_memory=True
        )
        seg_indptr_cpu[1:] = torch.cumsum(
            torch.tensor(seg_lens, dtype=torch.int32), dim=0
        )
        permutation_cpu = torch.tensor(order, dtype=torch.int32, pin_memory=True)

        shard = dataclasses.replace(
            self.batch_info,
            num_segments=num_segments,
            max_len=chunk_size,
            seg_indptr=seg_indptr_cpu.to(self.device, non_blocking=True),
            weight_indices=torch.tensor(
                seg_wi, dtype=torch.int32, pin_memory=True
            ).to(self.device, non_blocking=True),
            permutation=permutation_cpu.to(self.device, non_blocking=True),
        )
        self._cp_shard_cache_key = cache_key
        self._cp_shard_batch_info = shard
        self._cp_shard_covered = total
        return shard

    def _resolve_batch_info(
        self, pruned: Optional[LoRABatchInfo], x_rows: int
    ) -> LoRABatchInfo:
        """Pick the batch info whose covered rows match the actual rows of x.

        Priority: explicit ``pruned`` (lm_head) > CP shard view (dense / attn
        layers under prefill-CP run on this rank's token shard) > full
        pre-split view. The kernels index ``x`` via ``permutation``, so any
        covered count != x_rows is an OOB access; on a residual mismatch clamp
        the shard permutation (loud log) instead of going OOB.
        """
        picked = None
        if pruned is not None:
            picked = "pruned"
            result = pruned
        elif self._cp_prepare_ctx is not None:
            shard = self._cp_shard_batch_info_or_none()
            if shard is not None and self._cp_shard_covered == x_rows:
                picked = "shard"
                result = shard
            elif self._cp_full_covered == x_rows:
                picked = "full"
                result = self.batch_info
            elif shard is not None:
                logger.warning(
                    "CP LoRA batch shape mismatch: shard=%d full=%d x_rows=%d "
                    "(clamping shard permutation to stay in-bounds)",
                    self._cp_shard_covered,
                    self._cp_full_covered,
                    x_rows,
                )
                picked = "clamp"
                perm = torch.clamp(shard.permutation, max=max(x_rows - 1, 0))
                result = dataclasses.replace(shard, permutation=perm)
            else:
                picked = "full_silent_mismatch" if self._cp_full_covered != x_rows else "full"
                result = self.batch_info
        else:
            picked = "full_nocp" if self._cp_full_covered != x_rows else "nocp"
            result = self.batch_info
        if _LORA_CP_DEBUG:
            seg_lens = None
            if result is not None and result.num_segments:
                try:
                    indptr = result.seg_indptr[: result.num_segments + 1].tolist()
                    seg_lens = [indptr[i + 1] - indptr[i] for i in range(len(indptr) - 1)]
                except Exception:
                    seg_lens = "?"
            logger.info(
                "[LORA-CP-DBG] pick=%s x_rows=%d shard_cov=%d full_cov=%d "
                "nseg=%s segs=%s max_len=%s",
                picked,
                x_rows,
                self._cp_shard_covered,
                self._cp_full_covered,
                getattr(result, "num_segments", None),
                seg_lens,
                getattr(result, "max_len", None),
            )
        return result


    @staticmethod
    def _build_req_seg_indptr(forward_batch: ForwardBatch) -> torch.Tensor:
        """Build per-request cumulative token boundaries on CPU (pinned)."""
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode():
            indptr = torch.arange(bs + 1, dtype=torch.int32, pin_memory=True)
        else:
            seg_lens = generate_sequence_lengths(forward_batch, device="cpu")
            indptr = torch.zeros(bs + 1, dtype=torch.int32, pin_memory=True)
            torch.cumsum(seg_lens, dim=0, out=indptr[1:])
        return indptr

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_req: int,
    ):
        max_num_segments = (
            (num_tokens_per_req + MIN_CHUNK_SIZE - 1) // MIN_CHUNK_SIZE
        ) * max_bs_in_cuda_graph
        max_num_tokens = max_bs_in_cuda_graph * num_tokens_per_req
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=max_bs_in_cuda_graph,
                use_cuda_graph=True,
                seg_lens=torch.zeros(max_num_segments, dtype=torch.int32),
                seg_indptr=torch.zeros(max_num_segments + 1, dtype=torch.int32),
                weight_indices=torch.zeros(max_num_segments, dtype=torch.int32),
                permutation=torch.zeros(max_num_tokens, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                num_segments=None,  # Set per batch
                max_len=None,  # Not used in CSGMV backend
                req_seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                req_weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
            )

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
        chunk_size = self._determine_chunk_size(forward_batch)

        permutation, weight_indices_reordered = ChunkedSgmvLoRABackend._get_permutation(
            seq_weight_indices=weight_indices,
            forward_batch=forward_batch,
        )

        seg_weight_indices, seg_indptr = self._get_segments_info(
            weights_reordered=weight_indices_reordered,
            chunk_size=chunk_size,
        )
        num_segments = len(seg_weight_indices)

        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        )

        bs = forward_batch.batch_size
        req_wi_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        req_seg_indptr_cpu = self._build_req_seg_indptr(forward_batch)
        max_num_segments = 0
        has_unused_cuda_graph_segments = False

        if not use_cuda_graph:
            batch_info = LoRABatchInfo(
                bs=bs,
                num_segments=num_segments,
                max_len=chunk_size,
                use_cuda_graph=False,
                seg_indptr=torch.empty(
                    (num_segments + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (num_segments,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=torch.empty(
                    (len(permutation),), dtype=torch.int32, device=self.device
                ),
                seg_lens=None,
                req_seg_indptr=torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                ),
                req_weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ),
            )
        else:
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = bs
            batch_info.num_segments = num_segments
            batch_info.max_len = chunk_size
            max_num_segments = batch_info.weight_indices.shape[0]
            has_unused_cuda_graph_segments = num_segments < max_num_segments

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:num_segments].copy_(
            seg_weight_indices, non_blocking=True
        )
        if has_unused_cuda_graph_segments:
            batch_info.weight_indices[num_segments:max_num_segments].zero_()
        batch_info.seg_indptr[: num_segments + 1].copy_(seg_indptr, non_blocking=True)
        if has_unused_cuda_graph_segments:
            batch_info.seg_indptr[num_segments + 1 : max_num_segments + 1].fill_(
                int(seg_indptr[-1])
            )
        batch_info.permutation[: len(permutation)].copy_(permutation, non_blocking=True)
        batch_info.req_seg_indptr[: bs + 1].copy_(req_seg_indptr_cpu, non_blocking=True)
        batch_info.req_weight_indices[:bs].copy_(req_wi_tensor, non_blocking=True)

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)

        self.batch_info = batch_info
        # CP-shard context for dense/attn-layer csgmv calls (see
        # _cp_shard_batch_info_or_none): stash what the shard rebuild needs —
        # the forward_batch (CP state is read from parallel runtime + CPU seq
        # lens, never from late-attached metadata), the per-request adapter
        # ids, the chunk size, and the full-view covered row count.
        self._cp_prepare_ctx = (forward_batch, list(weight_indices), chunk_size)
        self._cp_shard_cache_key = None
        self._cp_shard_batch_info = None
        self._cp_shard_covered = -1
        self._cp_full_covered = int(len(permutation))
        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)
        )

    def _prepare_lm_head_batch_info(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        batch_info: LoRABatchInfo,
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:

        # Precompute lm_head_batch_info for pruned lm_head LoRA
        pruned_lens = get_lm_head_pruned_lens(forward_batch)
        lm_head_batch_info = None
        lm_head_pass_batch_infos = None

        if pruned_lens is not None:
            pruned_total = sum(pruned_lens)
            chunk_size = self._determine_chunk_size_for_tokens(pruned_total)
            lm_head_segments = merge_and_chunk_segments(
                weight_indices, pruned_lens, chunk_size=chunk_size
            )
            lm_head_batch_info = self._build_lm_head_batch_info(
                lm_head_segments, batch_info, chunk_size, pruned_total
            )

            # Precompute per-pass batch_infos for logprobs chunking
            pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)
            if pass_segments is not None:
                lm_head_pass_batch_infos = []
                for seg_wi, seg_lens_list in pass_segments:
                    pass_total = sum(seg_lens_list)
                    pass_chunk_size = self._determine_chunk_size_for_tokens(pass_total)
                    chunked_segments = merge_and_chunk_segments(
                        seg_wi, seg_lens_list, chunk_size=pass_chunk_size
                    )
                    lm_head_pass_batch_infos.append(
                        self._build_lm_head_batch_info(
                            chunked_segments,
                            batch_info,
                            pass_chunk_size,
                            pass_total,
                        )
                    )

        return lm_head_batch_info, lm_head_pass_batch_infos

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: LoRABatchInfo,
        chunk_size: int,
        expected_tokens: int,
    ) -> LoRABatchInfo:
        seg_weight_indices_cpu, seg_lens_cpu = lm_head_segments
        pruned_total = sum(seg_lens_cpu)
        num_segments = len(seg_weight_indices_cpu)

        weight_indices = torch.tensor(
            seg_weight_indices_cpu, dtype=torch.int32, device=self.device
        )
        seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=self.device)
        seg_indptr = torch.zeros(
            (num_segments + 1,), dtype=torch.int32, device=self.device
        )
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        # Identity permutation (lm_head tokens are in original order)
        permutation = torch.arange(pruned_total, dtype=torch.int32, device=self.device)

        return dataclasses.replace(
            batch_info,
            num_segments=num_segments,
            max_len=chunk_size,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            permutation=permutation,
            expected_tokens=expected_tokens,
        )

    @staticmethod
    def _get_permutation(seq_weight_indices, forward_batch: ForwardBatch):
        """
        Computes permutation indices for reordering tokens by their LoRA adapter assignments.

        This function implements the "gather" step in Chunked Segmented Gather Matrix Vector
        multiplication by creating a permutation that groups tokens by their LoRA adapter.
        Tokens using the same LoRA adapter are placed together to enable efficient batched
        computation.

        Example:
            seq_weight_indices = [0, 1, 0]  # 3 sequences using adapters [0, 1, 0]
            extend_seq_lens = [2, 1, 3]     # sequence lengths [2, 1, 3 tokens]

            # Creates row_weight_indices: [0, 0, 1, 0, 0, 0] (6 tokens total)
            # Returns permutation: [0, 1, 3, 4, 5, 2] (groups adapter 0 tokens together)
            # weights_reordered: [0, 0, 0, 0, 0, 1] (sorted by adapter)

        Args:
            seq_weight_indices: List of LoRA adapter indices for each sequence
            forward_batch (ForwardBatch): Batch information containing sequence lengths

        Returns:
            tuple: (permutation, weights_reordered) where:
                - permutation: Token reordering indices to group by adapter
                - weights_reordered: Sorted adapter indices for each token
        """
        with torch.device("cpu"):
            seq_weight_indices = torch.tensor(seq_weight_indices, dtype=torch.int32)
            seg_lens_cpu = generate_sequence_lengths(forward_batch)

            row_weight_indices = torch.repeat_interleave(
                seq_weight_indices, seg_lens_cpu
            )
            permutation = torch.empty(
                (len(row_weight_indices),), dtype=torch.long, pin_memory=True
            )
            torch.argsort(row_weight_indices, stable=True, out=permutation)
            weights_reordered = row_weight_indices[permutation]

            return permutation, weights_reordered

    def _get_segments_info(self, weights_reordered: torch.Tensor, chunk_size: int):
        """
        Computes segment information for chunked SGMV operations.

        This function takes the reordered weight indices and creates segments of fixed size
        (self.segment_size) for efficient kernel execution. Each segment contains tokens
        that use the same LoRA adapter, enabling vectorized computation.

        The segmentation is necessary because:
        1. GPU kernels work efficiently on fixed-size blocks
        2. Large groups of tokens using the same adapter are split into manageable chunks
        3. Each segment can be processed independently in parallel

        Example:
            weights_reordered = [0, 0, 0, 0, 0, 1]  # 5 tokens with adapter 0, 1 with adapter 1
            segment_size = 3

            # Creates segments:
            # Segment 0: tokens 0-2 (adapter 0), length=3
            # Segment 1: tokens 3-4 (adapter 0), length=2
            # Segment 2: token 5 (adapter 1), length=1

            # Returns:
            # weight_indices_list: [0, 0, 1] (adapter for each segment)
            # seg_indptr: [0, 3, 5, 6] (cumulative segment boundaries)

        Args:
            weights_reordered (torch.Tensor): Sorted adapter indices for each token
            chunk_size (int): Fixed size for each segment

        Returns:
            tuple: (weight_indices_list, seg_indptr) where:
                - weight_indices_list: LoRA adapter index for each segment
                - seg_indptr: Cumulative segment boundaries (CSR-style indptr)
        """
        with torch.device("cpu"):
            unique_weights, counts = torch.unique_consecutive(
                weights_reordered, return_counts=True
            )

            weight_indices_list = []
            seg_lens_list = []

            for weight_idx, group_len in zip(unique_weights, counts):
                group_len = group_len.item()
                num_segs = (group_len + chunk_size - 1) // chunk_size

                weight_indices_list.extend([weight_idx.item()] * num_segs)
                seg_lens_list.extend([chunk_size] * (num_segs - 1))
                seg_lens_list.append(group_len - (num_segs - 1) * chunk_size)

            seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32)

            weight_indices_list = torch.tensor(
                weight_indices_list, dtype=torch.int32, pin_memory=True
            )

            seg_indptr = torch.empty(
                (len(seg_lens) + 1,), dtype=torch.int32, pin_memory=True
            )
            seg_indptr[0] = 0
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

            return weight_indices_list, seg_indptr
