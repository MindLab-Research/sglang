import dataclasses
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
    get_batch_token_counts,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_lora
from sglang.srt.server_args import ServerArgs

MIN_CHUNK_SIZE = 16


class ChunkedSgmvLoRABackend(BaseLoRABackend):
    # GLM-5.3-flash MoL perf: all-base batches skip the full per-step
    # prepare_lora_batch (3 pinned-memory tensor allocs + permutation +
    # segment computation) and reset batch_info instead — the layers'
    # lora_active guards then route to the stock path. Mirrors the UNO
    # backend shortcut. LoRA-bearing batches still take the full prepare.
    skip_inactive_lora_batches = True

    # GLM-5.3-flash MoL perf: LoRA-A (shrink) overlap. The LoRA layer's
    # forward calls start_lora_a_overlap(x) BEFORE the base GEMM (see
    # ColumnParallelLinearWithLoRA.forward), which fires the chunked
    # shrink on a side stream; the base GEMM runs on the main stream in
    # parallel; apply_lora then consumes the precomputed shrink output
    # (waiting on the done event) instead of re-running it serially.
    # KNOWN ISSUE (default OFF): on the D-side decode graph this produced
    # a numeric corruption — the zero adapter's output diverged from base
    # starting at the 2nd token (FIRST_DIFF_IDX 7 vs the 430 baseline),
    # disappearing with A_OVERLAP=0. Root cause not yet isolated (suspect
    # stale-cache/stream race in capture/replay). Enabled only for debugging
    # with SGLANG_LORA_A_OVERLAP=1. See handover Phase 4b' notes.
    import os as _os

    supports_lora_a_overlap = _os.environ.get("SGLANG_LORA_A_OVERLAP", "0") == "1"

    # Fused attn shrink: wq_a/wkv share the same input x — concat their A
    # weights and fire ONE chunked shrink per layer instead of two.
    _fused_attn_enabled = _os.environ.get("SGLANG_FUSED_ATTN_SHRINK", "0") == "1"
    """
    Chunked LoRA backend using segmented matrix-vector multiplication.

    This backend is largely based on the SGMV (Segmented Gather Matrix-Vector multiplication) algorithm
    introduced in the Punica paper (https://arxiv.org/pdf/2310.18547). One main variation made here is to
    segment the input sequences into fixed-size chunks, which reduces excessive kernel launches especially
    when the LoRA distribution is skewed.
    """

    name = "csgmv"
    supports_prefill_cuda_graph = True

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        server_args: ServerArgs,
    ):
        super().__init__(max_loras_per_batch, device)
        self.max_chunk_size = get_lora().max_lora_chunk_size
        # LoRA-A overlap state (see supports_lora_a_overlap above). Event and
        # stream are kept alive on this backend instance so cuda-graph
        # captures referencing them are never torn down.
        self._overlap_a_out = None
        self._overlap_done_event = torch.cuda.Event()
        self._overlap_side_stream = None
        # Fused attn shrink pool: layer_key -> {"wq_a": A, "wkv": A} registered
        # at set_lora_info time so the FIRST run_lora_a_sgemm of a layer can
        # fire ONE concat shrink for both modules (they share input x).
        self._attn_a_pool: dict = {}
        # Per-forward fused cache: (data_ptr of x, fused_x2 [T, 2r], n_consumed)
        self._fused_attn_cache = None

    def register_attn_a(self, prefix: str, A: torch.Tensor) -> None:
        """Register an attention LoRA-A buffer for fused shrink pairing.

        Called from ColumnParallelLinearWithLoRA.set_lora_info; prefix is the
        base layer's full module prefix (e.g. "...layers.20.self_attn.wq_a").
        Pairing key strips the trailing module name.
        """
        import re as _re

        m = _re.search(r"(layers\.\d+)\.", prefix)
        layer_key = m.group(1) if m else prefix
        tail = prefix.rsplit(".", 1)[-1] if "." in prefix else prefix
        if tail in ("wq_a", "wkv", "wqkv_a"):
            self._attn_a_pool.setdefault(layer_key, {})[tail] = A

    def _try_fused_attn_shrink(self, x: torch.Tensor, A: torch.Tensor):
        """If this module's paired attn sibling is registered, fire ONE concat
        shrink for both and return this module's slice. Returns None when the
        pair is not available (fallback to separate shrink)."""
        if not self._fused_attn_enabled:
            return None
        cache = self._fused_attn_cache
        if cache is not None and cache[0] == x.data_ptr():
            # Second module of the pair — consume its slice.
            fused_x2, consumed = cache[1], cache[2]
            self._fused_attn_cache = (cache[0], fused_x2, consumed + 1)
            r = A.shape[1]  # single-slice A: [num_lora, r, K]
            return fused_x2[:, r:] if consumed == 0 else None
        # First module: find paired sibling from the pool.
        r = A.shape[1]
        # Pairing: the caller passes its own A; sibling A comes from the pool.
        # We cannot know our own tail here, so scan the pool for a sibling
        # whose A differs from ours and whose row count matches.
        for layer_key, mods in self._attn_a_pool.items():
            for tail, sib_A in mods.items():
                if sib_A is not A and sib_A.shape[1] == r and sib_A.shape[2] == A.shape[2]:
                    a_fused = torch.cat([A, sib_A], dim=1)
                    fused_x2 = chunked_sgmv_lora_shrink_forward(
                        x=x, weights=a_fused, batch_info=self.batch_info,
                        num_slices=2,
                    )
                    self._fused_attn_cache = (x.data_ptr(), fused_x2, 1)
                    return fused_x2[:, :r]
        return None

    def _get_overlap_side_stream(self) -> torch.cuda.Stream:
        if self._overlap_side_stream is None:
            self._overlap_side_stream = torch.cuda.Stream()
        return self._overlap_side_stream

    def start_lora_a_overlap(self, x, A, num_slices=1, *args, **kwargs) -> None:
        """Fire the chunked LoRA-A shrink on a side stream.

        Called by the LoRA layer BEFORE its base GEMM, so the shrink runs
        concurrently with the main-stream GEMM. The next shrink-consuming
        entry (run_lora_a_sgemm / run_qkv_lora / run_gate_up_lora) waits on
        the done event and reuses this output instead of re-running the
        shrink serially.
        """
        import os as _os

        side = self._get_overlap_side_stream()
        side.wait_stream(torch.cuda.current_stream())  # x ready on main
        with torch.cuda.stream(side):
            self._overlap_a_out = chunked_sgmv_lora_shrink_forward(
                x=x,
                weights=A,
                batch_info=self.batch_info,
                num_slices=num_slices,
            )
            if _os.environ.get("SGLANG_DEBUG_LORA"):
                _am = self._overlap_a_out.abs().max().item()
                print(
                    f"[AOV-START] slices={num_slices} shrink_absmax={_am:.6f} "
                    f"x_absmax={x.abs().max().item() if x.numel() else 0:.4f}",
                    flush=True,
                )
            self._overlap_done_event.record(side)

    def _consume_overlap_shrink(self):
        """Return the overlapped shrink output (waiting on the main stream),
        or None when no overlap was started for this forward."""
        import os as _os

        if self._overlap_a_out is None:
            if _os.environ.get("SGLANG_DEBUG_LORA"):
                print("[AOV-CONSUME] MISS (no cached shrink)", flush=True)
            return None
        out = self._overlap_a_out
        self._overlap_a_out = None
        torch.cuda.current_stream().wait_event(self._overlap_done_event)
        if _os.environ.get("SGLANG_DEBUG_LORA"):
            _am = out.abs().max().item()
            print(f"[AOV-CONSUME] HIT absmax={_am:.6f}", flush=True)
        return out

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert extra_embeddings is None, (
            "Extra embeddings for lora a is not supported yet in chunked backend"
        )
        return chunked_embedding_lora_a_forward(
            input_ids=input_ids,
            weights=weights,
            batch_info=self.batch_info,
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
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )
        # Fused attn shrink: when the paired sibling (wq_a/wkv) A is registered,
        # ONE concat shrink covers both — consume this module's slice here.
        if pruned_batch_info is None and stack_num == 1:
            fused_slice = self._try_fused_attn_shrink(x, weights)
            if fused_slice is not None:
                return fused_slice
        # LoRA-A overlap: if start_lora_a_overlap fired this forward on the
        # side stream, reuse its output instead of re-running the shrink.
        pre = None if pruned_batch_info is not None else self._consume_overlap_shrink()
        if pre is not None:
            return pre
        return chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            num_slices=stack_num,
        )

    def run_fused_attn_shrink(
        self,
        x: torch.Tensor,
        a_qkv: torch.Tensor,
        a_wqa: Optional[torch.Tensor] = None,
        a_wkv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused multi-module shrink: q_proj(3 slices) + wq_a + wkv share the
        same input x — concat their A weights along the slice dim and fire ONE
        chunked shrink. Returns [s, total_slices*r] which downstream expand
        calls consume via x_stride_0/x_slice_off."""
        if a_wqa is None or a_wkv is None:
            # Fallback: unfused shrink of just qkv part.
            r = 16
            n_q = a_qkv.shape[1] // r
            return chunked_sgmv_lora_shrink_forward(
                x=x, weights=a_qkv, batch_info=self.batch_info, num_slices=n_q,
            )
        r = a_wqa.shape[1]  # single-slice A: [num_lora, r, K]
        n_q = a_qkv.shape[1] // r  # q/k/v fused: 3 slices
        a_fused = torch.cat([a_qkv, a_wqa, a_wkv], dim=1)
        return chunked_sgmv_lora_shrink_forward(
            x=x, weights=a_fused, batch_info=self.batch_info,
            num_slices=n_q + 2,
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
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )
        return chunked_sgmv_lora_expand_forward(
            x=x,
            weights=weights,
            batch_info=batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_slice_size,
            base_output=base_output,
            x_stride_0=kwargs.get("x_stride_0"),
            x_slice_off=kwargs.get("x_slice_off", 0),
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

        # LoRA-A overlap: reuse the side-stream precomputed shrink when
        # start_lora_a_overlap fired this forward (see forward orchestration
        # in ColumnParallelLinearWithLoRA.forward).
        lora_a_output = self._consume_overlap_shrink()
        if lora_a_output is None and kwargs.get("fused_shrink") is not None:
            lora_a_output = kwargs["fused_shrink"]
        if lora_a_output is None:
            lora_a_output = chunked_sgmv_lora_shrink_forward(
                x=x,
                weights=qkv_lora_a,
                batch_info=self.batch_info,
                num_slices=n_slices,
            )
        import os as _os

        _dbg = _os.environ.get("SGLANG_DEBUG_LORA")
        _pre = base_output.abs().max().item() if (_dbg and base_output is not None) else None
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=qkv_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_qkv_out_dim,
            base_output=base_output,
            x_stride_0=kwargs.get("x_stride_0"),
            x_slice_off=kwargs.get("x_slice_off", 0),
        )
        if _dbg and base_output is not None:
            _post = base_output.abs().max().item()
            print(
                f"[AOV-EXPAND-qkv] pre={_pre:.6f} post={_post:.6f} "
                f"base_changed={_pre != _post}",
                flush=True,
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

        # lora_a_output: (s, 2 * r)
        # LoRA-A overlap: reuse the side-stream precomputed shrink when
        # start_lora_a_overlap fired this forward.
        lora_a_output = self._consume_overlap_shrink()
        if lora_a_output is None:
            lora_a_output = chunked_sgmv_lora_shrink_forward(
                x=x,
                weights=gate_up_lora_a,
                batch_info=self.batch_info,
                num_slices=2,
            )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=gate_up_lora_b,
            batch_info=self.batch_info,
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
        num_tokens, _ = get_batch_token_counts(forward_batch)
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
        with torch.device(self.device):
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

    def init_prefill_cuda_graph_batch_info(
        self, max_num_tokens: int, max_num_requests: Optional[int] = None
    ):
        # Worst-case chunk segments for any replay batch: ceil(N / chunk_top)
        # (bounded by 16 for the small tiers) plus one per adapter group.
        chunk_top = self._determine_chunk_size_for_tokens(max_num_tokens)
        max_num_segments = (
            max((max_num_tokens + chunk_top - 1) // chunk_top, 16)
            + self.max_loras_per_batch
        )
        max_bs = max_num_tokens if max_num_requests is None else max_num_requests
        with torch.device(self.device):
            self.prefill_cuda_graph_batch_info = LoRABatchInfo(
                bs=0,  # Set per batch
                use_cuda_graph=True,
                seg_lens=torch.zeros(max_num_segments, dtype=torch.int32),
                seg_indptr=torch.zeros(max_num_segments + 1, dtype=torch.int32),
                weight_indices=torch.zeros(max_num_segments, dtype=torch.int32),
                permutation=torch.zeros(max_num_tokens, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                num_segments=None,  # Set per batch
                max_len=None,  # Set per batch (chunk size)
                req_seg_indptr=torch.zeros(max_bs + 1, dtype=torch.int32),
                req_weight_indices=torch.zeros(max_bs, dtype=torch.int32),
            )
        self.prefill_cuda_graph_max_bs = max_bs
        self.prefill_cuda_graph_max_tokens = max_num_tokens

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
        use_prefill_cuda_graph: bool = False,
    ):
        chunk_size = self._determine_chunk_size(forward_batch)

        # Restored prep cache (SAFE variant): the R16/R19 crash bisect showed
        # the crash sources were id(batch_info) key reuse (base_backend) and
        # the pool-lifetime skip (fetch_new_loras) — NOT this value-keyed
        # cache. This restore fixes the original defect: the key now includes
        # the seq_lens fingerprint (multi-adapter permutations are
        # seq_lens-dependent; the old key could false-hit and OOB).
        _seq_fp = tuple(
            forward_batch.seq_lens.tolist()
        ) if forward_batch.seq_lens is not None else None
        _prep_key = (tuple(weight_indices), tuple(lora_ranks),
                     tuple(scalings), forward_batch.batch_size,
                     chunk_size, _seq_fp)
        _cached = getattr(self, "_prep_cache", {}).get(_prep_key)
        if _cached is not None:
            (permutation, weight_indices_reordered,
             seg_weight_indices, seg_indptr,
             lora_ranks_tensor, scalings_tensor, req_wi_tensor) = _cached
        else:
            permutation, weight_indices_reordered = (
                ChunkedSgmvLoRABackend._get_permutation(
                    seq_weight_indices=weight_indices,
                    forward_batch=forward_batch,
                )
            )
            seg_weight_indices, seg_indptr = self._get_segments_info(
                weights_reordered=weight_indices_reordered,
                chunk_size=chunk_size,
            )
            lora_ranks_tensor = torch.tensor(
                lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            scalings_tensor = torch.tensor(
                scalings, dtype=torch.float, pin_memory=True, device="cpu"
            )
            req_wi_tensor = torch.tensor(
                weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            if not hasattr(self, "_prep_cache"):
                self._prep_cache = {}
            self._prep_cache[_prep_key] = (
                permutation, weight_indices_reordered,
                seg_weight_indices, seg_indptr,
                lora_ranks_tensor, scalings_tensor, req_wi_tensor,
            )

        num_segments = len(seg_weight_indices)
        bs = forward_batch.batch_size
        req_seg_indptr_cpu = self._build_req_seg_indptr(forward_batch)
        max_num_segments = 0
        has_unused_cuda_graph_segments = False

        if use_prefill_cuda_graph:
            # In-place refresh of the prefill-sized static buffers; unused
            # segment slots no-op via the padded seg_indptr tail.
            batch_info = self.prefill_cuda_graph_batch_info
            batch_info.bs = bs
            batch_info.num_segments = num_segments
            batch_info.max_len = chunk_size
            max_num_segments = batch_info.weight_indices.shape[0]
            has_unused_cuda_graph_segments = num_segments < max_num_segments
        elif not use_cuda_graph:
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
        if use_prefill_cuda_graph:
            # Captured MoE kernels read every request slot; keep the tail empty.
            batch_info.req_seg_indptr[bs + 1 :].fill_(int(req_seg_indptr_cpu[-1]))
            batch_info.req_weight_indices[bs:].zero_()

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)

        self.batch_info = batch_info
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
            # lm_head LoRA runs in the eager tail, outside any captured graph.
            use_cuda_graph=False,
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
