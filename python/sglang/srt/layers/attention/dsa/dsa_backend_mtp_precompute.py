"""Multi-step precompute utilities for Native Sparse Attention backend.

This module provides optimization utilities for multi-step speculative decoding
by precomputing shared metadata once and copying it to multiple backend instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import seqlens_expand_triton
from sglang.srt.layers.attention.dsa.utils import compute_dsa_seqlens
from sglang.srt.layers.dcp.comm import (
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
)
from sglang.srt.layers.dcp.layout import (
    get_page_dcp_lens,
    localize_page_table_for_dcp_,
)
from sglang.srt.utils import is_cuda, is_hip

# (2026-08-20) Module-level cache for real DCP config — computed once at
# model init (NOT during CUDA graph capture) to avoid CUDA ops in hot path.
_CACHED_DCP = None

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_is_cuda = is_cuda()
_is_hip = is_hip()


@dataclass
class PrecomputedMetadata:
    """Precomputed metadata shared across multiple backend instances.

    Used for multi-step speculative decoding where multiple backends
    need identical metadata. Precomputing once and copying N times
    is much faster than computing N times.

    """

    # Basic seqlens
    cache_seqlens: torch.Tensor  # int32, [bs]
    cu_seqlens_k: torch.Tensor  # int32, [bs+1]

    # Page table
    page_indices: torch.Tensor  # int32, [bs, max_len] or [expanded_bs, max_len]
    real_page_table: Optional[torch.Tensor]  # int32, transformed version

    # DSA seqlens
    seqlens_expanded: torch.Tensor  # int32, [expanded_size]
    dsa_cache_seqlens: torch.Tensor  # int32, [expanded_size]
    dsa_cu_seqlens_k: torch.Tensor  # int32, [expanded_size+1]
    seqlens_expanded_size: int

    # Dimensions
    max_len: int  # for decode/draft_extend
    max_seqlen_k: int  # for target_verify

    # FlashMLA (optional)
    flashmla_metadata: Optional[torch.Tensor] = None


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths with padding."""
    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


def _localize_precomputed_page_dcp(
    cache_seqlens: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_index_topk: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dcp_size = get_attention_dcp_world_size()
    dcp_rank = get_attention_dcp_rank()
    if real_page_table is not None:
        localize_page_table_for_dcp_(real_page_table, dcp_size, dcp_rank)

    local_cache_seqlens = get_page_dcp_lens(
        cache_seqlens, dcp_size, dcp_rank, page_size
    ).to(torch.int32)
    local_expanded = get_page_dcp_lens(
        seqlens_expanded, dcp_size, dcp_rank, page_size
    ).to(torch.int32)
    local_dsa_seqlens = compute_dsa_seqlens(local_expanded, dsa_index_topk)
    return (
        local_cache_seqlens,
        compute_cu_seqlens(local_cache_seqlens),
        local_expanded,
        local_dsa_seqlens,
        compute_cu_seqlens(local_dsa_seqlens),
    )


class DeepseekSparseAttnBackendMTPPrecomputeMixin:
    """Mixin class providing metadata precomputation for multi-step speculative decoding.

    This mixin provides the _precompute_replay_metadata method and its helpers,
    which are used to optimize CUDA graph replay in multi-step scenarios.
    """

    def _repair_global_kv_slots_(self, *tensors) -> None:
        """In-place global->local repair for slot tensors gathered from
        req_to_token (2026-08-19/20 crash family). Some req_to_token cells
        hold GLOBAL DCP virtual slots (fingerprint: sequential allocator
        runs just above the local pool size, e.g. 1995115..1995119 vs
        cap 1855232); every draft-side reader of these tables (indexer
        paged-MQA, trtllm attention, transform kernels) assumes LOCAL ids
        and faults with an async IMA otherwise. Convert with the exact
        inverse the target KV-write uses: (g // (page*dcp))*page + g%page.
        DCP-enabled paths localize separately and never call this; in-pool
        values pass through unchanged.

        (2026-08-20 FINAL FIX) Use server_args dcp_size instead of
        get_attention_dcp_world_size() — the ContextVar-gated version
        returns 1 under @dcp_disabled (EAGLE draft), silently disabling
        this repair. The WRITE path (prepare_for_draft) already converts
        ≥cap values to page v; this READ path must match. Without this:
        write→page v, read→page 4v+k, draft attention reads wrong KV →
        accept cliff. CRITICAL: keep the _virtual check — only convert
        values ≥ pool_size. <cap values are already at correct positions
        (direct index). Converting them would DOUBLE-CONVERT (data at
        wrong position → crash). Module-level cache computed once at
        model init (NOT during CUDA graph capture).
        """
        global _CACHED_DCP
        if _CACHED_DCP is None:
            try:
                from sglang.srt.server_args import get_server_args
                from sglang.srt.runtime_context import get_parallel

                _dws = int(get_server_args().dcp_size or 1)
                _drank = (
                    get_parallel().tp_rank % _dws if _dws > 1 else 0
                )
                _CACHED_DCP = (_dws, _drank)
            except Exception:
                _CACHED_DCP = (1, 0)
        dws, drank = _CACHED_DCP
        if dws <= 1:
            return
        size = int(self.token_to_kv_pool.size)
        ps = int(self.real_page_size)
        for t in tensors:
            if t is None or t.numel() == 0:
                continue
            # (2026-08-20) Rank ownership: virtual page v (64-slot) is owned
            # by rank v%dcp. ONLY own-rank virtual ids may be mapped into
            # this rank's local pool — foreign-rank ids map to a DIFFERENT
            # rank's physical page, i.e. the wrong local page (EAGLE accept
            # cliff, cross-rank KV pollution). Route foreign lanes to slot 0
            # (in-pool, safe; garbage is intercepted by target verify).
            _own = ((t // ps) % dws) == drank
            _virtual = t >= size
            _repaired = (t // (ps * dws)) * ps + (t % ps)
            t.copy_(
                torch.where(
                    _virtual & _own,
                    _repaired,
                    torch.where(_virtual & ~_own, torch.zeros_like(t), t),
                )
            )

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> PrecomputedMetadata:
        """Precompute all shared metadata for multi-step backends.

        This function extracts and computes all operations that are
        identical across different backend instances in multi-step
        speculative decoding.

        Args:
            bs: Batch size
            req_pool_indices: Request pool indices [bs]
            seq_lens: Sequence lengths [bs]
            seq_lens_cpu: Sequence lengths on CPU [bs]
            forward_mode: Forward mode (decode/target_verify)

        Returns:
            PrecomputedMetadata containing all shared intermediate results
        """
        # Slice inputs to batch size
        seq_lens = seq_lens[:bs]
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Dispatch to mode-specific precomputation
        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        elif forward_mode.is_target_verify():
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for normal decode mode."""
        max_len = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]

        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_decode_metadata,
            )

            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            page_indices = torch.empty(
                (bs, max_len), dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            dsa_cu_seqlens_k = torch.empty(
                bs + 1, dtype=torch.int32, device=self.device
            )
            if self.real_page_size > 1:
                real_cols = (max_len + self.real_page_size - 1) // self.real_page_size
                real_page_table = torch.empty(
                    (bs, real_cols), dtype=torch.int32, device=self.device
                )
                real_page_table_arg = real_page_table
            else:
                real_page_table = None
                real_page_table_arg = page_indices

            fused_dsa_decode_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_len=max_len,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
            )
            seqlens_expanded = cache_seqlens
            seqlens_expanded_size = bs

            if not dcp_enabled():
                # Draft semantics: req_to_token cells can hold GLOBAL dcp
                # virtual slots; repair gathered tables to LOCAL ids before
                # any indexer/attention consumer (2026-08-20 IMA family).
                self._repair_global_kv_slots_(page_indices, real_page_table)

            if dcp_enabled():
                (
                    cache_seqlens,
                    cu_seqlens_k,
                    seqlens_expanded,
                    dsa_cache_seqlens,
                    dsa_cu_seqlens_k,
                ) = _localize_precomputed_page_dcp(
                    cache_seqlens,
                    real_page_table,
                    seqlens_expanded,
                    self.dsa_index_topk,
                    self.real_page_size,
                )

            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=max_len,
                max_seqlen_k=max_len,
                flashmla_metadata=flashmla_metadata,
            )

        # Convert to int32 and compute cumsum
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Get page indices from cache
        page_indices = self.req_to_token[req_pool_indices, :max_len].contiguous()
        if not dcp_enabled():
            self._repair_global_kv_slots_(page_indices)

        # Compute DSA seqlens
        dsa_cache_seqlens = compute_dsa_seqlens(
            cache_seqlens, dsa_index_topk=self.dsa_index_topk
        )
        seqlens_expanded = cache_seqlens
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # Compute DSA cumsum
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # Transform page table if needed
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None  # Will use page_indices directly

        if dcp_enabled():
            (
                cache_seqlens,
                cu_seqlens_k,
                seqlens_expanded,
                dsa_cache_seqlens,
                dsa_cu_seqlens_k,
            ) = _localize_precomputed_page_dcp(
                cache_seqlens,
                real_page_table,
                seqlens_expanded,
                self.dsa_index_topk,
                self.real_page_size,
            )

        # Compute FlashMLA metadata if needed
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for target verify mode."""
        max_seqlen_k = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]
        seqlens_expanded_size = bs * self.speculative_num_draft_tokens

        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_target_verify_metadata,
            )

            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            page_indices = torch.empty(
                (seqlens_expanded_size, max_seqlen_k),
                dtype=torch.int32,
                device=self.device,
            )
            seqlens_expanded = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cu_seqlens_k = torch.empty(
                seqlens_expanded_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if self.real_page_size > 1:
                real_cols = (
                    max_seqlen_k + self.real_page_size - 1
                ) // self.real_page_size
                real_page_table = torch.empty(
                    (seqlens_expanded_size, real_cols),
                    dtype=torch.int32,
                    device=self.device,
                )
                real_page_table_arg = real_page_table
            else:
                real_page_table = None
                real_page_table_arg = page_indices

            fused_dsa_target_verify_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_seqlen_k=max_seqlen_k,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
                next_n=self.speculative_num_draft_tokens,
            )

            if not dcp_enabled():
                # Draft-extend semantics: same global->local repair as the
                # decode branch (see _repair_global_kv_slots_).
                self._repair_global_kv_slots_(page_indices, real_page_table)

            if dcp_enabled():
                (
                    cache_seqlens,
                    cu_seqlens_k,
                    seqlens_expanded,
                    dsa_cache_seqlens,
                    dsa_cu_seqlens_k,
                ) = _localize_precomputed_page_dcp(
                    cache_seqlens,
                    real_page_table,
                    seqlens_expanded,
                    self.dsa_index_topk,
                    self.real_page_size,
                )

            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=-1,
                max_seqlen_k=max_seqlen_k,
                flashmla_metadata=flashmla_metadata,
            )

        # Cache seqlens with draft tokens
        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Page indices (repeated for each draft token)
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()
        if not dcp_enabled():
            self._repair_global_kv_slots_(page_indices)

        # Generate expanded seqlens on device. seq_lens_cpu is optional for DSA
        # CUDA graph replay, so this fallback must not require a host mirror.
        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        seqlens_expanded = seqlens_expand_triton(
            extend_seq_lens,
            cache_seqlens,
            bs * self.speculative_num_draft_tokens,
            self.speculative_num_draft_tokens,
        )

        # Compute DSA seqlens
        dsa_cache_seqlens = compute_dsa_seqlens(seqlens_expanded, self.dsa_index_topk)
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # DSA cumsum
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        if dcp_enabled():
            (
                cache_seqlens,
                cu_seqlens_k,
                seqlens_expanded,
                dsa_cache_seqlens,
                dsa_cu_seqlens_k,
            ) = _localize_precomputed_page_dcp(
                cache_seqlens,
                real_page_table,
                seqlens_expanded,
                self.dsa_index_topk,
                self.real_page_size,
            )

        # FlashMLA metadata
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,  # Not used in this mode
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
        )


# Backward-compat alias
DeepseekSparseAttnBackendMTPPrecomputeMixin = (
    DeepseekSparseAttnBackendMTPPrecomputeMixin
)
