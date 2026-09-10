from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state_on_swa,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: torch.Tensor, page_size: int) -> np.ndarray:
    return (kv_indices[::page_size] // page_size).cpu().numpy()


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    is_chunk_cache: bool = False,
) -> None:
    if req.kv is None:
        return

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    evict_floor = max(req.cache_protected_len, getattr(req, "swa_evict_floor", 0))
    if page_size > 1 and evict_floor > req.cache_protected_len:
        evict_floor = -(-evict_floor // page_size) * page_size
    req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, evict_floor)

    if is_chunk_cache:
        # Chunk cache builds no radix tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Radix cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    new_swa_evicted_seqlen = max(
        req.kv.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.kv.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.kv.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        maybe_evict_dsv4_state_on_swa(
            token_to_kv_pool_allocator, req_to_token_pool, req, new_swa_evicted_seqlen
        )
        req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    # [RADIX-EXTRAKEY-DIAG] 插入时的 extra_key（对比 match 侧是否一致）
    # 无条件打印（包括 None）——确认存和查的 extra_key 是否一致
    if getattr(maybe_cache_unfinished_req, "_diag_logged", 0) < 30:
        maybe_cache_unfinished_req._diag_logged = (
            getattr(maybe_cache_unfinished_req, "_diag_logged", 0) + 1
        )
        logger.info(
            f"[RADIX-EXTRAKEY-DIAG] INSERT extra_key={req.extra_key!r} "
            f"rid={getattr(req, 'rid', '?')} lora_id={getattr(req, 'lora_id', '?')} "
            f"lora_path={getattr(req, 'lora_path', '?')} "
            f"tokens_len={len(req.origin_input_ids) + len(req.output_ids)}"
        )

    tree_cache.cache_unfinished_req(req, **kwargs)


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # the two resources currently have the same lifecycle, thus simplify logic below
    assert (req.req_pool_idx is None) == (req.kv is None)
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    effective_kv_committed_len = req.effective_kv_committed_len()
    # [decode-radix fix 2026-09-05] Only insert the prefill/input segment into
    # the radix tree. Inserting decode-generated output tokens is wasted per-
    # batch churn (agent harness rarely reuses assistant output as a prefix)
    # and was correlated with batch-finish-window garbling + accept collapse.
    # Cap kv_len_to_handle at input length so the tree key stays within
    # origin_input_ids; after the first insert this is a no-op (radix match
    # reuses the existing path). Prefill side: output_ids is empty, so the
    # clamp is a no-op there.
    _input_len = len(req.origin_input_ids)
    _insert_len = min(effective_kv_committed_len, _input_len)
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=_insert_len,
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (req.req_pool_idx is None) == (req.kv is None)
    if req.req_pool_idx is None and req.kv is None:
        return

    # [decode-radix output-leak fix 2026-09-07] 8cd8707855 clamped the tree's
    # claim (kv_len_to_handle) to the input segment but left this free's start
    # at the unclamped committed length — the output pages (P(I), P(C)] were
    # freed by neither the tree's tail free ([floor_pg(I), I) releases only the
    # boundary page P(I)) nor this free ([ceil(C), alloc) starts past them) —
    # a permanent device-pool leak of ~max(0, |output|-pg) tokens per finished
    # request (measured +19~22K tokens/min at matched running/prealloc states
    # on the b300 spot; 105min = 21.3% of the pool; see
    # docs/agent/decode-radix-output-churn-kv-pollution.md §6). Clamp start_p
    # to the same boundary so the output region falls into this free — mirrors
    # #22373's strip_thinking_cache semantics where effective_kv_committed_len
    # itself returns min(committed, input_len).
    start_p, end_p = _insert_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv = None


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    global_server_args = get_server_args()
    # [decode-radix output-leak fix 2026-09-07] Align to the TREE/allocator's
    # page granularity (DCP: flag_page x dcp, e.g. 64x4=256), not the flag's
    # page (64). A flag-aligned start can land mid-page on DCP, overlapping the
    # boundary page that cache_finished_req's page-granular tail free already
    # released ([floor_pg(I), I) frees page P(I) whole) — a double free on
    # paths without a free_group wrapper (retract/release_req; the finish paths
    # are group-wrapped and dedup via free_group_end's torch.unique, but the
    # retract path frees immediately). The tree's page boundary is structurally
    # disjoint from the tail free's boundary page (floor vs ceil of the same
    # boundary). Non-DCP deployments: tree page == flag page, unchanged
    # behavior.
    page_size = getattr(tree_cache, "page_size", None) or global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373). The decode-radix output
    # clamp (8cd8707855 + this fix) does the same: start_p is the tree's claim
    # boundary (input segment), so start_p < end_p is legitimate — only
    # start_p > end_p (claim beyond allocation) is an accounting error.
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p <= end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
