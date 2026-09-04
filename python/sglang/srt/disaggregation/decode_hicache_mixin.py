"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        if self.scheduler.enable_decode_hicache:
            last_host_node = result.last_host_node
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                try:
                    # query_storage_hit_length does synchronous file I/O
                    # (os.path.exists per key in L3 cache). When L3 has
                    # 10000+ files and LRU eviction is active (os.remove
                    # competing for inode lock), this can block the main
                    # thread for minutes → all 8 ranks stuck in file I/O
                    # simultaneously → no NCCL work enqueued for 600s →
                    # NCCL watchdog kills the process (2026-08-31 crash:
                    # 30 rounds 3000 requests ok, then 600s hang at
                    # 21:45:02 → 21:55:03 NCCL timeout, enqueued=
                    # completed=2 on all ranks, PADDED-AR-DIVERGENCE=0
                    # because the check is AFTER this call).
                    # Fix: time-box the query. If it takes >5s, skip L3
                    # hit (return 0) and fall back to L2/PD transfer.
                    import time as _time
                    _t0 = _time.monotonic()
                    l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                        last_host_node,
                        suffix_tokens,
                        last_hash,
                        prefix_keys,
                    )
                    _elapsed = _time.monotonic() - _t0
                    if _elapsed > 5.0:
                        logger.warning(
                            "L3 storage hit query took %.1fs for rid=%s; "
                            "L3 has %d files. Skipping L3 hit to avoid "
                            "blocking the event loop (gloo all_reduce "
                            "timeout risk).",
                            _elapsed,
                            req.rid,
                            len(os.listdir(
                                self.tree_cache.cache_controller
                                .storage_backend.storage_dir
                            )) if hasattr(self.tree_cache, 'cache_controller')
                            and self.tree_cache.cache_controller is not None
                            and hasattr(self.tree_cache.cache_controller,
                                        'storage_backend')
                            and hasattr(self.tree_cache.cache_controller
                                        .storage_backend, 'storage_dir')
                            else -1,
                        )
                except Exception:
                    logger.warning(
                        "L3 storage hit query failed for rid=%s; "
                        "falling back to L2-only restore",
                        req.rid,
                        exc_info=True,
                    )
                    l3_storage_hit_length = 0

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=last_host_node if l3_storage_hit_length > 0 else None,
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = prefix_match.last_host_node
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid, node, suffix, last_hash, prefix_keys
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if self.decode_req.hicache_restore_status == HiCacheRestoreResult.FAILED:
            # 2026-09-03 fix (NCCL watchdog SIGABRT / batch-divergence crash):
            # a FAILED restore MUST enter the padded all_reduce as
            # KVPoll.Failed (=0, the min) so ALL ranks abort this request in
            # the SAME iteration. Previously FAILED fell through here,
            # returning the raw poll (Success/Transferring), while
            # pop_transferred's local `or hicache_restore_status == FAILED`
            # check aborted the request UNILATERALLY on the failing rank
            # only. Evidence (2026-09-03 02:10 crash): TP0's L2 load_back
            # shortfall (l1=26880 l2=323072 new_indices=0) → TP0 aborted
            # alone → [PADDED-AR] len=19 vs peers' 20 → TP1-7 COMMITted the
            # request into their running batch → batch membership divergence
            # → EAGLE verify eagle_sample broadcast (PG2) never completed on
            # 7 ranks → NCCL 600s watchdog → SIGABRT → whole engine down.
            return KVPoll.Failed
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
            # [abort host lock fix 2026-09-04] load_back() calls
            # inc_host_lock_ref to protect host slots from eviction while
            # the H→D DMA is in flight. On ACK success, the ack handler
            # (line 3191) pops ongoing_load_back and calls
            # dec_host_lock_ref. But on ABORT, this path only called
            # dec_lock_ref (device) — the host lock was LEAKED, keeping
            # host slots locked forever. Worse, if the node was later
            # force-evicted (e.g. _reclaim_full_host_duplicates ignoring
            # the stale host_lock_ref), the host slots would be freed
            # while an in-flight DMA was still reading → KV corruption
            # → accept rate degradation → toolcall format errors.
            node_id = decode_req.hicache_restored_node.id
            ongoing = self.tree_cache.ongoing_load_back.pop(node_id, None)
            if ongoing is not None:
                self.tree_cache.dec_host_lock_ref(
                    ongoing.node, ongoing.host_lock_params
                )
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match

        # Wait for L3 -> L2 prefetch to drain (skip when no L3 hit).
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            loaded_from_storage = self.tree_cache.pop_prefetch_loaded_tokens(
                dr.req.rid
            )
            # PD covers the l3 range (total_prefix_len = l1+l2 in decode.py).
            # Clear l3 so load_back only handles L2 (if any). This avoids
            # the L3 query/actual mismatch that causes garble and leak.
            # Prefetch host data remains in the host tree for future L2 hits.
            pm.l3_storage_hit_length = 0

        # Re-match: req.last_node / prefix_indices updated to current device state.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
            )
        )
        # Failback: total coverage < required prefix means device alloc likely failed.
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        self.tree_cache.inc_lock_ref(restored_node)

        if len(new_indices) == 0:
            # Whole prefix already on device; no DMA needed.
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        # Filter once: keep only PENDING reqs that still need restore work;
        # trivially-done reqs (no prefix_match / nothing to restore) flip to READY.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        # Phase A: advance in-flight DMAs to READY.
        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        counter = self.tree_cache.cache_controller.layer_done_counter
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        # Debug: log lock_ref state before commit
        A = prefix_match.last_device_node
        B = decode_req.hicache_restored_node
        def _get_lock_ref(node):
            from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
            cd = node.component_data[ComponentType.FULL]
            return cd.lock_ref if cd.value is not None else None
        logger.warning(
            f"COMMIT rid={decode_req.req.rid} A.id={A.id} A.lock_ref={_get_lock_ref(A)} "
            f"B.id={B.id} B.lock_ref={_get_lock_ref(B)} "
            f"B_is_child_of_A={B.parent is A} needs_restore={prefix_match.needs_local_restore} "
            f"l1={prefix_match.l1_prefix_len} l2={prefix_match.l2_host_hit_length} l3={prefix_match.l3_storage_hit_length}"
        )
        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)
        self.tree_cache.inc_lock_ref(decode_req.hicache_restored_node)
        restored_len = len(decode_req.hicache_restored_kv_indices)

        # With total_prefix_len = l1+l2 (PD covers l3), _pre_alloc wrote PD
        # indices at req_to_token[l1+l2 : fill_len]. L2 load_back may load
        # more than l2 (L3 prefetch put data in host tree, inflating host_hit).
        # The extra HiCache indices overlap with PD indices at
        # [l1+l2 : l1+restored_len]. Free the overwritten PD indices first
        # to prevent orphaned allocations (leak).
        total_prefix_len = getattr(decode_req.req, "cache_protected_len", 0)
        restore_end = prefix_match.l1_prefix_len + restored_len
        if total_prefix_len < restore_end:
            overlap_indices = self.tree_cache.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx
            ][total_prefix_len:restore_end]
            self.tree_cache.token_to_kv_pool_allocator.free(overlap_indices)

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, restore_end),
            ),
            decode_req.hicache_restored_kv_indices,
        )

        # Update cache_protected_len to cover the full restored range.
        # This prevents cache_finished_req's insert from freeing HiCache
        # indices as duplicates of tree values (load_back already committed
        # them to the tree). Without this, insert would free them while the
        # tree retains them → double count → leak.
        decode_req.req.cache_protected_len = restore_end

        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
