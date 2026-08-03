"""DCP (Decode Context Parallel) × PD debugging & validation helpers.

Env-gated via ``SGLANG_DCP_DEBUG``. When OFF (default) every function is a
near-zero-cost no-op (a single env-check + early return). When ON, each helper
validates a *design assumption* at runtime and logs the actual values, so a
single decode restart reveals exactly which assumption breaks — instead of a
silent wrong-answer or an opaque crash.

The assumptions validated here are the ones the DCP×PD design relies on but
could not be verified without a live DCP environment (see
docs/design/DCP_PD_LAYERSPLIT_DESIGN.md §3, §4):

  A1  global logical slot == token position  (so pos % dcp == loc % dcp)
  A2  local physical slot = global slot // dcp_size
  A3  DCP owner: loc % dcp_world_size == dcp_rank
  A4  top-k page localization: global_topk % dcp == rank  →  local = // dcp
  A5  LSE shape compatibility with cp_lse_ag_out_rs_mla  ([B, H])
  A6  head amplification: Q heads after all_gather == base * dcp_world_size
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENV = "SGLANG_DCP_DEBUG"
_cached: int | None = None


def dcp_debug_enabled() -> bool:
    global _cached
    if _cached is None:
        _cached = 1 if os.environ.get(_ENV, "0") not in ("0", "", "false", "False") else 0
    return _cached == 1


def _tag(msg: str) -> str:
    return f"[DCP-DEBUG] {msg}"


def log_kv_write(
    tag: str,
    loc: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    pool_size: int,
) -> None:
    """Validate A1/A2/A3 at KV-write time (set_mla_kv_buffer path).

    Checks:
      - every kept loc satisfies loc % dcp == dcp_rank  (A3, owner rule)
      - local slot = loc // dcp is within physical pool  (A2, no OOB)
      - global slot == position invariant hint
    """
    if not dcp_debug_enabled():
        return
    if loc.numel() == 0:
        return
    dws = dcp_world_size
    owner_ok = bool(torch.all(loc % dws == dcp_rank).item())
    local = loc // dws
    oob = int((local >= pool_size + 1).sum().item())  # +1 page slack
    logger.info(
        _tag(
            f"KV_WRITE {tag}: dcp={dws} rank={dcp_rank} n={loc.numel()} "
            f"owner_ok={owner_ok} loc[min,max]=[{int(loc.min())},{int(loc.max())}] "
            f"local[min,max]=[{int(local.min())},{int(local.max())}] pool_size={pool_size} "
            f"oob={oob}"
        )
    )
    if not owner_ok:
        logger.warning(
            _tag(
                f"KV_WRITE {tag}: OWNER VIOLATION — some loc % dcp != rank. "
                f"This means global_slot != position (A1 broken) or wrong dcp_rank."
            )
        )
    if oob > 0:
        logger.error(
            _tag(
                f"KV_WRITE {tag}: OOB — local slot >= pool_size ({pool_size}). "
                f"loc//dcp mapping (A2) is wrong; physical buffer will be corrupted."
            )
        )


def log_topk_localization(
    tag: str,
    global_topk: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    local_topk: torch.Tensor,
) -> None:
    """Validate A4: global top-k page localization to this DCP rank.

    global_topk: global page indices selected by indexer (same on all ranks).
    local_topk:  the subset this rank keeps, mapped to local physical slot.
    Correctness: local = global[mask] // dcp, where mask = global % dcp == rank.
    """
    if not dcp_debug_enabled():
        return
    dws = dcp_world_size
    mask = global_topk % dws == dcp_rank
    expected = global_topk[mask] // dws
    match = torch.equal(expected.to(local_topk.device), local_topk.to(expected.dtype))
    logger.info(
        _tag(
            f"TOPK_LOC {tag}: dcp={dws} rank={dcp_rank} "
            f"global_n={global_topk.numel()} local_n={local_topk.numel()} "
            f"match={match}"
        )
    )
    if not match:
        logger.error(
            _tag(
                f"TOPK_LOC {tag}: MISMATCH — local top-k != global[mask]//dcp. "
                f"Localization formula (A4) is wrong; sparse attention will read wrong pages."
            )
        )


def assert_lse_for_merge(
    tag: str,
    lse: torch.Tensor,
    batch_size: int,
    num_heads: int,
) -> None:
    """Validate A5: lse shape compatible with cp_lse_ag_out_rs_mla (expects [B, H]).

    cp_lse_ag_out_rs_mla (comm.py:104) does _ag_lse(lse) then all-gather across
    DCP ranks; lse must be [B, H] (or broadcastable). A wrong rank/shape silently
    corrupts the softmax merge.
    """
    if not dcp_debug_enabled():
        return
    if lse is None:
        logger.error(_tag(f"LSE {tag}: lse is None but return_lse was requested"))
        return
    logger.info(
        _tag(
            f"LSE {tag}: shape={tuple(lse.shape)} dtype={lse.dtype} "
            f"B={batch_size} H={num_heads}"
        )
    )
    # cp_lse_ag_out_rs_mla reshapes to [B, H]; be lenient but warn on surprises.
    if lse.dim() != 2:
        logger.warning(
            _tag(
                f"LSE {tag}: expected 2D [B,H], got {lse.dim()}D {tuple(lse.shape)}. "
                f"cp_lse_ag_out_rs_mla may need a reshape/view before merge."
            )
        )


def assert_q_head_amplification(
    tag: str,
    q_heads_before: int,
    q_heads_after: int,
    dcp_world_size: int,
) -> None:
    """Validate A6: all_gather_q_for_mla_decode amplifies Q heads by dcp_world_size."""
    if not dcp_debug_enabled():
        return
    expected = q_heads_before * dcp_world_size
    ok = q_heads_after == expected
    logger.info(
        _tag(
            f"Q_AMP {tag}: before={q_heads_before} after={q_heads_after} "
            f"dcp={dcp_world_size} expected={expected} ok={ok}"
        )
    )
    if not ok:
        logger.error(
            _tag(
                f"Q_AMP {tag}: head amplification mismatch. DSA forward_decode reshapes q "
                f"with layer.tp_q_head_num (un-amplified); must use * dcp_world_size under DCP."
            )
        )


def log_reshard(
    tag: str,
    kv_indices: "object",
    dcp_size: int,
    dcp_rank: int,
    out: "object",
) -> None:
    """Validate the PD reshard filter (filter_kv_indices_for_dcp_rank).

    Checks: out == kv_indices[mask] // dcp, mask = kv_indices % dcp == rank.
    """
    if not dcp_debug_enabled():
        return
    try:
        import numpy as _np

        ki = _np.asarray(kv_indices)
        mask = ki % dcp_size == dcp_rank
        expected = ki[mask] // dcp_size
        match = _np.array_equal(expected, _np.asarray(out))
        logger.info(
            _tag(
                f"RESHARD {tag}: dcp={dcp_size} rank={dcp_rank} "
                f"in_n={len(ki)} out_n={len(_np.asarray(out))} match={match}"
            )
        )
        if not match:
            logger.error(_tag(f"RESHARD {tag}: filter formula mismatch (A4)"))
    except Exception as e:  # instrumentation must never break the hot path
        logger.debug(_tag(f"RESHARD {tag}: validation skipped ({e})"))
