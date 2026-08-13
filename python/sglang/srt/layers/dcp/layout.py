# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

import torch

from sglang.srt.runtime_context import get_parallel


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def get_page_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
) -> torch.Tensor:
    """Return exact local lengths for round-robin ownership of whole pages.

    Full pages are assigned by ``global_page % dcp_size``. A partial tail page
    contributes only its valid tokens to its owner, preserving the distinct
    causal length of every EAGLE verify node.
    """
    if dcp_size == 1:
        return lens
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")

    full_pages = torch.div(lens, page_size, rounding_mode="floor")
    tail_tokens = torch.remainder(lens, page_size)
    local_full_pages = torch.div(full_pages, dcp_size, rounding_mode="floor")
    local_full_pages = local_full_pages + (dcp_rank < full_pages % dcp_size)
    owns_tail = full_pages % dcp_size == dcp_rank
    return local_full_pages * page_size + torch.where(
        owns_tail, tail_tokens, torch.zeros_like(tail_tokens)
    )


def localize_page_table_for_dcp_(
    page_table: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> None:
    """Localize a global page table in place for whole-page DCP ownership."""
    if dcp_size == 1:
        return
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")

    local_pages = page_table[:, dcp_rank::dcp_size].clone()
    valid = local_pages >= 0
    local_pages = torch.where(
        valid,
        torch.div(local_pages, dcp_size, rounding_mode="floor"),
        local_pages,
    )
    page_table.fill_(-1)
    page_table[:, : local_pages.shape[1]].copy_(local_pages)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = (
            kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
            // parallel.dcp_size
        )
    return kv_indices


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))
