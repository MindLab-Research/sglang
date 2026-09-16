# Copyright 2023-2024 SGLang Team
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
"""Regression tests for the KV host-offload index domain (retract path).

Context (2026-09-16, PD decode engine, all 8 TP ranks lost)::

    retract_decode -> release_req -> Req.offload_kv_cache
      -> allocator.get_cpu_copy -> MLATokenToKVPool.get_cpu_copy
      -> self.kv_buffer[layer_id][chunk_indices]   # <-- device-side assert

``Req.offload_kv_cache`` feeds ``req_to_token`` straight into ``get_cpu_copy``.
With DCP enabled those ids are *virtual* (the allocator hands out a
``size * dcp_size`` space) while the physical buffer only has
``size + page_size`` rows, so an out-of-range id hit the gather kernel and
poisoned the CUDA context.

These tests pin the two invariants of the fix:

1. ``localize_kv_offload_indices`` maps every id into the physical buffer
   (owner -> its slot, non-owner/invalid -> the scratch row), so the copy is
   in-range by construction.
2. ``assert_kv_offload_indices_in_range`` raises ``KvOffloadIndexError`` (a
   Python-level, catchable error) instead of letting the kernel assert.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.mem_cache import memory_pool as mp  # noqa: E402

SIZE = 1024
PAGE = 64
DWS = 8  # dcp size
RANK = 3


class _StubPool:
    """Minimal stand-in: the offload helpers only need size + page_size.

    The real (unbound) methods are bound here so the test exercises the shipped
    implementation, not a copy of it.
    """

    localize_kv_offload_indices = mp.MLATokenToKVPool.localize_kv_offload_indices
    assert_kv_offload_indices_in_range = (
        mp.MLATokenToKVPool.assert_kv_offload_indices_in_range
    )

    def __init__(self, size=SIZE, page_size=PAGE):
        self.size = size
        self.page_size = page_size


@pytest.fixture
def dcp(monkeypatch):
    monkeypatch.setattr(mp, "dcp_enabled", lambda: True)
    monkeypatch.setattr(mp, "get_attention_dcp_world_size", lambda: DWS)
    monkeypatch.setattr(mp, "get_attention_dcp_rank", lambda: RANK)
    yield


@pytest.fixture
def no_dcp(monkeypatch):
    monkeypatch.setattr(mp, "dcp_enabled", lambda: False)
    yield


def _expected_slot(v: int) -> int:
    """Mirror of the write-path mapping (``_write_mla_kv_buffer``)."""
    page = v // PAGE
    if page % DWS != RANK or v < 0 or v >= SIZE * DWS:
        return SIZE  # scratch row
    return (v // (PAGE * DWS)) * PAGE + (v % PAGE)


# --------------------------------------------------------------------------- #
# 1. localization puts every id inside the physical buffer
# --------------------------------------------------------------------------- #
def test_virtual_ids_are_mapped_into_the_physical_buffer(dcp):
    """The exact failure input: raw virtual ids exceed size + page_size."""
    pool = _StubPool()
    raw = torch.tensor(
        [
            RANK * PAGE,  # owner page, first token
            RANK * PAGE + 7,  # owner page, mid page
            (RANK + DWS) * PAGE + 1,  # owner page of a later round
            (RANK + 1) * PAGE,  # foreign page -> scratch
            SIZE * DWS - 1,  # last virtual id (way past the buffer)
            SIZE,  # already a physical id
            -1,  # garbage
        ],
        dtype=torch.int64,
    )
    cap = SIZE + PAGE
    assert int(raw.max()) >= cap, "precondition: raw ids are out of range (this is the crash)"

    loc = pool.localize_kv_offload_indices(raw)
    assert int(loc.min()) >= 0
    assert int(loc.max()) < cap
    # the guard the copy path now calls must pass after localization
    pool.assert_kv_offload_indices_in_range(loc, "test")
    assert [int(x) for x in loc] == [_expected_slot(int(v)) for v in raw]


def test_owner_pages_keep_their_slot_and_foreign_pages_go_to_scratch(dcp):
    pool = _StubPool()
    owner = torch.tensor([RANK * PAGE + 5, (RANK + DWS) * PAGE + 63])
    foreign = torch.tensor([(RANK + 1) * PAGE + 5, (RANK - 1) * PAGE])
    loc_owner = pool.localize_kv_offload_indices(owner)
    loc_foreign = pool.localize_kv_offload_indices(foreign)
    assert int(loc_owner.max()) < SIZE, "owner pages must land inside the pool"
    assert [int(x) for x in loc_foreign] == [SIZE, SIZE]


def test_all_pages_of_the_pool_map_within_range(dcp):
    """Sweep the whole virtual space: never out of range, scratch for foreign."""
    pool = _StubPool()
    raw = torch.arange(SIZE * DWS, dtype=torch.int64)
    loc = pool.localize_kv_offload_indices(raw)
    assert int(loc.min()) >= 0 and int(loc.max()) <= SIZE
    # owner ids outnumber foreign ones by exactly 1 : dcp-1
    n_scratch = int((loc == SIZE).sum())
    assert n_scratch == SIZE * DWS - SIZE


def test_without_dcp_indices_are_untouched(no_dcp):
    pool = _StubPool()
    raw = torch.tensor([0, 1, 63, 64, SIZE - 1], dtype=torch.int64)
    assert torch.equal(pool.localize_kv_offload_indices(raw), raw)


# --------------------------------------------------------------------------- #
# 2. the guard raises a catchable Python error (never a device assert)
# --------------------------------------------------------------------------- #
def test_guard_raises_on_out_of_range_indices():
    pool = _StubPool()
    cap = SIZE + PAGE
    for bad in (
        torch.tensor([cap], dtype=torch.int64),
        torch.tensor([0, cap + 4096], dtype=torch.int64),
        torch.tensor([-3], dtype=torch.int64),
    ):
        with pytest.raises(mp.KvOffloadIndexError):
            pool.assert_kv_offload_indices_in_range(bad, "test")

    # in-range (including the scratch row) must not raise
    pool.assert_kv_offload_indices_in_range(
        torch.tensor([0, SIZE, cap - 1], dtype=torch.int64), "test"
    )
    pool.assert_kv_offload_indices_in_range(torch.empty(0, dtype=torch.int64), "test")


def test_guard_is_reachable_from_the_offload_path():
    """``Req.offload_kv_cache`` must degrade, not propagate, a failed offload."""
    import inspect

    from sglang.srt.managers import schedule_batch

    src = inspect.getsource(schedule_batch.Req.offload_kv_cache)
    assert "except Exception" in src, "offload failure must be caught (OFFLOAD-SKIP)"
    assert "self.kv_cache_cpu = None" in src
    load_src = inspect.getsource(schedule_batch.Req.load_kv_cache)
    assert "self.kv_cache_cpu is None" in load_src, "resume must tolerate a skipped offload"


# --------------------------------------------------------------------------- #
# 3. the pool copy path applies the localization (source-level pin)
# --------------------------------------------------------------------------- #
def test_get_cpu_copy_localizes_before_indexing():
    import inspect

    for fn in (
        mp.MLATokenToKVPool.get_cpu_copy,
        mp.MLATokenToKVPool.load_cpu_copy,
    ):
        src = inspect.getsource(fn)
        assert "localize_kv_offload_indices" in src
        assert "assert_kv_offload_indices_in_range" in src

    dsa_get = inspect.getsource(mp.DSATokenToKVPool.get_cpu_copy)
    assert "localize_kv_offload_indices" in dsa_get, (
        "the page-indexed index_k companion buffer must use localized ids too"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
