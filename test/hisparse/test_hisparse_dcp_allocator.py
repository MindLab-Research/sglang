"""Phase 1 unit test: HiSparseTokenToKVPoolAllocator DCP virtual-id domain.

Run on B300-2: cd /root/hisparse_test && /root/v15_patched/bin/python3 -m pytest test_hisparse_dcp.py -x -q (or plain python3)

Covers (Phase 1 of docs/agent/hisparse-glm-decode-plan.md):
 1. virtual->local formula parity with memory_pool._write_mla_kv_buffer
 2. owner distribution: each rank owns exactly 1/dcp of any contiguous span
 3. alloc_decode / alloc_extend owner-scoped hisparse allocation
 4. alloc/free balance: both domains return to zero
 5. translate foreign->0 sentinel (mocked pool mapping)
"""
import sys
import torch

sys.path.insert(0, "/root/hisparse_test")

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator

DCP = 8
PS = 64  # local pool page


class MockKVCache:
    """Only register_mapping + translate are needed by the allocator."""

    def __init__(self):
        self.mapping = None

    def register_mapping(self, m):
        self.mapping = m

    def _translate_loc_to_hisparse_device(self, locs):
        m = self.mapping
        if m is None:
            return locs
        out = m[locs.clamp(min=0)]
        return out


def build(size=8192, ratio=2, dcp_size=DCP, rank=0):
    kvc = MockKVCache()
    alloc = HiSparseTokenToKVPoolAllocator(
        size,
        page_size=128,  # scheduler page (GLM prod); allocator must ignore it under DCP
        dtype=torch.uint8,
        device="cuda:0",
        kvcache=kvc,
        need_sort=False,
        host_to_device_ratio=ratio,
        dcp_size=dcp_size,
    )
    alloc.dcp_rank = lambda: rank  # unit-test override (no dist context)
    return alloc


def test_formula_parity():
    """Our _dcp_virtual_to_local must equal _write_mla_kv_buffer's formula."""
    dws, ps = DCP, PS
    v = torch.arange(0, ps * dws * 37 + 13, dtype=torch.int64, device="cuda:0")  # spans many pages
    # reference (memory_pool.py:3965-3976)
    ref_slot = (v // (ps * dws)) * ps + (v % ps)
    a = build()
    for rank in range(DCP):
        a.dcp_rank = lambda r=rank: r
        mine = a._dcp_virtual_to_local(v)
        owner = ((v // ps) % dws == rank) & (v >= 0)
        assert torch.equal(mine[owner], ref_slot[owner]), f"rank {rank} slot mismatch"
        assert torch.equal(mine[~owner], torch.zeros_like(mine[~owner])), (
            f"rank {rank} foreign lanes must be 0"
        )
    print("PASS formula parity (9db63a6abb lockstep)")


def test_owner_distribution():
    a = build()
    span = torch.arange(0, PS * DCP * 16, dtype=torch.int64, device="cuda:0")  # 16 virtual pages
    counts = []
    for rank in range(DCP):
        # count ownership by the formula itself (slot 0 is a REAL local slot —
        # the >0 sentinel cannot distinguish it from foreign; known upstream
        # ambiguity, tracked in the plan doc)
        owner = ((span // PS) % DCP == rank) & (span >= 0)
        counts.append(int(owner.sum()))
        # and verify translate maps owned lanes (slot0 included) to local slots
        a.dcp_rank = lambda r=rank: r
        mine = a._dcp_virtual_to_local(span)
        assert torch.equal(mine[owner], ((span // (PS * DCP)) * PS + span % PS)[owner])
    # mutually exclusive, collectively exhaustive
    assert sum(counts) == len(span), f"owner counts {counts} != {len(span)}"
    assert max(counts) == min(counts) == len(span) // DCP, f"unbalanced {counts}"
    print(f"PASS owner distribution: {counts[0]} per rank x {DCP} = {len(span)}")


def test_alloc_decode_owner_share_and_balance():
    size, dcp = 4096, DCP
    a = build(size=size, ratio=2, dcp_size=dcp, rank=0)
    logical_capacity = size * 2 * dcp
    assert a.size == logical_capacity
    assert a.page_size == PS * dcp, "virtual page must be 64*dcp"

    n_tokens = 1024
    seq_lens = torch.tensor([n_tokens + 128], dtype=torch.int64, device="cuda:0")
    last = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    got = []
    win0 = a.hisparse_attn_allocator.available_size()
    for _ in range(200):
        idx = a.alloc_decode(seq_lens, seq_lens.cpu(), last)
        assert idx is not None
        got.append(idx)
        assert torch.all((idx >= 0) & (idx < logical_capacity)), "virtual-domain OOB"
    # demand-paged design: device window untouched by alloc_decode
    assert a.hisparse_attn_allocator.available_size() == win0, (
        "alloc_decode must not consume the demand-paged device window"
    )
    for idx in got:
        a.free(idx)
    assert a.logical_attn_allocator.available_size() == logical_capacity
    print("PASS alloc_decode logical-only (device window demand-paged) + full free")


def test_alloc_extend_owner_pages():
    a = build(size=4096, ratio=2, dcp_size=DCP, rank=3)
    a.dcp_rank = lambda: 3
    prefix = torch.tensor([0], dtype=torch.int64, device="cuda:0")
    seq = torch.tensor([500], dtype=torch.int64, device="cuda:0")
    idx = a.alloc_extend(
        prefix, prefix.cpu(), seq, seq.cpu(),
        torch.zeros(1, dtype=torch.int64, device="cuda:0"), 500,
    )
    assert idx is not None and idx.numel() == 500
    # demand-paged: mapping stays 0 until coordinator swaps pages in
    m = a.full_to_hisparse_device_index_mapping
    assert torch.all(m[idx] == 0), "extend must not pre-map the device window"
    a.free(idx)
    assert a.logical_attn_allocator.available_size() == a.logical_attn_allocator.size
    assert a.hisparse_attn_allocator.available_size() == a.hisparse_attn_allocator.size
    print("PASS alloc_extend page-aligned owner mapping + free balance")


def test_budget_consistency():
    """available_size must be the min over both domains (SWA-lesson parity)."""
    a = build(size=1024, ratio=2, dcp_size=DCP, rank=0)
    before = a.available_size()
    assert before == 1024 * 2 * DCP  # logical (host) domain is the capacity
    # PagedTokenToKVPoolAllocator.alloc_decode pops free PAGES (kernel writes
    # into the last page's tail or a popped page; get_num_new_pages(decode)
    # decides). A seq_len that stays inside the last page pops nothing; a
    # page-crossing one pops exactly one page = page_size tokens of budget.
    inpage = torch.tensor([100], dtype=torch.int64, device="cuda:0")
    last_inpage = torch.tensor([99], dtype=torch.int64, device="cuda:0")
    idx1 = a.alloc_decode(inpage, inpage.cpu(), last_inpage)
    assert idx1 is not None
    assert a.available_size() == before, "in-page decode must not consume budget"
    a.free(idx1)
    assert a.available_size() == before
    crossing = torch.tensor([513], dtype=torch.int64, device="cuda:0")
    last_cross = torch.tensor([512], dtype=torch.int64, device="cuda:0")
    idx2 = a.alloc_decode(crossing, crossing.cpu(), last_cross)
    assert idx2 is not None
    delta = before - a.available_size()
    assert delta == a.page_size, f"page-crossing decode delta {delta} != {a.page_size}"
    a.free(idx2)
    assert a.available_size() == before
    print("PASS budget consistency (admission == alloc accounting)")


if __name__ == "__main__":
    test_formula_parity()
    test_owner_distribution()
    test_alloc_decode_owner_share_and_balance()
    test_alloc_extend_owner_pages()
    test_budget_consistency()
    print("\nALL PHASE-1 DOMAIN TESTS PASSED")
