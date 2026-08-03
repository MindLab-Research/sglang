from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )


@triton.jit
def _compact_dcp_kv_indices_kernel(
    page_table_ptr,  # [bs, topk] GLOBAL slots (or -1 padding)
    out_ptr,  # [bs, topk] output: owner local slots left-aligned, 0 pad
    dws: tl.constexpr,
    rank: tl.constexpr,
    topk: tl.constexpr,
):
    """Per-row: compact owner pages (slot % dws == rank) to the left, //dws,
    pad the rest with 0. Non-owner and -1 padding are skipped (left as 0)."""
    batch_id = tl.program_id(0)
    row = page_table_ptr + batch_id * topk
    out_row = out_ptr + batch_id * topk

    offs = tl.arange(0, topk)
    vals = tl.load(row + offs)  # global slots or -1

    # owner mask: slot % dws == rank (excludes -1: (-1)%dws != rank for dws>1)
    is_owner = (vals % dws == rank) & (vals >= 0)
    # prefix sum of owner count → destination position (0-based)
    owner_cumsum = tl.cumsum(is_owner.to(tl.int32), axis=0)  # [1..count]
    dst = owner_cumsum - 1  # [0..count-1] for owner slots

    page_size: tl.constexpr = 64
    allocator_page = page_size * dws  # 128

    # Fill padding slots with the first valid owner's LOCAL slot (not 0).
    # trtllm StaticTokenSparse reads ALL top_k entries (ignores seq_lens),
    # so 0 → kv_cache[0] (zero) would dilute attention. Repeat first valid
    # slot instead — same KV, harmless extra weight after softmax.
    first_val = tl.load(row).to(tl.int32)  # vals[0] (scalar from load of 1 element)
    first_local_val = (first_val // allocator_page) * page_size + (first_val % allocator_page) // dws
    tl.store(out_row + offs, tl.full(offs.shape, first_local_val, dtype=tl.int32))

    # write owner LOCAL TOKEN SLOTS to compacted positions.
    local_slots = (vals // allocator_page) * page_size + (vals % allocator_page) // dws
    tl.store(out_row + dst, local_slots, mask=is_owner)


def compact_dcp_kv_indices(
    page_table_1: torch.Tensor,
    dws: int,
    rank: int,
) -> torch.Tensor:
    """Compact owner pages to the left of each row, map to local slots via //dws.

    Args:
        page_table_1: [bs, topk] global logical slots (or -1 padding)
        dws: DCP world size
        rank: DCP rank

    Returns:
        [bs, topk] tensor: owner local slots left-aligned, 0-padded tail.
    """
    bs, topk = page_table_1.shape
    out = torch.zeros_like(page_table_1)
    _compact_dcp_kv_indices_kernel[(bs,)](
        page_table_1,
        out,
        dws=dws,
        rank=rank,
        topk=topk,
    )
    return out


def compact_dcp_kv_indices_inplace(
    page_table_1: torch.Tensor,
    out: torch.Tensor,
    dws: int,
    rank: int,
) -> None:
    """In-place version: write compacted owner pages to pre-allocated `out` buffer.
    CUDA-graph safe (no new allocations)."""
    bs, topk = page_table_1.shape
    _compact_dcp_kv_indices_kernel[(bs,)](
        page_table_1,
        out,
        dws=dws,
        rank=rank,
        topk=topk,
    )


# ---------------------------------------------------------------------------
# Inplace DCP shard kernel (Plan A)
#
# After PD transfer, the physical KV buffer has *contiguous* layout:
#   slot[s] = global slot s   (s = 0, 1, 2, ...)
#
# DCP requires *sharded* layout where each rank only stores its owned tokens
# (pos % dcp_size == dcp_rank). Two physical pages (2k, 2k+1) hold 128
# contiguous tokens = one allocator page. The shard compacts the 64 owner
# tokens into DCP page k (= physical page k):
#   - even page 2k   (p%2==0): owner tokens → page k,   slots [0    .. 31]
#   - odd  page 2k+1 (p%2==1): owner tokens → page k,   slots [32   .. 63]
#
# DCP page k = global // (dcp * page_size) = page_idx // 2.
# The compact kernel in _forward_trtllm returns global // (dcp * page_size)
# = DCP page k, so it reads from the correct page after shard.
# Decode's own writes (set_mla_kv_buffer) also target DCP page k
# (global // dcp → physical slot → page = global // (dcp * page_size)),
# so PD-transferred and decode-generated data share the same page. ✓
# ---------------------------------------------------------------------------


@triton.jit
def _inplace_shard_dcp_kernel(
    kv_ptr,             # flat buffer: (num_slots, 1, dim) or (num_slots, dim)
    page_indices_ptr,    # (num_pages,) physical page indices to shard
    num_pages,
    dcp_rank: tl.constexpr,
    dcp_size: tl.constexpr,
    page_size: tl.constexpr,   # 64
    half: tl.constexpr,        # page_size // dcp_size = 32 (unused in page-granular mode)
    dim: tl.constexpr,          # 576 (MLA) or 132 (DSA state per token)
    dim_padded: tl.constexpr,   # next_pow2(dim), e.g. 1024 or 256
    zero_non_owner: tl.constexpr,  # unused (page-granular: no non-owner within page)
):
    """Page-granular DCP shard: copy entire 64-token pages.

    Each physical page_64 belongs to rank (page_64 % dcp_size).
    If this page belongs to us, copy all 64 tokens to DCP page (page_64 // dcp_size).
    Result: every DCP page has 64 contiguous owner tokens — no non-owner mixing.
    buf_slot = (page_idx // dcp_size) * page_size + i == page_idx * half + i,
    matching set_mla_kv_buffer (loc // dws) and v2 (rpt[::dws]//dws).
    """
    pid = tl.program_id(0)
    if pid >= num_pages:
        return

    page_idx = tl.load(page_indices_ptr + pid).to(tl.int64)

    # Skip pages that don't belong to this rank
    if (page_idx % dcp_size) != dcp_rank:
        return

    # Source: entire page (64 contiguous tokens)
    src_slots = tl.arange(0, page_size)  # [0, 1, 2, ..., 63]

    # Destination: DCP page = page_idx // dcp_size, all 64 slots
    dst_page = page_idx // dcp_size
    dst_slots = tl.arange(0, page_size)  # [0, 1, 2, ..., 63]

    # Flat byte offsets (buffer element = 1 byte for fp8/uint8)
    page_bytes = page_size * dim
    src_byte = page_idx * page_bytes + src_slots * dim
    dst_byte = dst_page * page_bytes + dst_slots * dim

    dim_offs = tl.arange(0, dim_padded)
    dim_mask = dim_offs < dim
    src = tl.load(kv_ptr + src_byte[:, None] + dim_offs[None, :], mask=dim_mask[None, :])
    tl.store(kv_ptr + dst_byte[:, None] + dim_offs[None, :], src, mask=dim_mask[None, :])


@triton.jit
def _zero_dcp_non_owner_kernel(
    kv_ptr,
    page_indices_ptr,
    num_pages,
    dcp_rank: tl.constexpr,
    dcp_size: tl.constexpr,
    page_size: tl.constexpr,
    half: tl.constexpr,
    dim: tl.constexpr,
    dim_padded: tl.constexpr,
):
    """Zero the non-owner half of each DCP page (phase 2, no race).

    Runs AFTER _inplace_shard_dcp_kernel has written all owner data.
    Uses a grid barrier (host-side torch.cuda.synchronize between phases)
    to guarantee no write-write race with the owner write pass.
    """
    pid = tl.program_id(0)
    if pid >= num_pages:
        return
    page_idx = tl.load(page_indices_ptr + pid).to(tl.int64)
    dst_page = page_idx // dcp_size
    page_bytes = page_size * dim
    # Non-owner offset: opposite half of the owner's
    non_owner_offset = 0 if (page_idx % dcp_size) != 0 else half
    non_owner_slots = tl.arange(0, half) + non_owner_offset
    non_owner_byte = dst_page * page_bytes + non_owner_slots * dim
    dim_offs = tl.arange(0, dim_padded)
    dim_mask = dim_offs < dim
    tl.store(kv_ptr + non_owner_byte[:, None] + dim_offs[None, :], tl.zeros((half, dim_padded), dtype=tl.int8), mask=dim_mask[None, :])


def inplace_shard_dcp(
    kv_buffer: torch.Tensor,
    page_indices: torch.Tensor,
    dcp_rank: int,
    dcp_size: int,
    page_size: int = 64,
    dim: int = 576,
    zero_non_owner: bool = False,
) -> None:
    """Shard PD-transferred KV pages from contiguous layout to DCP layout.

    Args:
        kv_buffer: flat (num_slots, 1, dim) or (num_slots, dim) tensor
        page_indices: 1D int32/int64 tensor of physical page indices
        dcp_rank: this rank's DCP rank
        dcp_size: DCP world size
        page_size: physical page size (64)
        dim: per-token dimension (576 for MLA KV, 132 for DSA state per token)
        zero_non_owner: if True, zero the non-owner half of each DCP page.
            Used for index_k (state buffer) so MQA doesn't read stale data.
            Must be False for kv_buffer (compact kernel only reads owner entries).
    """
    num_pages = page_indices.shape[0]
    if num_pages == 0:
        return
    # Triton arange requires power-of-2; pad dim and mask the excess.
    dim_padded = 1
    while dim_padded < dim:
        dim_padded *= 2
    _inplace_shard_dcp_kernel[(num_pages,)](
        kv_buffer,
        page_indices,
        num_pages,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        page_size=page_size,
        half=page_size // dcp_size,
        dim=dim,
        dim_padded=dim_padded,
        zero_non_owner=zero_non_owner,
    )
