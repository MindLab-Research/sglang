"""Fused DSA KV-cache quantization and paged store.

Replaces the two-step ``quantize_k_cache_separate`` + ``set_mla_kv_buffer_triton``
with a single triton kernel that applies per-block fp8 quantization to k_nope,
keeps k_rope in bf16, and writes the mixed-dtype layout
``[nope_fp8(512) | scales_fp32(16) | rope_bf16(128)]`` directly into the paged
KV buffer at their byte offsets. This eliminates the intermediate tensor and one
kernel launch, and is byte-identical to the unfused path (lossless).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_parallel

_DIM_NOPE = 512
_DIM_ROPE = 64
_GROUP_SIZE = 128
_NUM_NOPE_BLOCKS = _DIM_NOPE // _GROUP_SIZE  # 4
_SCALE_BYTES = _NUM_NOPE_BLOCKS * 4  # 16
_ROPE_BYTES = _DIM_ROPE * 2  # 128
_TOTAL_BYTES = _DIM_NOPE + _SCALE_BYTES + _ROPE_BYTES  # 656
_NOPE_OFFSET = 0
_SCALE_OFFSET = _DIM_NOPE  # 512 bytes
_ROPE_OFFSET = _DIM_NOPE + _SCALE_BYTES  # 528 bytes

# fp8 e4m3fn is symmetric: max = -min = 448.
_FP8_MAX = 448.0


@triton.jit
def _fused_dsa_quant_store_kernel(
    k_nope_ptr,
    k_rope_ptr,
    nope_buf_ptr,
    scale_buf_ptr,
    rope_buf_ptr,
    loc_ptr,
    reserved_skip_index,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    nope_buf_stride_0: int,
    scale_buf_stride_0: int,
    rope_buf_stride_0: int,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    NUM_NOPE_BLOCKS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    cache_loc = tl.load(loc_ptr + token_id).to(tl.int64)

    # DCP-aware addressing: under DCP the slot id in ``loc`` is a *widened*
    # (virtual) id in ``[0, size * DCP_WORLD_SIZE)``. Keep only the tokens owned
    # by this rank and narrow the virtual id to the local row. Slot
    # ``reserved_skip_index`` (CUDA-graph padding) is never written. Invalid
    # tokens still compute a safe address (SIMT has no early return) but their
    # stores are masked off by ``is_valid``.
    is_valid = (cache_loc != reserved_skip_index) & (
        cache_loc % DCP_WORLD_SIZE == DCP_RANK
    )
    safe_loc = tl.where(is_valid, cache_loc, 0)
    safe_loc = safe_loc // DCP_WORLD_SIZE

    if block_id < NUM_NOPE_BLOCKS:
        # Quantize this nope block with its own per-block scale. Match the
        # unfused path byte-for-byte: quantize via ``y * (1 / y_s)`` (not
        # ``y / y_s``) and keep the defensive ``offs < DIM_NOPE`` mask.
        offs = block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        y = tl.load(
            k_nope_ptr + token_id * k_nope_stride_0 + offs, mask=mask, other=0.0
        ).to(tl.float32)
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, -FP8_MAX, FP8_MAX).to(
            nope_buf_ptr.dtype.element_ty
        )
        tl.store(
            nope_buf_ptr + safe_loc * nope_buf_stride_0 + offs,
            y_q,
            mask=mask & is_valid,
        )
        tl.store(
            scale_buf_ptr + safe_loc * scale_buf_stride_0 + block_id,
            y_s,
            mask=is_valid,
        )
    else:
        # Copy the rope part as bf16 (no quantization, position-sensitive).
        rope_block = block_id - NUM_NOPE_BLOCKS
        offs = rope_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        data = tl.load(
            k_rope_ptr + token_id * k_rope_stride_0 + offs, mask=mask, other=0.0
        )
        tl.store(
            rope_buf_ptr + safe_loc * rope_buf_stride_0 + offs,
            data,
            mask=mask & is_valid,
        )


def fused_dsa_quant_store(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    *,
    reserved_skip_index: int = 0,
) -> None:
    """Quantize k_nope to fp8 and store k_nope/k_rope into the DSA paged KV buffer.

    Args:
        k_nope: (num_tokens, 512) bf16.
        k_rope: (num_tokens, 64) bf16, already RoPE-applied.
        kv_buffer: (size, 1, 656) fp8 paged buffer, laid out as
            ``[nope_fp8(512) | scales_fp32(16) | rope_bf16(128)]``.
        loc: (num_tokens,) int32 slot ids. Under DCP these are *widened* virtual
            ids in ``[0, size * attn_dcp_size)``; the kernel filters by rank
            ownership and narrows ``v // attn_dcp_size`` to the local row.
        reserved_skip_index: slot id never written (CUDA-graph padding slot,
            default 0); pass -1 to disable skipping.
    """
    num_tokens = k_nope.shape[0]
    kv_buffer = kv_buffer.contiguous()
    # The kernel indexes the last dim with stride 1; guarantee it the same way
    # the unfused path (quantize_k_cache_separate) does.
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()

    # Three dtype-different views over the same buffer at different byte offsets.
    buf_flat = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[1], -1)
    nope_buf = buf_flat[:, :, _NOPE_OFFSET : _NOPE_OFFSET + _DIM_NOPE]
    scale_buf = buf_flat[:, :, _SCALE_OFFSET : _SCALE_OFFSET + _SCALE_BYTES].view(
        torch.float32
    )
    rope_buf = buf_flat[:, :, _ROPE_OFFSET : _ROPE_OFFSET + _ROPE_BYTES].view(
        torch.bfloat16
    )

    num_blocks_per_token = _NUM_NOPE_BLOCKS + 1  # 4 nope blocks + 1 rope block
    _fused_dsa_quant_store_kernel[(num_tokens, num_blocks_per_token)](
        k_nope,
        k_rope,
        nope_buf,
        scale_buf,
        rope_buf,
        loc,
        reserved_skip_index,
        k_nope.stride(0),
        k_rope.stride(0),
        nope_buf.stride(0),
        scale_buf.stride(0),
        rope_buf.stride(0),
        DCP_RANK=get_parallel().attn_dcp_rank,
        DCP_WORLD_SIZE=get_parallel().attn_dcp_size,
        GROUP_SIZE=_GROUP_SIZE,
        DIM_NOPE=_DIM_NOPE,
        DIM_ROPE=_DIM_ROPE,
        NUM_NOPE_BLOCKS=_NUM_NOPE_BLOCKS,
        FP8_MAX=_FP8_MAX,
    )


def test_correctness() -> bool:
    """Verify fused_dsa_quant_store is byte-identical to the unfused path."""
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

    torch.manual_seed(42)
    num_tokens, size = 128, 256

    k_nope = torch.randn(num_tokens, _DIM_NOPE, dtype=torch.bfloat16, device="cuda")
    k_rope = torch.randn(num_tokens, _DIM_ROPE, dtype=torch.bfloat16, device="cuda")
    # Slot 0 is the reserved CUDA-graph padding slot and is skipped by the
    # kernel (reserved_skip_index=0); start from 1 so every token is written.
    loc = torch.randint(1, size, (num_tokens,), dtype=torch.int32, device="cuda")

    # Unfused reference: concat + quantize + index copy.
    kv_buffer_ref = torch.zeros(
        size, 1, _TOTAL_BYTES, dtype=torch.float8_e4m3fn, device="cuda"
    )
    cache_k = torch.cat([k_nope.unsqueeze(1), k_rope.unsqueeze(1)], dim=-1).unsqueeze(
        1
    )
    cache_k_quant = quantize_k_cache(cache_k).squeeze(1).squeeze(1)
    kv_buffer_ref[loc] = cache_k_quant.unsqueeze(1)

    # Fused path.
    kv_buffer_fused = torch.zeros(
        size, 1, _TOTAL_BYTES, dtype=torch.float8_e4m3fn, device="cuda"
    )
    fused_dsa_quant_store(k_nope, k_rope, kv_buffer_fused, loc)

    # Byte-level comparison of the three regions.
    ref_u8 = kv_buffer_ref.view(torch.uint8)[loc]
    fused_u8 = kv_buffer_fused.view(torch.uint8)[loc]

    nope_match = torch.equal(ref_u8[:, :, :_DIM_NOPE], fused_u8[:, :, :_DIM_NOPE])
    scale_match = torch.equal(
        ref_u8[:, :, _DIM_NOPE : _DIM_NOPE + _SCALE_BYTES],
        fused_u8[:, :, _DIM_NOPE : _DIM_NOPE + _SCALE_BYTES],
    )
    rope_match = torch.equal(
        ref_u8[:, :, _DIM_NOPE + _SCALE_BYTES :],
        fused_u8[:, :, _DIM_NOPE + _SCALE_BYTES :],
    )

    assert nope_match, "nope fp8 quantized values differ"
    assert scale_match, "per-block fp32 scales differ"
    assert rope_match, "rope bf16 bytes differ"
    print("test_correctness passed: nope/scale/rope all byte-identical")
    return True


def test_dcp_correctness() -> bool:
    """Verify DCP ownership filter + row narrowing route each token correctly.

    Uses widened (virtual) slot ids ``v`` and confirms each token is written
    exactly once, into ``kv_buffer[v % dcp_size]`` at local row
    ``v // dcp_size``, and nowhere else.
    """
    torch.manual_seed(2026)
    dcp_size, local_size = 4, 32
    num_tokens = local_size * dcp_size

    k_nope = torch.randn(num_tokens, _DIM_NOPE, dtype=torch.bfloat16, device="cuda")
    k_rope = torch.randn(num_tokens, _DIM_ROPE, dtype=torch.bfloat16, device="cuda")
    # Unique widened ids covering [0, local_size * dcp_size): each (rank, row)
    # pair is owned by exactly one token, so any leak is detectable.
    loc = torch.randperm(num_tokens, device="cuda").to(torch.int32)

    kv_buffers = [
        torch.zeros(
            local_size, 1, _TOTAL_BYTES, dtype=torch.float8_e4m3fn, device="cuda"
        )
        for _ in range(dcp_size)
    ]

    def _views(buf):
        flat = buf.view(buf.shape[0], buf.shape[1], -1)
        return (
            flat[:, :, _NOPE_OFFSET : _NOPE_OFFSET + _DIM_NOPE],
            flat[:, :, _SCALE_OFFSET : _SCALE_OFFSET + _SCALE_BYTES].view(
                torch.float32
            ),
            flat[:, :, _ROPE_OFFSET : _ROPE_OFFSET + _ROPE_BYTES].view(
                torch.bfloat16
            ),
        )

    for r in range(dcp_size):
        nope_buf, scale_buf, rope_buf = _views(kv_buffers[r])
        _fused_dsa_quant_store_kernel[(num_tokens, _NUM_NOPE_BLOCKS + 1)](
            k_nope,
            k_rope,
            nope_buf,
            scale_buf,
            rope_buf,
            loc,
            0,
            k_nope.stride(0),
            k_rope.stride(0),
            nope_buf.stride(0),
            scale_buf.stride(0),
            rope_buf.stride(0),
            DCP_RANK=r,
            DCP_WORLD_SIZE=dcp_size,
            GROUP_SIZE=_GROUP_SIZE,
            DIM_NOPE=_DIM_NOPE,
            DIM_ROPE=_DIM_ROPE,
            NUM_NOPE_BLOCKS=_NUM_NOPE_BLOCKS,
            FP8_MAX=_FP8_MAX,
        )

    loc_cpu = loc.cpu().tolist()
    k_nope_f = k_nope.float()
    for i, v in enumerate(loc_cpu):
        owner, row = v % dcp_size, v // dcp_size
        # The per-block fp32 scale (max|seg| / FP8_MAX) is unique to this token,
        # so it is a precise fingerprint of where the token was written.
        _, scale_buf, _ = _views(kv_buffers[owner])
        for b in range(_NUM_NOPE_BLOCKS):
            expected = (
                k_nope_f[i, b * _GROUP_SIZE : (b + 1) * _GROUP_SIZE].abs().max()
                / _FP8_MAX
            )
            got = scale_buf[row, 0, b]
            if got != expected:
                raise AssertionError(
                    f"token {i} (v={v}) expected rank {owner} row {row} block {b} "
                    f"scale {expected.item()}, got {got.item()}"
                )
        # No other rank may hold this token at the same local row.
        for r2 in range(dcp_size):
            if r2 == owner:
                continue
            _, scale2, _ = _views(kv_buffers[r2])
            if scale2[row, 0].count_nonzero() != 0:
                raise AssertionError(
                    f"token {i} (v={v}) leaked into rank {r2} row {row}"
                )
    print("test_dcp_correctness passed: ownership filter + row narrowing correct")
    return True


if __name__ == "__main__":
    test_correctness()
    test_dcp_correctness()
