"""
DSA 专用融合 kernel：per-block fp8 量化(k_nope) + bf16 保留(k_rope) + paged buffer 直接写入
替代 quantize_k_cache() + kv_buffer[loc] = tensor 两步

混合 dtype 写入方案：传 3 个指针（fp8 nope / fp32 scale / bf16 rope），各指向同一 buffer 的不同偏移
"""
import torch
import triton
import triton.language as tl

DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
NUM_NOPE_BLOCKS = DIM_NOPE // GROUP_SIZE  # 4
SCALE_BYTES = NUM_NOPE_BLOCKS * 4  # 16
ROPE_BYTES = DIM_ROPE * 2  # 128
TOTAL_BYTES = DIM_NOPE + SCALE_BYTES + ROPE_BYTES  # 656
NOPE_OFFSET = 0
SCALE_OFFSET = DIM_NOPE  # 512 bytes
ROPE_OFFSET = DIM_NOPE + SCALE_BYTES  # 528 bytes


@triton.jit
def _fused_dsa_quant_store_kernel(
    k_nope_ptr,       # (num_tokens, 512) bf16
    k_rope_ptr,       # (num_tokens, 64) bf16
    nope_buf_ptr,     # (size, 1, 512) fp8
    scale_buf_ptr,    # (size, 1, 4) fp32
    rope_buf_ptr,     # (size, 1, 64) bf16
    loc_ptr,          # (num_tokens,) int32
    k_nope_stride_0,
    k_rope_stride_0,
    nope_buf_stride_0,
    scale_buf_stride_0,
    rope_buf_stride_0,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    NUM_NOPE_BLOCKS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    cache_loc = tl.load(loc_ptr + token_id).to(tl.int64)

    if block_id < NUM_NOPE_BLOCKS:
        # --- nope block: per-block fp8 量化 ---
        offs = block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        y = tl.load(k_nope_ptr + token_id * k_nope_stride_0 + offs).to(tl.float32)
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, -FP8_MAX, FP8_MAX).to(nope_buf_ptr.dtype.element_ty)

        # 写 fp8 量化值
        dst_q = nope_buf_ptr + cache_loc * nope_buf_stride_0 + offs
        tl.store(dst_q, y_q)

        # 写 fp32 scale（1 个 fp32 per block）
        dst_s = scale_buf_ptr + cache_loc * scale_buf_stride_0 + block_id
        tl.store(dst_s, y_s)
    else:
        # --- rope: 直接拷贝 bf16 ---
        rope_block = block_id - NUM_NOPE_BLOCKS
        offs = rope_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        data = tl.load(k_rope_ptr + token_id * k_rope_stride_0 + offs, mask=mask, other=0.0)

        dst = rope_buf_ptr + cache_loc * rope_buf_stride_0 + offs
        tl.store(dst, data, mask=mask)


def fused_dsa_quant_store(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    kv_buffer: torch.Tensor,  # (size, 1, 656) fp8
    loc: torch.Tensor,
):
    num_tokens = k_nope.shape[0]
    # 确保 contiguous
    kv_buffer = kv_buffer.contiguous()

    # 创建 3 个 view（指向同一块内存的不同偏移）
    buf_flat = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[1], -1)  # (size, 1, 656) bytes

    # nope: (size, 1, 512) fp8
    nope_buf = buf_flat[:, :, NOPE_OFFSET:NOPE_OFFSET + DIM_NOPE]
    # scale: view as fp32 (size, 1, 4)
    scale_buf = buf_flat[:, :, SCALE_OFFSET:SCALE_OFFSET + SCALE_BYTES].view(torch.float32)
    # rope: view as bf16 (size, 1, 64)
