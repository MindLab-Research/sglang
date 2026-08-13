import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_shrink_config
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel


@cached_triton_kernel(
    lambda _, kwargs: (
        kwargs["K"],
        kwargs["NUM_SLICES"],
        kwargs["BLOCK_M"],
        kwargs["SPLIT_K"],
    )
)
@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA shrink operations.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    SPLIT_K>1 splits the K (input) dimension across multiple CTAs. Each CTA
    computes a partial sum over its K-slice and atomically accumulates into the
    output. This is essential on Blackwell (SM103): with a single decode token
    the grid is otherwise (1, num_segs) — one CTA per segment, leaving 147 of
    148 SMs idle. SPLIT_K=4..8 maps K=6144 onto 4-8 CTAs per (segment, N-block),
    raising SM occupancy by that factor.

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_sk = pid % SPLIT_K
    pid_n = pid // SPLIT_K

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_global = (k * SPLIT_K + pid_sk) * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_global[None, :] < K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_global[:, None] < K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * SPLIT_K * x_stride_1
        w_ptrs += BLOCK_K * SPLIT_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    if SPLIT_K == 1:
        tl.store(output_ptr, partial_sum, mask=output_mask)
    else:
        tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")


def _pick_split_k(M: int, N: int, K: int, num_segs: int, block_m: int) -> int:
    """Choose SPLIT_K so the total CTA count reaches a Blackwell-worthy occupancy.

    On SM103 (148 SMs), a decode batch of one token produces grid =
    (cdiv(N,BLOCK_N) * SPLIT_K, num_segs). With N=16, BLOCK_N=16, num_segs=1
    that is just (SPLIT_K, 1) — we want SPLIT_K to fill as many SMs as useful.
    We cap at 8 to avoid atomic_add contention on tiny (16,16) tiles; the K
    loop then still covers 6144/8 = 768 per CTA.
    """
    # effective number of tokens handled by the kernel
    effective_tokens = max(1, min(M, num_segs * block_m))
    if effective_tokens >= 64:
        # enough parallelism from BLOCK_M alone
        return 1
    # small batch: parallelize over K. Split into at most 8 chunks.
    block_k_min = 64
    return min(8, max(1, K // block_k_min))


def chunked_sgmv_lora_shrink_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, num_slices * r, input_dim)
    # output: (s, num_slices * r)
    # num_slices: qkv=3, gate_up=2, others=1
    # when called with multiple slices, the weights.shape[-2] will be num_slices * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Block shapes — use auto-tuned config if available, else defaults
    BLOCK_M = batch_info.max_len
    # weights shape is (num_lora, num_slices * rank, input_dim)
    MAX_RANK = weights.shape[1] // num_slices
    config = get_lora_shrink_config(
        K=weights.shape[2], R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M
    )
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K

    num_segments = batch_info.num_segments
    segment_grid = (
        batch_info.weight_indices.shape[0]
        if batch_info.use_cuda_graph
        else num_segments
    )
    SPLIT_K = _pick_split_k(S, N, K, segment_grid, BLOCK_M)
    grid = (
        SPLIT_K * triton.cdiv(N, BLOCK_N),
        segment_grid,
    )

    # Optional launch params from tuned config
    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]

    output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    _chunked_lora_shrink_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=segment_grid,
        # constants
        N=N,
        K=K,
        NUM_SLICES=num_slices,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        **extra_kwargs,
    )

    return output
