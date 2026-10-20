"""Experimental BF16 merge semantics without a materialized merged weight.

The base and FP32 low-rank factors are read directly. A program reconstructs
only its weight tile in registers, rounds it to BF16, and consumes it in the
activation GEMM. No per-adapter dense tensor or persistent weight cache exists.
This is intentionally separate from the ordinary additive LoRA kernels.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_bf16_linear(
    X,
    W,
    A,
    B,
    Y,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    B_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DELTA_PRECISION: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ri = tl.arange(0, BR)
    bb = tl.load(
        B + ni[:, None] * B_STRIDE + ri[None, :],
        (ni[:, None] < N) & (ri[None, :] < R),
        other=0,
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        kk = ki * BK + tl.arange(0, BK)
        aa = tl.load(
            A + ri[:, None] * K + kk[None, :],
            (ri[:, None] < R) & (kk[None, :] < K),
            other=0,
        )
        delta = tl.dot(bb, aa, input_precision=DELTA_PRECISION)
        base = tl.load(
            W + ni[:, None] * K + kk[None, :],
            (ni[:, None] < N) & (kk[None, :] < K),
            other=0,
        ).to(tl.float32)
        # Match the offline export's separate FP32 multiply and add, followed
        # by exactly one BF16 conversion. Launch with FP fusion disabled.
        merged = (base + delta * SCALE).to(tl.bfloat16)
        xx = tl.load(
            X + mi[:, None] * K + kk[None, :],
            (mi[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        acc = tl.dot(xx, tl.trans(merged), acc)
    if HAS_BIAS:
        acc += tl.load(Bias + ni, ni < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N)
    )


def tiled_bf16_linear(
    x, weight, a, b, rank, scaling, bias=None, *, delta_precision="ieee"
):
    """Compute X @ BF16(W + scale * B @ A).T, using tile-local storage only.

    A must be the active packed slice; B may retain a larger pool rank stride.
    The caller handles packed projection slices and token-to-adapter routing.
    The IEEE path is the accuracy control. TF32x3 is an explicit experiment,
    never a silent substitute. This function does not alter any input tensor.
    """
    if delta_precision not in {"ieee", "tf32x3"}:
        raise ValueError("delta_precision must be ieee or tf32x3")
    if not (x.ndim == weight.ndim == a.ndim == b.ndim == 2):
        raise ValueError("expected two-dimensional activation, weight and factors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("tiled BF16 requires BF16 activations and base weights")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("original FP32 low-rank factors must be preserved")
    if not (x.is_cuda and x.device == weight.device == a.device == b.device):
        raise ValueError("all tensors must share a CUDA device")
    if not all(t.is_contiguous() for t in (x, weight, a, b)):
        raise ValueError("contiguous inputs are required")
    if not isinstance(rank, int) or not 1 <= rank <= b.shape[1] <= 64:
        raise ValueError("expected 1 <= active rank <= pool rank <= 64")
    if not math.isfinite(scaling):
        raise ValueError("scaling must be finite")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk or a.shape != (rank, k) or b.shape[0] != n:
        raise ValueError("incompatible projection and factor shapes")
    if bias is not None and (
        bias.shape != (n,)
        or bias.device != x.device
        or bias.dtype != torch.bfloat16
        or not bias.is_contiguous()
    ):
        raise ValueError("expected a contiguous BF16 bias on the same device")
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0 or n == 0:
        return output
    _tiled_bf16_linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x,
        weight,
        a,
        b,
        output,
        bias,
        m,
        n,
        k,
        rank,
        b.shape[1],
        scaling,
        bias is not None,
        delta_precision,
        BM=16,
        BN=64,
        BK=64,
        BR=max(16, triton.next_power_of_2(rank)),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _grouped_tiled_bf16_linear(
    X,
    W,
    A,
    B,
    Y,
    Bias,
    Rows,
    Tiles,
    Scales,
    Present,
    N: tl.constexpr,
    K: tl.constexpr,
    Y_WIDTH: tl.constexpr,
    A_SLOT_STRIDE: tl.constexpr,
    B_SLOT_STRIDE: tl.constexpr,
    B_RANK_STRIDE: tl.constexpr,
    PACKED_INDEX: tl.constexpr,
    N_START: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PRESENCE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    tile = tl.program_id(0)
    slot = tl.load(Tiles + tile * 4)
    rank = tl.load(Tiles + tile * 4 + 1)
    row_start = tl.load(Tiles + tile * 4 + 2)
    row_count = tl.load(Tiles + tile * 4 + 3)
    scale = tl.load(Scales + tile)
    if HAS_PRESENCE:
        rank = tl.where(tl.load(Present + slot) != 0, rank, 0)
    mi = tl.arange(0, BM)
    rows = tl.load(Rows + row_start + mi, mi < row_count, other=0)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ri = tl.arange(0, BR)
    bb = tl.load(
        B
        + slot * B_SLOT_STRIDE
        + (N_START + ni[:, None]) * B_RANK_STRIDE
        + ri[None, :],
        (ni[:, None] < N) & (ri[None, :] < rank),
        other=0,
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        kk = ki * BK + tl.arange(0, BK)
        base = tl.load(
            W + ni[:, None] * K + kk[None, :],
            (ni[:, None] < N) & (kk[None, :] < K),
            other=0,
        ).to(tl.float32)
        if rank > 0:
            aa = tl.load(
                A
                + slot * A_SLOT_STRIDE
                + (PACKED_INDEX * rank + ri[:, None]) * K
                + kk[None, :],
                (ri[:, None] < rank) & (kk[None, :] < K),
                other=0,
            )
            delta = tl.dot(bb, aa, input_precision="ieee")
            merged = (base + delta * scale).to(tl.bfloat16)
        else:
            merged = base.to(tl.bfloat16)
        xx = tl.load(
            X + rows[:, None] * K + kk[None, :],
            (mi[:, None] < row_count) & (kk[None, :] < K),
            other=0,
        )
        acc = tl.dot(xx, tl.trans(merged), acc)
    if HAS_BIAS:
        acc += tl.load(Bias + ni, ni < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * Y_WIDTH + N_START + ni[None, :],
        acc,
        (mi[:, None] < row_count) & (ni[None, :] < N),
    )


def grouped_tiled_bf16_linear(
    x,
    weight,
    a,
    b,
    output,
    rows,
    tiles,
    scales,
    packed_index,
    start,
    bias=None,
    present=None,
):
    """Write one packed projection for all prevalidated routing tiles.

    Tiles contain (slot, active rank, row offset, row count), with at most 16
    rows each. They and the row permutation are prepared once per forward
    batch, shared by every layer. A uses packed active-rank slices, B the pool
    rank stride. Only the output is written; no merged weight is allocated.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("grouped tiled BF16 requires BF16 activations and weights")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("original FP32 low-rank factors must be preserved")
    if not (x.ndim == weight.ndim == output.ndim == 2 and a.ndim == b.ndim == 3):
        raise ValueError("expected matrix inputs/output and three-dimensional pools")
    tensors = (x, weight, a, b, output, rows, tiles, scales)
    if not all(
        t.is_cuda and t.device == x.device and t.is_contiguous() for t in tensors
    ):
        raise ValueError("contiguous tensors on one CUDA device are required")
    if (
        rows.dtype != torch.int64
        or tiles.dtype != torch.int32
        or scales.dtype != torch.float32
    ):
        raise ValueError("invalid grouped routing metadata dtypes")
    if (
        rows.ndim != 1
        or tiles.ndim != 2
        or tiles.shape[1] != 4
        or scales.shape != (tiles.shape[0],)
    ):
        raise ValueError("invalid grouped routing metadata shapes")
    n, k = weight.shape
    if (
        x.shape[1] != k
        or output.shape[0] != x.shape[0]
        or output.dtype != torch.bfloat16
        or not 0 <= start <= output.shape[1] - n
        or a.shape[0] != b.shape[0]
        or a.shape[2] != k
        or b.shape[1] < start + n
        or not 1 <= b.shape[2] <= 64
        or not 0 <= packed_index < a.shape[1] // b.shape[2]
    ):
        raise ValueError("incompatible grouped projection and factor pools")
    if present is not None and (
        present.shape != (a.shape[0],)
        or present.dtype != torch.int32
        or present.device != x.device
        or not present.is_contiguous()
    ):
        raise ValueError("expected one presence flag per factor-pool slot")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != torch.bfloat16
        or bias.device != x.device
        or not bias.is_contiguous()
    ):
        raise ValueError("expected a contiguous BF16 bias on the same device")
    if not tiles.shape[0] or not n:
        return
    _grouped_tiled_bf16_linear[(tiles.shape[0], triton.cdiv(n, 64))](
        x,
        weight,
        a,
        b,
        output,
        bias,
        rows,
        tiles,
        scales,
        present,
        n,
        k,
        output.shape[1],
        a.stride(0),
        b.stride(0),
        b.shape[2],
        packed_index,
        start,
        bias is not None,
        present is not None,
        BM=16,
        BN=64,
        BK=64,
        BR=max(16, triton.next_power_of_2(b.shape[2])),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
