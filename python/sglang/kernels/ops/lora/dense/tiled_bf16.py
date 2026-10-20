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
