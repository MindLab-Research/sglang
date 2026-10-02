// Fused gated-SwiGLU + LoRA activation -> MXFP8 (E4m3 + UE8M0 32-block) quant
// for the E4m3/MXFP4 MoE-LoRA path (DeepSeek-V4.1).
//
// PRECISION FIX (zero-LoRA == base): the previous unfused chain materialized
// the activated row as bf16 (activationKernel -> activated_bf16 ->
// invokeMxFP8Quantization) — one extra RN rounding the fused kernel's GEMM1
// epilogue never performs. The bf16 perturbation shifts each 32-block's amax,
// flipping UE8M0 exponents at the PosInf rounding boundaries — the dominant
// residual noise between zero-LoRA and base. This kernel keeps the activated
// values in fp32 registers and quantizes them DIRECTLY with the exact
// cvt_warp_fp16_to_mxfp8 recipe (quantization_utils.cuh): fp32 amax,
// SFValue = amax * reciprocal(448), UE8M0 via
// __nv_cvt_float_to_e8m0(..., cudaRoundPosInf), outputScale = 1/SF,
// +/-448 saturation, RN E4m3, swizzled SF via get_sf_out_offset_128x4/8x4.
//
// The gated-SwiGLU (incl. the OAI alpha/beta/clamp scalars and the LoRA-delta
// placement) replicates moe::dev::activation::gatedSiluScalars
// element-for-element (xLinear = even column + delta[upper half],
// xGlu = odd column + delta[lower half]), so each fp32 act value is bitwise
// equal to the unfused activation's PRE-rounding value.
// activation_lora_input keeps the bf16 contract (LoRA-side bridge only — it
// never feeds the base numerics).
//
// Layout: one block per expanded row (grid = num_tokens*top_k), one thread per
// 32-element SF block (BLOCK_SIZE must cover innerHalf/32). Requires
// innerHalf % 32 == 0 and innerHalf/32 <= BLOCK_SIZE. All global accesses are
// 16B-aligned (row strides innerDim/innerHalf are multiples of 16 bytes for
// the supported models). Padding rows (permutedIdx == -1) write zeros to
// activation_lora_input and SKIP the quant outputs — those buffer rows stay
// uninitialized, never read into a valid output (same proven precedent as the
// NvFP4 fused kernel).
#pragma once

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace flashinfer {
namespace sgl_fused_act_mxfp8_quant {

namespace tk = tensorrt_llm::kernels;

// Replicates gatedSiluScalars (trtllm_fused_moe_dev_kernel.cu) with hoisted
// scalars: same clamp / sigmoid / beta instruction sequence per element, so
// the fp32 value matches the unfused kernel bitwise (pre-rounding). With all
// scalars at their identity defaults this is silu(xGlu) * xLinear exactly.
inline __device__ float fusedGatedSilu(float alpha, float beta, bool hasClamp,
                                       float limit, float xLinear, float xGlu) {
  if (hasClamp) {
    xGlu = fminf(xGlu, limit);
    xLinear = fmaxf(fminf(xLinear, limit), -limit);
  }
  // x * sigmoid(alpha * x), i.e. silu(x) generalized by alpha.
  float const act = xGlu / (1.0f + expf(-alpha * xGlu));
  return act * (xLinear + beta);
}

template <uint32_t BLOCK_SIZE, bool SF_128x4>
__global__ void fusedActMxfp8QuantKernel(
    int m,         // num expanded rows (num_tokens * top_k)
    int innerHalf, // inter; multiple of 32; innerHalf/32 <= BLOCK_SIZE
    int innerDim,  // 2 * inter
    __nv_bfloat16 const *__restrict__ gateUp, // [.., innerDim] interleaved
                                               // (xLinear even, xGlu odd), by
                                               // permutedIdx
    __nv_bfloat16 const *__restrict__ loraDelta, // [.., innerDim] contiguous
                                                  // [xGlu | xLinear], by
                                                  // expandedIdx
    __nv_bfloat16 *__restrict__ loraInputOut,    // [.., innerHalf] by
                                                 // expandedIdx
    int32_t const *__restrict__ expandedIdxToPermutedIdx,
    float const *__restrict__ alphaPtr, float const *__restrict__ betaPtr,
    float const *__restrict__ clampLimitPtr,
    uint8_t *__restrict__ weightOutput, // e4m3 [.., innerHalf] by permutedIdx
    uint8_t *__restrict__ scaleOutput) { // swizzled UE8M0 SF
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int SF_VEC = 32;
  int const expandedIdx = blockIdx.x;
  if (expandedIdx >= m) {
    return;
  }
  int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
  int const numBlks = innerHalf / SF_VEC;
  int const blk = threadIdx.x;
  int const h0 = blk * SF_VEC;

  if (permutedIdx < 0) {
    // Padding row: zeros to the LoRA-side bridge (mirrors the unfused
    // activation kernel); quant outputs skipped — padding rows never reach a
    // valid output.
    if (blk < numBlks && loraInputOut != nullptr) {
      int64_t const liBase = (int64_t)expandedIdx * innerHalf + h0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        *reinterpret_cast<int4 *>(&loraInputOut[liBase + j * 8]) =
            make_int4(0, 0, 0, 0);
      }
    }
    return;
  }
  if (blk >= numBlks) {
    return;
  }

  // Hoisted OAI scalars (pure loads — identical values to the per-element
  // reads in gatedSiluScalars).
  float const alpha = alphaPtr != nullptr ? alphaPtr[0] : 1.0f;
  float const beta = betaPtr != nullptr ? betaPtr[0] : 0.0f;
  bool const hasClamp = clampLimitPtr != nullptr;
  float const limit = hasClamp ? clampLimitPtr[0] : 0.0f;

  int64_t const permBase = (int64_t)permutedIdx * innerDim;
  int64_t const expBase = (int64_t)expandedIdx * innerDim;
  int64_t const liBase = (int64_t)expandedIdx * innerHalf + h0;
  int64_t const wBase = (int64_t)permutedIdx * innerHalf + h0;

  // ---- load 32 interleaved (xLinear, xGlu) bf16 pairs: 8 x int4 ----
  union {
    int4 v[8];
    __nv_bfloat16 b[64];
  } gu;
  {
    int4 const *gp =
        reinterpret_cast<int4 const *>(gateUp + permBase + 2 * h0);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      gu.v[j] = gp[j];
    }
  }

  // ---- compute the fp32 act values (never materialized to bf16) ----
  float act[SF_VEC];
  float amax = 0.0f;
  if (loraDelta != nullptr) {
    // delta layout per expandedIdx row: xGlu half at [h], xLinear half at
    // [innerHalf + h] (activationKernel / activationKernelOpt).
    union {
      int4 v[4];
      __nv_bfloat16 b[32];
    } dGlu, dLin;
    {
      int4 const *gluP =
          reinterpret_cast<int4 const *>(loraDelta + expBase + h0);
      int4 const *linP =
          reinterpret_cast<int4 const *>(loraDelta + expBase + innerHalf + h0);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        dGlu.v[j] = gluP[j];
        dLin.v[j] = linP[j];
      }
    }
#pragma unroll
    for (int j = 0; j < SF_VEC; ++j) {
      float const xLinear = (float)gu.b[2 * j] + (float)dLin.b[j];
      float const xGlu = (float)gu.b[2 * j + 1] + (float)dGlu.b[j];
      act[j] = fusedGatedSilu(alpha, beta, hasClamp, limit, xLinear, xGlu);
      amax = fmaxf(amax, fabsf(act[j]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < SF_VEC; ++j) {
      float const xLinear = (float)gu.b[2 * j];
      float const xGlu = (float)gu.b[2 * j + 1];
      act[j] = fusedGatedSilu(alpha, beta, hasClamp, limit, xLinear, xGlu);
      amax = fmaxf(amax, fabsf(act[j]));
    }
  }

  // ---- LoRA-side bridge: bf16(act) by expandedIdx (contract unchanged) ----
  if (loraInputOut != nullptr) {
    union {
      int4 v[4];
      __nv_bfloat16 b[32];
    } li;
#pragma unroll
    for (int j = 0; j < SF_VEC; ++j) {
      li.b[j] = __float2bfloat16_rn(act[j]);
    }
    int4 *lp = reinterpret_cast<int4 *>(loraInputOut + liBase);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      lp[j] = li.v[j];
    }
  }

  // ---- MXFP8 quant, exact cvt_warp_fp16_to_mxfp8 recipe, fp32 inputs ----
  float SFValue = amax * tk::reciprocal_approximate_ftz(448.0f);
  __nv_fp8_e8m0 tmpSFVal;
  tmpSFVal.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE,
                                        cudaRoundPosInf);
  SFValue = static_cast<float>(tmpSFVal);
  // Check SFValue != 0 (not amax != 0): the E8M0 conversion can underflow very
  // small amax values to zero; the reciprocal of 0 is inf and would NaN the
  // denormal inputs (same guard as the stock kernel).
  float const outputScale =
      SFValue != 0.0f ? tk::reciprocal_approximate_ftz(SFValue) : 0.0f;

  union {
    int4 v[2];
    uint8_t b[32];
  } q;
#pragma unroll
  for (int j = 0; j < SF_VEC / 2; ++j) {
    float2 const f2 = make_float2(act[2 * j] * outputScale,
                                  act[2 * j + 1] * outputScale);
    // Saturate to +/-E4M3 max so inf/overflow map to the max finite value
    // instead of NaN (matches torch's saturating float8_e4m3fn cast and the
    // stock kernel's clamp). cvt_float2_to_fp8x2 with __NV_SATFINITE is the
    // same elementwise RN conversion fp32_vec_to_e4m3 uses.
    float2 const sat = make_float2(fmaxf(fminf(f2.x, 448.0f), -448.0f),
                                  fmaxf(fminf(f2.y, 448.0f), -448.0f));
    uint16_t const packed =
        __nv_cvt_float2_to_fp8x2(sat, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint16_t *>(&q.b[2 * j]) = packed;
  }
  {
    int4 *wp = reinterpret_cast<int4 *>(weightOutput + wBase);
    wp[0] = q.v[0];
    wp[1] = q.v[1];
  }

  // ---- swizzled SF byte (R128c4 / R8c4 — the layout the down GEMM reads) --
  int64_t const sfOffset =
      SF_128x4 ? tk::get_sf_out_offset_128x4(0, permutedIdx, blk, m, numBlks)
               : tk::get_sf_out_offset_8x4(0, permutedIdx, blk, m, numBlks);
  scaleOutput[sfOffset] = tmpSFVal.__x;
#else
  (void)m;
  (void)innerHalf;
  (void)innerDim;
  (void)gateUp;
  (void)loraDelta;
  (void)loraInputOut;
  (void)expandedIdxToPermutedIdx;
  (void)alphaPtr;
  (void)betaPtr;
  (void)clampLimitPtr;
  (void)weightOutput;
  (void)scaleOutput;
#endif
}

// Host launch. Callers must guarantee inter % 32 == 0 and inter / 32 <= 128.
inline void launchFusedActMxfp8Quant(
    int m, int innerHalf, int innerDim, __nv_bfloat16 const *gateUp,
    __nv_bfloat16 const *loraDelta, __nv_bfloat16 *loraInputOut,
    int32_t const *expandedIdxToPermutedIdx, float const *alphaPtr,
    float const *betaPtr, float const *clampLimitPtr, uint8_t *weightOutput,
    uint8_t *scaleOutput, tensorrt_llm::QuantizationSFLayout sfLayout,
    cudaStream_t stream) {
  dim3 const grid(static_cast<uint32_t>(m));
  dim3 const block(128);
  if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    fusedActMxfp8QuantKernel<128, true><<<grid, block, 0, stream>>>(
        m, innerHalf, innerDim, gateUp, loraDelta, loraInputOut,
        expandedIdxToPermutedIdx, alphaPtr, betaPtr, clampLimitPtr,
        weightOutput, scaleOutput);
  } else {
    fusedActMxfp8QuantKernel<128, false><<<grid, block, 0, stream>>>(
        m, innerHalf, innerDim, gateUp, loraDelta, loraInputOut,
        expandedIdxToPermutedIdx, alphaPtr, betaPtr, clampLimitPtr,
        weightOutput, scaleOutput);
  }
}

} // namespace sgl_fused_act_mxfp8_quant
} // namespace flashinfer
