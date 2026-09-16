// Peer-to-peer (NVLink) all-reduce for bf16 tensors — single node, TP group.
//
// Protocol mirrors Ferrite's `ferrite_p2p_ar_v5` (store / publish / reduce with
// an epoch double-buffer and a per-rank ready table); the difference is the
// entry points here take bf16 input/output so we can drop-in replace the NCCL
// `AllReduce_Sum_bf16_RING_LL` that costs ~10% of a decode step.
//
// Numerics: the partials are bf16 (torch's dist.all_reduce sees bf16 partials),
// so this kernel converts bf16 -> fp32 for the staging store, sums the `world`
// partials in fp32 in ascending-rank order, and rounds the result back to bf16
// on write-out. That is exactly the reference semantics (bf16 partials, fp32
// accumulation, bf16 result).
//
// Layout (all owned by the caller, see python .../p2p_all_reduce.py):
//   staging_local : float32 [2][world][stride]   (double buffered by epoch&1)
//   ready_local   : uint32  [world]              (peers stamp their readiness here)
//   staging_tbl   : float*  [world]              (peer staging bases, device array)
//   ready_tbl     : uint32* [world]              (peer ready rows, device array)
//   epoch         : uint32* (1)                  (this rank's round counter)
//
// Two launches, same stream:
//   1) store : grid (blocks, world) — one block row per peer; every thread does
//              exactly one remote float4 store, so the `world` peer writes issue
//              concurrently instead of serially.
//   2) pubred: publish my readiness into every peer's ready row, then poll all
//              peers' stamps, then sum the staging rows (ascending rank) into out.
// `blockDim.x >= world` is required by the publish/poll arm (thread r handles
// peer r), hence 64 threads for world <= 64.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static inline int p2p_ar_block_threads(int world) {
    if (world <= 64) return 64;
    return (world > 256) ? 1024 : 256;
}

static inline int p2p_ar_grid_blocks(int n, int threads) {
    // grid on the float4 count of the *bf16* input: 8 bf16 per 16B
    const int n8 = (n + 7) >> 3;
    int b = (n8 + threads - 1) / threads;
    return b < 1 ? 1 : b;
}

// ---------------------------------------------------------------------------
// store: write MY partial into EVERY peer's staging slot for this epoch.
// ---------------------------------------------------------------------------
__global__ void p2p_ar_store_bf16_kernel(
    const __nv_bfloat16* __restrict__ partial,
    float* const* __restrict__ staging_tbl,   // [world] peer staging bases
    const unsigned* __restrict__ epoch,       // this rank's round counter
    int world, int my_rank, int n, int stride) {
    const unsigned e = *epoch;
    const int rr = blockIdx.y;                // this block's peer (gridDim.y == world)
    const int step = gridDim.x * blockDim.x;
    const int n8 = (n + 7) >> 3;              // 8 bf16 per 16B
    const int n4 = n >> 2;                    // float4 count of the fp32 staging
    const size_t base_row =
        (size_t)((e & 1u) * (unsigned)world + (unsigned)my_rank) * (unsigned)stride;

    // head: 8 bf16 -> 8 float (two float4 stores)
    for (int i8 = blockIdx.x * blockDim.x + threadIdx.x; i8 < (n8 & ~1); i8 += step) {
        const uint4 h = reinterpret_cast<const uint4*>(partial)[i8];
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&h);
        float4 lo, hi;
        lo.x = __bfloat162float(b2[0].x); lo.y = __bfloat162float(b2[0].y);
        lo.z = __bfloat162float(b2[1].x); lo.w = __bfloat162float(b2[1].y);
        hi.x = __bfloat162float(b2[2].x); hi.y = __bfloat162float(b2[2].y);
        hi.z = __bfloat162float(b2[3].x); hi.w = __bfloat162float(b2[3].y);
        float* dst = staging_tbl[rr] + base_row;
        reinterpret_cast<float4*>(dst)[i8 * 2 + 0] = lo;
        reinterpret_cast<float4*>(dst)[i8 * 2 + 1] = hi;
    }
    // tail: whole float4 words not covered by the 8-wide loop
    for (int i4 = (n8 & ~1) * 2 + blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += step) {
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(partial) + i4 * 2;
        float4 v;
        v.x = __bfloat162float(h2[0].x); v.y = __bfloat162float(h2[0].y);
        v.z = __bfloat162float(h2[1].x); v.w = __bfloat162float(h2[1].y);
        reinterpret_cast<float4*>(staging_tbl[rr] + base_row)[i4] = v;
    }
    // scalar tail (n % 4)
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        staging_tbl[rr][base_row + i] = __bfloat162float(partial[i]);
    }
    __threadfence_system();   // peer-visible stores before the flag (next launch)
}

// ---------------------------------------------------------------------------
// pubred: publish readiness to every peer, poll peers' stamps, sum into out.
// ---------------------------------------------------------------------------
__global__ void p2p_ar_pubred_bf16_kernel(
    unsigned* const* __restrict__ ready_tbl,   // [world] peer ready rows
    const unsigned* __restrict__ epoch,
    const float* __restrict__ staging_local,   // [2][world][stride]
    const unsigned* __restrict__ ready_local,  // [world] my ready row
    __nv_bfloat16* __restrict__ out,
    int world, int my_rank, int n, int stride) {
    const unsigned e = *epoch;
    const unsigned want = e + 1u;

    // publish: thread r stamps peer r's ready row at index my_rank
    if (threadIdx.x < (unsigned)world) {
        ready_tbl[threadIdx.x][my_rank] = want;
    }
    __threadfence_system();
    __syncthreads();

    // poll: every block waits for all peers (cheap; peers are within ~10us)
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++) {
            while (*(volatile const unsigned*)&ready_local[r] != want) __nanosleep(200);
        }
    }
    __syncthreads();

    const float* my_rows = staging_local + (size_t)(e & 1u) * (unsigned)world * (unsigned)stride;
    const int step = gridDim.x * blockDim.x;
    const int n4 = n >> 2;
    for (int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += step) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r = 0; r < world; r++) {
            const float4 v = reinterpret_cast<const float4*>(my_rows + (size_t)r * stride)[i4];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        __nv_bfloat162 o0, o1;
        o0.x = __float2bfloat16(acc.x); o0.y = __float2bfloat16(acc.y);
        o1.x = __float2bfloat16(acc.z); o1.y = __float2bfloat16(acc.w);
        __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out) + i4 * 2;
        o[0] = o0;
        o[1] = o1;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += step) {
        float acc = 0.f;
        for (int r = 0; r < world; r++) acc += my_rows[(size_t)r * stride + i];
        out[i] = __float2bfloat16(acc);
    }
    // advance the round counter (stream-ordered with the next launch)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *const_cast<unsigned*>(epoch) = want;
    }
}

extern "C" cudaError_t p2p_ar_bf16(
    const __nv_bfloat16* partial, float* const* staging_tbl,
    unsigned* const* ready_tbl, unsigned* epoch,
    const float* staging_local, const unsigned* ready_local,
    __nv_bfloat16* out, int n, int world, int my_rank, int stride,
    cudaStream_t s) {
    const int threads = p2p_ar_block_threads(world);
    const int blocks = p2p_ar_grid_blocks(n, threads);
    p2p_ar_store_bf16_kernel<<<dim3(blocks, world, 1), threads, 0, s>>>(
        partial, staging_tbl, epoch, world, my_rank, n, stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    p2p_ar_pubred_bf16_kernel<<<blocks, threads, 0, s>>>(
        ready_tbl, epoch, staging_local, ready_local, out, world, my_rank, n, stride);
    return cudaGetLastError();
}

// In-place variant (sglang's TP AR is in-place): out == partial is allowed only
// if the caller keeps a separate input buffer; here out is the caller's tensor.
extern "C" cudaError_t p2p_ar_bf16_inplace(
    __nv_bfloat16* buf, float* const* staging_tbl, unsigned* const* ready_tbl,
    unsigned* epoch, const float* staging_local, const unsigned* ready_local,
    int n, int world, int my_rank, int stride, cudaStream_t s) {
    return p2p_ar_bf16(buf, staging_tbl, ready_tbl, epoch, staging_local,
                        ready_local, buf, n, world, my_rank, stride, s);
}
