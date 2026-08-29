#pragma once

#include "hicache.cuh"
#include <limits>

namespace {

struct HicacheRelayoutParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_src;
  const void* __restrict__ k_ptr_src;
  const void* __restrict__ v_ptr_src;
  uint32_t num_pages;
  uint32_t num_layers;
  uint32_t page_size;
};

template <typename IndexType, int64_t kElementSize, bool kIsMLA>
__global__ void hicache_relayout_kernel(const __grid_constant__ HicacheRelayoutParams params) {
  using namespace device;
  using pack_t = uint4;
  static_assert(kElementSize % 16 == 0, "hicache_relayout_kernel requires 16-byte aligned element size");
  constexpr uint32_t kVecBytes = 16;
  constexpr uint32_t kVecPerItem = kElementSize / kVecBytes;

  const auto& [k_cache_dst, v_cache_dst, indices_src, k_ptr_src, v_ptr_src, num_pages, num_layers, page_size] = params;
  const auto k_ptr_src_arr = static_cast<const void* const*>(k_ptr_src);
  const auto v_ptr_src_arr = static_cast<const void* const*>(v_ptr_src);
  const auto tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const auto total_vecs = static_cast<uint64_t>(num_pages) * page_size * num_layers * kVecPerItem;

  for (uint64_t linear_vec_id = tid; linear_vec_id < total_vecs; linear_vec_id += stride) {
    const auto page_id =
        static_cast<uint32_t>(linear_vec_id / (static_cast<uint64_t>(page_size) * num_layers * kVecPerItem));
    const auto page_vec_id =
        static_cast<uint32_t>(linear_vec_id % (static_cast<uint64_t>(page_size) * num_layers * kVecPerItem));
    const auto token_in_page = page_vec_id / (num_layers * kVecPerItem);
    const auto token_vec_id = page_vec_id % (num_layers * kVecPerItem);
    const auto layer_id = token_vec_id / kVecPerItem;
    const auto vec_id = token_vec_id % kVecPerItem;
    const auto src_page = static_cast<uint32_t>(static_cast<const IndexType*>(indices_src)[page_id]);
    const auto src_token = src_page + token_in_page;
    // [RELAYOUT-BOUNDS] Xid31 probe: device-side OOB check on src offset.
    const int64_t _src_off = static_cast<int64_t>(src_token) * kElementSize + static_cast<int64_t>(vec_id) * kVecBytes;
    const int64_t _dst_off = static_cast<int64_t>(linear_vec_id) * kVecBytes;
    const uint64_t _src_max = static_cast<uint64_t>(num_pages) * page_size * kElementSize;
    const uint64_t _dst_max = static_cast<uint64_t>(num_pages) * page_size * num_layers * kElementSize;
    if (static_cast<uint64_t>(_src_off + kVecBytes) > _src_max ||
        static_cast<uint64_t>(_dst_off + kVecBytes) > _dst_max) {
      printf("[RELAYOUT-OOB] layer=%u page_id=%u src_page=%u tok=%u vec=%u src_off=%lld dst_off=%lld elem=%lld ps=%u np=%u nl=%u\\n",
             layer_id, page_id, src_page, token_in_page, vec_id,
             static_cast<long long>(_src_off), static_cast<long long>(_dst_off),
             static_cast<long long>(kElementSize), page_size, num_pages, num_layers);
      continue;
    }
    const auto src_k = pointer::offset(
        static_cast<const void*>(k_ptr_src_arr[layer_id]),
        static_cast<int64_t>(src_token) * kElementSize + static_cast<int64_t>(vec_id) * kVecBytes);
    const auto dst_k =
        pointer::offset(static_cast<void*>(k_cache_dst), static_cast<int64_t>(linear_vec_id) * kVecBytes);
    const auto vec_k = details::load_nc(reinterpret_cast<const pack_t*>(src_k));
    details::store_nc(reinterpret_cast<pack_t*>(dst_k), vec_k);

    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(
          static_cast<const void*>(v_ptr_src_arr[layer_id]),
          static_cast<int64_t>(src_token) * kElementSize + static_cast<int64_t>(vec_id) * kVecBytes);
      const auto dst_v =
          pointer::offset(static_cast<void*>(v_cache_dst), static_cast<int64_t>(linear_vec_id) * kVecBytes);
      const auto vec_v = details::load_nc(reinterpret_cast<const pack_t*>(src_v));
      details::store_nc(reinterpret_cast<pack_t*>(dst_v), vec_v);
    }
  }
}

template <int64_t kElementSize, bool kIsMLA>
inline void launch_hicache_relayout_kernel(
    const HicacheRelayoutParams& params,
    int64_t num_pages,
    int64_t num_layers,
    int64_t page_size,
    bool use_int32,
    DLDevice device) {
  using namespace host;

  constexpr uint32_t kRelayoutBlockSize = 256;
  constexpr uint32_t kVecPerItem = kElementSize / 16;
  const auto total_vecs = static_cast<uint64_t>(num_pages) * page_size * num_layers * kVecPerItem;
  const auto kernel = use_int32 ? hicache_relayout_kernel<int32_t, kElementSize, kIsMLA>
                                : hicache_relayout_kernel<int64_t, kElementSize, kIsMLA>;
  if (total_vecs == 0) {
    return;
  }

  const auto grid = div_ceil(total_vecs, static_cast<uint64_t>(kRelayoutBlockSize));
  RuntimeCheck(
      grid <= std::numeric_limits<uint32_t>::max(), "HiCache staged relayout: CUDA grid size exceeds uint32 range");
  LaunchKernel(static_cast<uint32_t>(grid), kRelayoutBlockSize, device)(kernel, params);
}

}  // namespace
