# mapping on device memory, host memory and memory allocator

import logging
from typing import Optional

import torch

from sglang.srt.layers.dcp.comm import (
    dcp_disabled as _dcp_disabled,
    dcp_enabled as _dcp_enabled,
    get_attention_dcp_rank as _get_attention_dcp_rank,
    get_attention_dcp_world_size as _get_attention_dcp_world_size,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
        indexer_types: Optional[list] = None,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
            indexer_types=indexer_types,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        mapping = getattr(self, "full_to_hisparse_device_index_mapping", None)
        if mapping is None:
            # Draft worker path: HiSparseTokenToKVPoolAllocator not created
            # (allocator reused from target), so no mapping registered.
            # Identity fallback: device index == logical index.
            return compressed_indices
        # DCP-aware: callers hand us VIRTUAL ids (req_to_token domain,
        # capacity=size*dcp). Convert to the local full domain with the
        # ownership filter first — the exact formula used by
        # memory_pool._write_mla_kv_buffer's DCP branch (9db63a6abb). Foreign
        # ids resolve to 0 (the unmapped sentinel) instead of leaking another
        # request's device slot through a stale mapping entry.
        if _dcp_enabled():
            _dws = _get_attention_dcp_world_size()
            _ps = self.page_size  # 64
            _slot = (compressed_indices // (_ps * _dws)) * _ps + (
                compressed_indices % _ps
            )
            _valid = compressed_indices >= 0
            _owner = (
                (compressed_indices // _ps) % _dws == _get_attention_dcp_rank()
            ) & _valid
            local = torch.where(_owner, _slot, torch.zeros_like(compressed_indices))
            mapped = mapping[local]
            # mapping[0] may hold a live slot; force foreign lanes to 0.
            return torch.where(_owner, mapped, torch.zeros_like(mapped))
        return mapping[compressed_indices]

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        mapping = getattr(self, "full_to_hisparse_device_index_mapping", None)
        if mapping is None:
            return compressed_indices
        return mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        forward_mode=None,
    ):
        if _dcp_enabled():
            # DCP mode: decode-generated KV goes directly to the device KV buffer
            # via the standard DCP virtual->local translate (super's path).
            # The hisparse device mapping is only for coordinator-managed
            # swap-in of PD-transferred KV (prefill's KV), NOT for decode's
            # own newly-generated tokens. Skipping hisparse translate here
            # fixes the "mapping all-zero -> write to slot 0 -> garbage -> EOS"
            # bug that produced only 1 token then stopped.
            super().set_mla_kv_buffer(
                layer, loc, cache_k_nope, cache_k_rope, forward_mode
            )
        else:
            # Non-DCP: hisparse translate (virtual -> hisparse device window)
            loc = self.translate_loc_to_hisparse_device(loc)
            super().set_mla_kv_buffer(
                layer, loc, cache_k_nope, cache_k_rope, forward_mode
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        if _dcp_enabled():
            return super().get_mla_kv_buffer(layer, loc, dst_dtype)
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")
