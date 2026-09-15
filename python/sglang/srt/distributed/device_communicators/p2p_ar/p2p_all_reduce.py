"""Peer-to-peer NVLink all-reduce for bf16 TP all-reduce (single node).

Drop-in replacement for the NCCL bf16 TP all-reduce that costs ~10% of a decode
step (profile: `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` 10.7%).  The kernel is
`p2p_ar_bf16.cu` in this directory: it mirrors Ferrite's `ferrite_p2p_ar_v5`
protocol (store -> publish -> pubred with an epoch double-buffer and a per-rank
ready table) but takes bf16 input/output, so the swapping is transparent to the
rest of the model.

Enable with:
    SGLANG_P2P_AR=1                      # flag (default: off)
    SGLANG_P2P_AR_LIB=/path/libp2p_ar.so # built by ./build.sh

Safety: everything is preallocated at init (graph-capture safe, no allocation on
the hot path), and `should_p2p_ar()` gates on dtype/size/world so anything
unsupported silently falls back to the existing (NCCL) path.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# staging is [2][world][stride] fp32 -> allow ~64 MB per rank by default
_DEFAULT_MAX_ELEMS = int(os.environ.get("SGLANG_P2P_AR_MAX_ELEMS", 8 * 1024 * 1024))
_IPC_HANDLE_BYTES = 64  # cudaIpcMemHandle_t


def p2p_ar_enabled() -> bool:
    return os.environ.get("SGLANG_P2P_AR", "0") == "1"


class P2PAllReduce:
    """Owns the staging/ready/epoch buffers and the peer pointer tables."""

    def __init__(self, group, device: torch.device, max_elems: Optional[int] = None):
        self.group = group
        self.device = device
        self.world = torch.distributed.get_world_size(group=group)
        self.rank = torch.distributed.get_rank(group=group)
        self.disabled = True
        self.max_elems = int(max_elems or _DEFAULT_MAX_ELEMS)
        # stride is in fp32 elements and must be a multiple of 4 (float4 stores)
        self.stride = (self.max_elems + 3) // 4 * 4

        lib_path = os.environ.get("SGLANG_P2P_AR_LIB", "")
        if not lib_path or not os.path.isfile(lib_path):
            logger.warning("p2p_ar: SGLANG_P2P_AR_LIB not set/found (%r); disabled", lib_path)
            return
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:  # pragma: no cover - configuration error
            logger.warning("p2p_ar: cannot load %s: %s; disabled", lib_path, e)
            return
        self.lib.p2p_ar_bf16.restype = ctypes.c_int
        self.lib.p2p_ar_bf16.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self._cudart = ctypes.CDLL("libcudart.so")

        if self.world < 2 or self.world > 8:
            logger.info("p2p_ar: world=%d not in [2,8]; disabled", self.world)
            return
        try:
            self._init_buffers()
        except Exception as e:  # pragma: no cover - any IPC/IPC-open failure
            logger.warning("p2p_ar: init failed (%s); disabled", e)
            return
        self.disabled = False
        logger.info(
            "p2p_ar: enabled (world=%d rank=%d stride=%d elems, lib=%s)",
            self.world, self.rank, self.stride, lib_path,
        )

    # ------------------------------------------------------------------ setup
    def _init_buffers(self):
        dev = self.device
        # One flat byte blob per rank so a single IPC handle exposes everything.
        self.staging = torch.zeros(2 * self.world * self.stride, dtype=torch.float32, device=dev)
        self.ready = torch.zeros(self.world, dtype=torch.uint32, device=dev)
        self.epoch = torch.zeros(1, dtype=torch.uint32, device=dev)
        self._blob = torch.empty(
            self.staging.numel() * 4 + self.ready.numel() * 4 + self.epoch.numel() * 4,
            dtype=torch.uint8, device=dev,
        )
        # carve the blob: staging | ready | epoch
        off = 0
        self._staging_off = off
        off += self.staging.numel() * 4
        self._ready_off = off
        off += self.ready.numel() * 4
        self._epoch_off = off

        # publish my handle, collect everyone's
        import torch.distributed as dist

        handle = (ctypes.c_char * _IPC_HANDLE_BYTES)()
        base = self._blob.data_ptr()
        self._check(self._cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(base)))
        gathered = [torch.zeros(_IPC_HANDLE_BYTES, dtype=torch.uint8, device=dev) for _ in range(self.world)]
        mine = torch.frombuffer(bytearray(handle), dtype=torch.uint8).to(dev)
        dist.all_gather(gathered, mine, group=self.group)

        peer_bases = []
        self._opened = []
        for r in range(self.world):
            if r == self.rank:
                peer_bases.append(base)
                continue
            h = (ctypes.c_char * _IPC_HANDLE_BYTES).from_buffer_copy(
                gathered[r].cpu().numpy().tobytes()
            )
            ptr = ctypes.c_void_p()
            self._check(
                self._cudart.cudaIpcOpenMemHandle(
                    ctypes.byref(ptr), h, ctypes.c_uint(1)  # cudaIpcMemLazyEnablePeerAccess
                )
            )
            self._opened.append(ptr)
            peer_bases.append(ptr.value)

        # device-side pointer tables
        st = [b + self._staging_off for b in peer_bases]
        rd = [b + self._ready_off for b in peer_bases]
        self.staging_tbl = torch.tensor(st, dtype=torch.int64, device=dev)
        self.ready_tbl = torch.tensor(rd, dtype=torch.int64, device=dev)
        self.staging_local = torch.Tensor().set_(self._blob, self._staging_off, (self.staging.numel(),)).view(torch.float32)
        self.ready_local = torch.Tensor().set_(self._blob, self._ready_off, (self.world,)).view(torch.uint32)

    def _check(self, err):
        if int(err) != 0:
            raise RuntimeError(f"cuda error {int(err)}")

    # ------------------------------------------------------------------- gate
    def should_p2p_ar(self, x: torch.Tensor) -> bool:
        return (
            not self.disabled
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.is_contiguous()
            and x.numel() > 0
            and x.numel() <= self.max_elems
            and x.numel() % 4 == 0
            and x.device == self.device
        )

    # ------------------------------------------------------------------ launch
    def all_reduce_inplace(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        rc = self.lib.p2p_ar_bf16(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(self.staging_tbl.data_ptr()),
            ctypes.c_void_p(self.ready_tbl.data_ptr()),
            ctypes.c_void_p(self.epoch.data_ptr()),
            ctypes.c_void_p(self.staging_local.data_ptr()),
            ctypes.c_void_p(self.ready_local.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(n), ctypes.c_int(self.world), ctypes.c_int(self.rank),
            ctypes.c_int(self.stride), ctypes.c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"p2p_ar_bf16 failed: cuda {rc}")
        return x


_GLOBAL: Optional[P2PAllReduce] = None


def get_p2p_ar(group, device) -> Optional[P2PAllReduce]:
    """Lazily create the singleton (per process)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = P2PAllReduce(group, device)
    return _GLOBAL
