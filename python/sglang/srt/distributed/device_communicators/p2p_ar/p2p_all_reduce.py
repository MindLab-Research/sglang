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


class _IpcMemHandle(ctypes.Structure):
    """cudaIpcMemHandle_t -- 64 opaque bytes that must be passed BY VALUE.

    Passing a ``ctypes.c_char * 64`` array (or any pointer) instead hands the
    runtime the *address* of the bytes: the ABI classifies this struct as MEMORY,
    so the callee copies sizeof(cudaIpcMemHandle_t) bytes out of the argument
    slot, reads the pointer value as the handle payload and fails with
    cudaErrorInvalidValue (1). Verified on the B300 boxes with a two-process
    probe (allocate+get in process A, open in process B):

        c_char*64 array,  flags=1 -> err=1
        struct by value,  flags=1 -> err=0   (ptr resolved)
        struct by value,  flags=0 -> err=1   (peer access flag is required)

    That is exactly the "p2p_ar: init failed (cuda error 1); disabled" seen on
    both PD nodes.
    """

    _fields_ = [("bytes", ctypes.c_char * _IPC_HANDLE_BYTES)]


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
        # Explicit prototypes: the IPC handle is a 64-byte struct passed BY VALUE
        # to cudaIpcOpenMemHandle (see _IpcMemHandle). Without these, ctypes
        # marshals the argument as an integer/pointer and the open fails with
        # cudaErrorInvalidValue (1).
        self._cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
        self._cudart.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_IpcMemHandle), ctypes.c_void_p,
        ]
        self._cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
        self._cudart.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), _IpcMemHandle, ctypes.c_uint,
        ]
        self._cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
        self._cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]

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
        #
        # NOTE: two traps here, both hit in production:
        #  1. `handle.bytes` on a `c_char * 64` field is truncated at the first
        #     NUL byte (the handle is mostly zeros: 19/64 non-zero), so it must be
        #     read with ctypes.string_at(byref(...), 64) -- otherwise `mine` ends
        #     up 4 (or 0) elements long and the collective fails with
        #     "invalid tensor size at index 0 (expected (4), got (64))".
        #  2. the handle exchange must run on CPU tensors: the group handed to us
        #     is the gloo TP group, which does not do CUDA tensors.
        import torch.distributed as dist

        handle = _IpcMemHandle()
        base = self._blob.data_ptr()
        self._check(self._cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(base)))
        raw = ctypes.string_at(ctypes.byref(handle), _IPC_HANDLE_BYTES)
        assert len(raw) == _IPC_HANDLE_BYTES, len(raw)
        mine = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        gathered = [torch.zeros(_IPC_HANDLE_BYTES, dtype=torch.uint8) for _ in range(self.world)]
        dist.all_gather(gathered, mine, group=self.group)

        peer_bases = []
        self._opened = []
        for r in range(self.world):
            if r == self.rank:
                peer_bases.append(base)
                continue
            h = _IpcMemHandle.from_buffer_copy(gathered[r].cpu().numpy().tobytes())
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
        # Views into the CUDA blob. NOTE: do NOT build these with
        # `torch.Tensor().set_(self._blob, ...)` -- torch.Tensor() is a CPU tensor
        # and set_() onto a CUDA storage now raises "Attempted to set the storage
        # of a tensor on device cpu to a storage on different device cuda:N".
        # Slicing the CUDA blob keeps the storage on the right device.
        self.staging_local = self._blob[
            self._staging_off : self._staging_off + self.staging.numel() * 4
        ].view(torch.float32)
        self.ready_local = self._blob[
            self._ready_off : self._ready_off + self.ready.numel() * 4
        ].view(torch.uint32)

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
