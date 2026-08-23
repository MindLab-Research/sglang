from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    @staticmethod
    def derive_forward_mode(gathered_modes: List[int]) -> Optional[ForwardMode]:
        """Collapse the gathered per-DP-rank forward modes into one weight-table
        bucket; the input is rank-identical, so the recv decision is too."""
        active = set(gathered_modes) - {
            ForwardMode.IDLE.value,
            ForwardMode.PREBUILT.value,
        }
        if not active:
            return None  # globally idle: same bucket as "no last batch"
        if active - {ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.EXTEND  # any extend-like rank: prompt recv
        if active == {ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def __init__(self, server_args: ServerArgs):
        self._use_synced_mode = server_args.enable_dp_attention
        self._counter = 0
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def _pick_mode(self, last_batch: Optional[ScheduleBatch]) -> Optional[ForwardMode]:
        # The recv decision must be identical on every rank in the request
        # broadcast. Local modes differ across DP ranks (IDLE vs DECODE), so
        # use the rank-consistent mode derived from the MLP sync all-gather.
        if last_batch is None:
            return None
        if self._use_synced_mode:
            return last_batch.recv_skipper_forward_mode
        return last_batch.forward_mode

    def handle(self, last_batch: Optional[ScheduleBatch]) -> bool:
        # 2026-08-23 deadlock fix (1104 collective wedge, R5 case50):
        # the recv decision MUST be identical on every rank. The previous
        # weighted counter accumulated per-rank last_batch weights — when
        # local batch state diverges transiently the counters drift, one
        # rank skips the request broadcast while another enters it, and the
        # collective sequence on the shared tp_cpu_group desyncs (observed:
        # TP0 stuck in broadcast, 6 ranks stuck in pop_transferred's
        # all_reduce, engine idle-deadlocked with GPU 0%). The invocation
        # count is lockstep-identical across ranks by construction — every
        # rank calls recv_requests exactly once per loop iteration and the
        # iterations are barrier-synchronized — so this decision is
        # rank-invariant regardless of local batch state.
        self._counter += 1
        if self._counter >= self._threshold:
            self._counter = 0
            return True
        return False
