from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk

logger = logging.getLogger(__name__)


class PDHiddenEventManager:
    """Backend-agnostic PD hidden streaming state helper.

    Backends own transport-specific READY/ACK packets and hidden-row data
    transfer. This component centralizes waiter, lifetime, and request-done
    state that can be reused by Mooncake, NIXL, MORI, or future backends.
    """

    def __init__(self, owner):
        self.owner = owner
        self.done_rooms = set()
        self.done_lock = threading.Lock()
        self.chunk_acks = defaultdict(int)
        self.chunk_ack_cv = threading.Condition()
        self.ack_waiters = {}
        self.room_waiters = defaultdict(deque)
        self.inflight_chunks = {}
        self.inflight_lock = threading.Lock()
        self.active_transfers = defaultdict(int)
        self.active_cv = threading.Condition()
        self.ready_chunks: Dict[int, List[dict]] = defaultdict(list)
        self.ready_lock = threading.Lock()
        self.ack_completions = queue.SimpleQueue()
        self.ack_pending_counts = defaultdict(int)
        self.ack_completion_cv = threading.Condition()
        self.acked_chunks: Dict[int, List[dict]] = defaultdict(list)
        self.acked_lock = threading.Lock()
        self.ack_wakeup_endpoint: Optional[str] = None
        self.ack_wakeup_receiver = None
        self.ack_wakeup_sender = None
        self.ack_wakeup_lock = threading.Lock()

    def supports_streaming(self) -> bool:
        return True

    def init_prefill_state(self) -> None:
        self.done_rooms.clear()
        self.chunk_acks.clear()
        self.ack_waiters.clear()
        self.room_waiters.clear()
        self.inflight_chunks.clear()
        self.active_transfers.clear()

    def init_decode_state(self) -> None:
        self.ready_chunks.clear()
        self.ack_completions = queue.SimpleQueue()
        self.ack_pending_counts.clear()
        self.acked_chunks.clear()
        self.ack_wakeup_endpoint = f"inproc://pd-hidden-ack-{id(self)}"
        self.ack_wakeup_receiver = self.owner._zmq_ctx.socket(zmq.PULL)
        self.ack_wakeup_receiver.bind(self.ack_wakeup_endpoint)
        self.ack_wakeup_sender = self.owner._zmq_ctx.socket(zmq.PUSH)
        self.ack_wakeup_sender.setsockopt(zmq.LINGER, 0)
        self.ack_wakeup_sender.connect(self.ack_wakeup_endpoint)

    def mark_request_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        with self.done_lock:
            room = int(bootstrap_room)
            if room in self.done_rooms:
                return
            self.done_rooms.add(room)
        pool = self.owner.pd_hidden_pool
        state_idx = self.state_index()
        if (
            pool is not None
            and state_indices is not None
            and state_idx is not None
            and state_idx < len(state_indices)
        ):
            indices = state_indices[state_idx]
            if indices is not None and len(indices) > 0:
                pool.free([int(idx) for idx in indices])

    def pop_request_done(self, bootstrap_room: int) -> bool:
        with self.done_lock:
            room = int(bootstrap_room)
            if room not in self.done_rooms:
                return False
            self.done_rooms.remove(room)
            return True

    def park_chunk_for_ack(
        self,
        *,
        transfer_queue: FastQueue,
        kv_chunk: TransferKVChunk,
        prefill_rank: int,
        expected_count: int,
        timeout_s: float = 300.0,
    ) -> bool:
        """Park a streaming hidden chunk until all Decode ACKs arrive.

        Returns True when the chunk was parked. False means ACKs were already
        available and the caller can finish the chunk immediately.
        """
        if kv_chunk.pd_hidden_start is None:
            return False
        key = (
            int(kv_chunk.room),
            int(prefill_rank),
            int(kv_chunk.pd_hidden_start),
        )
        expected_count = int(expected_count)
        if expected_count <= 0:
            kv_chunk.pd_hidden_ack_ready = True
            return False
        with self.chunk_ack_cv:
            if self.chunk_acks.get(key, 0) >= expected_count:
                self.chunk_acks[key] -= expected_count
                if self.chunk_acks[key] <= 0:
                    self.chunk_acks.pop(key, None)
                kv_chunk.pd_hidden_ack_ready = True
                return False
            if key in self.ack_waiters:
                raise RuntimeError(
                    "PD hidden ACK waiter already exists: "
                    f"room={key[0]}, prefill_rank={key[1]}, hidden_start={key[2]}"
                )
            kv_chunk.pd_hidden_ack_expected_count = expected_count
            self.ack_waiters[key] = (transfer_queue, kv_chunk)

        # Notify retransmission: the chunk-READY control message travels on a
        # lossy zmq PUSH socket (LINGER=0 rotation on any disconnect event —
        # common/conn.py::_connect), so the FIRST notify can be silently
        # dropped while later messages on the fresh socket go through
        # (observed 2026-08-24 windowed mode: chunk [17920,34304) notify lost
        # on exactly one rank pair; decode then hard-failed on the NEXT
        # chunk's out-of-order arrival, and the un-ACKed chunk parked here
        # held the room's single-in-flight slot for the full 300s ACK
        # timeout). Re-send the notify every few seconds until the ACK
        # arrives; decode dedups duplicates (re-ACKs them), so this is
        # idempotent and self-terminating.
        renotify_args = getattr(kv_chunk, "pd_hidden_renotify_args", None)
        if renotify_args is not None:
            self._start_renotify_loop(key, renotify_args)

        def on_timeout() -> None:
            with self.chunk_ack_cv:
                waiter = self.ack_waiters.pop(key, None)
            if waiter is None:
                return
            _, timed_out_chunk = waiter
            timed_out_chunk.pd_hidden_ack_timed_out = True
            self.owner.record_failure(
                key[0],
                "Timed out waiting for PD hidden chunk ACK: "
                f"prefill_rank={key[1]}, hidden_start={key[2]}",
            )
            self.owner.update_status(key[0], KVPoll.Failed)
            transfer_queue.put(timed_out_chunk)
            self.wake_ack_waiters(key[0])

        timer = threading.Timer(float(timeout_s), on_timeout)
        timer.daemon = True
        timer.start()
        return True

    # Re-notify cadence: healthy chunks ACK in well under a second, so the
    # loop only ever fires for notifies that were actually lost. 1s keeps the
    # worst-case notify-loss recovery (one dropped first notify) under ~1s
    # instead of ~3s, which directly bounds the TTFT tail on the PD path.
    RENOTIFY_INTERVAL_S = 1.0

    def _start_renotify_loop(self, key, renotify_args: dict) -> None:
        def tick() -> None:
            with self.chunk_ack_cv:
                alive = key in self.ack_waiters
            if alive:
                try:
                    self.owner.notify_pd_hidden_chunk_ready(**renotify_args)
                except Exception:
                    # Transient zmq error on a rotated socket; the next tick
                    # retries on a fresh connection.
                    logger.debug(
                        "PD hidden re-notify send failed (will retry): "
                        "room=%s hidden_start=%s",
                        key[0],
                        key[2],
                    )
            if alive:
                timer = threading.Timer(
                    self.RENOTIFY_INTERVAL_S, tick
                )
                timer.daemon = True
                timer.start()

        first = threading.Timer(self.RENOTIFY_INTERVAL_S, tick)
        first.daemon = True
        first.start()

    def wake_ack_waiters(self, room: int) -> None:
        room = int(room)
        with self.chunk_ack_cv:
            waiters = [
                self.ack_waiters.pop(key)
                for key in list(self.ack_waiters)
                if key[0] == room
            ]
            room_waiters = list(self.room_waiters.pop(room, []))
        for transfer_queue, kv_chunk in waiters:
            transfer_queue.put(kv_chunk)
        for transfer_queue, kv_chunk in room_waiters:
            transfer_queue.put(kv_chunk)

    # -- inflight bookkeeping (guarded by chunk_ack_cv, shared with the
    # room_waiters/park machinery so park-and-check can be atomic) --

    def get_inflight_chunk(self, room: int):
        with self.chunk_ack_cv:
            return self.inflight_chunks.get(int(room))

    def set_inflight_chunk(self, room: int, key) -> None:
        with self.chunk_ack_cv:
            self.inflight_chunks[int(room)] = key

    def pop_inflight_chunk(self, room: int, match=None) -> None:
        with self.chunk_ack_cv:
            room = int(room)
            if match is None or self.inflight_chunks.get(room) == match:
                self.inflight_chunks.pop(room, None)

    def park_chunk_behind_room(
        self, transfer_queue: FastQueue, kv_chunk: TransferKVChunk
    ) -> bool:
        """Atomically park the chunk behind the room's inflight chunk.

        Returns True when parked. False means the inflight chunk finished in
        the race window between the caller's inflight read and this park —
        the classic missed-wakeup: ``finish_streaming_chunk`` popped the
        inflight and its ``wake_next_room_waiter`` fired while
        ``room_waiters`` was still empty, so parking now would orphan this
        chunk forever (its notify is never sent -> decode-side out-of-order
        raise). The caller must proceed with the chunk instead.
        """
        room = int(kv_chunk.room)
        with self.chunk_ack_cv:
            if room not in self.inflight_chunks:
                return False
            self.room_waiters[room].append((transfer_queue, kv_chunk))
            return True

    def wake_next_room_waiter(self, room: int) -> None:
        with self.chunk_ack_cv:
            room_waiters = self.room_waiters.get(int(room))
            if not room_waiters:
                return
            transfer_queue, kv_chunk = room_waiters.popleft()
            if not room_waiters:
                self.room_waiters.pop(int(room), None)
        transfer_queue.put(kv_chunk)

    def handle_chunk_ack(self, room: int, prefill_rank: int, hidden_start: int) -> None:
        key = (int(room), int(prefill_rank), int(hidden_start))
        waiter_to_wake = None
        with self.chunk_ack_cv:
            self.chunk_acks[key] += 1
            waiter = self.ack_waiters.get(key)
            if waiter is not None:
                _, kv_chunk = waiter
                expected_count = kv_chunk.pd_hidden_ack_expected_count
                if self.chunk_acks[key] >= expected_count:
                    self.chunk_acks[key] -= expected_count
                    if self.chunk_acks[key] <= 0:
                        self.chunk_acks.pop(key, None)
                    self.ack_waiters.pop(key, None)
                    kv_chunk.pd_hidden_ack_ready = True
                    waiter_to_wake = waiter
            self.chunk_ack_cv.notify_all()
        if waiter_to_wake is not None:
            transfer_queue, kv_chunk = waiter_to_wake
            transfer_queue.put(kv_chunk)

    def pop_ready_chunks(self, room: int) -> List[dict]:
        with self.ready_lock:
            return self.ready_chunks.pop(int(room), [])

    def append_ready_chunk(self, room: int, chunk: dict) -> None:
        with self.ready_lock:
            self.ready_chunks[int(room)].append(chunk)

    def submit_chunk_ack(
        self,
        *,
        event,
        remote: str,
        dst_port: int,
        room: int,
        prefill_rank: int,
        hidden_start: int,
        is_last_hidden_chunk: bool,
    ) -> None:
        completion = {
            "remote": remote,
            "dst_port": int(dst_port),
            "room": int(room),
            "prefill_rank": int(prefill_rank),
            "hidden_start": int(hidden_start),
            "is_last_hidden_chunk": bool(is_last_hidden_chunk),
        }
        with self.ack_completion_cv:
            self.ack_pending_counts[int(room)] += 1

        def trigger_wakeup() -> None:
            with self.ack_wakeup_lock:
                self.ack_wakeup_sender.send(b"ACK_READY")

        if event is None:
            completion["success"] = True
            self.ack_completions.put(completion)
            trigger_wakeup()
            return

        def wait_for_injection() -> None:
            try:
                with torch.cuda.device(self.owner.kv_args.gpu_id):
                    event.synchronize()
                completion["success"] = True
            except Exception:
                logger.exception(
                    "PD hidden injection completion failed: room=%s start=%s",
                    room,
                    hidden_start,
                )
                completion["success"] = False
            self.ack_completions.put(completion)
            trigger_wakeup()

        threading.Thread(
            target=wait_for_injection,
            name=f"PDHiddenAckWaiter-{room}",
            daemon=True,
        ).start()

    def drain_ack_completions(self) -> None:
        while True:
            try:
                completion = self.ack_completions.get_nowait()
            except queue.Empty:
                return

            room = int(completion["room"])
            try:
                if completion.pop("success"):
                    self.owner.ack_pd_hidden_chunk(
                        remote=completion["remote"],
                        dst_port=int(completion["dst_port"]),
                        room=room,
                        prefill_rank=int(completion["prefill_rank"]),
                        hidden_start=int(completion["hidden_start"]),
                    )
                    with self.acked_lock:
                        self.acked_chunks[room].append(completion)
                else:
                    self.owner.record_failure(
                        room,
                        "PD hidden injection CUDA completion failed: "
                        f"hidden_start={completion['hidden_start']}",
                    )
                    self.owner.update_status(room, KVPoll.Failed)
            except Exception:
                logger.exception(
                    "Failed to send PD hidden chunk ACK: room=%s start=%s",
                    room,
                    completion["hidden_start"],
                )
                self.owner.record_failure(
                    room,
                    "Failed to send PD hidden chunk ACK: "
                    f"hidden_start={completion['hidden_start']}",
                )
                self.owner.update_status(room, KVPoll.Failed)
            finally:
                with self.ack_completion_cv:
                    self.ack_pending_counts[room] -= 1
                    if self.ack_pending_counts[room] <= 0:
                        self.ack_pending_counts.pop(room, None)
                    self.ack_completion_cv.notify_all()

    def pop_acked_chunks(self, room: int) -> List[dict]:
        with self.acked_lock:
            return self.acked_chunks.pop(int(room), [])

    def wait_ack_completions(self, room: int, timeout_s: float = 300.0) -> bool:
        room = int(room)
        deadline = time.monotonic() + float(timeout_s)
        with self.ack_completion_cv:
            while self.ack_pending_counts.get(room, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.ack_completion_cv.wait(timeout=min(remaining, 1.0))
        return True

    def begin_transfer(self, room: int) -> None:
        with self.active_cv:
            self.active_transfers[int(room)] += 1

    def end_transfer(self, room: int) -> None:
        with self.active_cv:
            room = int(room)
            count = self.active_transfers.get(room, 0) - 1
            if count <= 0:
                self.active_transfers.pop(room, None)
            else:
                self.active_transfers[room] = count
            self.active_cv.notify_all()

    def wait_transfers_quiesced(self, room: int, timeout_s: float = 300.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        room = int(room)
        with self.active_cv:
            while self.active_transfers.get(room, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Timed out waiting for PD hidden transfers to quiesce: "
                        "room=%s active=%s",
                        room,
                        self.active_transfers.get(room, 0),
                    )
                    return False
                self.active_cv.wait(timeout=min(remaining, 1.0))
            return True

    def state_index(self) -> Optional[int]:
        for idx, state_type in enumerate(self.owner.kv_args.state_types):
            if state_type == StateType.PD_HIDDEN:
                return idx
        return None

    def has_state(self, state_indices: Optional[List]) -> bool:
        idx = self.state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return False
        indices = state_indices[idx]
        return indices is not None and len(indices) > 0

    def without_state(self, state_indices: Optional[List]) -> Optional[List]:
        idx = self.state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return state_indices
        ret = list(state_indices)
        ret[idx] = None
        return ret

    def release_state_indices(self, kv_chunk: TransferKVChunk) -> Optional[List]:
        release_indices = kv_chunk.pd_hidden_release_indices
        if not release_indices:
            return kv_chunk.state_indices
        idx = self.state_index()
        if idx is None or not kv_chunk.state_indices or idx >= len(kv_chunk.state_indices):
            return kv_chunk.state_indices
        ret = list(kv_chunk.state_indices)
        ret[idx] = [int(x) for x in release_indices]
        return ret

    def free_state_indices(self, state_indices: Optional[List]) -> None:
        pool = self.owner.pd_hidden_pool
        state_idx = self.state_index()
        if (
            pool is None
            or state_idx is None
            or state_indices is None
            or state_idx >= len(state_indices)
        ):
            return
        indices = state_indices[state_idx]
        if indices is not None and len(indices) > 0:
            pool.free([int(idx) for idx in indices])

    def free_chunk_rows(self, kv_chunk: TransferKVChunk) -> None:
        self.free_state_indices(self.release_state_indices(kv_chunk))

    def release_or_mark_done(self, kv_chunk: TransferKVChunk) -> None:
        if kv_chunk.pd_hidden_start is not None:
            self.free_chunk_rows(kv_chunk)
        else:
            self.mark_request_done(
                kv_chunk.room,
                self.release_state_indices(kv_chunk),
            )

    def finish_streaming_chunk(
        self,
        kv_chunk: TransferKVChunk,
        hidden_inflight_key: Optional[Tuple[int, int]],
    ) -> None:
        self.free_chunk_rows(kv_chunk)
        # Pop the inflight marker AND wake the next room waiter in ONE
        # critical section: park_chunk_behind_room checks-and-appends under
        # the same lock, so a chunk either (a) observes the inflight marker
        # and is guaranteed to be in room_waiters before this wake runs, or
        # (b) observes no marker and proceeds — the missed-wakeup orphan race
        # (2026-08-23 windowed-mode out-of-order: notify for chunk
        # [14592,30976) never sent while [30976,...) overtook it) is closed.
        with self.chunk_ack_cv:
            room = int(kv_chunk.room)
            if (
                hidden_inflight_key is not None
                and self.inflight_chunks.get(room) == hidden_inflight_key
            ):
                self.inflight_chunks.pop(room, None)
            room_waiters = self.room_waiters.get(room)
            if room_waiters:
                transfer_queue, waiter_chunk = room_waiters.popleft()
                if not room_waiters:
                    self.room_waiters.pop(room, None)
            else:
                transfer_queue = waiter_chunk = None
        if waiter_chunk is not None:
            transfer_queue.put(waiter_chunk)

    def mark_session_failed_and_sync(
        self,
        *,
        kv_chunk: TransferKVChunk,
        req,
        prefill_unique_rank: int,
        failure_reason: str,
    ) -> None:
        session_id = req.mooncake_session_id
        with self.owner.session_lock:
            self.owner.session_failures[session_id] += 1
            if self.owner.session_failures[session_id] >= 1:
                self.owner.failed_sessions.add(session_id)
                logger.error("Session %s failed.", session_id)
        self.owner.record_failure(kv_chunk.room, failure_reason)
        self.owner.update_status(kv_chunk.room, KVPoll.Failed)
        self.wake_ack_waiters(kv_chunk.room)
        self.owner.sync_status_to_decode_endpoint(
            req.endpoint,
            req.dst_port,
            req.room,
            KVPoll.Failed,
            prefill_unique_rank,
        )
