# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from errno import ENXIO
from pathlib import Path
from typing import List

import psutil

logger = logging.getLogger(__name__)


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def pyspy_dump_schedulers(scheduler_only=False, timeout=10.0):
    """Dump local scheduler stacks within a shared deadline.

    Crash handling must eventually exit even if ptrace/py-spy is stuck. Run
    py-spy directly so subprocess.run can kill and reap the actual debugger.
    """
    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [psutil.Process().pid]
    deadline = time.monotonic() + timeout
    for pid in pids:
        for attempt, native_flag in enumerate(["--native", ""]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Pyspy dump deadline exceeded; continuing crash cleanup.")
                return
            cmd = ["py-spy", "dump"]
            if native_flag:
                cmd.append(native_flag)
            cmd.extend(["--pid", str(pid)])
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=min(5.0, remaining),
                )
                logger.error(f"Pyspy dump for PID {pid} ({cmd}):\n{result.stdout}")
                break
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                OSError,
            ) as e:
                logger.error(
                    f"Pyspy failed ({cmd}). Error: {getattr(e, 'stderr', None) or e}"
                )
                if attempt == 1:
                    logger.error(f"All pyspy dump attempts failed for PID {pid}.")


def trigger_cuda_user_coredump(scheduler_only=False) -> bool:
    """Return whether any CUDA dump pipe was successfully triggered."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )
        return False

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return False
    else:
        procs = [psutil.Process()]

    triggered = False
    for proc in procs:
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
                triggered = True
            finally:
                os.close(fd)
            logger.error(
                "Triggered CUDA user coredump for PID %s via %s",
                proc.pid,
                pipe_path,
            )
        except FileNotFoundError:
            logger.error(
                "CUDA coredump pipe not found for PID %s: %s. Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
                pipe_path,
            )
        except OSError as e:
            if e.errno == ENXIO:
                logger.error(
                    "CUDA coredump pipe has no reader for PID %s: %s",
                    proc.pid,
                    pipe_path,
                )
            else:
                logger.exception(
                    "Failed to trigger CUDA user coredump for PID %s via %s",
                    proc.pid,
                    pipe_path,
                )
    return triggered
