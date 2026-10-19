"""Fatal serving-worker failures must not look like a successful PID-1 exit."""

import os
import signal
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager import (
    ServerStatus,
    SignalHandler,
    TokenizerManager,
)
from sglang.srt.utils.common import kill_process_tree
from sglang.srt.utils.cudacore_pyspy_dump_utils import (
    pyspy_dump_schedulers,
    trigger_cuda_user_coredump,
)
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestFatalExitStatus(unittest.TestCase):
    def test_self_kill_fallback_preserves_requested_exit_code(self):
        # Linux PID 1 can survive SIGKILL; emulate that with a no-op kill.
        process = Mock()
        process.children.return_value = []
        for code in (0, 1):
            with (
                self.subTest(code=code),
                patch("sglang.srt.utils.common.psutil.Process", return_value=process),
            ):
                with self.assertRaises(SystemExit) as caught:
                    kill_process_tree(os.getpid(), exit_code=code)
                self.assertEqual(caught.exception.code, code)

    def test_sigquit_marks_unhealthy_and_requests_failure_exit(self):
        manager = SimpleNamespace(
            server_status=ServerStatus.Up,
            _subprocess_watchdog=Mock(),
            dump_requests_before_crash=Mock(),
        )
        with patch("sglang.srt.managers.tokenizer_manager.kill_process_tree") as kill:
            SignalHandler(manager).running_phase_sigquit_handler()
        self.assertEqual(manager.server_status, ServerStatus.UnHealthy)
        manager._subprocess_watchdog.stop.assert_called_once()
        manager.dump_requests_before_crash.assert_called_once()
        kill.assert_called_once_with(os.getpid(), exit_code=1)

    def test_pyspy_timeout_uses_shared_budget_and_direct_child(self):
        with (
            patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.collect_scheduler_processes",
                return_value=[SimpleNamespace(pid=10), SimpleNamespace(pid=20)],
            ),
            patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.time.monotonic",
                side_effect=[0, 0, 6, 11],
            ),
            patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.subprocess.run",
                side_effect=subprocess.TimeoutExpired("py-spy", 5),
            ) as run,
        ):
            pyspy_dump_schedulers(scheduler_only=True)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([x.kwargs["timeout"] for x in run.call_args_list], [5.0, 4.0])
        self.assertTrue(all(isinstance(x.args[0], list) for x in run.call_args_list))
        self.assertTrue(
            all(not x.kwargs.get("shell", False) for x in run.call_args_list)
        )

    def test_missing_pyspy_is_nonfatal_to_cleanup(self):
        with (
            patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.collect_scheduler_processes",
                return_value=[SimpleNamespace(pid=10)],
            ),
            patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.subprocess.run",
                side_effect=FileNotFoundError("py-spy missing"),
            ),
        ):
            pyspy_dump_schedulers(scheduler_only=True)

    def test_watchdog_notifies_parent_before_failing_diagnostics(self):
        watchdog = WatchdogRaw.__new__(WatchdogRaw)
        watchdog.debug_name = "test"
        watchdog.get_counter = lambda: 0
        watchdog.is_active = lambda: True
        watchdog.watchdog_timeout = 10
        watchdog.soft = False
        watchdog.parent_process = Mock()

        def diagnostic():
            watchdog.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
            raise RuntimeError("diagnostics failed")

        watchdog.dump_info = diagnostic
        with (
            patch(
                "sglang.srt.utils.watchdog.time.perf_counter", side_effect=[0, 1, 20]
            ),
            patch("sglang.srt.utils.watchdog.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "diagnostics failed"),
        ):
            watchdog._watchdog_once()

    def test_disabled_cuda_dump_returns_without_opening_pipe(self):
        with (
            patch.dict(os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "0"}),
            patch("sglang.srt.utils.cudacore_pyspy_dump_utils.os.open") as open_pipe,
        ):
            self.assertFalse(trigger_cuda_user_coredump())
        open_pipe.assert_not_called()

    def test_cuda_dump_reports_success_only_after_pipe_write(self):
        for error, expected in ((FileNotFoundError(), False), (None, True)):
            with (
                self.subTest(expected=expected),
                patch.dict(os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}),
                patch(
                    "sglang.srt.utils.cudacore_pyspy_dump_utils.collect_scheduler_processes",
                    return_value=[SimpleNamespace(pid=10)],
                ),
                patch(
                    "sglang.srt.utils.cudacore_pyspy_dump_utils._resolve_cuda_coredump_pipe_path",
                    return_value="/test/pipe",
                ),
                patch(
                    "sglang.srt.utils.cudacore_pyspy_dump_utils.os.open",
                    side_effect=error,
                    return_value=42,
                ),
                patch("sglang.srt.utils.cudacore_pyspy_dump_utils.os.write") as write,
                patch("sglang.srt.utils.cudacore_pyspy_dump_utils.os.close"),
            ):
                self.assertEqual(
                    trigger_cuda_user_coredump(scheduler_only=True), expected
                )
                self.assertEqual(write.call_count, int(expected))

    def test_crash_waits_for_cuda_dump_only_when_triggered(self):
        for triggered in (False, True):
            manager = SimpleNamespace(
                crash_dump_folder=None, crash_dump_performed=False
            )
            with (
                self.subTest(triggered=triggered),
                patch.object(
                    envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH, "get", return_value=False
                ),
                patch.object(
                    envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH, "get", return_value=True
                ),
                patch.object(
                    envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS,
                    "get",
                    return_value=60,
                ),
                patch(
                    "sglang.srt.managers.tokenizer_manager.collect_scheduler_processes",
                    return_value=[Mock()],
                ),
                patch(
                    "sglang.srt.managers.tokenizer_manager.trigger_cuda_user_coredump",
                    return_value=triggered,
                ),
                patch("sglang.srt.managers.tokenizer_manager.time.sleep") as sleep,
            ):
                TokenizerManager.dump_requests_before_crash(manager)
                self.assertEqual(
                    [call.args[0] for call in sleep.call_args_list],
                    [5, 60] if triggered else [5],
                )


if __name__ == "__main__":
    unittest.main()
