"""Fatal serving-worker failures must not look like a successful PID-1 exit."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.tokenizer_manager import ServerStatus, SignalHandler
from sglang.srt.utils.common import kill_process_tree
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


if __name__ == "__main__":
    unittest.main()
