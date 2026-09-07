"""Unit tests for --router auto-registration with sgl-model-gateway.

Covers URL derivation, the WorkerConfigRequest payload mapping, the
register/retry/give-up ladder, and best-effort deregistration. All network
calls are mocked; no gateway or model is needed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import router_registration
from sglang.srt.entrypoints.router_registration import (
    _build_registration_payload,
    _derive_worker_url,
    deregister_from_router,
    register_with_router,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_server_args(**overrides):
    """Minimal ServerArgs stand-in with the fields router_registration reads."""
    defaults = dict(
        router="http://router:30000",
        router_advertise_url=None,
        host="10.1.2.3",
        port=31000,
        ssl_certfile=None,
        served_model_name="glm-5.3",
        disaggregation_mode="null",
        api_key=None,
        disaggregation_bootstrap_port=8998,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _reset_registration_state():
    router_registration._registration = None


class _FakeResponse:
    def __init__(self, status_code=202, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {"worker_id": "42"}
        self.text = text

    def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body


class TestDeriveWorkerUrl(unittest.TestCase):
    def setUp(self):
        _reset_registration_state()

    def test_explicit_advertise_url_wins(self):
        args = _make_server_args(router_advertise_url="http://nated:9999/")
        self.assertEqual(_derive_worker_url(args), "http://nated:9999")

    def test_explicit_host_used_directly(self):
        args = _make_server_args(host="192.168.1.10")
        self.assertEqual(_derive_worker_url(args), "http://192.168.1.10:31000")

    def test_bind_all_interfaces_autodetects_ip(self):
        args = _make_server_args(host="0.0.0.0")
        with patch.object(
            router_registration, "_detect_local_ip", return_value="172.31.45.101"
        ):
            self.assertEqual(_derive_worker_url(args), "http://172.31.45.101:31000")

    def test_autodetect_fallback_to_loopback(self):
        args = _make_server_args(host="")
        with patch.object(
            router_registration, "_detect_local_ip", return_value=None
        ):
            self.assertEqual(_derive_worker_url(args), "http://127.0.0.1:31000")

    def test_ssl_certfile_upgrades_scheme(self):
        args = _make_server_args(ssl_certfile="/tmp/cert.pem")
        self.assertEqual(_derive_worker_url(args), "https://10.1.2.3:31000")


class TestBuildRegistrationPayload(unittest.TestCase):
    def setUp(self):
        _reset_registration_state()

    def test_regular_worker(self):
        args = _make_server_args()
        self.assertEqual(
            _build_registration_payload(args, "http://10.1.2.3:31000"),
            {
                "url": "http://10.1.2.3:31000",
                "model_id": "glm-5.3",
                "worker_type": None,
                "runtime": "sglang",
            },
        )

    def test_pd_prefill_worker(self):
        args = _make_server_args(disaggregation_mode="prefill")
        payload = _build_registration_payload(args, "http://p:1")
        self.assertEqual(payload["worker_type"], "prefill")
        self.assertEqual(payload["bootstrap_port"], 8998)
        self.assertNotIn("api_key", payload)

    def test_pd_decode_worker(self):
        args = _make_server_args(disaggregation_mode="decode")
        payload = _build_registration_payload(args, "http://d:1")
        self.assertEqual(payload["worker_type"], "decode")
        # bootstrap_port is a prefill-only field on the gateway
        self.assertNotIn("bootstrap_port", payload)

    def test_api_key_forwarded(self):
        args = _make_server_args(api_key="sk-secret")
        payload = _build_registration_payload(args, "http://10.1.2.3:31000")
        self.assertEqual(payload["api_key"], "sk-secret")


class TestRegisterWithRouter(unittest.TestCase):
    def setUp(self):
        _reset_registration_state()

    def test_success_stores_registration_state(self):
        args = _make_server_args()
        with patch.object(
            router_registration.requests,
            "post",
            return_value=_FakeResponse(
                202, {"worker_id": "uuid-1", "status": "accepted"}
            ),
        ) as mock_post:
            self.assertTrue(register_with_router(args))
        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args
        self.assertEqual(url[0], "http://router:30000/workers")
        self.assertEqual(kwargs["json"]["url"], "http://10.1.2.3:31000")
        self.assertEqual(kwargs["json"]["runtime"], "sglang")
        self.assertIsNotNone(router_registration._registration)
        self.assertEqual(router_registration._registration.worker_id, "uuid-1")

    def test_retries_then_succeeds(self):
        args = _make_server_args()
        responses = [
            router_registration.requests.exceptions.ConnectTimeout("down"),
            _FakeResponse(202, {"worker_id": "uuid-2"}),
        ]

        def fake_post(*a, **kw):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(
            router_registration, "REGISTER_RETRY_INTERVAL_SECS", 0
        ), patch.object(router_registration.requests, "post", side_effect=fake_post):
            self.assertTrue(register_with_router(args))
        self.assertEqual(router_registration._registration.worker_id, "uuid-2")

    def test_client_error_aborts_without_retry(self):
        args = _make_server_args()
        with patch.object(
            router_registration.requests,
            "post",
            return_value=_FakeResponse(400, None, text="bad request"),
        ) as mock_post, patch.object(
            router_registration, "REGISTER_RETRY_INTERVAL_SECS", 0
        ):
            self.assertFalse(register_with_router(args))
        # 4xx must not be retried
        self.assertEqual(mock_post.call_count, 1)
        self.assertIsNone(router_registration._registration)

    def test_server_error_is_retried(self):
        args = _make_server_args()
        responses = iter([_FakeResponse(503), _FakeResponse(202, {"worker_id": "u"})])

        with patch.object(
            router_registration, "REGISTER_RETRY_INTERVAL_SECS", 0
        ), patch.object(
            router_registration.requests,
            "post",
            side_effect=lambda *a, **kw: next(responses),
        ):
            self.assertTrue(register_with_router(args))

    def test_gives_up_after_max_attempts(self):
        args = _make_server_args()
        attempts = []

        def always_timeout(*a, **kw):
            attempts.append(1)
            raise router_registration.requests.exceptions.ConnectionError("down")

        with patch.object(
            router_registration, "REGISTER_MAX_ATTEMPTS", 3
        ), patch.object(
            router_registration, "REGISTER_RETRY_INTERVAL_SECS", 0
        ), patch.object(
            router_registration.requests, "post", side_effect=always_timeout
        ):
            self.assertFalse(register_with_router(args))
        self.assertEqual(len(attempts), 3)
        self.assertIsNone(router_registration._registration)

    def test_response_without_json_body(self):
        """A 2xx without a parseable body still counts as registered."""
        args = _make_server_args()
        with patch.object(
            router_registration.requests, "post", return_value=_FakeResponse(202, None)
        ):
            self.assertTrue(register_with_router(args))
        # No worker_id -> no deregistration state (id-less DELETE impossible)
        self.assertIsNone(router_registration._registration)


class TestDeregisterFromRouter(unittest.TestCase):
    def setUp(self):
        _reset_registration_state()

    def test_noop_when_never_registered(self):
        # Must not raise and must not touch the network.
        deregister_from_router()

    def test_deletes_by_worker_id(self):
        router_registration._registration = router_registration._RegistrationState(
            router_url="http://router:30000",
            worker_id="uuid-1",
            worker_url="http://10.1.2.3:31000",
        )
        with patch.object(
            router_registration.requests, "delete"
        ) as mock_delete:
            mock_delete.return_value = _FakeResponse(204)
            deregister_from_router()
        mock_delete.assert_called_once_with(
            "http://router:30000/workers/uuid-1",
            timeout=router_registration.DEREGISTER_TIMEOUT_SECS,
        )
        self.assertIsNone(router_registration._registration)

    def test_missing_worker_on_router_is_ok(self):
        router_registration._registration = router_registration._RegistrationState(
            router_url="http://router:30000",
            worker_id="gone",
            worker_url="http://10.1.2.3:31000",
        )
        with patch.object(
            router_registration.requests,
            "delete",
            return_value=_FakeResponse(404),
        ):
            deregister_from_router()  # must not raise

    def test_network_error_swallowed(self):
        router_registration._registration = router_registration._RegistrationState(
            router_url="http://router:30000",
            worker_id="uuid-1",
            worker_url="http://10.1.2.3:31000",
        )
        with patch.object(
            router_registration.requests,
            "delete",
            side_effect=router_registration.requests.exceptions.ConnectionError("down"),
        ):
            deregister_from_router()  # must not raise
        # State is consumed either way; a second call is a no-op.
        self.assertIsNone(router_registration._registration)


if __name__ == "__main__":
    unittest.main()
