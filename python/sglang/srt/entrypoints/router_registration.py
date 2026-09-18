"""Auto-registration of sglang workers with an sgl-model-gateway router.

The Rust router (``sgl-model-gateway``) exposes a dynamic worker-management
API on the public port:

- ``POST   /workers``          body ``WorkerConfigRequest`` → 202 + worker_id
- ``DELETE /workers/{id}``      remove by the UUID returned above
- ``GET    /workers``           list

When the server is launched with ``--router <url>``, it registers itself once
it is healthy (after warmup) and deregisters on graceful shutdown. This keeps
the "start router first, then start workers" workflow free of manual
``curl POST /workers`` steps.

This module is import-light (stdlib + ``requests`` only) so unit tests can
exercise it without a GPU or a live FastAPI app.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import time
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Bounded retry: the router may come up after the worker (start order is not
# guaranteed). 60 attempts x 5s = 5 minutes, then the worker gives up and
# stays up unregistered (health probes on the router side will surface it).
REGISTER_RETRY_INTERVAL_SECS = 5.0
REGISTER_MAX_ATTEMPTS = 60
REGISTER_TIMEOUT_SECS = 5.0
DEREGISTER_TIMEOUT_SECS = 3.0


@dataclasses.dataclass
class _RegistrationState:
    """Bookkeeping for one successful registration (for deregistration)."""

    router_url: str
    worker_id: str
    worker_url: str


# Module-level state: written by the warmup thread (register_with_router),
# read/cleared by the HTTP shutdown path (deregister_from_router). Assignment
# of a single object reference is atomic under the GIL, which is all the
# synchronization the two touch points need.
_registration: Optional[_RegistrationState] = None


def _derive_worker_url(server_args) -> str:
    """The URL the router will use to reach this worker.

    --router-advertise-url wins (e.g. behind NAT / a proxy). Otherwise derive
    from --host/--port: when binding to all interfaces, auto-detect the
    local routable IP (UDP connect trick, no packet is actually sent) since
    0.0.0.0 is not a dialable address for the router.
    """
    if server_args.router_advertise_url:
        return server_args.router_advertise_url.rstrip("/")

    scheme = "https" if server_args.ssl_certfile else "http"
    host = server_args.host
    if not host or host in ("0.0.0.0", "::"):
        host = _detect_local_ip() or "127.0.0.1"
    return f"{scheme}://{host}:{server_args.port}"


def _detect_local_ip() -> Optional[str]:
    """Local routable IPv4 (best effort; None if offline)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # No packet is sent for a UDP connect; it just picks the route.
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return None


def _build_registration_payload(server_args, worker_url: str) -> dict:
    """Map ServerArgs onto the gateway's WorkerConfigRequest JSON body.

    Gateway contract highlights (sgl-model-gateway WorkerConfigRequest):
    - ``url`` is required, ``runtime`` is required ("sglang" here).
    - ``worker_type``: "prefill"/"decode" for PD workers, null otherwise.
    - ``bootstrap_port`` is consumed for PD prefill workers (the router
      derives bootstrap_room routing from it).
    """
    worker_type = None
    if server_args.disaggregation_mode in ("prefill", "decode"):
        worker_type = server_args.disaggregation_mode

    payload = {
        "url": worker_url,
        "model_id": server_args.served_model_name,
        "worker_type": worker_type,
        "runtime": "sglang",
    }
    if server_args.api_key:
        # The router needs the worker's key to call its endpoints.
        payload["api_key"] = server_args.api_key
    if worker_type == "prefill":
        payload["bootstrap_port"] = server_args.disaggregation_bootstrap_port
    return payload


def register_with_router(server_args) -> bool:
    """POST this worker to ``{server_args.router}/workers`` until it sticks.

    Called from the warmup thread once the server is healthy. Retries on
    connection errors / 5xx (router not up yet); a 4xx means the payload is
    wrong and aborts immediately. Never raises: an unregistered worker keeps
    serving, and the router's own probes/scale-down will surface the gap.

    Returns True when registered, False when attempts were exhausted.
    """
    router_url = server_args.router.rstrip("/")
    worker_url = _derive_worker_url(server_args)
    payload = _build_registration_payload(server_args, worker_url)

    for attempt in range(1, REGISTER_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                f"{router_url}/workers",
                json=payload,
                timeout=REGISTER_TIMEOUT_SECS,
            )
        except requests.exceptions.RequestException as e:
            logger.info(
                "Router registration attempt %d/%d failed (router at %s "
                "unreachable: %s); retrying in %.0fs",
                attempt,
                REGISTER_MAX_ATTEMPTS,
                router_url,
                e,
                REGISTER_RETRY_INTERVAL_SECS,
            )
        else:
            if resp.status_code in (200, 201, 202):
                global _registration
                worker_id = None
                try:
                    worker_id = resp.json().get("worker_id")
                except ValueError:
                    pass
                logger.info(
                    "Registered with router %s as worker %s (worker_id=%s)",
                    router_url,
                    worker_url,
                    worker_id,
                )
                if worker_id:
                    _registration = _RegistrationState(
                        router_url=router_url,
                        worker_id=worker_id,
                        worker_url=worker_url,
                    )
                return True
            if 400 <= resp.status_code < 500:
                # Bad payload (e.g. rejected model_id): retrying is useless.
                logger.error(
                    "Router registration rejected by %s with %d: %s. "
                    "Giving up on registration; the worker keeps serving.",
                    router_url,
                    resp.status_code,
                    resp.text[:500],
                )
                return False
            logger.info(
                "Router registration attempt %d/%d got %d from %s; retrying "
                "in %.0fs",
                attempt,
                REGISTER_MAX_ATTEMPTS,
                resp.status_code,
                router_url,
                REGISTER_RETRY_INTERVAL_SECS,
            )

        if attempt < REGISTER_MAX_ATTEMPTS:
            time.sleep(REGISTER_RETRY_INTERVAL_SECS)

    logger.error(
        "Could not register with router %s after %d attempts; the worker "
        "keeps serving but the router will not route to it.",
        router_url,
        REGISTER_MAX_ATTEMPTS,
    )
    return False


def deregister_from_router() -> None:
    """Best-effort DELETE of this worker's registration on shutdown.

    Safe to call unconditionally (no-op when never registered or when the
    registration attempt never succeeded); never raises so it can sit in a
    `finally:` block. Missing registrations are ignored by the router (404).
    """
    global _registration
    state = _registration
    if state is None:
        return
    _registration = None
    try:
        resp = requests.delete(
            f"{state.router_url}/workers/{state.worker_id}",
            timeout=DEREGISTER_TIMEOUT_SECS,
        )
        if resp.status_code in (200, 202, 204, 404):
            logger.info(
                "Deregistered worker %s from router %s",
                state.worker_url,
                state.router_url,
            )
        else:
            logger.warning(
                "Deregistration from router %s returned %d: %s",
                state.router_url,
                resp.status_code,
                resp.text[:500],
            )
    except requests.exceptions.RequestException as e:
        logger.warning(
            "Deregistration from router %s failed: %s", state.router_url, e
        )
