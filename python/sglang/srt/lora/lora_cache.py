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
"""Local cache for remotely-hosted LoRA adapters: content-addressed reuse + GC.

Why this module exists
----------------------
``lora_manager.resolve_lora_local_path`` downloads a LoRA archive from an
``http(s)://`` URL and unpacks it under ``SGLANG_LORA_CACHE_DIR`` (default
``/root/glm52_local/loras``). Deployments hand out **pre-signed** URLs whose
query string (``X-Amz-Date`` / ``X-Amz-Signature`` / ``X-Amz-Expires``) changes
on every request. Keying the cache on ``sha1(full_url)`` therefore never hit
twice: the same multi-GB artifact was re-downloaded and re-extracted again and
again (10 verified re-downloads of one adapter on the B300 PD cluster, 4.8 TB
of duplicated cache, 97% disk).

Two behaviours live here:

1. **Content-addressed dedup** -- the cache key is derived from
   ``scheme://host/path`` with the signature (query) stripped, so a refreshed
   pre-signed URL for the same artifact maps to the same directory and an
   existing copy is reused instead of downloaded. Directories written under the
   old URL-hash key are still found, by artifact basename.

2. **GC** -- unloading an adapter stamps ``.unloaded_at`` in its cache
   directory, which is deleted only once ``SGLANG_LORA_CACHE_TTL_SEC``
   (default 24 h) has elapsed since the unload. Each sweep also reclaims
   duplicate directories of the same artifact and leftover archives. A
   directory is never removed while it is in the ``in_use`` set, while a
   download lock (``<dir>.lock``) exists, or while it was touched within the
   TTL -- so a copy another rank is downloading or serving is safe.

Env knobs (all optional)::

    SGLANG_LORA_CACHE_DIR        cache root       (default /root/glm52_local/loras)
    SGLANG_LORA_CACHE_TTL_SEC    retention        (default 86400 = 24 h)
    SGLANG_LORA_CACHE_GC         enable GC        (default 1)
    SGLANG_LORA_CACHE_DEDUPE     reuse by content-addressed key + basename scan
                                                  (default 1)

The module is stdlib-only on purpose, so it can be unit-tested without
torch/CUDA.
"""

import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_CACHE_DIR = "/root/glm52_local/loras"
DONE_MARKER = ".done"
UNLOADED_MARKER = ".unloaded_at"
ARCHIVE_SUFFIXES = (".tar.zst", ".tzst", ".tar.gz", ".tgz", ".tar")
LOCK_SUFFIX = ".lock"

# "<artifact-id>-<sha1[:8]>" -- the shape written by this module and the legacy
# full-URL-hash key, so both can be grouped by artifact.
_KEY_RE = re.compile(r"^(?P<base>.+)-(?P<hash>[0-9a-f]{8})$")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def cache_root() -> str:
    return os.environ.get("SGLANG_LORA_CACHE_DIR") or DEFAULT_CACHE_DIR


def cache_ttl_sec() -> float:
    """How long an unloaded adapter stays on disk (default 24 h)."""
    return _env_float("SGLANG_LORA_CACHE_TTL_SEC", 24 * 3600.0)


def gc_enabled() -> bool:
    return _env_bool("SGLANG_LORA_CACHE_GC", True)


def dedupe_reuse_enabled() -> bool:
    return _env_bool("SGLANG_LORA_CACHE_DEDUPE", True)


def dedupe_gc_enabled() -> bool:
    return _env_bool("SGLANG_LORA_CACHE_DEDUPE", True)


# --------------------------------------------------------------------------- #
# Naming / key derivation
# --------------------------------------------------------------------------- #
def is_remote(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def artifact_identity(url: str) -> str:
    """URL identity that survives pre-signed-URL refresh: scheme://host/path.

    The query string (``X-Amz-*`` signature parameters) is deliberately dropped
    -- it changes on every request for the very same artifact.
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def artifact_basename(url: str) -> str:
    """Archive file name without directory and without the archive extension."""
    base = os.path.basename(urlparse(url).path)
    for ext in ARCHIVE_SUFFIXES:
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def cache_key(url: str) -> str:
    """Stable, content-addressed cache key for a remote adapter URL."""
    digest = hashlib.sha1(artifact_identity(url).encode()).hexdigest()[:8]
    return f"{artifact_basename(url)}-{digest}"


def split_key(name: str) -> Tuple[str, Optional[str]]:
    """Split ``<base>-<sha1[:8]>`` (returns ``(name, None)`` when it does not fit)."""
    m = _KEY_RE.match(name)
    if m and not name.endswith(LOCK_SUFFIX):
        return m.group("base"), m.group("hash")
    return name, None


# --------------------------------------------------------------------------- #
# Inspection helpers
# --------------------------------------------------------------------------- #
def _is_complete(path: str) -> bool:
    return os.path.exists(os.path.join(path, DONE_MARKER))


def _activity_ts(path: str) -> float:
    """Newest filesystem timestamp of a cache directory."""
    try:
        newest = os.stat(path).st_mtime
    except OSError:
        return 0.0
    marker = os.path.join(path, DONE_MARKER)
    try:
        newest = max(newest, os.stat(marker).st_mtime)
    except OSError:
        pass
    unloaded = _unloaded_ts(path)
    if unloaded is not None:
        newest = max(newest, unloaded)
    return newest


def _unloaded_ts(path: str) -> Optional[float]:
    """Read ``.unloaded_at`` (seconds since epoch); None when absent/invalid."""
    try:
        with open(os.path.join(path, UNLOADED_MARKER)) as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def cache_dir_for_path(path: str, root: Optional[str] = None) -> Optional[str]:
    """Return the managed cache directory that contains ``path``.

    ``resolve_lora_local_path`` may return a nested adapter dir (the tarball can
    store it as ``<target>/adapter``), so walk up to the directory carrying the
    ``.done`` marker.
    """
    root = os.path.abspath(root or cache_root())
    cur = os.path.abspath(path)
    for _ in range(8):
        if not cur.startswith(root) or len(cur) <= len(root):
            return None
        if _is_complete(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent
    return None


def find_cached_adapter(
    root: Optional[str], key: str, base: Optional[str] = None
) -> Optional[str]:
    """Locate an already-extracted copy of this artifact.

    Exact key first; then (unless dedup is disabled) any directory with the same
    artifact basename and a ``.done`` marker, preferring the most recent one.
    This makes copies downloaded under the legacy full-URL-hash key reusable.
    """
    root = root or cache_root()
    exact = os.path.join(root, key)
    if _is_complete(exact):
        return exact
    if not base or not dedupe_reuse_enabled():
        return None
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    candidates = []
    for name in entries:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_complete(path):
            continue
        if split_key(name)[0] == base:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_activity_ts)


def touch_loaded(path: str) -> None:
    """Mark a cache directory as (re)used: clears a stale unload stamp."""
    if not path or not os.path.isdir(path):
        return
    try:
        os.remove(os.path.join(path, UNLOADED_MARKER))
    except OSError:
        pass
    try:
        os.utime(path, None)
    except OSError:
        pass


def mark_unloaded(path: str, now: Optional[float] = None) -> bool:
    """Stamp a cache directory as unloaded; it becomes GC-eligible after the TTL."""
    if not path or not os.path.isdir(path):
        return False
    root = os.path.abspath(cache_root())
    if not os.path.abspath(path).startswith(root):
        # Only managed cache directories are GC'able; never stamp a user path.
        return False
    try:
        with open(os.path.join(path, UNLOADED_MARKER), "w") as fh:
            fh.write(str(now if now is not None else time.time()))
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Remote adapter download (curl + tar extraction)
# --------------------------------------------------------------------------- #
def find_adapter_dir(target: str) -> str:
    """Locate the directory that contains adapter_config.json.

    The checkpoint tarball may nest the LoRA under e.g. ``adapter/`` (full
    checkpoint layout), so return the innermost dir that looks like a LoRA
    adapter.
    """
    if os.path.exists(os.path.join(target, "adapter_config.json")):
        return target
    try:
        for d in sorted(os.listdir(target)):
            sub = os.path.join(target, d)
            if os.path.isdir(sub) and os.path.exists(
                os.path.join(sub, "adapter_config.json")
            ):
                return sub
    except OSError:
        pass
    return target


def curl_download(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` with curl.

    ``requests`` streaming can stall on some B300 nodes; curl is verified
    stable there. On those nodes ``/usr/local/bin/curl`` is a multi-threaded
    Range wrapper (``deploy/curl-parallel-wrapper.sh``) which shell PATH
    resolution picks up here automatically.
    """
    subprocess.run(
        [
            "curl",
            "-sfL",
            "--retry",
            "8",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "10",
            "-o",
            dest,
            url,
        ],
        check=True,
        timeout=3600,
    )


def resolve_remote_adapter(
    name: str,
    path: str,
    download=None,
    root: Optional[str] = None,
    logger=None,
    poll_interval: float = 1.0,
    timeout: float = 3600.0,
) -> str:
    """Return a local adapter directory for ``path``, downloading when needed.

    ``path`` may be an ``http(s)://`` URL (downloaded + unpacked under the cache
    root) or a local path (returned unchanged). Cache identity is
    content-addressed: an already-extracted copy is reused, so a refreshed
    pre-signed URL never triggers a second download.

    Only ONE TP rank downloads (``mkdir`` lock); the other ranks wait for the
    ``.done`` marker instead of downloading the multi-GB checkpoint again.
    """
    if not is_remote(path):
        return path

    root = root or cache_root()
    download = download or curl_download
    log = logger.info if logger is not None else (lambda *a, **k: None)

    if is_remote(name):
        base = artifact_basename(name)
        key = cache_key(name)
    else:
        # Registered under a plain name (e.g. loaded from a local dir): keep the
        # historical 1:1 mapping name -> directory.
        base, key = None, name
    target = os.path.join(root, key)
    marker = os.path.join(target, DONE_MARKER)

    cached = find_cached_adapter(root, key, base)
    if cached is not None:
        touch_loaded(cached)
        log(
            "lora %s already cached at %s (key=%s%s)",
            name,
            cached,
            key,
            "" if cached == target else ", dedup hit — no download",
        )
        return find_adapter_dir(cached)

    # Extension detection must use the URL *path* (strip query params).
    url_path = urlparse(path).path
    is_zst = url_path.endswith(".tar.zst") or url_path.endswith(".tzst")
    suffix = ".tar.zst" if is_zst else ".tar.gz"

    lock_dir = target + LOCK_SUFFIX
    deadline = time.time() + timeout
    while not os.path.exists(marker):
        if time.time() > deadline:
            raise RuntimeError(f"timed out waiting for lora {name} to download")
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            # another TP rank is downloading; wait for the marker
            time.sleep(poll_interval)
            continue
        try:
            archive = os.path.join(root, f"{key}{suffix}")
            log("downloading lora %s from %s", name, path)
            download(path, archive)
            log("extracting lora %s -> %s", name, target)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            os.makedirs(target, exist_ok=True)
            if is_zst:
                subprocess.run(
                    ["tar", "--zstd", "-xf", archive, "-C", target], check=True
                )
            else:
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(target)
            os.remove(archive)

            # Hoist a single nested top-level directory if present.
            entries = [os.path.join(target, e) for e in os.listdir(target)]
            if len(entries) == 1 and os.path.isdir(entries[0]):
                nested = entries[0]
                for e in os.listdir(nested):
                    shutil.move(os.path.join(nested, e), os.path.join(target, e))
                os.rmdir(nested)
            # Done marker (extraction succeeded) + usage stamp for the GC.
            with open(marker, "w") as f:
                f.write("ok")
            touch_loaded(target)
        finally:
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass

    log("lora %s ready at %s", name, target)
    return find_adapter_dir(target)


# --------------------------------------------------------------------------- #
# GC
# --------------------------------------------------------------------------- #
class CacheGCScheduler:
    """Run :func:`gc` on a daemon thread, at most one sweep at a time.

    Why this exists (2026-09-17, AWS B300 MoL pair ``3.37.20.114`` / ``15.164.0.39``):
    ``gc()`` used to be called *synchronously* from
    ``LoRAManager.load_lora_adapter()`` / ``unload_lora_adapter()`` — i.e. while the
    engine holds ``lora_update_lock``. With a 55 GB adapter directory a sweep takes
    minutes (tar / dedup / prune), so every later LoRA op queued behind it:

    * ``Start unload Lora adapter`` at 14:16:20 never returned (no per-rank lines);
    * loads submitted from 14:19 onward never ran (no ``LORA-ASSIGN``);
    * the control-plane deploy therefore never got its per-engine ACK and sat at
      ``FAILED``/``LOADING`` for ~30 min while the platform retried the deploy.

    The sweep is idle-time housekeeping: it must never block the update path.
    Scheduling is O(1); the caller keeps holding no lock while the GC runs.

    ``gc_enabled()`` (``SGLANG_LORA_CACHE_GC``) is still honoured here, and only one
    sweep runs at a time — a second ``schedule()`` while a sweep is alive is a no-op,
    because the sweep in flight already sees the current directory set.

    Residual race (documented, not solved here): ``in_use`` is a snapshot taken at
    schedule time. A directory that becomes in use *after* the snapshot can still be
    reclaimed if its TTL already expired; ``resolve_lora_local_path`` re-downloads in
    that case, so the cost is a re-download rather than a correctness bug.
    """

    def __init__(self, *, name: str = "lora-cache-gc") -> None:
        self._name = name
        self._thread: Optional[threading.Thread] = None
        self._last_result: Optional[Dict[str, object]] = None

    def schedule(self, in_use: Iterable[str] = (), logger=None) -> bool:
        """Start a sweep if none is running. Returns True when one was started."""
        if not gc_enabled():
            return False
        thread = self._thread
        if thread is not None and thread.is_alive():
            return False
        self._thread = threading.Thread(
            target=self._run,
            args=(set(in_use), logger),
            name=self._name,
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, in_use: set, logger) -> None:
        try:
            result = gc(in_use=in_use, logger=logger)
        except Exception:
            if logger is not None:
                logger.exception("LoRA cache GC failed (background); continuing")
            return
        self._last_result = result
        deleted = (result or {}).get("deleted") or []
        if deleted and logger is not None:
            logger.info(
                "LoRA cache GC: removed %d dir(s), freed %.2f GB",
                len(deleted),
                float((result or {}).get("freed_bytes", 0)) / 1e9,
            )

    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Join the in-flight sweep (tests / shutdown)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


def gc(
    root: Optional[str] = None,
    ttl: Optional[float] = None,
    in_use: Iterable[str] = (),
    now: Optional[float] = None,
    logger=None,
    dedupe: Optional[bool] = None,
) -> Dict[str, object]:
    """Reclaim cache space.

    Deletes, under ``root``:

    * directories whose ``.unloaded_at`` is older than ``ttl`` (unloaded and
      past retention),
    * duplicate directories of one artifact (same basename): keep the most
      recently used one, drop the older copies once they are older than ``ttl``,
    * incomplete directories (no ``.done``) older than ``ttl`` with no active
      download lock,
    * leftover archives older than ``ttl``.

    Never deletes a directory that is in ``in_use``, that has an active
    ``<dir>.lock`` (download in flight), or that was touched within ``ttl``.

    Returns ``{"deleted": [...], "kept": int, "freed_bytes": int}``.
    """
    result: Dict[str, object] = {"deleted": [], "kept": 0, "freed_bytes": 0}
    root = root or cache_root()
    ttl = cache_ttl_sec() if ttl is None else float(ttl)
    now = time.time() if now is None else float(now)
    dedupe = dedupe_gc_enabled() if dedupe is None else bool(dedupe)
    if ttl <= 0 or not os.path.isdir(root):
        return result

    in_use_real = {os.path.realpath(p) for p in in_use if p}
    log = logger.info if logger is not None else (lambda *a, **k: None)
    deleted: List[str] = []
    freed = 0

    def _drop(path: str, reason: str) -> None:
        nonlocal freed
        size = dir_size(path) if os.path.isdir(path) else _file_size(path)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as exc:  # pragma: no cover - best effort
            log("lora cache GC: failed to remove %s (%s)", path, exc)
            return
        deleted.append(path)
        freed += size
        log(
            "lora cache GC: removed %s (%.2f GB, %s)",
            path,
            size / 1e9,
            reason,
        )

    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return result

    complete_dirs: List[str] = []
    for name in entries:
        if name.startswith(".") or name.endswith(LOCK_SUFFIX):
            continue
        path = os.path.join(root, name)
        if os.path.islink(path):
            continue
        if os.path.isfile(path):
            if name.endswith(ARCHIVE_SUFFIXES) and _mtime(path) < now - ttl:
                _drop(path, "stale archive")
            continue
        if not os.path.isdir(path):
            continue
        if os.path.realpath(path) in in_use_real:
            result["kept"] += 1
            continue
        lock_active = os.path.exists(path + LOCK_SUFFIX)
        if not _is_complete(path):
            if not lock_active and _activity_ts(path) < now - ttl:
                _drop(path, "incomplete download")
            else:
                result["kept"] += 1
            continue
        unloaded = _unloaded_ts(path)
        if unloaded is not None and now - unloaded > ttl:
            _drop(path, "unloaded > TTL ago")
            continue
        complete_dirs.append(path)

    if dedupe and complete_dirs:
        groups: Dict[str, List[str]] = {}
        for path in complete_dirs:
            groups.setdefault(split_key(os.path.basename(path))[0], []).append(path)
        for base, paths in groups.items():
            if len(paths) < 2:
                continue
            paths.sort(key=_activity_ts, reverse=True)
            for stale in paths[1:]:
                if os.path.realpath(stale) in in_use_real:
                    continue
                if os.path.exists(stale + LOCK_SUFFIX):
                    continue
                if _activity_ts(stale) >= now - ttl:
                    continue
                _drop(stale, f"duplicate of {base}")
                complete_dirs.remove(stale)

    result["deleted"] = deleted
    result["freed_bytes"] = freed
    result["kept"] = result["kept"] + len(complete_dirs)  # type: ignore[operator]
    return result


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _file_size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0
