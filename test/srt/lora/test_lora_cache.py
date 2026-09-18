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
"""Unit tests for sglang.srt.lora.lora_cache.

The module is stdlib-only, so these tests run without torch/CUDA:

    python3 test/srt/lora/test_lora_cache.py     # local, no pytest needed
    pytest test/srt/lora/test_lora_cache.py
"""

import contextlib
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MODULE_PATH = os.path.join(_REPO_ROOT, "python", "sglang", "srt", "lora", "lora_cache.py")


def _load_module():
    """Load lora_cache.py by path (avoids importing the sglang package)."""
    spec = importlib.util.spec_from_file_location("lora_cache_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lc = _load_module()

TTL = 24 * 3600.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signed_url(base_path: str, signature: str, host: str = "tos.example.com") -> str:
    return (
        f"https://{host}/adapters-v1/{base_path}?x-id=GetObject"
        f"&X-Amz-Date=20260915T112352Z&X-Amz-Expires=86400&X-Amz-Signature={signature}"
    )


def _make_archive(dst: str, nested: bool = False) -> None:
    """Build a tar.gz containing an adapter (optionally nested one level)."""
    work = tempfile.mkdtemp()
    adapter = os.path.join(work, "adapter") if nested else work
    os.makedirs(adapter, exist_ok=True)
    with open(os.path.join(adapter, "adapter_config.json"), "w") as fh:
        fh.write('{"r": 16}')
    with open(os.path.join(adapter, "adapter_model.safetensors"), "wb") as fh:
        fh.write(b"\x00" * 1024)
    if nested:
        top = os.path.join(work, "payload")
        os.makedirs(top, exist_ok=True)
        shutil.move(adapter, os.path.join(top, "adapter"))
        work = top
    with tarfile.open(dst, "w:gz") as tf:
        for name in os.listdir(work):
            tf.add(os.path.join(work, name), arcname=name, recursive=True)
    shutil.rmtree(work, ignore_errors=True)


def _make_cache_dir(root: str, name: str, mtime: float = None, unloaded_at=None) -> str:
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as fh:
        fh.write("{}")
    with open(os.path.join(path, ".done"), "w") as fh:
        fh.write("ok")
    if unloaded_at is not None:
        with open(os.path.join(path, ".unloaded_at"), "w") as fh:
            fh.write(str(unloaded_at))
    if mtime is not None:
        os.utime(os.path.join(path, ".done"), (mtime, mtime))
        os.utime(path, (mtime, mtime))
    return path


@contextlib.contextmanager
def _cache_env(root: str):
    saved = {k: os.environ.get(k) for k in (
        "SGLANG_LORA_CACHE_DIR", "SGLANG_LORA_CACHE_TTL_SEC",
        "SGLANG_LORA_CACHE_GC", "SGLANG_LORA_CACHE_DEDUPE",
    )}
    os.environ["SGLANG_LORA_CACHE_DIR"] = root
    os.environ.pop("SGLANG_LORA_CACHE_TTL_SEC", None)
    os.environ.pop("SGLANG_LORA_CACHE_GC", None)
    os.environ.pop("SGLANG_LORA_CACHE_DEDUPE", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _tmpdir():
    d = tempfile.mkdtemp(prefix="lora_cache_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 1. content-addressed key
# --------------------------------------------------------------------------- #
def test_cache_key_is_signature_independent():
    artifact = "01a0a4c8-4f18-7d33-85ae-ee49b290fe82/01a0a4c8-4f18-7d33-85ae-ee49b290fe82.tar.zst"
    u1 = _signed_url(artifact, "aaaa1111")
    u2 = _signed_url(artifact, "bbbb2222")
    assert u1 != u2
    assert lc.cache_key(u1) == lc.cache_key(u2), "refreshed signature must reuse the cache key"
    other = _signed_url("01a0dead-0000-0000-0000-000000000000/x.tar.zst", "aaaa1111")
    assert lc.cache_key(u1) != lc.cache_key(other)
    assert lc.artifact_basename(u1) == "01a0a4c8-4f18-7d33-85ae-ee49b290fe82"
    assert lc.artifact_identity(u1).startswith("https://tos.example.com/adapters-v1/")
    assert "Signature" not in lc.artifact_identity(u1)


def test_key_splits_back_into_base():
    url = _signed_url("01a0a4c8-4f18-7d33-85ae-ee49b290fe82/a.tar.zst", "sig")
    key = lc.cache_key(url)
    base, digest = lc.split_key(key)
    assert base == lc.artifact_basename(url)
    assert digest and len(digest) == 8


# --------------------------------------------------------------------------- #
# 2. download once, then reuse (dedup)
# --------------------------------------------------------------------------- #
def test_downloads_once_then_reuses_across_signatures():
    with _tmpdir() as root, _cache_env(root):
        src = os.path.join(root, "src.tar.gz")
        _make_archive(src)
        artifact = "01a0a4c8-4f18-7d33-85ae-ee49b290fe82/a.tar.gz"
        downloads = []

        def fake_download(url, dest):
            downloads.append(url)
            shutil.copyfile(src, dest)

        u1 = _signed_url(artifact, "sig-one")
        p1 = lc.resolve_remote_adapter(u1, u1, download=fake_download, root=root)
        assert len(downloads) == 1
        assert os.path.exists(os.path.join(p1, "adapter_config.json"))
        assert os.path.exists(os.path.join(root, lc.cache_key(u1), ".done"))

        # Same artifact, refreshed pre-signed URL -> cache hit, no download.
        u2 = _signed_url(artifact, "sig-two")
        p2 = lc.resolve_remote_adapter(u2, u2, download=fake_download, root=root)
        assert len(downloads) == 1, "refreshed signed URL must not re-download"
        assert os.path.realpath(p1) == os.path.realpath(p2)
        assert not any(n.endswith(".lock") for n in os.listdir(root))


def test_reuses_legacy_url_hash_directory():
    """Copies written by the old sha1(full_url) key are still reused."""
    with _tmpdir() as root, _cache_env(root):
        base = "01a0a4c8-4f18-7d33-85ae-ee49b290fe82"
        legacy = _make_cache_dir(root, f"{base}-deadbeef")
        artifact = f"{base}/{base}.tar.zst"
        downloads = []

        def fake_download(url, dest):  # pragma: no cover - must not be called
            downloads.append(url)

        p = lc.resolve_remote_adapter(
            _signed_url(artifact, "sig"), _signed_url(artifact, "sig"),
            download=fake_download, root=root,
        )
        assert not downloads, "existing local copy must be reused, not downloaded"
        assert os.path.realpath(p) == os.path.realpath(legacy)


def test_local_path_passthrough():
    with _tmpdir() as root, _cache_env(root):
        assert lc.resolve_remote_adapter("/local/adapter", "/local/adapter", root=root) == "/local/adapter"


def test_nested_adapter_dir_is_hoisted_and_located():
    with _tmpdir() as root, _cache_env(root):
        src = os.path.join(root, "src.tar.gz")
        _make_archive(src, nested=True)
        artifact = "01a0aaaa-0000-0000-0000-000000000000/a.tar.gz"
        url = _signed_url(artifact, "sig")

        def fake_download(u, dest):
            shutil.copyfile(src, dest)

        adapter_dir = lc.resolve_remote_adapter(url, url, download=fake_download, root=root)
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        # cache_dir_for_path must walk back up to the managed directory
        cache_dir = lc.cache_dir_for_path(adapter_dir, root=root)
        assert cache_dir == os.path.join(root, lc.cache_key(url))
        assert lc.cache_dir_for_path("/somewhere/else") is None


def test_zstd_archive_path_when_available():
    """`.tar.zst` URLs go through `tar --zstd` (skipped where zstd is absent)."""
    if not shutil.which("zstd"):
        print("      (skipped: no zstd binary on this host)")
        return
    with _tmpdir() as root, _cache_env(root):
        work = tempfile.mkdtemp()
        with open(os.path.join(work, "adapter_config.json"), "w") as fh:
            fh.write("{}")
        with open(os.path.join(work, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"\x00" * 64)
        src = os.path.join(root, "src.tar.zst")
        subprocess.run(["tar", "--zstd", "-cf", src, "-C", work, "."], check=True)
        shutil.rmtree(work, ignore_errors=True)
        artifact = "01a0bbbb-0000-0000-0000-000000000000/a.tar.zst"
        url = _signed_url(artifact, "sig")

        def fake_download(u, dest):
            shutil.copyfile(src, dest)

        adapter_dir = lc.resolve_remote_adapter(url, url, download=fake_download, root=root)
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        assert os.path.isdir(os.path.join(root, lc.cache_key(url)))


# --------------------------------------------------------------------------- #
# 3. GC: unload + TTL
# --------------------------------------------------------------------------- #
def test_gc_deletes_only_after_ttl_since_unload():
    with _tmpdir() as root:
        expired = _make_cache_dir(root, "expired-11111111", unloaded_at=time.time() - TTL - 60)
        fresh = _make_cache_dir(root, "fresh-22222222", unloaded_at=time.time() - 60)
        never_unloaded = _make_cache_dir(root, "live-33333333")
        res = lc.gc(root=root, ttl=TTL, in_use=[])
        deleted = {os.path.basename(p) for p in res["deleted"]}
        assert deleted == {"expired-11111111"}, deleted
        assert os.path.isdir(fresh) and os.path.isdir(never_unloaded)
        assert res["freed_bytes"] > 0


def test_gc_never_deletes_in_use_dir():
    with _tmpdir() as root:
        expired = _make_cache_dir(root, "expired-11111111", unloaded_at=time.time() - TTL - 60)
        res = lc.gc(root=root, ttl=TTL, in_use=[expired])
        assert res["deleted"] == []
        assert os.path.isdir(expired)


def test_gc_respects_download_lock():
    with _tmpdir() as root:
        incomplete = os.path.join(root, "downloading-44444444")
        os.makedirs(incomplete)
        os.mkdir(incomplete + ".lock")  # as created by resolve_remote_adapter
        old = time.time() - TTL - 3600
        os.utime(incomplete, (old, old))
        res = lc.gc(root=root, ttl=TTL)
        assert res["deleted"] == [], "dir with an active .lock must not be reclaimed"
        # without the lock the stale partial download is garbage
        os.rmdir(incomplete + ".lock")
        res = lc.gc(root=root, ttl=TTL)
        assert [os.path.basename(p) for p in res["deleted"]] == ["downloading-44444444"]


def test_gc_dedupes_same_artifact_keeping_newest():
    with _tmpdir() as root:
        old_ts = time.time() - TTL - 3600
        _make_cache_dir(root, "artifact-aaaaaaaa", mtime=old_ts)
        _make_cache_dir(root, "artifact-bbbbbbbb", mtime=old_ts + 10)
        fresh = _make_cache_dir(root, "artifact-cccccccc")  # touched now -> kept
        res = lc.gc(root=root, ttl=TTL)
        deleted = {os.path.basename(p) for p in res["deleted"]}
        # Keep exactly one copy per artifact: the most recently used one.
        assert deleted == {"artifact-aaaaaaaa", "artifact-bbbbbbbb"}, deleted
        assert os.path.isdir(fresh), "recently used copy is never reclaimed"


def test_gc_dedupe_can_be_disabled():
    with _tmpdir() as root, _cache_env(root):
        os.environ["SGLANG_LORA_CACHE_DEDUPE"] = "0"
        old_ts = time.time() - TTL - 3600
        a = _make_cache_dir(root, "artifact-aaaaaaaa", mtime=old_ts)
        _make_cache_dir(root, "artifact-bbbbbbbb", mtime=old_ts + 10)
        res = lc.gc(root=root, ttl=TTL)
        assert res["deleted"] == [] and os.path.isdir(a)


def test_gc_removes_stale_archives_only():
    with _tmpdir() as root:
        stale = os.path.join(root, "artifact-aaaaaaaa.tar.zst")
        fresh = os.path.join(root, "artifact-bbbbbbbb.tar.zst")
        for path, ts in ((stale, time.time() - TTL - 60), (fresh, time.time())):
            with open(path, "wb") as fh:
                fh.write(b"x" * 16)
            os.utime(path, (ts, ts))
        res = lc.gc(root=root, ttl=TTL)
        assert [os.path.basename(p) for p in res["deleted"]] == ["artifact-aaaaaaaa.tar.zst"]
        assert os.path.exists(fresh)


def test_gc_zero_ttl_is_noop_and_env_parsing():
    with _tmpdir() as root, _cache_env(root):
        expired = _make_cache_dir(root, "expired-11111111", unloaded_at=time.time() - 10)
        assert lc.gc(root=root, ttl=0)["deleted"] == []
        assert os.path.isdir(expired)
        # default TTL is 24h, GC on, dedup on
        assert lc.cache_ttl_sec() == TTL and lc.gc_enabled() and lc.dedupe_reuse_enabled()
        os.environ["SGLANG_LORA_CACHE_TTL_SEC"] = "60"
        os.environ["SGLANG_LORA_CACHE_GC"] = "0"
        assert lc.cache_ttl_sec() == 60.0 and not lc.gc_enabled()
        os.environ["SGLANG_LORA_CACHE_TTL_SEC"] = "not-a-number"
        assert lc.cache_ttl_sec() == TTL  # malformed -> default


# --------------------------------------------------------------------------- #
# 4. unload stamping
# --------------------------------------------------------------------------- #
def test_mark_unloaded_and_touch_loaded_roundtrip():
    with _tmpdir() as root, _cache_env(root):
        d = _make_cache_dir(root, "artifact-aaaaaaaa")
        assert lc.mark_unloaded(d, now=1234.0)
        assert abs(lc._unloaded_ts(d) - 1234.0) < 1e-6
        lc.touch_loaded(d)
        assert lc._unloaded_ts(d) is None, "reusing a dir clears the unload stamp"
        # outside the cache root: never stamped
        outside = tempfile.mkdtemp()
        try:
            assert not lc.mark_unloaded(outside)
        finally:
            shutil.rmtree(outside, ignore_errors=True)


def test_gc_keeps_dir_that_was_reused_after_unload():
    """A reload inside the TTL window must protect the directory from GC."""
    with _tmpdir() as root, _cache_env(root):
        d = _make_cache_dir(root, "artifact-aaaaaaaa")
        lc.mark_unloaded(d, now=time.time() - TTL - 3600)
        assert lc.gc(root=root, ttl=TTL)["deleted"], "expired unload is reclaimed"
        d2 = _make_cache_dir(root, "artifact-bbbbbbbb")
        lc.mark_unloaded(d2, now=time.time() - TTL - 3600)
        lc.touch_loaded(d2)  # reloaded -> stamp cleared
        assert lc.gc(root=root, ttl=TTL)["deleted"] == []
        assert os.path.isdir(d2)


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# GC scheduling (2026-09-17: the sweep must never run on the LoRA update path)
# --------------------------------------------------------------------------- #
def test_gc_scheduler_is_non_blocking_and_single_flight():
    """``schedule()`` 必须立刻返回，且同一时刻只跑一次 sweep。

    现场（AWS B300 对，2026-09-17）：GC 以前在 ``lora_update_lock`` 内同步跑，
    55GB 目录下要分钟级 —— 一次 unload 之后再没有任何 LoRA op 完成（``Start unload``
    14:16:20 无果、14:19 起的 load 全排队），CP 部署因此永远拿不到 per-engine ACK。
    """
    orig_gc = lc.gc
    calls = []

    def slow_gc(**kwargs):
        calls.append(kwargs.get("in_use"))
        time.sleep(0.4)
        return {"deleted": [], "freed_bytes": 0}

    lc.gc = slow_gc
    try:
        sched = lc.CacheGCScheduler()
        t0 = time.time()
        started = sched.schedule(in_use={"/live-1"}, logger=None)
        elapsed = time.time() - t0
        assert started is True
        assert elapsed < 0.2, f"schedule() 阻塞了 {elapsed:.3f}s"
        assert sched.schedule(in_use={"/live-2"}, logger=None) is False
        sched.wait(timeout=5)
        assert calls == [{"/live-1"}], calls
        assert sched.running() is False
    finally:
        lc.gc = orig_gc


def test_gc_scheduler_honours_env_switch():
    """``SGLANG_LORA_CACHE_GC=0`` 时不调度（现场就是用这个开关兜住的）。"""
    orig_gc = lc.gc
    orig_env = os.environ.get("SGLANG_LORA_CACHE_GC")

    def must_not_run(**kwargs):
        raise AssertionError("GC 不该被调度")

    lc.gc = must_not_run
    os.environ["SGLANG_LORA_CACHE_GC"] = "0"
    try:
        assert lc.CacheGCScheduler().schedule(in_use=set(), logger=None) is False
    finally:
        lc.gc = orig_gc
        if orig_env is None:
            os.environ.pop("SGLANG_LORA_CACHE_GC", None)
        else:
            os.environ["SGLANG_LORA_CACHE_GC"] = orig_env


def test_gc_scheduler_swallows_errors():
    """GC 抛异常不能影响调用方 —— 它现在跑在后台线程里。"""
    orig_gc = lc.gc

    def boom(**kwargs):
        raise RuntimeError("disk on fire")

    lc.gc = boom
    try:
        sched = lc.CacheGCScheduler()
        assert sched.schedule(in_use=set(), logger=None) is True
        sched.wait(timeout=5)
        assert sched.running() is False
    finally:
        lc.gc = orig_gc

def _run_all():
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
        and not inspect.isgeneratorfunction(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
