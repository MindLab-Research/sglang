"""Exercise adapter alias reuse, failed loads, draining and LRU on a dedicated server.

This is a deterministic TP1 diagnostic against same-service references, not D/M
precision acceptance. Never run against a shared production service.
"""

import argparse
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()

    def call(path, data=None):
        req = Request(
            a.url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urlopen(req, timeout=a.timeout) as r:
                status, raw = r.status, r.read().decode()
        except HTTPError as e:
            status, raw = e.code, e.read().decode()
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        with lock, (a.output / "api.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "path": path,
                        "request": data,
                        "status": status,
                        "body": body,
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        return status, body

    def ok(path, data=None):
        status, body = call(path, data)
        assert status == 200, (path, status, body)
        return body

    info = ok("/get_server_info")
    (a.output / "server-info.json").write_text(json.dumps(info, indent=2))
    cfg = info.get("server_args", info)
    assert cfg["tp_size"] == 1 and cfg.get("attn_dp_size", 1) == 1, "TP1 only"
    assert cfg["enable_lora"]
    assert (
        a.aliases > cfg["max_loaded_loras"]
    ), "aliases must exceed the server loading budget"
    corpus = json.loads(a.corpus.read_text())
    ids = max(corpus, key=lambda row: len(row["input_ids"]))["input_ids"]
    assert 2 <= len(ids) and len(ids) + 256 <= cfg["context_length"]
    adapters = json.loads(a.adapters.read_text())
    assert len(adapters) >= 2 and len({x["path"] for x in adapters}) == len(adapters)
    assert a.cycles >= len(adapters) and a.bad_loads > 0 and a.lru_rounds > 0
    (a.output / "config.json").write_text(
        json.dumps(
            {**vars(a), "adapters": adapters, "input_ids": ids}, default=str, indent=2
        )
    )
    active = set()

    def load(alias, path):
        ok("/load_lora_adapter", {"lora_name": alias, "lora_path": path})
        active.add(alias)

    def unload(alias):
        ok("/unload_lora_adapter", {"lora_name": alias})
        active.discard(alias)

    def request(alias, count=16):
        return {
            "input_ids": ids,
            "lora_path": alias,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }

    def compare(body, ref, count=16):
        assert (
            len(body["output_ids"]) == count and body["output_ids"] == ref["output_ids"]
        )
        x, y = [z["meta_info"]["output_token_logprobs"] for z in (body, ref)]
        assert len(x) == len(y) == count
        assert [z[1] for z in x] == [z[1] for z in y] == body["output_ids"]
        assert all(p[0] == q[0] for p, q in zip(x, y)), "repeat logprob drift"

    alias = a.alias_prefix + "reuse"
    refs = {}
    try:
        for row in adapters:
            load(alias, row["path"])
            ok("/flush_cache", {})
            refs[row["path"]] = ok("/generate", request(alias))
            unload(alias)
        (a.output / "baseline.json").write_text(json.dumps(refs))
        for i in range(a.cycles):
            path = adapters[i % len(adapters)]["path"]
            load(alias, path)
            compare(ok("/generate", request(alias)), refs[path])
            unload(alias)
            assert (
                call("/generate", request(alias))[0] == 400
            ), "unloaded alias accepted"
        load(alias, adapters[0]["path"])
        # A failed load may not poison an unrelated healthy, resident adapter.
        for _ in range(a.bad_loads):
            status, _ = call(
                "/load_lora_adapter",
                {"lora_name": a.alias_prefix + "bad", "lora_path": a.missing_adapter},
            )
            assert (
                400 <= status < 500
            ), "invalid adapter must be rejected with a client error"
            compare(ok("/generate", request(alias)), refs[adapters[0]["path"]])
        ok("/flush_cache", {})
        reference = ok("/generate", request(alias, 256))
        assert len(reference["output_ids"]) == 256
        ok("/flush_cache", {})
        first = threading.Event()
        events, failures, timing = [], [], {}

        def stream():
            try:
                req = Request(
                    a.url + "/generate",
                    data=json.dumps(request(alias, 256) | {"stream": True}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=a.timeout) as r:
                    for raw in r:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        row = json.loads(payload)
                        events.append(row)
                        if row.get("output_ids"):
                            timing.setdefault("first_token", time.monotonic())
                            first.set()
                timing["stream_done"] = time.monotonic()
            except Exception as e:  # noqa: BLE001
                failures.append(repr(e))
                first.set()

        t = threading.Thread(target=stream, daemon=True)
        t.start()
        assert first.wait(a.timeout) and not failures and "first_token" in timing
        assert "stream_done" not in timing, "no measured overlap with generation"
        timing["unload_started"] = time.monotonic()
        unloaded = []

        def drain():
            try:
                unloaded.append(ok("/unload_lora_adapter", {"lora_name": alias}))
                timing["unload_done"] = time.monotonic()
            except Exception as e:  # noqa: BLE001
                failures.append(repr(e))

        u = threading.Thread(target=drain, daemon=True)
        u.start()
        rejected = False
        deadline = time.monotonic() + a.timeout
        while time.monotonic() < deadline and "stream_done" not in timing:
            status, _ = call("/generate", request(alias, 1))
            if status == 400:
                rejected = "stream_done" not in timing
                break
            assert status == 200, status
        t.join(a.timeout)
        u.join(a.timeout)
        assert not t.is_alive() and not u.is_alive() and not failures, failures
        assert rejected and unloaded, "new request was not rejected during draining"
        active.discard(alias)
        compare(events[-1], reference, 256)
        (a.output / "inflight-events.json").write_text(json.dumps(events))
        (a.output / "inflight-reference.json").write_text(json.dumps(reference))
        aliases = [a.alias_prefix + f"lru_{i}" for i in range(a.aliases)]
        for i, name in enumerate(aliases):
            load(name, adapters[i % len(adapters)]["path"])
        for _ in range(a.lru_rounds):
            for i, name in enumerate(aliases):
                compare(
                    ok("/generate", request(name)),
                    refs[adapters[i % len(adapters)]["path"]],
                )
        for name in aliases:
            unload(name)
            assert call("/generate", request(name))[0] == 400
        ok("/health")
        summary = {
            "passed": True,
            "serial_cycles": a.cycles,
            "invalid_loads": a.bad_loads,
            "lru_requests": a.aliases * a.lru_rounds,
            "lru_aliases": a.aliases,
            "inflight_output_tokens": 256,
            "new_requests_rejected_while_draining": rejected,
            "timing": timing,
            "max_repeat_logprob_abs_nat": 0,
            "scope": "TP1 deterministic same-service natural-output repeat checks; not independent D/M acceptance",
        }
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        for name in sorted(active):
            call("/unload_lora_adapter", {"lora_name": name})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--adapters", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cycles", type=int, default=100)
    p.add_argument("--bad-loads", type=int, default=100)
    p.add_argument("--aliases", type=int, default=12)
    p.add_argument("--lru-rounds", type=int, default=9)
    p.add_argument("--alias-prefix", default="mol_check_")
    p.add_argument("--missing-adapter", default="/work/mol/does-not-exist")
    p.add_argument("--timeout", type=float, default=300)
    main(p.parse_args())
