"""CPU HTTP protocol tests; these do not establish serving/GPU correctness."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/lora" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperationalClients(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus.json"
        self.corpus.write_text(
            json.dumps([{"id": "long", "input_ids": list(range(256))}])
        )
        self.adapters = self.root / "adapters.json"
        self.adapters.write_text(
            json.dumps([{"name": "A", "path": "/A"}, {"name": "B", "path": "/B"}])
        )
        self.loaded, self.cached = {}, set()
        self.streaming = threading.Event()
        self.stream_done = threading.Event()
        self.zero_hit, self.drift, self.different_tokens = False, False, False
        self.owner = self
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def send(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/get_server_info":
                    return self.send(
                        {
                            "server_args": {
                                "tp_size": 1,
                                "attn_dp_size": 1,
                                "enable_lora": True,
                                "context_length": 4096,
                                "max_loaded_loras": 2,
                            }
                        }
                    )
                return self.send("ok")

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/flush_cache":
                    owner.cached.clear()
                    return self.send("flushed")
                if self.path == "/load_lora_adapter":
                    if "does-not-exist" in data["lora_path"]:
                        return self.send("missing adapter", 400)
                    owner.loaded[data["lora_name"]] = data["lora_path"]
                    return self.send("loaded")
                if self.path == "/unload_lora_adapter":
                    owner.loaded.pop(data["lora_name"], None)
                    if owner.streaming.is_set():
                        owner.stream_done.wait(2)
                    return self.send("unloaded")
                alias = data.get("lora_path")
                if alias and alias not in owner.loaded:
                    return self.send("unloaded", 400)
                n = data["sampling_params"]["max_new_tokens"]
                value = 3 if owner.loaded.get(alias) == "/B" else 2
                if owner.different_tokens and alias:
                    value += 5
                hit = alias in owner.cached
                owner.cached.add(alias)

                def body(count):
                    return {
                        "output_ids": [value] * count,
                        "meta_info": {
                            "cached_tokens": 128 if hit and not owner.zero_hit else 0,
                            "output_token_logprobs": [
                                [
                                    -1.0 + (0.5 if owner.drift and alias else 0),
                                    value,
                                    None,
                                ]
                            ]
                            * count,
                        },
                    }

                if data.get("stream"):
                    owner.streaming.set()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(("data: " + json.dumps(body(1)) + "\n\n").encode())
                    self.wfile.flush()
                    deadline = time.monotonic() + 2
                    while alias in owner.loaded and time.monotonic() < deadline:
                        time.sleep(0.001)
                    # Keep generation in flight while the client observes unregister.
                    time.sleep(0.1)
                    self.wfile.write(
                        (
                            "data: " + json.dumps(body(n)) + "\n\ndata: [DONE]\n\n"
                        ).encode()
                    )
                    owner.streaming.clear()
                    owner.stream_done.set()
                    return
                self.send(body(n))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def cache_args(self):
        return SimpleNamespace(
            output=self.root / "cache",
            dynamic_url=self.url,
            merged_url=self.url,
            corpus=self.corpus,
            adapter_name="check",
            adapter_path="/A",
        )

    def test_full_lifecycle_protocol_and_cleanup(self):
        args = SimpleNamespace(
            url=self.url,
            corpus=self.corpus,
            adapters=self.adapters,
            output=self.root / "life",
            timeout=5,
            aliases=3,
            cycles=100,
            bad_loads=100,
            lru_rounds=36,
            alias_prefix="test_",
            missing_adapter="/does-not-exist",
        )
        load("check_lifecycle").main(args)
        summary = json.loads((args.output / "summary.json").read_text())
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["serial_cycles"], 100)
        self.assertEqual(summary["lru_requests"], 108)
        self.assertFalse(self.loaded)

    def test_actual_cache_hit_success(self):
        args = self.cache_args()
        load("check_cache_restore").main(args)
        self.assertTrue(
            json.loads((args.output / "summary.json").read_text())["passed"]
        )
        self.assertFalse(self.loaded)

    def test_zero_hit_cannot_pass(self):
        self.zero_hit = True
        args = self.cache_args()
        with self.assertRaises(AssertionError):
            load("check_cache_restore").main(args)
        self.assertFalse((args.output / "summary.json").exists())
        self.assertFalse(self.loaded)

    def test_mismatched_native_path_cannot_pass(self):
        self.different_tokens = True
        args = self.cache_args()
        with self.assertRaises(AssertionError):
            load("check_cache_restore").main(args)
        self.assertFalse((args.output / "summary.json").exists())
        self.assertFalse(self.loaded)

    def test_large_score_drift_cannot_pass(self):
        self.drift = True
        args = self.cache_args()
        with self.assertRaises(AssertionError):
            load("check_cache_restore").main(args)
        self.assertFalse((args.output / "summary.json").exists())
        self.assertFalse(self.loaded)


if __name__ == "__main__":
    unittest.main()
