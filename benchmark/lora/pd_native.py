"""Direct SGLang prefill/decode protocol for controlled acceptance tests."""

import concurrent.futures
import copy
import json
import time
import uuid
from pathlib import Path

from compare_merged import call


class NativePDClient:
    def __init__(self, prefill_url, decode_url, bootstrap_port, output):
        self.prefill_url = prefill_url
        self.decode_url = decode_url
        self.bootstrap_port = bootstrap_port
        self.output = Path(output)
        self.serial = 0

    def call(self, url, path, data=None):
        if path in ("/flush_cache", "/load_lora_adapter", "/unload_lora_adapter"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                futures = [
                    ex.submit(call, u, path, data)
                    for u in (self.prefill_url, self.decode_url)
                ]
                return [f.result() for f in futures]
        if path != "/generate":
            return call(url, path, data)
        requests = [
            {k: v[i] for k, v in data.items()} for i in range(len(data["input_ids"]))
        ]
        prefill, decode = [], []
        for request in requests:
            request = copy.deepcopy(request)
            request.update(
                bootstrap_host="127.0.0.1",
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=uuid.uuid4().int & ((1 << 63) - 1),
            )
            p, d = copy.deepcopy(request), copy.deepcopy(request)
            if "custom_logit_processor" in request:
                for row, suffix, step in ((p, ".prefill", 0), (d, ".decode", 1)):
                    params = row["sampling_params"]["custom_params"]
                    params["capture_file"] += suffix
                    params["capture_start_step"] = step
            prefill.append(p)
            decode.append(d)

        def pack(rows):
            return {k: [r[k] for r in rows] for k in rows[0]}

        def timed(endpoint, payload):
            started = time.monotonic()
            return call(endpoint, path, pack(payload)), time.monotonic() - started

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(timed, self.prefill_url, prefill)
            df = ex.submit(timed, self.decode_url, decode)
            (p_result, p_seconds), (d_result, d_seconds) = pf.result(), df.result()
        serial = self.serial
        self.serial += 1
        for side, payload, result, elapsed in (
            ("prefill", prefill, p_result, p_seconds),
            ("decode", decode, d_result, d_seconds),
        ):
            (self.output / f"pd-{serial}-{side}.json").write_text(
                json.dumps(
                    {
                        "requests": payload,
                        "responses": result,
                        "http_elapsed_s": elapsed,
                    }
                )
            )
        assert len(p_result) == len(d_result) == len(requests)
        merged = copy.deepcopy(d_result)
        for p, d in zip(p_result, merged):
            assert len(p["output_ids"]) == 1
            assert p["output_ids"][0] == d["output_ids"][0]
            for key, value in p["meta_info"].items():
                if key.startswith("input_"):
                    d["meta_info"][key] = value
        return merged
