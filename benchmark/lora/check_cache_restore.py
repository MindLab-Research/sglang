"""Verify actual KV prefix hits on D and M using identical natural token paths.

Only output-token logprobs on an equal greedy path are checked. Full-prompt
precision and adapter identity isolation require their separate matrices.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)

    def call(side, path, data=None):
        url = {"D": a.dynamic_url, "M": a.merged_url}[side]
        req = Request(
            url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=7200) as r:
            raw = r.read().decode()
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        with (a.output / "api.jsonl").open("a") as f:
            f.write(
                json.dumps({"side": side, "path": path, "request": data, "body": body})
                + "\n"
            )
        return body

    ids = max(json.loads(a.corpus.read_text()), key=lambda x: len(x["input_ids"]))[
        "input_ids"
    ]
    assert len(ids) >= 128, "use a prefix long enough to create reusable pages"
    request = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    call(
        "D",
        "/load_lora_adapter",
        {"lora_name": a.adapter_name, "lora_path": a.adapter_path},
    )
    try:
        responses = {}
        for side in ("D", "M"):
            info = call(side, "/get_server_info")
            (a.output / f"{side}-server-info.json").write_text(
                json.dumps(info, indent=2)
            )
            call(side, "/flush_cache", {})
            for mode in ("cold", "hot"):
                body = call(
                    side,
                    "/generate",
                    request | ({"lora_path": a.adapter_name} if side == "D" else {}),
                )
                (a.output / f"{side}-{mode}.json").write_text(json.dumps(body))
                hit = body["meta_info"]["cached_tokens"]
                assert hit == 0 if mode == "cold" else hit > 0, (side, mode, hit)
                responses[side + "-" + mode] = body
        ref = responses["M-cold"]
        assert len(ref["output_ids"]) == 16
        ref_lp = ref["meta_info"]["output_token_logprobs"]
        assert len(ref_lp) == 16 and [r[1] for r in ref_lp] == ref["output_ids"]
        rows = []
        for label, body in responses.items():
            assert (
                body["output_ids"] == ref["output_ids"]
            ), "natural paths differ; use forced-prefix scorer"
            lp = body["meta_info"]["output_token_logprobs"]
            assert len(lp) == 16 and [r[1] for r in lp] == ref["output_ids"]
            errors = sorted(abs(x[0] - y[0]) for x, y in zip(lp, ref_lp))
            assert all(math.isfinite(x) for x in errors)
            mean, p99 = (
                sum(errors) / len(errors),
                errors[math.ceil(0.99 * len(errors)) - 1],
            )
            assert mean <= 0.01 and p99 <= 0.1, (label, mean, p99)
            rows.append(
                {
                    "label": label,
                    "cached_tokens": body["meta_info"]["cached_tokens"],
                    "mean_abs_nat": mean,
                    "p99_abs_nat": p99,
                    "max_abs_nat": max(errors),
                }
            )
        for side in ("D", "M"):
            x, y = [
                responses[side + "-" + mode]["meta_info"]["output_token_logprobs"]
                for mode in ("cold", "hot")
            ]
            assert all(
                p[0] == q[0] for p, q in zip(x, y)
            ), "cold/hot self drift under deterministic configuration"
        summary = {
            "passed": True,
            "prompt_tokens": len(ids),
            "output_scores_per_response": 16,
            "rows": rows,
            "cold_hot_repeat_noise_max_abs_nat": 0,
            "scope": "One prefix, actual D/M cache hits and equal natural output paths; no full-prefill or cross-adapter isolation acceptance",
        }
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        call("D", "/unload_lora_adapter", {"lora_name": a.adapter_name})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dynamic-url", required=True)
    p.add_argument("--merged-url", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--adapter-name", default="mol_cache_check")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
