"""Check that one server scores a causal prefix consistently across lengths.

This is a small diagnostic, not dynamic-versus-merged acceptance. Use the same
frozen token corpus as compare_merged.py. Compare each prompt input logprob with
an independently executed prefix's next-token logprob for the SAME token ID.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


def call(url, path, data=None):
    request = Request(
        url + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except ValueError:
        return body


def main(args):
    corpus = json.loads(args.corpus.read_text())
    if args.case_id:
        matches = [case for case in corpus if case["id"] == args.case_id]
        if len(matches) != 1:
            raise ValueError("--case-id must identify exactly one corpus entry")
        case = matches[0]
    else:
        case = max(corpus, key=lambda case: len(case["input_ids"]))
    tokens = case["input_ids"]
    positions = sorted({p for p in args.positions if 0 < p < len(tokens)})
    if not positions:
        raise ValueError("At least one position must be inside the prompt")
    if bool(args.adapter_name) != bool(args.adapter_path):
        raise ValueError("--adapter-name and --adapter-path must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request-config.json").write_text(
        json.dumps(
            {
                "url": args.url,
                "case": case,
                "positions": positions,
                "adapter_name": args.adapter_name,
                "adapter_path": args.adapter_path,
            },
            indent=2,
        )
    )
    if args.adapter_name:
        call(
            args.url,
            "/load_lora_adapter",
            {"lora_name": args.adapter_name, "lora_path": args.adapter_path},
        )
    request = {
        "input_ids": tokens,
        "return_logprob": True,
        "logprob_start_len": 0,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    }
    if args.adapter_name:
        request["lora_path"] = args.adapter_name
    call(args.url, "/flush_cache", {})
    full = call(args.url, "/generate", request)
    (args.output / "full.json").write_text(json.dumps(full))
    scores = full["meta_info"]["input_token_logprobs"]
    assert len(scores) == len(tokens)
    rows = []
    for position in positions:
        call(args.url, "/flush_cache", {})
        prefix = call(
            args.url,
            "/generate",
            request
            | {
                "input_ids": tokens[:position],
                "token_ids_logprob": [tokens[position]],
            },
        )
        (args.output / f"prefix-{position}.json").write_text(json.dumps(prefix))
        a = scores[position]
        b = prefix["meta_info"]["output_token_ids_logprobs"][0][0]
        assert a[1] == b[1] == tokens[position], (position, a, b)
        error = abs(a[0] - b[0])
        assert math.isfinite(error), (position, a, b)
        rows.append(
            {
                "position": position,
                "token_id": tokens[position],
                "full_logprob": a[0],
                "prefix_logprob": b[0],
                "abs_nat": error,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    errors = sorted(row["abs_nat"] for row in rows)
    summary = {
        "case": case["id"],
        "adapter": args.adapter_name,
        "positions": rows,
        "mean_abs_nat": sum(errors) / len(errors),
        "p99_abs_nat": errors[math.ceil(0.99 * len(errors)) - 1],
        "max_abs_nat": max(errors),
        "exact_matches": sum(error == 0 for error in errors),
        "count": len(errors),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--positions", type=int, nargs="+", default=[1, 63, 127, 950, 1024]
    )
    parser.add_argument("--adapter-name")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
