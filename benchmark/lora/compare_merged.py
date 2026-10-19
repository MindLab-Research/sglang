"""Compare dynamic LoRA with an independently deployed merged model.

The two servers must use --enable-custom-logit-processor and share --capture-dir
with this client. The test processor records UNMODIFIED FP32 log-softmax before
forcing a common continuation, so steps >=1 execute actual cached decode.
API output-token logprobs after forcing are intentionally never used as scores.
"""

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class CaptureAndForce(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        import json

        for i, params in enumerate(custom_param_list):
            # Overlap scheduling can leave req.output_ids one token behind
            # the sampler. This request-owned dict persists across batches.
            step = params.get("_capture_step", 0)
            params["_capture_step"] = step + 1
            token = params["continuation"][step]
            original = logits[i].float()
            score = original.log_softmax(-1)[token].item()
            argmax = original.argmax(-1).item()
            row = {
                "step": step,
                "phase": "prefill_last" if step == 0 else "decode",
                "token": token,
                "logprob": score,
                "argmax": argmax,
            }
            with open(params["capture_file"], "a") as f:
                f.write(json.dumps(row) + "\n")
            logits[i].fill_(-float("inf"))
            logits[i, token] = 0
        return logits


def call(url, path, data=None):
    req = Request(
        url + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=7200) as f:
        body = f.read().decode()
    try:
        return json.loads(body)
    except ValueError:
        return body


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def stats(pairs):
    if not pairs:
        return {"count": 0, "passes_logprob_gate": None, "status": "not_measured"}
    errors = [abs(a[0] - b[0]) for a, b in pairs]
    assert errors and all(math.isfinite(x) for x in errors)
    return {
        "count": len(errors),
        "mean_abs_nat": sum(errors) / len(errors),
        "p99_abs_nat": quantile(errors, 0.99),
        "max_abs_nat": max(errors),
        "argmax_matches": sum(a[1] == b[1] for a, b in pairs),
        "argmax_match_fraction": sum(a[1] == b[1] for a, b in pairs) / len(errors),
        "passes_logprob_gate": sum(errors) / len(errors) <= 0.01
        and quantile(errors, 0.99) <= 0.1,
    }


def main(args):
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus.read_text())[: args.limit]
    endpoints = {"D": args.dynamic_url, "M": args.merged_url}
    call(
        args.dynamic_url,
        "/load_lora_adapter",
        {"lora_name": args.adapter_name, "lora_path": args.adapter_path},
    )
    processor = CaptureAndForce.to_str()
    all_pairs = {"prefill": [], "prefill_last": [], "decode": []}
    repeat_pairs = {side: {phase: [] for phase in all_pairs} for side in endpoints}
    per_case = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for case in corpus:
            first_scores = {}
            prompt = case["input_ids"]
            common = {
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.decode_tokens,
                    "ignore_eos": True,
                },
            }
            reference = call(args.merged_url, "/generate", common)
            (args.output / f"{case['id']}-reference-response.json").write_text(
                json.dumps(reference)
            )
            continuation = reference["output_ids"]
            assert len(continuation) == args.decode_tokens, (
                f"{case['id']}: requested {args.decode_tokens} output tokens, "
                f"got {len(continuation)}; see saved reference response"
            )
            (args.output / f"{case['id']}-continuation.json").write_text(
                json.dumps(
                    {
                        "input_ids": prompt,
                        "continuation": continuation,
                        "reference": "M greedy, frozen before all D/M scoring rounds",
                    }
                )
            )
            for repeat in range(args.rounds):
                for cache in ("cold", "hot"):
                    if cache == "cold":
                        for url in endpoints.values():
                            call(url, "/flush_cache", {})
                    futures = {}
                    captures = {}
                    started = time.monotonic()
                    for side, url in endpoints.items():
                        name = f"{case['id']}-r{repeat}-{cache}-{side}"
                        path = args.capture_dir / f"{name}.jsonl"
                        if path.exists():
                            raise ValueError(f"Capture file already exists: {path}")
                        captures[side] = path
                        req = common | {
                            "custom_logit_processor": processor,
                            "sampling_params": common["sampling_params"]
                            | {
                                "custom_params": {
                                    "capture_file": str(path),
                                    "continuation": continuation,
                                }
                            },
                        }
                        if not args.skip_prefill_logprobs:
                            req.update(
                                return_logprob=True,
                                logprob_start_len=0,
                                top_logprobs_num=1,
                            )
                        if side == "D":
                            req["lora_path"] = args.adapter_name
                        futures[side] = executor.submit(call, url, "/generate", req)
                    results = {
                        side: future.result() for side, future in futures.items()
                    }
                    records = {}
                    for side, response in results.items():
                        (
                            args.output / f"{case['id']}-r{repeat}-{cache}-{side}.json"
                        ).write_text(json.dumps(response))
                        assert response["output_ids"] == continuation, (side, response)
                        rows = [
                            json.loads(x)
                            for x in captures[side].read_text().splitlines()
                        ]
                        assert [x["step"] for x in rows] == list(
                            range(len(continuation))
                        )
                        records[side] = rows
                    case_pairs = {"prefill": [], "prefill_last": [], "decode": []}
                    d_meta, m_meta = (results[s]["meta_info"] for s in ("D", "M"))
                    if not args.skip_prefill_logprobs:
                        d_lp, m_lp = (
                            d_meta["input_token_logprobs"],
                            m_meta["input_token_logprobs"],
                        )
                        assert len(d_lp) == len(m_lp) == len(prompt)
                        d_top, m_top = (
                            d_meta["input_top_logprobs"],
                            m_meta["input_top_logprobs"],
                        )
                        for pos in range(1, len(prompt)):
                            assert d_lp[pos][1] == m_lp[pos][1] == prompt[pos]
                            case_pairs["prefill"].append(
                                (
                                    (d_lp[pos][0], d_top[pos][0][1]),
                                    (m_lp[pos][0], m_top[pos][0][1]),
                                )
                            )
                    for d, m in zip(records["D"], records["M"]):
                        phase = d["phase"]
                        case_pairs[phase].append(
                            ((d["logprob"], d["argmax"]), (m["logprob"], m["argmax"]))
                        )
                    row = {
                        "case": case["id"],
                        "round": repeat,
                        "cache": cache,
                        "cached_tokens": {
                            side: result["meta_info"].get("cached_tokens")
                            for side, result in results.items()
                        },
                        "prompt_tokens": len(prompt),
                        "continuation_tokens": len(continuation),
                        "elapsed_s": time.monotonic() - started,
                    }
                    for phase, pairs in case_pairs.items():
                        all_pairs[phase].extend(pairs)
                        row[phase] = stats(pairs)
                        if not pairs:
                            continue
                        row[phase]["worst_positions"] = sorted(
                            (
                                {
                                    "position": i + 1 if phase == "prefill" else i,
                                    "D_logprob": d[0],
                                    "M_logprob": m[0],
                                    "abs_nat": abs(d[0] - m[0]),
                                    "D_argmax": d[1],
                                    "M_argmax": m[1],
                                }
                                for i, (d, m) in enumerate(pairs)
                            ),
                            key=lambda x: x["abs_nat"],
                            reverse=True,
                        )[:5]
                        for side_index, side in enumerate(endpoints):
                            scores = [pair[side_index] for pair in pairs]
                            key = (side, phase)
                            if key not in first_scores:
                                first_scores[key] = scores
                            else:
                                repeat_pairs[side][phase].extend(
                                    zip(first_scores[key], scores)
                                )
                    per_case.append(row)
                    with (args.output / "cases.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
    summary = {
        "adapter": args.adapter_name,
        "cases": len(corpus),
        "rounds": args.rounds,
        "decode_tokens": args.decode_tokens,
        "prefill_logprobs_measured": not args.skip_prefill_logprobs,
        "cache_modes": ["cold", "hot"],
        "cache_mode_note": (
            "hot means no explicit flush; scoring from logprob_start_len=0 "
            "can prevent prompt prefix reuse. Check per-case cached_tokens; "
            "this run does not establish prefix-cache restore correctness."
        ),
        "gates": {"mean_abs_nat": 0.01, "p99_abs_nat": 0.1},
        "scores": {phase: stats(pairs) for phase, pairs in all_pairs.items()},
        "repeat_noise": {
            side: {phase: stats(pairs) for phase, pairs in phases.items() if pairs}
            for side, phases in repeat_pairs.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-url", default="http://127.0.0.1:30000")
    parser.add_argument("--merged-url", default="http://127.0.0.1:30002")
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--skip-prefill-logprobs",
        action="store_true",
        help=(
            "Only score prefill-last and cached decode. Avoids full-prompt logits "
            "for long-context diagnostics; prompt-token equivalence is not measured."
        ),
    )
    main(parser.parse_args())
