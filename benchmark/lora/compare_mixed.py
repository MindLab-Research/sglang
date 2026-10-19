"""Compare isolated and mixed LoRA requests on one dedicated SGLang server.

This checks routing/batching consistency, not dynamic-versus-merged equivalence.
Adapters JSON is a list of {"name": ..., "path": ...}; base is always included.
The server requires --enable-custom-logit-processor and a shared capture path.
"""

import argparse
import json
import random
from pathlib import Path

from compare_merged import call, stats
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class CaptureAndForce(CustomLogitProcessor):
    # Defined in the executable so dill embeds it rather than requiring the
    # benchmark module to be importable in the serving worker.
    def __call__(self, logits, custom_param_list=None):
        import json

        import torch.distributed as dist

        for i, params in enumerate(custom_param_list):
            step = params.get("_capture_step", 0)
            params["_capture_step"] = step + 1
            token = params["continuation"][step]
            original = logits[i].float()
            row = {
                "step": step,
                "token": token,
                "logprob": original.log_softmax(-1)[token].item(),
                "argmax": original.argmax(-1).item(),
            }
            with open(params["capture_file"] + f".rank-{dist.get_rank()}", "a") as f:
                f.write(json.dumps(row) + "\n")
            logits[i].fill_(-float("inf"))
            logits[i, token] = 0
        return logits


def main(args):
    adapters = json.loads(args.adapters.read_text())
    names = [a["name"] for a in adapters]
    assert len(names) == len(set(names)) and all(names)
    corpus = json.loads(args.corpus.read_text())[: args.cases]
    assert corpus and len({c["id"] for c in corpus}) == len(corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    info = call(args.url, "/get_server_info")
    (args.output / "server-info.json").write_text(json.dumps(info, indent=2))
    config = info.get("server_args", info)
    replicas = config["tp_size"] // config["attn_dp_size"]
    processor = CaptureAndForce.to_str()
    loaded = []
    references = []
    sampling = {
        "temperature": 0,
        "max_new_tokens": args.decode_tokens,
        "ignore_eos": True,
    }

    def request(ref, label):
        capture = args.capture_dir / (label + ".jsonl")
        assert not list(capture.parent.glob(capture.name + ".rank-*")), capture
        return {
            "input_ids": ref["input_ids"],
            "lora_path": ref["adapter"],
            "custom_logit_processor": processor,
            "sampling_params": sampling
            | {
                "custom_params": {
                    "capture_file": str(capture),
                    "continuation": ref["continuation"],
                }
            },
        }

    def read_scores(label, continuation):
        paths = sorted(args.capture_dir.glob(label + ".jsonl.rank-*"))
        assert len(paths) == replicas, (label, paths, replicas)
        rank_rows = [[json.loads(s) for s in p.read_text().splitlines()] for p in paths]
        rows = rank_rows[0]
        assert all(other == rows for other in rank_rows[1:])
        assert [r["step"] for r in rows] == list(range(args.decode_tokens))
        assert [r["token"] for r in rows] == continuation
        return [(r["logprob"], r["argmax"]) for r in rows]

    try:
        for adapter in adapters:
            call(
                args.url,
                "/load_lora_adapter",
                {
                    "lora_name": adapter["name"],
                    "lora_path": adapter["path"],
                    "pinned": config["attn_dp_size"] > 1,
                },
            )
            loaded.append(adapter["name"])
        for adapter in [None] + names:
            for case in corpus:
                call(args.url, "/flush_cache", {})
                ref = {
                    "id": f"{adapter or 'base'}-{case['id']}",
                    "adapter": adapter,
                    "input_ids": case["input_ids"],
                }
                greedy = call(
                    args.url,
                    "/generate",
                    {
                        "input_ids": ref["input_ids"],
                        "lora_path": adapter,
                        "sampling_params": sampling,
                    },
                )
                ref["continuation"] = greedy["output_ids"]
                assert len(ref["continuation"]) == args.decode_tokens
                (args.output / (ref["id"] + "-greedy.json")).write_text(
                    json.dumps(greedy)
                )
                call(args.url, "/flush_cache", {})
                label = ref["id"] + "-isolated"
                response = call(args.url, "/generate", request(ref, label))
                assert response["output_ids"] == ref["continuation"]
                (args.output / (label + ".json")).write_text(json.dumps(response))
                ref["scores"] = read_scores(label, ref["continuation"])
                references.append(ref)
        (args.output / "references.json").write_text(json.dumps(references))
        rng = random.Random(args.seed)
        pairs = {"prefill_last": [], "decode": []}
        cases = []
        for repeat in range(args.rounds):
            order = list(range(len(references)))
            rng.shuffle(order)
            call(args.url, "/flush_cache", {})
            labels = [f"r{repeat}-{references[i]['id']}" for i in order]
            requests = [
                request(references[i], label) for i, label in zip(order, labels)
            ]
            batch = {key: [req[key] for req in requests] for key in requests[0]}
            responses = call(args.url, "/generate", batch)
            assert len(responses) == len(order)
            (args.output / f"mixed-r{repeat}.json").write_text(json.dumps(responses))
            for i, label, response in zip(order, labels, responses):
                ref = references[i]
                assert response["output_ids"] == ref["continuation"]
                actual = read_scores(label, ref["continuation"])
                row = {
                    "id": ref["id"],
                    "round": repeat,
                    "cached_tokens": response["meta_info"].get("cached_tokens"),
                }
                for phase, selected in [
                    ("prefill_last", slice(0, 1)),
                    ("decode", slice(1, None)),
                ]:
                    compared = list(zip(actual[selected], ref["scores"][selected]))
                    pairs[phase].extend(compared)
                    row[phase] = stats(compared)
                cases.append(row)
            print(
                json.dumps(
                    {
                        "round": repeat,
                        "requests": len(responses),
                        "scores": {phase: stats(p) for phase, p in pairs.items()},
                    }
                ),
                flush=True,
            )
        summary = {
            "purpose": "isolated versus mixed scheduling; not D/M acceptance",
            "adapters_including_base": len(adapters) + 1,
            "isolated_requests": len(references),
            "mixed_requests": len(cases),
            "seed": args.seed,
            "scores": {phase: stats(p) for phase, p in pairs.items()},
            "cases": cases,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        assert all(
            row[phase]["passes_logprob_gate"]
            for row in cases
            for phase in pairs
        ), summary
    finally:
        for name in reversed(loaded):
            call(args.url, "/unload_lora_adapter", {"lora_name": name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    main(parser.parse_args())
