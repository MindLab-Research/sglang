"""Export/replay same-topology scores when D and M cannot coexist on the GPUs.

Run `reference` against the independent offline merge, stop it, then `dynamic`
against the base with LoRA. The sealed reference includes all raw responses,
unmodified scores, common continuations and per-file SHA256. Both stages must
use the same topology label and resolved serving configuration.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from compare_merged import call, stats
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class CaptureAndForce(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        import json

        import torch.distributed as dist

        for i, params in enumerate(custom_param_list):
            step = params.get("_capture_step", params.get("capture_start_step", 0))
            params["_capture_step"] = step + 1
            if step >= len(params["continuation"]):
                # Overlap decoding can sample an already-completed row once
                # more before filtering it from the next batch. The response
                # and capture reader still require the exact continuation.
                # Keep the unused logits unchanged and record this separately.
                path = params["capture_file"] + f".unused.rank-{dist.get_rank()}"
                with open(path, "a") as f:
                    f.write(json.dumps({"step": step}) + "\n")
                continue
            token = params["continuation"][step]
            original = logits[i].float()
            row = {
                "step": step,
                "token": token,
                "logprob": original.log_softmax(-1)[token].item(),
                "argmax": original.argmax(-1).item(),
            }
            path = params["capture_file"] + f".rank-{dist.get_rank()}"
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logits[i].fill_(-float("inf"))
            logits[i, token] = 0
        return logits


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch(requests):
    return {key: [row[key] for row in requests] for key in requests[0]}


def read_scores(response, path, case, continuation, replicas, pd=False):
    assert response["output_ids"] == continuation
    rows = []
    for suffix in (".prefill", ".decode") if pd else ("",):
        paths = sorted(path.parent.glob(path.name + suffix + ".rank-*"))
        assert len(paths) == replicas, (paths, replicas)
        rank_rows = [
            [json.loads(line) for line in p.read_text().splitlines()] for p in paths
        ]
        assert all(other == rank_rows[0] for other in rank_rows[1:])
        rows.extend(rank_rows[0])
    assert [row["step"] for row in rows] == list(range(len(continuation)))
    assert [row["token"] for row in rows] == continuation
    meta = response["meta_info"]
    lp, top = meta["input_token_logprobs"], meta["input_top_logprobs"]
    assert len(lp) == len(case["input_ids"])
    assert [row[1] for row in lp[1:]] == case["input_ids"][1:]
    return {
        "prefill": [(v[0], t[0][1]) for v, t in zip(lp[1:], top[1:])],
        "prefill_last": [(rows[0]["logprob"], rows[0]["argmax"])],
        "decode": [(r["logprob"], r["argmax"]) for r in rows[1:]],
    }


def main(args):
    dynamic = args.stage == "dynamic"
    reference = args.reference
    if dynamic:
        seal = json.loads((reference / "seal.json").read_text())
        for filename, sha in seal["files"].items():
            assert digest(reference / filename) == sha, filename
        contract = json.loads((reference / "contract.json").read_text())
        assert contract["topology"] == args.topology
        corpus = contract["corpus"]
        rounds, ntokens, batch_size = (
            contract[k] for k in ("rounds", "decode_tokens", "batch_size")
        )
    else:
        corpus = json.loads(args.corpus.read_text())[: args.limit]
        rounds, ntokens, batch_size = args.rounds, args.decode_tokens, args.batch_size
        assert corpus and len({c["id"] for c in corpus}) == len(corpus)
        assert rounds > 0 and ntokens > 1 and batch_size > 0
        contract = {
            "topology": args.topology,
            "corpus": corpus,
            "rounds": rounds,
            "decode_tokens": ntokens,
            "batch_size": batch_size,
        }
    args.output.mkdir(parents=True, exist_ok=False)
    # Captures are sealed alongside their responses. Both server and client
    # must see this output at the same absolute path.
    captures = args.output / "captures"
    captures.mkdir()
    write(args.output / "contract.json", contract)
    info = call(args.url, "/get_server_info")
    config = info.get("server_args", info)
    assert bool(config["enable_lora"]) == dynamic
    replicas = config["tp_size"] // config["attn_dp_size"]
    assert config["pp_size"] == 1
    write(args.output / "server-info.json", info)
    write(args.output / "model-info.json", call(args.url, "/model_info"))
    write(args.output / "provenance.json", json.loads(args.provenance.read_text()))
    if args.prefill_url:
        from pd_native import NativePDClient

        prefill_info = call(args.prefill_url, "/get_server_info")
        write(args.output / "prefill-server-info.json", prefill_info)
        prefill_config = prefill_info.get("server_args", prefill_info)
        assert prefill_config["disaggregation_mode"] == "prefill"
        assert config["disaggregation_mode"] == "decode"
        assert prefill_config["tp_size"] == config["tp_size"]
        assert prefill_config["attn_dp_size"] == config["attn_dp_size"] == 1
        assert prefill_config["disaggregation_transfer_backend"] != "fake"
        assert config["disaggregation_transfer_backend"] != "fake"
        transport = NativePDClient(
            args.prefill_url, args.url, args.bootstrap_port, args.output
        )
        request_call = transport.call
    else:
        request_call = call
    if dynamic:
        ref_info = json.loads((reference / "server-info.json").read_text())
        # Model paths and LoRA support deliberately differ. Numerical execution
        # settings and topology must agree; keep full configs for audit too.
        keys = (
            "tp_size",
            "dp_size",
            "attn_dp_size",
            "pp_size",
            "dtype",
            "kv_cache_dtype",
            "enable_tf32_matmul",
            "context_length",
            "chunked_prefill_size",
            "max_running_requests",
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
            "linear_attn_backend",
            "linear_attn_prefill_backend",
            "linear_attn_decode_backend",
            "bf16_gemm_backend",
            "enable_deterministic_inference",
            "disable_cuda_graph",
            "json_model_override_args",
            "max_mamba_cache_size",
            "mamba_ssm_dtype",
            "mamba_radix_cache_strategy",
            "disable_radix_cache",
            "disable_chunked_prefix_cache",
            "page_size",
            "disaggregation_mode",
            "disaggregation_transfer_backend",
            "dcp_size",
            "dcp_comm_backend",
            "dcp_replicate_q_proj",
        )
        actual, expected = config, ref_info.get("server_args", ref_info)
        differences = {
            k: [actual.get(k), expected.get(k)]
            for k in keys
            if actual.get(k) != expected.get(k)
        }
        assert not differences, differences
        if args.prefill_url:
            ref_prefill = json.loads(
                (reference / "prefill-server-info.json").read_text()
            )
            ref_prefill = ref_prefill.get("server_args", ref_prefill)
            differences = {
                k: [prefill_config.get(k), ref_prefill.get(k)]
                for k in keys
                if prefill_config.get(k) != ref_prefill.get(k)
            }
            assert not differences, differences
        request_call(
            args.url,
            "/load_lora_adapter",
            {
                "lora_name": args.adapter_name,
                "lora_path": args.adapter_path,
                "pinned": config["attn_dp_size"] > 1,
            },
        )
    sampling = {"temperature": 0, "max_new_tokens": ntokens, "ignore_eos": True}
    processor = CaptureAndForce.to_str()
    phases = ("prefill", "prefill_last", "decode")
    pairs, noise, first, cases = (
        {p: [] for p in phases},
        {p: [] for p in phases},
        {},
        [],
    )
    started = time.monotonic()
    for start in range(0, len(corpus), batch_size):
        group = corpus[start : start + batch_size]
        name = f"references-{start}.json"
        if dynamic:
            refs = json.loads((reference / name).read_text())
        else:
            request_call(args.url, "/flush_cache", {})
            refs = request_call(
                args.url,
                "/generate",
                batch(
                    [
                        {"input_ids": c["input_ids"], "sampling_params": sampling}
                        for c in group
                    ]
                ),
            )
        assert len(refs) == len(group)
        write(args.output / name, refs)
        continuations = [r["output_ids"] for r in refs]
        assert all(len(c) == ntokens for c in continuations)
        for repeat in range(rounds):
            for cache in ("cold", "hot"):
                label = f"g{start}-r{repeat}-{cache}"
                if cache == "cold":
                    request_call(args.url, "/flush_cache", {})
                requests, paths = [], []
                for i, (case, cont) in enumerate(zip(group, continuations)):
                    path = captures / f"{label}-i{i}.jsonl"
                    paths.append(path)
                    row = {
                        "input_ids": case["input_ids"],
                        "return_logprob": True,
                        "logprob_start_len": 0,
                        "top_logprobs_num": 1,
                        "custom_logit_processor": processor,
                        "sampling_params": sampling
                        | {
                            "custom_params": {
                                "capture_file": str(path),
                                "continuation": cont,
                            }
                        },
                    }
                    if dynamic:
                        row["lora_path"] = args.adapter_name
                    requests.append(row)
                responses = request_call(args.url, "/generate", batch(requests))
                write(args.output / f"{label}-responses.json", responses)
                assert len(responses) == len(group)
                scores = [
                    read_scores(res, path, case, cont, replicas, bool(args.prefill_url))
                    for res, path, case, cont in zip(
                        responses, paths, group, continuations
                    )
                ]
                write(args.output / f"{label}-scores.json", scores)
                expected_scores = (
                    json.loads((reference / f"{label}-scores.json").read_text())
                    if dynamic
                    else scores
                )
                for i, (case, score, expected) in enumerate(
                    zip(group, scores, expected_scores)
                ):
                    row = {
                        "case": case["id"],
                        "round": repeat,
                        "cache": cache,
                        "cached_tokens": responses[i]["meta_info"].get("cached_tokens"),
                    }
                    for phase in phases:
                        assert len(score[phase]) == len(expected[phase])
                        compared = list(zip(score[phase], expected[phase]))
                        pairs[phase].extend(compared)
                        row[phase] = stats(compared)
                        key = (case["id"], phase)
                        if key in first:
                            noise[phase].extend(zip(first[key], score[phase]))
                        else:
                            first[key] = score[phase]
                    cases.append(row)
                    with (args.output / "cases.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                print(
                    label,
                    len(cases),
                    {p: stats(v) for p, v in pairs.items()},
                    flush=True,
                )
    summary = {
        "stage": args.stage,
        "topology": args.topology,
        "cases": len(corpus),
        "case_pairs": len(cases),
        "rounds": rounds,
        "decode_tokens": ntokens,
        "cache_modes": ["cold", "hot"],
        "batch_size": batch_size,
        "elapsed_s": time.monotonic() - started,
        "scores": {p: stats(v) for p, v in pairs.items()} if dynamic else None,
        "repeat_noise": {p: stats(v) for p, v in noise.items()},
        "note": "hot means no flush, inspect cached_tokens for actual cache reuse",
    }
    write(args.output / "summary.json", summary)
    if not dynamic:
        write(
            args.output / "seal.json",
            {
                "files": {
                    str(p.relative_to(args.output)): digest(p)
                    for p in sorted(args.output.rglob("*"))
                    if p.is_file()
                }
            },
        )
    else:
        request_call(args.url, "/unload_lora_adapter", {"lora_name": args.adapter_name})
        assert all(
            v["passes_logprob_gate"] for v in summary["scores"].values()
        ), summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "dynamic"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prefill-url")
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--adapter-name", default="frozen-A")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    if args.stage == "reference" and args.corpus is None:
        parser.error("reference requires --corpus")
    if args.stage == "dynamic" and (
        args.reference is None or args.adapter_path is None
    ):
        parser.error("dynamic requires --reference and --adapter-path")
    main(args)
