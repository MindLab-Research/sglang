#!/usr/bin/env python3
"""TPOT benchmark: base vs kv_shared LoRA vs normal LoRA (1101, port 31500)."""
import json
import urllib.request

BASE = "http://127.0.0.1:31500"


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bench(label, lora=None, n=3):
    # Warm-up + prefix write
    post(
        "/generate",
        {
            "text": "Warm up prefix. " * 50,
            "sampling_params": {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True},
            **({"lora_path": lora} if lora else {}),
        },
    )
    tpots = []
    for _ in range(n):
        r = post(
            "/generate",
            {
                "text": "Answer with a long explanation: what is 6 times 7 and why? " * 20,
                "sampling_params": {"max_new_tokens": 128, "temperature": 0, "ignore_eos": True},
                **({"lora_path": lora} if lora else {}),
            },
        )
        mi = r["meta_info"]
        tpots.append(mi["e2e_latency"] / mi["completion_tokens"] * 1000)
    avg = sum(tpots) / len(tpots)
    runs = ", ".join(f"{t:.2f}" for t in tpots)
    print(f"{label}: TPOT avg {avg:.2f} ms (runs: {runs})")


bench("base (no lora)")
bench("kv_shared_a", lora="kv_shared_a")
bench("normal_isolated", lora="normal_isolated")
