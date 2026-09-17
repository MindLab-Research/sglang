#!/usr/bin/env python3
"""Classify a torch-profiler trace of a decode window into step-time buckets.

    zcat <id>-TP-0.trace.json.gz | python3 tools/analyze_step_profile.py /dev/stdin 60

Only `cat == "kernel"` events are counted.  The profiler also emits
`gpu_user_annotation` spans that NEST (step[TARGET_VERIFY bs=N], EAGLE_VERIFY,
scheduler.run_batch ...); including them inflates the total ~4x and mis-buckets
everything (first version of this script reported "eagle 59.6%" — an artifact).

Buckets answer "where does a decode step go" for a GLM-5.3 MoL + DCP=8 + EAGLE
stack: MoE GEMM / attention / communication (AR vs AG vs RS) / elementwise /
LoRA(MoL) / EAGLE / norm / other.
"""

import json
import re
import sys
from collections import defaultdict

RULES = [
    ("comm_allreduce", r"all_?reduce|AllReduce"),
    ("comm_allgather", r"all_?gather|AllGather"),
    ("comm_reducescatter", r"reduce_?scatter|ReduceScatter"),
    ("comm_p2p", r"\bsend\b|\brecv\b|_p2p|p2p_|broadcast"),
    ("moe_gemm", r"fused_moe|moe_|grouped_gemm|m_grouped|deep_gemm|fp8_gemm|scaled_mm|gemm_"),
    ("attn", r"triton_.*attn|flash|mla|attention|dsa_|mha|paged|trtllm|fmha"),
    ("lora", r"csgmv|bgmv|sgmv|lora"),
    ("eagle", r"draft|eagle|topk|verify|tree"),
    ("norm", r"rms|layer_?norm|mhc|mean|std"),
    (
        "elementwise",
        r"elementwise|vectorized|\badd\b|\bmul\b|copy|cat|index|gather|scatter|where|"
        r"arange|silu|gelu|activ|cast|convert|fill|zero|clamp|slice|reshape|contiguous|"
        r"reduce|cub::|Sort|Scan",
    ),
]


def classify(name: str) -> str:
    for bucket, pat in RULES:
        if re.search(pat, name, re.I):
            return bucket
    return "other"


def main() -> None:
    path = sys.argv[1]
    declared_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(path) as f:
        doc = json.load(f)
    events = doc.get("traceEvents") or doc.get("events") or []

    per_bucket: dict = defaultdict(float)
    per_kernel: dict = defaultdict(float)
    n_kernels = 0
    span_min = span_max = None

    for ev in events:
        if ev.get("ph") != "X" or (ev.get("cat") or "").lower() != "kernel":
            continue
        dur = float(ev.get("dur") or 0.0)
        name = ev.get("name") or ""
        n_kernels += 1
        per_bucket[classify(name)] += dur
        per_kernel[name] += dur
        ts = float(ev.get("ts") or 0.0)
        span_min = ts if span_min is None else min(span_min, ts)
        span_max = ts + dur if span_max is None else max(span_max, ts + dur)

    total = sum(per_bucket.values())
    if total <= 0:
        print("no kernel events found; top-level keys:", list(doc)[:8])
        return
    span_ms = (span_max - span_min) / 1000.0 if span_min is not None else 0.0

    print(f"trace={path}")
    print(
        f"kernel events={n_kernels}, GPU busy total={total / 1000.0:.1f} ms, "
        f"wall span={span_ms:.1f} ms"
    )
    if declared_steps:
        print(
            f"num_steps(declared)={declared_steps} -> "
            f"GPU busy/step={total / 1000.0 / declared_steps:.2f} ms, "
            f"wall/step={span_ms / declared_steps:.2f} ms"
        )
    print()
    print("%-20s %12s %8s" % ("bucket", "ms", "%"))
    print("-" * 44)
    for b in sorted(per_bucket, key=lambda x: -per_bucket[x]):
        print("%-20s %12.1f %7.1f%%" % (b, per_bucket[b] / 1000.0, 100.0 * per_bucket[b] / total))

    print()
    print("=== top 25 kernels by GPU time ===")
    for name, dur in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:25]:
        print("  %9.1f ms  %6.1f%%  %s" % (dur / 1000.0, 100.0 * dur / total, name[:110]))


if __name__ == "__main__":
    main()
