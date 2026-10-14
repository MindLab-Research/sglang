#!/usr/bin/env python3
"""nsys 导出物 → 按 kernel 家族分桶的 µs 表（20 层摊销 + 发射间隔/gap）。

输入:
  *.sqlite          nsys export --type sqlite   (首选, 有 start/end/stream)
  或 gpu_trace_*.csv (nsys stats --report cuda_gpu_trace --format csv, 兜底)
用法:
  python3 nsys_extract.py --rep /root/prof/kern_rnd.sqlite --layers 20 [--csv ...] [--json out.json]
输出: 每桶 total_us / per_layer_us / per_call_us / calls / %；以及按步边界的 kernel-busy 与 gap。
"""
from __future__ import annotations
import argparse, csv, json, re, sqlite3, statistics, sys
from collections import defaultdict

# 家族正则（可扩；命名对齐 docs/DSV41_MOL_FUSED_KERNEL_DESIGN.md §0 的做差桶）
BUCKETS = [
    ("gemm_base_fused",  r"t128x16x256_s6|Gemm1.*fused_act|nvfp4.*block_scale.*fused"),
    ("gemm_lora_unfused", r"t128x8x512_s3|Gemm2::Runner|unfused"),
    ("delta_shrink",     r"_chunked_lora_shrink|merged_experts_fused_moe_lora_add|_fused_lora_delta_kernel|lora.*shrink"),
    ("delta_expand",     r"_chunked_lora_expand|lora.*expand"),
    ("gemv_k1_gateup",   r"gemv.*gate|gate_up.*gemv|k1_"),
    ("gemv_k2_down",     r"gemv.*down|down.*gemv|k2_"),
    ("quant",            r"quant|cvt_warp_fp16_to_mxfp8|fp8.*cast|scaled_mm_quant|mxfp8"),
    ("align_permute",    r"permute|align|pack_topk|pad|scatter|expand_index"),
    ("activation",       r"activationDeepSeek|silu|gatedSilu|swiglu"),
    ("finalize",         r"finalize"),
    ("attn_lora_sgmv",   r"sgmv|csgmv|lora.*attn"),
    ("attn",             r"attention|fmha|mla|flashinfer.*attn|decode_attention"),
    ("norm_rope",        r"rmsnorm|norm|rope"),
    ("other",            r".*"),
]
# 整链聚合：单核胜负不作数，只有「链 Σ（含发射税）」才是判官（rt-runtime 2026-10-13 共识）
CHAINS = {
    "moE_overlay_chain": ["gemm_base_fused", "quant", "activation", "align_permute", "finalize"],
    "moE_lora_chain":    ["gemm_lora_unfused", "gemv_k1_gateup", "gemv_k2_down", "delta_shrink",
                          "delta_expand", "quant", "activation", "align_permute", "finalize"],
    "attn_lora_chain":   ["attn_lora_sgmv", "attn"],
}

def classify(name: str) -> str:
    for b, pat in BUCKETS:
        if re.search(pat, name, re.I): return b
    return "other"

def load_sqlite(path):
    con = sqlite3.connect(path); cur = con.cursor()
    tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    ktab = next((t for t in tabs if "CUPTI_ACTIVITY_KIND_KERNEL" in t.upper()), None)
    if not ktab: raise SystemExit(f"no kernel table in {path}; tables={tabs[:20]}")
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({ktab})")]
    sc = "start" if "start" in cols else [c for c in cols if c.lower().startswith("start")][0]
    ec = "end" if "end" in cols else [c for c in cols if c.lower().startswith("end")][0]
    nc = "nameId" if "nameId" in cols else None
    st = "streamId" if "streamId" in cols else None
    q = f"SELECT {sc},{ec},{st or 'NULL'},{(f'(SELECT value FROM StringIds WHERE id={nc})' if nc else 'NULL')} FROM {ktab}"
    return [(float(s), float(e), stream, name or "?") for s, e, stream, name in cur.execute(q)]

def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            g = lambda *ks: next((r[k] for k in ks if k in r and r[k]), None)
            s, e, n = g("Start (ns)", "Start"), g("End (ns)", "End"), g("Name", "Kernel Name")
            if s is None: continue
            rows.append((float(s), float(e), g("Stream"), n or "?"))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep"); ap.add_argument("--csv"); ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--gap-threshold-us", type=float, default=2.0, help=">该值的间隙才算 dispatch gap")
    ap.add_argument("--json"); ap.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "RND_JSON"),
        help="两个 --json 输出的做差表（base vs lora），自动给 Δ/per-layer")
    a = ap.parse_args()
    if a.compare:
        import json as _j
        A, B = (_j.loads(open(x).read()) for x in a.compare)
        keys = sorted(set(A) | set(B) - {"_meta", "_chains"})
        print(f"{'bucket':<18}{'base_us':>10}{'lora_us':>10}{'delta_us':>10}{'delta/layer':>13}{'x':>7}")
        for k in keys:
            if k.startswith("_"): continue
            bu = A.get(k, {}).get("total_us", 0.0); lu = B.get(k, {}).get("total_us", 0.0)
            print(f"{k:<18}{bu:>10.1f}{lu:>10.1f}{lu-bu:>10.1f}{(lu-bu)/(a.layers):>13.1f}{(lu/bu if bu else float('nan')):>7.2f}")
        return
    if a.rep: rows = load_sqlite(a.rep)
    elif a.csv: rows = load_csv(a.csv)
    else: sys.exit("need --rep or --csv")
    rows.sort(key=lambda x: x[0])
    unit = 1e-3 if rows and rows[-1][0] < 1e9 else 1.0   # ns->us
    buckets = defaultdict(lambda: {"us": 0.0, "calls": 0})
    for s, e, stream, name in rows:
        b = classify(name); d = (e - s) * unit
        buckets[b]["us"] += d; buckets[b]["calls"] += 1
    # 按流做 gap 统计（同一流相邻 kernel 之间的空档 = 发射间隙 / 未覆盖的 PDL 缺口）
    by_stream = defaultdict(list)
    for s, e, stream, name in rows: by_stream[stream].append((s, e))
    gaps = []
    for stream, ev in by_stream.items():
        ev.sort()
        for i in range(1, len(ev)):
            g = (ev[i][0] - ev[i - 1][1]) * unit
            if g > a.gap_threshold_us: gaps.append(g)
    total = sum(v["us"] for v in buckets.values())
    print(f"total kernel busy = {total/1000:.1f} ms over {len(rows)} launches "
          f"({len(by_stream)} streams, gaps>{a.gap_threshold_us}us: n={len(gaps)}, "
          f"sum={sum(gaps)/1000:.1f} ms, p50={statistics.median(gaps) if gaps else 0:.2f} us)")
    print(f"{'bucket':<18}{'total_us':>10}{'per_layer_us':>14}{'per_call_us':>12}{'calls':>8}{'share':>8}")
    out = {}
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]["us"]):
        pl = v["us"] / a.layers; pc = v["us"] / max(v["calls"], 1)
        print(f"{b:<18}{v['us']:>10.1f}{pl:>14.1f}{pc:>12.2f}{v['calls']:>8d}{100*v['us']/total:>7.1f}%")
        out[b] = {"total_us": v["us"], "per_layer_us": pl, "per_call_us": pc, "calls": v["calls"]}
    print("\n-- 整链 Σ（含发射/间隙税，per-layer）--")
    for cname, members in CHAINS.items():
        us = sum(buckets[m]["us"] for m in members if m in buckets)
        calls = sum(buckets[m]["calls"] for m in members if m in buckets)
        print(f"{cname:<20} total={us/1000:8.3f} ms  per_layer={us/a.layers:7.1f} us  calls={calls}")
        out.setdefault("_chains", {})[cname] = {"total_us": us, "per_layer_us": us / a.layers, "calls": calls}
    gap_pl = (sum(gaps) / a.layers) if gaps else 0.0
    print(f"{'dispatch gap':<20} total={sum(gaps)/1000:8.3f} ms  per_layer={gap_pl:7.1f} us  n={len(gaps)}")
    out["_meta"] = {"total_us": total, "launches": len(rows), "gap_n": len(gaps), "gap_sum_us": sum(gaps),
                    "gap_p50_us": statistics.median(gaps) if gaps else None, "gap_per_layer_us": gap_pl}
    if a.json: open(a.json, "w").write(json.dumps(out, indent=2)); print(f"wrote {a.json}")
if __name__ == "__main__": main()
