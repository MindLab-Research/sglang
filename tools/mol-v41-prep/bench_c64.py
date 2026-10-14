#!/usr/bin/env python3
"""conc=64 / n=256 / 长 prompt 的 client 侧 TPOT p50 采集器（骨架，未在 1101 跑过）。

口径（与 docs/DSV41_LORA_SUB10_PLAN.md §5 一致）:
  - cases_50.json 真实请求体，长 prompt(10k+ tok)，out tokens=4096，ignore_eos，seed=777
  - 并发 64 常驻，共 n=256 请求；每请求 TPOT = (last_token_ts - ttft) / (n_content_chunks - 1)
  - 丢弃：decode_window < 0.5s 或 content chunk < 4 的样本；报告 p50/p90 + n + 失败数
  - 每轮 bench 前用 ROUND_BEGIN 输出时间戳，便于和服务端 pd_decode.log 窗口对齐

⚠️ 与已验证工具 /home/smith/src/mint-bench/bench/case50/bench_ttft_tpot.py 互校：
   同一轮双跑，若 p50 相对偏差 >2%，以 bench_ttft_tpot.py 为准（本脚本弃用）。
"""
from __future__ import annotations
import argparse, asyncio, json, random, statistics, sys, time
from pathlib import Path
import httpx

DEFAULT_CASES = "/home/smith/src/mint-bench/bench/case50/cases_50.json"

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--endpoint", default="http://127.0.0.1:31000/v1/chat/completions")
    p.add_argument("--api-key", default="")
    p.add_argument("--adapter", default=None, help="moe_rnd / moe_zero / none(base)")
    p.add_argument("--model", default=None)
    p.add_argument("--conc", type=int, default=64)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--out-tokens", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--tag", default="unnamed")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--max-tokens-cap", type=int, default=4096,
                   help="覆盖 payload 里的 max_tokens，保证长 decode 窗口")
    p.add_argument("--min-decode-window", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", default=None)
    p.add_argument("--dump-tokens", default=None,
                   help="jsonl: 每请求 token 流 + logprob md5 + nan/inf（供 gate_zero_bitexact.sh compare）")
    return p.parse_args()

async def one(client, url, headers, payload, sem, res, args):
    async with sem:
        t0 = time.perf_counter(); ttft = None; last = None; nch = 0; ntok = 0
        toks_list, lps = [], []
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code != 200:
                    res["fail"] += 1; res["errors"].append(f"HTTP {r.status_code}")
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"): continue          # 跳过 : keepalive / 空行
                    body = line[5:].strip()
                    if body == "[DONE]": break
                    try: chunk = json.loads(body)
                    except Exception: continue
                    for ch in (chunk.get("choices") or []):
                        d = ch.get("delta") or {}
                        txt = d.get("content") or d.get("reasoning_content") or ""
                        if not txt: continue
                        now = time.perf_counter()
                        if ttft is None: ttft = now
                        last = now; nch += 1
                        toks_list.append(txt)
                        for lp in (ch.get("logprobs") or {}).get("content") or []:
                            if isinstance(lp, dict) and lp.get("logprob") is not None: lps.append(lp["logprob"])
                    u = chunk.get("usage") or {}
                    if u.get("completion_tokens"): ntok = int(u["completion_tokens"])
        except Exception as e:
            res["fail"] += 1; res["errors"].append(f"{type(e).__name__}: {e}")
            return
        if ttft is None or last is None: res["fail"] += 1; return
        window = last - ttft
        toks = max(ntok - 1, nch - 1)
        if window < args.min_decode_window or toks < 4 or toks <= 0:
            res["skip"] += 1; return
        import hashlib
        res["dump"].append({"id": getattr(payload, "_id", None) or f"req{len(res['dump']):04d}",
                            "tokens": toks_list, "logprob_md5": hashlib.md5(repr(lps).encode()).hexdigest() if lps else None,
                            "nan_inf_count": sum(1 for v in lps if v != v or abs(v) == float("inf")),
                            "n_tokens": nch})
        res["tpot_ms"].append(window / toks * 1e3)
        res["ttft_ms"].append((ttft - t0) * 1e3)
        res["e2e_s"].append((last - t0))

async def run_round(args, cases, rnd):
    payloads = [(dict(c) if isinstance(c, dict) else c) for c in cases]
    random.Random(args.seed + rnd).shuffle(payloads)
    payloads = payloads[: args.n]
    for pl in payloads:
        pl["max_tokens"] = min(int(pl.get("max_tokens") or args.max_tokens_cap), args.max_tokens_cap)
        pl["ignore_eos"] = True; pl["temperature"] = 0.0; pl["stream"] = True
        if args.seed is not None: pl["seed"] = args.seed
        if args.model: pl["model"] = args.model
        if args.adapter and args.adapter not in ("none", "base"):
            pl["model"] = args.adapter   # smg 以 model 字段选 adapter；按盒子上实际约定改这里
    headers = {"Content-Type": "application/json"}
    if args.api_key: headers["Authorization"] = f"Bearer {args.api_key}"
    res = {"tpot_ms": [], "ttft_ms": [], "e2e_s": [], "fail": 0, "skip": 0, "errors": [], "dump": []}
    sem = asyncio.Semaphore(args.conc)
    limits = httpx.Limits(max_connections=args.conc + 8, max_keepalive_connections=args.conc)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        await asyncio.gather(*(one(client, args.endpoint, headers, p, sem, res, args) for p in payloads))
    return res

async def main():
    a = parse_args()
    cases = json.loads(Path(a.cases).read_text())
    if isinstance(cases, dict): cases = cases.get("cases") or cases.get("requests") or []
    report = {"tag": a.tag, "adapter": a.adapter, "conc": a.conc, "n": a.n, "rounds": []}
    for r in range(a.rounds):
        print(f"ROUND_BEGIN tag={a.tag} round={r} ts={time.time():.3f} adapter={a.adapter}", flush=True)
        res = await run_round(a, cases, r)
        s = {"round": r, "ok": len(res["tpot_ms"]), "fail": res["fail"], "skip": res["skip"],
             "tpot_p50_ms": pct(res["tpot_ms"], 50), "tpot_p90_ms": pct(res["tpot_ms"], 90),
             "tpot_mean_ms": statistics.fmean(res["tpot_ms"]) if res["tpot_ms"] else None,
             "ttft_p50_ms": pct(res["ttft_ms"], 50), "errors": res["errors"][:5]}
        report["rounds"].append(s)
        report.setdefault("_dump", []).extend(res["dump"])
        print("ROUND_END " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "errors"}), flush=True)
    all_tpot = [s["tpot_p50_ms"] for s in report["rounds"] if s["tpot_p50_ms"]]
    report["client_agg_p50_ms"] = statistics.median(all_tpot) if all_tpot else None
    report["verdict"] = None
    if report["client_agg_p50_ms"]:
        report["verdict"] = "PASS" if report["client_agg_p50_ms"] <= 6.000 and (a.adapter and a.adapter != "none") else "MEASURED"
    if a.dump_tokens and report.get("_dump"):
        with open(a.dump_tokens, "w") as f:
            for r in report["_dump"]: f.write(json.dumps(r) + "\n")
        print(f"wrote {a.dump_tokens} ({len(report['_dump'])} reqs)")
    if a.out: Path(a.out).write_text(json.dumps(json.loads(json.dumps(report, default=str)), indent=2))
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
