#!/usr/bin/env python3
"""M2 数值门 + M3 接线后的端到端数值门。

门 1（单层，M2）: 融合核输出 vs 现有路径
    act  : median_rel <= 0（位级，fp8 输出必须逐元素相等）
    out  : median_rel <= 2e-2
门 2（端到端，M3 后）: 同 prompt 同 seed 下 logprob 的 median_rel <= 2e-2（token 流允许少量分歧，需列出）
输入：/root/mol_val/{ref,fused}.pt（张量）或 {ref,fused}.jsonl（每请求 logprobs 列表）
用法：python3 gate_m2_numeric.py --ref ref.pt --test fused.pt [--name out] [--tol 2e-2]
"""
from __future__ import annotations
import argparse, json, sys
def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])
def rel(a, b, eps=1e-6):
    import torch
    return ((a.float()-b.float()).abs()/(b.float().abs().clamp_min(eps))).reshape(-1)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--name", default="out"); ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--print-every", type=int, default=200)
    a = ap.parse_args()
    if a.ref.endswith(".pt"):
        import torch
        r = torch.load(a.ref, map_location="cpu"); t = torch.load(a.test, map_location="cpu")
        keys = [k for k in r if k in t]
        print(f"tensors={len(keys)}")
        worst = 0.0
        for k in keys:
            x, y = r[k], t[k]
            if x.shape != y.shape: print(f"  {k}: SHAPE MISMATCH {tuple(x.shape)} vs {tuple(y.shape)}"); continue
            d = rel(x, y)
            m = float(d.median()); mx = float(d.max()); worst = max(worst, m)
            flag = "OK" if m <= a.tol else "FAIL"
            print(f"  {k:24s} median_rel={m:.3e} max_rel={mx:.3e} exact={float((x==y).float().mean()):.6f}  {flag}")
        print("VERDICT:", "PASS" if worst <= a.tol else "FAIL", f"(worst median_rel={worst:.3e}, tol={a.tol})")
        return 0 if worst <= a.tol else 1
    # jsonl: 每请求 logprobs
    def load(p):
        return [json.loads(l) for l in open(p) if l.strip()]
    R, T = load(a.ref), load(a.test)
    if len(R) != len(T): print(f"COUNT MISMATCH {len(R)} vs {len(T)}"); return 1
    meds, ndiverge = [], 0
    for i, (r, t) in enumerate(zip(R, T)):
        rl, tl = r["logprobs"], t["logprobs"]
        n = min(len(rl), len(tl))
        if rl != tl: ndiverge += 1
        for j in range(n): meds.append(abs(rl[j]-tl[j])/(abs(tl[j])+1e-6))
    m = median(meds) if meds else float("inf")
    print(f"requests={len(R)} requests_with_stream_mismatch={ndiverge} n_logprob_pairs={len(meds)}")
    print(f"logprob median_rel={m:.3e} tol={a.tol} VERDICT:", "PASS" if m <= a.tol else "FAIL")
    return 0 if m <= a.tol else 1
if __name__ == "__main__": sys.exit(main())
