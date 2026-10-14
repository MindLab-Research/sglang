#!/usr/bin/env python3
"""S1 判据守卫：P2 GEMV 是否值得做 —— 必须同形状、同分母，否则拒绝下结论。

rt-runtime 补正（2026-10-13）：
  现状 rnd 每层 MoE = M_base_overlay + 93µs（实测增量）
  P-A 后每层        = M_gemv（K1+K2 把 base GEMM 一起换掉，是整核替换）
  正确判据:  M_gemv ≤ M_base_overlay + 29µs   ⇔  M_gemv ≤ 现状rnd每层 − 64µs
  ⇒ microbench **必须同时报 overlay 链基线**（gemm_base_fused+quant+activation+finalize），
    只报 K1+K2+argsort+finalize 对「93/29」= 拿替换后整核比替换前增量 ⇒ 拒绝判决。

输入 JSON（microbench 产出契约）:
{
  "shape": {"A": 384, "experts_per_rank": 64, "rows_per_expert_min": 1, "rows_per_expert_max": 8},
  "overlay_chain_per_layer_us": {"gemm_base_fused": x, "quant": x, "activation": x, "finalize": x},
  "gemv_chain_per_layer_us":    {"k1": x, "k2": x, "argsort": x, "finalize": x},
  "current_rnd_per_layer_us": 93.0        # 可选，用于交叉核对分母
}
用法: python3 check_p2_budget.py --microbench-json s1.json
"""
from __future__ import annotations
import argparse, json, sys

TARGET_HEADROOM_US = 29.0     # 6.000ms 目标允许的每层 LoRA 开销
CURRENT_DELTA_US = 93.0       # 现状 rnd − base（per layer，实测 1.867ms/20）
REQ_SHAPE = {"A": 384, "experts_per_rank": 64, "rows_per_expert_min": 1, "rows_per_expert_max": 8}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--microbench-json", required=True)
    ap.add_argument("--tol-us", type=float, default=10.0, help="分母交叉核对容差")
    a = ap.parse_args(); M = json.loads(open(a.microbench_json).read())

    sh = M.get("shape") or {}
    bad = {k: (sh.get(k), v) for k, v in REQ_SHAPE.items() if sh.get(k) != v}
    if bad:
        print(f"REFUSED: 形状不符 R7 铁律（要求 A=384 / 64 experts/rank / 1–8 行每专家）。实际={sh}")
        print("         微基准的分布若与真实 decode 不一致，会重犯 P1 deepgemm 的错误（结论反向）。")
        return 2
    ov, gv = M.get("overlay_chain_per_layer_us"), M.get("gemv_chain_per_layer_us")
    if not ov:
        print("REFUSED: 缺 overlay 链基线 ⇒ 拿替换后整核比替换前增量，苹果比橘子。"
              "必须同形状同测 gemm_base_fused+quant+activation+finalize。")
        return 3
    if not gv:
        print("REFUSED: 缺 gemv 链测量（k1/k2/argsort/finalize）"); return 3
    M_ov, M_gv = sum(ov.values()), sum(gv.values())
    cur = M.get("current_rnd_per_layer_us", CURRENT_DELTA_US + M_ov)
    print(f"同形状 per-layer: overlay_chain={M_ov:.1f}µs  gemv_chain={M_gv:.1f}µs")
    for k, v in sorted(ov.items(), key=lambda kv: -kv[1]): print(f"    overlay  {k:<16}{v:8.1f} µs")
    for k, v in sorted(gv.items(), key=lambda kv: -kv[1]): print(f"    gemv     {k:<16}{v:8.1f} µs")
    implied = cur - M_ov
    if abs(implied - CURRENT_DELTA_US) > a.tol_us:
        print(f"WARN: 分母交叉核对失败 —— 现状每层={cur:.1f}µs − overlay={M_ov:.1f}µs "
              f"⇒ 隐含增量 {implied:.1f}µs，与实测 93µs 差 >{a.tol_us}µs。"
              "两套测量的形状/口径可能不一致，先对齐再判。")
    budget = M_ov + TARGET_HEADROOM_US
    head = budget - M_gv
    print(f"判据: M_gemv({M_gv:.1f}) ≤ M_overlay({M_ov:.1f}) + {TARGET_HEADROOM_US:.0f} = {budget:.1f} µs"
          f"  （等价于 ≤ 现状每层 − 64µs）")
    if head >= 0:
        print(f"VERDICT: P2 可行（余量 {head:.1f}µs/层 ≈ 全模型 {head*20/1000:.3f}ms）⇒ M3 接线优先。")
    else:
        print(f"VERDICT: P2 判死（超预算 {-head:.1f}µs/层 = {M_gv/budget:.1f}× 预算）"
              "⇒ 转 P-B/P-C + 第四刀；M3 仅留作数值基线/对照。")
    return 0

if __name__ == "__main__": sys.exit(main())
