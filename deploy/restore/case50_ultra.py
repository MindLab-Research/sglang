#!/usr/bin/env python3
"""case50_ultra.py — 超高并发 case50 压测（固定并发，验证 decode 稳定性）

从 cases_50.json 读 50 个真实请求体，按 rounds 轮 × 50 cases 构造任务，
用 Semaphore 限制同时 in-flight 数（超高并发 = --conc 128/160）。

指标: ok/fail、耗时 p50/p95、乱码/短响应检测、崩溃检测（失败模式）。
用法:
  python3 case50_ultra.py --conc 128 --rounds 3 --endpoint http://127.0.0.1:30000/v1/chat/completions --key sk-glm52-pd
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx


async def send_one(client, endpoint, payload, idx):
    t0 = time.time()
    try:
        # 与 stress80_mix.py 一致：完整 URL post（不设 base_url，避免拼接重复路径）
        r = await client.post(endpoint, json=payload, timeout=900)
        dt = time.time() - t0
        ok = r.status_code == 200
        text = r.text
        # 短响应/疑似乱码启发式（供参考，不判失败）
        suspicious = ok and len(text) < 80
        return {
            "idx": idx,
            "ok": ok,
            "code": r.status_code,
            "t": round(dt, 1),
            "len": len(text),
            "suspicious": suspicious,
        }
    except Exception as e:
        return {
            "idx": idx,
            "ok": False,
            "code": 0,
            "err": str(e)[:120],
            "t": round(time.time() - t0, 1),
            "len": 0,
            "suspicious": False,
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=128, help="同时 in-flight 请求数")
    ap.add_argument("--rounds", type=int, default=3, help="50 cases 循环轮数（总请求=50×rounds）")
    ap.add_argument("--endpoint", default="http://127.0.0.1:30000/v1/chat/completions")
    ap.add_argument("--key", default="sk-glm52-pd")
    ap.add_argument("--cases", default="/root/bench_case50/cases_50.json")
    ap.add_argument("--model", default="glm52-fp8-official", help="served model 名（cases_50 的旧名会被覆盖）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)
    # cases_50.json 的 payload 用旧模型名 (Macaron-V1-Venti)，当前 served 模型
    # 是 glm52-fp8-official — 强制覆盖，否则 router 404 拒绝（请求到不了 decode）。
    payloads = [
        dict(c["payload"], model=args.model) for c in cases[:50]
    ]
    random.seed(args.seed)

    tasks = []
    for rnd in range(args.rounds):
        for i, p in enumerate(payloads):
            tasks.append((rnd * 1000 + i, p))
    random.shuffle(tasks)

    sem = asyncio.Semaphore(args.conc)
    t0 = time.time()
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {args.key}"},
        timeout=900,
        limits=httpx.Limits(max_connections=args.conc * 2, max_keepalive_connections=args.conc),
    ) as client:

        async def run(t):
            idx, p = t
            async with sem:
                return await send_one(client, args.endpoint, p, idx)

        results = await asyncio.gather(*[run(t) for t in tasks])
        elapsed = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    times = sorted(r["t"] for r in ok)
    p50 = times[len(times) // 2] if times else -1
    p95 = times[int(len(times) * 0.95)] if times else -1
    susp = [r for r in ok if r["suspicious"]]
    codes = {}
    for r in results:
        codes[r["code"]] = codes.get(r["code"], 0) + 1
    print(
        f"RESULT total={len(results)} ok={len(ok)} fail={len(fail)} "
        f"elapsed={elapsed:.1f}s conc={args.conc} rounds={args.rounds}"
    )
    print(f"  p50={p50}s p95={p95}s suspicious_short={len(susp)} codes={codes}")
    if fail:
        print(f"  first_fails={fail[:5]}")

    # 崩溃检测：失败比例高且 HTTP 层错误 = 可能 decode 挂了
    if len(fail) > len(results) * 0.5:
        print("  !! HIGH FAILURE RATE — decode may have crashed")


asyncio.run(main())
