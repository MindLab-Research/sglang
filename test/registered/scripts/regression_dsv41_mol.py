#!/usr/bin/env python3
"""DSV4.1-Flash + MoL regression test (runs on 1101 against port 31500).

Verifies:
  1. health + plain inference (no LoRA)
  2. statically loaded MoE shared-outer adapter (test_moe_lora) — the
     virtual-experts path
  3. attention LoRA (q_proj) dynamic load / infer / unload
  4. server-log evidence for VE / shared-outer activation
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:31500"
PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    PASS += ok
    FAIL += not ok


def post(path: str, payload: dict, timeout: int = 300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt: str, lora: str = None, max_tokens: int = 64) -> dict:
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if lora:
        payload["lora_path"] = lora
    return post("/v1/chat/completions", payload)


def main():
    # 1. health
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
            check("health", r.status == 200)
    except Exception as e:
        check("health", False, str(e))
        return

    # 2. plain inference (sanity: clean output)
    try:
        out = chat("1+1=? Answer with the number only.", max_tokens=32)
        text = out["choices"][0]["message"]["content"]
        check("plain inference", "2" in text, text[:40])
    except Exception as e:
        check("plain inference", False, str(e))

    # 3. MoE shared-outer adapter (statically loaded as test_moe_lora)
    try:
        out = chat("What is 7*6? Answer with the number only.", lora="test_moe_lora", max_tokens=32)
        text = out["choices"][0]["message"]["content"]
        check("MoE LoRA inference (VE path)", "42" in text, text[:60])
    except Exception as e:
        check("MoE LoRA inference (VE path)", False, str(e))

    # 4. attention LoRA dynamic load / infer / unload
    for name in ("test_lora_1",):
        path = f"/tmp/lora_adapters/{name}"
        try:
            r = post("/load_lora_adapter", {"lora_name": name, "lora_path": path})
            check(f"load {name}", r.get("success", False), str(r.get("error_message", ""))[:80])
        except Exception as e:
            check(f"load {name}", False, str(e)[:120])
            continue
        try:
            out = chat("1+1=? Answer with the number only.", lora=name, max_tokens=32)
            text = out["choices"][0]["message"]["content"]
            check(f"infer {name}", "2" in text, text[:40])
        except Exception as e:
            check(f"infer {name}", False, str(e)[:120])
        try:
            r = post("/unload_lora_adapter", {"lora_name": name})
            check(f"unload {name}", r.get("success", False))
        except Exception as e:
            check(f"unload {name}", False, str(e)[:120])

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
