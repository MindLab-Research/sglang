#!/usr/bin/env python3
"""Verify kv_shared LoRA under HiCache (--enable-hierarchical-cache).

kv_shared requests carry extra_key=None, which flows unchanged through the
device radix tree, the L2 host tree (child_key), and the L3 storage hash
seed (storage_namespace_seed) — all three layers therefore share the base
model's namespace. This test verifies the basics hold under HiCache:

  1. kv_shared adapters load and infer normally.
  2. A kv_shared request hits the shared prefix (cached_tokens > 0, possibly
     served from host).
  3. A normal (non-kv_shared) LoRA stays isolated (cached_tokens == 0).
  4. Outputs are sane (LoRA path didn't break inference).
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:31500"
PROMPT = (
    "The following is a stable context block used to exercise the shared "
    "prefix cache across requests. " * 30
    + "Now answer: what is 6 times 7? Answer with the number only."
)
PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    PASS += ok
    FAIL += not ok


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        return {"success": False, "error_message": f"HTTP {e.code}: {body}"}


def load_lora(name):
    try:
        post("/unload_lora_adapter", {"lora_name": name})
    except Exception:
        pass
    return post("/load_lora_adapter", {"lora_name": name, "lora_path": f"/tmp/lora_adapters/{name}"})


def generate(prompt, lora=None, max_tokens=64):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    if lora:
        payload["lora_path"] = lora
    return post("/generate", payload)


def main():
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        check("health (hicache mode)", r.status == 200)

    for name in ("kv_shared_a", "kv_shared_b"):
        r = load_lora(name)
        check(f"load {name}", r.get("success"), r.get("error_message", "")[:70])

    out_a = generate(PROMPT, lora="kv_shared_a")
    cached_a = out_a["meta_info"].get("cached_tokens", 0)
    check("kv_shared_a first request ok", bool(out_a.get("text", "").strip()),
          f"cached={cached_a}")

    out_b = generate(PROMPT, lora="kv_shared_b")
    cached_b = out_b["meta_info"].get("cached_tokens", 0)
    check("kv_shared_b hits shared prefix", cached_b > 0, f"cached_tokens={cached_b}")

    out_base = generate(PROMPT)
    cached_base = out_base["meta_info"].get("cached_tokens", 0)
    check("base hits shared prefix", cached_base > 0, f"cached_tokens={cached_base}")

    r = load_lora("normal_isolated")
    check("load normal_isolated", r.get("success"), r.get("error_message", "")[:70])
    out_n = generate(PROMPT, lora="normal_isolated")
    cached_n = out_n["meta_info"].get("cached_tokens", 0)
    check("normal LoRA stays isolated", cached_n == 0, f"cached_tokens={cached_n}")

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
