#!/usr/bin/env python3
"""Verify kv_shared LoRA behavior on 1101 (port 31500).

Scenarios:
  1. fail-safe: adapter declaring kv_shared + v_proj target must be REJECTED.
  2. shared namespace: kv_shared_a prompt, then kv_shared_b (different weights)
     with the same prefix → second request hits the cached prefix
     (cached_tokens > 0) AND the outputs differ (LoRA actually applies).
  3. isolation regression: a normal (non-kv_shared) LoRA with the same prefix
     must NOT hit the shared prefix (cached_tokens == 0 on first call).
Run on 1101 after the server is up.
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:31500"
PROMPT = (
    "You are a helpful assistant. Here is a long context paragraph for "
    "prefix-cache testing. " * 30
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_lora(name):
    return post("/load_lora_adapter", {"lora_name": name, "lora_path": f"/tmp/lora_adapters/{name}"})


def generate(prompt, lora=None, max_tokens=64):
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
    }
    if lora:
        payload["lora_path"] = lora
    r = post("/generate", payload)
    return r


def main():
    # 1. fail-safe: bad_kv_vproj must be rejected
    r = load_lora("bad_kv_vproj")
    check("fail-safe rejects kv_shared+v_proj", not r.get("success"),
          r.get("error_message", "")[:90])

    # 2. shared namespace across kv_shared adapters
    r = load_lora("kv_shared_a")
    check("load kv_shared_a", r.get("success"), r.get("error_message", "")[:60])
    r = load_lora("kv_shared_b")
    check("load kv_shared_b", r.get("success"), r.get("error_message", "")[:60])

    out_a = generate(PROMPT, lora="kv_shared_a")
    cached_a = out_a["meta_info"].get("cached_tokens", 0)
    text_a = out_a["text"]

    out_b = generate(PROMPT, lora="kv_shared_b")
    cached_b = out_b["meta_info"].get("cached_tokens", 0)
    text_b = out_b["text"]

    check("kv_shared_b hits shared prefix", cached_b > 0, f"cached_tokens={cached_b}")
    check("kv_shared outputs differ (LoRA applies)", text_a != text_b,
          f"a={text_a[:20]!r} b={text_b[:20]!r}")

    # base request must also hit the shared prefix (same namespace)
    out_base = generate(PROMPT)
    cached_base = out_base["meta_info"].get("cached_tokens", 0)
    check("base hits shared prefix", cached_base > 0, f"cached_tokens={cached_base}")

    # 3. isolation regression: normal LoRA must NOT hit the shared prefix
    r = load_lora("normal_isolated")
    check("load normal_isolated", r.get("success"), r.get("error_message", "")[:60])
    out_n = generate(PROMPT, lora="normal_isolated")
    cached_n = out_n["meta_info"].get("cached_tokens", 0)
    check("normal LoRA stays isolated (no hit)", cached_n == 0,
          f"cached_tokens={cached_n}")

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
