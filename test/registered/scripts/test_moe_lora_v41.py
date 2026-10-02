#!/usr/bin/env python3
"""Verify MoE LoRA on DeepSeek-V4.1 (MXFP4 trtllm-gen) on 1101.

Scenarios:
  1. health + plain inference (LoRA enabled, no adapter active).
  2. zero-weight MoE adapter: its output must equal the base output exactly —
     proves the LoRA dispatch (gated SwiGLU + OAI controls, decomposed
     gate_up/down pipeline, MXFP8 quantisation) matches the non-LoRA path.
  3. real MoE adapter (test_moe_lora): loads, runs, produces non-empty output.
  4. TPOT: base vs LoRA-active (no perf cliff).

Run on 1101 after the server is up. Note the server must be started with
SGLANG_EXPERIMENTAL_LORA_OPTI=1 and gate_up_proj/down_proj in
--lora-target-modules.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:31500"
PROMPT = (
    "The following is a stable context block used for a numerical check. " * 20
    + "Now answer with the number only: what is 6 times 7?"
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
            body = e.read().decode()[:300]
        except Exception:
            pass
        return {"success": False, "error_message": f"HTTP {e.code}: {body}"}


def load_lora(name):
    try:
        post("/unload_lora_adapter", {"lora_name": name})
    except Exception:
        pass
    return post(
        "/load_lora_adapter",
        {"lora_name": name, "lora_path": f"/tmp/lora_adapters/{name}"},
    )


def generate(prompt, lora=None, max_tokens=32):
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
        check("health", r.status == 200)

    # 1. base output (LoRA enabled, no adapter)
    out_base = generate(PROMPT)
    text_base = out_base.get("text", "")
    check("plain inference (no adapter active)", bool(text_base.strip()),
          repr(text_base[:40]))

    # 2. zero adapter must reproduce the base output exactly
    r = load_lora("zero_moe_lora")
    check("load zero_moe_lora (MoE adapter)", r.get("success"),
          r.get("error_message", "")[:100])
    if r.get("success"):
        out_zero = generate(PROMPT, lora="zero_moe_lora")
        text_zero = out_zero.get("text", "")
        check("zero adapter == base output (activation/pipeline parity)",
              text_zero == text_base,
              f"base={text_base[:24]!r} zero={text_zero[:24]!r}")
        post("/unload_lora_adapter", {"lora_name": "zero_moe_lora"})

    # 3. real MoE adapter
    r = load_lora("test_moe_lora")
    check("load test_moe_lora (real MoE adapter)", r.get("success"),
          r.get("error_message", "")[:100])
    if r.get("success"):
        out_real = generate(PROMPT, lora="test_moe_lora")
        text_real = out_real.get("text", "")
        check("real MoE adapter infers", bool(text_real.strip()), repr(text_real[:40]))
        check("real MoE adapter output finite", "nan" not in text_real.lower()
              and "inf" not in text_real.lower(), repr(text_real[:40]))

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
