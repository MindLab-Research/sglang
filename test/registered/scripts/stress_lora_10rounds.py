#!/usr/bin/env python3
"""10-round LoRA stress cycle on 1101 (matches the release-verification bar).

3 adapters x (load + 3 inferences + unload) x 10 rounds.
Run on 1101 after the server is ready.
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:31500"
ADAPTERS = ("test_lora_1", "test_lora_2", "test_lora_3")
ROUNDS = 10
PROMPTS = (
    "1+1=? Answer with the number only.",
    "Write a Python one-liner to reverse a list.",
    "用一句话解释什么是快速排序。",
)


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt, lora, max_tokens=48):
    return post(
        "/v1/chat/completions",
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "lora_path": lora,
        },
    )


def main():
    fails = []
    for rnd in range(1, ROUNDS + 1):
        for name in ADAPTERS:
            r = post("/load_lora_adapter", {"lora_name": name, "lora_path": f"/tmp/lora_adapters/{name}"})
            if not r.get("success"):
                fails.append((rnd, name, "load", r.get("error_message", "")[:80]))
                continue
            for p in PROMPTS:
                try:
                    out = chat(p, name)
                    text = out["choices"][0]["message"]["content"]
                    if not text.strip():
                        fails.append((rnd, name, "infer", "empty output"))
                except Exception as e:
                    fails.append((rnd, name, "infer", str(e)[:80]))
            r = post("/unload_lora_adapter", {"lora_name": name})
            if not r.get("success"):
                fails.append((rnd, name, "unload", r.get("error_message", "")[:80]))
        print(f"round {rnd}/{ROUNDS} done", flush=True)

    if fails:
        print(f"\nFAILED ({len(fails)} issues):")
        for f in fails[:10]:
            print("  ", f)
        sys.exit(1)
    print(f"\nALL {ROUNDS} rounds x {len(ADAPTERS)} adapters x load+3infer+unload PASS")


if __name__ == "__main__":
    main()
