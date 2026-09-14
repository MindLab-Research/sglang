#!/usr/bin/env python3
"""分析引擎日志里的 LoRA 装载分段耗时：每次 load 的总时长 + 每个 rank 的 下载→完成 耗时。"""
import re, sys, collections, datetime

path = sys.argv[1] if len(sys.argv) > 1 else "/root/prefill_glm53mol.log"
ts = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
lid = re.compile(r"lora_id=([0-9a-f]{32})")

loads = collections.OrderedDict()   # lora_id -> dict
starts = []                          # (ts, url_tail)  "Start load Lora adapter"
unloads = []                         # (ts, url_tail)

for ln in open(path, errors="ignore"):
    m = ts.match(ln)
    if not m:
        continue
    t = m.group(1)
    if "Start load Lora adapter" in ln:
        u = ln.split("Lora name=", 1)[-1].strip()
        starts.append((t, u[-80:]))
        continue
    if "Start unload Lora adapter" in ln:
        u = ln.split("Lora name=", 1)[-1].strip()
        unloads.append((t, u[-80:]))
        continue
    i = lid.search(ln)
    if not i:
        continue
    d = loads.setdefault(i.group(1), {"start": [], "down": [], "comp": []})
    if "loading starts" in ln:
        d["start"].append(t)
    elif "downloading lora" in ln:
        d["down"].append(t)
    elif "loading completes" in ln:
        d["comp"].append(t)


def p(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


print("=== %s ===" % path)
print("--- Start load / Start unload 调用（tokenizer_manager 侧）---")
for t, u in starts[-12:]:
    print("  LOAD   %s  ...%s" % (t[11:], u[-46:]))
for t, u in unloads[-8:]:
    print("  UNLOAD %s  ...%s" % (t[11:], u[-46:]))

print("--- 每次 load 的 rank 进度与总时长 ---")
for l, d in loads.items():
    if not d["start"]:
        continue
    s = p(d["start"][0])
    e = p(d["comp"][-1]) if d["comp"] else None
    dur = ("%.1f min" % ((e - s).total_seconds() / 60)) if e else "未完成"
    print("  %s start=%s ranks: starts=%d downloads=%d completes=%d  dur=%s"
          % (l[:8], d["start"][0][11:], len(d["start"]), len(d["down"]), len(d["comp"]), dur))
    pairs = list(zip(d["down"], d["comp"]))
    if pairs:
        durs = [(p(b) - p(a)).total_seconds() for a, b in pairs]
        print("      每 rank 下载→完成: %s  (中位 %.0fs, 最大 %.0fs)"
              % (", ".join("%.0fs" % x for x in durs), sorted(durs)[len(durs) // 2], max(durs)))
    if d["comp"]:
        gaps = [(p(d["comp"][i + 1]) - p(d["comp"][i])).total_seconds() for i in range(len(d["comp"]) - 1)]
        if gaps:
            print("      rank 完成间隔: %s" % ", ".join("%.0fs" % g for g in gaps))
