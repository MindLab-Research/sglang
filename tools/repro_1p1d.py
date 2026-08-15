#!/usr/bin/env python3
"""1P1D 复现脚本：填高 pool 水位 + 缓存命中 + DCP 传输触发"""
import urllib.request, json, sys, time, threading

URL = "http://8.213.215.2:18888/v1/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer MOL_API_KEY_1P1D"}

def send(rid, content, max_tokens=30, timeout=180):
    payload = json.dumps({
        "model": "deepseek-v4-pro-0813",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers=HEADERS)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(resp.read())
        usage = d.get("usage", {})
        print(f"[{rid}] OK {time.time()-t0:.1f}s usage={usage}", flush=True)
        return True
    except Exception as e:
        print(f"[{rid}] FAIL {time.time()-t0:.1f}s {e}", flush=True)
        return False

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fill"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # 每个请求唯一长文本（~20K tokens：内容 + 唯一后缀避免完全命中，部分命中缓存）
    base_text = "量子计算与人工智能融合的前沿技术研究报告：量子比特的相干时间、门保真度、纠错阈值、表面码与拓扑编码、变分量子本征求解器、量子退火与绝热演化、量子机器学习中的核方法。" * 30

    def worker(rid):
        # 唯一后缀 → 部分缓存命中（前缀相同命中，后缀新算）
        content = base_text + f" 第{rid}节：具体技术细节与实验数据对比分析。"
        send(rid, content)

    if mode == "fill":
        # 并发发 n 个长请求（填 pool 水位）
        threads = []
        for i in range(n):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
            time.sleep(0.5)  # 错开避免超时
        for t in threads:
            t.join()
    elif mode == "hit":
        # 重发相同内容（命中 HiCache 缓存 + DCP 传输高位页）
        for i in range(n):
            send(f"hit-{i}", base_text + f" 第1节：具体技术细节与实验数据对比分析。")

if __name__ == "__main__":
    main()
