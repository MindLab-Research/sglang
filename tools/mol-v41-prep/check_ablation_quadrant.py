#!/usr/bin/env python3
"""R9 判别器（含 rt-runtime 的拓扑二分）：SKIP_DELTA 生效性 × MoE 段拓扑 × p50 落点 → 唯一判决。

「无 delta kernel」有四种成因，光看 p50 分不开，必须看拓扑：
  (a) 无 active adapter            → 分解链在(n_decomp>0)、delta=0
  (b) adapter 活跃但请求侧未选路    → 同上，需请求侧证据（smg 日志 + /v1/models）
  (c) 走了 stock 分流（BYPASSED / nolora 图）→ n_stock>0 且 n_decomp==0
       代码依据：lora_dispatch.py:780-787 遇 BYPASSED 直接 return fused_experts_none_to_flashinfer_trtllm；
                 deepseek_v2.py:112 _lora_forces_standard_topk() 决定 topk 是否强制 STANDARD
  (d) 桶正则没命中 / 窗口没抓到    → n_stock==0 且 n_decomp==0

用法: python3 check_ablation_quadrant.py --nsys-json kern_C2.json --bench-json C2.json [--pid <decode_pid>]
"""
from __future__ import annotations
import argparse, json, sys

RT_LO, RT_HI = 6.44, 6.60      # rt-runtime 先验（回收量上界，右收）
MY_LO, MY_HI = 6.90, 7.20      # rt-measure 先验
C1_REF, C1_TOL = 7.281, 0.05   # C1 复现窗口 ±5%

DELTA_BUCKETS = ("delta_shrink", "delta_expand", "gemv_k1_gateup", "gemv_k2_down")
DECOMP_BUCKETS = ("gemm_lora_unfused", "activation_lora_input", "decomposed_moe_lora")
STOCK_BUCKETS = ("gemm_base_fused",)

def calls(d, names): return sum(int(d.get(k, {}).get("calls", 0)) for k in names)
def p50_of(b):
    v = b.get("client_agg_p50_ms")
    if v is None:
        rs = [r for r in b.get("rounds", []) if r.get("tpot_p50_ms")]
        v = sorted(r["tpot_p50_ms"] for r in rs)[len(rs)//2] if rs else None
    return v
def env_flag(pid):
    try: env = open(f"/proc/{pid}/environ","rb").read().decode(errors="replace").split("\0")
    except Exception as e: return None, f"读不到: {e}"
    return dict(x.split("=",1) for x in env if "=" in x).get("SGLANG_MOE_LORA_SKIP_DELTA"), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys-json", required=True); ap.add_argument("--bench-json", required=True)
    ap.add_argument("--pid", type=int, default=None); a = ap.parse_args()
    N = json.loads(open(a.nsys_json).read()); p = p50_of(json.loads(open(a.bench_json).read()))
    n_delta, n_decomp, n_stock = calls(N, DELTA_BUCKETS), calls(N, DECOMP_BUCKETS), calls(N, STOCK_BUCKETS)
    print(f"kernel 计数: delta/VE={n_delta}  decomp_lora={n_decomp}  stock_fused={n_stock}   C2 p50={p} ms")
    if a.pid:
        e, err = env_flag(a.pid); print(f"/proc/{a.pid}/environ SGLANG_MOE_LORA_SKIP_DELTA={e!r} {err or ''}")
        if e != "1":
            print("VERDICT R9-HARD: env 没进 server 进程（不问 p50）。该轮作废，先修 launcher export。"); return 1
    if p is None: print("VERDICT: 无 p50 数据，无法判别"); return 2
    if n_stock > 0 and n_decomp > 0:
        print("VERDICT 前置失败: R8 混批嫌疑 —— 同一窗口同时出现 stock 与分解链 ⇒ 必须先按 NVTX 步边界把"
              " lora/nolora 图分开（decode_cuda_graph_runner.py:1104-1125），否则 p50 是两图混合值，不可用于归因。"); return 1
    if n_delta > 0:
        print("VERDICT 格3: delta 类 kernel 仍在跑 ⇒ SKIP_DELTA 未生效（env 未 export / 走了别的 dispatch 分支）。"
              "先查 /proc/<pid>/environ 与 launcher，再谈结论。"); return 0
    if n_decomp == 0 and n_stock == 0:
        print("VERDICT (d): MoE 段都没抓到 ⇒ 桶正则或采样窗口有问题，先修提取器，别下结论。"); return 2
    if n_decomp == 0 and n_stock > 0:
        print("VERDICT 格4c: 走了 stock 分流（BYPASSED / nolora 图）⇒ C2 对该 batch 无效，"
              "不是『delta 免费』。补足三样证据：请求侧 adapter 命中（smg 日志）+ /v1/models 有 dsv41_moe_rnd "
              "+ 拓扑二分（本条）。"); return 0
    if RT_LO <= p <= RT_HI:
        print("VERDICT 格1: rt-runtime 先验成立。剩余 ~1.0ms 全在分解 C++ 核 + pad/contiguous 桥 + alloc ⇒ P-A 必须整段替换。")
    elif MY_LO <= p <= MY_HI:
        print("VERDICT 格2: rt-measure 先验成立。VE 链比估算轻，大头在 attn csgmv / 快照污染 ⇒ P-A 上限上修 ~5.6。")
    elif abs(p - C1_REF) <= C1_TOL:
        print("VERDICT 格5: delta 链几乎免费（两方先验双双落空）⇒ 1.87ms 全在别处，归因方向必须重开。")
    elif abs(p - C1_REF) <= C1_TOL and n_delta == 0:
        print("VERDICT: 与 C1 同值，先按格5/格4(a)(b) 复核请求侧证据。")
    else:
        print("VERDICT: 落在已知格之外 ⇒ 先查口径（R6 --cuda-graph-max-bs-decode / 外部流量 / key 污染），不下结论。")
    return 0

if __name__ == "__main__": sys.exit(main())
