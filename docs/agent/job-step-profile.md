# decode step 时间分解（torch-profiler，1021/1022 GLM-5.3 MoL）

> 用途：给"通信到底占多少 / 优化哪个方向"提供**实测**依据（而不是估算）。
> 本文的方法与数据支撑 PR #23（P2P 全归约，替 TP 维 AR）的收益上限判断。

## 1. 采集方法（**不重启、不改启动参数**）

引擎自带 profiler 端点（`http_server.py` 的 `/start_profile`、`/stop_profile`），
`ProfileReq` 支持 `num_steps`（到步数自动停）、`output_dir`、`activities`：

```bash
curl -X POST http://127.0.0.1:30200/start_profile -H 'Content-Type: application/json' \
  -d '{"profile_id":"prof_x","num_steps":60,"output_dir":"/root/prof",
       "activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}'
# 60 步后自动停（无需 stop_profile），8 个 rank 各落一份
#   /root/prof/<id>-TP-{0..7}.trace.json.gz   (~32 MB/rank)
```

**窗口纪律**（前作教训）：
- 采样前 **warmup**（cuda-graph 捕获 / JIT / KV 池稳态都要先过去）；
- **同 bs / 同 seq_len 才可比**（混批次均值会误判）；
- profiler 自身有开销：本集群实测 **wall/step 比不采样时高 ~10–16%** →
  **占比可信，绝对 latency 以日志法为准**（`step_ms = #running-req × accept_len ÷ gen_throughput`）。

## 2. 分析口径

`tools/analyze_step_profile.py`（Node 上也有一份 `/root/analyze_step_profile.py`）：

```
zcat /root/prof/<id>-TP-0.trace.json.gz | python3 analyze_step_profile.py /dev/stdin 60
```

规则（**只统计 `cat=="kernel"`**！profiler 的 `gpu_user_annotation` 是嵌套 span，
把它们算进来会把总量虚高 ~4 倍并让分类错位 —— 我们第一版就踩过：`eagle 59.6%` 是假的）：

| bucket | 命中规则（节选） |
|---|---|
| comm_allreduce / allgather / reducescatter / p2p | `all_?reduce|AllReduce` / `all_?gather` / `reduce_?scatter` / `send|recv|_p2p` |
| moe_gemm | `fused_moe|moe_|grouped_gemm|deep_gemm|fp8_gemm|scaled_mm|gemm_` |
| attn | `triton_.*attn|flash|mla|attention|dsa_|trtllm|fmha` |
| lora | `csgmv|bgmv|sgmv|lora` |
| eagle | `draft|eagle|topk|verify|tree` |
| norm | `rms|layer_?norm|mhc` |
| elementwise | `elementwise|vectorized|add|mul|copy|cat|index|gather|scatter|silu|cast|...` |

## 3. 实测结果（decode 1022，60 步窗口，TP0）

| bucket | bs≈11（基线） | **bs=23** | **bs=20** |
|---|---|---|---|
| **moe_gemm** | 43.4% | **45.7%** | **44.6%** |
| **comm_allreduce（TP 维 AR）** | **10.7%** | **10.5%** | **10.4%** |
| other | 7.9% | 7.4% | 7.5% |
| attn（DSA / fmhaSm100f） | 7.1% | 7.3% | 7.2% |
| elementwise | 7.7% | 6.9% | 6.9% |
| norm | 6.2% | 5.7% | 5.7% |
| **comm_allgather** | 5.5% | **5.7%** | **6.7%** |
| lora（MoL） | 5.0% | 4.6% | 4.6% |
| **comm_reducescatter（f32）** | 3.9% | **4.2%** | **4.1%** |
| eagle | 2.6% | 2.1% | 2.2% |
| **通信合计** | **20.1%** | **20.3%** | **21.2%** |
| GPU busy/step ｜ wall/step | 71.1 ｜ 85.6 ms | 85.4 ｜ 99.3 ms | 82.2 ｜ 95.5 ms |
| GPU 占空（busy/wall） | 83% | 86% | 86% |

Top kernel（bs=23，绝对耗时）：

```
 1905.4 ms  37.2%  fused_moe_kernel
  524.6 ms  10.2%  ncclDevKernel_AllReduce_Sum_bf16_RING_LL
  278.2 ms   5.4%  ncclDevKernel_AllGather_RING_LL
  213.1 ms   4.2%  ncclDevKernel_ReduceScatter_Sum_f32_RING_LL
  210.7 ms   4.1%  _moe_lora_shrink_splitk_kernel
  174.0 ms   3.4%  fmhaSm100fKernel_...TokenSparse...          (DSA)
  131.6 ms   2.6%  _chunked_lora_shrink_kernel
  103.0 ms   2.0%  _chunked_lora_expand_kernel
   87.0 ms   1.7%  at::native::elementwise_kernel (direct_copy)
   82.9 ms   1.6%  at::native::sbtopk::gatherTopK              (EAGLE topk1)
   66.2 ms   1.3%  deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl
```

容器位置（bs≈11 窗口，供"其它"对照）：`p2p_ar/p2p_ar_bf16.cu` 等同族 —— 见 PR #23。

## 4. 结论

1. **通信合计 20–21%，且不随 batch 变大而恶化**（20.1% → 20.3% → 21.2%）；
2. 通信里 **TP 维 bf16 AllReduce 是最大单项：10.4–10.7%**（占通信 ~52%），
   RING 算法（`..._RING_LL`）→ 有 P2P 替换空间；
3. 另一半是 **AllGather 5.5–6.7% + ReduceScatter(f32) 3.9–4.2%**（DCP/其它 pattern），
   **不是 all-reduce，P2P AR 替不了**，需要各自优化；
4. **单点最大仍是 MoE GEMM 44–46%**（`fused_moe_kernel` 36–37% + `deep_gemm` ~1.3%）——
   AR 之外的最大收益在这里；
5. ⇒ **P2P AR 的收益上限 ≈ step 时间 −5~6%**（把 10.5% 的 AR 砍一半以上）。

## 5. 采集脚手架（节点上，自动攒窗口）

`/root/watch_bs_profile.sh`：每 5 分钟检查 decode 并发，**>16 就自动**采 60 步
（`/start_profile`）→ 等 8 份 trace 落盘（**要求 size 稳定 + `gzip -t` 通过**，否则会读到
半成品 → `JSONDecodeError`，我们踩过）→ 自动分析 TP0 → 写
`/root/prof/report_bs16_<时刻>.txt`。带单例锁（不会两个 profile 撞车），
只留 TP0 且最多 3 份（控盘）。
