# DeepSeek-V4.1-Flash 单机 4P+4D PD 分离性能报告

**测试日期**：2026-10-03
**环境**：1101 单机 8×B300（TP4 Prefill + TP4 Decode，IB 8×mlx5，cgroup v1 3.6 TiB，宿主内存 3.5 TiB）
**软件栈**：sglang fork `release/dsv41-flash-mol` + sgl-router mini-lb（端口 31000）+ mooncake KV 传输（RDMA）+ HiCache（P 端，ratio 4）+ CUDA Graph（D 端 max_bs 128）+ dSpark 投机解码（DSPARK，两端同开，uninitialized SPS 表 bootstrap）
**模型**：DeepSeek-V4.1-Flash（MXFP4，draft 权重内嵌 checkpoint）
**LoRA**：4 个随机初始化 MoL 适配器 L0-L3（rank 16 / alpha 32，权重限 21-39 层，`lora_kv_shared: true`——V4.1 persistent KV 只由 source 层 (2,8,14,20) 写入，21-39 层 LoRA 不影响任何缓存 KV）
**负载**（除非注明）：`sglang.benchmark.serving` random 数据集——4K token 输入 / 2048 输出 / 96 请求齐发（乱码 prompt 保证输出拉满，隔离 decode 性能）；flush-cache 冷态。

---

## 1. 四维性能矩阵

| 维度 | 聚合输出 tok/s | 总吞吐 tok/s | TPOT median | E2E mean | TTFT median | 成功/错误 |
|---|---:|---:|---:|---:|---:|---|
| **A. base（无 LoRA，dSpark 关）** | 3,400.45 | 10,568 | 14.46 ms | 17.42 s | 3.11 s | 96/0 |
| **B. LoRA 动态挂载（L0-L3 均匀混合，dSpark 关）** | 3,391.03 | 10,539 | 14.35 ms | 17.55 s | 3.57 s | 96/0 |
| **C. MoL Harness 全链路（route hop + 目标 LoRA，dSpark 关）** | 1,809.9 | — | —（见 2.2） | 32.68 s (p50) | — | 96/0 |
| **D. base + dSpark 投机解码（开启）** | **5,367.89** | **16,682** | **6.49 ms** | **9.59 s** | 2.91 s | 96/0 |

服务端错误全维度为 0（illegal memory / No kernel found / CUDA error）。

## 2. 分析

### 2.1 LoRA 动态挂载 = 零开销（B vs A：-0.3%）

4 路适配器按请求轮换（uniform），聚合输出 3,391 vs 3,400（-0.3%，噪声带内），TPOT/E2E 持平。机制：
- VE（virtual-experts）shrink/expand 是 Triton 分组融合 kernel，FLOPs 相对主 GEMM ≈ 1.4%（rank 16）
- delta 注入融合在 flashinfer 侧 C++ activation/finalize kernel 内（零额外 HBM 往返）
- 整段序列（含 VE）被 CUDA graph 捕获，launch 开销摊平
- **生产意义：MoL 多路由服务的 decode 侧无任何税**。

### 2.2 MoL Harness 全链路开销 = 47%（C vs B：1,810 vs 3,391）

每请求走 harness 完整流程：L0 route hop（路由 prompt 4.2K token + 32 token 决策，临时分支不提交）→ 严格两行解析 → 目标 LoRA 完整生成。

开销解剖（96 并发）：
- **route hop p50 8.96 s**：每请求额外一次 4.2K prefill → P 端总负载翻倍（96×8.2K≈787K token），P 端饱和排队
- gen p50 22.13 s（含 TTFT 排队后移）
- E2E p50 32.68 s = route 9.0 + gen 22.1（vs 直连 17.4 s，1.87 倍）
- 路由分布 L0:85 / L2:11，parse_error=0（两行契约 100% 解析成功）

**缓解方向**（按收益排序）：① route prompt 瘦身（去掉完整用户 prompt，用摘要/首尾窗口，route prefill 降到 <1K）② 路由决策缓存（同 session 前缀复用上次路由）③ route hop 与正式生成共享 prefill（route prompt 设计为正式 prompt 的前缀，radix 直接命中）。

### 2.3 dSpark 投机解码 = +58%（D vs A：5,368 vs 3,400）

- **配置要点**：V4.1 PD 分离下 DSPARK 是唯一允许的投机算法，且 **P/D 两端必须同开**（否则 `DSpark PD layout mismatch`——draft tokens/KV 布局在 P↔D 间强一致）
- draft 权重内嵌 checkpoint（`--speculative-draft-model-path` 自动默认到 `--model-path`）
- SPS 表缺省时 planner 自动构造 uninitialized 表（bootstrap 模式）——无需先离线生成表即可启动
- **实测接受率 0.22-0.37**（accept len 2.10-2.85）——bootstrap 表的保守成绩
- TPOT 14.46→6.49 ms（2.2 倍），E2E 17.42→9.59 s，聚合输出 +58%
- **进一步优化路径**：`dspark_sps_profiler` 在线采样（--base-url 指向运行中的 DSpark server）→ `dspark_sts_fit` 拟合 STS 校准 → 配置真表重启。预期接受率可显著提升（当前 0.37 上限还远低于标称潜力）

### 2.4 user339 真实流量补充（RPM 100，非本矩阵主负载）

平均 prompt 136.9K token / 输出 104 token（IO 比 1259:1），回放 1120 行：
- HiCache 修复后 TTFT p50 29.13→10.69 s（**2.7 倍**），p99 76.8→28.5 s
- 聚合输出 126.6-138.9 tok/s——贴住**数学上限**（RPM × 平均输出长度 ≈ 173 tok/s）；该流量形态下 D 端稳态仅 1-2 个并发流（`#running-req: 1`），瓶颈是到达率×输出长度而非引擎
- 168 组重复请求 + 前缀对实测可命中（受控同 prompt 重发 4.2 倍加速）

## 3. 本轮修复的三个性能 bug（皆有定罪证据）

1. **HiCache 从未挂载**：`--hicache-ratio` 只是比例参数，总开关 `--enable-hierarchical-cache` 缺失；且 sglang `host_memory.py` 的 cgroup v1 headroom 把 3.4 TiB 可回收 page cache 计入"已用"（`memory.usage_in_bytes` 含 `total_inactive_file`）→ 53 GiB 假上限。修复：补开关 + usage 扣 `total_inactive_file`（镜像 MemAvailable 口径）。ratio 8 会 OOM（全池镜像 1-2 TB 挤爆 cgroup 3.6 TiB），定 ratio 4。
2. **D 端 CUDA graph 超界**：`--cuda-graph-max-bs-decode 64`，96 并发时 91 流 fallback eager——日志铁证 `cuda graph: False` 时 gen throughput 415 tok/s vs `True` 时 3,914 tok/s（8 倍差）。修复：max_bs 128，聚合输出 707→3,400 tok/s。
3. **dSpark 单端开必崩**：PD 分离下 DSPARK 要求两端同开（KV 布局校验 `both servers must enable DSpark`）。

## 4. 结论

- **MoL 生产就绪**：LoRA 动态路由 decode 零开销，harness 全链路 47% 开销可优化（route prompt 瘦身/共享 prefill）
- **dSpark + PD 分离可行且收益显著**（+58%，bootstrap 表即可达），SPS/STS 正式标定后还有空间
- 系统能力边界清晰：4K/2048 负载下 D 端 4×B300 聚合 3,400（关 spec）~5,368（开 spec）tok/s；超长上下文低输出流量（user339 形态）的聚合输出由到达率×输出长度决定（数学上限），与引擎无关
- 原始数据：`/root/bench_pd/.tmp/sglangbench_long*.jsonl`、`mol_harness_results.json`、`sglangbench_dspark.jsonl`

## 附录：launch_pd.sh 当前配置

```
P（GPU 0-3）：--disaggregation-mode prefill --cuda-graph-max-bs-decode 8
             --enable-hierarchical-cache --hicache-ratio 4
D（GPU 4-7）：--disaggregation-mode decode --cuda-graph-max-bs-decode 128
COMMON：--tp 4 --enable-lora --max-lora-rank 16 --lora-use-virtual-experts
       --experts-shared-outer-loras --lora-target-modules q_proj gate_up_proj down_proj
       --speculative-algorithm DSPARK（两端同开）
       --served-model-name deepseek-v4.1-flash
```
