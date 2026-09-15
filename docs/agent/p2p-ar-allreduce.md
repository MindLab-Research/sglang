# P2P all-reduce (bf16, NVLink peer-to-peer) —— flag 启用

> 目标：把 decode step 里 **10.7% 的 TP 维 bf16 all-reduce**（profile 实测
> `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`）换成 NVLink P2P kernel。
> 协议照搬 Ferrite 的 `ferrite_p2p_ar_v5`（store → publish → pubred + epoch 双缓冲 +
> per-rank ready 表），入口改成 **bf16 进/出、fp32 累加**，以对齐 torch/NCCL 的 bf16 AR 语义。

## 1. 开关

| env | 作用 | 默认 |
|---|---|---|
| `SGLANG_P2P_AR` | `1` = 启用（走 P2P AR），`0` = 关闭 | `0`（关） |
| `SGLANG_P2P_AR_LIB` | `libp2p_ar.so` 路径（`build.sh` 产物） | 空 → 自动降级 NCCL |
| `SGLANG_P2P_AR_MAX_ELEMS` | staging 容量上限（元素数，决定单次 AR 的最大 tensor） | 8M（fp32 staging ≈64 MB/rank） |

## 2. 构建

```bash
cd python/sglang/srt/distributed/device_communicators/p2p_ar
./build.sh                      # 产出 libp2p_ar.so
# B300 必须 long-form gencode（AGENTS §8）：
#   -gencode arch=compute_103a,code=sm_103a
```

**独立 .so，不需要重建 sglang / sgl_kernel**（用 ctypes 加载）。

## 3. 生效路径（代码）

| 文件 | 作用 |
|---|---|
| `.../p2p_ar/p2p_ar_bf16.cu` | 2 个 kernel：`p2p_ar_store_bf16_kernel`（把自己 partial 写进每个 peer 的 staging 槽 → publish）、`p2p_ar_pubred_bf16_kernel`（publish 自己的 ready → poll 各 peer → 按 rank 升序 fp32 累加 → 写回 bf16） |
| `.../p2p_ar/p2p_all_reduce.py` | IPC peer 指针表（`cudaIpcGetMemHandle`/`OpenMemHandle`）、staging `[2][world][stride]`、ready 表、epoch 计数器、ctypes 启动、`should_p2p_ar()` 门控 |
| `.../parallel_state.py` | `GroupCoordinator.__init__` 建 `self.p2p_ar`；`_resolve_outplace_all_reduce_method` 优先返回 `"p2p"`；`_all_reduce_out_place` 分发（clone 后原地 reduce，遵守 out-of-place 契约） |

## 4. 数值口径（重要）

partials 进 AR 前**已经是 bf16**（torch `dist.all_reduce` 看到的就是 bf16 部分和）→ kernel
**bf16 → fp32 存入 staging → fp32 升序累加 → bf16 写回**，与参考实现（bf16 partials + fp32 累加
+ bf16 结果）一致。Ferrite 那份注释也记录过：**少了这步 bf16 舍入会引入系统性 ~0.5 ulp 偏移**，
在 40 层残差流里放大到 attn_o 0.63–0.73% / moe_o 1.7%。

## 5. 约束与降级（silently fall back）

| 条件 | 行为 |
|---|---|
| `dtype != bf16`、非连续、`numel % 4 != 0`、`numel > MAX_ELEMS` | 回落原路径（NCCL / custom-AR） |
| `world < 2` 或 `> 8`（PVP blockDim 约束 64） | 建通信子即 disabled |
| 非单机 / 无 NVLink P2P / IPC 打开失败 | disabled + warn，回落 |
| cuda-graph | 全部 buffer 初始化时预分配、无热路径分配；epoch 在 kernel 内推进 → **可捕获** |

## 6. 验证计划（未做，需要能重启的环境）

1. **正确性**：8 rank torchrun，同一个 bf16 tensor 分别过 P2P 与 NCCL，逐元素比对
   （容差按 bf16：`max|Δ| ≤ 1 ulp`；期望完全一致，因为两者都是 fp32 累加 + bf16 舍入）；
2. **图捕获**：`torch.cuda.graphs.CUDAGraph` 里连跑 100 次 P2P AR，比对第 1 次与第 100 次结果（epoch 推进不能错位）；
3. **A/B 性能**：用 `docs/agent/job-step-profile.md` 的 profile 口径（同 bs/seq_len），
   对比 `SGLANG_P2P_AR=0/1` 的 `nccl AllReduce` 是否被 kernel 取代、step 时间变化；
   判据：AR 那 10.7% 至少砍掉一半（Ferrite 实测 AR 从 5.9 µs 级降到 1.5 µs 级）；
4. **回归**：cases_50 全过 + 输出无乱码（AR 数值路径变了，必须验）。

## 7. 上线方式

`SGLANG_P2P_AR=1` + `SGLANG_P2P_AR_LIB=/path/libp2p_ar.so` 是**启动参数**，需**重启**引擎生效
（decode 端先上，prefill 端同理）。未验证前不要在生产 1021/1022 打开。
