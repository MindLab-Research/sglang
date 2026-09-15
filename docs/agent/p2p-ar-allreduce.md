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

## 8. 2026-09-15 部署实录：init 链路 3 个真 bug（全部实测定位）

在 1021/1022（8.213.214.14）首次真正启停验证 `SGLANG_P2P_AR=1`，双端 8 rank 全部：

```
p2p_ar: init failed (cuda error 1); disabled
```

逐层剥出 3 个 bug（本特性从未跑过 ⇒ 全部是首次暴露）：

### 8.1 `cudaIpcMemHandle_t` 必须按值传（cuda error 1 的真身）

`cudaIpcOpenMemHandle` 第二个参数是 64 字节结构体，ABI 归 **MEMORY 类 = 按值传递**。
原实现传 `ctypes.c_char * 64` 数组，ctypes 会把它当**指针**传 → callee 把指针值当句柄内容
读 → `cudaErrorInvalidValue(1)`。

两进程实验（A 进程 allocate+get，B 进程 open，同一张卡）：

| 传参形式 | flags | 结果 |
|---|---|---|
| `c_char*64` 数组（原实现） | `cudaIpcMemLazyEnablePeerAccess` | **err=1** |
| 64B `ctypes.Structure` **按值** | `cudaIpcMemLazyEnablePeerAccess` | **err=0**（ptr 解析成功） |
| 结构体按值 | 0 | err=1 ⇒ **peer-access flag 必需** |

修复：新增 `_IpcMemHandle(ctypes.Structure)`；`cudaIpcGetMemHandle` 传 `byref(struct)`、
`cudaIpcOpenMemHandle` 传 **struct 实例**；并显式设 `argtypes/restype`（否则 ctypes 退回
整型/指针 marshalling，同样得到 err=1）。

### 8.2 `handle.bytes` 在第一个 NUL 处截断（第二层误导性错误）

IPC 句柄 64 字节中只有 ~19 字节非零；对 `c_char * 64` 字段取 `.bytes`/`.value` 会**在第一个
NUL 处截断** → `mine` 变 4 字节（个别 rank 0 字节）→

```
ProcessGroupGloo::allgather: invalid tensor size at index 0 (expected (4), got (64))
both buffer length (0) and count (-1) must not be 0
```

修复：`ctypes.string_at(ctypes.byref(handle), 64)` 取全量 + `assert len(raw) == 64`。
同族坑：句柄交换必须用 **CPU 张量** —— 传进来的 group 是 **gloo**，不支持 CUDA 张量。

### 8.3 `build.sh` 静默编错架构（丢 sm_103a）

`set -euo pipefail` + `nvcc --help | grep -q compute_103a`：grep 命中即退出 → nvcc 收
SIGPIPE(141) → 管道整体判失败 → 走"不支持"分支 → 产物只有 `sm_90a/sm_80` cubin，而
**build.sh 仍返回 0**。B300(SM103) 上加载即失败，且只在 load 时暴露。

修复：先抓 help 到变量再 grep（`NVCC_HELP="$("$NVCC" --help 2>&1 || true)"`）+ 不支持时打
WARNING；重建后 `cuobjdump -lelf` 确认含 `sm_103a`。

### 8.4 blob 视图不能再用 `torch.Tensor().set_(cuda_blob, ...)`

修完 8.1–8.3 后暴露的第四个 init bug（双端 8 rank）：

```
p2p_ar: init failed (Attempted to set the storage of a tensor on device "cpu" to a
        storage on different device "cuda:N". This is no longer allowed; the
        devices must match.)
```

`staging_local` / `ready_local` 原本写成
`torch.Tensor().set_(self._blob, off, shape)`：`torch.Tensor()` 是 **CPU** 张量、`self._blob`
在 **CUDA**，现代 torch 拒绝把 CPU 张量的 storage 重指到 CUDA storage。
修复：直接**切片 CUDA blob** 再 `.view(dtype)`（storage 天然在正确设备）。

### 8.5 该部署的 profiler 不可用（验证手段受限）

`/stop_profile` 报 `RuntimeError: Profiling is not in progress`，trace 永不落盘 ⇒ 内核级证据
只能走日志探针（同 `MOL-PROBE` 思路：在 host 侧把真正交给 kernel 的 config/grid 打一次），
profile 口径的 A/B 需要另找环境或先修 profiler。

> 教训：P2P AR 这条链上 **4 个 bug 全部只在真机启停时才暴露**（IPC 句柄 ABI、字节截断、
> 编译架构探测的 SIGPIPE、张量 storage 设备语义）。"能 import、能编 .so、逻辑自洽"
> 完全不能替代一次真实启停。
