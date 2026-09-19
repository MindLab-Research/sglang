# MoL 融合 kernel 单机验证：启动即崩的四个根因（2026-09-19）

> 场景：b300-4（`43.202.208.136`，8×B300）用 GLM-5.3 单机 MoL 验证
> PR #27（LoRA mmap fast load）+ PR #31（per-group routing / fused delta）。
> 四个 bug 全部让引擎在 **CUDA graph capture** 阶段崩，且**报错位置与真正根因无关**。

## 症状

引擎每次都在 CUDA graph capture 阶段崩溃：

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
Exception: Capture cuda graph failed: Triton Error [CUDA]: an illegal memory access was encountered
[2026-09-19 05:42:20] Received sigquit from a child process.
```

**误导点**：栈顶是 **base MoE down-proj kernel**（`fused_moe.py:715` →
`fused_moe_triton_kernels.py:907 invoke_fused_moe_kernel`），那个 kernel 一行都没改。

真正原因是**前一个自定义 kernel 已经 IMA 污染了 CUDA context**，后续任意
kernel launch 才把错误抛出来。

> **判据**：capture 期间报 IMA 且栈顶落在无关 kernel 上时，必须往前找**第一个**
> 执行的自定义 kernel，而不是盯着栈顶。

---

## 根因 1：`envs.XXX` 缺 `.get()` → `EnvField.__bool__` raise

```
File ".../kernels/ops/moe/virtual_experts.py", line 1699, in merged_experts_fused_moe_lora_add
    if _per_group_routing_enabled():
File ".../srt/environ.py", line 109, in __bool__
    raise RuntimeError(
RuntimeError: Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`
```

新写的开关访问函数**返回 `EnvField` 对象本身**而不是它的值：

```python
# BAD
def _per_group_routing_enabled() -> bool:
    return envs.SGLANG_LORA_PER_GROUP_ROUTING      # EnvField, 不是 bool

# GOOD
def _per_group_routing_enabled() -> bool:
    from sglang.srt.environ import envs
    return envs.SGLANG_LORA_PER_GROUP_ROUTING.get()
```

`sglang.srt.environ.EnvField.__bool__` / `__len__` 会显式 raise，用**运行时 trap**
替代静默的真值判断。顺带去掉 module 级缓存（`os.getenv` 很便宜，且便于运行时 A/B）。

---

## 根因 2：`num_valid_tokens` 用了 worst-case buffer 尺寸

`token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)` 的上界必须等于
**真实** flattened token-expert 槽位数 `topk_ids.numel()`。两个新 wrapper 却传了
`sorted_token_ids.numel()`——这是 moe_align 输出的 **worst-case 尺寸**缓冲区长度：

```
sorted_token_ids.numel() = 695     <- 传错的（缓冲区容量）
topk_ids.numel()         = 128     <- 正确的（真实槽位数）
num_tokens_post_padded   = 512
```

缓冲区内 `[128, 695)` 是**未初始化 / freed-memory 复用**的垃圾值。

micro bench 判据（`poison = 411 ∈ [128, 695)`）：

| bound | 结果 |
|---|---|
| `695`（错） | poison 通过 mask → `token_rows = 411 // 8 = 51` > `hidden` 的 16 行 → **out-of-range reads** |
| `128`（对） | mask 拒绝 → 与 torch 参考一致（max diff 0.022 @ bf16） |

大垃圾值（`2**30`）**不会**触发——它同时超出两个上界。**只有落在
`[真实bound, 错误bound)` 区间的小正整数才越界**，所以它是内存状态相关的间歇故障。

---

## 根因 3：部署链路两层缓存 —— "改了但没生效"

**现象**：源码里 grep 已确认修复存在，引擎仍报**完全相同**的旧错误。

两个原因叠加：

### 3a. overlay 打包自错误的工作区状态

修复在 `git stash push` / `stash pop` 往返中被旧版本覆盖，**从未写入文件**。
工具返回 "Successfully replaced" 不代表改动一定留在最终产物里 —— 后续的
`git checkout` / `stash pop` 都可能把它冲掉。

> **铁律**：改完必须**立刻**用 `grep -c` 断言写入；生成产物（tar/镜像/overlay）前
> 再断言一次。

### 3b. 旧 `.pyc` 进了 overlay

打包 overlay 时未排除 `__pycache__`；且容器（root）会把编译产物写回**挂载目录**，
下一次 `rm` 甚至因权限失败：

```
virtual_experts.cpython-312.pyc   05:26   <- 容器编译的缓存
virtual_experts.py                05:25   <- 源文件
```

`.pyc` 比 `.py` **新** → Python 判定缓存有效 → 加载旧代码。
（AGENTS.md §8 已有此铁律，本次再次踩中。）

**正确打包 / 解压**：
```bash
tar czf overlay.tar.gz --exclude='__pycache__' --exclude='*.pyc' -C python sglang
ssh node 'sudo rm -rf <overlay_dir>; mkdir -p <overlay_dir>; cd <overlay_dir>; tar xzf ...'
```
必须 `sudo rm -rf`：容器以 root 运行，写回的 `.pyc` 属主是 root，普通用户删不掉。

> **md5 校验不够**：md5 只覆盖 `.py`，看不见 `.pyc` 的遮蔽。

### 容器内断言（重启前的最后一道闸）

```python
import inspect
from sglang.kernels.ops.moe import virtual_experts as ve
print(ve.__file__)                                   # 确认是挂载的 overlay
assert ".get()" in inspect.getsource(ve._per_group_routing_enabled)
print(ve._per_group_routing_enabled())               # True，不 raise
```

**只有这一步通过，才允许重启。**

---

## 根因 4：output 是 3D 而 kernel 按 2D 索引 —— stride 错位

修完根因 1/2 后 capture **仍然** IMA。kernel 级 micro bench 却正好 PASS ——
因为 **micro bench 传的是 2D**，而真实 hook 传的 **3D**。

`lora_moe_runners.py` 明确写出形状：

```python
M, top_k, gate_up_dim = intermediate_cache.shape   # <-- 3D!
...
merged_experts_fused_moe_lora_add(output=intermediate_cache, ...)
```

wrapper 把 3D tensor 的 stride 当成 (行, 列) 用：

```python
output.stride(0), output.stride(1),      # 3D: (top_k*N, N)  ← 错!
```

kernel 内部按 flattened 槽位索引：

```python
token_rows = offs_token // top_k
result = tl.load(output_ptr + token_rows * stride_om + offs_n * stride_on)
```

对 3D `[M, top_k, N]`：
- `stride_om = top_k * N`（应该是 `N`）
- `stride_on = N`（应该是 `1`）

→ `offs_token * top_k * N` 与 `offs_n * N` **双重越界** → IMA。

**修复**（两个 wrapper 各一处）：
```python
if output.dim() == 3:
    output = output.reshape(-1, output.shape[-1])
```
contiguous 的 3D `reshape(-1, N)` 是 view，原地写入仍然生效。

**判据**：复现测试同时跑 **2D 与 3D 两条路径**，要求都不 IMA 且结果一致 —— 只测
2D 会漏掉这个 bug（这正是第一轮 micro bench 的盲区）。

---

## 验证方法（可复用）

### 1. kernel 级 micro bench（不起引擎，秒级迭代）

1. **C1 正确性**：`out[slot] += B_l @ (A_l @ hidden[t])` 与 torch 逐槽参考对拍，
   断言 `max abs diff <= bf16 容差` **且 delta 非零**（防止"什么都没算"也判过）。
2. **C2 垃圾 padding + 正确 bound**：slack 填 `2**30 / -1 / 123456789` → 不崩、结果不变。
3. **C3 复现旧 bug**：填 `poison = real + (bad-real)//2`（落在两界之间）→ 错误 bound
   观察到越界读，正确 bound 拒绝同一份数据。
4. **形状矩阵**：必须覆盖**真实形状**——本次的教训是 NE=8 太小、且 output 用 2D
   掩盖了 3D 的 stride bug。真实值：`NE=1024, K=1536/6144, N=6144/1536, NT=64`。
5. **`_align_block_size_large`**：`num_experts >= 1024` 时 **不能**用
   `moe_align_block_size`（native 只处理小专家数），必须走 `_align_block_size_large`
   → `_align_block_size_jit`。

### 2. 通用教训

- `sorted_token_ids` 这类 **worst-case 缓冲区的 `numel()` 不能当合法索引上界**。
- 自定义 Triton kernel 报错栈可能指向**后一个** kernel（IMA 是异步的）。
- **micro bench 的输入必须与真实调用路径同构**（维度/布局/专家数），否则
  "PASS" 是虚假安全感 —— 本次两个 bug（2 的 bound、4 的 3D）都因形状不对而漏掉。
- 部署产物要断言**内容**，不是断言**文件存在**。

## 相关

- PR #31 `feat/lora-per-group-routing`（修复 commit `025a4b6000`）
- 复现脚本：`/nvme/tmp/repro_ima_real_shapes.py`（b300-4）
- micro bench：`/nvme/tmp/micro_bench_lora_delta.py`（b300-4）

## 根因 5：fused delta kernel 用 stride(0)/stride(1) 而非 stride(-2)/stride(-1)

修完根因 1-4 后，EAGLE + LoRA 仍然 SIGQUIT 崩溃（base 正常 837 tok/s，
3-LoRA 全部 CUDA error）。崩溃栈异步报告在 `_write_mla_kv_buffer` 的 loc
断言，真正的 IMA 来自前面的 fused delta kernel。

**根因**：我们的 `_fused_lora_delta_kernel` 用 `output.stride(0)` 和
`output.stride(1)` 传递行/列 stride。当 output 是 **3D** `[M, topk, N]`
（EAGLE TARGET_VERIFY 的 `intermediate_cache1`/`intermediate_cache3`）时：

- `stride(0) = topk * N`（不是行 stride！）
- `stride(1) = N`（不是列 stride！）

→ `offs_token * (topk * N)` 严重越界 → IMA。

**为什么原版不崩**：上游 `invoke_fused_moe_kernel` 用 `C.stride(-2)` 和
`C.stride(-1)`：
- 3D `[M, topk, N]`：`stride(-2) = stride(1) = N`，`stride(-1) = stride(2) = 1` ✓
- 2D `[M*topk, N]`：`stride(-2) = stride(0) = N`，`stride(-1) = stride(1) = 1` ✓

`stride(-2)/stride(-1)` 对 2D 和 3D 都取到正确的行/列 stride。

**修复**：两个 wrapper 的 `output.stride(0), output.stride(1)` →
`output.stride(-2), output.stride(-1)`，与原版一致。同时删掉 3D reshape
（不再需要，因为 stride(-2)/(-1) 对 3D 正确）。

**验证**：micro bench（NE=1024 + 2D/3D 双路径）ALL PASS，2D vs 3D agreement
= 0.000000（逐位一致）。

**教训**：自定义 kernel 的 stride 传递要用 `stride(-2)/stride(-1)` 而非
`stride(0)/stride(1)`，与上游 `invoke_fused_moe_kernel` 一致——前者对 N-D
tensor 取最后两维的 stride（始终是行/列），后者只对 2D 正确。
