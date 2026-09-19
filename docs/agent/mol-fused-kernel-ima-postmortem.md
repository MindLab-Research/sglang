# MoL 融合 kernel 单机验证：启动即崩的根因排查记录（2026-09-19）

> 场景：b300-4（`43.202.208.136`，8×B300）用 GLM-5.3 单机 MoL 验证
> PR #27（LoRA mmap fast load）+ PR #31（per-group routing / fused delta）。
> 三个 bug 都让引擎在**启动阶段**（CUDA graph capture）崩，且**报错位置与真正根因无关**。

## 症状

引擎每次都在 CUDA graph capture 阶段崩溃：

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  at fused_moe_kernel[grid] -> _init_handles -> load_binary
```

**误导点**：栈顶是 **base MoE kernel**（`fused_moe_triton_kernels.py:907`），但那个
kernel 一行都没改。真正原因是**前一个 kernel 已经 IMA 污染了 CUDA context**，
后续任意 kernel launch 才把错误抛出来。

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
    ...
    return envs.SGLANG_LORA_PER_GROUP_ROUTING      # -> EnvField, 不是 bool

# GOOD
def _per_group_routing_enabled() -> bool:
    from sglang.srt.environ import envs
    return envs.SGLANG_LORA_PER_GROUP_ROUTING.get()
```

`sglang.srt.environ.EnvField.__bool__` / `__len__` 会显式 raise，用**运行时 trap**
替代静默的真值判断。这个坑只在**新加的 flag** 上暴露（老 flag 在 b300-glm52 里不存在，
不会走到这条路径）。顺带去掉 module 级缓存（`os.getenv` 很便宜，且便于运行时 A/B）。

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

原版 `_invoke_moe_lora_shrink_splitk` 一直传 `topk_ids.numel()`——新 kernel 抄了
mask 逻辑但漏了对齐该契约。

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

打包 overlay 时未排除 `__pycache__`：

```
virtual_experts.cpython-312.pyc   05:26   <- 容器编译的缓存
virtual_experts.py                05:25   <- 源文件
```

`.pyc` 比 `.py` **新** → Python 判定缓存有效 → 加载旧代码。
（AGENTS.md §8 已有此铁律，本次再次踩中。）

**正确打包**：
```bash
tar czf overlay.tar.gz --exclude='__pycache__' --exclude='*.pyc' -C python sglang
```
解压前还要 `rm -rf` 目标目录（残留的 `.pyc` 不会被解压覆盖）。

> **md5 校验不够**：md5 只覆盖 `.py`，看不见 `.pyc` 的遮蔽。

### 容器内断言（重启前的最后一道闸）

```python
import inspect
from sglang.kernels.ops.moe import virtual_experts as ve
print(ve.__file__)                                   # 确认是挂载的 overlay
assert ".get()" in inspect.getsource(ve._per_group_routing_enabled)
print(ve._per_group_routing_enabled())               # True，不 raise
print(ve._fused_delta_enabled())                     # True
```

**只有这一步通过，才允许重启。** 只看文件 md5 / grep 都不足以证明进程会加载新代码。

---

## 验证方法（可复用）

### micro bench（不起引擎，kernel 级对拍）

1. **C1 正确性**：`out[slot] += B_l @ (A_l @ hidden[t])` 与 torch 逐槽参考对拍，
   断言 `max abs diff <= bf16 容差` **且 delta 非零**（防止"什么都没算"也判过）。
2. **C2 垃圾 padding + 正确 bound**：slack 填 `2**30 / -1 / 123456789` → 不崩、结果不变。
3. **C3 复现旧 bug**：填 `poison = real + (bad-real)//2`（落在两界之间）→ 错误 bound
   观察到越界读，正确 bound 拒绝同一份数据。

### 通用教训

- `sorted_token_ids` 这类 **worst-case 缓冲区的 `numel()` 不能当合法索引上界**。
- 自定义 Triton kernel 报错栈可能指向**后一个** kernel（IMA 是异步的）。
- 部署产物要断言**内容**，不是断言**文件存在**。

## 相关

- PR #31 `feat/lora-per-group-routing`（修复 commit `a38969e49b`）
- micro bench：`/tmp/micro_bench_lora_delta.py`（b300-4）
