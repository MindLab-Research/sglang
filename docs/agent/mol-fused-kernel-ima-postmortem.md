# MoL 融合 kernel 单机验证：两个启动即崩 bug 的排查记录（2026-09-19）

> 场景：在 b300-4（`43.202.208.136`，8×B300）上用 GLM-5.3 单机 MoL 验证
> PR #27（LoRA mmap fast load）+ PR #31（per-group routing / fused delta）。
> 两个 bug 都让引擎在**启动阶段**就崩，且**报错位置与真正根因无关**。

## 症状

引擎每次都在 CUDA graph capture 阶段崩溃，栈顶指向**base MoE kernel**：

```
File ".../kernels/ops/moe/fused_moe_triton_kernels.py", line 907, in invoke_fused_moe_kernel
    fused_moe_kernel[grid](
  File ".../triton/compiler/compiler.py", line 465, in _init_handles
    self.module, self.function, ... = driver.active.utils.load_binary(...)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  -> Exception: Capture cuda graph failed
```

**误导点**：栈顶是 base kernel，但 base kernel 完全没改。真正的原因是
**前一个 kernel 已经 IMA，污染了 CUDA context**，后续任意 kernel launch
（这里是 `load_binary`）才把错误抛出来。凡是在 capture 期间报 IMA 且栈顶
落在无关 kernel 上的，都要往前找**第一个**执行的自定义 kernel。

## 根因 1：`envs.XXX` 缺 `.get()` → `__bool__` raise

```
File ".../kernels/ops/moe/virtual_experts.py", line 1596, in merged_experts_fused_moe_lora_add
    if _per_group_routing_enabled():
File ".../srt/environ.py", line 109, in __bool__
    raise RuntimeError(
RuntimeError: Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`
```

新写的开关访问函数**返回了 `EnvField` 对象本身**而不是它的值：

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

`sglang.srt.environ.EnvField.__bool__` / `__len__` 会显式 raise，用**运行时
trap**替代静默的真值判断——所以任何 `if envs.FLAG:` 或 `if _flag():`（函数
返回 field）都会立刻炸。这个坑只在**新加的 flag**上暴露：老的 flag 在
b300-glm52 里根本不存在，不会走到这条路径。

顺带把 module 级缓存去掉（每次 `os.getenv` 很便宜，且便于运行时 A/B 切换）。

## 根因 2：`num_valid_tokens` 用了 padded buffer 尺寸

`token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)` 的上界是
**token 索引的合法范围**，必须等于 `topk_ids.numel()`（真实 token×expert 槽位
数）。两个新 wrapper 却传了 `sorted_token_ids.numel()`——这是 moe_align 输出的
**worst-case 尺寸**缓冲区长度，远大于真实值：

```
sorted_token_ids.numel() = 695     <- 传错的（缓冲区长度）
topk_ids.numel()         = 128     <- 正确的（真实槽位数）
num_tokens_post_padded   = 512
```

缓冲区内 `[128, 695)` 这一段是 **未初始化 / freed-memory 复用**留下的垃圾值。

判据（micro bench 实测，`poison = 411 ∈ [128, 695)`）：

| bound | 结果 |
|---|---|
| `695`（错） | `poison=411` 通过 mask → `token_rows = 411 // 8 = 51` > `hidden_states` 的 16 行 → **out-of-range reads happened** |
| `128`（对） | mask 拒绝 → 结果与参考实现一致（max diff 0.022 @ bf16） |

大垃圾值（如 `2**30`）不会触发——因为它同时超出两个上界。**只有落在
`[真实bound, 错误bound)` 区间的小正整数值才会越界**，这也解释了为什么它是
时序/内存状态相关（freed-memory 内容随机），表现为间歇性 IMA。

对比原版 `_invoke_moe_lora_shrink_splitk` 一直是传 `topk_ids.numel()`——新
kernel 抄了 mask 逻辑但漏了对齐这个契约。

## 修复

- `virtual_experts.py`：两个 wrapper（`_invoke_fused_lora_delta`、
  `_invoke_fused_lora_delta_all_groups`）增加显式 `num_valid_tokens` 参数，
  两个调用点传 `topk_ids.numel()`；文档字符串写明该契约与误用后果。
- `environ.py`：开关访问一律 `.get()`，去掉 module 级缓存。

## 验证方法（可复用）

`micro bench`（不需要起引擎，直接 kernel 级对拍）三个 case：

1. **C1 正确性**：`out[slot] += B_l @ (A_l @ hidden[t])` 与 torch 逐槽参考对拍
   → `max abs diff 0.022`（bf16 容差内），并断言 delta 确实非零（防止
   "什么都没算"也判过）。
2. **C2 垃圾 padding + 正确 bound**：把 `sorted_token_ids` 真实长度之后的
   slack 填成 `2**30 / -1 / 123456789` → 三种都不崩、结果不变。
3. **C3 复现旧 bug**：填 `poison = real + (bad-real)//2`（落在两界之间）→
   错误 bound 观察到越界读，正确 bound 拒绝同一份数据。

**教训**：`sorted_token_ids` 这类 worst-case 缓冲区的 `numel()` **不能**当
合法索引上界用；凡是要 mask 索引的地方都要拿真实 token 槽位数。

## 相关

- PR #31 `feat/lora-per-group-routing`
- micro bench 脚本：`/tmp/micro_bench_lora_delta.py`（b300-4）
