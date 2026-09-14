# LoRA 更新路径"永不卡死"修复（2026-09-13，SD-1021/1022）

## 1. 症状（训练侧报的）

```
smg registry:  lora-01a09b49-…  state=FAILED
               error="load timed out on launch-pd after 2160s"
现象：adapter 部署永远不完成；期间【推理完全正常】（health 200、4-token 探针 200/0.6s、
      在跑 job 持续出字、token 数递增），只有 LoRA 加载/卸载这条路全灭。
```

## 2. 排查：**不是 LRU**（两个硬判据）

| 判据 | 事实 |
|---|---|
| 槽位是否满 | decode 只挂 1 个 adapter（base + 01a09a15），`--max-loaded-loras 3` ⇒ 不需要驱逐 |
| 是否发生驱逐 | 全日志真正的 eviction **只有 1 次**（grep `evict` 的 182249 次命中是 `pool memory leak … evictable=0` 告警，**误命中**）|

## 3. 根因：LoRA 控制路径上有**无上限等待**

decode 日志全量时间线（决定性证据）：

```
最后一次成功完成 = 17:31:10（load uid=72b9d2e1 → slot=1，8 rank completes）
此后【每一次】LoRA 操作都没有 completion：
  22:10:56 Start load     ← 用一个 fake URL 做探针（第一个卡住的 op）
  22:11:02 / 22:11:36 / 22:11:42 Start load|unload
  22:45:56 / 22:46:36 Start unload        ← 网关 2100s 超时后的清理
  23:07:10 Start load     ← 训练侧真实 adapter，被前面堵死 → 2160s 超时 FAILED
计数失衡：Start load 16 / Start unload 29  vs  completes 176 / 88，且 17:31 之后 0 completion
```

代码位置：`python/sglang/srt/managers/tokenizer_control_mixin.py`

```python
# 序列化 LoRA 更新；注释明确写了"不阻塞推理"
self.lora_update_lock = asyncio.Lock()          # tokenizer_manager.py:516
...
async with self.lora_update_lock:                # ← 取锁【无超时】
    ...
    result = (await self.update_lora_adapter_communicator(obj))[0]   # ← 等后端 ack【无超时】
    ...
await self.lora_registry.wait_for_unload(lora_id)                    # ← 等在飞请求排空【无超时】
```

**一个 op 不返回 ⇒ 它永久持有 `lora_update_lock` ⇒ 之后所有 LoRA op 永久排队**；而推理走别的路径，
所以表现为「推理健康、LoRA 部署永远超时」。**这就是 bug 的本质：等待没有上界。**

（触发者是一次用 fake URL 做的验证探针 —— 但"传错链接"绝不该导致永久卡死，这是引擎侧缺上界。）

## 4. 修复（引擎侧，`tokenizer_control_mixin.py`）

1. **取锁有界**：新增 `_lora_update_guard(op)`，`asyncio.wait_for(lock.acquire(), timeout)`；
   超时即抛错，**不再无限排队**（`async with self._lora_update_guard("…")` 替换原三处 `async with self.lora_update_lock`）。
2. **后端等待有界**：新增 `_lora_wait_bounded(op, awaitable)`，包住
   `update_lora_adapter_communicator(...)`（权重下载 + 张量装载 + 调度器 ack）与
   `lora_registry.wait_for_unload(...)`（等请求排空）——两处都是原来会永久挂住的点。
3. **锁一定释放**：守卫用 `try/finally: lock.release()`，异常/超时路径也释放。
4. **快速失败**：三个 handler 的 `except` 从 `ValueError` 扩到
   `(ValueError, asyncio.TimeoutError, TimeoutError)` → 返回 `success=False` → HTTP **400**
   → 网关立刻判 `load failed`（不再挂到 2160s）。
5. **超时可配**：`SGLANG_LORA_UPDATE_TIMEOUT_SECS`（默认 **900s**，远小于网关 2100s/驱动 2160s）。

## 5. 验证（部署后实测）

| 用例 | 结果 |
|---|---|
| **同一 fake URL 复现**（曾把 decode 卡死 35 分钟） | `202 → LOADING(11s) → FAILED(31s)`，error=`load failed on launch-pd; check weights exist on all cluster` ✅ **31 秒快速失败** |
| 卡死是否复发 | 失败后注册表其它条目（base/L0/L2）不受影响；后续操作照常执行 ✅ |
| 别名 DELETE（上一处修复） | `DELETE …/models/<uuid>` → **200 / 0.05s** ✅ |
| 三端重启后 | prefill 30100 / decode 30200 `health_generate`=200；smg(31000) `/health`=200；端到端 chat 200 ✅ |

## 6. 运维要点（这次事故的直接教训）

- ⛔ **部署/删除 LoRA 的验证探针禁止用远端 URL**：引擎会**真去下载**；
  要造就用**本地不存在的路径**（秒失败，不触发下载重试）。
  同一教训已记在 `docs/agent/lora-management-api.md` §7.3。
- ⛔ **引擎侧任何"等后端/等请求"都必须有超时**：缺上界 = 一次坏输入换永久卡死。
- ✅ PD 重启必须**配对**（prefill+decode）；网关(smg)与引擎一同重启可消除 registry 脏状态。
- 判"卡死"必须带前提（AGENTS.md §8 诊断铁律）：**有在飞请求 + 停滞**才算；空闲线程等待是正常态。

---

## 7. 第二种失败模式（同一晚遇到）：权重 URL 被 TOS 拒 → curl 22 → 400

**症状**：adapter 部署 `FAILED`，error=`load failed on launch-pd; check weights exist on all cluster`；
引擎日志里 rank 仍逐个 `downloading lora → loading completes`（部分 rank 命中本地缓存而成功）。

**判据（3 分钟出结论，不要猜）**：
1. **直接打引擎**拿原始 error（smg 只记 status，不记 body）：
   ```bash
   curl -s -m 600 -X POST http://10.0.0.75:30100/load_lora_adapter -H 'Content-Type: application/json' \
     -d "{\"lora_name\":\"<URL>\",\"lora_path\":\"<URL>\"}"
   # → {"success":false,"error_message":"Command '['curl','-sfL','--retry','8',…]' returned non-zero exit status 22"}
   ```
   `curl exit 22` = `-f` 模式下服务端返回 4xx/5xx。
2. **从 smg 日志取【完整】URL**（不截断）后**绕过我们全部代码**直接 GET：
   ```bash
   grep -m1 "load_lora_adapter request sent" /root/smg_glm53mol.log | grep -oE "model=https://[^ ]+" | sed 's/^model=//' | xargs -I{} curl -s -o /dev/null -w '%{http_code}\n' --max-time 60 -r 0-1023 {}
   ```
   实测得到 **403 + `<Code>AuthorizationQueryParametersError</Code>`**（签名/查询参数非法，不是过期：
   `X-Amz-Date` + `Expires=86400` 仍在有效期内）。

**归因**：smg `resolve_model_path()` 是**原样透传**、引擎 `curl -o <dest> <lora_path>` **不做任何参数加工**
（`grep x-id|GetObject` 两侧皆 0）→ 直接 curl 同样 403 ⇒ **URL 本身被 TOS 拒，非本仓库代码问题**；
训练侧需重新生成 URL（注意 `x-id=GetObject` 这类 SDK 后加、未参与签名的参数会让 TOS 报
`AuthorizationQueryParametersError`）。

**我们侧可改进（待定）**：smg 的 `load_lora_adapter rejected by engine` 日志应带上引擎响应体
（现在只有 `status=400`，导致必须手工直连引擎才能拿到 `curl 22` 这条关键信息）。

---

## 8. 分析工具

`tools/analyze_lora_log.py <engine.log>` —— 扫引擎日志重建每次 LoRA 装载的分段耗时：
`lora_id` 维度聚合 `loading starts / downloading / loading completes`，输出
每次 load 的总时长、每 rank 的「下载→完成」耗时、相邻 rank 完成间隔（>60s 标 ⚠）。
用途：3 秒内回答"装载本身慢不慢"（`starts=16 / completes=16` = 同一 lora_id 装了两批，
两批间隔数十分钟 = **外部重复触发**，不是单次装载慢）；冷启动 `3.7min`（8 rank×28s）
是正常量级。

本次排查用法：`python3 tools/analyze_lora_log.py /root/prefill_glm53mol.log`

---

## 9. 第三种根因：LoRA 用量计数器泄漏（2026-09-14 修复）

§3 的"无上限等待"是**放大器**；本节是**独立的第二个触发源**。修了 §3（等待有界）之后，
它的表现变成：**这次 unload 900s 后 400 快速失败，但那个 adapter 永远卸不掉**
（每次重试都一样）——引擎侧槽位/显存一直被占着，而 `/v1/models` 里它已经消失。

### 9.1 机制

- 每个用到 adapter 的请求在 `_resolve_lora_path()` 里 `acquire()` 记一笔账，由 `release()` 销账；
  而这版代码里 `release()` **只有两处**：正常完成（`_handle_batch_output`）与
  `finish_reason=abort` 且 status∈{500,503}。
- `unload_lora_adapter` 的顺序是 `unregister()` → `wait_for_unload()`（等计数归零）→ 下发后端。
  **只要有一笔账没销，计数永远不归零**，unload 就等不到（§4 之后 = 900s 超时 400）。
- 漏账点（都是"丢 state、但等不到调度器回复"的清理路径）：

  | 位置 | 触发 | 为什么必漏 |
  |---|---|---|
  | `_handle_abort_req()` | 调度器回送的 `AbortReq`：等待队列 abort、disagg transfer/prealloc/retracted abort、chunked-prefill abort | 等待队列里的请求**只回这一条 AbortReq**，之后不会再有 batch output ⇒ 没人销账 |
  | `_discard_pending_req_states()` | handler 失败：输入校验失败、400 abort 抛错、客户端断连（type1/type3）抛错 | 只 `pop(rid)`，不销账 |

### 9.2 现场判据（GLM-5.3 MoL PD decode，2026-09-13/14）

```
22:40:24  Start unload Lora adapter ... 01a090dd        ← 之后 17 小时 0 completion
          /v1/models 里已无 01a090dd（= unregister 成功）
          后端 0 条 "LoRA adapter unloading starts"
02:08 起  9 次 load/unload 全部只留下 "Start ..."，后端 0 条 loading/unloading starts
旁证：同期 26 条 `Received output for rid=... but the state was deleted in TokenizerManager.`
      = "状态先被删、最终输出才到"的形态（abort 回显先赢）——正是漏账的形状
```

**一句话判据**：`Start (load|unload) Lora adapter` 与后端每 rank 的
`LoRA adapter (loading|unloading) starts` **不配对**，且 `/v1/models` 与引擎实际占用不一致。

### 9.3 修复

1. `_handle_abort_req()`：删 state 前 `_schedule_lora_release(state)` 销账（一次请求一次）。
2. `_discard_pending_req_states()`：区分两种情形——
   **没送到调度器**（`ReqState.dispatched=False`）→ 直接删 + 销账；
   **已送出**（`dispatched=True`）→ 发 abort 并**保留** state，由调度器的 abort/finish 回执销账
   （否则要么漏账，要么在请求还在用 adapter 时提前销账 → 可能被卸掉）。
3. 诊断：unload 排空超时时打印 `pending_usage`（还差几笔账），把"真的还在用"与"账漏了"分开。

### 9.4 与 §3 的关系

- §3（等待有界）= **兜底**：无论什么原因卡住都只卡 900s、快速失败、锁一定释放；
- §9（销账配对）= **根治触发源**：让 unload 能真正完成，不留"卸不掉的 adapter"。
  两者都需要：只做 §3 → adapter 一直 400；只做 §9 → 未知原因仍可能锁死通道。
