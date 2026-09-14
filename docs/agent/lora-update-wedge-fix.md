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
