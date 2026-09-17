# PD 下「请求内隐式重载 adapter」把 bootstrap 卡死（2026-09-14 结案）

## 1. 症状

训练 job（或任何带 `lora_path` 的 `/generate`）在 PD 集群上**静默挂 10 分钟然后失败**：

```
prefill 日志：Prefill bootstrap failed ... KVTransferError(bootstrap_room=...):
              Request ... timed out after 600.0s in KVPoll.Bootstrapping
smg jobs  ：job status=failed  error="empty SSE response"   （0 token，无其它线索）
decode 日志：该 bootstrap_room / rid 【零记录】（只有一次 "LoRA adapter loading completes"）
```

客户端可见：TTFT 无进展 → 600s 后失败；期间两个引擎都没有任何 batch。

## 2. 机制（逐层证据）

1. **PD 的 KV 索引协商与 radix 无关**：KV 池在 decode，decode 必须先分配槽位并把 `dst_kv_indices` 经
   `send_metadata` 回给 prefill（`disaggregation/decode.py::pop_preallocated` → `send_metadata`），
   prefill 才知道往哪写。`--disable-radix-cache` 只让「命中长度」= 0（全量 KV 都要传），
   **"分配+回索引"任何 PD 请求都躲不掉**。所以 `KVPoll.Bootstrapping` 超时 = **该回索引的一侧没回**。
2. 请求引用的 adapter **当前未加载**时，**两个引擎都在请求路径内**做隐式重载：
   ```
   prefill 16:21:05 POST /generate 200 OK → Reloading evicted adapter → Start load Lora adapter
   decode  16:21:05 同样开始 load
   ```
   decode 在加载期间既不分配 KV 也不回索引 → prefill 干等 → 600s 超时。
3. 加载本身可以很慢（OSS 大文件 / 网络抖动），也可能一直完不成；
   实测 decode `num_queue_reqs=0 / num_used_tokens=512` —— 请求根本没进 decode 调度器。
4. **与 `continue_from` 无关**：普通 job（不带续跑）+ 同一个 `lora_path` 一样卡；
   不带 adapter 的 job 全部正常。`continue_from` 只是**继承 `lora_path`** 而更容易撞上。

## 3. 修复（两处代码，缺一不可）

### A. 引擎：PD 下禁止请求内隐式重载（根因）

`managers/tokenizer_manager.py::_resolve_lora_path` 的隐式重载分支
（`unregistered_loras` 循环里）加：`disaggregation_mode` 非空（prefill/decode）时
**不加载，直接 raise ValueError** → HTTP 400 + 可执行指引（`POST /load_lora_adapter ...`）。

理由：PD 里请求路径内的加载必然发生在**两个引擎各自的请求路径**中，无法保证另一端也已就绪，
只会把 bootstrap 握手拖到超时；显式加载还能保证**双端一致**。

### B. jobs 层：派发前预检所有引擎（防线）

`sgl-model-gateway/src/control_plane/jobs.rs`：任务带 `lora_path` 时，`execute_task` 先逐引擎
`GET /v1/models` 确认 adapter 在场（引擎按 path 注册）；缺则**立即 fail 该任务**并附同一条指引
（`ensure_lora_loaded` / `engine_has_lora` / `fail_task_preflight`）。
把 10 分钟静默等待变成**立刻可执行的错误**。

## 4. 判据

- 修复前：`grep "timed out after 600.0s in KVPoll.Bootstrapping" prefill.log` 有命中，且对应
  job 报 `empty SSE response`。
- 修复后：
  - 带未加载 adapter 的请求**秒级**返回 400，报文含 `POST /load_lora_adapter` 指引；
  - jobs 里该任务 `failed` 且 error 同义（不再出现 `empty SSE response`）；
  - prefill 日志**不再出现** `timed out after 600.0s in KVPoll.Bootstrapping`；
  - adapter 显式加载到双端后，同一请求正常完成。

## 5. 运维要点

- **不要在 PD 下依赖"请求触发自动加载"**：adapter 必须先显式加载到**每个**引擎（prefill + decode），
  再提交引用它的请求。
- 排查这类问题时三条一起看：① prefill 的 `KVPoll.Bootstrapping` 超时；② decode 侧
  `num_queue_reqs` / 是否有该 bootstrap_room 的记录；③ 两引擎 `/v1/models` 是否都列出该 adapter。
