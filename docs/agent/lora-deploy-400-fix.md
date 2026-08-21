# LoRA 部署 400 双 Bug 修复 + 训练任务接口（2026-08-21）

> 背景：训练侧通过 smg `POST /v1/control/models` 提交 OSS URL 部署 LoRA，
> 出现"卡 LOADING / engine_count=0 / DRAINING 不消失 / DELETE 挂起 / 加载成功却 400"。
> 本轮共修复 5 层问题并新增训练任务 API。验收：OSS URL 部署 ACTIVE、
> LoRA 请求 200、DELETE 秒级。

## 修复链（按发现顺序）

### 1. smg deploy.rs（Rust，router 侧）
- **load 无超时**：`client()` 3600s + engine 分钟级加载 → 卡 LOADING 永久静默。
  修复：`LOAD_TIMEOUT_SECS=2100` 守护 + load 请求发出/接受日志。
- **DELETE 对非 ACTIVE 态等 drain 1800s**：LOADING/FAILED 态无流量却挂 30 分钟。
  修复：快速路径（QUEUED/LOADING/FAILED 跳过 drain），仍尝试 unload 清理半载。
- **迟到复活**：load 期间被 DELETE 的部署会在 load 完成后 add_child_model 复活。
  修复：load 返回后校验部署仍是本引擎的 LOADING 态，否则回滚 unload。
- **引擎注册键不匹配**：load 用 `lora_name=<name>` 注册，运行时请求按 `lora_path`
  查找 → "never been loaded" 400。修复：引擎侧注册键统一用 path
  （load/unload/rollback/delete 全部传 path），控制面仍用 name 标识。
- **400 无日志**：load/unload 被拒时只打 status。修复：打印引擎响应 body
  （含 success=false 的 error_message，是真实原因）。

### 2. lora_manager.py（Python，engine 侧）`resolve_lora_local_path`
- **URL 拼 cache 目录非法**（400 #1）：smg 改传 `lora_name=URL` 后，
  `target = join(cache_root, name)` 变成 `/root/glm52_local/loras/https://...`。
  修复：URL → `basename(去压缩后缀)-sha1(url)[:8]` 派生合法 cache key。
- **archive 文件名同 bug**（400 #2 直接根因）：`archive = join(cache_root, f"{name}{suffix}")`
  仍是非法 URL 路径 → curl 写文件失败 exit 22 → success=False → 400。
  修复：`f"{key}{suffix}"`（与 target 同 key）。
- **curl 重试弱**：OSS 瞬时 4xx 时 1 秒放弃。修复：`--retry 8 --retry-delay 3
  --retry-all-errors --connect-timeout 10`。
- 锁机制（mkdir lock + `.done` marker，单 rank 下载其余复用）**本来就正确**，
  无需改。

### 3. tokenizer_control_mixin.py + communicator.py（engine 侧）
- **400 无日志**：注入 `[LOAD/UNLOAD-LORA-FAIL]`（tokenizer 聚合层，打
  name/path/lora_id/error_message）与 `[LOAD/UNLOAD-LORA-400]`（http_server 层，
  打 name/path/error_message/loaded_adapters）。
- **字段名陷阱**：`LoRAUpdateOutput` 的字段是 `error_message`，不是 `message`；
  `merge_results` 里 `r.message` 是死代码（`__call__` 直接返回 result_values，
  从不调用 merge）。日志取值必须用 `error_message`。
- **FanOutCommunicator 类型过滤**（防御性加固）：`fan_out=dp_size(1)` +
  `handle_recv` 无类型检查，IPC 流上无关消息可能被误计为结果。注入
  `expected_type=resp_type` 过滤 + 丢弃告警日志。

### 4. 事件循环卡死（decode，独立事故）
12:11 训练侧首次 OSS 部署把 decode 事件循环卡死（pop_transferred all_reduce
悬死），后续所有 LoRA 请求排在死循环后永不处理（表现为"加载慢"假象）。
**处置：重启 decode 恢复**。判据：py-spy 8 rank 全停在同一 collective。

## 训练任务接口（新增，smg `control_plane/jobs.rs`）

- `POST /v1/control/jobs` 提交任务数组（OpenAI completions 格式 + 可选
  `lora_path`，不设=基模；job 级 `{lora_path, requests:[...]}` 可整体默认）。
- 内部转原生 `/generate`（return_logprob + stream），SSE 聚合产出
  `output_ids` / `output_token_logprobs` / `output_logprob_entries`（三元组
  `[logprob, token_id, top_k]`，**累计模式**非增量——按增量解析会堆积错位，
  已有单元测试锁定）。
- 轮询 `/v1/control/jobs/{id}`、下载 `/result`（整体或单 task）、DELETE 清理。
- 并发 64（env `SMG_JOBS_MAX_CONCURRENCY`）、结果落盘 `SMG_JOBS_DIR`
  （重启恢复）、48h 保留。
- API 文档：`docs/agent/training-jobs-api.md`。

## OSS 下载慢根因（非 bug，链路问题）

- `mint-dev.oss-accelerate.aliyuncs.com`（阿里加速端点）：B300 出口 IP 被 GDS
  调度到**新加坡边缘**（161.117.243.27），源站在北京 → 双向跨境，单流
  0.5-2.4MB/s 抖动。给阿里云支持的 request id：`6A880FDD8BE59FAA1EEFED59`
  （x-amz-request-id 头）。
- **北京直连端点快 10-30 倍**（训练侧换 URL host 即可，签名流程不变）：
  - `mint-dev.oss-cn-beijing.aliyuncs.com`：9.6MB/s，418MB≈46s（Expires 24h）
  - `tos-mint-ckpt.tos-cn-beijing.volces.com`（火山 TOS）：14.6MB/s，<30s
    （注意 TOS `X-Tos-Expires=3600` 仅 1h 窗口）
  - 火山 TOS 加速端点 22.2MB/s 也可用
- adapter 包实测 418MB 压缩 / 29GB 解压（全量 target-modules）。
- 签名 URL 的 query 变化会使 sha1 cache-key 失效（同 adapter 重复下载）；
  如训练侧每次新签名，可考虑 key 改用去 query 的 path 部分。

## 验收记录

| 项 | 结果 |
|---|---|
| 本地路径部署 | 17s ACTIVE（19s 优化正常） |
| OSS URL 部署（修复后） | 下载 8m09s（当时链路）+ 解压 35s + 加载 4.5m → ACTIVE |
| LoRA 请求（lora_path=URL） | 200，decode 8 rank loading completes |
| DELETE | 0.24s（原 30 分钟挂起） |
| 引擎日志 | 400 时完整打印 name/path/error_message |
