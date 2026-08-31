---
name: dpsk-pro-ops
description: "DeepSeek-V4-Pro PD 集群运维 runbook（1P1D B300 集群 8.213.214.14）。触发条件：/dpsk-pro-ops，或用户提到 V4 Pro 部署、PD 分离、DSpark、decode 卡死、prefill 死锁、KV 传输、bootstrap、router 503、集群重启、case50 压测。"
---

# DeepSeek-V4-Pro PD 集群运维

## 集群拓扑

| 节点 | SSH | 角色 | Python | 端口 |
|---|---|---|---|---|
| B300-1 | `ssh -p 1021 root@8.213.214.14` | prefill + router + gateway | `/root/sglang_venv/bin/python3` | prefill=30100, router=31000 |
| B300-2 | `ssh -p 1022 root@8.213.214.14` | decode (DCP=4 + DSpark) | `/root/v15_patched/bin/python3` | decode=30200 |

- 公网入口：`8.213.214.14:31000`（router），API key `sk-glm52-pd`
- 模型：`deepseek-v4-pro-0813`（FP4 expert + FP8 attention，853GB 官方权重）
- 代码路径：`/root/sglang_venv/.../sglang/srt/`（prefill）、`/root/v15_patched/.../sglang/srt/`（decode）

## 启动脚本

```bash
# prefill (1021)
bash /root/start_v4_prefill.sh

# decode (1022)
bash /root/start_v4_decode.sh

# router (1021)
bash /root/start_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200
```

**必须两端同时启动**（decode 会 reconnect prefill 的 8998 bootstrap 端口，先后启动会刷 reconnect 日志）。

## 规范重启流程

1. **备份日志**：`cp /root/prefill_v4.log /root/prefill_v4.log.bak_$(date +%Y%m%d_%H%M%S)`（两端都备份）
2. **同步代码**：`rsync -az -e "ssh -p <PORT>" python/sglang/srt/<file> root@8.213.214.14:<DEST>/<file>`（逐条 rsync，不要目录+文件混合）
3. **清缓存**：`find <venv>/lib/python3.12/site-packages/sglang -name '__pycache__' -type d | xargs rm -rf`
4. **杀进程**：`pkill -9 -f 'sglang[.]launch_server'; pkill -9 -f 'sglang::scheduler'`（注意 `.` 转义防自杀）
5. **确认清零**：`ps aux | grep -aE 'launch_server|sglang::scheduler' | grep -v grep | wc -l`（必须 0）
6. **同时启动**：两端 `nohup bash /root/start_v4_*.sh > /root/<name>_restart.out 2>&1 &`
7. **等 health 200**：`curl -s -o /dev/null -w '%{http_code}' http://localhost:<port>/health`
8. **重启 router**：`kill -9 $(lsof -ti :31000); bash /root/start_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200`
9. **验证请求**：`curl -s -X POST http://127.0.0.1:31000/v1/chat/completions -H 'Authorization: Bearer sk-glm52-pd' -d '{"model":"deepseek-v4-pro-0813","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'`

## 已知问题与陷阱

### 1. prefill collective 死锁（538ee9d0d1 基线）

**症状**：health 200、零崩溃、请求全超时。py-spy 显示 8 rank 全卡在 `all_reduce`（prefill.py:1061 iteration barrier）或 `all_gather`（dp_attn.py:158）。

**根因**：`538ee9d0d1` 的专用 gloo 组（`pf_barrier_gloo_group`）只改了 `pop_bootstrapped`，漏了其他 5 处 poll collective。但**不要试图修复**——2026-08-17 的 4 次修复尝试（419f592de9/d8b28e71a7/373041b738/7af2c4f8ff）全部引入了更严重的死锁。当前基线（`538ee9d0d1` + `d013ed4541`）是能跑的版本。

**判据**：
- py-spy 并发抓 8 rank（串行抓会产生假象）
- 全停在同一 all_reduce = iteration barrier 死锁
- 1 rank broadcast + 7 rank all_reduce = recv_requests 与 pop_bootstrapped 错位
- 7 rank all_reduce + 1 rank all_gather = barrier 与 get_next 跨后端死锁

**处理**：不要改 prefill collective 代码。如果死锁，重启 prefill。

### 2. 请求随机永久卡死（结构性修复：预留先于 bootstrap + 非阻塞释放）

**症状**：低负载下偶尔一个请求永久不返回，health 200、零崩溃、日志无错误。

**根因链**（三个环节）：
1. decode 端 bootstrap（`receiver.init`）先于 hidden rows 分配 → sender 进入 `WaitingForInput`（prefill 侧无超时）
2. pool 满时 `dspark_pool.alloc()` 返回 `None` → `pop_preallocated` 无限 `continue`（不发 KV indices）→ sender 永久卡死
3. `_release_pd_hidden_rows` 等 ACK **阻塞 300s** 后 `return` 不释放 → rows 泄漏 → pool 永久满（回到 2）；且该调用跑在调度线程 → 整个 decode 冻结 300s

**为什么 cd15c85d68（超时兜底）被替换**：用户要求"根本不可能卡"——超时兜底仍会卡 600s 才 abort，且 300s 阻塞等待冻结调度线程。

**结构性修复（decode 端纯本地，零超时依赖，不触碰 collective/poll）**：
- **预留先于 bootstrap**（`_try_prealloc_pd_hidden_rows`，在 `add()` fast path 和 `_resolve_pending_reqs` 的 `init()` 前调用）：
  用 prefix-free 上界 `min(_rebootstrap_prefill_len, pool.size)` 预留 rows。pool 满 → 请求留在 `pending_reqs` **不 bootstrap** → sender 根本不存在 → 无卡可谈（背压而非死锁）。pop 时按实际 `pd_hidden_window_rows` 裁剪多余部分还回 pool。
- **释放非阻塞化**（`_release_pd_hidden_rows` + `drain_pending_pd_hidden_releases`）：
  `wait_ack_completions(room, timeout_s=0.0)` 非阻塞探测（ACK 是 decode 本地的 CUDA 注入完成事件，正常 µs 级完成）；未完成则 park 到 `_pending_pd_hidden_releases`，每 tick 由 `pop_preallocated` 顶部 drain 重试——**永不放弃（不泄漏）、永不阻塞调度线程、无需任何超时**。
- **alloc-None 分支改 fail-fast**：预留机制下已 bootstrap 的请求必有 reserved rows，`alloc()==None` 只能是不变量违规 → 立即 abort(503) + `kv_receiver.abort()`（通知 prefill 释放 sender），绝不 spin。
- **failed_reqs 统一释放**：pop 循环尾对全部 failed_reqs 释放 hidden rows（幂等），覆盖 6 个 config-abort 分支的预留泄漏。

**不变量证明**：`pd_hidden_len = upper_bound - total_prefix_len ≤ upper_bound` 且 `total_prefix_len ≤ len-1`（match 少最后 1 token）→ `window_rows = min(hidden_len, pool.size) ≤ min(upper, pool.size) = len(reserved)`，切片恒有效。

**验证 grep**：`PD hidden ACK still pending at release; parking`（park 发生，正常少见）、`invariant violated`（不应出现）。

**临时缓解（仍有效）**：router 的 `--request-timeout-secs 3600`（1 小时）会最终超时返回 503。

### 3. DSpark draft CUDA graph（d013ed4541，已开启）

**背景**：`dspark_worker_v2.py:296-309` 原本在 PD 模式下禁用 draft CUDA graph（eager 模式）。`d013ed4541` 重开了它（删掉 PD 禁用，加 env gate `SGLANG_DSPARK_PD_DISABLE_DRAFT_CUDA_GRAPH`）。

**效果**：draft verify graph 捕获成功（热缓存 ~18s），accept rate 0.4-0.5 正常。

**冷启动注意**：首次启动 draft graph 捕获需要 ~9 分钟（JIT 编译 flashinfer fp4 MoE kernel），后续热缓存 ~18s。

### 4. router "进程活着但转发卡死"

**症状**：router health 200，但请求 503/超时。router.log 最后一条 "started processing request" 后无 "finished"。

**处理**：`kill -9 $(lsof -ti :31000); bash /root/start_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200`

### 5. DSpark 不支持 grammar-constrained decoding

**症状**：带 `response_format: json_schema` 或部分 `tools` 的请求返回 400：
```
DFLASH speculative decoding does not support grammar-constrained decoding yet.
```

**根因**：`dflash_utils.py:803-814`，`json_schema`/`regex`/`ebnf`/`structural_tag` 四件套任一触发即拒绝。

**处理**：已知限制，不影响非约束请求。

### 6. decode reconnect 日志

**症状**：prefill 重启后 decode 日志刷 `Attempting to reconnect to 10.0.0.75:8998...`

**处理**：必须一起重启 decode（`pkill` decode + 重新启动），不能只重启 prefill。

## 关键代码位置

| 关注点 | 路径 |
|---|---|
| prefill 事件循环 | `disaggregation/prefill.py:1046` (`event_loop_normal_disagg_prefill`) |
| iteration barrier | `disaggregation/prefill.py:1061` (`all_reduce` on `pf_barrier_gloo_group`) |
| pop_bootstrapped | `disaggregation/prefill.py:638` |
| recv_requests broadcast | `scheduler_components/request_receiver.py:141` (`_broadcast_reqs_across_ranks`) |
| DSpark draft graph | `speculative/dspark_components/dspark_worker_v2.py:296` (`init_cuda_graphs`) |
| decode prealloc 队列 | `disaggregation/decode.py:1295` (`pop_preallocated`) |
| hidden pool alloc | `disaggregation/decode.py:1399` (`dspark_pool.alloc`) |
| sender poll | `disaggregation/mooncake/conn.py:2700` (`MooncakeKVSender.poll`) |
| receiver poll | `disaggregation/mooncake/conn.py:2920` (`MooncakeKVReceiver.poll`) |
| 部署脚本 | `/root/start_v4_prefill.sh`、`/root/start_v4_decode.sh`、`/root/start_pd.sh` |

## 当前部署版本

- **HEAD**：`49b3b0e4b7`（回退到 `d013ed4541` draft graph 开启 + `538ee9d0d1` prefill collective 基线）
- **dspark_worker_v2.py**：draft graph 已开启（PD 模式不再禁用）
- **prefill.py**：`538ee9d0d1` 基线（专用 gloo 组创建 + pop_bootstrapped 用专用组，其他 5 处 poll 用原组）
- **decode.py / conn.py / scheduler.py**：`d013ed4541` 状态（无超时兜底，无自适应窗口）
- **environ.py**：`d013ed4541` 状态（无 `SGLANG_PD_HIDDEN_STREAMING_*`、无 `SGLANG_DISAGGREGATION_RETRACT_TIMEOUT`）

## 不要做的事

1. **不要改 prefill collective 代码**——4 次修复全部引入更严重死锁
2. **不要在 prefill/decode 事件循环里加超时兜底**——会改变 poll 行为导致 collective 错位
3. **不要用 `scp`**——用 `rsync`（scp 在 B300 节点会挂起）
4. **不要 rsync 目录+文件混合**——会平铺（逐条 rsync）
5. **不要忘记清 `__pycache__`**——旧 .pyc 会让进程加载没有最新修复的代码
6. **不要忘记重启 router**——worker 重启后 router 需要重新发现 worker
7. **不要串行抓 py-spy**——会因时间跨度产生"rank 散布"假象，必须并发抓
