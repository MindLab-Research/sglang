# PD TTFT 确认链延迟调查(2026-08-28,2P1D GLM-5.2)

## 现象

PD 分离 + decode radix 命中场景,cache-hit TTFT 与**上下文长度成正比**:
- 269K prompt 命中:TTFT ≈ 6.2s(SSE 首 chunk 口径)
- 672K prompt 完整命中:TTFT ≈ 13.5-14s
- 用户观察:上下文越短 TTFT 越短;命中缓存后 TTFT 仍 25s+(多次重复请求)

## 排除项(全部实测铁证)

| 嫌疑 | 判定 | 证据 |
|---|---|---|
| DCP 2D reshard 全量传输 | 无罪 | `[DCP-PD-INIT] decode_prefix_len=671744 to_send=331`(差量 99.95%) |
| decode 端 bootstrap/匹配/预分配慢 | 无罪 | `[BS-T] handshake-done + prealloc-done` 均在 t+2s |
| tokenize | 无罪 | 1218 msgs 实测 0.35s |
| prefill 计算 | 无罪 | 完整命中只算 64 token(`#cached-token: 672064`) |
| 传输带宽 | 无罪 | 57GB 全量 send→arrived 仅 9s;28MB 差量反而 14s |
| prefill 端 radix 断链 | 无罪 | 完整命中 `#cached-token: 671744`(断链疑云后来证实是内容分叉的 prefix-match 正常语义) |

## 真凶:send→kv-arrived 确认链的固定延迟

[BS-T] 探针实测(672K 完整命中):

```
t+2   decode handshake-done + prealloc-done   (decode bootstrap 无罪)
t+4   prefill batch 完成(算 64)→ send_kv_chunk
t+18  decode kv-arrived (hidden_done AND kv_done)
      ↑ 14s 空窗,与数据量无关(28MB 差量 14s vs 57GB 全量 9s)
```

**注意**:[BS-T] kv-arrived 是 4 个探针(见修复节)的最后一个;4 个探针把 bootstrap 链
(入队→握手→预分配→KV 到达)完全切开,前 3 跳都在 t+2,14s 全部压在最后一跳
(prefill send → decode 感知)。

### 确认链的跳(prefill → decode)

```
send_kv_chunk
 → sender.send(入队)
 → transfer worker: batch_transfer_sync(同步等数据到达 decode)
 → maybe_send_extra(states: SWA/DSA/C128/PD_HIDDEN)
 → send_aux(同步)
 → update_status(Success)
 → sync_status_to_decode_endpoint ×N(串行 TCP,已改并行)
 → decode 控制线程 update_status(Success)
 → decode scheduler 下一 tick poll → [BS-T] kv-arrived
```

### 最后的未闭合环(待环境恢复验证)

14s 在以上哪几跳,两个候选:
1. **mooncake batch_transfer_sync 的传输时间本身慢**:默认路径
   `process_layers` 把 78 层合并成**一次** batch(排除逐层调用开销),但 **DCP
   分片 + CP 过滤把连续 6 页按 `pos%8` 打散到 8 个 decode rank** → 块粒度碎片化
   → TCP 小块(每 rank 每层 ~41 token)传输效率崩塌。对照组:Q1 全量 672K 连续
   → 每层 1 个 730MB 大块 → 6.3GB/s。**小块 vs 大块的 mooncake TCP 吞吐差 ~2700×**
   (2.3MB/s vs 6.3GB/s)是最后未定量的一点。
2. **decode scheduler tick 检测延迟**:py-spy 实锤 idle 时 decode MainThread 连续
   快照卡 `fast_barrier`(≥3s)和 `process_batch_result_decode.synchronize`(4s,
   verify 672K 的 GPU 等待)→ tick 周期被 8-rank collective + GPU 等待拉长 →
   Success 状态的感知延迟最多 1 个慢 tick。

## 修复(commit 记录)

| commit | 内容 |
|---|---|
| `ac2defe0c7` | [BS-T] 4 探针进 decode.py(prealloc.add / handshake-done / prealloc-done / kv-arrived)+ hidden renotify 3s→1s(notify 走有损 zmq PUSH,首轮丢失最坏恢复 ~3s→~1s) |
| `951d5aa4bc` | sync_status_to_decode_endpoint 串行 TCP×N → executor 并行(确认链尾跳) |

诊断工具另在容器层(未进 repo,属测试集群镜像层):prefill.py 的 2 处
iteration-barrier→fast_barrier patch、utils.py 的 NCCL fast_barrier + FB-PATH
日志版。

## 环境恢复后的验证 runbook(10 分钟)

1. decode 容器 `docker cp` repo 版 decode.py + `__pycache__` 清理 + restart
   (注意:先配对重启 prefill,否则 mooncake 会话失效全 503)
2. prefill 置 `SGLANG_DEBUG_DIAG=1`(PDH-SEND/PDH-RECV/PDH-ACK-SUBMIT 日志)
3. 发 1 次树热请求,grep `[BS-T]` + `[PDH-`:
   - `PDH-SEND`(prefill)与 `PDH-RECV`(decode)的差 = notify 链延迟
   - `kv-arrived` 与 `PDH-ACK-SUBMIT` 的差 = decode tick 检测延迟
   - `send_kv_chunk`(t+4)与传输完成差 = batch_transfer_sync 时长
4. 按缺口位置修:
   - batch_transfer_sync 慢 → DCP 小块合并(跨层/跨 rank 聚合 transfer_blocks)
   - notify 链慢 → notify 改同步 TCP(弃有损 zmq PUSH)
   - tick 检测慢 → 控制线程 update_status 后直接唤醒 scheduler(事件驱动),
     不等下一 tick 的 fast_barrier/collective 链

## 教训

- "上下文越短 TTFT 越短"在 PD + decode radix 下不是差量传输问题(差量早就生效),
  而是确认链固定开销在小传输下占比放大——**优化方向是砍固定开销(并行化/事件驱动),
  不是减数据量**。
- 完整命中请求的 prefill 端只算 64 token(`#cached-token: 671744`),TTFT 的大头
  全在传输完成确认链——**别被"命中了为什么还慢"迷惑,先切分确认链再谈优化**。
