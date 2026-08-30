# smg SSE 快照+重放协议 — 设计与实现（2026-08-30）

> 目标：客户端 SSE 请求带特殊 header（含 key）→ smg 对响应流做快照；客户端意外断开后，
> 用同 header+key 重连 → 重放缓存内容 + 无缝续传。RL 训练长 SSE（30min+、100K token）断线容错。
> smg 代码：`sgl-model-gateway/`（Rust）。实现 commit 见 git log `sse-snapshot`。

## 0. 用户决策（2026-08-30 确认）

| 决策点 | 取值 |
|---|---|
| 快照 max bytes | **默认无限**（不设单流上限） |
| SSE 正常结束 | **快照立即删除**（client 收完整流 → 删；只有"断开"的快照保留供重连） |
| sse-snapshot-enabled | **默认开**（无需灰度开关灰度） |
| 断开续读 token 计费 | TokenGuard 在 detached drain 完成时归还（与现行为对齐：流终归还） |
| 重放速率 | burst（快速 catch-up，背压由 client TCP 自适应） |

## 1. 协议（客户端侧）

```
POST /v1/chat/completions          （任何 SSE 端点，stream=true）
X-Sse-Snapshot-Key: <client-generated-uuid>   # 客户端生成；重连用同 key

响应头: X-Sse-Snapshot: first | replay | replay+live
```

| 场景 | 行为 |
|---|---|
| 带新 key | 正常转发 + tee 快照（`first`） |
| 同 key，快照存在且 live（upstream 在流） | burst 重放缓存 → tail 新 chunk → 结束（`replay+live`） |
| 同 key，快照存在且 done | 全量重放至流尾（`replay`），重放完正常结束 → **快照删除** |
| 不带 key | 完全旁路，现行为不变 |
| 流正常结束（client 持连到尾） | **快照删除**（client 已拿全量） |
| client 断开 | 快照保留 + detached drain（upstream 继续读到尾） |

重放 **byte-level**（原始 SSE bytes 逐字节回放，含 usage/finish/[DONE]）——不解析不重算，100% 一致。

## 2. 核心组件

### 2.1 SseSnapshot（`routers/snapshot.rs`）

```rust
struct SseSnapshot {
    chunks: Vec<Bytes>,              // append-only 原始 SSE bytes
    len: u64,                        // 总字节数（统计用）
    done: AtomicBool,                // [DONE] sentinel 或 upstream 流尾
    detached: AtomicBool,            // client 已断开（upstream 由后台 task 继续读）
    live: broadcast::Sender<Bytes>,   // detached 期间的新 chunk（重连 tail 用）
    created: Instant, last_access: Instant,
}
struct SnapshotRegistry { map: DashMap<String, Arc<SseSnapshot>> }
```

### 2.2 生命周期（三态）

```
Live ──client 持连正常结束──► 删除（registry.remove）
  │
  └─client 断开（tx.closed）──► Detached（upstream 后台 drain 到尾）
                                 ├─ 重连 → replay（+tail）→ 正常结束 → 删除
                                 └─ 永不重连 → TTL 兜底删除（防泄漏）
```

- **正常结束删除**：转发 task 读到 `[DONE]`/流尾且 client 未断 → `registry.remove(key)`
- **Detached**：`tx.closed()` → 不 break，继续 `res.next()` → append + `live.send()`（无接收者惰性丢弃）→ 到尾后 done=true（快照保留，等重连）
- **重放**：registry hit → 构造重放 stream（`chunks[0..len]` burst + `!done` 则 `live.subscribe()` tail）→ 重放流正常结束 → 删除
- **TTL 兜底**：detached done 后 N 分钟无重连 → 删除（默认 30min，防 key 泄漏无限积累；正常结束/重放完即时删，不依赖 TTL）

### 2.3 配置

| 项 | 默认 | 说明 |
|---|---|---|
| `--sse-snapshot-enabled` | **true** | 总开关 |
| `--sse-snapshot-max-sessions` | 4096 | registry 条目上限（LRU 驱逐最旧 detached） |
| `--sse-snapshot-ttl-secs` | 1800 | detached 快照无重连兜底删除 |
| ~~max-bytes~~ | **无** | 单流大小不限（用户决策） |

## 3. 数据流改造（挂钩点）

### 3.1 PD router（`routers/http/pd_router.rs` ~1000）

现状（已有基础设施）：后台 task + `tx/rx` channel 转发 upstream `bytes_stream`；
`tx.closed()` = client 断开检测（现 break 取消 upstream）；`memmem::find(&chunk, b"data: [DONE]")` sentinel 检测已有。

```
挂钩 1（handler 入口）: 提取 X-Sse-Snapshot-Key → registry 查询
  hit  → 不转发 upstream，构造重放 Response（X-Sse-Snapshot: replay[+live]）
  miss → 正常转发 + 快照 tee
挂钩 2（转发 task 内）:
  chunk 出现 → snap.append(chunk.clone()) + tx.send(chunk)   // tee
  [DONE]/流尾 → snap.finish() + if !detached { registry.remove(key) }  // 正常结束即删
  tx.closed() → snap.set_detached() + 继续循环 drain upstream   // 不再 break
```

### 3.2 OpenAI router（`routers/openai/router.rs` 637-650）

现状是 drop 链（`Body::from_stream(BreakerTrackedStream)`——client 断开 → upstream 直接 abort）。
改造成同 pd_router 的 **task+channel 模式**（`spawn_snapshot_stream()` 抽到 `streaming_utils.rs` 两路复用）：
- tx.send(client) + snap.append(tee)
- tx.closed() → detached drain（不 abort upstream）
- [DONE] 检测 → finish + 正常结束删除

### 3.3 重放流构造（`routers/snapshot.rs`）

```
fn replay_stream(snap: Arc<SseSnapshot>, registry, key) -> impl Stream<Item=Result<Bytes, Error>> {
    // 1. snapshot chunks 前缀（读锁 clone 引用，burst yield）
    // 2. !done → live.subscribe() tail（新 chunk 广播）
    // 3. done 且前缀发完 → 流尾（+ registry.remove(key)——重放完成即删）
}
```

TokenGuardBody 不 wrap 重放响应（不占 token bucket——重放不耗上游算力）。

## 4. 边界

- **同 key 并发**：broadcast 多读者支持；重放中的二次断开 → 再连再重放（chunks 幂等）
- **上游 error**：error chunk 进快照 + finish；client 在连正常结束 → 删；断开 → 保留可重放（重放看到 error 段后结束）
- **重放期间上游完成**：tail 部分自然收到 finish 信号（broadcast close）→ 流尾
- **多副本 smg**：快照在进程内存，单实例假设（当前部署满足：1102 smg 单点 / 18888 单点）；v2 需 registry 外置
- **正常结束删除时序**：remove 在流尾 chunk 发送成功后（tx.send Ok 且非 closed）——client 已收到全部字节
- **engine 侧**：smg 不 abort upstream → sglang 正常生成到 `[DONE]`（无 cancel 波动）

## 5. 内存

正常流：快照随流删除（零积累）。仅断开流驻留：RL 100K token ≈ 5-10MB/session；
4096 sessions 上限 + 30min TTL 兜底。`Vec<Bytes>` 共享 engine chunk 引用（append 时 clone Bytes 句柄，零拷贝）。

## 6. 测试

1. 单元：SseSnapshot 三态转换/tee/detach-drain/正常删除/TTL；replay_stream byte 一致性
2. 集成：mock upstream SSE generator + client drop at N + reconnect 同 key → 重放输出 == 完整输出（byte diff）；正常结束 → registry 空
3. 压测：RL 16 并发长 SSE 随机断连（复现 2026-08-22 场景）→ 全部重放成功 + TokenGuard 无泄漏 + circuit breaker 记录正确
4. 回归：不带 header 的流完全旁路（现路径零改动验证）

## 7. 实现顺序

1. `routers/snapshot.rs`：SseSnapshot + SnapshotRegistry + replay stream + 单测
2. `streaming_utils.rs`：`spawn_snapshot_stream()` 公共 task 模式
3. pd_router 挂钩（入口查询 + task tee/detach）
4. openai router channel 化改造 + 挂钩
5. AppContext 注入 + 配置（enabled 默认 true）
6. cargo build + test + 集成验证
