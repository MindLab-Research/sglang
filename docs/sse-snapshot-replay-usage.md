# smg SSE 断点接续（Snapshot + Replay）使用指南

> 面向调用方（RL 训练 / 长流客户端）。内部设计文档见 `docs/agent/sse-snapshot-replay.md`。
> 实现：sgl-model-gateway commit `f86a4e45e9`（协议）+ `926d30ff1f`（首请求响应头补齐）。

## 1. 解决什么问题

SSE 长流（RL 训练 30min+ / 100K+ token 输出）中途断线（网络抖动、客户端重启、代理超时），
此前只能整请求重发——重新 prefill、重新计费、输出前功尽弃。

开启后：**客户端断开，引擎继续生成（不中断）**；客户端用同一个 key 重连，
**逐字节重放缓存 + 无缝续收 live 尾流**，与一次完整连接输出 100% 一致。

## 2. 快速开始（curl）

```bash
KEY=$(uuidgen)

# ① 首次请求：带 X-Sse-Snapshot-Key
curl -N http://<smg>/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-Sse-Snapshot-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[...],"stream":true,"max_tokens":10000}' &
# 响应头会带: X-Sse-Snapshot: first   ← 快照已生效

# ② 模拟断线：kill 掉 curl（或客户端崩溃/断网）

# ③ 重连：同一个 KEY 重发同样请求
curl -N http://<smg>/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-Sse-Snapshot-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[...],"stream":true,"max_tokens":10000}'
# 响应头: X-Sse-Snapshot: replay+live（原流未结束时）
#          或 X-Sse-Snapshot: replay   （原流已结束，纯重放）
# 输出 = 断开前的全部内容 + 后续新内容，与从未断线完全一致
```

## 3. 协议

```
请求头:  X-Sse-Snapshot-Key: <client-generated-uuid>   # HTTP 头不区分大小写
响应头:  X-Sse-Snapshot: first | replay | replay+live
```

| 场景 | 行为 | 响应头值 |
|---|---|---|
| 新 key 首请求 | 正常转发 + 边转发边快照 | `first` |
| 同 key 重连，原流仍在生成 | burst 重放缓存 → 无缝接 live 尾流 | `replay+live` |
| 同 key 重连，原流已结束 | 全量重放到流尾（含 `[DONE]`） | `replay` |
| 不带 key | 完全旁路，行为与旧版一致 | （无此头） |
| 客户端持连到流正常结束 | 快照立即删除（下同 key 重发 = 新请求） | — |
| 客户端断开 | 快照保留 + 引擎继续生成到 `[DONE]`（detached drain，不 cancel 上游） | — |

**重放是 byte-level 的**：原始 SSE 字节逐字节回放（含 usage / finish_reason / `[DONE]`），
不解析不重算，与首连接输出完全一致。

## 4. 关键语义（务必了解）

1. **断开 ≠ 取消生成**：客户端断开后 smg 不 abort 上游，引擎继续生成到流尾。
   断开期间的内容都进快照，重连全量拿回——**零丢失**。
2. **重放不重复计费/不重复推理**：registry 命中直接重放，不 dispatch 到引擎、
   不占 token bucket。
3. **快照删除时机**（默认策略，防泄漏）：
   - 客户端持连正常结束 → **立即删**（已拿全量，无需重连）
   - 重放流正常结束 → **立即删**（重连只保证一次完整重放；**第二次重连同 key = 新请求**）
   - 断开后无人重连 → TTL 1800s 兜底删
   - 快照总数上限 4096（LRU 驱逐最旧的 detached）
4. **同 key 并发**：重放中再断再连，可再次重放（chunks 幂等）；多读者并发读同一快照安全。
5. **单流大小不限**（快照字节数无上限，决策见设计文档 §0）。

## 5. Python 断线重连完整示例（httpx）

```python
import uuid, httpx

URL = "http://<smg>/v1/chat/completions"
HEADERS = {
    "Authorization": "Bearer <API_KEY>",
    "Content-Type": "application/json",
    "X-Sse-Snapshot-Key": str(uuid.uuid4()),   # 本次生成会话的唯一 key
}
PAYLOAD = {"model": "...", "messages": [...],
           "stream": True, "max_tokens": 10000}

def stream_with_resume(max_retries: int = 10):
    out = bytearray()
    for attempt in range(max_retries):
        try:
            with httpx.stream("POST", URL, headers=HEADERS, json=PAYLOAD,
                              timeout=httpx.Timeout(connect=10, read=None)) as r:
                snap_mode = r.headers.get("x-sse-snapshot", "")
                print(f"[attempt {attempt}] mode={snap_mode}")  # first / replay(+live)
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        out += line.encode()
                    if line == "data: [DONE]":
                        return bytes(out)          # 完整输出
            return bytes(out)                      # 流尾即成功（非 DONE 结束的合规流）
        except (httpx.ReadError, httpx.ReadTimeout, httpx.ConnectionError) as e:
            print(f"[attempt {attempt}] dropped ({e!r}), reconnecting with same key...")
            continue                                # 同 key 自动重连 → replay(+live)
    raise RuntimeError("exhausted retries")

full = stream_with_resume()
```

要点：**key 在整个生成会话内不变**（首请求生成，重连复用）；`read=None` 关闭客户端
读超时（长流靠 TCP keepalive + 本协议容断）；重连后无需去重——replay 从头给全量，
客户端可整体覆盖本地缓冲（示例的 `out` 每次重连从空开始重新收集）。

## 6. smg 侧配置

| 启动参数 | 默认 | 说明 |
|---|---|---|
| `--sse-snapshot-enabled` | `true` | 总开关（关闭则该 header 完全旁路） |
| `--sse-snapshot-max-sessions` | `4096` | 快照条目上限（LRU 驱逐最旧 detached） |
| `--sse-snapshot-ttl-secs` | `1800` | 断开快照无重连兜底删除（秒） |

## 7. 注意事项与边界

- **key 唯一性由客户端负责**：不同请求必须不同 key（uuid4）。同 key 复用 = 请求重放语义。
- **重放只保证一次**：重放流正常结束即删。若重放过程又断，可再连（同 key 第三次重放，
  幂等）；重放完成后同 key 再发 = 普通新请求（`first`）。
- **单实例假设**：快照在 smg 进程内存。若前端有 LB 轮询多副本 smg，**重连必须落回同一实例**
  （sticky session 或单入口）；当前部署（1102 smg:31000 单点 / 18888 单点）天然满足。
- **仅覆盖 PD 路由**（`/v1/chat/completions` 走 pd_router 的部署模式）。openai 直连路由
  的快照支持留 v1.1（生产 smg 全走 PD 模式已覆盖）。
- 需要 `stream: true`（SSE）请求；非流式请求不受影响也不生效。
- 与 smg 总请求超时（`--request-timeout-secs`）独立：超时是服务端主动断流的另一维度，
  超长流请确认该值足够（RL 场景建议 ≥ 生成时长上限）。

## 8. 部署状态

| 入口 | 状态 |
|---|---|
| 1102 smg:31000（B300 测试对，api `sk-glm52-pd`） | **已部署**（commit `926d30ff1f`） |
| 其它 smg 实例 | 以各自部署版本为准（协议实现在 `f86a4e45e9`+`926d30ff1f` 之后即含） |

验证方法：任意 `stream: true` 请求带 `X-Sse-Snapshot-Key` → 响应头出现
`X-Sse-Snapshot: first` 即生效；掐断重连同 key → `replay` / `replay+live`。

## 9. 参考

- 内部设计（数据流/生命周期/内存分析）：`docs/agent/sse-snapshot-replay.md`
- 实现：`sgl-model-gateway/src/routers/snapshot.rs`（协议常量
  `x-sse-snapshot-key` / `x-sse-snapshot`）、`routers/http/pd_router.rs`（tee 挂钩）
