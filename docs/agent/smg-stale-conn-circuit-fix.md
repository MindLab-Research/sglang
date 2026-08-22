# smg→engine 死连接复用：SSE 断流与 circuit 熔断根因修复（2026-08-22，`a5c3672361`）

> 三次事故闭环：08-20 15:49 首现 → 08-21 多次 → 08-22 23:03 UTC RL 训练中
> 12+ 流同秒齐断把 decode circuit 打开 81 秒（jobs 503）。

## 根因（单一）

引擎 HTTP 是 uvicorn，`SGLANG_TIMEOUT_KEEP_ALIVE` 默认 **5s**——空闲连接
服务端 5 秒即关。而 smg 所有引擎通信的 reqwest 客户端连接池 idle 超时
**大于 5s**：

| 客户端 | 修复前 idle | 后果 |
|---|---|---|
| 主客户端（流式 /generate、chat） | **50s**（`DEFAULT_POOL_IDLE_TIMEOUT_SECS`） | 死连接被复用 → 流中途断 → `error decoding response body` |
| jobs.rs（训练任务执行器） | **300s**（方向写反，注释还吹"above upstream"） | 比 default 再放大 3 倍 |
| deploy.rs / control_plane 各客户端 | default 90s | 同类风险 |

**RL 触发模式**：64 并发 burst → 批次间空闲 >5s → 下一波并发同时捞出同一批
死连接 → 同秒 12+ 流齐断 → circuit 瞬间 open（81s）→ 所有请求 503。

## 修复（机制性消灭，非调参）

所有引擎通信客户端统一 **`pool_idle_timeout(4s) < uvicorn 5s`**——池淘汰
必然先于服务端关闭，死连接不可能被捞出。6 处：`config/types.rs`（50→4，
主客户端）、`jobs.rs`（300→4）、`deploy.rs` ×2、`control_plane/mod.rs` ×3。

保险（下次引擎自然重启生效）：decode 启动脚本
`export SGLANG_TIMEOUT_KEEP_ALIVE=120`。

## 为什么以前"挂了无法自愈"这次能恢复

- 引擎真挂：circuit open→half_open 试探→永远失败→open↔half_open 抖动，需人工
- 本次连接层瞬断：half_open 试探**重连即成功**→closed，2m21s 自愈
  （decode 进程 18h36m 未重启，全程健在）

## 验证（现场回归）

- 原触发条件复刻（空闲 8s 后连打 3 发）：200/200/200，新日志 SSE 错误 **0**
- jobs e2e：提交→3s completed→ids/logprobs 对齐→delete 全绿
- circuit 全程 closed

## 排查判据（若再出现 SSE 断流）

1. `grep 'error decoding response body' /root/smg_router.log`——同秒多条
   = 本病复发（检查是否有客户端漏改/新增未设 4s）
2. `grep 'Circuit breaker state transition'` 看 open 窗口与恢复路径
3. 引擎是否真挂：进程 uptime + 引擎日志批次输出（活的=连接层问题）
