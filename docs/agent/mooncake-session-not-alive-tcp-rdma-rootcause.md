# mooncake PD "session not alive" 双根因报告（TCP 背压 + RDMA 无损网络缺失）

> 日期：2026-08-27  
> 集群：2P1D（`.6`/`.7` prefill + `.9` decode），mooncake KV transfer  
> 模型：glm52-coding-venti-fp8，CP=8 prefill + DCP=8 decode + EAGLE

## 1. 症状

`Prefill transfer failed ... Decode instance could be dead, remote mooncake session X is not alive`  
`Decode transfer failed ... Failed to get kvcache from prefill instance, it might be dead`

- **间歇性**：短请求正常，高并发长请求必发
- **rdma 和 mooncake_tcp 两种后端都出现相同症状**
- health 全 200，`.9:15002` 在监听，`.6/.7 → .9` 有 767 条 ESTAB

## 2. 根因一：TCP transport 背压溢出 + "一次失败永久拉黑"放大

### 2.1 触发：mooncake TcpTransport 队列打满

`.6` prefill C++ 层日志（`tcp_transport_lane_impl.h:1330`）：
```
W20260827 10:57:21 TCP lane queue-full rejection count: 1, 2, 4, 8, ..., 8192
```

源码确认（`mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp`）：
- 默认容量极小：`lanes_per_peer=4`、`max_queued_transfers_per_peer=1024`、`max_pending_admissions_per_peer=1024`、`admission_timeout=1000ms`、`slice_size=64KB`
- 队列满 → `enqueuePooledTransfer` 走 `hard_rejection` → `failWorkItem(QUEUE_FULL)` → `TransferStatusEnum::FAILED` 直接返回调用方（不排队等待）

16 个 prefill CP rank × 8 decode session 的广播模式下（`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1`），每 chunk 切成 64KB slice 入队，4 条 lane 吞吐有限 → 63 并发请求瞬间打满 1024+1024 → 传输直接判 FAILED。

### 2.2 放大：sglang "失败 1 次即永久拉黑"

```python
# conn.py:2164
if ret != 0:  # batch_transfer_sync 返回非 0
    self._mark_session_failed_and_sync(...)

# hidden_events.py:543-545
session_failures[session_id] += 1
if session_failures[session_id] >= 1:   # 阈值 = 1！
    failed_sessions.add(session_id)     # 进程生命周期内永久拉黑
```

之后所有请求走 `conn.py:2043` `if req.mooncake_session_id in self.failed_sessions` → 快速报 `not alive`（不重试）→ 同步 Failed 给 decode → `"Failed to get kvcache from prefill"`。

### 2.3 恢复机制默认关闭

`environ.py:497`：`SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)` — probe 自动恢复循环默认关闭，黑名单永不摘除。

### 2.4 修复

**env 级（已部署验证通过）：**
```bash
MC_TCP_LANES_PER_PEER=16                    # 4→16（max=16）
MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384   # 1024→16K（max=65535）
MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=16384
MC_TCP_ADMISSION_TIMEOUT_MS=10000            # 1s→10s
MC_TCP_SLICE_SIZE=262144                     # 64KB→256KB
SGLANG_ENABLE_FAILED_SESSION_PROBE=1         # 开启黑名单自动恢复
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=10
```

**代码级（bind-mount override 已部署）：**  
`hidden_events.py:449` 阈值 `>= 1` → `>= 3`（单次瞬时背压不永久杀 session）。

### 2.5 关键源码位置

| 文件 | 行 | 内容 |
|------|-----|------|
| `mooncake_transfer_engine.py` | 236-265 | `batch_transfer_sync` → `engine.batch_transfer_sync_write` 返回 ret |
| `mooncake/conn.py` | 2164 | `if ret != 0: _mark_session_failed_and_sync` |
| `hidden_events.py` | 533-556 | `mark_session_failed_and_sync`（阈值 1→3 已改） |
| `mooncake/conn.py` | 2043 | `if req.mooncake_session_id in self.failed_sessions` → "not alive" |
| `mooncake/conn.py` | 2641-2676 | `_run_one_probe_pass` / `_failed_session_probe_loop` |
| `mooncake tcp_transport.cpp` | 296-330 | `kDefaultLanesPerPeer=4` / `kDefaultQueuedTransfersPerPeer=1024` |
| `mooncake tcp_transport_lane_impl.h` | 488-600 | `enqueuePooledTransfer` → `hard_rejection` → `failWorkItem(QUEUE_FULL)` |

## 3. 根因二：RDMA 无损网络缺失

### 3.1 为什么 rdma 和 tcp 都报相同错误

mooncake 的 RDMA transport 和 TCP transport 共用 **同一个 Python 层放大器**（`hidden_events.py` 的 `failed_sessions` 黑名单）。RDMA 传输失败一次 → 同样永久拉黑 → 同样的 `not alive` 报错。

### 3.2 RDMA 失败的物理层根因

**交换机/NIC 无 PFC（Priority Flow Control）配置：**
- `mlnx_qos` 未安装 → NIC 侧 PFC/ETS 未配置
- `tc qdisc` 是普通 `fq_codel`（无 DCB 优先级）
- RoCE 流量跑在 best-effort Ethernet 上 → 拥塞时交换机直接丢包

**MTU = 1500（标准，非 jumbo 9000）：**
- 1MB KV chunk → ~700 个 1500B 包 → 交换机缓冲压力大
- jumbo frames 可减至 ~114 个包，6× 减少

**`.9` bond0 退化（3/4 链路）：**
- `eno768np0` NO-CARRIER（DOWN）→ bond0 只剩 3 条 100GbE 链路

### 3.3 硬件计数器实证

`.6` prefill（sender）mlx5_13 累计错误（计数器已静止，历史遗留）：
```
req_cqe_error          = 29791    ← CQE 错误（包丢失/损坏）
req_cqe_flush_error    = 26623    ← QP error 后 flush
local_ack_timeout_err  = 19007    ← ACK 超时
roce_adp_retrans       = 20668    ← 自适应重传
roce_slow_restart_cnps = 39675    ← DCQCN 拥塞通知（被要求减速）
```
健康 RoCE 这些应该全是 0。

`.9` decode（receiver）dmesg：
```
[Aug 26 21:18:10] mlx5_13/1: QP 4938 error: local protection error (0x3a 0x0 0x93)
[Aug 26 21:18:10] mlx5_13/1: QP 4942 error: local protection error (0x3a 0x0 0x93)
```
QP memory window 失效 → 连接 reset。

`.6` mlx5_13 错误占 4 张 HCA 总量的 68%（29791 / 43779），说明流量偏向 mlx5_13（符合 `--disaggregation-ib-device mlx5_13` + 可能的 `MC_MLX5_QP_UDP_SPORTS=60001` 单源端口钉死单链路）。

### 3.4 流量集中疑点

若 `MC_MLX5_QP_UDP_SPORTS=60001`（单值）+ `MC_MLX5_QP_LAG_PORT_BALANCE=0`：
- `rdma_endpoint.cpp:1284`：`sport = sports[qp->qp_num % 1] = 60001`（所有 QP 同一源端口）
- bond0 `layer3+4` 哈希 → 固定源端口 → 固定哈希 → 全部流量钉死在一条物理链路
- 4×100GbE bond 降为 1×100GbE

> ⚠️ 无法 100% 确认此 env 当时是否应用（旧容器已删，bash history 无记录），但 `.6` 错误分布（mlx5_13 占 68%）支持流量集中假说。

### 3.5 RDMA 修复建议（如果要切回）

1. **交换机配置 PFC + ETS + ECN**（RoCE 无损网络前提）
2. **MTU 改 9000**（bond0 所有成员口 + 交换机端口）
3. **修复 `.9` eno768np0 DOWN**（物理链路，需现场检查）
4. `MC_MLX5_QP_LAG_PORT_BALANCE=1`
5. `MC_MLX5_QP_UDP_SPORTS=60001,60002,60003,60004`
6. 保留 Python 修复（probe 恢复 + 阈值 3）

### 3.6 RDMA 源码关键位置

| 文件 | 行 | 内容 |
|------|-----|------|
| `config.cpp` | 298-305 | `MC_RETRY_CNT`（默认 9）|
| `config.cpp` | 409-416 | `MC_SLICE_TIMEOUT`（默认 -1 = 不超时）|
| `rdma_endpoint.cpp` | 1240-1295 | `MC_MLX5_QP_LAG_PORT_BALANCE` + `MC_MLX5_QP_UDP_SPORTS` 实现 |
| `config.h` | 65 | `retry_cnt = 9`（重试耗尽 → FAILED）|
| `config.h` | 85 | `slice_timeout = -1`（默认不超时）|

## 4. 为什么 TCP 方案是正确的

| 维度 | TCP | RDMA |
|------|-----|------|
| 拥塞控制 | TCP Cubic/BBR（自带）| 需 PFC+ECN（缺失）|
| 丢包恢复 | TCP 重传（自动）| RDMA CQE error → 传输失败 |
| 背压 | TCP 窗口 + mooncake queue（可调）| 硬件丢包 → CQE error |
| 配置复杂度 | env 可调（`MC_TCP_*`）| 需交换机+NIC 联配 |
| 当前集群状态 | ✅ 已修复验证 | ❌ 无 PFC/MTU1500/bond退化 |

TCP 自带拥塞控制（Cubic/BBR），不需要无损网络就能可靠工作。RDMA 在没有 PFC 的网络上 = "丢包以太网上的 UDP" → 任何高负载必然丢包。

## 5. 部署变更记录

### env 变更（`fix_run2.py` ENV 段）
```python
"MC_TCP_LANES_PER_PEER=16",
"MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384",
"MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=16384",
"MC_TCP_ADMISSION_TIMEOUT_MS=10000",
"MC_TCP_SLICE_SIZE=262144",
"SGLANG_ENABLE_FAILED_SESSION_PROBE=1",
"SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=10",
```

### 代码变更（bind-mount override）
`hidden_events.py:449`：`>= 1` → `>= 3`（三节点 `/root/sglang-overrides/hidden_events.py`）

### 验证
- 3 引擎 health 200 ✓
- probe loop 启动 ✓（`Starting failed-session probe loop (interval=10.0s)`）
- TCP 16 lanes 生效 ✓（`MC_FORCE_TCP is set, using TCP transport only` + `MC_TCP_LANES_PER_PEER=16`）
- 用户确认压测通过 ✓

## 6. 判据工具

```bash
# TCP queue-full 检查
docker logs cv-prefill 2>&1 | grep "queue-full rejection"

# session 黑名单检查
docker logs cv-prefill 2>&1 | grep "Session.*failed"

# probe 恢复检查
docker logs cv-prefill 2>&1 | grep "recovered via probe"

# RDMA 硬件错误检查
cat /sys/class/infiniband/mlx5_13/ports/1/hw_counters/req_cqe_error
cat /sys/class/infiniband/mlx5_13/ports/1/hw_counters/local_ack_timeout_err

# QP error 检查
dmesg -T | grep "QP.*error"
```
