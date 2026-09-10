# B300 NVLink Fabric 驱动层退化 → PD KV 传输失败（2026-09-06）

> **结论先行**：这不是代码/镜像问题。同事用**原 image + 原参数**部署同样复现。
> 根因是 **B300 (sm_103) 的 NVIDIA 驱动 595.91.07 存在 NVLink multicast fabric 状态机失配**，
> 导致 decode 端 TP8 内部 NCCL collective 间歇性卡死 → 跨节点 KV 传输 ACK 超时 → session failed。
> 处置需在**硬件/驱动层**进行，代码层修改无效。

---

## 1. 故障现象

- prefill 日志：`Decode instance could be dead, remote mooncake session <ip:port> is not alive`
- decode 日志：`Failed to get kvcache from prefill instance, it might be dead` / `... unknown reason from another rank`
- probe 自愈（`89635826e2` un-blacklist）在跑，但每次恢复后**新 session 又立即失败**，
  形成「失败→拉黑→probe 恢复→再失败」死循环。
- decode 端 `batch_transfer_sync` 返回非 0（`session_failures≥1` 即永久拉黑）。

## 2. 排除代码的过程（重要教训）

1. **换端口重启**（bootstrap 30011→30012）→ 无效。**端口不是根因**。
2. **验证 conn.py 版本配套**（本地版 e79f3463 + prefill 835fb1fc 带 page_size）→ 已配套，无效。
3. **恢复 memory_pool.py 到 PR13 基线**（去掉 `_dcp_virtual_to_physical` 修复）→ e2e 仍失败。**DCP 修复不是根因**。
4. **关键排他逻辑**：`tp_size=8` 的 decode，其内部 8 GPU 的 NCCL 通信**必然走 NVLink**。
   我们改的所有代码（mooncake 版本、DCP 修复、memory_pool.py、端口、env）都**不改变这个硬件依赖**。
5. 同事用**原 image + 原参数**复现同一问题 → **彻底排除代码因素**。

> **教训（本分支死锁家族通用）**：`seq_len`/`gate`/`collective` 的 rank-invariant 修复都救不了
> **底层 GPU 互联硬件故障**——这类问题需要先查 dmesg 的 NVLink/NVRM 错误，而不是改代码。

## 3. 机器侧证据（只读取证）

### 3.1 prefill 节点 (3.37.20.114) —— 最重
```
[13:36~15:53] NVRM: nvCheckOkFailedNoLog: Check failed: NVLink fabric state cached by the driver is out of sync
  [NV_ERR_FABRIC_STATE_OUT_OF_SYNC] (0x00000087) @ mem_multicast_fabric.c:3119   ×688
[同时段]     NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0       ×122
```
时间戳（近 3h）：`13:36, 13:51, 13:55, 14:04, 14:08, 14:29, 14:32, 15:00, 15:03, 15:36, 15:40, 15:49, 15:53`
——其中 `14:29~15:53` **精确落在 14:27 服务重启之后的失败窗口**。每次 `9` 个（对应 8 GPU + 1 同步点报）。

### 3.2 decode 节点 (15.164.0.39)
```
[上午时段] NVRM: Xid (PCI:0000:*): 43, pid=python3, channel 0x00000002       ×9
             ← 正是上午调试 decode crash 的时间段（同批进程）
[15:49~15:54] NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0
[3.2天前]  Xid 145 RLW_SRC_TRACK Nonfatal XC1 i0 Link 00-17                 ×872
```
历时累计（`uptime≈3.9天`，机器从未重启，只重启过容器，dmesg 跨多天累计）。

### 3.3 py-spy 实锤 decode 卡在 GPU 组内通信（非网络）
```
decode pid=951 (MainThread active+gil):
    all_reduce (torch/distributed/distributed_c10d.py:3068)
    _padded_all_reduce_min (utils.py:205)
    pop_transferred (decode.py:3242)
    event_loop_overlap_disagg_decode (decode.py:3543)
```
——卡在 **NCCL all_reduce（走 NVLink）**，不是 TCP/socket 层。
之前 tcpdump 已证明 TCP 层健康（0 丢包 0 重传 0 RST），排除网络。

## 4. 因果链（完整闭合）

```
NVLink fabric 状态不同步 (0x87 ×688) + knvlink send fail (×122)
    │   所有 bootstrap/PFX-SYNC 的跨 rank collective 走 NCCL all_reduce（走 NVLink）
    ▼
decode 端 TP8 组内 NCCL all_reduce 卡死 (py-spy: pop_transferred→_padded_all_reduce_min)
    │   decode 无法完成 forward → 无法 ACK KV 传输
    ▼
prefill 端 batch_transfer_sync 等不到 ACK → 超时返回非0 → session_failures≥1 → 永久拉黑
    ▼
probe 自愈(89635826e2)恢复 session → 下次传输又卡 decode 端 NCCL → 再次失败 → 死循环
    ▼
本分支可见症状：「is not alive」/「Failed to get kvcache from prefill」/「Decode instance could be dead」
```

## 5. 排除项（机器表面健康）

| 检查 | decode | prefill | 结论 |
|---|---|---|---|
| GPU HBM ECC (corrected/uncorrected) | 0/0 | 0/0 | 无坏块 |
| GPU remapped rows (correctable/uncorrectable/pending) | 0/0/No | 0/0/No | 无 retire |
| GPU 温度 | 39~48°C | 39~48°C | 正常 |
| GPU 时钟 | 2032 MHz (P0) | 2032 MHz (P0) | 正常 |
| GPU 在册 | 8/8 | 8/8 | 无掉卡 |
| NVLink 每链路速度 | 53.125 GB/s 全速 | 53.125 GB/s 全速 | 链路物理在 |
| IB (ConnectX-7) | State Active/LinkUp, 固件 28.48.1132 | State Active/LinkUp | 正常 |
| 近 5~25s 新 Xid/fabric | 0 | 0 | 当前离线平稳 |

## 6. 环境关键值

- NVIDIA 驱动：**595.91.07**，CUDA 13.2
- VBIOS：97.10.7E.00.03
- MLX5 固件：28.48.1132（ConnectX-7，MT2910）
- dmesg 另有 `mlx5_fw_version_query: fw query isn't supported by the FW` ×174（ConnectX-7 固件与驱动查询交互异常，次要信号）

## 7. 处置路径

| 优先级 | 动作 | 说明 |
|---|---|---|
| 1 | 确认 595.91.07 对 B300(sm_103) 的 NVLink fabric bug → 升级/降级驱动 | 最可能一击命中 |
| 2 | 检查 NVSwitch/NVLink 硬件层与固件 | 0x87 失配或来自 NVSwitch 固件 |
| 3 | 若驱动无效 → 判 GPU 互联硬件/驱动缺陷，隔离换机 | 参照 1100 Xid13 硬件判决案例 |

## 8. 验证判据（驱动/换机修复后）

- prefill dmesg 出现 `FABRIC_STATE_OUT_OF_SYNC`/`knvlinkSendInbandData Failed` 的新条目 = **仍复现**
- 驱动调整后，`nvidia-smi` 下跑一次 e2e（1+1=2），TPOT/TTFT 正常且无 `is not alive` = **修复**
- 压力时段同步 `grep -c FABRIC_STATE_OUT_OF_SYNC` dmesg 增量 = 0
