# TODO — b300-glm52 分支（2026-08-18 16:45 CST 更新）

> 会话遗留待办按优先级排列。已完成大项见文末「已闭环」。

---

## P0 — 生产在烧

### 1. GLM 1P2D（1101/1100/1103）prefill 反复 CUDA 崩溃未解决 ⛔
- **现状**：prefill 每跑 ~20 分钟崩一次（`illegal memory access`，报错点漂移：13:38 DSA attention / 14:14 LoRA 虚拟专家 kernel——CUDA 异步错误，报错点≠出错点）。有人反复重启（13:17/13:50/14:38/14:48），**非我操作**。
- **已排除**：L2_0818001 权重坏（新旧 L2 都崩，shape 全同）、venv 版本错（torch/triton 与稳定机一致）、有人同步新代码（8 文件 md5 三方对比，远端是 8/5 老快照）。
- **主要嫌疑**：H1 流量触发型越界（长 prompt × LoRA chunk 边界切换 16/32/128 × 虚拟专家 E=1024）；H2 旧 router 把请求路由到已变 V4 的 1102/1104 端口造成污染（15:51 有人已把 router 修成正确 1P2D 拓扑）。
- **下一步**：
  1. 先观察 router 修好后还崩不崩（15:51 修复后窗口）；
  2. 再崩则 `CUDA_LAUNCH_BLOCKING=1` 重启 prefill 复现，拿真正出错 kernel；
  3. 绕过 router 直打 prefill 30100，长 prompt × L0/L2 各 LoRA 隔离测试；
  4. 远端 srt/ 与 `git archive 6a00bbde3`（8/5 稳定快照）全量 md5 对比排除散改。
- **前置**：与同时操作该集群的同事对齐（router 15:51、start_pd.sh 14:33 均被他人改动）。
- **注意**：18777 供网依赖此集群恢复；当前真实日志 `prefill.log`（非 `_v4`，`_v4` 是 V4 Pro 1P1D 的）。

### 2. V4 Pro 1P1D 窗口修复（W=16384）观察期 → 固化
- **现状**：已部署 `8ae56004ba`+`1910bb619c`，实测 68 万 token 大请求 20s 完成、判据全干净。**尚未经过高峰流量验证**。
- **下一步**：观察一个高峰窗口（看 Grafana `b300-pd-1p1d-pro` 面板 + `grep PDH-PARK / PDH-ADMIT / PDH-SEND-FAIL / "invariant violated"`），无异常后把 `SGLANG_PD_HIDDEN_RECV_WINDOW=16384` 写死进 `start_v4_*.sh`（当前是命令行 env 传入，重启不带就回旧行为）。
- **硬约束**：W ≥ prefill `--chunked-prefill-size`（16384），否则 mismatch（虽已被捕获不再冻结，但请求会失败）。

---

## P1 — 待办

### 3. case50 压测验证（V4 Pro 1P1D）
- 上次修复部署后遗留的验收项：跑 bench-pd（case50，含 abort 洪峰）验证无永久挂起、无乱码；判据 `invariant violated`=0、`PDH-PARK` 短暂出现后 `PDH-ADMIT`。

### 4. 本地工作区 8 个未提交文件处置
- `dsa_backend.py`、`hisparse.py`、`hisparse_memory_pool.py`、`kv_cache_configurator.py`、`model_runner.py`、`model_runner_kv_cache_mixin.py`、`forward_mla.py`、`.md/remote-lora-download.md`（+115/-16）。
- 是上一会话 GLM HiSparse 探索的半成品，**rsync 整树同步会带上生产**（rsync 用工作区不是 HEAD）。
- 处置：review 后 WIP commit 存档，或 `git checkout --` 丢弃。等用户定。

### 5. 未 push 的 commits
- 本地领先远端：`1910bb619c`（窗口准入）、`8ae56004ba`（transfer_worker 异常捕获）、`df49cbae22`/`fc93d88f86`（文档）+ 更早的 `6eb1c7d1bb`/`29b60a034b`/`1c9e1c3275` 等。
- `git push origin b300-glm52`（等用户确认）。

### 6. 窗口修复 v2：sender 自适应 sub-chunk
- 解除 W ≥ chunk 约束：sender 按 decode 窗口切子块发送，允许更小 W → 更高并发长请求。
- 设计文档 `docs/agent/pd-hidden-window-design.md` §8 已记录议题。非紧急（当前 W=16384/P=65536 够 4 路并发长请求）。

---

## P2 — 清理项

### 7. VictoriaMetrics 里 2P3D namespace 的历史脏数据
- 1102/1104 改 V4 Pro 前，旧 collector 把它们的数据写进了 `b300-pd-1p3d`。collector 已修正，但 VM 历史数据要等 retention 过期或手动 `delete_series`。2P3D 面板短期闪现 pro 数据即此因。

### 8. `.md/remote-lora-download.md` 归档
- 排查 L2 时建的笔记（untracked）。并入 4 一起处置。

---

## 已闭环（本轮会话）

| 事项 | 交付 |
|---|---|
| V4 Pro 1P1D 个别请求永久卡死（结构性修复） | `1c9e1c3275` + 报告 `dspark-pd-stuck-req-postmortem.md` |
| 606s 长请求静默楔死（窗口化准入 + 形式化证明） | `1910bb619c` + 设计文档 `pd-hidden-window-design.md` |
| 16:13 窗口部署后双端冻结（transfer_worker 异常裸杀） | `8ae56004ba`（异常→ret=-1 走既有失败同步） |
| V4 Pro 1P1D 重新部署（最新代码 + 同参数） | 1102/1104 W=16384，E2E + 68 万 token 重放通过 |
| 1P1D 独立 Grafana 面板 + otel router 上报 | uid=`b300-pd-1p1d-pro`，1102 独立 collector，2P3D 污染 target 已摘 |
| 吞吐口径修正（输入/输出吞吐 counter rate 15s 窗口） | 面板 8 个 target 修正 |

## 关键速查
- V4 Pro 1P1D：1102(prefill)/1104(decode)，router 1102:31000（key `sk-glm52-pd`），日志 `*_v4.log`，启动脚本 `/root/start_v4_*.sh`。
- GLM 1P2D：1101(prefill+router+gateway+proxy)/1100/1103，公网 18777，日志 `prefill.log`/`decode.log`。
- 判据 grep：`PDH-ADMIT|PDH-PARK|PDH-SEND-FAIL|invariant violated|PADDED-AR-FAIL`。
