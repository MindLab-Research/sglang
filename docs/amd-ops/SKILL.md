---
name: amd-ops
description: "AMD MI300X PD (Prefill-Decode) 集群运维 runbook — Azure VMSS `1p3d-rdma-01` (RG-MI300X, Spot 实例), SGLang v0.5.14 + AITER, GLM-5.2/Macaron-V1-Coding-Venti FP8. Use when: 报告集群状态、重启/部署 prefill/decode/router、处理容器崩溃/退化/乱码/驱逐、清理 cache/config、节点 re-init (deallocate/reimage/驱逐后)、排查 SIGABRT/断言崩溃、otel 指标缺失、任何涉及本集群的运维操作。Covers 2p2d (node-0/5 prefill + node-1/4 decode + router), 镜像 v17-patched / ropefix6 / cacheaware-fix, 一键启动 bootstrap, Spot 驱逐恢复, 流式 usage。"
---

# AMD MI300X PD Cluster Ops

Runbook for the Azure MI300X PD (Prefill-Decode) SGLang cluster. **Status checks
are read-only/safe; stopping/starting containers or re-deploying touches
production — confirm with the user first.**

## Topology (2026-08-22, 2p2d)

| 节点 | 公网 IP | 内网 IP | 角色 | 端口 |
|------|---------|---------|------|------|
| node-0 | <NODE0_PUBLIC_IP> | <NODE0_IP> | **prefill** | 30100 |
| node-5 | <NODE5_PUBLIC_IP> | <NODE5_IP> | **prefill** (2p2d 冗余) | 30100 |
| node-1 | <NODE1_PUBLIC_IP> | <NODE1_IP> | **decode + router** | 30200 / 30000 |
| node-4 | <NODE4_PUBLIC_IP> | <NODE4_IP> | **decode** | 30200 |

- VMSS `1p3d-rdma-01`, RG `RG-MI300X`, 全部 `Standard_ND96isr_MI300X_v5` (8×MI300X 192GB)
- **Spot 实例** (priority=Spot, evictionPolicy=Deallocate, maxPrice=-1) — **会被 Azure 随时驱逐**
- 模型: `glm52-step1369-fp8` (Macaron-V1-Coding-Venti FP8, 756GB) @ `/nvme/models/coding-venti-fp8`
- SSH: `vmadmin@<公网IP>`, key `~/.ssh/id_ed25519` (已注入 VMSS 模型, reimage 不失联)
- 客户端: `http://<NODE1_PUBLIC_IP>:30000/v1`, model `macaron-v1-coding-venti`, api-key `<API_KEY>`
- ⚠️ **鉴权 (08-22 实测)**: router 只认 `Authorization: Bearer <API_KEY>`; `api-key:` header 返回 401
- otel: 四节点 collector 抓 localhost:30100/30200, 推送 <OTEL_ENDPOINT>:4317 (ns `mi300x-pd-debug-glm`, x-api-key 见 bootstrap otel 分支)

### 2p2d 由来 (2026-08-21 实验)
- 为消除 prefill 单点, node-5 从 decode 改为第二台 prefill (router `-p` 可多传)
- router 命令: `bootstrap_pd_v17.sh router -p <NODE0_IP> -p <NODE5_IP> -d <NODE1_IP> -d <NODE4_IP>`
- ⚠️ **实验发现 prefill 故障转移不对称**: 停 node-5 (非协调者) → 10/10 成功; 停 node-0 (bootstrap room 协调者) → 0/10 失败!
  - 根因: `follow_bootstrap_room` 策略, node-0 挂掉后 room 失效, router 不路由到 node-5
  - 结论: **2p2d 不能消除 node-0 单点**, 需试 `--prefill-policy round_robin` (未验证)

## SSH 公钥 (2026-08-20 已注入 VMSS 模型)

- VMSS 模型 publicKeys: **2 个** — `ccss@caosongdeMacBook-Air.local` (原) + `cjiwe@outlook.com` (本地运维)
- 注入命令: `az vmss update -g RG-MI300X -n 1p3d-rdma-01 --add virtualMachineProfile.osProfile.linuxConfiguration.ssh.publicKeys keyData="<pubkey>" path="/home/vmadmin/.ssh/authorized_keys"`
- **作用**: reimage/deallocate 后新实例自动带两个公钥, 避免失联
- ⚠️ 历史教训 (08-20): reimage node-4 后 SSH Permission denied — VMSS 只有 ccss key, 本地 key 失联 → 用 run-command 注入 cjiwe key 才恢复; 已把 cjiwe 永久加入 VMSS 模型
- 若某节点 SSH refused (publickey) → 检查该节点 authorized_keys 是否含本地 key; 缺失则:
  `az vmss run-command invoke -g RG-MI300X -n 1p3d-rdma-01 --instance-id <ID> --command-id RunShellScript --scripts "echo '<pubkey>' > /home/vmadmin/.ssh/authorized_keys && chmod 600 /home/vmadmin/.ssh/authorized_keys && chown vmadmin:vmadmin /home/vmadmin/.ssh/authorized_keys"`
- ⚠️ run-command 有 Conflict 限制 (上一次未完成时新请求被拒), 等待 60s+ 重试

## 镜像 (ACR <ACR>.azurecr.io/mindverse/sglang:)

| Tag | 实际用途 (08-21 实测) | 说明 |
|-----|------|------|
| `v0.5.14-cp-layersplit-v17-patched` | **decode** | digest e007901a, 91.1GB |
| `v0.5.14-cp-layersplit-v17-patched-ropefix6` | **prefill 专用** (PREFILL_IMAGE) | 源头 contiguous + MoE config 补全 (digest 47592fe8) |
| `v0.5.14-cp-layersplit-v17-cacheaware-fix` | **router 专用** (ROUTER_IMAGE) | cache_aware 路由修复 (digest 67bdef1f) — 用户铁律 |

**bootstrap 变量逻辑** (08-21):
- `PREFILL_IMAGE` = **ropefix6** (prefill 容器)
- `ROUTER_IMAGE` = cacheaware-fix (router 容器, 用户铁律: router 必须专用此镜像)
- `IMAGE` = `${V17_IMAGE:-v17-patched}` (decode 容器)
- ⚠️ 08-20 早前实测 router 曾跑 v17-patched 属部署偏差, 已修正回 cacheaware-fix (以 skill 为准)

## 一键启动 / 恢复 (零上下文目标)

唯一入口: `/opt/sglang-config/bootstrap_pd_v17.sh` (节点) = `/home/cjw/glm/scripts/bootstrap/bootstrap_pd_v17.sh` (本地源)。幂等。

```bash
# 全量恢复 (deallocate/reimage/驱逐后 NVMe 全空, 每节点跑):
[node-0] bash bootstrap_pd_v17.sh all-prefill
[node-5] bash bootstrap_pd_v17.sh all-prefill
[node-1] bash bootstrap_pd_v17.sh all-decode -p <NODE0_IP> -p <NODE5_IP> -d <NODE1_IP> -d <NODE4_IP>
[node-4] bash bootstrap_pd_v17.sh all-decode -p <NODE0_IP> -p <NODE5_IP> -d <NODE1_IP> -d <NODE4_IP>

# 快速恢复 (仅 restart, NVMe 保留):
bash bootstrap_pd_v17.sh fast-start [prefill|decode|all]

# 单步: init / prefill / decode / router -p <P>... -d <D>... / otel [prefill|decode|auto] / warmup / status / stop / recovery-log
# 动态 worker 管理 (无需重启 router):
#   bash bootstrap_pd_v17.sh register -r <ROUTER_IP> [-t prefill|decode]   # worker 主动注册 (HTTP 202 Accepted = 成功)
#   curl -X DELETE -H "Authorization: Bearer <API_KEY>" http://<ROUTER>:30000/workers/<id>   # 动态移除
# all-prefill/all-decode 支持 -r <ROUTER_IP>: 就绪后自动主动注册 (驱逐恢复自动回挂)
```

**注意**: `-p` 可多传 (多 prefill), `-d` 必须列出全部 decode IP; 2p2d 用双 `-p`。
**⚠️ node-4 误启 router**: all-decode 分支带 `-p/-d` 参数时, **每个 decode 节点都会启动 sglang-router** — node-4 上会出现多余 router 容器 (实测 08-22 复现)。router 应由 node-1 独占; 建议: 在 node-4 手动 `docker stop sglang-router` 或待 bootstrap 加 hostname 判断 (待办)。
**验证**: 各节点 `/health` 200 → router 公网 `curl <NODE1_PUBLIC_IP>:30000/health` → `warmup` 3×200 → Grafana up 指标 (prefill×2 + decode×2 + router×1)。
**耗时**: 从零 ~12-15min (RAID 1min + Docker 30s + 镜像 2-3min + 模型 5min + 服务 3-6min)。

## 流式 usage (2026-08-22 已启用)

- `--stream-response-default-include-usage` 已加入 bootstrap **prefill + decode** 启动参数
- 效果: 流式请求**默认返回 usage chunk** (choices:[], 含 reasoning_tokens), 客户端无需传 stream_options
- ⚠️ **router (launch_router) 无此参数** (仅 launch_server 有) — 勿给 router 加
- 计费: 服务端始终统计 usage (日志/metrics), 该参数只是回传给客户端

## Spot 驱逐 (2026-08-21 实锤, 频繁发生!)

**观测方法**:
```bash
# 驱逐事件 (evictSpotVM):
az monitor activity-log list -g RG-MI300X --max-events 100 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for e in sorted([e for e in d if 'evict' in e.get('operationName',{}).get('value','')], key=lambda x:x.get('eventTimestamp','')):
    print(e.get('eventTimestamp','?'))"
# VM 状态:
az vmss get-instance-view -g RG-MI300X -n 1p3d-rdma-01 --instance-id <ID> --query "statuses[].code" -o tsv
# Agent 状态:
az vmss get-instance-view -g RG-MI300X -n 1p3d-rdma-01 --instance-id <ID> --query "vmAgent.statuses[0].displayStatus" -o tsv
```

**实测记录** (2026-08-20~22): 驱逐频繁且集中 — 08-20 05:18/05:30/05:34 + 23:21/00:34/00:47/01:48 (三节点先后被驱逐) + **08-22 02:08/06:29 (node-0 连续两次)** — **该 region ND96isr_MI300X_v5 Spot 容量严重不足, 驱逐频繁, node-0 (prefill) 是重点目标**。

**⚠️ router 静态 worker 列表不感知驱逐**: node-0 被驱逐后 router 的 `/get_server_info` 仍显示 workers_count=4 (启动时的静态注册), 但请求打到 node-0 会失败 — **判断节点真实状态必须查 VMSS 实例视图 (PowerState) 或节点 /health, 不能只看 router workers_count**。

**驱逐后处理**: 被驱逐节点 `PowerState/deallocated` → `az vmss start` 拉起 → 按"节点 re-init"流程恢复 (all-prefill/all-decode)。**无 Azure 驱逐压力量化指标** (VMSS metrics 只有 CPU/网络/磁盘)。

**可选对策** (未实施): ① Scheduled Events 预警 (IMDS 轮询, 驱逐前 30s 通知) ② cloud-init 自动恢复 ③ 转 on-demand (免驱逐但贵) ④ 换 region。

## 节点 re-init (deallocate/reimage/驱逐后必做)

1. **冷启动最可靠**: `az vmss start` → SSH refused 时, **`az vmss restart` 常无效**, 用 `az vmss deallocate` → `az vmss start` (冷启动, 历史验证有效; agent Not Ready 需等 10-15min)
2. 等 VM Agent Ready (vmAgent.statuses[0].displayStatus = Ready), **SSH 通** (多次 start/restart 后仍 refused → 再 deallocate→start 一次)
3. reimage 后需重传: `scp scripts/bootstrap/bootstrap_pd_v17.sh config/env.sh → /opt/sglang-config/`
4. `bash bootstrap_pd_v17.sh all-prefill` / `all-decode` (模型缺失时 init 自动拉)
5. containerd 缓存损坏 (`crc32 mismatch`/`invalid tar header`) → 清 `/nvme/containerd`:
   ```bash
   sudo systemctl stop docker containerd
   sudo rm -rf /nvme/containerd && sudo mkdir -p /nvme/containerd && sudo chown 1000:1000 /nvme/containerd
   sudo systemctl start containerd docker
   ```
   然后重跑 init。⚠️ 勿用 `pkill -f containerd` (杀 SSH 链), 用 systemctl。⚠️ 必须 `rm -rf /nvme/containerd` 整目录删, `rm -rf /nvme/containerd/*` 可能残留 snapshotter 挂载点。
6. **docker pull 卡死** (新故障 08-22 node-0): 症状 = pull 进程活着但 `/nvme` 长时间 0 增长 + containerd 日志无活动 (死锁), 之后 `du` 命令陷入 D 状态 (IO 卡死) 导致 `systemctl stop docker` 也卡住 → **直接 `az vmss restart`** (保留 NVMe, 重启清掉 D 状态进程), 重启后整目录清 `/nvme/containerd` 再重跑 init。
7. **containerd 二进制 panic** (新故障 08-21) → 见故障 §7

## 持久化实测 (node-5 实验 2026-08-20) — 修正 "restart 必清" 错误记录

| 场景 | NVMe 数据 | 系统盘配置 (fstab/mdadm/daemon.json/symlink/脚本) |
|------|-----------|--------------------------------------------------|
| **restart** (az vmss restart) | ✅ **保留** (自动挂载 md0) | ✅ 保留 |
| **deallocate→start** (Spot 驱逐) | ❌ 清空 (RAID 需重建, 模型重下) | ✅ **保留** |
| **reimage** | ❌ 清空 | ❌ 清空 (需重传脚本) |

- **restart 后**: NVMe 数据在 → 直接 `fast-start` 或重启容器, 无需重下模型
- **deallocate→start 后**: 配置在但数据空 → 需 `all-prefill`/`all-decode` 全量 (RAID+镜像+模型)
- 结论: 只有 deallocate/reimage 才需全量恢复; restart 是快速恢复
- ⚠️ 08-07 "restart 后 NVMe 必清" 记录为误判 (应为 deallocate 场景)

## 已知故障 & 修复 (2026-08 实锤)

### 1. prefill SIGABRT — 根因链: dsa_cp_round_robin_split_data 上游 bug (ropefix5 源头修复, 08-20)
- 症状: prefill Exited(137), scheduler SIGABRT(-6), stack 含 eagle_worker_v2 + dsa_indexer + rotary_embedding + aiter JIT
- **根因 (代码级确认)**: `dsa_cp_round_robin_split_data` (dsa/utils.py:140) 不等分路径
  `return input_[indices]` **缺 .contiguous()** → stride=cp_size=8 非连续 view
- **触发条件**: EAGLE draft 边界 (tokens % 8 != 0) + batch size 1 (M:1 GEMM) → 单元素 rank 得 [1] stride=[8]
- **两崩溃点同根**: ① aiter rope 断言 `stride(1)==1` ② rebuild_cp_kv_cache view-alias
- **修复演进**: ropefix1 .contiguous() ❌ → ropefix2 reshape ⚠️ → ropefix3 clone(memory_format) ✅ → ropefix4 rebuild .clone() ✅ → **ropefix5 源头 contiguous ✅**
- 镜像: `v17-patched-ropefix5` (digest e6db03da) → **ropefix6 基础上再补 MoE config (见 §2b)**
- **验证**: 修复后 `[1] stride=[8]` 场景 0 次 (此前每次崩溃前必现 8 次)

### 2. MoE triton config JSON 是正确 tuned 参数 (勿删!) — 两次事故!
- `moe_runner/triton_utils/configs/triton_3_6_0/*MI300X*.json` (镜像 Jul 10 预编译)
  是 autotune 参数 (BLOCK_SIZE 等), **缺失/删除 → fallback 默认配置 → FP8 数值乱码**
- 与 `/root/.triton`、`/root/.tilelang` (编译缓存, 可清/挂载覆盖) 是两类东西
- ⚠️ 用户曾建议清 `*L20D*` (NVIDIA L20D config, 本集群无此文件) — 勿同理删 MI300X config

### 2b. **MoE config 被固化进镜像** (08-20 第 2 次事故, 退化根因!)
- 症状: 输出 FP8 数值乱码 (`0.0.0,1,0,1,1, or` 数字夹杂), reasoning_content 乱码, finish=length
- 根因: 08-19 误删 configs 后 `docker commit` 把**删掉 config 的状态固化进 ropefix2-5 镜像**
  → prefill (ropefix5) 一直缺 MI300X config → MoE fallback 默认 → 乱码; decode 用原始 v17-patched 正常
- **验证方法**: 对比容器内 `triton_3_6_0` 目录: 原始镜像 17 文件含 1 个 MI300X config (时间戳 Jul 10);
  被固化镜像 16 文件 0 个 MI300X config (时间戳 Aug 19)
- **修复**: 从原始镜像提取缺失 config → 补入新镜像 → **ropefix6** (digest 47592fe8, 17 文件完整)
  `docker cp <config> ropefix6-build:<path>` → docker commit
- **教训**: docker commit 固化容器状态, 构建新镜像前必须确认 configs 完整; 镜像表必须记录 digest 供对比

### 3. 循环退化 (Hi there 无限重复) — 未根因, 勿改配置
- 症状: 长上下文请求输出 `Hi there! How can I help you today? 😊</think>` 无限循环,
  accept rate 崩溃 (0.91→0.02), 或 accept rate 1.00 全接受
- 已验证无效的假设 (勿再试): MoE config 删除 / `--disaggregation-decode-enable-radix-cache`
  去掉 / `SGLANG_ENABLE_UNIFIED_RADIX_TREE=0`
- 当前策略: 保留最初版本配置, 用户决定不排查

### 4. 循环测试方法 (用户定)
- 串行请求 `input="hi"`, `max_tokens=512`, **`finish_reason=length` = 退化**
- 脚本 `/tmp/stream_deg_test.py <n> <concurrency> <max_tokens> <base_url>` (base_url 是第 4 参数!)
- ⚠️ GLM-5.2 是 reasoning 模型: 短 max_tokens 下 content 为空属正常 (token 全花在 reasoning_content);
  判断退化看 reasoning_content 是否数字乱码, 而非 content 是否为空

### 5. otel NODE_LABEL 冲突 (Grafana 只有 1 个 decode up) — 已修 (08-20)
- 根因: otel 分支 decode 角色 NODE_LABEL 默认 node-1 → node-4 标签冲突被去重
- 修复: 脚本已从 hostname 自动推导 (`1p3drdma00000N → node-N`); 手工: `sudo env NODE_LABEL=node-4 bash ... otel decode`
- 验证: `ssh node-X 'sudo cat /opt/sglang-config/otel/otel-collector.yaml | grep node:'`

### 6. decode `/server_info` 端点僵死 (08-18/08-20/08-21 多次实锤, 周期性)
- 症状: warmup 503 + router 日志 "Failed job: type=AddWorker ... Step timeout after 10s" + workers_count < 期望
- 特征: decode /health 200 但 `/server_info` 15-20s 超时; **/health /get_load /metrics 全正常, 只有 server_info 卡**
- **僵死可"转移"**: node-1/node-4 交替僵死 (重启 router 后受害者轮转) — 疑 server_info handler 内部 lock
- 与 router 镜像无关 (cacheaware-fix / v17-patched 均触发); decode 运行 2-7h 后必现
- **处理**: 重启 decode (`bash bootstrap_pd_v17.sh decode`) + 重启 router; 根因待查

### 7. containerd 二进制 panic (新故障 08-21) — apt reinstall 修复
- 症状: `systemctl start containerd` 失败; 手动 `/usr/bin/containerd --version` 也 panic:
  `panic: runtime error: invalid memory address` @ `github.com/davecgh/go-spew/spew.init()` (regexp.MustCompile)
- 根因: `moby-containerd 2.3.3-2` (Azure HPC 镜像, msft-golang 编译) 二进制损坏 — 与 /nvme 缓存无关!
- **修复**: `sudo apt-get install --reinstall -y moby-containerd` (升级到 2.3.4-2) → 服务正常
- **鉴别**: 清缓存无效 (症状不同: 缓存损坏是拉镜像失败, 二进制 panic 是 containerd 起不来 + --version 也崩)

## 运维铁律

- **router 必须用 cacheaware-fix 镜像** (用户铁律, 以 skill 为准); decode 用 v17-patched; prefill 用 ropefix6
- **decode/prefill 重建后必须重启 router** (bootstrap room 重新协调), 否则 "No available decode workers"
- **Spot 驱逐频繁**: 节点 deallocated + SSH 不通 → 先查 evictSpotVM 活动日志确认驱逐, 再冷启动恢复
- **冷启动** (deallocate→start) 比 restart 更可靠修复 sshd/Agent; 失败可重复一次
- **otel 部署**: 脚本自动推导 NODE_LABEL; 手工配必须每节点不同
- 勿并行 docker pull 跨节点 (containerd 锁竞争); 勿在拉镜像时 restart docker
- kill 用 `kill <pid>` 精确 PID, 勿 `pkill -f` (误杀 SSH 链)
- ACR tag 拉取失败 → 按 digest 拉 `sha256:e007901a...`
- node-1 公网 30200 不通是正常 (router 走内网 <NODE1_IP>:30200); 检查健康用 `ssh node-X 'curl localhost:30200/health'`
- **构建新镜像前检查 configs 完整性** (triton_3_6_0 应 17 文件含 MI300X config), 防止 docker commit 固化缺失状态
- **每次部署/恢复后必须更新文档**: `docs/agent/update-loop.md` (本集群硬规则) — 故障→fault-triage.md, 拓扑→topology.md, skill 同步, 高危坑→AGENTS.md GOTCHAS

## 参考

- `/home/cjw/glm/scripts/bootstrap/bootstrap_pd_v17.sh` — 部署脚本 (本地源, 同步到节点 /opt/sglang-config/)
- `/home/cjw/glm/config/env.sh` — ACR 凭据 + 镜像变量
- `/home/cjw/glm/AGENTS.md` — 项目上下文 + GOTCHAS
- `/home/cjw/glm/docs/agent/` — topology.md / recovery.md / fault-triage.md / update-loop.md
- `/home/cjw/glm/experiments/2026-08-20-prefill-crash*` — 崩溃现场与根因分析
- archival memory: "MoE triton config", "prefill SIGABRT 崩溃根因", "EAGLE draft", "otel NODE_LABEL", "Spot 驱逐", "containerd panic"
