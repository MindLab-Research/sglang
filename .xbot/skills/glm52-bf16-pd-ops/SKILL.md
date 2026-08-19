---
name: glm52-bf16-pd-ops
description: "GLM-5.2 BF16 + 4 LoRA 1P1D PD 集群运维 runbook（B300 8.213.215.2）。触发条件：/glm52-bf16-pd-ops，或用户提到 GLM-5.2 BF16 部署、EAGLE draft graph、DCP=8、prefill CP=8、LoRA 虚拟专家、KV cache 容量、start_glm52_bf16_pd.sh。"
---

# GLM-5.2 BF16 + 4 LoRA 1P1D PD 集群运维

## 集群拓扑

| 节点 | SSH | 角色 | Python | 端口 |
|---|---|---|---|---|
| B300-1 | `ssh -p 1021 root@8.213.215.2` | prefill (CP=8) + router + gateway | `/root/sglang_venv/bin/python3` | prefill=30100, router=31000 |
| B300-2 | `ssh -p 1022 root@8.213.215.2` | decode (DCP=8 + EAGLE) | `/root/v15_patched/bin/python3` | decode=30200 |

- 公网入口：`8.213.215.2:31000`（router），API key `sk-glm52-pd`
- 模型名：`glm52-bf16-pd`
- 模型路径：`/root/glm52_local/bf16`（GLM-5.2 BF16，78层/256专家/MTP=1，282 shards）
- LoRA：`/root/glm52_local/loras/L0..L3`（4 个，`--lora-use-virtual-experts`，rank 16）
- 部署脚本：`/root/start_glm52_bf16_pd.sh`（两端同 md5）

## 配置速查

| 项 | prefill | decode |
|---|---|---|
| TP | 8 | 8 |
| CP/DCP | 无 CP（2026-08-19 起；CP×LoRA 修复 `c1946917cb` 已进主线，恢复 CP 前必须部署该修复） | DCP=4（`--dcp-size 4`） |
| 投机解码 | 无 | EAGLE（`--speculative-algorithm EAGLE --speculative-draft-model-path /root/glm52_local/bf16 --speculative-num-steps 5 --speculative-eagle-topk 1`） |
| KV dtype | FP8 E4M3（默认，scaling=1.0） | 同 |
| mem-fraction | 0.85 | 0.90 |
| page size | 64 | 128 |
| HiCache | ❌（2026-08-19 移除，排查时排除，未恢复） | ❌ |
| LoRA | 4（L0-L3）+ **VE flag 必须** | 4（L0-L3）+ VE |

**必须两端同时启动**（decode 会 reconnect prefill 的 8998 bootstrap，先后启动会刷 reconnect）。

## 启动命令

```bash
# prefill (1021)
cd /root && nohup bash /root/start_glm52_bf16_pd.sh prefill > /root/prefill_restart.out 2>&1 &

# decode (1022) —— 和 prefill 同时
cd /root && nohup bash /root/start_glm52_bf16_pd.sh decode > /root/decode_restart.out 2>&1 &

# router (1021)
cd /root && bash /root/start_glm52_bf16_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200
```

## 规范重启流程

1. **备份日志**：`cp /root/prefill.log /root/prefill.log.bak_$(date +%Y%m%d_%H%M%S)`（decode 同理，注意是 `prefill.log`/`decode.log` 不是 `prefill_v4.log`）
2. **同步代码**：`rsync -az -e "ssh -p <PORT>" python/sglang/srt/<file> root@8.213.215.2:<DEST>/<file>`（逐条 rsync）
3. **清缓存**：`find <venv>/lib/python3.12/site-packages/sglang -name '__pycache__' -type d | xargs rm -rf`
4. **杀进程**：`pkill -9 -f 'sglang[.]launch_server'; pkill -9 -f 'sglang::scheduler'; pkill -9 -f 'sglang::router'`
5. **确认清零**：`ps aux | grep -aE 'launch_server|sglang::scheduler' | grep -av grep | wc -l`（必须 0）
6. **两端同时启动**（prefill + decode）
7. **等 health**：prefill ~70s、decode ~80s（`curl http://localhost:<port>/health`）
8. **重启 router**：`kill -9 $(lsof -ti :31000); bash /root/start_glm52_bf16_pd.sh router --prefill 10.0.0.75:30100 --decode 10.0.0.67:30200`
9. **验证**：curl 一个 chat 请求（见下）

## 验证请求

```bash
curl -s -X POST http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-glm52-pd' \
  -d '{"model":"glm52-bf16-pd","messages":[{"role":"user","content":"1+1等于几？只回答数字"}],"max_tokens":64,"temperature":0}'
```

## KV cache 容量（实测，CP=8+layersplit 旧形态；当前无 CP 形态数值更大，待重测）

| | prefill | decode |
|---|---|---|
| max_total_num_tokens/rank | ~118K（TP6/7=130K） | ~199K |
| 总 KV 容量 | ~967K tokens | ~1.6M tokens |
| 单 rank 可用显存（给 KV） | ~4.1GB | ~16.3-17.3GB |
| 显存 used/卡 | ~272GB/275GB | ~257GB/275GB |

decode KV 是 prefill 的 1.6 倍，因为 prefill 权重 + CP layer-split 开销把可用显存压到 4GB。

## 已知问题与陷阱

1. **⛔ VE flag 缺失 = 首个 LoRA 请求必崩（2026-08-19 结案）**：prefill 块曾丢
   `--lora-use-virtual-experts` → 走无加固 classic `fused_moe_lora` kernel →
   CUDA IMA。已修复（脚本双块补回，备份 `.bak.nove_*`）。**对照实验前先逐条
   diff 两边启动 flags**；空续行 `\` 是 flag 被删的物证。
2. **BF16 权重 199GB**：CP=8+layersplit 形态需 mem-fraction ≥0.93；当前无 CP
   形态 0.85 正常（AGENTS §5.1.1 规范值）
3. **CP×LoRA 崩溃已根修**（commit `c1946917cb`，2026-08-19）：csgmv batch_info
   与 CP shard 行数错配 → OOB。恢复 CP 部署前确认双端代码 ≥ 该 commit
4. **LoRA 触发语义**：只认请求体 `lora_path` 字段（按 adapter 名匹配）；
   `model:xxx:L0` colon 语法不生效；`model` 字段引擎不校验
5. **LoRA 懒加载**：`--lora-paths` 只注册，首个请求才加载。判断真用了 LoRA：
   `grep -E "Start load Lora|loaded weights" /root/{prefill,decode}.log`；
   `curl :30200/metrics | grep lora_pool_slots_used`（decode 常驻 ≥1）
6. **LoRA 缓存隔离**：radix cache 的 extra_key 拼 lora_id，跨 adapter/base 不
   命中（正确性要求）；同 prompt 多 adapter 各存独立 KV，显存按份数涨
7. **PD 重启成套**：只重启 prefill → decode mooncake 会话过期
   （`KVTransferError: Aborted by AbortReq`）；router 熔断器记死旧 prefill →
   503。顺序：杀干净 → prefill+decode 同时 → router
8. **router/smg control plane**（commit `c1663ec790`）：`GET
   /v1/control/models`（key `sk-control-pd-2026`）返回引擎已注册 LoRA（动态切
   lora 数据源）；POST 传 URL/路径部署（swap 槽位只数 lora 不算 base）；
   `v1/completions` 端点在 smg PD 模式卡死（既有 bug，用 chat/completions）
9. **启动缺函数/import 错误**（已修，commit `c561d36e45`，历史保留）
10. **prefill collective 死锁**（V4 Pro 遗留）：`538ee9d0d1` 基线，不要改 prefill collective 代码
11. **router 转发卡死**：health 200 但请求 503/超时 → kill + 重启 router
12. **EAGLE draft graph**：冷启动捕获 ~9 分钟（热缓存 ~18s），draft model = BF16 自带 MTP
13. **不用 scp**、**rsync 逐条**、**清 __pycache__**、**py-spy 并发抓**（同 V4 Pro）

## 关键代码位置

| 关注点 | 路径 |
|---|---|
| 部署脚本 | `/root/start_glm52_bf16_pd.sh` |
| EAGLE draft graph | `speculative/eagle_worker_v2.py` |
| prefill CP layer-split | `mem_cache/cp_layersplit_pool.py`、`layers/utils/cp_utils.py` |
| DCP | `disaggregation/decode.py`（`--dcp-size 4`） |
| LoRA 虚拟专家 | `lora/backend/base_backend.py`、`lora/lora_manager.py` |
| LoRA csgmv CP shard 修复 | `lora/backend/chunked_backend.py`（`_cp_shard_batch_info_or_none`） |
| smg control plane | `sgl-model-gateway/src/control_plane/{deploy.rs,mod.rs}` |

## 当前部署版本

- 代码 = 本地 git HEAD（2026-08-19 起）：含 `c1946917cb`（CP×LoRA 根修）、
  `0eccdf831c`（LoRA sync-load）、`c1663ec790`（smg control plane）
- prefill.py：`538ee9d0d1` 基线（不要动 collective）
- 部署脚本 `start_glm52_bf16_pd.sh` 已同步两端（md5 一致），双块均带 VE flag
