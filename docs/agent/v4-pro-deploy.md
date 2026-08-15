# V4 Pro 0813 部署知识（B300 1P1D 集群）

## 集群拓扑（当前运行配置）

| 节点 | 角色 | 内网 | 服务端口 |
|---|---|---|---|
| B300-1 (SSH 1021) | prefill + router | 10.0.0.75 | 30100 (engine) / 31000 (router) / 29001 (prometheus) |
| B300-2 (SSH 1022) | decode | 10.0.0.67 | 30200 |

- 模型：`deepseek-v4-pro-0813`（官方 FP4 expert + FP8 attention 混合量化，853GB / 66 shards）
- 公网：`8.213.215.2:18888` → router 31000，API key `sk-glm52-pd`
- venv：1021 `/root/sglang_venv`，1022 `/root/v15_patched`
- 启动脚本：`/root/start_v4_prefill.sh`（1021）、`/root/start_v4_decode.sh`（1022）
- 一键部署脚本：`/root/deploy_pro.sh`（1021 上，含验证/替换/重启/router/测试全链）

## ⛔ 核心教训：第三方重打包 checkpoint

**本地 `/root/deepseek_v4_pro_0813_repacked_bad`（已备份）是第三方重打包的变体，非官方文件**。识别方法（HTTP Range 只下载 shard 头几 MB 解析 safetensors JSON header）：

| 项目 | 重打包变体（坏） | 官方 |
|---|---|---|
| config.json | Flash 的值（4096/1024/64头/43层/256e） | **7168/1536/128头/61层/384e** + dspark_layers=[58,59,60] |
| expert w1.weight | `[3072, 5376]`（6-bit 两级缩放，1.5×） | `[3072, 3584]`（K/2 packed FP4） |
| expert w1.scale | `[3072, 336]`（3B/64elem） | `[3072, 224]`（标准 per-32 e8m0） |
| shard2 tensors | 1568 | 2337 |

**症状**：load 时 `RuntimeError: size of tensor a (224) must match b (336)`（scale 行数）、`(3584) vs (5376)`（weight 行数）——比值恰好 3:2。总字节数守恒（831GiB）极具迷惑性。

**官方 config 关键值**：hidden_size=7168, q_lora_rank=1536, num_attention_heads=128, head_dim=512, num_hidden_layers=61, n_routed_experts=384, num_experts_per_tok=6, expert_dtype=fp4, moe_intermediate_size=3072, dspark_block_size=5, dspark_target_layer_ids=[58,59,60]。

## 正确部署步骤

1. **下载官方权重**（两端并行，`hf download deepseek-ai/DeepSeek-V4-Pro-0813 --local-dir /root/dsv4_pro_official --max-workers 8`，实测 150-470MB/s，约 25 分钟）
2. **验证官方格式**（`/tmp/verify_official.py`：shard2 头部 w1.weight shape 必须是 `[3072, 3584]`）
3. **替换**：旧目录 mv 到 `_repacked_bad`，`cp -al`（硬链接，不占额外空间）官方权重到运行目录
4. **decode 端 DSpark**：Pro 0813 自带 DSpark head（config 里 dspark_target_layer_ids），只加 `--speculative-algorithm DSPARK`，**不要加 EAGLE flags**（num-steps/eagle-topk/num-draft-tokens）
5. **MoE backend**：`--moe-runner-backend flashinfer_mxfp4`（B300=SM100 系走 TRT-LLM FP4 kernel：`trtllm_fp4_block_scale_moe`）
6. **重启前必杀残留 GPU 进程**（`nvidia-smi --query-compute-apps=pid | xargs kill -9`）——旧 scheduler 各占 121GB 显存会导致 OOM；8998 端口占用会导致 bootstrap bind 失败

## 容量与性能（实测）

| | prefill | decode |
|---|---|---|
| max_total_num_tokens | 11,466,496 | 10,292,224 |
| context_len | 1M | 1M |
| max_running_requests | 256 | 64 |

64 并发 bench（4K in / 512 out）：64/64 成功，Output 684 tok/s（峰值 1707），Median TPOT 8.4ms，Median TTFT 12.3s（排队），0 crash。

## 已知行为（非 bug）

- **>1M token 请求正确返回 400**：`Token indices sequence length is longer than the specified maximum (1051973 > 1048576)`——agent 侧 token 估算（xbot maybeCompress）与实际 tokenizer 计数有 ~8% 偏差，1M 附近会偶发超限。475K/489K 实测正常处理。
- 401 = 未带 `Authorization: Bearer sk-glm52-pd`（router 鉴权正常工作）。
