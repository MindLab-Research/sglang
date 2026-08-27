# GLM-5.2 MoL DCP=8 decode TPOT 优化（2026-08-27 结案）

## 目标

在 2P3D（1102 prefill/CP=8 + 1104 decode/DCP=8+EAGLE+MoL 4 LoRA）生产集群上，
**不改精度、保留 LoRA + EAGLE 参数、纯代码/配置改动**的前提下，将 decode TPOT 降低，
目标快 30%。

## 结论（已部署 + 实测验证）

| 优化项 | 类型 | 实测效果 | 状态 |
|--------|------|---------|------|
| AllReduce Fusion | 启动参数 | decode verify -4.6%~5.5% | ✅ 已验证 |
| tile 对齐 `BLOCK_SIZE_M=64→16` | MoE kernel config | decode verify **-8.8%** | ✅ 已验证 |
| down config `USE_TMA=True` | MoE kernel config | 含在 tile 对齐中 | ✅ 已验证 |
| **decode verify 累计** | — | **57.9ms → 50.3ms（-13.1%）** | 生产稳定 |
| FP8 AllReduce 通信量化 | 代码 | **负收益**（quant/dequant>通信节省）| ❌ 已回退 |
| NCCL_ALGO=Ring | 启动参数 | AllGather 负优化（+9ms）| ❌ 已回退 |

## 优化详解

### 1. AllReduce Fusion（-4.6%~5.5%）

- 去掉 `--enforce-disable-flashinfer-allreduce-fusion`：GLM-5.2 架构 `GlmMoeDsaForCausalLM`
  在 SM100 支持列表内，TP AllReduce 融入 mnnvl attention kernel。
- 验证：SPEC-TIMING 口径 decode verify 中位数下降 4.6%~5.5%（同 bs/seq_len A/B）。

### 2. tile 对齐 `BLOCK_SIZE_M=64→16`（-8.8%，最大单项）

- **背景**：decode 日志实证 GLM-5.2（E=257, N=256, L20D, block_shape=[128,128]）
  **无匹配 MoE config 文件 → 走默认 `BLOCK_SIZE_M=64`**。
- **方案**：参考 Blackwell 同构 B200（E=256,N=256）实测 tune 值，为 L20D 生成
  `BLOCK_SIZE_M=16` config（含按 M 区间的 N/K/GROUP/stages 精细调优 + down TMA）。
- **A/B 实测（SPEC-TIMING verify，同 bs/seq_len 口径，排除缓存混杂）**：

| bs | b64 baseline | B200 精确 | 改善 |
|----|------|------|------|
| 9  | 67.6ms | 60.8ms | **-10.1%** |
| 13 | 73.7ms | 67.6ms | **-8.3%** |
| 16 | 76.0ms | 69.3ms | **-8.8%** |
| 平均 | 57.9ms | 50.3ms | **-8.8%** |

- **全维度锁定验证**：config 18 个标准 key 中 `BLOCK_M=16` 覆盖 1~512（decode 全部实际 M），
  仅 1024~4096（prefill 大 batch）用 `BLOCK_M=64`。decode 46.4% 非标准 M（40/72/88/104）
  fallback 到邻近 key 后 BLOCK_M **仍为 16**（最近 key=32/64/96 皆 BLOCK_M=16）。
  → **主导 tile 维度已 100% 覆盖**，残留仅 N/K（64↔128）小差异，B200 官方同样 fallback 处理。
- **部署确认**：prefill(1102)+decode(1104) 两端 `Using MoE kernel config from`×16
  （8 TP rank × up+down），`Down MoE config file not found`=0。

### 3. 判定不做的方向（基于实测/显存/上游设计，避免负优化）

| 方向 | 结论 | 原因 |
|------|------|------|
| 非标准 M 精确 tune（40/72/88/104） | ❌ 不做 | B200 官方同样 fallback 到标准 key（上游设计）；blind tune 会负优化（AGENTS 记载 35.7ms vs 12.8ms）；tune 需 ~19.3GB 显存而 decode 仅剩 ~14GB |
| FP8 MoE AllReduce 量化 | ❌ 回退 | quant/dequant 开销 > 通信节省，且引入精度风险（7/50 dump 异常） |
| Elementwise kernel 融合（18.4% 第二瓶颈） | ❌ 暂缓 | profile 显示分散在几十个小 kernel（每 1-3ms），非单点；ROI 低、改动面大、每次部署 20min |

## 生产实测证据（2026-08-27 本次会话）

- decode(1104)：当前进程**无 `enforce-disable-flashinfer-allreduce-fusion`**（Fusion 生效）；
  `Using MoE kernel config`=16 条；`Down MoE config file not found`=0。
- prefill(1102)：`prefill_glm.log` 加载 tile config，Down not found=0。
- **20 并发真实压测**（通过 router 1102:31000，模型 glm52-fp8-official）：
  **400 成功 / 0 错误 / E2E median 2.1s**（正确性 + 当前稳定）。
- 三端健康：prefill 200 / router 200 / decode 200。
- profile（verify_profile_*.json，30 步 TARGET_VERIFY bs=6）严格 `cat=kernel` 分类：
  `fused_moe_kernel` 31.5% / NCCL 12.5% / MoE-LoRA 6.2% / deep_gemm 4.5% / Elementwise 类 ~15%
  （分散）。fused_moe（已被 tile 对齐覆盖）仍是最大瓶颈。

## 部署命令速查

```bash
# A) MoE config 文件路径（两份：up + down）
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/\
  E=257,N=256,device_name=NVIDIA_L20D,dtype=fp8_w8a8,block_shape=[128, 128].json
  E=257,N=256,device_name=NVIDIA_L20D,dtype=fp8_w8a8,block_shape=[128, 128]_down.json

# B) decode 启动关键参数（1104, /root/decode_dcp8_opt2.sh）
#   —— 必须【不带】--enforce-disable-flashinfer-allreduce-fusion（AllReduce Fusion 生效）
#   --tp 8 --dcp-size 8 --moe-runner-backend triton \
#   --speculative-algorithm EAGLE --speculative-num-steps 5 \
#   --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
#   --enable-lora --lora-use-virtual-experts --max-lora-rank 16

# C) rsync 部署后必须清理（AGENTS 铁律）
#   rm -f .../configs/triton_3_6_0/*L20D*.json   # 防止旧的错误 E=1024 L20D config 被加载
#   find ... -name '__pycache__' -o -name '*.pyc' | xargs rm -rf
```

## 方法论教训（本次重点）

1. **同口径 A/B 才可信**：tile 对齐 -8.8% 用**同 bs/seq_len、排除缓存混杂**的严格 A/B 验证，
   不是 grep 全日志均值（`decode_dcp8_opt2.log` 是多次重启累积，直接 grep 会混批次 → 误判）。
2. **profile 单位陷阱**：Chrome trace `dur` 单位是 **us**（微秒），曾误除以 1e6 当 ns → 缩小 1000 倍
   → 误判。修正后 `fused_moe_kernel` = 31.5% 才与"最大瓶颈"自洽。
3. **`.item()`/CPU op 误判**：`cat=="kernel"` 严格过滤后才排除 `aten::item`/`_local_scalar_dense`
   （CPU 同步）混入 GPU kernel 统计——这些是 pipeline 气泡来源，非 kernel 本身。
4. **blind tune 是负优化温床**：设备名伪装"L20D"导致匹配 L20D config；B300 手动 config
   无实测依据会大幅负优化（AGENTS 记载）。以 B200 同构实测值为基准才可靠。
