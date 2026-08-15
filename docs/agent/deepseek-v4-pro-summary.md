# DeepSeek V4 Pro 0813 PD 部署技术报告

> 2026-08-15 完成。汇总 DSV4 Pro 0813 在 B300 1P1D 集群（prefill 1021 + decode 1022）
> 上 PD 分离部署的全部代码改动，共 4 个 commit。各专题的实现细节见对应文档。

## 0. 改动全景

| commit | 主题 | 核心文件 | 行数 |
|---|---|---|---|
| `20856e6d3f` | 双端 radix cache 全链路（树不持有 SWA） | `unified_radix_cache.py`、`swa_component.py`、`decode.py` | +262/-21 |
| `d300afc750` | prefill CP + DSpark PD hidden 共存（3.5×） | `deepseek_v4.py`、`mooncake/conn.py`、`deepseek_v4_hook.py` | +231/-67 |
| `f29e50bc93` | 双侧 hidden pool 配对 + ACK 插桩 | `mooncake/conn.py` | +40 |
| `1d5d602855` | 运维脚本更新（recover/repro/博客） | 脚本 + docs | +2276 |

专题文档索引：
- 部署全流程 + 重打包 checkpoint 识别 → `v4-pro-deploy.md`
- 编译层三修复（DeepGEMM sm_103a / tilelang flock / CCCL）→ `b300-compile-fixes.md`
- DSpark PD 五个死锁修复链 → `dspark-pd-deadlocks.md`
- CP + DSpark hidden 三层共存 → `dsv4-cp-dspark.md`
- 双端 radix + 树不持有 SWA → `decode-radix-swa.md`

---

## 1. 双端 radix cache（`20856e6d3f`）

### 动机

DSPARK 的 hidden state 吃 target 中间层激活（`dspark_target_layer_ids=[58,59,60]`），
**不可缓存、不可从 KV 反推**。prefill 单开命中后 hidden 覆盖缺口必崩：
`hidden_start != decode_prefix_len` → 500 / 流式 out-of-order → abort。因此两端必须同开 radix。

### 设计核心 —— `swa_served_from_tree = False`（树不持有 SWA）

| 机制 | 作用 |
|---|---|
| validator 恒过 | 不对 SWA 做校验 |
| insert overlap 跳过 | 防双重 free |
| 尾窗重算 | 保证 SWA 新鲜 |
| finish 时 `free_swa` 补位释放 | 防池泄漏 |

### 配套

- `SGLANG_DECODE_RADIX_ALLOW_SWA=1`（decode 端）
- `--disaggregation-decode-enable-radix-cache`（decode 侧 flag）
- 修了 4 层 bug：hidden clamp、len-1 match、SWA validator、SWA 池泄漏

---

## 2. CP layer-split + DSpark hidden 共存（`d300afc750`）

### 收益

`--enable-prefill-cp --cp-strategy interleave`（attn_cp=8 切 DSA indexer 的 n²），
200K 冷前缀 prefill drain **11K → 38.7K tok/s（3.5×）**，garble 无损。

### 三层修复

| # | 文件 | 症状 | 修复 |
|---|---|---|---|
| ① | `deepseek_v4_hook.py` | 首启崩溃 `--enable-prefill-context-parallel and --enable-nsa-prefill-context-parallel are mutually exclusive` | flag 归一化先于 hook 把 MLA 别名置 True，hook 再置 NSA 别名 → 互斥断言。**hook 置 NSA 别名时显式清 MLA 别名** |
| ② | `deepseek_v4.py` | `NotImplementedError: DSpark aux hidden-state capture is not supported ... CP` | 删 v1 硬禁；层循环捕获 CP 分片行，尾部对 aux hidden 做 `cp_all_gather_rerange_output` 重组回全序列（与 final hidden 同法，下游 PD hidden 管道零改动） |
| ③ | `mooncake/conn.py` | hidden 8 倍重复发送 → 有序流 out-of-order 全 500 | CP 稀释 prefill attn_tp 8→1，decode rank-mapping 走广播（`required_dst_info_num=8`）；`TransferInfo` 增 `decode_engine_rank`（metadata 帧 12，`-1` 兼容），worker hidden 双循环**对角过滤**，8×8 → 8 对 1:1（KV 广播不变） |

---

## 3. 双侧 hidden pool 配对（`f29e50bc93`）

decode recv pool 决定窗口大小；prefill sender pool 必须装同尺寸。
prefill pool 仍 16384 而所需窗口 65536 → `DSpark hidden rows exceed prefill hidden pool capacity`
→ 传输失败 → abort 后 decode hidden window 泄漏。

**修复**：双侧 pool 对齐为 `65536` + ACK 链插桩（`PDH-SEND/RECV/ACK*`）。

---

## 4. 诊断支撑（`SGLANG_DEBUG_DIAG`，默认关）

gate 12 个热路径日志点（`DSP-STAGE/DSP-ACCEPT/DCP-XFER/DCP-PD-IDX/DEBUG-CAN/
DEBUG-INIT/CACHE_UNFINISHED/SWA-INSERT/PDH-*`）。开启前 decode 1281 行/s ≈ 3-5ms/步纯日志税。

另两个 gate：
- `SGLANG_ENABLE_TREE_SANITY_CHECK`（默认关）—— decode idle 的 radix `sanity_check` 是 O(全树) 多遍，DSV4 + decode-radix 首次触发，大树每次 idle 卡数秒 → health 200 但卡死、吞吐 0.81 tok/s。
- prefill chunk 8192→16384 + HiCache 三级（L1/L2/L3 file）。

---

## 5. 运维脚本（`1d5d602855`）

- `recover_b300_v4_pd.sh`：从 V4 Flash 0731 更新到 Pro 0813（CP interleave、双端 radix、hidden pool 65536、去 EAGLE flags，Pro 自带 DSpark head）
- `tools/repro_1p1d.py`：旧 IP/旧模型 → 公网 `8.213.215.2:18888` + `deepseek-v4-pro-0813`
- `docs/merge_l2.py` + `prep_fused_deploy.sh`：GLM L2 融合脚本（沿用）
- `docs/dcp-blog*.html`：DCP KV resharding 技术博客
- `.gitignore`：忽略 `.xbot/`

---

## 6. 「乱码/无限循环」专项调查（2026-08-15 深夜，结论：非服务 bug）

### 6.1 症状

64 并发 replay 下 decode `accept rate 0.99 / accept len 5.95`，DSP-ACCEPT 日志大量 3-token
循环（`[5272,12975,65]`→`CTRL_`、`[65,5745,79]`→`_glm`、`[425,482,65]`→`abort_`），
最终输出如 `abort_abort_abort...`、`_glm_glm_glm...`。用户 agent 低概率遇到同样循环。

### 6.2 排查链（全部做完的二分实验）

| 实验 | 配置 | 结果 |
|---|---|---|
| 部署卫生全套 | md5 对齐 + 清 pycache + 删 L20D + 清 HiCache 旧 KV | 循环依旧 → 排除部署漂移 |
| radix 复用嫌疑 | 双端 radix OFF + HiCache OFF | 4 并发 2/4 循环 → 排除 |
| prefill CP 嫌疑 | CP OFF（radix 仍 OFF） | 4 并发 1/4 循环 → 排除 |
| batch/并发嫌疑 | **串行单请求 × 10** | **5/10 循环（50%）** → 排除并发 |
| **采样温度对照** | 同一乱码 prompt：temp=0 × 6 vs temp=1.0/top_p=0.95 × 6 | **greedy 5/6 循环；temp=1.0 0/6 循环** |

### 6.3 根因

**replay.csv 数据集含大量乱码/污染行**（尾部是无意义多语言 token 流），请求体自带
`temperature=0`（greedy）。乱码 prompt 使模型输出分布尖锐，greedy 解码陷入 token 级
循环吸引子（`CTRL_`/`abort_`/`OPTIONS_` 都是 prompt 尾部片段的自然续写）。**KV cache
与 target 分布完全正常**——temp=1.0 下同样 prompt 零循环、输出正常。

### 6.4 accept rate 语义修正（重要认知）

- **accept 0.99 + 输出循环** = 模型自身循环（分布退化为确定性重复），DSPARK draft
  （吃 target 中间层激活）完美跟随 → 全接受。这是数据 × 采样的**模型行为**。
- 反直觉的是：**真 KV 污染（乱码）时 accept rate 反而会掉**——target 分布变随机，
  draft 不再与 target 一致，verify 大量拒绝。所以「accept 接近 1 = 乱码」不总成立；
  判据是 **accept 异常高 + 输出 token 级短循环**，且循环内容是 prompt 片段复读。

### 6.5 修复（mint-bench `bee3355`）

- `run_bench.sh` + `bench.yml` 新增 `replay_temperature` input（透传 `--temperature`）。
- 验收标准：64 并发 replay 用 `replay_temperature=1.0` 跑，accept rate 回落至正常区间。

### 6.6 排查中发现的运维新坑（已写入启动脚本注释位）

1. **router `--policy cache_aware` 与 prefill radix OFF 不兼容**：关 radix 后 router 的
   prefill 健康探测持续失败 → 503 卡死（今晚 router 三次卡死的根因）。二分实验关 radix
   时必须同时换 `--policy round_robin`。
2. **decode 无 DSPARK 起不来**（PD 模式 bootstrap 依赖 spec_metadata 协商）：
   `--speculative-algorithm` 去掉后 decode 报 `Failed to get kvcache from prefill`，
   无法用「关投机解码」做二分。
3. **PD 双端重启顺序铁律再确认**：任一端重启后另一端必须跟着重启重建 bootstrap 会话，
   否则 room 配对错乱（prefill 处理 room X、decode 等 room Y）。

---

## 7. 64 并发 replay 实测

### 7.1 temp=0（原始 CSV 请求体，历史数据）

run `2026-08-15T171606Z`（修复部署卫生后）：204 请求全 401（key 错误，作废）。
run `2026-08-15T171959Z`：accept rate 从 0.365 → 0.99 渐进锁死（乱码行逐批陷入循环）。

### 7.2 temp=1.0（验收 run `31904368671`，mint-bench bee3355）

| 指标 | temp=0（旧） | **temp=1.0（验收）** |
|---|---|---|
| accept rate（全程） | 0.365 → **0.99 锁死** | **0.12–0.19 稳定**（终值 0.145） |
| finish=length（循环锁死到 max_tokens） | **63/204（31%）** | **1/182（0.5%）** |
| DSP-ACCEPT 短循环序列占比 | 61% | **0%** |
| 成功数 | 204/204 | 182/204（22 个排队期客户端超时） |
| cache_ratio p50 | — | 0.996（双端 radix 正常） |
| prompt/completion tokens | — | 22.8M / 732K |

验收结论：**循环消失、accept rate 不再异常接近 1**。0.14 的绝对值是 temp=1.0 ×
乱码行（平坦分布）下投机解码的诚实结果——draft 给 argmax、target 随机采样，命中率
天然低；正常业务 prompt（分布尖锐）的 accept rate 不受影响。生产侧若遇到 agent 循环，
优先检查客户端采样参数（避免 temp=0 长上下文 greedy）。