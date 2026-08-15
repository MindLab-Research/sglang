# DSV4 Prefill CP + DSpark PD hidden —— 三层修复链与 3.5× 长上下文加速

> 2026-08-15 完成：DeepSeek V4 Pro 0813 1P1D 上开启 prefill CP（attn_cp=8，DSA indexer
> n² 切分）+ DSPARK PD hidden 传输共存。200K 冷前缀 prefill drain 11K → 38.7K tok/s
> （3.5×），正确性（出师表 garble）无损，0 崩溃。

## 1. 开启方式（1P1D 当前生产配置）

```bash
# prefill (1021 /root/start_v4_prefill.sh) 关键 flags：
--chunked-prefill-size 16384 --max-prefill-tokens 16384   # CP 前 8192
--enable-prefill-cp --cp-strategy interleave
--enable-hierarchical-cache --hicache-ratio 1 --hicache-write-policy write_back \
--hicache-mem-layout page_first --hicache-storage-backend file --file-storage-path /root/hicache
# 注意：GLM 的 --enable-dsa-prefill-cp-layersplit 不适用于 V4（见 §2 flag 语义）
# decode 不变（DSPARK + dcp4 + decode radix）
```

**DSV4 专用 CP hook**（`arg_groups/deepseek_v4_hook.py::validate_deepseek_v4_cp`）：
`--enable-prefill-cp --cp-strategy interleave` 自动配置 round-robin-split +
`attn_cp_size = tp_size`（8）+ `enable_dp_attention` + `moe_dense_tp_size=1`，并关
`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`。日志标记：`Enable Context Parallel for DeepSeekV4,
attn_cp_size=8`。**不支持 zigzag，只支持 interleave**。

## 2. 三层 bug 修复链（全必装）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | 首启崩溃 `--enable-prefill-context-parallel and --enable-nsa-prefill-context-parallel are mutually exclusive` | flag 归一化（`_handle_prefill_cp_aliases`）先跑，看不到 dsv4 backend，把 MLA 别名置 True；随后 DSV4 hook 又置 NSA 别名 True → 互斥断言 | `deepseek_v4_hook.py`：hook 置 `enable_dsa_prefill_context_parallel=True` 时显式 `enable_prefill_context_parallel = False` |
| 2 | 首个请求崩溃 `NotImplementedError: DSpark aux hidden-state capture is not supported together with ... CP` | v1 硬禁（怕的就是 #3） | `deepseek_v4.py`：删 gate；层循环内每 CP rank 捕获自己 round-robin 分片的行（原逻辑），循环尾部对每个 aux 张量做 `cp_all_gather_rerange_output` 重组回全序列（与 final hidden 同法）。chunked-prefill 限制单次 forward ≤16384 行，all-gather 瞬时 ~230MB 无忧 |
| 3 | 每个请求 500：decode 报 `PD streaming hidden chunk arrived out of order: expected_start=10, chunk_start=0` | **CP 把 prefill attn_tp 8→1**（被 attn_cp 稀释）。decode `_resolve_rank_mapping`（common/conn.py:557）按 `attn_tp_size 比较` 走进"prefill TP1 广播"分支 → `required_dst_info_num=8` → **每个 prefill rank 向全部 8 个 decode rank 发 KV+hidden 通知**（64 份，8 倍冗余）→ PD hidden 是每 decode rank 一条有序流（`next_start` 严格递增），第二份重复 chunk 即 out-of-order → KVTransferError | **对角配对**：① `TransferInfo` 新增 `decode_engine_rank`（bootstrap metadata 帧 12，旧版缺帧=-1 兼容）② decode `send_metadata` 附带 `kv_args.engine_rank` ③ prefill transfer worker 的 hidden 计数/发送双循环过滤 `req.decode_engine_rank in (-1, prefill_unique_rank)` —— 8 prefill × 8 decode 变成 8 对 1:1 对角（KV 广播保持不变，只有 hidden 单流化） |

插桩定位法（这次的关键手段）：`SGLANG_DEBUG_DIAG=1` 时 prefill 打 `[PDH-SEND] tp_rank/rid/room/start/row_len`、decode 打 `[PDH-RECV] engine_rank/room/prefill_rank/start`。坏形态：
`engine_rank=7 收到 prefill_rank=5,4,1... 同一 (room,start)`；好形态：`engine_rank=N ← prefill_rank=N`
恰好一份。

## 3. flag 语义辨析（易混）

- `--enable-prefill-cp --cp-strategy interleave`：**DSV4/V4-Flash 用这套**（进 DSV4 hook）
- `--enable-dsa-prefill-cp-layersplit`：GLM-5.2 DSA backend 的层切分机制，**与 V4 hook 互斥**（同开会触发 #1 的断言）——V4 上别加
- `SGLANG_ENABLE_DSA_PREFILL_CP_LAYERSPLIT_UNEVEN`：GLM 61%8≠0 的 UNEVEN 支持，V4 不需要（残留 env 无害但建议清掉）

## 4. 性能与正确性数据（2026-08-15 实测，1P1D）

- 200K 冷前缀 prefill drain：11,000 → **36,870~39,586 tok/s**（3.4-3.6×，与 GLM DSA CP8 区间一致）
- chunk 16384（原 8192）：CP 前提下摊薄每 chunk 固定开销（barrier/broadcast/process_result）
- 出师表 garble PASS；PDH 对角 8×1 份；out-of-order=0；双端 0 crash；idle 泄漏检查 0

## 5. 相关：热路径日志 gate（同日落地）

`SGLANG_DEBUG_DIAG`（默认关）统一 gate 12 个热路径日志点：DSP-STAGE/DSP-ACCEPT
（dspark_worker_v2）、DSP-WIN（planner）、DSP-ALLOC（allocation）、DCP-XFER(-BROADCAST)
（mooncake/conn）、DCP-PD-IDX/DEBUG-CAN/DEBUG-INIT（decode）、CACHE_UNFINISHED
（unified_radix_cache）、PDH-SEND（prefill）。开启前 decode 日志 1281 行/s（≈3-5ms/步纯税），
开后 0。诊断时两端脚本临时 `export SGLANG_DEBUG_DIAG="1"`，用完删。

另：`SGLANG_ENABLE_TREE_SANITY_CHECK`（默认关）gate decode idle 的 radix
`sanity_check()`——O(全树) 多遍，DSV4+decode-radix 首次满足触发条件，大树每次 idle 卡数秒
（"health 200 但卡死"、吞吐塌到 0.81 tok/s 的根因）。

## 6. ⛔ PD hidden 双侧 pool 必须配对（2026-08-15 深夜死锁修复）

**streaming hidden 是配对滚动窗口协议**：decode recv pool（`SGLANG_PD_HIDDEN_RECV_POOL_TOKENS`，
默认=max_prefill_tokens）决定窗口尺寸（dst_indices 长度）；prefill sender pool
（`SGLANG_PD_HIDDEN_POOL_TOKENS`，默认同）必须装得下同尺寸窗口，且**源行只在对侧 ACK
后才释放**（`Streaming source rows are released only after the matching hidden chunk ACK`）。

**死锁链（撞过一次）**：只扩 decode pool（8192→65536）不扩 prefill（16384）→ 大请求
窗口 65536 > prefill 池 → `hidden rows exceed prefill hidden pool capacity` 传输失败 →
abort 后 decode 侧已占的 65536 行**不释放**（失败路径泄漏）→ 后续一切请求
`PD decode hidden pool blocked prealloc: free_rows=0` → "发个请求把服务卡死"。

**修复 = 两侧对齐 65536**（两端启动脚本 env；显存代价 prefill+decode 各 ~2.8GB，B300 可承受）：
```bash
# prefill 脚本: export SGLANG_PD_HIDDEN_POOL_TOKENS="65536"
# decode  脚本: export SGLANG_PD_HIDDEN_RECV_POOL_TOKENS="65536"
```
验证（2026-08-16 00:00）：100K E2E 200/5.8s；**630K（真实场景）E2E 200/23.6s**；
ACK 对账 SEND=480=ACK=480 完全闭环；事后小请求 0.55s。

**已知残留**：传输失败路径的 decode 窗口行释放仍可疑（池对齐后失败本身罕见，未复现）；
换 pool 尺寸时**两侧必须同时改**；验证大请求必须等端到端 200，不能只看 prefill 吞吐
（"3.5× 全绿"曾是假阴性——出师表单窗口 + 吞吐采样都不触发多窗口 ACK 滚动）。

ACK 链诊断（DIAG 开时）：`[PDH-SEND]`（prefill 发窗口）/`[PDH-RECV]`（decode 收通知）/
`[PDH-ACK-SUBMIT]`（decode 发 ACK）/`[PDH-ACK]`（prefill 收 ACK），按 (rid,start) 对账即知断在哪跳。
