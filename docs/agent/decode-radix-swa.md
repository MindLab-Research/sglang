# DSV4 PD 双端 Radix Cache（hybrid-SWA decode radix）——设计与修复链

> 2026-08-15 完成的功能：DeepSeek V4 Pro 0813 PD 集群 prefill + decode **两端同时开 radix cache**，
> 多轮/重放负载 TTFT 3-30× 改善（64 并发重放吞吐 5.8K → 18.5K tok/s）。本文记录架构决策与四层 bug 修复链。

## 1. 为什么 DSpark 模型必须两端同开（不能只 prefill 开）

- **PD 前缀复用协议是 decode 主导**：decode 告知 `decode_prefix_len`（它已持有的前缀），
  prefill 只计算/传输 `[decode_prefix_len, len)`（`prefill.py: req.start_send_idx = decode_prefix_len`）。
- **KV page 可缓存可重发**（radix 树持有），但 **DSpark hidden states 不行**：DSpark head 吃 target
  第 58/59/60 层激活，只能由 prefill 前向产出、流式传输给 decode；不在树里、不能从 KV 反推。
- prefill 单开 radix 命中 [0, N) 后：KV 可从树重发（`maybe_send_cached_prefix_chunk`），但 hidden [0, N)
  无处可取 → 要么协议拒绝（`dspark_disaggregation.py:152 hidden_start != decode_prefix_len`），要么
  重算前向（收益归零）。EAGLE 没这个问题：draft 是独立小模型，bootstrap 状态 decode 侧重建。
- 结论：DSpark PD 有意义的前缀复用只有"两端同开"（decode 认领 → prefill 算后缀 + 后缀 hidden）。

## 2. 开启方式（1P1D 当前配置）

```bash
# decode (1022 /root/start_v4_decode.sh)
export SGLANG_DECODE_RADIX_ALLOW_SWA="1"   # 降级 SWA 硬护栏为 warning
--disaggregation-decode-enable-radix-cache
# prefill (1021 /root/start_v4_prefill.sh)：去掉 --disable-radix-cache 即可
```

## 3. 核心设计：`swa_served_from_tree = False`（树不持有 SWA）

DSV4 是 hybrid-SWA（`DeepseekV4ForCausalLM*` 在 `model_config.py:hybrid_swa_archs`）。decode-disagg
radix 模式下（`kv_cache_builder.py` 在 decode+hybrid_swa+decode_radix 时置位）：

1. **SWA match validator 恒 True** —— FULL 匹配不被树 SWA 存活状态门控（上一请求滑窗后，树 SWA
   永远在当前匹配边界之外，门控必然全拒）。
2. **SWA insert/复活路径全部跳过**（`swa_component.py` 的 overlap/unevict/commit 三处 early return）
   —— 关键收益：`_restore_device_value_with_locked_full` 会 free 请求 incoming FULL slots 而
   req_to_token 还引用它们 + finish 二次 free → **双重 free / 池 over-release / KV 污染**，跳过后消失。
3. **`swa_reprefill_tail_tokens() = window + page`** —— 两端 match 都按尾部窗口截断，prefill 重算
   最后一个窗口 → SWA 随 KV transfer 每次全新到达（正确性由传输保证，树不背锅）。
4. **finish 时 `free_swa(kv_indices[:page_aligned_len])`**（`unified_radix_cache.cache_finished_req`）
   —— 树不接管 SWA 后释放端的补位；经 mapping 过滤（窗口外头部 mapping=0 跳过），刻意不走
   `free()`/free_group（会连 FULL 一起 free，而 FULL 已归树）。

## 4. 四层 bug 修复链（按发现顺序，全部必装）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | `PD streaming hidden chunk arrived out of order: expected_start=0, chunk_start=13568` → KVTransferError → 请求 abort（agent 重放全挂） | prefill radix 命中未 clamp 到 decode 承诺：prefill 从 13568 发 hidden，decode 期望 0 | `schedule_batch.py:init_next_round_input` 把 `disagg_decode_prefix_len` 并入 match `key_limit` |
| 2 | `pack_int_lists: TypeError len(None)` → decode 8 rank 崩 | identical replay 全量命中 → `pd_hidden_len=0` → hidden 块跳过 → `state_indices` 里是 None | decode `_match_prefix_and_lock` match key 去掉最后一个 token（对齐聚合调度 `len-1` 不变量：末 token 必须重算产出首 logits + hidden bootstrap）；None→空 int32 数组兜底 |
| 3 | `AssertionError: new_prefix_len=3328, len(new_indices)=0` → 崩 | SWA validator 要求累计 ≥ window 的存活 SWA 才认节点；PD 传输只给尾部 ~180 token 窗口 → 凑不齐 → re-match 恒 0；且触发复活路径的双重 free（`#full token: -107520`） | 第 3 节设计（树不持有 SWA）；该断言降级为 `[SWA-INSERT]` tripwire warning（正常应永不触发） |
| 4 | idle 时 `pool memory leak detected! [full]...[swa] total=1029120, available=1028864` → 崩 | 树不接管 SWA 后，finish 请求的 SWA 窗口 slots 无人释放（漏 256/请求，1M 池 ~4000 请求耗尽） | 第 3 节第 4 点（`free_swa` 补位） |

**注意报错格式**：`_check_all_pools` 的 full+swa 消息用 `\n` 拼接后一起报 `[full]` 行——数字平衡也别信，
看下一行 `[swa]` 才是真凶。

## 5. 诊断工具（保留）

- `SGLANG_DECODE_RADIX_DIAG=1`（decode）：`[DRX-DIAG] match/insert(prebuilt)/insert(post)` 日志，
  含 key_len/l1/l2_host/device_indices/cache_protected/swa_evicted。
- `pool_stats_observer.py` over-release canary：`[DRX-DIAG] pool over-release`（available+evictable>size
  = 双重 free 探测器，正常应零）。
- `unified_radix_cache.py` `[SWA-INSERT]` tripwire：insert 深度 > SWA-validated match 时告警（正常零）。
- 权威命中证据：prefill 日志 `#cached-token: N`（N=decode 承诺）↔ decode 日志 `[DCP-PD-IDX] prefix_len=N`。

## 6. 验证结果（2026-08-15，1P1D V4 Pro 0813）

- 多轮延迟：3.8s → 2.0s → 1.7s（逐轮命中）；出师表背诵乱码检测 PASS。
- 64 并发 4096in/512out 同 seed 三连（RUN2/3 = 重放轰炸）：64/64 × 3，零 crash/零 leak/零 kv_err/
  零 over_release；RUN3 吞吐 18,505 tok/s（radix-off 基线 5,772）。
- idle 泄漏检查（on_idle 不变量）通过：full/swa 两池全平衡。

## 7. 已知残留

- RUN2 的 Median TPOT 30-42ms / Max ITL 6.7s 尖峰：重放轮 prefill 侧 chunk 调度排队所致（TTFT 已
  大幅下降），非 radix 正确性问题；可后续调 prefill chunked-prefill 顺序优化。
- prefill 端仍走"树持有 SWA"的常规路径（`swa_served_from_tree=True`），只受 clamp 约束——目前稳定。
