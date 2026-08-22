# CP Round-Robin Split Pad-Row Garbling（2026-08-22 结案）

> ⚠️ 2026-08-22 03:27 UTC：gate 修复 `f1421f2241` 已被 commit `08deb10807` **revert**
> （当前 HEAD `ff525bd0dd` 的 `can_dsa_prefill_cp_round_robin_split` 已无
> `seq_len % cp_size == 0` 检查，见 `srt/layers/attention/dsa/utils.py:109-124`）。
> 下文"修复后 28 连发 0 乱码"对应的是 revert 前的状态；bug 面已重新打开，正在重新
> 深挖 -1 哨兵体系下残余的未守卫消费点（embedding kernel、MoE-LoRA stamp、
> control 路径负索引回绕等）。
>
> ✅ **终局结论（2026-08-22 17:5x UTC）**：根因不在 CP。当日 A/B 消融链定罪
> **DCP decode 双 bug**（`5974a4fc56`：index_k 低水位直通错位 + topk -1 lane →
> slot 0 污染，见 `dcp-virtual-id-domain-fix.md` §7）——CP 关掉乱码依旧，DCP 关掉
> 才干净。本文件描述的 pad-row 现象是 DCP bug 在 LoRA+CP 组合下的显影之一，
> gate 修复 f1421f2241 的"28 连发 0 乱码"实为撞上了 DCP bug 的低水位窗口
> （重启后水位重置）。此文件保留作现象记录，**结论以 §7 为准**。

## 症状

MoL 线上 L2（及任意 LoRA）请求间歇性输出乱码，两种签名并存：

1. **模板 token 汤**：`</arg_value></arg_value><arg_key>1</arg_value>...`、`<arg_key>limit</arg_key>`×4 循环、`</think>` 泄漏进 content/args
2. **路径/参数损坏**（更隐蔽）：`cd /home/smith/src/xbot/web610>/dev/null/e/smartypec==7g0.7/nman0;`、`src/x2>/dev/null`、`web/src/x0.0.0;1>/dev/null3` —— 高频上下文 token（/dev/null 等）污染参数

关键特征：
- **+1~2 个输入字符 → 乱码消失**（用户观察，实验证实）
- 全缓存命中（只 prefill ~54 token）也乱 → tiny extend 足以触发
- 单发干净（16K chunk 整除 cp_size 无 padding），**并发混合长度批次 ~30% 乱码率**
- base 模型完全免疫（无 LoRA delta）
- 与上下文长度无直接关系（150K 截断干净，476K 单发也干净——**乱的是 padding 不是长度**）

## 根因

`can_dsa_prefill_cp_round_robin_split`（`srt/layers/attention/dsa/utils.py`）只检查
`seq_len >= cp_size`，**不检查整除性**。非整除 extend 被补齐到 cp_size=8 对齐后才进
round-robin 分片，pad 行流入 LoRA segment 路径（chunked_backend.py 的 -1 哨兵体系）
后在某条未完全守卫的路径上污染真实行——输出是**有限值的错误 token**（非 12cda1cd60
的 NaN→'!' 汤，是同族不同枝）。54 token extend → 2 个 pad 行就够触发。

## 修复（commit f1421f2241）— ⚠️ 已被 revert（08deb10807，2026-08-22 03:27）

gate 增加 `seq_len % cp_size == 0`：非整除批次整体跳过 CP 分片（无 padding 即无 bug）。
`extend_num_tokens` 是 batch-prep 期计算的 rank-invariant 值（d4d23041d 死锁家族安全）。
性能损失仅限尾部小块/混合奇数批次；16K chunked-prefill 主路径本身整除，保留 CP 加速。

> ⚠️ **状态更正（2026-08-22 05:5x）**：`08deb10807` 已 revert 本修复——当前 HEAD 的
> `can_dsa_prefill_cp_round_robin_split`（dsa/utils.py:109-124）**没有整除性检查**，
> pad 行类分歧（embedding/MLP 层 `covered != x_rows` → `_resolve_batch_info` clamp 分支）
> 重新激活。此文件描述的"修复后 28 连发 0 乱码"验证的是已 revert 的代码。

## 验证（1102/1104 测试对，v15 镜像树 + 修复文件）

用录制到的真实乱码 payload（47.6 万 token、71 tools、model=L2+lora_path、temp 默认 1.0）：
- 修复前：11 路并发重放 ~35 次乱码 ~12 次；对撞实验 PAIR_d1_A（原始）乱 / PAIR_d1_B（+1 字符）干净
- 修复后：**28 连发（含确定性乱码的原始 payload + 4 路并发×6 轮）0 乱码**

## 复现/判别方法论（可复用）

1. **录制代理**（本地 :8022 → SSH 隧道 → 1101 gateway，强制 model=L2+lora_path，SSE 逐
   chunk 透传 + 双向字节落盘 `/tmp/recap/`）抓到用户真实乱码请求 → `req_006.json`
2. 同 payload 打 base / L0/L1/L3 / 截断 150K / 单发 vs 并发 → 收敛到"LoRA+padding"
3. **并发对撞**（A=原始、B=+1 字符同时发）是定位对齐依赖 bug 的利器
4. 乱码检测启发式：`</arg_value>`/`</think>`/`<tool_call>` 计数 + 重复片段 + 路径损坏
   正则（`/dev/null|0\.0\.0|x2>|src/x[0-9]`）——纯 token 计数会漏掉"参数损坏"型乱码

## 踩坑记录（本次调查走的弯路）

- 代码统一排除了 v15 树漂移后乱码依旧 → 不是代码版本问题
- flush_cache"暂时恢复"是巧合（乱码与缓存无关，全缓存命中也乱）
- 合成 148K 无 tools 复现永远是干净的——**必须用真实 tools payload + 并发混合长度**
- 乱码请求 accept rate 或 ≈1.0（死循环）或异常低（噪声采样）——不可单独作为定位依据
- 生产三节点 v15 树历史上从未整树统一（1101 差 43 文件、1100 差 50、互不一致）——
  2026-08-22 已统一到本地 HEAD；逐文件 rsync 的修复方式留下了 transform_index 旧版等暗雷
