# UTF-8 变长编码导致 logprobs token 字节丢失（U+FFFD）问题修复记录

> B300 1P1D · glm52-fp8-official（glm53 rank-32 MoL）· sglang OpenAI serving 层
> 版本：2026-09-06 · 影响面：所有带 logprobs 的 OpenAI 接口（chat/completions）

---

## 1. 问题现象

对包含 4 字节 CJK 扩展 B 区字符的文本（如 `𠮷` = U+20BB7 = UTF-8 `F0 A0 AE B7`），引擎按 byte-level BPE 将其切成 **4 个单字节 fragment token**（`[F0] [A0] [AE] [B7]`，每个单字节都是 vocab 中的独立 token）。

网关返回的 logprobs 中，这 4 个 fragment 的 `bytes` 全部被腐蚀成 **U+FFFD 的 UTF-8 编码 `[239,191,189]`**，而非真实字节 `[240]/[160]/[174]/[183]`。

```
实际返回（错误）:
  logprobs.content[].bytes = [239,191,189] ×4   ← '\ufffd' 的 UTF-8 编码

期望返回（正确）:
  logprobs.content[].bytes = [240], [160], [174], [183]   ← 引擎原始 fragment 字节
```

**完整 token 不受影响**：`' 中文'` → `[32,228,184,173,230,150,135]` ✓，`'🙂'` → `[240,159,153,130]` ✓。只有 fragment token 损坏。

### 触发条件（三层 AND）

1. 字符不在 BPE 词表中 → 被切成单字节 fragment token（生僻字 / CJK 扩展 B 区 / 部分组合字符）
2. 请求开启了 `logprobs`（否则不返回 logprobs 字段，完全无感）
3. 客户端读取了 `logprobs.content[].bytes` 字段（或 completions 的 tokens 字符串）

普通对话（不开 logprobs）完全不受影响——content 文本始终正确（引擎整段 decode 会正确合并 fragment）。

---

## 2. 根因分析（三层信息丢失链）

引擎对每个生成 token 保留三元组 `(logprob, token_id, token_text)`——**token_id 全程正确保留**。信息丢失发生在 OpenAI serving 层的两次"文本化"转换：

### 第 1 层：引擎 detokenize（信息已损，行为合理）

```python
# tokenizer_manager.py
def detokenize_logprob_tokens(self, token_logprobs_val, token_logprobs_idx, decode_to_text):
    if not decode_to_text:
        return [(logprob, token_id, None) ...]      # ← token_id 保留
    token_texts = self.tokenizer.batch_decode([[idx] for idx in token_logprobs_idx])
    return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))
```

单字节 `0xF0` 单独 decode 不是合法 UTF-8 序列 → tokenizer 输出替换字符 `'\ufffd'`。
**注**：这是字节级 BPE 的固有性质——单字节 fragment 没有合法 UTF-8 文本形态，任何 tokenizer（含 tiktoken）单独 decode 都会给 U+FFFD。该层行为不算 bug；正确表达 fragment 只能靠 bytes 字段 / token_id。

### 第 2 层：serving_chat 用文本反推 bytes（真正的 bug）

```python
# serving_chat.py _process_logprobs_tokens（修复前）
token_bytes = list(token.encode("utf-8"))   # token 已是 '\ufffd' → [239,191,189]  ← 错！
```

`logprobs.tokens` 来自引擎第 1 层 decode 出的 `token_text`（fragment 已变 `'\ufffd'`），再 `encode("utf-8")` 反推 bytes → 把显示文本的 U+FFFD 当成了 token 的真实字节。**bytes 是字节级信息，正确来源是 token_id，却错误地走了文本通道**。

### 第 3 层：serving_completions 字符串通道（同源损坏）

```python
# utils.py to_openai_style_logprobs（修复前）
for logprob, _, token_text in token_logprobs:
    ret_logprobs.tokens.append(token_text)   # '\ufffd'，无 bytes 字段，string 形式丢失
```

OpenAI completions 老 API 的 LogProbs 结构无 per-token bytes 字段，tokens 为字符串——fragment 同样以 `'\ufffd'` 呈现。

---

## 3. 影响范围

| 维度 | 是否受影响 |
|---|---|
| `/v1/chat/completions`（流式 + 非流式） | ✅ logprobs.content[].bytes 损坏 |
| `/v1/completions`（流式 + 非流式） | ✅ logprobs.tokens 显示 '\ufffd' |
| 普通请求（不开 logprobs） | ❌ content 文本始终正确 |
| `/generate`（引擎内部接口） | ❌ 保留原始 token_id |
| 完整 token（'中文'/'🙂' 等在词表内） | ❌ bytes 正确 |

**同源部署影响**：该 bug 位于所有 sglang 部署共用的 OpenAI serving 层（`sglang/srt/entrypoints/openai/`），因此 18888（B300 1P1D）、13.124:30000（glm53）、8.222:31000（2P3D glm-5.3）等端点均存在同样问题——只是 18888 为最初复现与修复对象。

---

## 4. 修复方案

**核心原则**：bytes 只能从 token_id 恢复，绝不用 detokenized 显示文本反推。

### 4.1 utils.py：新增 GPT-2 byte_decoder 表 + token_id_to_bytes

byte-level BPE 的 vocab token 是"原始 UTF-8 字节经 GPT-2 `bytes_to_unicode` 映射后的可打印字符"，因此 `convert_ids_to_tokens(token_id)` 的结果可逐字符经逆表还原为原始字节：

```python
_BYTE_DECODER: Dict[str, int] = {}

def _build_byte_decoder() -> Dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord(chr(0xA1)), ord(chr(0xAC)) + 1))
          + list(range(ord(chr(0xAE)), ord(chr(0xFF)) + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b); cs.append(2**8 + n); n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(cs, bs))

def token_id_to_bytes(tokenizer, token_id) -> Optional[List[int]]:
    global _BYTE_DECODER
    if not _BYTE_DECODER:
        _BYTE_DECODER = _build_byte_decoder()
    try:
        piece = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None
    if piece is None:
        return None
    out = bytearray()
    for ch in piece:
        b = _BYTE_DECODER.get(ch)
        if b is None:
            return None          # 非 byte-level 可表示 → 调用方回退 display 文本
        out.append(b)
    return list(out) if out else None
```

**实测验证**（部署端 B300-2 tokenizer）：`'ð'→[240]`、`'ł'→[160]`、`'®'→[174]`、`'·'→[183]`；完整 token 还原与 utf-8 encode 完全一致。

### 4.2 serving_chat.py：直接消费引擎三元组（token_id）

替换 `_process_logprobs_tokens` 的错误反推逻辑，新增 `_build_token_logprobs_from_raw` 直接遍历引擎三元组 `(logprob, token_id, token_text)`：

```python
def _build_token_logprobs_from_raw(self, output_token_logprobs, output_top_logprobs, use_token_index=False):
    tokenizer = self.tokenizer_manager.tokenizer
    for token_idx, item in enumerate(output_token_logprobs):
        logprob, token_id, token_text = item
        token_bytes = token_id_to_bytes(tokenizer, token_id)
        if token_bytes is None:
            token_bytes = list((token_text or "").encode("utf-8"))
        # top_logprobs 同理：top_id → token_id_to_bytes
        ...
```

`_process_response_logprobs`（非流式）与 `_process_streaming_logprobs`（SSE 流式）两处调用点均改走该函数；删除含错误逻辑的旧 `_process_logprobs_tokens`。

### 4.3 serving_completions.py + utils.py：string 通道 latin-1 无损化

completions 老 API LogProbs 无 bytes 字段，fragment 以 latin-1 单字符可逆表示（每字节一字符）：

```python
def to_openai_style_logprobs(..., tokenizer=None):
    def append_token_logprobs(token_logprobs):
        for logprob, token_id, token_text in token_logprobs:
            token_text = _lossless_token_text(tokenizer, token_id, token_text)
            ret_logprobs.tokens.append(token_text)
            ...
```

`_lossless_token_text`：display 文本含 U+FFFD 且 token_id 可还原时，用 `bytes(raw).decode("latin-1")` 输出可逆单字节字符；否则保留原文本。serving_completions 两处调用传 `tokenizer=self.tokenizer_manager.tokenizer`。

---

## 5. 修改文件清单（本地仓库与两端 venv 三方 md5 一致）

| 文件 | 改动 |
|---|---|
| `sglang/srt/entrypoints/openai/utils.py` | +byte_decoder 表、+`token_id_to_bytes`、+`_lossless_token_text`、`to_openai_style_logprobs` 支持 tokenizer |
| `sglang/srt/entrypoints/openai/serving_chat.py` | +`_build_token_logprobs_from_raw`，两处调用改走；删旧 `_process_logprobs_tokens` |
| `sglang/srt/entrypoints/openai/serving_completions.py` | 两处 `to_openai_style_logprobs` 传 tokenizer |

同步路径：本地 `/home/smith/src/sglang-b300-glm52/python/sglang/srt/entrypoints/openai/`
↔ B300-1 `/root/sglang_venv/lib/python3.12/site-packages/sglang/srt/entrypoints/openai/`
↔ B300-2 `/root/v15_patched/lib/python3.12/site-packages/sglang/srt/entrypoints/openai/`

部署动作：改文件 → 双端重启 prefill/decode 引擎（`start_prefill_glm53mol.sh` / `start_decode_glm53mol.sh`）→ smg 熔断器约 75s 自动恢复。

---

## 6. 验证结果

### 6.1 chat SSE（bytes 通道）

```
== /v1/chat/completions SSE ==
fragment=12, 单字节值=[160,174,183,240]
✅ 真实字节 [240,160,174,183]（不再是 U+FFFD [239,191,189]）
```

完整 token 不受影响：`' 中文'` → `[32,228,184,173,230,150,135]` ✓；`'🙂'` → `[240,159,153,130]` ✓

### 6.2 completions（string 通道）

```
U+FFFD tokens: 0
✅ fragment 以 latin-1 可逆表示: 'ð'(0xF0) / '\xa0'(0xA0) / '®'(0xAE) / '·'(0xB7)
```

### 6.3 跨 SSE chunk 提示

一个生僻字 = 多个 fragment token，会分别出现在不同 SSE chunk 的 `logprobs.content[]` 中。客户端还原生僻字需**自拼连续 fragment 的 bytes 再按 UTF-8 边界解码**，不能按 chunk 边界或 logprobs 条目数当字符边界。content 文本流本身已正确合并，不受影响。

---

## 7. 复盘要点

1. **bytes 是字节级信息，永远应从 token_id 恢复，不要从 detokenized 显示文本反推**——文本通道天生有损（U+FFFD 无合法文本形态）
2. detokenize 单 fragment 给 U+FFFD 是字节级 BPE 的必然，不是 bug；bug 在 serving 层"拿注定有损的文本冒充字节真相"
3. 只伤"带 logprobs + 输出含 fragment token"的请求，且 logprobs 是必要条件；普通文本输出永不损坏
4. 该修复是共用 serving 层改动，一处修复双表面（chat/completions）同时生效；同源 sglang 部署如需同样修复须同步此 3 文件
