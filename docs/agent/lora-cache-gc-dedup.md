# LoRA 远程缓存：签名无关哈希去重 + 卸载后 24h GC

**日期**：2026-09-15（B300 PD 集群 SD-1021/1022 = `8.213.214.14`）
**改动**：新增 `python/sglang/srt/lora/lora_cache.py`；`python/sglang/srt/lora/lora_manager.py` 接入；单测 `test/srt/lora/test_lora_cache.py`
**触发**：排查"最近几天 LoRA 下载速度"时发现两个真问题——**同一个 adapter 被反复全量下载**（缓存 key 带签名），以及**缓存无回收**（磁盘 97%）。

---

## 1. 现场证据（1022 decode / 1021 prefill）

### 1.1 下载耗时（实测，`prefill/decode_glm53mol.log` 的 `downloading lora` → `extracting lora` → `ready at`）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 传输（curl 拉归档） | **11–12 s** | 09-13 → 09-15 共 ~60 次，区间 6–12 s，极稳 |
| 解压+落盘（tar --zstd） | **~20 s** | 单线程 15.2 GB 解压 ≈ 760 MB/s，**已是端到端主成本** |
| 端到端（DL → 8 rank ready） | 28–40 s | 只有 1 个 rank 真下，其余靠 `.lock`/`.done` 等 1–2 s |

单 adapter 解压后 = 15,277,245,049 B（`adapter_model.safetensors` 15,274,464,624 B + `adapter_config.json` 2.8 MB）。
归档 `.tar.zst` 解压后即删；抽 4×200 MiB 实测 zstd -3 压缩比 0.2363 / 0.2487 / 0.2498 / 0.2601（≈0.249）⇒ 归档 ≈ 3.8 GB ⇒ **有效带宽 ≈ 345 MB/s**（16 路 Range 并行；单流基线 ~21.8 MB/s，若单流需 170–380 s，实测 11 s）。

### 1.2 两个真问题

**(a) 缓存 key 掺了签名 ⇒ 每次下载都落新目录、永不命中**

旧代码（`lora_manager.resolve_lora_local_path`）：

```python
key = f"{base}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"   # name = 完整 URL（含签名）
```

预签名 URL 每次请求都变（`X-Amz-Date` / `X-Amz-Signature` / `X-Amz-Expires`）⇒ `sha1` 变 ⇒ `.done` 永不命中 ⇒ **同一 artifact 反复下载 + 反复解压**。

实证（1022）：174 个 `01a0*` 缓存目录中同一 UUID 出现 **10 / 7 / 6 / 4 / 4 / 3 / 3 份**；当天 `01a0a3ca` 在 14:46 与 17:54 各下一次（目录后缀 `-4ec3c3e2` vs `-69606578`）。

**(b) 缓存无回收**：`/root/glm52_local/loras` = **4.8 TB**，根分区 11 TB 用掉 9.5 TB（**97%**，剩 338 GB）。无 unload 生命周期、无 TTL、无重复清理。

---

## 2. 设计（`lora_cache.py`，纯 stdlib，便于无 torch 单测）

### 2.1 签名无关的内容寻址 key

```python
artifact_identity(url) = "{scheme}://{netloc}{path}"     # 丢掉 query（签名）
cache_key(url)         = "{basename(path) 去归档扩展名}-{sha1(identity)[:8]}"
```

同一 artifact 的刷新签名 URL ⇒ 同 key ⇒ 命中本地 `.done` ⇒ **直接复用，不下载**。
同时 `find_cached_adapter()` 会按 artifact basename 回扫：**旧 URL-哈希 key 写下的目录也能被复用**（生产上 174 个老目录立刻受益，无需重下）。

安全前提：artifact 以 ULID 命名、内容不可变（`adapters-v1/<uuid>/<uuid>.tar.zst`），故 path 即可作内容身份。

### 2.2 卸载后保留 24h 再删

- 加载时登记目录：`LoRAManager._lora_cache_dirs[lora_id] = cache_dir_for_path(local_path)`（嵌套 adapter 目录会向上找到带 `.done` 的缓存目录）。
- 卸载成功时写 `<cache_dir>/.unloaded_at = epoch`，日志：`LoRA cache: <name> unloaded, <dir> kept for 24 h before deletion`。
- 复用/新解压时 `touch_loaded()` → **删除 `.unloaded_at`**（重新加载即撤销待删状态，跨进程安全网）。
- 每次 **load 前** 与 **unload 后** 各跑一次 sweep（都是重操作，listdir 成本可忽略；生产 LoRA churn 频繁 ⇒ 无需 timer 线程）。

GC 删除条件（满足任一，且必须全部守卫通过）：

| 规则 | 说明 |
|---|---|
| `unloaded` 超期 | 有 `.unloaded_at` 且 `now - ts > TTL` |
| 重复副本 | 同 basename 多份 ⇒ 只保留最近使用的一份，其余在超 TTL 后删（**回收 10× 重复的关键**） |
| 半成品 | 无 `.done` 且无 `<dir>.lock` 且超 TTL |
| 残留归档 | `*.tar.zst/.tar.gz/...` 且超 TTL |

**永不删除**：`in_use`（本进程已加载的目录）、存在 `<dir>.lock`（下载在飞）、TTL 内被 touch 过（`touch_loaded` / `.done` mtime）。

env 旋钮（全部可选，读 `os.environ`，与既有 `SGLANG_LORA_CACHE_DIR` 同风格）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `SGLANG_LORA_CACHE_DIR` | `/root/glm52_local/loras` | 缓存根 |
| `SGLANG_LORA_CACHE_TTL_SEC` | `86400` | 卸载后保留时长 |
| `SGLANG_LORA_CACHE_GC` | `1` | 总开关 |
| `SGLANG_LORA_CACHE_DEDUPE` | `1` | 关闭后：不回扫旧 key 目录、不做重复副本 GC |

### 2.3 接线（`lora_manager.py`）

- `resolve_lora_local_path()` 变薄壳 → `lora_cache.resolve_remote_adapter()`（下载/解压/`.lock`/`.done` 流程原样搬过去，**不改变并发语义**）；`_find_adapter_dir` / `_stream_download` 保留为别名，避免外部引用断裂。
- `_load_lora_adapter()`：先 `_gc_lora_cache()`，再 resolve，再登记 cache dir。
- `_unload_lora_adapter()`：`mark_unloaded()` → `_gc_lora_cache()`。
- `_rollback_failed_lora_load()`：同步 pop `_lora_cache_dirs`。

---

## 3. 验证

`test/srt/lora/test_lora_cache.py`（stdlib-only，pytest 或无 pytest 直跑均可）：

```
python3 test/srt/lora/test_lora_cache.py     # 本地 16/16
# 节点（python3.12 + 真 zstd，覆盖 tar --zstd 分支）16/16
/root/v15_patched/bin/python3 /tmp/loratest/test/srt/lora/test_lora_cache.py
```

覆盖：签名刷新 key 不变 / 下载一次后复用（`downloads == 1`）/ 旧 URL-hash 目录复用 / 嵌套 `adapter/` 提升与 `cache_dir_for_path` / unload 超 TTL 才删 / TTL 内保留 / `in_use` 保留 / `.lock` 保留 / 重复副本仅留最新 / 残留归档 / malformed env 回退默认 / `.tar.zst` 分支（有 zstd 时）。

---

## 4. 运维

- 观察：`grep -E "already cached .*dedup hit|LoRA cache GC|kept for [0-9]+ h before deletion" <engine>.log`
- 首次生效：同一 adapter 的第二个签名 URL 起即命中（老目录也会被回扫复用），不再出现 `downloading lora`。
- 磁盘回收：下一次加载/卸载 sweep 自动清掉重复副本与超期目录；如需立即回收，可 `SGLANG_LORA_CACHE_TTL_SEC` 调小后触发一次加载/卸载，或直接删目录（**无需重启**：`lora_manager` 每次调用都读 env、现场扫盘）。
- 回滚：`SGLANG_LORA_CACHE_GC=0`（停 GC）/ `SGLANG_LORA_CACHE_DEDUPE=0`（停回扫），代码回退只需恢复 `lora_manager.py` 的薄壳函数。

## 5. 已知边界

- 复用判定基于 **artifact 身份（URL path）+ `.done`**，不做内容校验和；若平台以同名（同 ULID）覆盖写不同内容，缓存会命旧内容——当前 artifact 语义为不可变，故安全。
- GC 是"事件驱动 + 懒惰"的：长时间无 LoRA 加载/卸载时不会自行回收（磁盘不会因此增长）。要严格定时回收需另加 timer，本次刻意不做（避免引擎内起线程）。
- 8 个 TP rank 共享同一缓存目录：`in_use` 是进程内视图，跨 rank 保护依赖 `.unloaded_at`（只有真卸载才写）+ TTL 近期保护 + `.lock`。
