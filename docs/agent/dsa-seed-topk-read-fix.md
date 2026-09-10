# DSA Seed Top-K Read Error Fix — 2026-09-05

## 问题
GLM-5.3 PD 集群（1102/1104）accept rate 降至 0.01-0.13，出现乱码和自愈现象。

## 根因
`eagle_worker_v2.py` 中 `_draft_extend_for_decode()` 读取 DSA seed top-k 时使用了错误的索引：

```python
# 错误代码
dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]
```

`forward_mha.py` 的 indexer 将 per-request DSA seed 写入 `seed_buf[:bs]`（buffer 的前 bs 行），
但 `select_index` 指向的是 token 行号（如 3 reqs × 6 tokens = `[5, 11, 17]`），而不是 seed 行号 `[0, 1, 2]`。
这导致读取到 stale/garbage 数据而非 DSA seed，污染下一步 draft 的 sparse attention seed，
造成 accept rate 下降和乱码。

## 修复

### 文件
`python/sglang/srt/speculative/eagle_worker_v2.py`，commit `5085112225`，分支 `fix/eagle-seed-contract`

### 变更
```python
# 修复前（错误）
dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

# 修复后（正确）
try:
    _bs = batch.batch_size() if callable(batch.batch_size) else batch.batch_size
    if _bs > 0 and _bs <= dsa_extend_topk_capture.shape[0]:
        dsa_seed_topk_indices = dsa_extend_topk_capture[:_bs].clone()
    else:
        logger.warning(...)
        dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]
except Exception as e:
    logger.warning(...)
    dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]
```

### 关键点
- `batch.batch_size` 在此上下文中是方法不是属性（`callable()` 检查处理两种情况）
- `try/except` + fallback 保护，出错时回退到原行为不会 crash
- `[:bs].clone()` 读 buffer 前 bs 行（indexer 写入位置），`clone()` 脱离原 buffer

## 部署步骤（1:1 复刻）

### 1. 部署修复到 1104
```bash
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
  -e "ssh -p 1104 -o ConnectTimeout=15" \
  python/sglang/srt/speculative/eagle_worker_v2.py \
  root@8.222.11.182:/root/v15_patched/lib/python3.12/site-packages/sglang/srt/speculative/eagle_worker_v2.py
```

### 2. 重启 1104 decode
```bash
ssh -p 1104 root@8.222.11.182 '
  # SIGTERM 优先，等 30s 再 SIGKILL
  pid=$(ps aux | grep "sglang[.]launch_server" | grep -v grep | awk "{print \$2}" | head -1)
  [ -n "$pid" ] && kill $pid
  for i in $(seq 1 30); do ! ps -p $pid >/dev/null 2>&1 && break; sleep 1; done
  ps -p $pid >/dev/null 2>&1 && kill -9 $pid; sleep 2
  # 清残留
  ps aux | grep -E "sglang|launch_server" | grep -v grep | awk "{print \$2}" | xargs -r kill -9 2>/dev/null; sleep 2
  # 清缓存
  find /root/v15_patched/lib/python3.12/site-packages/sglang/srt/ -name "__pycache__" -o -name "*.pyc" | xargs -r rm -rf
  # 确认部署
  grep -n "_bs = batch" /root/v15_patched/lib/python3.12/site-packages/sglang/srt/speculative/eagle_worker_v2.py
  # 启动
  cd /root && setsid nohup bash start_glm53_decode.sh </dev/null >/root/glm53_decode.log 2>&1 &
'
```

### 3. 等 decode 启动（~3 分钟）
```bash
for i in $(seq 1 24); do
  code=$(ssh -p 1104 root@8.222.11.182 'curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:30200/health')
  [ "$code" = 200 ] && echo "ready" && break; sleep 10
done
```

### 4. 重启 1102 router（让 router 发现新 decode worker）
```bash
ssh -p 1102 root@8.222.11.182 '
  lsof -ti :31000 | xargs -r kill -9 2>/dev/null; sleep 2
  nohup /usr/local/bin/smg launch --pd-disaggregation \
    --prefill http://10.0.58.35:30100 --decode http://10.0.58.37:30200 \
    --host 0.0.0.0 --port 31000 --api-key sk-glm52-pd \
    --policy cache_aware --max-concurrent-requests 64 \
    --health-check-timeout-secs 300 --disable-circuit-breaker \
    --request-timeout-secs 3600 --log-level info --prometheus-port 29004 \
    >/root/smg_31000.log 2>&1 </dev/null &
  sleep 3; curl -s -o /dev/null -w "router=%{http_code}" http://localhost:31000/health
'
```

### 5. 验证公网推理
```bash
curl --noproxy '*' -sS -m 30 -X POST http://8.222.11.182:31000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-glm52-pd' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"只回复OK"}],"max_tokens":8}'
```

### 6. 启动 bench 压测
```bash
nohup python3 /home/smith/.xbot/skills/bench-pd/assets/bench_ttft_tpot.py \
  --rpm 200 --endpoint http://8.222.11.182:31000/v1/chat/completions \
  --api-key sk-glm52-pd --out /tmp/bench_final.json --timeout 900 \
  --dump-dir /tmp/responses_final \
  --cases /home/smith/.xbot/skills/bench-pd/assets/cases_50.json \
  --abort-count 5 --abort-min-sec 60 --abort-max-sec 180 --seed 42 \
  > /tmp/bench_final.log 2>&1 &
```

### 7. 验证 accept rate
```bash
ssh -p 1104 root@8.222.11.182 '
  grep -a "Decode batch" /root/glm53_decode.log | grep -oE "accept rate: [0-9.]+" | awk -F": " "{print \$2}" | sort -n | head -5
  # <0.1 应为 0
  grep -a "Decode batch" /root/glm53_decode.log | grep -oE "accept rate: [0-9.]+" | awk -F": " "{print \$2}" | awk "\$1<0.1{c++}END{print c+0}"
  # DSA-SEED fallback 应为 0（修复生效）
  grep -c "DSA-SEED" /root/glm53_decode.log
'
```

## 日志保存到 CPFS
```bash
# 1104 上执行
ssh -p 1104 root@8.222.11.182 '
  mkdir -p /mnt/workspace/logs/glm53-accept-bug
  cp /root/glm53_decode.log /mnt/workspace/logs/glm53-accept-bug/decode_$(date +%Y%m%d_%H%M%S)_fix_deployed.log
'
```

## 验证结果（2026-09-05 11:36 UTC+8 部署后）
- 1184+ decode batches，<0.1 = 0 ✅
- <0.15 = 1（仅冷启动第一个 batch 0.13，KV transfer 刚完成）
- DSA-SEED fallback = 0（修复生效，无 fallback 触发）
- crashes = 0
- 稳态 accept rate = 0.20-0.38
- bench: http=200，公网推理正常

## 相关文件
- 修复: `python/sglang/srt/speculative/eagle_worker_v2.py` (~line 1035-1060)
- indexer 写入: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` (seed_buf[:bs])
- graph buffer: `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` (line 242, buffers.dsa_seed_topk_capture)
- Git commit: `5085112225` on branch `fix/eagle-seed-contract`
- MD5 (local == deployed): `f6d7c0a9ba407c7ed8e4809ef689903d`

## 1104 部署环境
- Python venv: `/root/v15_patched`
- sglang path: `/root/v15_patched/lib/python3.12/site-packages/sglang`
- 启动脚本: `/root/start_glm53_decode.sh`
- 日志: `/root/glm53_decode.log`
- 模型: `/root/glm53_local/`
