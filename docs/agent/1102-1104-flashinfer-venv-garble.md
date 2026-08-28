# 1102/1104 GLM-5.3 乱码根因与修复（2026-08-29，v15_patched flashinfer 泄漏）

## 现象

1102（prefill）+ 1104（decode）裸机 v15_patched 部署 GLM-5.3（mooncake RDMA + DCP=4 + EAGLE）：
**每次请求第一个 SSE 正常，然后卡半天，输出 token 汤乱码**（HTML/CSS/随机字符混合）。
decode 日志 accept rate **0.03-0.06**（病态，健康基准 0.24-0.43），gen throughput 43-49 tok/s。

## 根因（排除法 + 实锤）

**v15_patched venv 的 system_packages.pth 泄漏系统 flashinfer 0.6.12 给 venv flashinfer 0.6.14 → trtllm kernel 资源版本错配 → kernel 计算静默错乱**（AGENTS 已记载此坑的 IMA 崩溃形态，本次是静默错乱形态：首 token 对 → 后续 trtllm kernel 持续输出错 → draft 全拒 + 乱码）。

证据链：
1. `/root/v15_patched/lib/python3.12/site-packages/system_packages.pth` → 指向 `/usr/local/lib/python3.12/dist-packages`（系统包泄漏进 venv sys.path）
2. 系统装有 **flashinfer_python-0.6.12**（旧版）+ venv 是 **flashinfer_python-0.6.14**（`import flashinfer` 显示 0.6.14 但 cubin/JIT 资源查找会命中系统的 0.6.12）
3. `dsa_decode_backend='trtllm'`（DSA decode 走 flashinfer trtllm kernel）→ 版本错配的 kernel 资源 → 计算错乱
4. 1102 的 venv flashinfer `data` 目录只有 **75M**（AGENTS 记载完整应为 **88M**）→ cubin 资源缺失加剧 JIT 回退错配

**排除项**（全部实测）：
- srt 代码 13 文件 md5 全一致（dsa_indexer/dsa_topk_backend/eagle_worker_*/decode/utils/conn×2/memory_pool/kv_cache_configurator/model_runner/scheduler/schedule_batch = repo 最新，含全部乱码修复）
- `.pyc` 无旧字节码抢跑（.pyc 比 .py 新）
- L20D triton config 无残留
- RDMA 丢包计数 0
- decode radix（用户确认：decode 不开 radix 默认全量传，语义正确）

## 与 docker 部署（38.255.28.x，无乱码）的区别

| 项 | 1102/1104（乱码） | docker 部署（无乱码） |
|---|---|---|
| 部署 | 裸机 v15_patched venv | cv-prefill/cv-decode 镜像 |
| flashinfer | **系统 0.6.12 泄漏 + venv 0.6.14（错配）** | cubin 0.6.13 + python 0.6.14 配套干净 |
| venv data | 1102 只有 75M（缺 13M） | 完整 |

## 修复（AGENTS 记载的确切方案："删系统 cubin + 同步完整 88M data"）

1. 删系统 flashinfer（1102/1104）：`mv /usr/local/lib/python3.12/dist-packages/flashinfer* /root/backup_flashinfer_sys/`（备份保留）
2. 1102 的 venv data 补齐 75M→88M：从 1104（同版本 0.6.14）tar 传输替换 `flashinfer/data`
3. 双节点清 `__pycache__`（flashinfer + sglang srt）
4. 配对重启（提取原进程 cmdline+environ → kill → nohup 同命令重启，日志追加 /root/glm53_{prefill,decode}.log）

## 验证（重启后实测）

- HTTP 200，4.7s 完成（修复前"卡半天"）
- content 正常中文流畅输出（修复前 token 汤乱码）
- **accept rate 0.32-0.45**（修复前 0.03-0.06；AGENTS 健康基准 0.24-0.43 ✓）
- accept len 2.58-3.23（基准 2.2-3.2 ✓）
- gen throughput 103-130 tok/s（修复前 43-49，2.5×）

## 教训

- **裸机 venv 部署必须检查 system_packages.pth**（AGENTS 已有记载："v15_patched 的 system_packages.pth 会暴露系统 flashinfer_cubin 0.6.12"——之前是 IMA 崩溃形态，**本次是静默错乱形态：不崩溃但 kernel 计算错 → 乱码 + accept 崩**，更隐蔽）
- **乱码 + accept rate 病态（<0.1）的组合 → 先查 venv 依赖版本错配**（flashinfer/sgl-kernel），再查代码/传输
- 判据：`pip list`/`dist-info` 对比系统与 venv 的同名包版本；`du -sh venv/flashinfer/data` 应 88M
- AGENTS 该条坑需要补记：泄漏的另一种表现形态（静默乱码 vs IMA 崩溃）
