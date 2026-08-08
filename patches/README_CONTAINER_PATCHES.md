# 容器内修改的文件 (不在 sglang git 仓库中)

> 这些文件在 Docker 容器内修改，机器重启/容器删除后会丢失。
> 镜像 `test-mi300x-v3-cg:latest` 中的文件不会丢失（镜像层），只有容器层修改会丢失。

## 1. AITER rope.py — v17 clone patch

**文件路径** (容器内): `/opt/venv/lib/python3.10/site-packages/aiter/aiter/ops/rope.py`

**修改内容**: 8 个 rope 函数加 `positions.clone()` — 修复 DSA/CP/DCP 下 positions 2D 连续断言失败

**提取方式**:
```bash
ssh node-0 'sudo docker cp sglang-decode:/opt/venv/lib/python3.10/site-packages/aiter/aiter/ops/rope.py /tmp/rope_patched.py'
scp node-0:/tmp/rope_patched.py patches/rope_patched.py
```

**重新应用方式**:
```bash
scp patches/rope_patched.py node-0:/tmp/rope_patched.py
ssh node-0 'sudo docker cp /tmp/rope_patched.py sglang-decode:/opt/venv/lib/python3.10/site-packages/aiter/aiter/ops/rope.py'
# 同样给 prefill 容器
scp patches/rope_patched.py node-1:/tmp/rope_patched.py
ssh node-1 'sudo docker cp /tmp/rope_patched.py sglang-prefill:/opt/venv/lib/python3.10/site-packages/aiter/aiter/ops/rope.py'
```

## 2. triton 3.5.0 — 源码编译

**文件路径** (容器内): `/nvme/tmp/triton35-src/` (编译源码), `/opt/venv/lib/python3.10/site-packages/triton/` (安装)

**用途**: ROCm backend, `aiter_can_use_preshuffle_paged_mqa()=True` → DSA 用 paged-MQA path (page_size=64)

**重新编译方式**:
```bash
# 在容器内
git clone https://github.com/triton-lang/triton.git /nvme/tmp/triton35-src
cd /nvme/tmp/triton35-src
git checkout v3.5.0  # 或对应的 release tag
pip install -e .  # 源码编译, 需要 ROCm 环境
# 两节点 prefill + decode 都需要
```

## 3. triton_kernels v17-patched

**文件路径** (容器内): `/opt/venv/lib/python3.10/site-packages/triton_kernels/`

**用途**: 去掉 `constexpr_function` 装饰器, 兼容 triton 3.5+

**重新应用方式**: 从 v17-patched 镜像提取, 或从源码安装:
```bash
pip install triton-kernels==0.0.x  # 对应 v17 版本
# 然后手动去掉 constexpr_function 装饰器
```

## sglang 仓库内修改的文件 (已 git commit + push)

已 commit 到 `mi300x-pd-glm52` 分支 (commit 273896bdfc):

1. `python/sglang/srt/layers/dcp/comm.py` — dcp_enabled() 去掉 is_cuda() 检查
2. `python/sglang/srt/disaggregation/decode.py` — kv_transfer_page_size fix + 调试日志
3. `python/sglang/srt/disaggregation/mooncake/conn.py` — b300 官方 DCP reshard + 调试日志
4. `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` — DCP=8 跳过 assert page_size==1
5. `python/sglang/srt/layers/rotary_embedding/factory.py` — SGLANG_USE_AITER_ROPE 默认关
6. `python/sglang/srt/server_args.py` — dsa_backend 默认 tilelang

之前的 commit (457f11b467): aiter.py, topk.py, lora/layers.py 乱码修复
