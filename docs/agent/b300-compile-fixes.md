# B300 集群编译/JIT 问题修复记录（2026-08-15）

三个独立的编译层问题，全部表现为 prefill crash → decode 报 `reconnect to 8998`（断连是后果非根因）。

## 1. DeepGEMM JIT：sm_103 缺 `a` 后缀

**症状**：16 并发触发新 shape → `tf32_hc_prenorm_gemm` JIT → ptxas 报 `Instruction 'tcgen05.fence' not supported on .target 'sm_103'` → 8 rank 齐崩。

**根因**：B300 真实 capability=(10,3)=SM103（device_name 伪装 "NVIDIA L20D"），DeepGEMM 拼 `--gpu-architecture=sm_103`（无 a）。tcgen05 指令需要 `sm_103a`。且 CUDA 13.2 的 `-arch=sm_103a` 短格式在 fatbin 模式下会丢 a 后缀，**必须用 `-gencode arch=compute_103a,code=sm_103a` 长格式**。

**修复**：直接替换 pip 包内 nvcc 为 wrapper（`nvidia/cu13/bin/nvcc` → 脚本，原文件改名 `nvcc.real`），拦截重写。注意：
- `DG_JIT_NVCC_COMPILER` env **无效**（DeepGEMM 0.1.4 不读它，实际从 `find_cuda_home()` → `CUDA_HOME/bin/nvcc` 找）
- 改 `CUDA_HOME` 会触发 flashinfer JIT 全量重编（崩溃），不能动
- 16 并发前最大安全 batch 是 768 token（1536 触发新 shape 编译）

## 2. tilelang JIT：并发编译竞争 → TVM C 层崩溃

**症状**：空闲后 16 并发同时到达 → 8 个 scheduler 进程同时 cache-miss → 同时 `tilelang.lower`/`BuildTileLangCUDA` → TVM C 层崩溃。

**根因**：`KernelCache` 只用 `threading.Lock`（进程内），8 个 scheduler 是独立进程——跨进程无锁 → staging 目录/disk cache 竞争。

**修复**：patch 两端 `tilelang/cache/kernel_cache.py`（备份 `.orig`）——cache-miss 编译段包 `fcntl.flock` 跨进程文件锁（`.compile.flock`）+ 锁内 double-check `_load_kernel_from_disk`（先到的编译，后到的直接加载）。

## 3. tilelang JIT：`cuda/atomic` 头文件缺失

**症状**：编译报 `fatal error: cuda/atomic: No such file or directory`。

**根因**：tilelang `lower.py` 只传 `-I TILELANG_TEMPLATE_PATH` 和 `-I CUTLASS_INCLUDE_DIR`，缺 CCCL include。CUDA 13.2 的 nvcc 不自带 CCCL（旧版自带）。

**修复**：patch 两端 `tilelang/engine/lower.py`——options 加 `"-I<venv>/lib/python3.12/site-packages/nvidia/cuda_cccl/include"`。

## 修复后验证

v39：72/72 全通过（含 16 并发 + 8 abort 洪峰 + 空闲 60s 后 16 并发 ×2 + 二轮 abort 洪峰），两端 0 crash。

## 注意

- 两个 tilelang patch 都是**对 pip 包内文件的原位修改**，`pip install --force-reinstall tilelang` 会打回原形——重装后必须重打 patch
- MHC prewarm（`SGLANG_DSV4_MHC_PREWARM`）当前设 0：flock 后预热并发安全，但该模型的 prewarm shape 编译报 RuntimeError（`_MSC_VER` 相关），保持关闭；运行期首 shape 编译由 flock 串行化，安全
