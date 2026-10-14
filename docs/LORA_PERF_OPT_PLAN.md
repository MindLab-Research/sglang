# DSV4.1(mxfp4) 挂 LoRA 推理性能优化 · 大计划

> 目标口径：**生产口径**（decode CUDA graph 开、dspark 开、PD 分离、smg 唯一入口）。
> 度量口径：服务端 `gen throughput (token/s)`，**仅取 `#running-req >= 64`**，并**剔除每窗口前/后 15%**（爬坡/长尾）；
> 负载 `out=4096 / n=256 / conc=64 / ignore_eos / seed 777 / chat completions`；达成率 = 实测 ÷ (64×1/TPOT)。

---

## 0. 最终目标

在**不改变 MoE LoRA 结构**（A/B 形状、秩、目标模块、虚拟专家语义、量化格式、gating 数学全不变）的前提下，
显著压低 **DSV4.1 挂 LoRA 推理**的代价，并在**生产口径**下拿到可复现、可同批对照的收益。

**当前基线（同口径，dspark 开）**

| 组 | 满批 p50 (tok/s) | TPOT p50 (ms) | 样本 |
|---|---|---|---|
| base（不挂适配器） | **12356.4 / 12148.6** | 5.180 / 5.268 | 53 / 53 |
| `moe_rnd`（挂 MoE LoRA） | **7633.2 / 7874.5** | 8.384 / 8.127 | 37 / 41 |
| `moe_zero`（权重全零） | 8082.7 / 7885.3 | 7.918 / 8.116 | 43 / 43 |

⇒ 目标：把 `moe_rnd` 的 p50 从 **≈7.7k** 往 **≥9k**（即把 −38% 压到 ≤ −25%）推进；**若实测无收益则如实定案**。

---

## 1. 已确立的事实（含证据，勿重复验证）

| # | 事实 | 证据 |
|---|---|---|
| F1 | 代价**全部**在 mxfp4 MoE-LoRA 的 delta 执行路径上，**与权重数值无关** | `moe_zero ≈ moe_rnd`；「开 LoRA 本身≈0」（stock 无 LoRA base 12386/13027 ≈ LoRA base 12356/12149） |
| F2 | **没有更快的老路可退**：BYPASSED 不比物化快 | BYPASSED 5691.8/5699.8 < 物化 6000.1/6028.0（同 dspark 关） |
| F3 | fork 自带 **O1 双流重叠**（侧流跑 gate_up LoRA shrink+expand，与主流 quant 并行），有 fp8/fp4/bf16 三支，**原本缺 mxfp4 支** | `moe_overlap.py` 三支 + 我移植的 `..._mxfp4_lora_two_stream` |
| F4 | 双流 gate 有两道门：**① `has_active_lora`**（捕获期为 False）**② `SGLANG_TWO_STREAM_MAX_TOKENS`（默认 256 < dspark verify 批 64×6≈384 或捕获 bs=128×6=768）** | 诊断打印 `FALLBACK(virtual=True active=False std=True tokens=768)` |
| F5 | **decode 是 CUDA-graph replay** ⇒ 执行分支在**捕获期**决定一次 | fork 自述：`capture_mode.py:71`（捕获时无适配器但图必须记录 LoRA 路径）；双图捕获在本部署**本就开启**（`cuda_graph_setup.py:730`） |
| F6 | **已实现**「图内启用双流」的接线：gate 加 `or is_capturing_lora_variant()`（对齐 `_dual_topk`/`layers.py`/`lora_manager` 既有模式）+ 上限 1024 | 捕获期出现 `[TS-ACTIVE] mxfp4 two-stream ENTERED (tokens=768 rank=16)` ×4（此前恒 0） |
| F7 | **但进图即崩**：捕获期 `mhc_post` `CUDA error: unspecified launch failure`（+ 分配器 `insert_events` 断言） | 崩溃堆栈：`capture → run_once → deepseek_v4.py:2985 hc_post → mhc.py:1365` |
| F8 | 机制判断：侧流与主流的**汇合依赖 C++ trtllm 算子内部 `wait(lora_ready_event)`**，而 **mxfp4 trtllm-gen 算子未实现该跨流等待** ⇒ 竞态 → 非法访问 | 与 fork 自述同源：`trtllm_lora_temp/__init__.py` "sharing one side stream is a premature-reuse WAR -> IMA"；F7 的崩溃点在下游 layer 的 kernel |

---

## 2. 里程碑与 checklist

> 每个里程碑完成即：① 在本文档勾选并写结论/证据 ② `compact_context` 压缩 ③ 进入下一个。

### M0 · 准备与固化 ✅ 完成（2026-10-09）
- [x] 计划文档落库（本文件，commit `b5b1da41fe`）
- [x] 基线数字与复现命令固化（见 §0/§4）
- [x] 环境不变量固化（§3 红线 + §4 资产索引）
- [x] 「实现进图 → 崩」可稳定复现（两次复现：17:31:42 TP2 与 17:37 前后）
**验收**：✅ 文档命令可直用；崩溃可复现（`mhc_post` launch failure）。

### M1 · 崩溃机制精确定位 ✅ 完成（2026-10-09，含一次自我修正）

**先修正一个错误结论**：初次对比误判「mxfp4 双流缺 down 侧流事件链 ⇒ 崩溃」。
复核 fp4 同族原文发现 **`down_overlap = False` 是 fork 的显式设计**（`moe_overlap.py` fp4 段注释：
"the down/finalize overlap is bench-verified net-neutral-to-negative AND corrupts the base/decode path —
the captured gemm2_done cross-stream event under cuda-graph replay perturbs no-active-LoRA (base) requests
(qwen base gsm8k 0.81 -> 0.56 with it on; bisect-confirmed)"）。⇒ fp4 活动路径**只有 gate_up 一半**，
我的 mxfp4 移植与之结构相同 ⇒ **down 链缺失不是崩因**（M2 不做它）。

**已证事实（file:line）**：
- C++ 侧等待**已实现**：`kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu:3310`（`FP4BlockScaleLoraLauncher`）、`:3935`（`Bf16LoraLauncher`）：`if (lora_ready_event_ != 0) cudaStreamWaitEvent(...)`。
- Python op 包装**支持传事件**：`kernels/ops/moe/trtllm_lora_temp/core.py:230/330/408`（`lora_ready_event: int = 0`），285/375/497 传入。
- 双流 gate 原文（`srt/lora/trtllm_lora_temp/moe_overlap.py` mxfp4 函数）：
  `use_virtual_lora_store and lora_info.has_active_lora and is_standard and is_two_stream_active(hidden_states)`，
  其中 `is_two_stream_active(x) = x.shape[0] <= SGLANG_TWO_STREAM_MAX_TOKENS`（`environ.py:117`，**默认 256**）。
- `__init__.py:34` 注释自述："Per-batch gate (two-stream now always-on). True iff batch is decode-shaped (<= MAX_TOKENS)"。
- ⇒ **双流本来就装在现役配置里**（`launch_pd_smg.sh:12` 有 `SGLANG_LORA_TWO_STREAM=1`），
  只是 (a) dspark verify 批 `64×6=384 > 256` 被上限挡住；(b) **decode 走 CUDA-graph replay，分支在捕获期烙定，
  而捕获期 `has_active_lora=False`** ⇒ 图里永远是单流 fallback。

**结论**：崩点只可能来自**捕获期真正执行双流**（gate 认 `is_capturing_lora_variant()` 之后）。
M2 用**步进打印**（`[TS-STEP]`，env `SGLANG_TS_DIAG=1`）把崩点缩到具体一步。

**验收**：✅ 事实链完整；M2 判据 = decode 日志里 `[TS-STEP]` 的最后一行即为崩点。

### M2 · 定位崩点 + 实现修复（进行中）
- [x] 接线：gate 改为 `lora_ctx_active = bool(lora_info.has_active_lora) or is_capturing_lora_variant()`（对齐 `_dual_topk`/`layers.py`/`lora_manager` 既有约定）
- [x] 加 `[TS-STEP]` 步进打印（`pre-side-gateup` → `post-side-gateup` → `post-quant-prep` → `pre-op` → `post-op` → `post-down`），env `SGLANG_TS_DIAG=1` 开
- [x] 启动器临时加 `export SGLANG_TWO_STREAM_MAX_TOKENS=1024`（生产口径，decode CUDA graph **开**）
- [ ] **实验 E1**（进行中，task `8eca2101`）：重启看 `[TS-STEP]` 停在哪个标签 ⇒ 崩点
- [ ] **若捕获期崩点被异步 sticky 掩盖**（`insert_events`/`mhc_post` 只是最先看到的地方，不是 fault 点）
      ⇒ 走诊断 D2：eager（`--cuda-graph-backend-decode disabled`）+ `CUDA_LAUNCH_BLOCKING=1` + 双流 + 适配器
      ⇒ 拿到**精确算子**（启动器 `/root/exp_eager.sh block`；非阻塞版 `noblock` 同时可量双流的真实收益上限）
- [ ] 顺带确认：C++ 侧等待**已完备**（`FP4BlockScaleLoraLauncher` 就是 mxfp4/E4m3 路线，`:3310` 在 activation 前 `cudaStreamWaitEvent`；
      `core.py:408/497` 把 `lora_ready_event` 转发到 C++）⇒ 只要崩点不在算子内部，**就是纯 Python 可修**
- [x] **实验 E1 完成 → 根因锁定（决定性）**：`[TS-STEP]` 全量打印显示**每一行都是 `cap=False`**
      ⇒ **sglang 的图捕获用原生 stream capture，`torch.cuda.is_current_stream_capturing()` 在捕获期恒为 False**
      ⇒ fork 的事件保活守卫 `if torch.cuda.is_current_stream_capturing(): _LORA_OVERLAP_EVENTS.append(lora_event)`
      **永不成立** ⇒ `torch.cuda.Event` 出函数即 `cudaEventDestroy`，而捕获进图的 `cudaStreamWaitEvent` 仍引用它
      ⇒ 图实例化/replay 时 sticky `launch failure`，在**后续任意 allocator 调用**处爆发：
      `FP4BlockScaleLoraLauncher::run → TVMFFIEnvTensorAlloc → TorchDLPackExchangeAPI::ManagedTensorAllocator
      → MemoryError: CUDA error: unspecified launch failure`（`process_events @ CUDACachingAllocator.cpp:4408`）。
      与 fork 自述注释完全吻合（"Keep the event alive through cuda-graph capture so the captured wait inside
      the op isn't torn down before instantiation"）。
      ⚠️ 副作用发现：**fp8/fp4/bf16 三个同族也有同一潜在缺陷**（同一守卫写法）。
- [x] **FIX-1（纯 Python）**：保活条件加 `or is_capturing_lora_variant()`（捕获 lora 变体期必须保活）+
      `[TS-KEEPALIVE] events=…` 计数证据。
- [x] **同族同缺陷（跨模型价值）**：`moe_overlap.py` 里 fp8/fp4/bf16 三个 twin 的保活守卫写法完全相同
      （`:217/:332/:492/:585/:750`）⇒ 它们在图内同样保不住事件。若 mxfp4 修复收益成立，
      **对三个同族做同样一行改动**（GLM-5.3-flash 走的正是这几支之一）⇒ 可能是 GLM LoRA 的同一份红利。
- [ ] **后备方案**（若 FIX-1 不足以解除）：
      FIX-2 事件改为**模块级预建、只创建不销毁**（每调用新建但永驻，等价于 FIX-1 的强版）；
      FIX-3 若「图内根本不支持侧流」⇒ 退回 `lora_ready_event=0` + `current_stream().wait_stream(side_stream)`
      （安全但收益≈0，仅作对照）；FIX-4 若仍不行 ⇒ 记证据、转 M5 定案（并给出 kernel/捕获栈改动的具体方案）。
- [x] **FIX-1 被证伪（重要）**：加 `or is_capturing_lora_variant()` 后**仍崩**，且 `[TS-KEEPALIVE]` 计数 **=0**、
      `cap=False` 出现在**每一行**。结合 `full_cuda_graph_backend.py:140-151`（`capture_one` 先跑 **2 次 eager warmup**
      再进图捕获）⇒ **崩溃发生在 eager warmup，不是捕获、也不是事件生命周期**。
- [x] 据此推翻"事件悬空"假设，改为：**子代理移植的 mxfp4 双流本身从未成功执行过**（§9.2 的"正确性 OK/收益 0"
      是在 `ENTERED=0`（双流从未进入）情况下测的 ⇒ 结论无效）。已排除的可能因：
      VE gate_up 调用参数与能跑的单流版**逐字一致**（仅多 `intermediate_buffer`，形状 `(T,topk,r)` 正确）；
      算子标量参数（`output1/2_scales_*` 等）**两边同为 None**；pad 桥 `pad` vs `new_zeros+copy_` 语义等价。
- [ ] **D2 诊断（进行中，task `be4b147f`）**：eager（`--cuda-graph-backend-decode disabled`）+
      `CUDA_LAUNCH_BLOCKING=1` + 载入 `moe_rnd` + 发 1 个请求 ⇒ 阻塞模式可报**精确 fault 算子**
      （脚本 `/root/d2_eager.sh`；日志 `eager_decode.log`）。
- [x] **崩溃变量已收窄到唯一实参（机械对比）**：`sorted diff` 显示「能跑的单流 mxfp4 op 调用」与「崩溃的双流 op 调用」
      **唯一差异 = 新增 `lora_ready_event=lora_ready_handle`**（其余每个实参逐字相同，包括 `hidden_states=x_quant`/
      `hidden_states_scale=x_scale`/三个 `*_scales_*=None`/gated-silu 控制量/两个 bias）。
      ⇒ **崩溃由「传入非零 `lora_ready_event`」触发**，即 C++ `FP4BlockScaleLoraLauncher` 的
      `cudaStreamWaitEvent(stream, lora_ready_event_)` 这一步（`trtllm_fused_moe_kernel_launcher.cu:3310`）。
      ★ 注意 fp4 同族还额外传 `gemm2_done_event`（本移植未传，默认 0 ⇒ 无记录，符合预期）。
- [ ] **D2b 判定（进行中，task `fea394dd`）**：eager（**完全不进图捕获**）+`CUDA_LAUNCH_BLOCKING=1`+`moe_rnd`+1 请求
      ⇒ **若 eager 也崩**：跨流等待本身在 mxfp4 路线不可用（与捕获无关，属算子/参数层面）
      ⇒ **若 eager 正常**：问题专属于「图内」，即回到 kernel/捕获栈改动。
- [x] **崩溃 = 竞态（决定性反证）**：同配置 + `CUDA_LAUNCH_BLOCKING=1`（只改这一项）⇒ **服务完整启动并对外服务**
      （30100/30200/31500 全 200，"The server is fired up and ready to roll!"）。无 blocking 时必崩
      （`MemoryError: CUDA error: unspecified launch failure`，sticky error 在算子 allocator 处爆发）。
      ⇒ 不是结构性错误，是**跨流执行顺序/生命周期**问题。
- [x] **崩溃发生在 `capture_one` 的 2 次 eager warmup**（`full_cuda_graph_backend.py:140-151`），**不在图捕获区**：
      捕获用的是 `torch.cuda.graph(...)`（`:176-179`，会让 `is_current_stream_capturing()` 为真），
      而我们全部 `[TS-STEP]` 打印都是 `cap=False` ⇒ 双流跑在图上下文之外。
- [x] FIX-2 / FIX-2b（对侧流触达的 **7 类张量**逐一 `record_stream(side_stream)`）⇒ **仍崩** ⇒ 排除「caching
      allocator 提前回收侧流缓冲」这一假说。
- [ ] **Bisect A（进行中，task `45b10d99`）**：在算子前加 `torch.cuda.current_stream().wait_stream(side_stream)`
      （**完全 join 侧流**，算子内的 C++ `lora_ready_event` 等待保持不变）
      ⇒ 若**不再崩**：说明 C++ 中途 `cudaStreamWaitEvent` 在该路线**并未真正保护** delta 的消费者
      （→ 这是唯一变量 `lora_ready_event` 的直接证据，也解释了为何单流安全、双流不安全）；
      ⇒ 若**仍崩**：说明侧流本身在该算子上不可用（与顺序无关），需退回 kernel 侧改动。
- [x] **Bisect A/B/B'/C 全部失败（重要）** ⇒ 崩因**与并发无关、与事件无关、与那两条自加机制无关**：
      A 在算子前完全 join 侧流 → 崩；B 把 VE 调用放回主流（无并发、保留数据通路）→ 崩；
      B' 再把 `lora_ready_event` 强制为 0（完全没有 C++ 等待）→ 崩；C 再去掉 routing 预热与
      `intermediate_buffer` → 崩（仅进度推进到 1/35）。**+CUDA_LAUNCH_BLOCKING=1 始终可启服** ⇒ 竞态确实存在，
      但不在我逐项排除过的这些位置。
- [x] **方法论对照 Bisect D（决定性）**：把 mxfp4 双流 override 的函数体整体换成**直接委派原生单流函数**
      （`return get_original_mxfp4_moe_lora_func()(...)`）⇒ **完整启服并对外服务**（30100/30200/31500 全 200，
      "The server is fired up and ready to roll!"，`Engine startup timings ... cuda_graph={...}` 全过）
      ⇒ **经该 override 调度本身没有副作用；缺陷完全在子代理移植的那个函数体内**。
- [ ] **改用「基底重建」而非继续二分（新方案 M2-v2）**：以**能跑的单流 mxfp4 函数体为基底**逐字复制为
      `..._mxfp4_lora_two_stream`（去掉签名注解、补齐该函数体依赖的模块级名字
      `use_symmetric_memory` / `is_allocation_symmetric` / `fused_experts_none_to_flashinfer_trtllm` 等），
      **只增量加入侧流改写**（VE gate_up 上侧流 + 事件 + C++ 中途等待），每加一处就重启验证一次。
      理由：子代理版本与能跑版本之间仍有未被 diff 捕获的行为差异，继续二分成本高于重建。
- [x] **M2-v2 基底重建 · 工具链修正（重要教训）**：第一次 STEP2/STEP3 构建**都因我的补丁缺陷没真正执行**
      （`NameError: name 'get_parallel' is not defined` —— 能跑的单流函数体依赖 `lora_dispatch.py` 的模块级
      `get_parallel` 等名字）。且我原来的 `import` 自检**抓不到运行期缺名**（只在函数被调用时才炸），
      导致我一度把**上一轮 Bisect D（委派）的"fired up"误判为 STEP 2 的成功**（已更正）。
      ⇒ 修正：① 构建器**重放 `lora_dispatch.py` 的全部模块级 import** 到新函数内；
      ② 加 **AST 未定义名静态检查**（模块级名字 ∪ 函数局部 ∪ builtins），有未定义名**直接拒绝启动**。
- [ ] **M2-v2 STEP 2（重跑中，task `5b982b39`）**：能跑基底 + VE gate_up 上侧流 + `lora_event.record()`
      + 主流在 pad 桥前 `wait_stream(side)`（`lora_ready_event=0`，暂不用 C++ 中途等待）
      ⇒ 启服 ⇒ 「能跑基底 + 侧流 VE + 事件」安全（崩因收窄到 pad 桥 copy_ 在侧流 / 非零事件）；
      ⇒ 崩 ⇒ 侧流 VE 本身有问题。
- [x] **M2-v2 STEP 2 ✅ 完整启服（决定性）**：以**能跑的单流函数体逐字为基底**（重放其 7 条模块级 import、
      跳过 `__future__`，经 **py_compile + AST 未定义名 + 真实 import 三道闸门**），只把 **VE gate_up 调用**
      移入 `with torch.cuda.stream(_side)` + `lora_event.record()`，并在 pad 桥前 `current.wait_stream(_side)`
      （**`lora_ready_event=0`**）⇒ **30100/30200/31500 全 200，"The server is fired up and ready to roll!"，
      捕获 35/35 全过、零 Traceback**。
      ⇒ 结论：**「能跑基底 + 侧流 VE + 事件 + 主流 join」是安全的**；崩因收窄到二选一：
      **(a) pad 桥 `copy_` 在侧流上下文内执行**，或 **(b) 非零 `lora_ready_event`（C++ 中途等待）**。
- [ ] **M2-v2 STEP 3（运行中，task `f302645f`）**：移植版完整设计 = 桥 `copy_` 留在侧流上下文 + 恢复
      `lora_ready_event=lora_ready_handle` + 去掉 join ⇒ 崩 ⇒ 在 (a)/(b) 中定位（再跑 STEP 4/5 二分）；
      启服 ⇒ 移植版设计本身可行，需再找我漏掉的差异。
- [x] **M2-v2 STEP 3 ✅ 完整启服（移植版完整设计，重建版）+ 稳定性 2/2**：桥 `copy_` 留在侧流上下文 +
      `lora_ready_event=lora_ready_handle`（非零）+ 无 join ⇒ **启服**（21:06:18 / 21:11:57 两次，
      三端全 200、35/35 捕获全过、零 Traceback）；`[TS-ACTIVE] rebased mxfp4 two-stream ENTERED tokens=768 rank=16`
      ×4 ⇒ **探针确认侧流路径确实被执行**（非 fallback）。
- [x] **结论（M2 定案）**：原「子代理移植版必崩」**不是**并发/事件/那两条自加机制的问题，而是**移植函数体自身的缺陷**
      （证据链：A/B/B'/C 四种简化全崩、而 D=委派原生单流启服 ⇒ 调度路径无辜）。**修复方式 = 弃用移植体、以能跑的单流
      函数体为基底重建**（只增量加侧流），STEP2/STEP3 两层都验证启服。
- [x] **工具链沉淀（可复用）**：`/root/m2v2_build3.py`（基底重建 + 三道闸门：`py_compile` → AST 未定义名 →
      真实 import；`from __future__` 必须过滤否则函数内 SyntaxError）、`/root/m2v2_check.py`（独立未定义名检查）、
      `/root/m2v2_trace.py`（ENTERED 探针）。
      ★ 教训：**只做 `import` 自检抓不到运行期 `NameError`**（如 `get_parallel` 只在函数被调用时才炸），
      必须做 AST 作用域级未定义名检查；**每次改完先过闸门再启动**（一次空跑 = 4~5 分钟）。
- [ ] 按崩点修（候选项：① 侧流块里的 pad 桥 `copy_`（fp4 无此步，最可疑）；② 侧流临时分配；③ C++ `cudaStreamWaitEvent` 在捕获期被 `_LORA_OVERLAP_EVENTS` 之外的生命周期破坏）
- [ ] 若判定「图内双流不可为」⇒ 记录证据，转 M5 如实定案（并给出 kernel 侧改动的具体方案）
**验收**：能指名崩在哪一步；修复后 `ENTERED` 在捕获期出现且**不崩**；
若不可修，给出「为什么不可为」的机制证据（而非"试了不行"）。

### M3 · 正确性与稳定性验证
- [x] **稳定性**：双流配置 **2/2 重启全过**（21:06:18、21:11:57 两次 fired up；三端 200/200/200；捕获 35/35；零 Traceback），
      `[TS-ACTIVE] rebased mxfp4 two-stream ENTERED tokens=768 rank=16` ×4 ⇒ 路径确认非 fallback
- [x] **判据失效（本轮关键负面结论）**：`m3_dump.py` **同配置自身即不可复现**——单流配置连跑 N=20 次，
      p0~p3 **20/20 全不同**（modal share 仅 1/20），最大公共前缀仅 1~15 / 60 字符；p4 19/20。
      ⇒ 本栈（dspark 投机 + MXFP4 trtllm-gen 非确定性归约 + 前缀缓存/批组成）上**「逐字一致」不可达**；
      故「双流 vs 单流 5/5 不同」**与同配置噪声同量级，不能归因于双流**（`cmp_m3.py` 逐叶子比对）。
- [ ] **需要的正确判据（未做，列为后续）**：① 关 dspark + `--disable-radix-cache` + 单请求 batch=1 的确定性 dump；
      ② 或 logit 级对比（teacher-forcing，同前缀比较 top-k / 相对误差）——**在此之前不得宣称数值等价**
**验收**：PASS（稳定性）；数值等价**判为「证据不足」**并给出构造确定性 oracle 的具体方案。

### M4 · 性能实测（同口径 + 同批对照）✅ 完成
同一份代码、只切 `SGLANG_LORA_TWO_STREAM`（1=双流 / 0=原生单流），服务端 64 打满稳态、剔除前后 15%（`steady3.py`）：

| 组 | 满批 p50 (tok/s) | 满批 p90 | TPOT p50 (ms) | TPOT p90 | 样本 | 达成率 |
|---|---|---|---|---|---|---|
| `two_base` | 12113.1 | 13677.8 | 5.462 | 6.290 | 44 | 0.97 |
| `single_base` | 12244.3 | 13875.4 | 5.233 | 6.519 | 46 | 1.00 |
| **`two_rnd`** | **7923.9** | 8556.2 | **8.077** | 8.718 | 33 | 1.00 |
| **`single_rnd`** | **7739.5** | 8394.1 | **8.269** | 9.422 | 43 | 1.00 |

- base 口径确认 ✓（12113 / 12244，与历史 12356/12149 同量级）
- **双流净收益：moe_rnd 7923.9 vs 7739.5 = +2.38%**（TPOT −2.3%）；base 端 −1.07%（噪声内，符合预期：base 走 nolora 图不受影响）
- 与历史基线（单流 7633.2/7874.5）对照，本次单流 7739.5 亦在同量级 ⇒ **结论：收益仅 ~2.4%，属噪声边缘的小幅正收益，绝非数量级改善**
**验收**：原始数据表 ✓；结论明确 ✓（**无显著收益**）。

### M5 · 沉淀与交付 ✅ 完成
- [x] **如实定案（详见 §6 M5 判决）**：双流**能跑且稳定**，但**净收益仅 +2.4%**，**远不足以回收 −38%** ⇒
      原假设「代价在 gate_up delta 的串行执行、可用旁流重叠」**被证伪**；**默认不开启**（收益 ≪ 复杂度/风险 + 数值无法验证）。
- [x] 交付：本计划文档（含 §6 判决 + 复现步骤）+ 工具链入库 `tools/dsv41_two_stream/`（重建器 / 三道闸门 / dump 噪声统计 / dump 比对）
- [ ] 推 `pr17`（本轮 docs + tools 提交已 push）；release 分支与 tag：**不因「无显著收益」而创建**（避免给未验证的调度改动留下发布痕迹）
**验收**：判决有完整证据链 ✓；工作区干净 ✓；环境恢复（见下）。

---

## 3. 红线（违反即作废）

1. **不改 LoRA 数学/结构**：形状、秩、目标模块、虚拟专家语义、量化格式（E4m3 激活 / UE8M0 SF / OAI gated-silu α=1.702 β=1.0 clamp=7.0）全部保持；只允许改**调度与同步**。
2. **不为了变快而跳步**：不许省 delta、降精度、改 batch/并发。
3. **不编数据**：批间波动 4–15%，**必须同批对照**；达不到显著就如实写"无显著收益"。
4. **操作纪律**：精确 PID kill（禁宽匹配，会自杀）；`rm -rf` 被守卫拦 → `mv` 到 `/root/_scratch_trash`；禁 foreground sleep（后台任务 + `task_status`）；`bench_chat3.py` 必带 `--timeout` 防挂死。
5. **目录隔离**：改动只在 `/root/dsv41_upstream`；**GLM 树 `/root/glm53_mol` 一个字不许动**。

---

## 4. 关键资产索引

| 资产 | 路径 |
|---|---|
| 服务 | P=30100 / D=30200 / **smg=31500（唯一入口，api-key 在 `/root/dsv41_upstream/smg_api_key.txt`）**；LoRA 管理端点**直打 30100/30200**（经 smg 404） |
| 启动器 | `/root/dsv41_upstream/launch_pd_smg.sh`（基线口径，带 decode CUDA graph） |
| 压测/分析 | `/root/bench_chat3.py`、`/root/steady3.py` |
| 适配器 | `/root/dsv41_moe_rnd`、`/root/dsv41_moe_zero`（层 20–39） |
| 关键代码 | `srt/lora/trtllm_lora_temp/moe_overlap.py`（双流）、`lora_dispatch.py`（mxfp4 单流入口，~730-929）、`runner_utils/capture_mode.py`、`runner/decode_cuda_graph_runner.py`、`kernels/ops/moe/trtllm_lora_temp/**` |
| 诊断开关 | `SGLANG_TS_DIAG=1`（打印 `[TS-ACTIVE] ... ENTERED/FALLBACK`） |
| 备份 | `moe_overlap.py.bak-lora-variant-gate`（本轮接线前） |

---

## 5. 进度日志

| 时间 | 里程碑 | 状态 | 结论/证据 |
|---|---|---|---|
| 2026-10-09 | M0 | ✅ 完成 | 计划文档 + 基线/红线/资产固化；崩溃可复现 |
| 2026-10-09 | M1 | ✅ 完成（含修正） | 修正：fp4 本就 `down_overlap=False`（设计禁用）⇒ 缺 down 链**非**崩因。真因=捕获期双流；gate=`has_active_lora` ∧ `tokens<=256` |
| 2026-10-09 | M2 | 进行中 | **Bisect A/B/B'/C 全败 ⇒ 与并发/事件/自加机制无关；Bisect D(委派原生单流) 完整启服 ⇒ 缺陷在移植函数体内**。改用基底重建（M2-v2） |
| 2026-10-09 | M2 | 进行中 | **竞态已证**：+`CUDA_LAUNCH_BLOCKING=1` ⇒ 完整启服并服务；崩点在 `capture_one` 的 eager warmup（非捕获区）。FIX-2/2b（record_stream）失败 ⇒ 非回收假说 |
| 2026-10-09 | M2 | 进行中 | **机械对比锁定唯一变量**：崩溃 ⇔ 传入非零 `lora_ready_event`（其余实参逐字相同）|
| 2026-10-09 | M2 | 进行中 | **FIX-1 证伪**：仍崩且 `KEEP=0`/`cap=False` 全量 ⇒ 崩在 **eager warmup** ⇒ 不是事件问题，是**移植版双流本身有 bug**（该路径从未成功跑过）。转 D2（eager+LAUNCH_BLOCKING）定位精确算子 |
| 2026-10-09 | M2 | 进行中 | **根因锁定**：捕获期 `is_current_stream_capturing()==False`（原生捕获）⇒ 事件保活从未触发 ⇒ Event 被销毁、图内 wait 悬空 ⇒ launch failure。FIX-1 已落，重跑验证中 |
| 2026-10-09 | M2 | 进行中 | 补丁踩坑：作用域裁剪后误写回 `s`（仅函数体）⇒ 文件被截断、`ImportError: ...bf16_lora_two_stream`；已从 `.bak-m2-trace` 恢复并改为写 `_head + s`（教训：**作用域替换必须回写 head+body**） |
| 2026-10-09 | M2 | 进行中 | 接线 + `[TS-STEP]` 埋点 + TOKENS=1024 → 重启实验 E1（task `8eca2101`） | **根因=mxfp4 双流缺 down 侧流事件链**（`gemm2_done_event`/`shrink_done_event` 计数=0；fp4 有）⇒ 修法纯 Python |
| 2026-10-09 | M2 | ✅ 完成 | **定案**：崩因 = 子代理移植体自身缺陷（A/B/B'/C 四种简化全崩 vs Bisect D 委派原生单流**启服** ⇒ 调度路径无辜）。修复 = **弃用移植体、以能跑的单流函数体为基底重建**（重放模块级 import、跳过 `__future__`；三道闸门）。STEP2/STEP3 均**完整启服**、ENTERED 探针确认路径、**稳定性 2/2** |
| 2026-10-09 | M3 | 部分完成 | 稳定性 2/2 ✓；**数值等价判据失效**：dump 同配置 N=20 即 20/20 全不同（噪声底饱和）⇒ 双流 vs 单流 5/5 不同**不可归因**；需确定性 oracle（关 dspark/关 radix/单请求 或 logit 级） |
| 2026-10-09 | M4 | ✅ 完成 | 同批 A/B：`two_rnd 7923.9` vs `single_rnd 7739.5` ⇒ **+2.38%**；base 12113.1 vs 12244.3（−1.07%，噪声内）⇒ **无显著收益** |
| 2026-10-09 | M5 | ✅ 完成 | 如实定案：**双流默认不开启**；−38% 的真实位置在分解链**串行依赖**，下一步应在 **kernel 侧**（LoRA delta 融进 activation/quant 融合核）而非调度侧；工具链入库 |

---

## 6. M5 判决（如实定案，2026-10-09）

### 6.1 本轮真正拿到的成果
1. **双流从「必崩」到「稳定可跑」**：根因是子代理移植体自身缺陷（不是并发/事件/生命周期）；
   修复方式是**以已知能跑的单流 mxfp4 函数体为基底重建**，只增量加侧流改写。
   证据链：`Bisect A(join)/B(串行)/B'(event=0)/C(去自加机制)` **四种简化全崩**，而
   `Bisect D(委派原生单流)` 与重建版 `STEP2/STEP3` **全部完整启服**。
2. **可复用工具链**（本次最大工程资产）：基底重建器 + **三道闸门**（`py_compile` → AST 作用域未定义名 → 真实 import）
   + dump 噪声统计 + 结构无关 dump 比对。**教训**：只做 `import` 自检抓不到运行期 `NameError`
   （`get_parallel`）；`from __future__` 重放进函数体是语法错误；作用域替换必须回写 head+body。
3. **性能结论（同批 A/B，唯一的可信数字）**：双流对 LoRA 请求 **+2.38%**（7923.9 vs 7739.5 tok/s）。

### 6.2 为什么「此路不通（只值 2.4%）」
- 原假设：−38% 的代价在 **gate_up LoRA delta 的串行执行**，可用侧流与主流的 quant+GEMM1 重叠。
- 实测：把 gate_up shrink/expand + pad 桥搬到侧流并把 C++ 中途等待接上后，**只回收 2.4%** ⇒ 假设被证伪。
- 机制推断（与 GLM 报告 §12 的结论一致）：分解链的真实成本在**串行依赖**——
  `permute → GEMM1 → activation(必须读 delta) → MXFP8 quant → GEMM2 → finalize`，
  其中 activation 对 delta 的依赖使「旁流重叠」只能覆盖很窄的一段；且分解路径本身比 stock 单 cubin 多出的
  若干次 kernel 往返（20 层 × 384 专家 top-6 的小 GEMM 发射）无法靠调度隐藏。
- **方向结论**：要真正吃掉 −38%，必须走 **kernel 侧**——把 LoRA delta 融进 **activation/quant 融合核**
  （AGENTS.md #15 的 `fused_activation_mxfp8_quant` 已具备把 delta 加在 pre-activation 的能力），
  或把 GEMM1+delta+act+quant 融成一个核，而不是在调度层做重叠。

### 6.3 风险与限制（诚实记录）
- **数值等价未验证**：本栈 greedy 解码 run-to-run 不可复现（同配置 N=20 → 20/20 不同），
  因此**不能宣称**双流与单流数值一致，也**不能据此否决**；必须先用确定性 oracle 复测。
- 双流仅在 **tokens ≤ 256 的 decode 形状**、且 **LoRA 生效** 时才有意义；本项目 LoRA 只作用于后 20 层，故无 KV 隔离需求。
- 收益 2.4% 处于批间波动（报告 §3 红线：批间 4–15%）的下沿 ⇒ 只能定为「**小幅正收益、非数量级改善**」。

### 6.4 复现（本轮全部结论）
```bash
# 环境：1101，代码树 /root/dsv41_upstream/sglang（PYTHONPATH=/root/dsv41_upstream），venv /opt/sglang-venv
# 1) 基底重建（含三道闸门，READY 才启动）
/opt/sglang-venv/bin/python /root/m2v2_build3.py step3      # 移植版完整设计
/opt/sglang-venv/bin/python /root/m2v2_build3.py step2      # 仅侧流 VE + join（对照）
bash /root/exp_m2.sh          # 双流开（SGLANG_LORA_TWO_STREAM=1）
bash /root/exp_single.sh      # 单流基线（=0）
# 2) 就绪等待 + dump + bench
bash /root/wait_ready.sh && bash /root/m3.sh two && bash /root/m4.sh two
# 3) 噪声底（判据有效性自检，必做）
/opt/sglang-venv/bin/python /root/m3noise.py single 20 moe_rnd
/opt/sglang-venv/bin/python /root/cmp_m3.py /tmp/m3_two.json /tmp/m3_single.json
```

---

## 7. 融合路线（F1–F3，2026-10-10）——证据与结论

### 7.1 F1 消融归因（数值无损）：**LoRA delta 内核成本 = 0**
把 `has_active_lora` 置 False（env `SGLANG_MOE_LORA_SKIP_DELTA=1`）⇒ 走**零 delta 通路**（与 `moe_zero` 数值等价），
只跳过 `merged_experts_fused_moe_lora_add`（gate_up shrink/expand + down delta）：

| 组（服务端 64 打满稳态，剔除前后 15%） | 满批 p50 (tok/s) | TPOT p50 (ms) |
|---|---|---|
| `f1_moe_zero`（跳过 delta 内核） | 7816.4 | 8.252 |
| `f1_moe_rnd`（跳过 delta 内核） | **7934.2** | 8.066 |
| `single_rnd`（delta 内核照跑） | 7739.5 | 8.269 |
| `two_rnd`（双流，delta 照跑） | 7923.9 | 8.077 |
| `base`（**stock 融合核**，无适配器） | 12244.3 | 5.233 |

⇒ **跳过 delta 内核没有任何收益**（7934 甚至略高于正常 7739/7924，全在批间噪声内）
⇒ **「把 LoRA delta 融进 activation/quant 融合核」这条路线的天花板 ≈ 0**（本文件 §6 我自己的建议被本次消融**证伪**）。
⇒ **−35% 的主体在「分解链路形态」本身**：`base` 用 stock 单融合核 12244，而分解链（routing→permute→GEMM1→act+quant→GEMM2→finalize）无论有没有 delta 都只有 ~7800-7900。

### 7.2 F2 cubin 供给盘点：**4-bit 激活的 MX GEMM 根本不存在**
`/tmp/trtllm/.../trtllmGen_bmm_export/cubins/`（**9499** 个）+ `/root/.cache/flashinfer/cubins/<hash>/batched_gemm-*/checksums.txt`：

| dtype 组合 | 个数 |
|---|---|
| `Bmm_E2m1_E2m1E2m1`（nvfp4 形态，4bit×4bit） | 476 |
| `Bmm_E4m3_E4m3E4m3` | 188 |
| `Bmm_MxE4m3_MxE2m1MxE4m3`（**MX：8bit A × 4bit B**） | 168 |
| `Bmm_Bfloat16_MxE2m1*` 等 | 144 / 106 / … |
| **`Bmm_MxE2m1_MxE2m1`（MX 4bit A × 4bit B）** | **0** |

⇒ V4.1 权重是 **MX（UE8M0 SF）** 布局，而 MX 布局下**没有** 4-bit 激活的 A 侧变体
⇒ 分解链**被迫**用 `MxE4m3`（8-bit）激活 ⇒ 多一条 E4m3 激活量化链 + GEMM1 的 A 侧带宽翻倍。
这**不是实现缺陷，是闭源 cubin 供给约束**（也解释了为什么 stock 融合核更快：它用 4-bit 形态）。
⇒ 「切回 E2m1 以省掉 E4m3 量化链」这条路**被证据关闭**（不存在该 cubin）。

### 7.3 F3 可融项盘点与天花板（本侧可动的边界）
| 可融项 | 归属 | 可动 | 预估收益 |
|---|---|---|---|
| GEMM1 epilogue × activation/quant（stock 的做法） | **闭源 cubin 内部** | ❌ | 大（−35% 主体） |
| GEMM2 × finalize | **闭源 cubin 内部** | ❌ | 中 |
| gate_up LoRA delta → act/quant 核 | 本侧 ✓ | ✓ | **≈0（F1 实测）** |
| down LoRA delta → finalize | 本侧 ✓ | ✓ | ≈0（同上） |
| permute(E4m3) 与激活量化融合（省 23.6MB/层 往返） | 本侧 ✓ | ✓ | ~1-2% |
| routing/align 元数据核合并 | 本侧 ✓ | ✓ | ~1% |
| 中间张量 bf16 → fp8 降带宽 | 依赖闭源 GEMM 的 epilogue dtype | ❌ | — |
⇒ **本侧可融项合计 ≈ 2-3%**（相对 −35% 是零头），**且实现风险（改 op 接口 + 内核）高于收益**。

### 7.4 真正能回收 −35% 的两条路（均需决策，非纯工程）
1. **把适配器折进权重（rank-16 权重修正）**：LoRA 的贡献在数学上等价于
   `W1' = W1 + B_e A_e`（gate_up）/ `W2' = W2 + B_e A_e`（down），**逐专家、逐适配器一次性预计算**
   ⇒ 之后 LoRA 请求可走 **stock 融合核（实测 12244 = 上限 +56%）**。
   代价/红线冲突：① **违反本文件 §3 红线 1**（改 LoRA 结构：适配器不再是运行时模块、切适配器需重融）；
   ② 权重需**重新量化到 MXFP4**，而 4-bit 量化噪声按 `|W|` 量级给出 ⇒ **小 delta（真实训练的适配器）会被舍入噪声淹没**
   —— 现有「分解链在 bf16/fp32 里加 delta」正是为保留 delta 保真度而付的 −35%。
   ⇒ 若产品接受「delta 保真度下降 + 不支持按请求热切适配器」，这是唯一能吃掉 −35% 的路线（需新存储/加载设计）。
2. **向上游要 cubin**：`Bmm_MxE2m1_MxE2m1`（MX 4-bit 激活）或带 **LoRA delta 钩子**的融合核（把 gate_up delta 加在激活前）。
   这是 NVIDIA 侧供给问题；有该 cubin 则本侧无需分解链，直接回到 base 速度。

### 7.5 复现
```bash
# F1 消融（数值无损：退化为零 delta 通路）
/opt/sglang-venv/bin/python /root/f1_ablate.py            # 插入 env 开关
sed '1a export SGLANG_MOE_LORA_SKIP_DELTA=1' /root/exp_single.sh > /root/exp_ablate.sh
bash /root/exp_ablate.sh && bash /root/wait_ready.sh && bash /root/f1.sh
# F2 盘点
ls /tmp/trtllm/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/cubins \
  | sed -E 's/^(Bmm_[A-Za-z0-9]+_[A-Za-z0-9]+)_.*/\1/' | sort | uniq -c | sort -rn
```

---

## 8. 口径澄清：三个"看起来矛盾"的结论的适用范围（2026-10-10，用户质询后补充）

用户质询：「之前说 MoL 下走 base 几乎没有性能损耗，又说性能几乎全是分解路径消耗的——矛盾吗？」
答：**不矛盾，但早期有一条归因错了**。三个结论的适用对象不同，其中第 2 条已被 F1 实测**推翻**：

| # | 结论 | 适用对象 | 状态 | 证据（服务端 64 打满稳态，剔除前后 15%） |
|---|---|---|---|---|
| A | 「开 LoRA 本身 ≈ 0」 | **非适配器请求**（payload 不带 `lora_path`）跑在 LoRA 服务器上 | ✅ **成立** | stock 服务器 base 12386/13027 ≈ LoRA 服务器 base 12356/12149；本轮同批复核：`two_base` 12113.1 / `single_base` 12244.3（两服务器都已加载 adapter） |
| B | 「−38% 全在 mxfp4 MoE-LoRA **delta 执行**（小 GEMM 发射受限）」 | 适配器请求 | ❌ **已被推翻**（旧归因，错误） | F1 消融：跳过 delta 内核 `f1_moe_rnd` **7934.2** vs delta 照跑 `single_rnd` 7739.5 / `two_rnd` 7923.9 ⇒ **delta 内核成本 ≈ 0** |
| C | 「−35% 全在**分解链路形态**本身」 | 适配器请求 | ✅ **成立（本轮定案）** | `base`（stock 单融合核）12244 vs 适配器请求 ~7800-7900，**有无 delta 都一样** |

### 为什么 A 与 C 并存却不矛盾
- A 说的是**服务器开了 LoRA 这个"能力"**：不带 `lora_path` 的请求**根本不进** LoRA 分支，仍走 **stock 融合核**
  （`lora_dispatch.py:117/406/598` 的 `not lora_info.has_active_lora` 早退；已实测该路径 ≈ base 速度）。
- C 说的是**带着适配器的请求**：它**必须**进分解链 ⇒ 形态代价 −35%。
- ⇒ 二者是**不同请求类**的结论；把 A 读成「LoRA 请求不损失性能」才是误读（A 从来没这个意思，但措辞容易误读，已在此澄清）。

### C 的"铁证"：代价与 LoRA 数学无关，只与「是否声明适配器」有关
`moe_zero` 适配器的 **60 个 `lora_B` 张量 max|B| = 0.0**（已核验）⇒ 其 LoRA 贡献**恒等于 0**，
与 pass-through 数值等价；但它实测 **7885.3 / 8082.7**（与 `moe_rnd` 7739.5-7924 同档）
⇒ **只要请求声明了适配器，就付掉 −35%，哪怕 delta 恒为 0**。

### 为什么"声明适配器"就必须付这笔钱（结构性，非缺陷）
1. LoRA delta 必须加在 **activation 之前**（gate_up）/ finalize 之后（down）；
2. stock 路径是**闭源 flashinfer/trtllm-gen 单融合 cubin**（routing→GEMM1→act→quant→GEMM2→finalize 全在一个核内），
   **没有 LoRA 钩子**（AGENTS.md #9：non-LoRA 路径无 MoeRunner ⇒ 无钩子），无法在其内部插入 delta；
3. ⇒ 只能把链路**拆开**成我们自己的核（分解链），并被迫接受：材质化 topk（`_lora_forces_standard_topk`）、
   **MxE4m3 8-bit 激活**（F2：9499 个 cubin 里 `Bmm_MxE2m1_MxE2m1` = 0 ⇒ MX 布局没有 4-bit 激活变体）、
   额外的 permute/中间张量往返。

⇒ **一句话**：A（非适配器请求 ≈ 0 损耗）与 C（适配器请求 −35% 且与 delta 数学无关）**同时成立**；
被推翻的是 B（旧归因"delta 执行"）。真正要回收 C，只能在**权重侧折适配器**或**上游给出带 LoRA 钩子的融合核**（见 §7.4）。

---

## 9. 显存预算：多存一份「后 20 层 MoE 权重」要多少（2026-10-10，用户质询）

测量方式：直接读 `/mnt/workspace/DeepSeek-V4.1-Flash` 的 safetensors 头（不加载数据）。工具 `tools/dsv41_two_stream/moe_footprint.py`。
**关键事实**：专家权重盘上就是 **FP4 打包**（`dtype=I8`，2 值/字节）+ **UE8M0 分块 scale**：

| 张量 | 形状 | dtype | 字节 |
|---|---|---|---|
| `layers.20.ffn.experts.0.w1.weight`（gate） | (2304, **2560**) 打包 | I8 | 5.898 MB |
| `layers.20.ffn.experts.0.w1.scale` | (2304, 160) | F8_E8M0 | 0.369 MB |
| `layers.20.ffn.experts.0.w2.weight`（down） | (5120, 1152) | I8 | 5.898 MB |
| `layers.20.ffn.experts.0.w3.weight`（up） | (2304, 2560) | I8 | 5.898 MB |
⇒ 单专家 ≈ **17.9 MiB**；单层（384 专家 + shared + gate）≈ **6.76 GiB**；
**后 20 层 MoE 合计 = 138,451 MiB = 135.21 GiB**（其中 routed experts 134.5 GiB）；
**TP=4 下每卡 +33.80 GiB**。逻辑权重数 = **271.79 G 元素**。

### 换成别的格式（按 271.79 G 元素算）
| 副本格式 | 全模型后20层 | **每卡（tp=4）** |
|---|---|---|
| **FP4 打包 + UE8M0 scale（同现状）** | ~127 GiB | **≈ 31.6 GiB**（实测盘上 33.8 GiB/卡） |
| FP8 | 253.1 GiB | ≈ 63.3 GiB |
| **BF16** | **506.3 GiB** | **≈ 126.6 GiB** |
| 现有 rank-16 适配器（对照） | **13.6 MiB** | 13.6 MiB |

### 对 §7.4 路径①（把适配器折进权重）的含义
- 每适配器一份合并权重 = **+31.6~33.8 GiB/卡（FP4）** 或 **+126.6 GiB/卡（BF16）**；
  而现有适配器只要 **13.6 MiB** ⇒ **放大约 2500×**。
- 现网配置 `--mem-fraction-static 0.78`、B300 单卡 275 GiB、捕获后 `available_gpu_mem≈52 GB`
  ⇒ +33.8 GiB/卡 = 吃掉约 **12%** 单卡容量且**直接压缩 KV pool**（吞吐下降）；BF16 版本（delta 保真所必需）**根本放不下**。
- ⇒ **结论**：权重折合路线在显存上**只对「单一固定适配器」勉强可行**（且要牺牲 KV/吞吐）；
  BF16 保真版不可行；多适配器热切不可行（除非用"替换基座权重"而非"多存一份"，那就失去热切能力）。

---

## 10. 「形态差」的具体名字 + 零额外显存的候选路线（2026-10-10，用户质询"有没有不浪费空间的办法"）

### 10.1 cubin 层面的新证据（比 §7.2 的"缺文件"更硬）
| 家族 | 文件数 | 带 swiGlu | block 布局 | 含义 |
|---|---|---|---|---|
| `Bmm_Bfloat16_MxE2m1Bfloat16` | **144** | **96** | **`Ab32`** | **bf16 激活**（无激活量化）+ MxE2m1 权重 + **block-32（MX/UE8M0）= 我们的权重格式** ✓ + `clmp`(144) 与 swiGlu 融合 |
| `Bmm_Bfloat16_MxE2m1MxE4m3` | 106 | 0 | `Ab32` | bf16 激活，无 swiGlu |
| `Bmm_E4m3_MxE2m1E4m3` | 40 | 40 | — | E4m3 激活（weight-only fp8 专用） |
| **我们分解链在用**：`Bmm_MxE4m3_MxE2m1MxE4m3` | 168 | — | `Ab32` | **8bit E4m3 激活** ⇒ 必须先做激活量化，且**无 swiGlu 融合** |
| `Bmm_E2m1_E2m1E2m1`（4bit 激活） | 476 | 88 | **`Ab16_Bb16`（block-16 = NVFP4/FP8-scale）** | **与我们的 block-32 MX 权重格式不兼容**（不是"缺一个文件"） |

⇒ **"−35% 的形态差"有了具体名字**：
- **stock/base 形态**：`bf16 激活 + MxE2m1(=我们的权重格式, block32) + swiGlu/clmp 融合 epilogue`，**无激活量化步**；
- **我们的分解链形态**：`E4m3(8bit) 激活 + 独立激活量化核 + 无 swiGlu 融合`（因为 LoRA delta 必须加在 activation 之前，而融合 epilogue 吃不下 data-dependent 的 delta）。
- ⇒ **4-bit 激活不可用的主因 = MX（block-32）传统里根本没有 4-bit 激活的 A 侧变体**（`Bmm_MxE2m1_MxE2m1` = 0 个）
  —— **这与权重格式无关，权重不需要动**（基模 MoE 本来就是 MXFP4：`w1.weight` I8 打包 (2304,2560)、
  `w1.scale` F8_E8M0 (2304,160) = block-32/UE8M0 ✓，config `weight_block_size [32,32] + scale_fmt ue8m0 + expert_dtype fp4` ✓）。
- **更正（原文措辞有误）**：只有当我们想去"借"现存那 476 个 4-bit 激活 cubin（`Bmm_E2m1_E2m1E2m1`）时，
  才需要改权重格式——因为它们在 **block-16（NVFP4 / FP8-scale）** 传统里，而我们是 block-32 MX。
  ⇒ 那是**一条我不想推荐的支线**，**不是**"要靠近 base 就必须重量化"。
- ★ **好消息**：MX 家族允许 **bf16 作 A 侧**（`Bmm_Bfloat16_MxE2m1Bfloat16`，144 个，96 带 swiGlu、144 带 clmp，`Ab32`），
  它与我们**现有权重完全兼容** ⇒ 方案③（把分解链 GEMM1 的 A 侧从 `MxE4m3`(8bit+量化步) 换成 **bfloat16**）
  是**零权重改动、零额外显存**的。

### 10.2 零/低额外显存的候选路线
| # | 方案 | 额外显存/卡 | 能否接近 base | 说明/风险 |
|---|---|---|---|---|
| ① | **单适配器离线合并**（把适配器烘进权重，当普通模型服务，不带 `lora_path`） | **0** | ✅ **就是 base（实测 12244）** | 不能多适配器热切；切换需重写 ~135 GiB（~30–60 s）；改动小（一条离线合并脚本） |
| ② | **K 增广等价式** `W x + B(A x) = [W\|B]·[x; A x]` | **≈0.15 GiB**（只存 rank-16 的 A/B 块） | ❌ 对本问题无用 | 数学恒等、几乎零显存，但它只等价于"合并权重"——而 **stock 融合核内部自己做 gather**，吃不下需要 per-(token,expert) 增广的输入 ⇒ 只能用在我们的分解 GEMM 上，而分解链里 delta 内核实测≈0 |
| ③ | **改分解链形态**：GEMM1 从 `MxE4m3`(8bit A + 量化) 换成 **`Bfloat16_MxE2m1Bfloat16`**（bf16 A、同权重格式、可选 swiGlu 变体），并融合我们自己的核 | **0** | ❓ 待归因（可能吃掉 −35% 一大块） | 需改 op 接口/kernel；**必须先做分阶段归因**才知道值不值 |
| ④ | 全部专家**重量化成 NVFP4(block-16)** 以启用 4-bit 激活家族 | 0（略省） | ❓ | 全模型级改造 + 改变 base 数值 + 大验证 |
| ⑤ | 上游给**带 LoRA 钩子的融合核** | 0 | ✅ | 依赖 NVIDIA |

### 10.3 结论（本轮）
- **"零额外显存 + base 速度" 现在就可行的只有 ①（单适配器离线合并）**；
- **多适配器热切 + 接近 base** 在现有闭源 cubin 下**无低成本解**（①不支持多适配器、②对本问题无效、④代价大）；
- **③ 是唯一"零显存 + 可能吃掉 −35% 一大块"的技术方向**，但**必须先做 150µs/层的分阶段归因**——
  工具已具备：`trtllm_fused_moe_kernel_launcher.cu` 里已有 `bench_permute`/`bench_nvfp4_quant`/`bench_activation`/`bench_fused_permute_quant`/`bench_fused_act_quant` 五个微基准（尚需包 FFI 暴露到 Python），
  可对 decode 真实形状（T=768, topk=6, hidden=5120, inter=2304）逐阶段计时，并与 stock 融合核同形状对拍。

---

## 11. 逐阶段归因实测（2026-10-10）——③ 的判决与新的零显存目标

方法：把 overlay 里**已用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 导出的 5 个微基准**（`bench_permute` / `bench_nvfp4_quant` /
`bench_activation` / `bench_fused_permute_quant` / `bench_fused_act_quant`）在**服务进程内**跑（env `SGLANG_STAGE_BENCH=1`，
注入在 `trtllm_lora_temp/__init__.py` 尾部 ⇒ overlay 已在本进程构建、零重编译、卡独占）；
形状取真 decode：H=5120, INTER=2304, topk=6（T=64/256/768）。8 个 TP rank 的数字一致。

| 阶段 | T=64 | T=256 | T=768 | ×20 层（T=256） |
|---|---|---|---|---|
| `permute`（gather） | 4.99 | 10.45 | 22.11 µs | **209 µs/step** |
| `activation`（含 delta，未融合） | 7.56 | 18.67 | 49.46 | 373 |
| **`fused_act_quant`**（实际使用的融合核） | 6.34 | 10.43 | 26.78 | **209** |
| `nvfp4_quant` | 4.35 | 6.37 | 12.43 | 127 |
| `fused_permute_quant` | 6.35 | 12.98 | 30.95 | 260 |

**⇒ 结论：分解链"自己的核"（permute 10.5 + 融合 act/quant 10.4）≈ 21 µs/层 ≈ 0.42 ms/step，
只占实测差距（8.25−5.23 = 3.0 ms/step）的 ~14%；其余 ~86%（≈2.6 ms/step）在两个 batched GEMM 内**
（闭源 cubin：`MxE4m3 × MxE2m1`）。

### 对"③ 把激活换成 bf16"的判决：**不写**
- ③ 只能影响上述 ~14% 里的"激活/量化"部分（`_prepare_flashinfer_mxfp8_activations` + 融合 act/quant 的 E4m3 量化 +
  `sgl_gather_swizzle_ue8m0_sf`），**动不了占 86% 的闭源 GEMM**；
- 且 launcher 里 bf16 输入分支是 `Dtype::E2m1` + per-token scaling（需要不存在的 `Bmm_MxE2m1_MxE2m1`），
  要试 `Bmm_Bfloat16_MxE2m1Bfloat16` 必须**新写 C++ 分支**；
- ⇒ **收益上限 ≈ 14% × （其中量化相关部分占比）**，风险是新写 C++ + 违反 §3 红线 1（改激活量化格式）。
  **结论：不值得写。**（与 F1「delta 内核 ≈ 0」同一逻辑：链内小件都不值钱，大头在形态/闭源核。）

### 新的零显存目标：**GEMM 的 group padding / 调度**
decode 稳态下 384 个专家、每组只有 1~4 行；`moe_align_block_size` 的 **block_size** 直接决定 padded 行数
（按 128 块 ⇒ 384×128 ≈ 49K 行 vs 有用 ~1.5K 行，32× 空转）。这块**在我们自己的调度里**，
相关代码：`kernels/ops/moe/trtllm_lora_temp/moe_lora_merged_align.py`（fused align，含 block_size）、
`virtual_experts.py`（`BLOCK_SIZE_M`/`num_tokens_post_padded`）。
⇒ 下一步：查该 block_size 的实参、与 stock 融合核的调度对比、试调小（**零显存、零权重改动**）。

### 环境教训（本轮踩坑，已修）
- **`/tmp` 是 30G 独立分区且被 13.2G 旧 `tmp*.json.tmp` 残留占满 ⇒ 编译器写 ptxas 临时文件失败**；
  已 `mv` 到 `/root/_scratch_trash`，恢复 12G。
- 独立进程跑 overlay 微基准会**触发全量 JIT 重编译**（且磁盘满时把缓存条目打成"需重建"状态，拖慢后续所有启动 ≈30 分钟）
  ⇒ **教训：overlay 的微基准必须在服务进程内跑，且先确认 /tmp 有空间。**

---

## 12. 那 86% 的 GEMM 侧归因：**padding 块被闭源 cubin 锁死在 t128**（2026-10-10）

### 证据链
1. C++ 里 `total_num_padded_tokens` 由 **`tile_tokens_dim`** 决定（`trtllm_fused_moe_kernel_launcher.cu:407`
   `...getTotalNumPaddedTokens(args->num_experts, tile_tokens_dim)`），而 `tile_tokens_dim` 来自
   **所选 GEMM tactic 的 M-tile**（`:143-160` `computeSelectedTileN`）⇒ **padding 块 = M-tile**。
2. cubin 库实测（`/tmp/trtllm/.../trtllmGen_bmm_export/cubins`，1437 个 batched-GEMM cubin）：
   | 家族 | M-tile |
   |---|---|
   | 我们用的 `Bmm_MxE4m3_MxE2m1MxE4m3` | **全部 `t128`**（N/K 有 8/16/32/64/128/256 变化） |
   | stock 的 `Bmm_Bfloat16_MxE2m1Bfloat16` | **也全部 `t128`** |
   | 全库 | **1437/1437 全是 `t128`** |
3. ⇒ decode 稳态（384 专家、每组 1~4 行）下 padded 行数 ≈ 384×128 ≈ **49K**，有用行 ~1.5K ⇒ **~30× 空转**；
   而 **库里没有任何 <128 的 M-tile 变体**可选。
4. AGENTS.md #13 记录的 "`Gemm2::Runner(E2m1, E2m1, BF16)` … the E2m1×E2m1 **tile-8** cubin does not exist"
   **正是同一件事**：runner 的配置空间里有 tile-8，但 **NVIDIA 没发对应的 cubin**。

### 结论
- **那 86%（~2.6 ms/step）的 GEMM 侧不可在我们这侧优化**：形态（`MxE4m3` 8-bit A + `t128` M-tile）由**闭源 cubin 供给**决定；
  我们唯一能动的仍只有链自身那 14%。
- ⇒ **「零额外显存 + 接近 base」在本侧无解**，除非：
  - **上游补 cubin**：`Bmm_MxE4m3_MxE2m1MxE4m3` 的 **tile-8/tile-16 M-tile** 变体（消掉 30× padding 空转），
    或带 LoRA delta 钩子的融合核（把 gate_up delta 加在激活前）。**这是一个具体、可向上游提单的诉求**；
  - 或 **① 单适配器离线合并**（零额外显存、直接 base = 12244）。

### 本轮"试一下"的完整账（用户问"效果会不会比较好"）
| 尝试 | 结果 |
|---|---|
| ③ 激活换 bf16（stock 形态） | ❌ 不写：launcher 无该分支（需新写 C++）；且只能动 14% 里的一部分 |
| 逐阶段归因（5 个已导出微基准，服务进程内跑） | ✅ 完成：链自身核 21µs/层 = 0.42ms/step（**14%**）；GEMM 侧 2.6ms/step（**86%**） |
| padding/tile 调小（零显存、我们手里） | ❌ 不可为：**t128 被锁死**（1437/1437 全 t128，无更小 M-tile） |
| 环境 | ⚠️ 踩坑：`/tmp` 被 13.2G 旧残留占满（已清）+ overlay JIT 缓存被我的独立进程打进"需重建"（重编译 ~30min）。**教训：微基准只在服务进程内跑** |

### 下一步（二选一，均零额外显存）
1. **① 单适配器离线合并**：把适配器烘进权重当普通模型服务 ⇒ 直接 base 速度（12244）；代价=不能多适配器热切。
2. **向上游提单**（具体诉求已在上文写清）：tile-8/16 的 MxE4m3×MxE2m1 cubin，或带 LoRA 钩子的融合核。

---

## 13. 为什么 LoRA 的 GEMM 这么慢 + 能不能自己写超过闭源（2026-10-10）

### 13.1 定量：**慢在 padding 空转，不在 GEMM 本身效率**
实测（§11）：链自身核 0.42 ms/step（14%），**GEMM 侧 ≈2.6 ms/step（86%）**。
把它拆开算（TP=4 ⇒ 每 rank 分到 `384/4 = 96` 个专家）：

| 量 | 值 | 说明 |
|---|---|---|
| 稳态 decode 的 token 数 | T ≈ 256 | 64 并发 ×（1+draft≈3.6 接受长） |
| 每 rank 有用 row-expert 数 | 256×6 = **1536** | 全部 token 都在本 rank 路由到它那 96 个专家 |
| 每专家组行数 | 1536/96 ≈ **16** | |
| `moe_align_block_size` 每专家补齐到 M-tile=**128** | 96×128 = **12,288 padded 行** | |
| **padding 空转** | 12,288 / 1536 ≈ **8×** | ⇒ **GEMM 时间 ~87% 是 padding 浪费** |
| 有用 FLOPs（每 rank 每层） | GEMM1 4608×5120×256×6×2/4 + GEMM2 … ≈ **27 GFLOP** | |
| 实测 130 µs/层 ⇒ 隐含算力 | ≈ **1.1 PFLOPS/GPU** | ≈ B300 fp4 峰值的 30~40% ⇒ **效率是正常的，慢在 8× padding** |

⇒ **结论：我们的 batched GEMM 每-FLOP 效率并不差；慢是因为 `tile_tokens_dim`=M-tile=**t128** 强制把
每 16 行有用数据补成 128 行。stock 融合核能用小的 tileTokensDim（AGENTS.md #13 提到的 tile-8），
padding 只有 ~1.2× ⇒ 这就是 −35% 的主要来源。**

### 13.2 能不能自己写超过闭源？分两个目标，答案不同
- **"打平/超过 stock 融合核本身"**：不现实（它是 NVIDIA 调过的 tcgen05/TMA/SF 全融合核）。
- **"超过我们被迫使用的分解链"**：**现实，而且这是唯一能同时拿到「零额外显存 + base 级速度 + 热切适配器」的路**。
  因为我们不需要在每-FLOP 效率上赢，只需要**不浪费那 8×**：一个 **small-M（tile 8/16）的 grouped GEMM**
  每行效率即使只有 trtllm-gen 的 1/3，总时间仍能赢 ~2.5×。
  而 tile-8 **正是闭源里"配置有、cubin 没发"的那一档**（AGENTS.md #13 的 "tile-8 cubin does not exist"）
  ⇒ 自己写反而**没有 cargo-cult 的束缚**。

### 13.3 三条路线（按工程量从小到大）
| # | 做法 | 工程量 | 预期 |
|---|---|---|---|
| A | 把链里的两个 GEMM 换成**开源 small-M grouped GEMM**（DeepGEMM / CUTLASS-Triton，支持 MXFP4+MXFP8-A） | 中（集成+数值对齐） | 可能吃掉那 ~87% padding 的大头 |
| B | 自己写 **tile-8/16 grouped GEMM**（Triton/CUDA），专门打 decode 的小组形状 | 中高 | 同上，且完全可控 |
| C | 写**全融合 LoRA-MoE 核**（gather+GEMM1+delta+silu+quant+GEMM2+delta+finalize） | 高（数周） | 终态：base 级速度 + 零显存 |

### 13.4 先做**一次便宜的确认实验**（再决定是否投 A/B/C）
把链里的 op 在**同一 T** 下跑两种路由：
- **真实分散路由**（384 专家均摊 ⇒ 每组 ~16 行）；
- **集中路由**（同一批 token 都打到少数几个专家 ⇒ 每组数百行，padding 几乎为 0）。
⇒ 两者时间差 ≈ padding 成本。若差 ~8×（如 130µs → ~20µs/层），则 §13.1 成立、A/B/C 值得投；
若差很小，则我的 padding 模型错了，得回头重测。**这是投任何 kernel 工程前必须付的一次 1 小时成本。**

---

## 14. 【新目标】LoRA 请求 TPOT < 6 ms —— roadmap（2026-10-10，用户军令）

现状：`moe_rnd` TPOT 8.25 ms / base 5.23 ms ⇒ 差 **3.02 ms/step**。军令：**LoRA 请求压到 < 6 ms**。
§11/§13 的预算表明：**必须消掉 GEMM 侧的 padding 空转（2.6 ms），否则不可能达标**；
并再压掉链自身 14%（0.42 ms）的一部分以获得余量。

| 里程碑 | 内容 | 门（Gate） |
|---|---|---|
| **M1 确认** | padding 主导性验证（免写代码）：padded 行数 = 96 专家×128 = **12,288，与并发无关** ⇒ 若 padding 主导，**base↔rnd 步时差在不同并发下应近乎平坦**。跑 C=16/32/64 的 base/rnd 对拍（`/root/m1_sweep.sh`） | gap 平坦（≈2.6ms 不随 C 变）⇒ 模型成立；若 gap 随 C 线性增长 ⇒ padding 模型错，回头重测 |
| **M2 主攻** | 把链里 **GEMM1/GEMM2 换成 small-M（tile 8/16）grouped GEMM**：<br>· M2a 先查开源可复用（DeepGEMM / CUTLASS-Triton，需支持 **MXFP4(block32) 权重 + MXFP8-A**）<br>· M2b 否则自己写（Triton 先行，验证后再考虑 CUDA/tcgen05） | 逐阶段微基准：GEMM 侧 **2.6 ms → ≤0.5 ms/step**；数值与现链一致（同 ref 对拍） |
| **M3 收尾** | 压链自身 14%：permute(E4m3)与量化融合、metadata/align 合并 | 链自身 **0.42 → ≤0.25 ms/step** |
| **M4 交付** | 同批 A/B（64 打满稳态、剔除前后 15%）达标 + 文档 + push | **TPOT p50 < 6.0 ms**（且无崩溃、零 == base 平价不劣化） |

**预算核对**：5.23（base） + 0.4（新 GEMM） + 0.25（链路） ≈ **5.9 ms** ⇒ 刚好达标（余量 ~0.1ms）。
⇒ 两条都要做；M2 是主战场，M3 是达标的必要余量。

**M1 结果**：（待回填）

---

## 15. <6ms roadmap 执行记录（2026-10-10 下午）

### 15.1 M2a ✅ 免 padding 的核找到了
`deep_gemm 0.2.0` 提供：
```
m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe_a=None, recipe_b=None,
                                 compiled_dims='nk', disable_ue8m0_cast=False)
m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, …, ensure_zero_padding=True)
per_token_cast_to_fp8/fp4(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False)
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, grouped_layout, ks, gran_k)
```
= **fp8 A × fp4 B 的 masked（按每专家行数、免补齐）分组 GEMM** ⇒ 直接消掉 §13 的 ~8× padding 空转。

### 15.2 M2b 🔑 关键发现：仓库里已有 DeepGEMM 的 DSV4 MXFP4 MoE 路径，且代码在我们手里
- `srt/layers/moe/moe_runner/deep_gemm.py`：`DeepGemmMoeQuantInfo(..., is_fp4_experts, use_mxfp8)`
  注释原文 *"DSV4 mxfp4 layout flag; selects recipe_a=(1,128)/recipe_b=(1,32) downstream"*
  ⇒ **A 侧 MXFP8(1,128)、B 侧 MXFP4(1,32) = 我们的权重格式** ✓
- 路径：`masked_m` → masked grouped GEMM（gateup）→ **`_varlen_deep_gemm_situ_mul_quant(gateup_output, masked_m, …)`**
  → masked grouped GEMM（down）；DSV4 专用 JIT：`silu_and_mul_clamp` / `silu_and_mul_contig_post_quant`
- `mxfp4.py`：`layer._mxfp4_backend = "deep_gemm"`，开关 = `--moe-runner-backend deep_gemm`
  （`arg_groups/choices.py:133 MOE_RUNNER_BACKEND_CHOICES` 已含 `deep_gemm`）
- ⇒ **两个 LoRA delta 注入点已定位**（gateup 输出之后 / down 输出之后），delta 仍由现有 VE 核产出
  ⇒ 适配器仍 **13.6 MiB** ⇒ **零额外显存**；与之前所有闭源方案的本质区别：**这段代码我们能改**。
- 预算：base 5.23 + DeepGEMM MoE ≈0.4 + 链自身 ≤0.25 ≈ **5.9 ms** ⇒ 达标。

### 15.3 M1 作废重做（两次自身失误，如实记录）
1. **参数缺陷**：`--n C*4` + `out-tokens 2048` ⇒ 窗口 ~37s，稳态样本 0 ⇒ 稳态统计全空。
2. **操作事故**：M1 还在压测（C=32/64）时我启动了 DeepGEMM 后端的 boot ⇒ **把正在压测的栈杀掉**
   ⇒ C=64 `Connection refused`、C=32 rnd 24 个 `RemoteDisconnected`。日志 23 行、无 Traceback
   ⇒ **不是 LoRA 崩溃**。**教训：任何压测进行中，绝不起第二个 boot。**
   （唯一有效的 C=16 一轮：base agg 3559 vs rnd 2517，比 1.41×，方向与 padding 模型一致但不显著。）

### 15.4 下一步
1. 读 DeepGEMM 后端 bench（`--moe-runner-backend deep_gemm`，task `d0991419`）⇒ **M2 胜负手**：
   若该免 padding 路径在 decode 达到 stock 融合核水平（base 5.23ms）⇒ M2b 只需接线 + 两处 delta 钩子；
2. 重做 M1（长窗口、与 M4 同口径：out=4096、n=256），期间**绝不启动第二个 boot**；
3. M2b 加钩子 → M3（压链自身 14%）→ M4 同批 A/B 验收 TPOT p50 < 6.0ms。

---

## 16. M2 的"快捷路"被挡 + 真正的绕行方案（2026-10-10）

### 16.1 实测：`--moe-runner-backend deep_gemm` 对本模型无效
boot 报错（decode 端，TP0/TP1/TP2/TP3 全一致）：
```
NotImplementedError: Fp8MoEMethod does not expose quant info for 'deep_gemm'
```
原因：本 checkpoint 是 `quantization_config.quant_method = "fp8"` + `expert_dtype = "fp4"`
⇒ MoE 走 **`Fp8MoEMethod`**，而 DeepGEMM 的 quant-info 集成只写在
`srt/layers/quantization/mxfp4.py`（`Mxfp4FlashinferTrtllmMoEMethod` → `layer._mxfp4_backend = "deep_gemm"`）
⇒ **这条开关路需要先给 `Fp8MoEMethod` 补 DeepGEMM quant info 才有用**（可做，但不是我们的主路）。

### 16.2 主路：**绕开 runner/quant-method 管道，在我们自己的 LoRA MoE 调度里直接调 DeepGEMM masked 核**
（M2b 本来就要写这条 LoRA 路径，所以这不是额外工作）

| 步 | 动作 | 现成件 |
|---|---|---|
| 1 | standard topk → 每专家行数 `masked_m`（**免 padding**） | 现有 routing |
| 2 | 激活量化 `per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)` ⇒ E4m3 + UE8M0 | DeepGEMM |
| 3 | `m_grouped_fp8_fp4_gemm_nt_masked(A=act_fp8, B=w13_fp4, masked_m, expected_m, recipe_a=(1,128), recipe_b=(1,32))` | DeepGEMM（**消掉 8× padding**） |
| 4 | `gate_up += gate_up_lora_delta`（masked 布局） | 现有 VE 核（需布局适配） |
| 5 | 融合激活 + 量化（含 OAI gated-silu/clamp） | `silu_and_mul_masked_post_quant` / `silu_and_mul_clamp` |
| 6 | 同法第二个 masked grouped GEMM（down） | DeepGEMM |
| 7 | `output += down_lora_delta` | 现有 VE 核 |

- **权重**：现有 MXFP4（E2m1 + UE8M0 **block-32**）只需**一次性 repack** 到 DeepGEMM 布局
  （`transpose_packed_fp4` + `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`）⇒ **字节数不变、内存中性**；
- **适配器仍 13.6 MiB ⇒ 零额外显存**；数值会变（A 侧 block-128 而非 block-32；base 走 bf16 本无 bitwise 平价）；
- 预算不变：base 5.23 + DeepGEMM MoE ≈0.4 + 链自身 ≤0.25 ≈ **5.9 ms**。

### 16.3 立即要做的验证（M2 胜负手的正确形态）
写一个**独立微基准**（单独占卡）：对我们的真实 decode 形状（T=256、96 专家/rank、每组 ~16 行）
测 `m_grouped_fp8_fp4_gemm_nt_masked` 的耗时 ⇒ 若 20 层合计 ≈0.2~0.5 ms（对照现链 2.6 ms）
⇒ **M2 成立**，随后按 16.2 的 7 步实现；否则回到 §13 预算重新分解。
（注意：该微基准需要独占 GPU ⇒ **先停栈**；且**绝不在压测进行中起 boot**。）

---

## 17. M2 原型实测（2026-10-10 晚）：DeepGEMM masked 分组 GEMM 的**确切契约**

### 17.1 权威用法（照抄仓库里已在 DSV4 上工作的调用，`moe_runner/deep_gemm.py:755-779`）
```python
num_groups, m, k = hidden_states.shape          # A 是 3-D (num_groups, m, k)
n = w13_weight.size(1)
gateup_output = torch.empty((num_groups, m, n), dtype=torch.bfloat16, ...)   # D 也是 3-D
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
    (hidden_states, hidden_states_scale),   # A 对
    (w13_weight, w13_scale),                # B 对，3-D
    gateup_output, masked_m, expected_m, recipe_a=recipe_a, recipe_b=recipe_b)
```
**★ masked 的语义 = 缓冲区按 `m` 补齐，但核只处理 `masked_m[g]` 行** ⇒ **padding 在核层面被消掉**（正是我们需要的）；
⇒ 我们的 tile_tokens_dim=128 空转，正是"用 `m=128` 且 `masked_m[g]=真实行数`"能解决的问题。

### 17.2 实测到的两个契约坑（已定位，避免后人重踩）
1. 传 **2-D A** ⇒ `Assertion error (layout.hpp:46): t.dim() == N` ⇒ **必须 3-D 分组布局**；
2. **recipe 必须与 cast 的粒度配对**：实测 `per_block_cast_to_fp8((1536,5120))` 返回 **480 = (1536/128)×(5120/128)** 个 scale
   ⇒ 它是 **128×128 双向分块**（对应 `recipe_b=(128,128)`）；
   而 A 侧要 **per-token**：`per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)` ⇒ `(M, K/128)`，对应 `recipe_a=(1,128)`
   （与 §15 的 DSV4 注释 `recipe_a=(1,128)/recipe_b=(1,32)` 一致——**B 侧 32 就是我们的 MXFP4 block-32**）。
3. `per_block_cast_to_fp8` 无 `block_size` 参数（0.2.0 版），别按旧文档传参。

### 17.3 下一步（继续 M2 的最短路径）
1. 修原型：A 用 `per_token_cast_to_fp8(gran_k=128)`、B 用 `per_block_cast_to_fp8`（或用 sglang 的 `deep_gemm_wrapper`，
   它已把 SF 打包/对齐封装好：`DEEPGEMM_SCALE_UE8M0` 分支里 `_cast_to_e8m0_with_rounding_up` + `get_mn_major_tma_aligned_packed_ue8m0_tensor`）；
   **优先直接用 sglang 的 wrapper**（`grouped_gemm_nt_f8f8bf16_masked`）⇒ 少踩坑；
2. 量 `masked(m=16/组) vs padded(m=128/组)` ⇒ padding 惩罚因子 + 可达 GEMM 成本（目标：20 层 ≤0.5ms）；
3. 若成立 ⇒ 实现 §16.2 的 7 步（LoRA-aware DeepGEMM 路径，权重一次性 repack、适配器仍 13.6 MiB）；
4. 之后 M1 重做（长窗口）、M3（压链自身）、M4（同批 A/B 验收 TPOT p50 < 6.0ms）。

### 17.4 本轮环境态（交接必读）
- 1101 **栈已全停、8 卡已清空（0 MiB）**；重启用 `bash /root/exp_single.sh`（单流基线）或 `/root/exp_m2.sh`（双流）。
- **绝不在压测进行中启动第二个 boot**（本轮已因此毁掉 M1 的 C=32/64 数据）。
- `/tmp` 分区仅 30G：跑任何编译/微基准前先 `df -h /tmp`（曾被 13G 旧 `tmp*.json.tmp` 占满导致 ptxas 写盘失败）。

---

## 18. ❌ padding 假设被**直接证伪**（2026-10-10）—— M2 路线取消，-35% 的主因另有其物

### 18.1 实测（`tools/dsv41_two_stream/m2_proto_deepgemm.py` 的同族原型，DeepGEMM masked 分组 GEMM）
形状 = 真 decode（G=96 专家/rank、K=5120=hidden、N=4608=2·inter、A 用 per-token(1,128)、B 用 per-block(128,128)）：

| 用例 | 每组行数 | 总行数 | 单次 | ×20 层 |
|---|---|---|---|---|
| **masked**（真实有用行） | 16 | 1,536 | **440.5 µs** | **8,810 µs/step** |
| **padded**（我们被强制的 t128 形态） | 128 | 12,288 | 462.2 µs | 9,243 µs/step |

⇒ **padding 惩罚 = 1.05×**（不是 §13 推的 8×）。

### 18.2 机制（为什么）
B 侧每个专家需流式读入完整 `N×K` 权重：4608×5120×1B ≈ **23.6 MB/专家 × 96 = 2.26 GB/次调用**
⇒ 按 ~3–8 TB/s ≈ **280–750 µs**，与实测 440 µs 吻合
⇒ **该 GEMM 是"权重带宽受限"，不是"行数受限"** ⇒ **padding 行几乎免费**（masked 与否都要流同样的权重）。

### 18.3 结论（对本 roadmap 的直接影响）
- **M2（免 padding 拿回 2.6 ms）不成立** ⇒ §13 的"8× 空转占 87%"是**推断错误**（应为"权重流式主导"）；
- DeepGEMM 这版实现在本几何下**并不更快**（8.8 ms/step 单侧）——同样是权重流式受限；
- **−35%（3.0 ms/step）的现有分解**：0.42 ms（14%）= 链自身核（**已实测**）；其余 ~2.6 ms 的最可能解释是
  **分解链丢掉了 base 融合 cubin 内部的"权重流式与算力重叠"（单核软件流水线）**，
  而不是 padding 或 kernel 数量问题；
- ⇒ 要补回这 ~2.6 ms，只剩两条路：**(a) 融合（在闭源 cubin 内 ⇒ 不可及）**；
  **(b) 图内更激进的跨层重叠**（MoE 与相邻层 attention/其他工作重叠；需新图/流设计，收益未知、风险高）。

### 18.4 教训（方法论）
**"从公式推断成本"会误导**：§13 的 8× 空转是算出来的、看着很合理，实测只有 1.05×。
⇒ 任何 kernel/调度优化**必须先有该动作的直接微基准**，再投入实现（本轮 F1 与本节两次都印证了这条）。

---

## 19. 达标 <6ms 的唯一路径 ① 的实施计划（可直接开工；等待决策）

前提：roadmap 的 M2 前提（免 padding 拿 2.6ms）已被 §18 实测证伪 ⇒ **本侧零显存 + 多适配器热切 = 无解**。
唯一能达标的是 ①：**把适配器离线折进权重，当普通模型服务**（实测 base TPOT **5.233 ms** < 6 ms ✓，额外显存 0）。

### 19.1 实施步骤（离线，不影响线上服务形态）
1. **提取适配器**：对后 20 层（20–39）的每个 MoE 专家 e：
   `ΔW13_e = B13_e @ A13_e`（(2·inter, hidden)）、`ΔW2_e = B2_e @ A2_e`（(hidden, inter)）
   —— 适配器是 rank-16（13.6 MiB），该 GEMM 在 GPU 上秒级完成。
2. **合并到基座权重**：`W13'_e = W13_e + ΔW13_e`、`W2'_e = W2_e + ΔW2_e`（在**去量化后的 bf16 域**做加法，保 delta 精度）。
3. **重新量化到 MXFP4**（block-32 + UE8M0，含 w13 的 e4m3x2 scale）：
   - 这一步的**量化噪声按 |W| 量级给出** ⇒ **小 delta 会被淹没**（随机适配器无此问题，真实训练的适配器有）；
   - 若要保 delta 保真：可把 `ΔW` 只保留 **低秩残差** 的额外通道（超出本计划范围）或接受精度损失并**用评测**判是否可接受。
4. **落盘一份 merged checkpoint**（仅后 20 层 MoE 张量替换；约 135 GiB，/mnt 有 207 GB 可用）→ 用常规方式起服（**请求不带 `lora_path`**）⇒ 走 stock 融合核。
5. **验收（与 M4 同口径）**：64 打满稳态、剔除前后 15%、`bench_chat3.py --conc 64 --n 256 --out-tokens 4096`；目标 **TPOT p50 < 6.0 ms**（预期 ≈5.2 ms）。
6. **数值验收**：与"未合并 + 正常 LoRA 请求"的输出做**统计对比**（本栈 greedy 不可复现，见 §M3 ⇒ 只能比分布/公共前缀，不能比 bitwise）。

### 19.2 已知代价（必须让产品侧拍板）
- **不能按请求热切适配器**（切换 = 重写 ~135 GiB，约 30–60 s）；多适配器需多份权重（+31.6~33.8 GiB/卡/个）。
- **小 delta 的保真度下降**（MXFP4 重量化）；需用真实适配器 + 评测确认可接受。

### 19.3 若"多适配器热切"是硬需求
只剩两条（都不在本侧）：
1. **上游**：请 NVIDIA 提供**带 gate_up LoRA delta 钩子的融合 MoE 核**（或 `Bmm_MxE2m1_MxE2m1` + 小 M-tile 变体）；
2. **自研融合核**：仓库已有 act/quant/delta 的核（`dev_kernel.cu`、`fused_activation_mxfp8_quant.cuh`、VE delta 核），
   缺的是把 GEMM1+delta+act+quant+GEMM2+delta 融进一个 tcgen05 核 ⇒ 数周级工程。

---

## 20. ① 离线合并的数值可行性：**不成立**（2026-10-10，实测）——廉价路线至此全部关闭

### 20.1 实测（真实专家权重 + MXFP4 往返 + rank-16 delta）
`tools/dsv41_two_stream/path1_viability.py`（读 checkpoint 的 fp4 打包 + F8_E8M0 scale，去量化 → 叠加 delta → 按 MXFP4 重量化 → 量误差）：

| \|dW\|/\|W\| | \|dW\| rms | 重量化误差 rms | err/\|dW\| | delta 存活 |
|---|---|---|---|---|
| 3e-1 | 6.65e-3 | 2.67e-3 | 0.40 | 勉强 |
| 1e-1 | 2.22e-3 | 1.94e-3 | 0.87 | **否** |
| 3e-2 | 6.65e-4 | 1.06e-3 | 1.60 | **否** |
| 1e-2 | 2.22e-4 | 9.38e-4 | 4.23 | **否** |
| 3e-3 | 6.65e-5 | 9.44e-4 | 14.2 | **否** |

基线：**W 本身的 MXFP4 往返误差 = 0**（说明量化流程复现了原格式，可信）；量化噪声 ≈ 9.4e-4 rms ≈ **W rms 的 4%**。

### 20.2 结论
- **MXFP4 重量化噪声会淹没任何 < ~10%|W| 的 delta**；真实训练的 LoRA delta 正在这个量级
  ⇒ **① 离线合并会毁掉适配器效果**（我们手上的随机适配器 delta 很大 ⇒ 用它会"看起来能用"，但**不能代表真实适配器**，验收必须用真实适配器 + 评测）；
- 这条同时解释了**为什么必须走分解链**：delta 需要在 bf16/fp32 精度里加，4-bit 权重装不下。
- ⇒ **① 作为「零额外显存 + 达标 <6ms」的方案同样不成立。**

### 20.3 「<6ms + 真实 delta 保真 + 多适配器热切」——廉价路线**全部实测关闭**
| # | 路线 | 结论（实测） |
|---|---|---|
| 1 | 双流调度重叠 | +2.4% |
| 2 | LoRA delta 融进激活/量化核 | ≈0 |
| 3 | 免 padding（DeepGEMM masked） | 1.05×（权重带宽受限） |
| 4 | DeepGEMM 替代 GEMM | 更慢 |
| 5 | 激活换 bf16（stock 形态） | 只能动 14% 的一小部分，需新写 C++ |
| 6 | 本侧其它融合 | ~2-3% |
| 7 | **① 离线合并** | **数值上毁掉小 delta**（本节） |
| 8 | 多份合并权重（保多适配器） | +31.6~33.8 GiB/卡/适配器（FP4）/ +126.6 GiB（BF16） |
⇒ 剩余只有两条**非本侧/重工程**路径：
1. **上游**：带 gate_up LoRA delta 钩子的**融合 MoE 核**（把 delta 加在激活前、以 bf16/fp32 精度）；
2. **自研融合核**：仓库已有 act/quant/delta 的核（`dev_kernel.cu`、`fused_activation_mxfp8_quant.cuh`、VE delta 核），
   缺的是与 GEMM 融合的 tcgen05 部分 ⇒ 数周级。

---

## 21. ✅ 达标路径③：**TPOT 随并发上升 ⇒ 给 LoRA 流量限并发即可 <6ms（零 kernel 工作）**（2026-10-10）

### 21.1 实测曲线（同一台服务器；`bench_chat3.py`，out=4096 长窗口，`steady3.py` 剔前后 15%、minc=C）
| 并发 C | **TPOT p50** | TPOT p90 | 满批 p50 (tok/s) | 样本 |
|---|---|---|---|---|
| 8 | **3.392 ms** | 4.469 | 2358.7 | 109 |
| 16 | **4.774 ms** | 5.441 | 3351.2 | 123 |
| 24 | **5.030 ms** | 5.602 | 4775.2 | 108 |
| **32** | **5.934 ms ✅ <6** | 6.488 | 5403.2 | 100 |
| 64（原口径） | 8.269 ms ❌ | 9.422 | 7739.5 | 43 |

⇒ 曲线**单调**（自洽，可信）；**C≤32 时 LoRA 请求 TPOT p50 已 < 6 ms**。

### 21.2 结论与含义
- **目标 `<6ms` 可达，且不需要任何 kernel 工作**：只需把 **LoRA 流量的有效并发控制在 ≤32**
  （例：LoRA 流量单独一个实例/队列并设 `--max-running-requests 32`；或在网关侧限制 LoRA 客户端并发；或对 LoRA 请求做调度侧配额）。
- **代价**：吞吐 ~−30%（C=32 的 5403 tok/s vs C=64 的 7739 tok/s）；且 p90 会略超 6ms（6.49）⇒ 若以 p90 为口径需把上限收到 ~28。
- **"−35%" 的正确解读**：它是**批大小效应**（每步时间随 batch 增长 ⇒ 每请求 ITL 增长），
  不是"LoRA 每 token 算得更烂"；因此**用并发配额（延迟↔吞吐权衡）就能满足延迟 SLA**。
  base 在 C=64 是 5.23ms ⇒ 若要做**混合部署**：base 流量可跑 C=64，LoRA 流量限到 C≤32，互不干扰。

### 21.3 与前文的关系（避免误读）
- §11–§20 的所有"廉价路线关闭"结论**仍然成立**（它们解释的是"同一并发下 LoRA 比 base 慢"这个 per-step 差距）；
- 本节给出的是**另一个自由度**：**用并发换延迟**，它不需要修 per-step 差距就能满足 SLA。
- 若要**同时**满足"T=64 高并发 + <6ms + 真实 delta + 多适配器热切"，仍需 §20.3 的两条重工程路径（上游融合核 / 自研融合核）。

### 21.4 复现
```bash
bash /root/exp_single.sh && bash /root/wait_ready.sh && bash /root/path3_sweep.sh
# 或单点：bench_chat3.py --lora moe_rnd --conc 32 --n 256 --out-tokens 4096 --win ... --jsonl ...
#         steady3.py <win> /root/dsv41_upstream/pd_decode.log 15 15 32
```

---

## 22. C=64 硬指标（<6ms）的 kernel 级归因 + 唯一剩余杠杆（2026-10-10 晚）

### 22.1 归因（仓库自带 torch profiler/CUPTI，`/start_profile` 40 步窗口，TP0，总核时间 1423.8ms）
| 排名 | kernel | 总时间 | 次数 | 每次 |
|---|---|---|---|---|
| 1 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_**t128x8x512u2**…`（GEMM1） | **333.2 ms** | 1600 | **208 µs** |
| 2 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_**t128x8x512**…`（GEMM2） | **290.5 ms** | 1600 | **182 µs** |
| 3 | `flash_fwd_splitkv_mla_fp8_sparse`（注意力） | 109.4 | 1720 | 63.6 |
| 4 | cutlass dense blockscaled gemm | 88.0 | 9200 | 9.6 |
| 7 | `_chunked_lora_shrink_kernel` | 45.4 | 3200 | 14.2 |
| 10 | `_chunked_lora_expand_kernel` | 26.5 | 3200 | 8.3 |
| 12 | `mxfp8_quantize` | 21.8 | 3600 | 6.1 |
| 14 | `moe::dev::permute::permuteKernel` | 17.9 | 1600 | 11.2 |
| 18 | `sgl_fused_act_mxfp8_quant` | 14.3 | 1600 | 8.9 |
⇒ **两个 MoE GEMM 占 623/1424 = 44%，且每次只要 208/182 µs（2 次/层 × 20 层）**；LoRA VE 每次 14.2+8.3 µs（4 次/层 ≈ 45 µs/层）。

### 22.2 autotune 缓存揭示的杠杆（关键）
`/root/.cache/sglang/flashinfer/autotune/0.6.18/sm103/<hash>/rank_tp*.json` 是按 **形状 → [tile, tactic]** 键控的表。
**我们模型的缓存**（hash `2e6e45331bfdb0ab`，家族 `flashinfer::trtllm_fp4_block_scale_moe`）示例：
```
tokens=64 → [16,783]   tokens=256 → [16,783]   tokens=512 → [16,391]
tokens=1024 → [32,575] tokens=1792 → [32,287]  tokens=2048 → [64,3]
```
**但这批条目是 stock/base 的「融合核」家族**；剖析显示**我们 LoRA 链的两个 GEMM 属于另一族**（`bmm_Bfloat16_MxE2m1MxE4m3…`，batched-GEMM，`t128x8x512`），
**不在 autotune 表里** ⇒ 它的 tactic 由 overlay 的 `getDefaultValidConfigIndex(...)` 取**默认值**。

### 22.3 ⇒ 唯一剩余的零显存杠杆：**把 batched-GEMM 这一族的 tactic 暴露并 sweep**
- 该族 cubin 可用 tile：`t128x{8,16,32,64}x{128,256,512}`（§F2 清单，106 个文件）；
- 我们当前跑在 **N-tile=8 / K-tile=512** 的默认档（每次 208/182 µs，等价权重带宽约 5.4 TB/s，激活侧可能被 N/8 次重读）；
- 做法（与 AGENTS.md 里 GLM 线"vendored tactic 硬编码 → 暴露 tactic kwarg + sweep"同型）：
  ① 在 overlay 的 batched-GEMM 调用处暴露 `tactic`/`tile` 覆盖（env 或 kwarg）；
  ② 在真 decode 形状（T=64/256、96 专家/rank、hidden 5120、N=4608/2304）上 sweep 候选；
  ③ 取最优后固化。
- **预期**：若默认小 N-tile 是带宽浪费的来源，可吃掉 623 ms 的 20~40% ⇒ **1.5~3 ms/step ⇒ 直接跨过 <6ms**；
  若 sweep 证明默认已近最优（GLM 线的先例是"证明默认近最优"），则本杠杆关闭，需回到 §20.3 的两条重工程路径。

### 22.4 复现
```bash
# 抓 C=64 LoRA 图内每核时间（40 步窗口）
bash /root/prof_c64.sh          # 生成 /root/prof/*-TP-{0..3}.trace.json.gz
/opt/sglang-venv/bin/python /root/trace_agg.py /root/prof/*TP-0*.trace.json.gz 28
# 看 autotune 表
/opt/sglang-venv/bin/python /root/dump_our_moe.py
```

---

## 23. 真正的旋钮找到了：`config_index = [tile_N, config]`（Python 侧直接可给，无需改 C++）

### 23.1 源码事实（overlay `resolveMoeTileAndConfig`）
```cpp
// Python side convention: tactic is [tile_N, config]
TVM_FFI_ICHECK(config_index.size() == 2) ...;
const int64_t tile_N = config_index[0];
const int64_t config  = config_index[1];
if (tile_N == -1 || config == -1)   // 我们目前恒传 [-1,-1] ⇒ 走这条
  return {selectDefaultTileN(supported_tile_nums, num_tokens, top_k, local_num_experts), -1};
return {tile_N, config};
```
- ⇒ **第一个元素就是 tile（tokens-per-tile），第二个是 cubin 的 config 索引**；
- ⇒ 我们 `core.py:492` 硬编码 `[-1, -1]` ⇒ **两个 LoRA MoE GEMM 一直走默认 tile+默认 config**（stock/base 的融合核族则是按形状 autotune 的）；
- ⇒ **tile 这个旋钮从 Python 就能给，完全不必改 C++**（上一轮我改 `computeSelectedTileN` 是**走错了路**，白付了 40 分钟 overlay 重建，已回滚）。

### 23.2 如何**枚举**合法 (tile, config)（不用瞎猜）
导出里有 **`trtllm_get_valid_moe_configs(dtype_act, dtype_weights, fp8_quant_type, top_k, hidden, inter, num_experts, num_tokens, bool, int, bool, int) -> Array<Array<int>>`**（12 参）
⇒ 对我们的形状（topk=6、hidden=5120、inter=4608/2304、96 专家/rank、T=64/256）直接问出**每个 tile 下的合法 config 列表**。

### 23.3 扫描配方（推荐，稳健且便宜）
1. 用上面的探针枚举合法 (tile, config)；
2. 把 `config_index` 由 env 覆盖（**模块级读一次**，`SGLANG_LORA_MOE_TACTIC="<tile>,<config>"`）⇒ **每个候选一次重启（~10 分钟）**，避免上一轮"文件热更新"把启动搞挂的坑；
3. 每个候选跑一次同批 64 并发 bench（out=4096，n=96~256）；
4. 取最优固化；若 ≥64 的 tile/config 都不改善 ⇒ 该杠杆关闭。

### 23.4 本轮踩过的坑（写给后人）
- ❌ 改 overlay C++（`computeSelectedTileN`）⇒ **触发 40+ 分钟全量重建**，且 `tile_tokens_dim` 只管 padding（§18 实测 padding 仅 1.05×）⇒ **走错路**；
- ❌ `core.py` 运行期"读文件热更新"补丁 ⇒ **decode 端起不来**（已回滚，`syntax OK`）⇒ **宁可用模块级 env + 重启**；
- ❌ 扫描脚本 `wait_ready` 失败未 fail-fast ⇒ boot-while-booting 踩死 ⇒ **必须 fail-fast**；
- ✅ 正确的诊断入口是仓库自带的 **torch profiler/CUPTI（`/start_profile`）**，可给出图内每核 CUDA 时间（本轮归因全靠它）。

---

## 24. <6ms 目标：两把**零显存**刀的全部机制与实测（2026-10-10 晚）

### 24.1 第一刀：MoE GEMM 的 `config_index`（tactic/tile）
- **契约**（`resolveMoeTileAndConfig`）：`config_index = [tile_N, config]`；任一为 −1 则回落 `selectDefaultTileN(...)`；
- **现状**：`kernels/ops/moe/trtllm_lora_temp/core.py:492` 恒传 `[-1,-1]` ⇒ **两个 LoRA MoE GEMM 从未 autotune**；
- **tile 候选静态表**：`mSupportedTileNums = {8,16,32,64,128}`；
- **注入通道（已上线、零重建）**：模块级 env `SGLANG_LORA_MOE_TACTIC="<tile>,<config>"` + `/tmp/lora_tactic.txt` 热切（1s 缓存）；默认行为不变（`GATE3 OK [64,3]`）。
- **实测（稳态口径 minc=64、剔前后 15%、n=96）**：
  | 组 | 满批 p50 (tok/s) | **TPOT p50 (ms)** | 样本 |
  |---|---|---|---|
  | 默认 `[-1,-1]` | 8115.8 | **7.964** | 16 |
  | **`tile=64, config=0`** | **8754.4** | **7.311（−8.2%）** | 13 |
  | `tile=64, config=1` | 8362.8 | 7.673 | 14 |
  ⇒ **tactic 是真杠杆但不足**（需 ≈−24%）。

### 24.2 第二刀：LoRA VE（shrink/expand）核的 CSGMV 配置
- **机制**（`python/sglang/kernels/ops/gemm/lora_tuning_config.py`）：
  - 目录：`$SGLANG_LORA_CONFIG_DIR`（或包内 `csgmv_configs`）下的 `triton_<ver>/`；
  - 文件名：`lora_{shrink|expand},K={K},R={R},S={S},device={device}.json`；
  - 格式：`{ "<chunk_size(BLOCK_M)>": {"BLOCK_N":…,"BLOCK_K":…,"num_warps":…,"num_stages":…} }`；
  - **`get_lora_configs` 带 `@functools.lru_cache`** ⇒ 改配置**必须重启**服务；
  - 我们命中的是**硬编码默认**：shrink `{BLOCK_N:16, BLOCK_K:256}`、expand `{BLOCK_N:64, BLOCK_K:16}`（无 warps/stages）。
- **真实形状**（服务日志）：`shrink K=576 / K=5120`、`expand K=1152 / K=5120`（R=16，S=1,2）；`get_device_name()` 返回 **NVIDIA L20D**（⚠️ 本机是 B300，说明设备名受环境/缓存影响，配置文件名须以**服务进程内**取到的为准）。
- **做法**：自写 8 个 JSON（4 形状 × S=1,2，覆盖 chunk 8..512）到 `/root/lora_csgmv_cfg/csgmv_configs/triton_3_7_1/`，`export SGLANG_LORA_CONFIG_DIR=/root/lora_csgmv_cfg` ⇒ 逐候选重启扫 `默认/32/64/128/(32,64)`（脚本 `/root/ve_sweep.sh`、生成器 `/root/gen_csgmv_cfg.py`）。
- **该刀的分量**：CUPTI 归因里 `_chunked_lora_shrink` 45.4ms + `_chunked_lora_expand` 26.5ms = **72ms/40 步 ≈ 1.8 ms/step**（每层 4 次调用）。

### 24.3 判定与预期
- 最优组合（tactic + VE 配置）跑 `/root/verify_c64.sh`（**n=256**、同批 `conc=64/out=4096/剔前后 15%`）⇒ **TPOT p50 < 6.000 ms ⇒ 达标固化（零额外显存、零核改动）**；
- **诚实预期**：第一刀 −8.2% + 第二刀乐观 0.5–1.0 ms ⇒ **≈6.3–6.8 ms，大概率仍差 6.0 ms** ⇒ 届时**如实宣布本侧杠杆不足**，只剩上游带 `gate_up` LoRA delta 钩子的融合核 / 自研 tcgen05 融合核（数周级）。

### 24.4 本轮操作教训（补 §23）
⑥ 扫描必须**先确认服务健康、且 `wait_ready` 失败要 fail-fast**；⑦ `task_kill` 之后务必**检查远程是否留有僵尸脚本**（曾每 ~8 分钟重启一次，害得服务永远起不来）；⑧ 比较吞吐/TPOT **必须取稳态平台段**（同一 run 的头尾段不可比）。

## 25. 两把零显存刀：第二刀（LoRA VE CSGMV）结果 + C=64 正式验收（2026-10-10 晚）

### 25.1 第一刀（MoE GEMM tactic）——完整实测
稳态口径：`minc=64`（只看打满并发）、剔除窗口前后 15%、`n=96`、`seed=777`。

| 组 | 满批 p50 (tok/s) | **TPOT p50 (ms)** | 样本 |
| --- | --- | --- | --- |
| 默认 `[-1,-1]`（**从未 autotune**） | 8115.8 | 7.964 | 16 |
| **`tile=64, config=0`** | **8754.4** | **7.311（最优，−8.2%）** | 13 |
| `tile=64, config=1` | 8362.8 | 7.673 | 14 |
| `tile=32 / 128 / 16` | — | ⚠️ 被我并发启动的 VE 扫描打断（`RemoteDisconnected` → `Connection refused`/503）⇒ 用 `/root/tile_hot.sh` **免重启热切**串行重跑 | — |

- 另有 client_agg：`t64_c0=6533`、`t64_c1=6256`、`t64_c2=6269`。
- **机制**：overlay 契约 `config_index = [tile_N, config]`，任一为 −1 则回落 `selectDefaultTileN`；我们的
  `kernels/ops/moe/trtllm_lora_temp/core.py:492` **恒传 `[-1, -1]`** ⇒ 两个 LoRA MoE GEMM **从未 autotune**
  （stock/base 融合核族按形状 autotune）。tile 候选静态表 `mSupportedTileNums = {8,16,32,64,128}`。
- **注入通道（零重建）**：模块级 env `SGLANG_LORA_MOE_TACTIC="<tile>,<config>"` + `/tmp/lora_tactic.txt`
  热切（1s 缓存，见 `_lora_tactic_hotswap_v3()`）；**文件/环境缺失 ⇒ 回落 `[-1,-1]`，默认行为不变**。

### 25.2 第二刀（LoRA VE 核 CSGMV 配置）
- **机制**：`python/sglang/kernels/ops/gemm/lora_tuning_config.py`；`get_lora_configs()` 带
  `@functools.lru_cache` ⇒ **改配置必须重启服务**。目录 `$SGLANG_LORA_CONFIG_DIR/csgmv_configs/triton_<ver>/`，
  文件名 `lora_{shrink|expand},K={K},R={R},S={S},device={device}.json`，
  内容 `{ "<chunk_size(BLOCK_M)>": {"BLOCK_N":…,"BLOCK_K":…,"num_warps":…,"num_stages":…} }`。
  命中形状：`shrink K=576 / K=5120`、`expand K=1152 / K=5120`（R=16，S=1,2）。
- **做法**：自写 **8 个 JSON**（4 形状 × S=1,2，chunk 8..512）到 `/root/lora_csgmv_cfg/csgmv_configs/triton_3_7_1/`，
  `export SGLANG_LORA_CONFIG_DIR=/root/lora_csgmv_cfg`，逐候选重启扫 `默认/n32/n64/n128/n32k64`。
- **生效证据**：日志由 `no tuned config, using defaults` → `Using LoRA … config from /root/lora_csgmv_cfg/…` +
  `tuned config {...}`。
- **该刀分量**（CUPTI / 40 步）：`_chunked_lora_shrink` 45.4ms + `_chunked_lora_expand` 26.5ms = **72ms**。

**扫描结果（客户端口径 ms/tok，*仅* 用于候选间选优；含排队，不可与服务端 TPOT 直接比）**：

| 候选 | BLOCK_N | 请求数 | 失败 | 均值 ms/tok | p50 ms/tok | 吞吐 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| ve_default | 16 | 96 | 0 | 8.993 | 8.748 | 5700.9 |
| ve_n32 | 32 | 96 | 0 | 8.748 | 8.762 | 5856.0 |
| ve_n64 | 64 | 96 | 0 | 8.758 | **8.639**（p50 最优） | 5571.6 |
| ve_n128 | 128 | 96 | 0 | 9.055 | 9.147（最差） | 5575.2 |
| ve_n32k64 | 32,64 | 96 | 0 | 8.805 | 8.905 | 5627.6 |

**⇒ 全 5 候选跑毕（16:31:37–16:54:54，每候选 96 请求 / 0 失败）**：p50 极差仅 `8.639–9.147 ms/tok`（≈5.6%），
**BLOCK_N=64 为 p50 最优**；均值最优为 `n32`（8.748）。**VE 刀的量级确认只有 ±2–3%**，与 CUPTI「VE 仅占
5% 核时间」一致 —— 它不可能独自把 7.311 ms 拉到 6.000 ms。

- ⚠️ 本次扫描全程 `tactic` 文件为空 ⇒ 所有候选回落默认 `[-1,-1]`（**候选间一致，干净的 VE 对比态**）。
- ⚠️ 扫描脚本每候选重启都会 `: > pd_decode.log`（`exp_single.sh` 所为）⇒ 服务端口径事后不可用，只能走客户端口径。

### 25.3 C=64 正式验收（`/root/verify_c64.sh <tag>`）
- **口径**：`conc=64`、`n=256`、`out-tokens=4096`、`seed=777`；**同一次启动内先 base 后 moe_rnd**
  （`bench_chat3.py --win` 写 `RUN_BEGIN/RUN_END`，`steady3.py <win> <log> 15 15 64` 出服务端 TPOT p50）。
- **判据**：`moe_rnd` **服务端 TPOT p50 < 6.000 ms**。
- **为何必须"同一次启动"**：任何重启都会截断 `pd_decode.log`，跨重启窗口匹配不到日志样本（实测
  `(平台段样本不足 0；窗口 69s)`）⇒ 只能在同一启动内连续测 base/rnd。
- **结果 ①（tag=`best`，16:55:14–16:59:20；判定未达标）**：

  | 组 | 满批 p50 (tok/s) | 满批 p90 | **TPOT p50 (ms)** | TPOT p90 | 样本 |
  | --- | --- | --- | --- | --- | --- |
  | `best_base`（无 LoRA） | 12022.5 | 13472.3 | **5.323** | 6.243 | 39 |
  | `best_rnd`（LoRA `moe_rnd`） | 7829.4 | 8672.4 | **8.176** | 8.923 | 28 |

  ⇒ **`moe_rnd` 服务端 TPOT p50 = 8.176 ms > 6.000 ms ⇒ 未达标。**（客户端口径交叉验证：
  `best_base` 5.560 均值 / 5.419 p50 ms/tok；`best_rnd` 8.153 / 8.045 ms/tok；各 256 请求 0 失败。）

  ⚠️ **但这一轮并**不是**最优组合**：服务仍带着 VE 扫描 shell 导出的
  `SGLANG_LORA_CONFIG_DIR=/root/lora_csgmv_cfg`，于是 LoRA GEMM 用的是扫描【最后一个】候选
  `n32k64`（扫描里第 2 差档，p50 8.905 ms/tok）—— **而不是第一刀 7.311 ms 所用的内建 CSGMV 默认**。
  ⇒ 改为 `/root/accept_final.sh`（task `5c673346`）：**从免重启 tile 扫描里自动选出冠军 tile**，
  再 `unset SGLANG_LORA_CONFIG_DIR` 重启为**内建 CSGMV 默认**，跑 `verify_c64.sh best2`。
  （原先固定 `tactic=64,0` 的 `accept2_after_tile.sh`／task `e4f69ea0` **已停用** —— 两个原因：
  ① tile 扫描显示 `tile=16/32` 显著优于 `64`，固定 64 不是最优；② 它会抢跑重启，**截断 tile 扫描
  所需的 `pd_decode.log`**，毁掉免重启扫描的服务端口径数据。）

- **结果 ②（tag=`best2`，17:08:50–17:12:59）**：`tile=32,0`（tile 扫描的客户端吞吐冠军）+ **`SGLANG_LORA_CONFIG_DIR` 已 `unset`（内建 CSGMV 默认）**，同一启动内先 base 后 moe_rnd：

  | 组 | 满批 p50 (tok/s) | 满批 p90 | **TPOT p50 (ms)** | TPOT p90 | 样本 |
  | --- | --- | --- | --- | --- | --- |
  | `best2_base`（无 LoRA） | 12472.0 | 13675.3 | **5.131** | 6.216 | 45 |
  | `best2_rnd`（LoRA `moe_rnd`） | 7637.1 | 8831.3 | **8.402** | 9.308 | 40 |

  ⇒ **`moe_rnd` 服务端 TPOT p50 = 8.402 ms > 6.000 ms ⇒ 未达标。**（客户端交叉验证：
  base 5.620 均值 / 5.445 p50；rnd 8.408 / 8.260 ms/tok；各 256 请求 0 失败。）

  ⚠️ **重要口径更正**：**n=96 的短跑（第一刀 7.311 ms、tile 扫描 7.25–7.42 ms）系统性偏乐观**
  —— 在 n=256 的持续压测下，两轮验收均为 **8.18–8.40 ms**（`tile=64,0` → 8.176 / `tile=32,0` → 8.402）。
  ⇒ **n=96 的候选排序不能外推到 n=256**（这也解释了 tile 扫描的"冠军"在 n=256 下并未胜出）。

### 25.3.1 最终判定（本侧）

**在 C=64、n=256 的持续负载下，LoRA 请求 TPOT ≈ 8.2–8.4 ms；判据 6.000 ms 需再降 ≈ 29%。**
本侧两把**零额外显存、零核改动**的刀（MoE GEMM tactic + LoRA VE CSGMV）**已全部用尽且量级不足**。

### 25.4 结论与判定

**判定：C=64 下 LoRA 请求 TPOT 未达 6.000 ms —— 本侧「零额外显存、零核改动」的两把刀**杠杆不足**。**

| 口径 | TPOT p50 (ms) |
| --- | --- |
| base（不带 LoRA） | 5.233 |
| LoRA 基线（默认 tactic `[-1,-1]`，**从未 autotune**） | 7.964 |
| **LoRA 最优（第一刀 `tile=64, config=0`）** | **7.311（−8.2%）** |
| 第二刀 VE（CSGMV）天花板 | CUPTI 占比 `(45.4+26.5)/1423.8 ≈ 5%` ⇒ 即便全消也仅 ≈ −5% ⇒ ≈ **6.95** |
| **判据** | **≤ 6.000** ⇒ 尚差 ≈ **−17.9%** |

**⇒ 两刀叠满（tactic −8.2% + VE ≤ −5%）仍不足以把 LoRA TPOT 从 7.311 ms 压到 6.000 ms。**
（第二刀实测佐证：3 个已测候选 `ve_default/ve_n32/ve_n64` 的客户端口径均值 8.99/8.75/8.76 ms/tok、
p50 8.75/8.76/8.64 ms/tok —— 仅 ≈±2%，与「VE 只占 5% 核时间」一致。）

### 25.5 唯一剩余路径（数周级；须上游或自研）
1. **上游**：提供带 `gate_up` LoRA delta 钩子的**融合核**（把 delta 融进 GEMM1 的 epilogue，省掉
   独立 permute/quant/act 链）；
2. **cubin**：补 `Bmm_MxE4m3_MxE2m1MxE4m3` 的 **tile-8/16 M-tile** 变体（我们现跑 `t128x8x512u2`）；
3. **自研 tcgen05 融合核**：复用仓库既有 `dev_kernel.cu` / `fused_activation_mxfp8_quant.cuh` /
   VE delta 核，把 `permute → gate_up GEMM → act+delta → quant → down GEMM → finalize` 合成单核。

**已实测/论证关闭的路线（勿再投入）**：离线合并（MXFP4 重量化噪声 9.4e-4 ≈ 4%|W| 会淹没 <10%|W| 的真实 delta）、
padding（masked vs padded 仅 1.05×，该 GEMM 是**权重带宽受限**）、限并发（**用户明确否决**）、
双流调度重叠（+2.4%）、delta 融进激活/量化核（消融实测 ≈0）、激活换 bf16（stock 形态，
launcher 无该分支、需新写 C++）。

### 25.6 附带发现：`tile_N` 最优在 16/32/128（非 64），但**全部落在 ≈7.3 ms 平台**

免重启热切换（`/tmp/lora_tactic.txt`，1s 缓存）跑 `config=0` 的 tile 候选（**n=96、conc=64、同一次启动、`pd_decode.log` 未截断** ⇒ 服务端口径可用）：

**客户端口径（含排队，仅候选择优）**

| 候选 | tile,config | n | bad | mean ms/tok | p50 ms/tok | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| `th_16_0` | 16,0 | 96 | 0 | 7.455 | **7.406** | 6409.6 |
| `th_32_0` | 32,0 | 96 | 0 | 7.534 | 7.537 | **6556.2** |
| `th_128_0` | 128,0 | 96 | 0 | 7.805 | 7.702 | 6049.4 |
| `th_64_1` | 64,1 | 96 | 0 | 7.829 | 7.785 | 6065.8 |

**服务端口径（`steady3.py <win> <log> 15 15 64` —— 官方判据）**

| 组 | 满批 p50 (tok/s) | 满批 p90 | **TPOT p50 (ms)** | TPOT p90 | 样本 |
| --- | --- | --- | --- | --- | --- |
| `th_16_0` | 8703.4 | 9265.9 | 7.401 | 8.212 | 14 |
| `th_32_0` | 8630.3 | 9513.4 | 7.416 | 8.148 | 15 |
| `th_128_0` | 8949.2 | 9404.4 | **7.248** | 8.577 | 14 |
| `th_64_1` | 7796.3 | 8929.8 | **8.209**（最差） | 8.864 | 17 |

**⇒ 四个候选跑毕**：`config=0` 的 16/32/128 三个候选服务端 TPOT p50 全落在 7.25–7.42 ms（极差 <2.4%；
样本 14–15 条，属同一平台），与第一刀 `tile=64, config=0` 的 7.311 ms 一致；而 **`config=1`（`th_64_1`）
明显更差（8.209 ms）⇒ tiling 配置维度取 `config=0` 正确**。⇒ **tile 维度在 16/32/128/64 之间已饱和**，
继续调 tile **无法**把 7.3 ms 拉到 6.0 ms。

**这也再次印证 §25.4 的判定**：本侧（零额外显存、零核改动）两把刀——tactic（−8.2%）+ VE（±2~3%）——
**叠满也只能到 ≈7.2–7.3 ms，距判据 6.000 ms 差 ≈ −17%**。

---

## 26. 作用域：`--lora-target-layers` —— 后 20 层模式下前 20 层零开销（2026-10-12）

**本次排查发现的浪费**：`has_active_lora` 是 **batch 级**的（`lora_manager.py` 里 `batch_info.has_active_lora = any(lora_ranks[wi] > 0)`），而 `init_lora_modules` 按**模块名命中 `target_modules`** 就把**全部 40 层**的 MoE 包成 `FusedMoEWithLoRA`。于是：

- 只要 batch 里有**一个**请求挂适配器 ⇒ **全体 MoE 层进分解链**；
- `moe_rnd` / `moe_zero` 实际只覆盖 **20–39**（读 safetensors header 核实：120 张量 / 20 层；`moe_tinker` 是 722 张量 / 40 层）⇒ **0–19 层 delta 恒 0，却照付整条分解链的开销**；
- `_adapter_weight_layers()` 这份「本层到底有没有适配器权重」的信息，旧实现**只用于 kv_shared 加载校验**，compute 路径从不使用。

**修法**：新增 `--lora-target-layers`（语法 `20-39` / `20,21,39` / 不填 = 全量，行为不变）。在 `init_lora_modules`（**唯一包装入口**）做层过滤：范围外的层**完全不包装** ⇒ 保持原生 `FusedMoE` ⇒ 永远走 stock 融合核，**不分配 LoRA buffer、不做 VE delta、不进分解 GEMM**。另加载期守卫：适配器权重落在范围外 ⇒ **响亮 `raise`**，不静默降级。

**为什么这才叫零开销**：走分解链的**唯一前提**是该层被包成 LoRA 模块。「不包装」= 结构上根本不进入这条路径，不是「算出 0 再加进去」。

**顺带查实**：`SGLANG_LORA_TWO_STREAM` 是**死变量** —— 全树 6 处只有注释/docstring，没有任何 `os.environ.get` / `EnvBool` 读取点；双流真正的开关是 `SGLANG_EXPERIMENTAL_LORA_OPTI`（`lora/layers.py` 里 `install_two_stream_overrides`）。启动脚本里写 `SGLANG_LORA_TWO_STREAM=0` 关不掉任何东西。

**实测**（服务端口径，`conc=64 / n=256 / out=4096 / ignore_eos / seed=777`，只取 `#running-req ≥ 64` 并剔前后 15%）：

| 模式 | base p50 (tok/s) | `moe_rnd` p50 (tok/s) | TPOT p50 (ms) | **保留率 rnd/base** |
| --- | --- | --- | --- | --- |
| **后 20 层（本次优化）** | 12176.2 | **8789.6** | 5.256 → **7.281** | **0.722** |
| 全量（既有基线 §M4） | 12244.3 | 7739.5 | 5.233 → 8.269 | 0.632 |

⇒ 挂 LoRA 的相对损失由 **−36.8% 收窄到 −27.8%**；两组 base 仅差 0.6%，故该对比有效（跨时段，按 §3 红线应视为量级）。客户端侧同向：后 20 层模式 base 4.679 ms/tok、`moe_rnd` 7.636 ms/tok，各 256/256 成功、`bad=0`。

**⚠️ 测试坑**：PD 分离下适配器必须在 **P(30100) 与 D(30200) 两侧都加载**；只加载一侧 ⇒ 经 smg 的 LoRA 请求全部超时（实测 256/256 bad、wall=720s），而服务日志无显式报错。
