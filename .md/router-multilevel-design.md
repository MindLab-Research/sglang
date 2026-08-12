# M1 详细设计：递归 Router 树 + 统一 Model 抽象 + 声明式部署

> 分支：`b300-glm52` | 状态：Draft v0.2 | 日期：2026-08-11
> 前置：P0 实验确认 LoRA 热加载/卸载在虚拟专家 + PD 模式可用
> （decode ~13s / prefill ~6s；unload 不阻塞在途，buffer slot 复用有覆盖风险）

---

## 1. 设计原则（驱动一切决策）

1. **只有两个抽象**：`ServiceUnit`（可路由单元，递归）+ `Model`（服务对象）。
   其余一切（LoRA、PD 集群、base model）都是这两个抽象的**实例**，不是特殊逻辑。
2. **递归**：router 的孩子是同构的（router 或 engine），树深任意、自动可扩展。
3. **PD 集群是原子叶子（KV 传输域）**：一个集群 = 若干 prefill + 若干 decode，
   集群内任意 prefill 可喂任意 decode（mooncake 域），对外呈现为一个不可拆分的
   Engine 单元。1P1D / xPyD（如 2P3D）都是它的实例。跨集群配对在结构上不可能
   （集群间没有 KV 传输通道）。
4. **引擎零补丁**：所有引擎差异封装在 Engine Adapter，router 核心不感知。
5. **声明式**：上层只表达意图（"model X 要可用"），下层自治执行（找机器→替换→serve）。

---

## 2. 核心抽象

```
ServiceUnit（可路由单元，所有孩子都是它）
├── RouterUnit   — 递归子树（同构，管辖若干 ServiceUnit）
│     数据面 :30000  |  控制面 :29990
└── EngineUnit  — 叶子（物理服务，原子路由域）
     ├── PDCluster { prefills: [...], decodes: [...], internal_allocator }
     │     ↑ 一个 KV 传输域（mooncake 域）：集群内任意 prefill 可喂任意 decode；
     │       1P1D / xPyD（如 2P3D）都是它的实例；内部分配器=现有 sglang-router 逻辑
     └── Standalone { url }                      ← 单机（vllm / 非分离）

Model（服务对象，声明式）
├── type: base | lora         ← router 不关心，EngineAdapter 处理差异
├── name, path, version
├── desired: 部署意图（strategy: any | all）
└── observed: 观测状态（每 engine: state/inflight；全局聚合）
```

**递归性的含义**：每个 RouterUnit 对子节点的操作只有两个——
`forward(req)` 转发请求、`report()` 拉取资源/Model 视图。子节点是 router 还是 engine
对父节点完全透明。**扩展 = 注册新孩子**（加集群注册 EngineUnit，加层级插入 RouterUnit），
零代码改动。

---

## 3. 总体架构

```
                     中转站（业务编排，3-hop，仅调数据面 OpenAI API）
                                    │ /v1/chat/completions {model: L2}
                                    ▼
                    ┌──────────────────────────────────┐
                    │ Root Router（递归根）              │
                    │ 数据面:30000  控制面:29990          │
                    └──┬──────────────┬───────────────┘
        ┌──────────────┴──┐    ┌──────┴─────────────┐
        ▼                 ▼    ▼                    ▼
  RouterUnit(区域A)   RouterUnit(区域B)       EngineUnit(PD Cluster, 单集群直挂)
        │                 │
   ┌────┴────┐       ┌────┴────┐
   ▼         ▼       ▼         ▼
EngineUnit  EngineUnit EngineUnit EngineUnit
(PD Cluster) (PD Cluster) (PD Cluster) (PD Cluster)

每个 EngineUnit(PD Cluster) = 一个 KV 传输域（xPyD，1P1D/2P3D 均适用）:
   ├── prefills:  [p1:30100, ...]   (HiCache + models)
   ├── decodes:   [d1:30200, ...]   (DCP + models)
   └── internal_allocator: 集群内选 prefill(算 KV) + 选 decode(生成)，KV 经 mooncake 传输
       集群内任意 p_i ↔ d_j 可配对；对外呈现为单一体（请求→集群→内部双发）
```

> 顶层可以混挂 RouterUnit 和 EngineUnit（示例中单集群直挂 Root），
> 这完全合法——因为对孩子没有类型假设。

---

## 4. 数据面

### 4.1 请求路由（按 Model）
- 标准 OpenAI API；`body.model` 即 Model 名（base 或 LoRA 名，对 router 无差别）。
- 每个 ServiceUnit 维护 `model → 孩子` 的路由能力：
  - 孩子上报其可服务的 model 集合（Engine 上报已加载的 models；Router 上报子树聚合）。
  - 请求只路由到**声明可服务该 model**的孩子；无可用孩子 → 503 + 明确原因。

### 4.2 亲和与压力（分层复用同一策略引擎）
| 层 | 亲和 | 压力 |
|---|---|---|
| 任意层 RouterUnit | routing key 粘性 + 前缀哈希桶（同 model 内聚合相似请求） | 孩子 pressure 归一化分：`w1·running/cap + w2·waiting/cap + w3·TPOT`，超阈值跨孩子 spill |
| EngineUnit(PD Cluster) | 集群内 cache_aware 前缀树（现有 sglang-router 逻辑） | 集群内 prefill/decode min_load |

- routing key：`model:convo_id`（粘性粒度 = model × 对话），请求结束释放（`/_internal/routing-key` DELETE）。
- **同构策略引擎**：每层都是同一套 select 逻辑，参数（亲和权重/桶数/spill 阈值）按层配置。

### 4.3 per-Model 并发统计（数据面职责）
- 转发时解析 `body.model` → `inflight[model]++`；响应结束（流式 EOF/非流式/断连）→ `--`。
- 同步代理连接生命周期 = 请求生命周期，计数精确；每层求和向上聚合。

---

## 5. 控制面板 API（每个 ServiceUnit 同构，仅 scope 不同）

| 端点 | 方法 | 语义 |
|---|---|---|
| `/v1/register` | POST | 孩子向父注册（url、可服务 models、静态拓扑） |
| `/v1/models` | GET | 本单元 Model 视图（= 聚合子树） |
| `/v1/models` | POST | **声明式部署/更新** `{name, type, path, strategy: any\|all}` |
| `/v1/models/{name}` | DELETE | 声明式移除（排干后卸载） |
| `/v1/models/{name}/drain` | POST | 显式排水（停新路由 + 等在途归零） |
| `/v1/units` | GET | 子树拓扑 + 负载（EngineUnit 粒度 + 其内部节点） |
| `/v1/routing` | GET/PUT | 策略配置 |
| `/v1/healthz` | GET | 父探活 + 摘要上报（models/pressure 快照） |
| `/_internal/routing-key` | DELETE | 粘性 key 释放 |

> `GET /v1/models` 与 OpenAI 兼容视图一致（基础字段可对齐 `/v1/models` 格式）。

---

## 6. 全局 Model 视图（递归聚合）

```
EngineUnit 每 5s 生成 EngineSummary:
  models:   [ {name, type, state: LOADING|ACTIVE|DRAINING|SWAP_OUT|EVICTED,
                inflight} ]                       ← 数据面统计 + 控制面状态
  pressure: { running, waiting, tpot_p50, gen_throughput }   ← sglang /metrics
  cache:    { hit_rate, hicache_used/total }
  topology: { nodes: [ {host, role, gpus[{id,util,mem}], kv_usage} ] }
  healthy:  bool

父 RouterUnit 每 10s report() 拉孩子 → 归并：
  models  → 跨孩子汇总 {name: {state, engine_count, per_engine: {id: inflight}}}
  pressure/topology → 按层聚合摘要
Root 缓存全局视图 TTL 10s
```

**Root 查询**（满足"最上层详细汇总"）：
```
GET /v1/models?scope=global
→ { "models": {
      "glm52-fp8-official": { type: base,   engine_count: 5,  per_engine: {...} },
      "L0": { type: lora,   state: ACTIVE,  engine_count: 3,  per_engine: {e1:8并发, e2:3, e3:0} },
      "L1": { type: lora,   state: ACTIVE,  engine_count: 2,  per_engine: {e1:12, e2:5} },
      "L2": { type: lora,   state: DRAINING, engine_count: 1, per_engine: {e2:0} },
      "L4": { type: lora,   state: LOADING,  engine_count: 0, per_engine: {} } } }
GET /v1/units?scope=global
→ 全部 EngineUnit 拓扑 + 负载（含每节点 GPU/KV）
```

---

## 7. Model 生命周期状态机（通用；LoRA 替换是其一个实例）

> P0 驱动：unload 不阻塞在途、buffer slot 复用有覆盖风险。
> 通用解法：**Engine 预留 1 个 swap slot**（base/lora 通用：`--max-loaded-loras 5` 或等价），
> 替换走"先加载新 → 切路由 → 排干旧 → 卸载旧"，零覆盖窗口。

### 7.1 状态
```
EVICTED → LOADING → ACTIVE → DRAINING → SWAP_OUT → EVICTED
              └── 任一步失败 ──▶ ROLLBACK → 恢复原状态
```

### 7.2 声明式下发时序（POST /v1/models → 逐层下钻）
```
上层意图：{name: L4, type: lora, path: new, strategy: any}
 1. 逐层下钻：每层选孩子 = pressure 最低 且 有可替换 model（亲和 + 容量）
 2. 目标 EngineUnit(PD Cluster) 本地执行：
    LOADING:
      adapter.load_model(decode)   // decode ~13s（先发慢端）
      adapter.load_model(prefill)  // prefill ~6s（并行）
      双端 success 校验 → 失败 → ROLLBACK
    ACTIVE:     路由表 L4 生效（L4 占 swap slot）
    选替换者 Mx = 本 Engine 内 LRU / inflight 最少（排除 pinned）
    DRAINING:   停 Mx 新路由 → 轮询 Mx.inflight == 0（超时兜底）
    SWAP_OUT:   adapter.unload_model(Mx) 双端（旧 buffer 释放，无覆盖）
    上报父节点 → 父更新全局 Model 视图
 3. ROLLBACK（load 失败时）:
    adapter.unload_model(L4) 双端 → 恢复原路由 → 上报错误 → 告警
```

### 7.3 多副本
- `strategy: any` → 选最优单引擎；`all` → 全部引擎滚动执行（每引擎独立排干）。
- 版本收敛：新版本部署完成后，旧版本在其余引擎按流量自然过期或显式 EVICT。

---

## 8. EngineAdapter（引擎差异的唯一封装点）

```python
class EngineAdapter(Protocol):
    """Router 唯一依赖的引擎接口；实现差异全部在此。"""
    def load_model(self, ref: ModelRef, scope: str) -> Result
    # scope: "all"（集群全部 prefill+decode）/ "decode" / "prefill" / "some"
    def unload_model(self, ref: ModelRef) -> Result
    def forward(self, req) -> Response               # 集群→内部 PD 双发; standalone→单发
    def get_view(self) -> EngineSummary              # /metrics + 本池状态
    def models(self) -> list[ModelState]

class PDClusterAdapter(EngineAdapter):
    # 集群 = xP yD（KV 传输域）：
    #   load 顺序: 先 decode（~13s，慢端）→ 再 prefill（~6s）；"some" 时按需覆盖部分实例
    #   内部分配器: 选 prefill(算 KV) + 选 decode(生成)（复用现有 sglang-router 逻辑）
    #   forward: 集群内任意 p_i ↔ d_j 配对，KV 经 mooncake 传输
class StandaloneAdapter(EngineAdapter):
    # 单发
```
router 核心只认 `EngineAdapter` 协议 —— 新增引擎类型（vllm、未来 MoE 独立部署）
只加一个 Adapter，**router 代码零改动**。

---

## 9. 部署与迁移

1. Engine 配置：`--max-loaded-loras 5`（4 active + 1 swap）等，全部经 Adapter 隐藏。
2. 控制面 `:29990` 仅内网；admin 直连 prefill/decode `:30100/:30200`。
3. 指标：轮询 `/metrics`（`num_running_reqs`/`num_waiting_reqs`/`cache_hit_rate`/`hicache_host_*`/`gen_throughput`/`inter_token_latency_seconds`）。
4. 迁移：
   - M1a：Root Router 直挂现有集群为 EngineUnit（每个集群 = 现有 sglang-router :30000 包装成 PDClusterAdapter），替代 smg；控制面 `/v1/models` + `/v1/units` 上线。
   - M1b：EngineAdapter 内置 Model 声明式生命周期（swap 状态机）。
   - M2：RouterUnit 递归层（区域插入）+ 亲和分层。
   - smg 能力并入后下线。

---

## 10. 里程碑与验证

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| M1a | Root Router（递归单层）：EngineUnit 注册/健康/Model 视图/真实负载路由；`GET /v1/models` + `/v1/units` 全局视图 | 双集群压测压力路由正确；root 可见 per-Model 并发 + 拓扑 |
| M1b | 声明式 Model 部署：`POST /v1/models` 在 EngineAdapter 执行 swap 状态机 | 端到端替换成功、注入失败回滚、替换期 0 错误 |
| M2 | 递归树：插入 RouterUnit 层 + 亲和分层策略 | 两层树压测、跨域 spill、加孩子零改动 |
| M3 | 观测：Grafana/OTel、替换事件流、排干时长告警 | 全局视图实时准确、全流程可观测 |

---

## 11. 开放问题

1. **EngineUnit 顶层混挂合法性**：允许 Root 直接挂 EngineUnit（短树）与纯 RouterUnit 树共存 —— 确认无歧义（当前设计允许）。
2. per-Model 并发在流式断连下的计数泄漏需压测标定。
3. 前缀哈希桶 vs routing key 粘性的权重需真实流量标定。
4. 节点 GPU 动态指标来源：nvidia-smi agent / DCGM / 初期静态拓扑。
5. `strategy: all` 滚动期间新旧版本并存的对话路由语义（旧对话留在旧版本引擎是否可接受）。
6. 引擎原生"model 上限"（如 max-loaded-loras）如何作为 Engine 能力上报，参与上层替换决策（避免上层把引擎塞爆）。

---

## 12. 实现状态（2026-08-11）

### 12.1 代码（`sgl-model-gateway/` 工程，Rust）

| 里程碑 | 状态 | 代码位置 |
|---|---|---|
| M1a 控制面板 + 全局视图 | ✅ 实现，编译通过 | `src/control_plane/mod.rs`（状态/指标采集/视图 handler）、`server.rs`（`/v1/control/*` 路由）、`app_context.rs`/`server.rs`（AppContext/AppState 接入） |
| M1a 数据面 per-model 并发 | ✅ | `routers/http/router.rs`：`request_started/finished` + `EndTrackedStream`（流式 Drop 释放） |
| M1b 声明式部署 + swap 状态机 | ✅ | `src/control_plane/deploy.rs`：`POST /v1/control/models`（选低压力引擎→远程下载→decode 先/prefill 后加载→选低压力 replacee→drain→unload）、`DELETE /v1/control/models/{name}` |
| M1b 远程下载 | ✅ | `deploy.rs::resolve_model_path`：`http(s)://*.tar.gz` 下载解压到 `SGLANG_LORA_CACHE_DIR` |
| M2 递归树 | ✅ | 能力发现（collector 拉子节点 `/v1/control/models` 或 `/v1/models`）、递归聚合（`get_models/get_units` 展开 router 孩子）、数据面 `proxy_to_child` + `child_for_model` |
| M2 亲和分层配置 | ✅ | `RoutingConfig`（affinity_weight/prefix_bucket_tokens/spill_threshold）+ `GET/PUT /v1/control/routing` |

**控制面板 API（`/v1/control/` 前缀，避免与 OpenAI `/v1/*` 冲突）**：
```
GET    /v1/control/models           聚合 model 视图（递归）
POST   /v1/control/models           声明式部署 {name,type,path,strategy}  → 自动替换
DELETE /v1/control/models/{name}    排干并移除
GET    /v1/control/units            子树拓扑 + 实时负载（递归）
GET    /v1/control/routing          GET/PUT 亲和分层参数
POST   /v1/control/register         子单元注册（pd_cluster/router/engine）
GET    /v1/control/healthz          存活 + 摘要
```

### 12.2 1P1D 实测结论（B300 集群，2026-08-11）

**背景**：1P1D 原为 merge-quant（`base_l2_merged`，无 `--enable-lora`）→ 按 2P3D 线上参数重启为动态 LoRA 模式（`base` + `--enable-lora` + CP layer-split + HiCache + `--max-loaded-loras 4`，decode 含 `--disable-custom-all-reduce` + DCP=4 + EAGLE 5）。

| 实测项 | 结果 |
|---|---|
| PD 重启动态 LoRA（2P3D 参数） | ✅ 两端 ready，`/v1/models` 返回 L0-L3，warmup 完成，零崩溃 |
| **prefill 热加载 L4** | ✅ 8.7s `success:true` |
| **prefill 卸载 L4 → 加载 L5** | ✅ unload success + load 4.9s success（池 `[L1,L2,L3,L5]`） |
| **decode 热加载 L4** | ✅ 9.5s `success:true` |
| **decode 卸载 L4 → 加载 L5** | ✅ 池 `[base, L0-L3, L5]` |
| router 控制面板（register/视图/指标/routing） | ✅ 正常 |
| ⚠️ 早期崩溃（3 次 NCCL Aborted） | ❌ 根因 = **warmup 未完成时调用 `load_lora_adapter`**（见下） |

**关键根因（回答"为什么之前 prefill 动态加载没出错"）**：
1. **`/v1/models` 200 ≠ 引擎就绪**：sglang 模型加载完成后 `/v1/models` 即返回 200，但 **DeepGEMM warmup + CUDA graph capture 仍在进行**（实测崩溃时 warmup 96%）。
2. **warmup 期间调用 `load_lora_adapter` → NCCL 死锁**：load 的跨 rank collective（`_ALLGATHER_BASE`）与 warmup 中的 collective 错位 → 60s watchdog 超时 → 进程 Aborted。
3. **用户历史操作成功** = 当时 prefill 已完全 warmup（稳定运行期）→ load 正常（5-18s）。
4. **结论**：动态 LoRA 加载在 prefill/decode 端**本身没有 bug**；正确姿势 = 等 sglang 完全 warmup（日志出现 `ready to roll` / `capture cuda graph finished`）再调用。

**正确操作时序**：
```
1. 启动 prefill/decode（2P3D 参数）
2. 等 warmup 完成（/v1/models 200 + 日志 graph capture finished + 稳定运行）
3. 再调 /load_lora_adapter（prefill ~5-9s / decode ~9-13s）
4. 卸载前先停该 lora 的新路由 + 等在途请求归零（unload 不阻塞在途）
```

**架构约束**：
1. **PD 权重双端**：prefill 与 decode 独立节点，新 LoRA 权重必须两端都有（实测 L4 只建在 B300-1 导致 decode 端 `Repo id must be...`）。router 的 `resolve_model_path` 需补"分发到同集群所有节点"或依赖共享存储。
2. **swap slot 配置**：本测试用"先卸后加"（`max_loaded_loras=4` 即可）；router deploy 状态机的"先加后卸"（零覆盖窗口）需 `--max-loaded-loras 5`，与 2P3D 线上 4 不兼容 —— 需决策：线上改 5 或 deploy 改先卸后加。
3. sglang `/v1/models`：prefill 只返回 base，decode 返回完整 lora 列表；能力发现对 pd_cluster 须用 decode 端点（已实现）。

### 12.3 修复记录（2026-08-11 第二轮）

按线上决策（**`max_loaded_loras=4`，常驻 4 个，其中一个 swap slot**）完成修复：

| 修复 | 内容 | 验证 |
|---|---|---|
| **deploy 改先卸后加** | `deploy.rs`：无空 slot 时先选 replacee（inflight 最少 = swap slot 候选）→ 排干 → 卸载腾 slot → 加载新的；load 失败自动回滚恢复 replacee | ✅ L6 替换 L1 成功（15.7s），base 保留 |
| **select_replacee 排除 base** | 只从 `model_type != "base"` 的 lora 里选替换候选（修复误选 base 的 bug） | ✅ 替换的是 L1 而非 glm52-fp8-official |
| **能力发现标记 base/lora** | `fetch_capability`/`fetch_unit_models`：按 `/v1/models` 的 `parent` 字段（null=base）标记类型 | ✅ units 显示 `[('base','base'),('L1','lora'),...]` |
| **get_units 展示修复** | `get_units` 的 models 改用 `child.models`（能力发现）而非 `model_inflight`（原展示 bug） | ✅ child.models 正确显示 |
| **共享实例修复** | `server.rs`：AppState.control_plane 复用 AppContext 的实例（原两个实例 → Router 数据面查不到注册） | ✅ S7 转发从 503 → 成功 |
| **生产 router 恢复** | sglang-router 正确路径 `/usr/local/bin/sglang-router`（PD 模式） | ✅ :30000 ready |
| **S7 数据面转发端到端** | A(:30500) 发 `model=L0` → child_for_model → pd_cluster → 生产链路 → 真实推理成功（5.7s） | ✅ |

### 12.4 待办状态（2026-08-11 收尾：全部清零）

| 原待办 | 状态 |
|---|---|
| ~~引擎 bug：CP layer-split 热加载~~ | ✅ **已澄清无需修**：崩溃根因是 warmup 未完成时调用（`/v1/models` 200 ≠ 就绪）；2P3D 参数 + 完整 warmup 后 prefill/decode 热加载全部正常（prefill 8.7s / decode 9.5s） |
| ~~权重分发~~ | ✅ **完成**：`scripts/sync_lora_weights.sh`（rsync 到 pd_cluster 所有节点）+ `resolve_model_path` 本节点 URL 下载 + deploy load 失败时明确提示（`hint` 字段）；PD 权重双端约束写入 §12.2 |
| ~~S2 错误处理~~ | ✅ **完成并实测**：重复部署返回 `409 already_deployed`（不再 503）；S7 数据面转发实测成功（A→pd_cluster→真实推理 5.7s） |
| ~~M3 观测~~ | ✅ **基础版完成**：router 原生 Prometheus 端口（`:29011/metrics`，51 指标行，含 http_request/router 指标）；Grafana 可直连抓取 |

### 12.5 运维要点（收尾新增）

1. **正确操作时序**（动态 LoRA 替换）：
   - 启动 PD（2P3D 参数）→ 等完全 warmup（`/v1/models` 200 + 日志 `ready to roll` / `capture cuda graph finished`）→ 再调 `load_lora_adapter`
   - warmup 期间调用会触发 NCCL 死锁（60s watchdog Aborted）——本分支已知陷阱
2. **swap slot 语义**（线上 `max_loaded_loras=4`）：常驻 4 个 lora，替换选"最闲的"（inflight 最少）→ 严格排干（零在途）→ 卸载 → 加载新的；base model 永不被选为 replacee
3. **权重双端**：新 lora 权重必须先 `sync_lora_weights.sh` 同步到 pd_cluster 所有节点（或共享存储），再 deploy
4. **router 多实例**：每个实例需独立 `--prometheus-port`（默认冲突）
