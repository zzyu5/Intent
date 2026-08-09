# Intent Kernel 编译器：从 4 Kernel 到 10 Kernel 的共享 GPU Realization 与三后端实现报告

## 1. 报告范围与当前结论

本报告覆盖的阶段从 softmax、GEMM、attention、MoE 四类 kernel 已形成三后端活体路径开始，到当前扩展为 10 个 kernel、Triton / cuTile / TileLang 三个后端全部真实运行结束。

当前不是“为 10 个 kernel 写了 10 条翻译路径”，而是形成了下面这条共用编译链：

```text
Python Intent DSL
  │
  │ frontend lowering
  ▼
Canonical Intent Kernel MLIR
  │
  │ common kernel / ABI / region / operation analysis
  ▼
KernelFacts + ScheduleStructure
  │
  │ one shared GPU schedule policy
  ▼
intent_plan.realization(target = "gpu")
  │
  ├── target-neutral GPU machine decisions
  │   axis / ownership / traversal / boundary / residency / primitive
  │
  ├── intent_plan.search_space
  │   legal keys + canonical parameter roles
  │
  ├── Triton projection  ──► intent_triton.*
  ├── cuTile projection  ──► intent_cutile.*
  └── TileLang projection ─► intent_tilelang.*
             │
             │ shared lifecycle + shared Kernel IR traversal
             │ target-local per-op handlers
             ▼
      generated target source
             │
             │ backend compiler / autotuner
             ▼
        real CUDA execution
```

已经得到的可核验结果：

- 10 个 DSL kernel，分为 normalization、contraction、streaming、ragged 四组；
- 10 × 3 = 30 条模型级 GPU repro 全部数值通过；
- 25 条组合连接了仓库中的上游高性能实现，或由上游高性能原语诚实组合出的 baseline；
- 5 条组合在当前 source corpus 中没有等价 standalone baseline，明确报告 unavailable，但 generated kernel 仍实际运行并与 PyTorch reference 对数值；
- realization 和 emission 中没有任何一个新增 kernel 的名字判断；
- 扩展到 10 个 kernel 没有在 realization 或 emission 目录增加 kernel 专用文件；
- 三个后端读取同一份 GPU machine plan，差异只发生在 capability、target projection、语法物化和后端 tuner；
- TileLang 实际运行环境为 0.1.13。

## 2. 四 Kernel 阶段的起点与当时的问题

四个起始 kernel 是：

| 结构 | Kernel | 主要算法形态 |
|---|---|---|
| row reduction | stable softmax | max + exp + sum + normalize |
| tiled contraction | GEMM | M/N tile + K reduction |
| stateful streaming | flash attention | Q tile + K stream + online state |
| ragged / irregular | MoE | expert ownership + indirect gather + two contractions + scatter-add |

四条路径证明了 frontend 与三个 surface 可以工作，但当时仍有几个架构问题：

1. 三个 backend 各自组合 schedule，重复决定 program mapping、axis role、stream 和 ragged staging；
2. “同一台 GPU 的物理事实”仍散落在三个 target 路径中；
3. rowwise padding 证明仍带有 stable-softmax 痕迹；
4. scalar row result 没有完整 target materialization；
5. stream decision 和 ragged relation 带有 singleton 假设；
6. TileLang ragged emission 仍出现 `E`、`R`、`route_offsets`、`member_routes` 等 MoE 名称假设；
7. 四个 kernel 只能说明四个点可运行，还不能证明相同机制能自然组合出新的算法结构。

这一阶段首先统一了工具入口：

```text
examples/run/repro.sh <backend> <kernel>
                    │
                    ▼
          tools/intent-compile/intent-compile
```

`intent-compile` 是唯一 C++ 编译工具。Python frontend 只构造 Kernel MLIR，C++ 在同一进程中完成 realization、target projection 和 target source emission；Python runtime 负责加载并运行已经生成的 source。

## 3. 当前编译器的真实分层

### 3.1 Python frontend：只产生 canonical Kernel MLIR

Python frontend 继续负责：

- Python AST 与 closure/constexpr 解析；
- dtype、shape symbol、domain、region 和 source location 的 lowering 临时状态；
- 在构造 operation 时立即做局部语言检查；
- 直接生成 canonical Intent Kernel MLIR。

Python 中没有独立 typed Kernel IR，没有 Python Physical Plan，没有 Python target emitter。生产入口位于：

```text
python/intent/compiler/pipeline.py
  lower_to_mlir(...)
  run_compiler(...)
  target.materialize(...)
```

### 3.2 Common analysis：把 IR 转成事实，不转成第二套语义 IR

`lib/Target/Common/Analysis/Kernel.cpp` 构造 `KernelModel`。它只包含：

- `KernelABI`；
- canonical node/value ID 索引；
- structured region preorder 与父子关系；
- 唯一 kernel entry。

ABI analysis 验证函数参数与 `intent.parameter_nodes` / `intent.parameters` 一一对应，并保留 view access、shape、dtype、symbol 和 runtime scalar metadata。

随后 `lib/Target/Common/Realization/KernelFacts.cpp` 通过共享 operation traversal 收集 `KernelFacts`：

```text
domain source axes
partition / parallel ownership
value-axis provenance
vector domains
contraction domains and contraction facts
ordered-stream domains and state streams
ragged relation / outer / member ownership
scatter-reduction facts
boundary domains and proven masked-lane fill
whole-view loads
```

这些对象是 analysis facts，不是与 MLIR 平行的 kernel graph。

### 3.3 ScheduleStructure：只描述算法结构要求

`analyzeScheduleStructure(...)` 将 facts 压缩为：

- program domains；
- tiled program domains；
- vector domains；
- contraction domains；
- ordered stream domains；
- ragged ownership；
- scatter reductions；
- 唯一 program root。

它还检查每个 boundary domain 必须已经被 program、vector、stream 或 contraction 机制解释，不能让 target emitter 临时猜测索引含义。

### 3.4 Shared GPU SchedulePolicy：机器决策只做一次

当前唯一 GPU schedule 入口是：

```cpp
FailureOr<ScheduleDecision>
decideGpuSchedule(const KernelFacts &facts);
```

文件位于 `lib/Target/Common/Realization/SchedulePolicy.cpp`。它不接收 target 名称，也不接收 kernel 名称。

`ScheduleDecision` 保存：

| 字段 | 含义 |
|---|---|
| `programRoot` | 哪个 logical parallel region 成为 program root |
| `axes` | domain node、source axis、canonical role、tile role |
| `workerAxes` | logical ownership 映射到哪些 GPU worker axes |
| `traversal` | persistent / grouped / forward / expert-major 等遍历顺序 |
| `mapping` | row-strided / grouped-2d / stream / ragged 等机器机制 |
| `stateStreams` | 同一计划中所有 state-stream operation |
| `raggedRelations` | ragged relation decisions |
| `stages` | contraction pipeline 的 stage DAG |
| `autotuneKeys` | shape-dependent tuning key |
| `autotuneParameters` | target-neutral parameter roles |

当前共有五种共享 mapping：

| Kernel 结构事实 | Shared mapping | Shared traversal | Canonical tuner roles |
|---|---|---|---|
| one program + one vector domain | `row_strided` | `persistent` | fixed-row mechanism，不建 search space |
| two tiled program axes + one reduction axis | `grouped_2d_tiles` | `grouped` | `program_m`, `program_n`, `reduction`, `group_m` |
| query tile + ordered contraction stream | `multi_axis_stream` | `forward` | `query`, `stream` |
| row ownership + ordered stream，无 contraction | `row_stream` | `row_major_forward` | `stream` |
| ragged outer/member + scatter-add | `ragged_stages` | `expert_major` | `ragged_member`, `feature`, `reduction` |

GEMM 与 dual GEMM 都走 `grouped_2d_tiles`；attention 走 `multi_axis_stream`；online softmax 走 `row_stream`；MoE 与 grouped GEMM 都走 `ragged_stages`。这不是按类别分支入口，而是 facts 组合后的结果。

## 4. Masked-lane 证明如何从 softmax 特化变成通用数据流证明

边界 tile 超过真实 extent 时，load 的 padding 值必须保持算法语义。realizer 不允许 emitter 随意选择 zero 或 negative infinity。

当前 `lib/Target/Common/Realization/Proofs.cpp` 使用一个很小的 abstract value lattice：

```text
unknown
zero
negativeInfinity
```

`inferMaskedLaneFill(loaded)` 分别尝试证明 zero 和 negative infinity 能否沿 def-use 链安全传播：

| Operation | 允许的传播 |
|---|---|
| cast | 保持 padded value |
| negate | zero → zero |
| exp / exp2 | negativeInfinity → zero |
| multiply | zero → zero |
| lhs subtract | negativeInfinity → negativeInfinity |
| reduce add | 只接受 zero identity |
| reduce maximum | 只接受 negativeInfinity identity |
| masked store | 终止传播 |

无法证明时直接失败，不会默认填零继续发射。

这项改动同时覆盖：

- stable softmax 的 max/exp/sum；
- LayerNorm / RMSNorm 的 additive reductions；
- logsumexp 的 max 与 sum；
- online softmax 第一遍的 running max 和 exp-sum。

## 5. GPU Machine Plan：resolved decision 与 search space 分离

### 5.1 Plan Build

`lib/Target/GPU/Realization/Plan/Build.cpp` 将共享 `ScheduleDecision` 和 per-op facts 写入同一个 `intent_plan.realization(target = "gpu")`。

主要 Plan operation 为：

| Plan op | 当前保存的物理含义 |
|---|---|
| `device` | CUDA device index 与机器容量事实 |
| `axis` | Kernel domain node、source axis、canonical role、tile role |
| `program` | loop node、worker axes、traversal、mapping |
| `storage` | external / workspace / shared / private fragment / private scalar residency |
| `transfer` | access、domain nodes、boundary fill、defer、result space |
| `reduction` | semantic role、axis、keep-dims、input/result space |
| `pointwise` | semantic role、result space、reuse operand、materialization |
| `contract` | matrix primitive、accumulator type、transpose、operand/result space |
| `stream` | stream node、axis node、tile、order、carry space |
| `ragged` | relation、outer/member node、traversal |
| `stage` | ordinal、root contract、input/output value IDs |
| `atomic` | scatter-reduce node、combine / ordering / scope |

`MachinePlanIndex` 只是对该 MLIR region 中这些 operation 的临时分类索引，不是另一套 MachinePlan 类或第二种 IR。

### 5.2 Resolved realization 与 unresolved search space

决定源码结构的内容直接写入 realization：

- program mapping；
- ownership；
- traversal；
- legal axes；
- boundary semantics；
- value residency；
- matrix primitive role；
- stream/ragged/stage 结构。

需要 target tuner 选择的内容写入 `intent_plan.search_space`：

```text
key        = [shape symbols ...]
parameters = [canonical role names ...]
```

realizer 不选择 winner，也没有 cost model。三个 runtime tuning 模块把 canonical roles 映射为各 target 的参数与 candidate configs：

```text
python/intent/runtime/tuning/triton.py
python/intent/runtime/tuning/cutile.py
python/intent/runtime/tuning/tilelang.py
```

未知 role set 直接 `NotImplementedError`，没有默认候选兜底。

## 6. 一份 GPU Plan 如何投影为三个 surface

三个 projector 都读取相同的 `MachinePlanIndex`，然后在 realization body 中追加 target dialect operation：

```text
lib/Target/Triton/Projection/Project.cpp
lib/Target/CuTile/Projection/Project.cpp
lib/Target/TileLang/Projection/Project.cpp
```

### 6.1 Capability partition

三个 target 都显式声明物理概念属于三类中的哪一类：Intent 已决定、委托给下层、该 surface 模型中不存在。

| Physical concept | Triton | cuTile | TileLang |
|---|---|---|---|
| tile shape | Intent decided | Intent decided | Intent decided |
| ownership | Intent decided | Intent decided | Intent decided |
| traversal | Intent decided | Intent decided | Intent decided |
| boundary | Intent decided | Intent decided | Intent decided |
| value residency | Intent decided | Intent decided | Intent decided |
| matrix primitive role | Intent decided | Intent decided | Intent decided |
| layout | delegated | delegated | delegated |
| pipeline details | delegated | delegated | delegated |
| launch resources | delegated | delegated | delegated |
| register allocation | delegated | delegated | delegated |
| instruction selection | delegated | delegated | delegated |
| candidate configuration/ranking/winner | delegated to target runtime/backend | delegated to target runtime/backend | delegated to target runtime/backend |
| explicit on-chip buffer allocation | absent | absent | Intent decided / printed |

`verifyCapabilityPartition(...)` 要求每个 concept 恰好归入一类，并且必须与 target profile 完全匹配。

### 6.2 同一个 mapping 的不同 surface spelling

| Shared concept | Triton | cuTile | TileLang |
|---|---|---|---|
| `row_strided` | `grid_stride` | `persistent_rows` | `persistent_rows` |
| row tile | `BLOCK_SIZE` | `TILE_SIZE` | `TILE_SIZE` |
| M/N/K tile | `BLOCK_SIZE_M/N/K` | `TILE_SIZE_M/N/K` | `TILE_SIZE_M/N/K` |
| query/stream tile | `BLOCK_SIZE_Q/K` | `TILE_SIZE_M/N` | `TILE_SIZE_M/N` |
| reduction | `tl.max/sum` | `ct.max/sum` | `T.reduce_max/sum` |
| contraction | `tl.dot` | `ct.mma` | `T.gemm` |
| atomic add | `tl.atomic_add` | `ct.atomic_add` | `T.atomic_add` |
| cast | `tl.cast` | `ct.astype` / `ct.full_cast` | `T.cast` / `T.copy_cast` |

这里发生的是概念到语法/物化的映射，不重新选择 tile、ownership 或 traversal。

### 6.3 为什么 cuTile scalar 是 1 元 Tile

Kernel IR 中 scalar reduction/state 仍是 scalar。cuTile reduction 使用 keep-dims tile representation，因此 scalar initial state 在 cuTile projection 中映射为：

```python
ct.full((1,), value, dtype=...)
```

这不是把 cuTile 的 Tile type 上移到 Kernel IR，而是 `private_scalar` 在该 surface 中的物化方法。后续 pointwise 与 stream carry 都继续使用该 1 元 Tile。

### 6.4 为什么 TileLang scalar stream state 使用 local buffer

TileLang 需要显式可变存储。scalar state 不能只绑定到一个不可变 Python/TIR expression，因此 emitter 生成：

```python
stream_state_0 = T.alloc_local((1,), T.float32)
stream_state_0[0] = initial

for stream_tile in T.Pipelined(...):
    ...
    stream_state_0[0] = next_state
```

tensor state 则保持 fragment。这个差异位于 TileLang emission handler，不改变共享 Plan 中的 logical scalar type。

### 6.5 TileLang 的显式 storage

TileLang projection 将共享 residency 显式映射为：

```text
external/workspace  → global
shared              → shared
private_fragment    → fragment
private_scalar      → local
none                → none
```

Triton 与 cuTile 让下层推断更多分配细节，因此不会为了字段对齐而伪造同样的 target storage schema。

## 7. Emission：共享遍历，不共享字符串模板

### 7.1 Common lifecycle

`lib/Target/Common/Emission/Lifecycle.cpp` 固定 emission 生命周期：

```text
prepare
  → emit imports
  → emit kernel header
  → register per-op handlers
  → traverse canonical Kernel IR
  → emit runtime wrapper
```

`OperationHandlerRegistry` 是 operation name 到 `{enter, leave}` callback 的映射。遍历器对每个结构化 region 执行：

```text
enter(op)
  traverse nested regions
leave(op)
```

realization analysis、Plan build 和 target emission 分别使用自己的 handler registry，但共享同一 traversal 机制。

### 7.2 Target emitter 的合法职责

三个 source emitter 分别位于：

```text
lib/Target/Triton/Emission/
lib/Target/CuTile/Emission/
lib/Target/TileLang/Emission/
```

它们负责：

- 验证 target dialect binding；
- 将 ABI 和 Plan node 解析回 Kernel IR value/domain；
- 按 target spelling 发射每个 operation；
- 明确 materialize target-specific scalar/tile/buffer；
- 生成 target runtime wrapper；
- 缺少 binding 或不支持时在当前 op 上失败。

它们不负责：

- 根据 kernel 名称选算法；
- 重新选择 tile size role；
- 决定 ownership 或 traversal；
- 为同一 GPU 再做一次 schedule analysis；
- 在失败时切换到 Python/Torch fallback。

并不存在一个强行统一三种语法的“通用字符串 emitter”。共享的是 lifecycle、Plan、Kernel IR traversal 和 handler contract；每种语言只保留自己的叶子 spelling 与必须显式表达的物化。

## 8. 从 4 Kernel 扩到 10 Kernel 时补上的共享抽象

### 8.1 多 reduction 与 scalar row result

LayerNorm、RMSNorm、logsumexp 要求 row schedule 不再假定 softmax 固定 op 数量：

- LayerNorm：两个 additive reductions；
- RMSNorm：一个 square-sum reduction；
- logsumexp：max + sum，最终输出 scalar row value；
- stable softmax：max + sum，最终输出 full row tensor。

Plan transfer 和 emitter 因此允许 fixed-row output 为 rank 1 或 rank 2；TileLang scalar store 直接发射 scalar assignment，而不是强制 `T.copy` tensor。

### 8.2 一个 schedule 中多个 contraction op

dual GEMM 在同一 M/N/K ownership 下产生两个独立 `intent.contract`：

```text
gate  = contract(x, gate_weight)
value = contract(x, value_weight)
y     = relu(gate) * value
```

两个 contract 各自产生一个 `plan.contract`，共用相同 axis roles 与 `grouped_2d_tiles` mapping。没有 `dual_gemm` realizer，也没有 dual-GEMM emitter。

### 8.3 同一 stream axis 上的两个连续 state stream

online softmax 使用两个 state stream：

1. 第一遍扫描列 tile，更新 `(maximum, denominator)`；
2. 第二遍再次扫描列 tile，使用最终统计量写 normalized output。

`ScheduleDecision.stateStreams`、MachinePlan stream index 和三个 projector 都保存/遍历全部 stream node。它们共用一个 `stream_0` axis decision，但每个 state-stream operation 有独立 `plan.stream` binding。

### 8.4 Ragged pipeline 不再等于 MoE

MoE 有两个 contraction stages：

```text
gather input
  → W1 contract
  → ReLU
  → W2 contract
  → weighted scatter-add
```

grouped GEMM 只有一个 stage：

```text
gather rows
  → group weight contract
  → scatter-add
```

两者都由 `analyzeContractionPipeline(...)` 形成 stage list，再由 `ragged_stages` mechanism 发射。新增 grouped GEMM 证明 stage 数量不是 MoE matcher 的结果。

### 8.5 TileLang ragged 参数不再依赖 MoE 名称

grouped GEMM 首次在 TileLang 运行时暴露了旧 emitter 中的硬编码：outer dimension 被写成 `E`，member capacity 被写成 `R`，offset/index 参数被写成 `route_offsets/member_routes`。

修复后 `prepareRaggedStages()` 从 canonical `intent.ragged` 的两个 view-load operand 解析：

- offsets ABI view；
- indices ABI view；
- member dimension；
- outer program dimension；
- stage feature/reduction dimension。

如果 relation 不是 canonical rank-1 offsets/indices view，直接报错。修复后同一 emitter 同时覆盖 MoE 和任意命名的 grouped GEMM。

## 9. 当前 10 个 Kernel 与它们验证的机制

### 9.1 Kernel 目录

```text
examples/kernels/
├── contraction/
│   ├── gemm.py
│   └── dual_gemm.py
├── normalization/
│   ├── softmax.py
│   ├── layer_norm.py
│   ├── rms_norm.py
│   └── logsumexp.py
├── streaming/
│   ├── attention.py
│   └── online_softmax.py
└── ragged/
    ├── moe.py
    └── grouped_gemm.py
```

### 9.2 结构矩阵

| Kernel | 模型级 shape | Kernel IR 关键结构 | Shared mapping | 主要新增证明 |
|---|---|---|---|---|
| stable softmax | `8192×8192`, f32 | max/sum + full-row store | `row_strided` | 基础 row reduction |
| weighted LayerNorm | `8192×4096`, f32 | two sums + rsqrt + weight/bias | `row_strided` | 多 reduction、weighted normalization |
| weighted RMSNorm | `8192×4096`, f32 | square-sum + rsqrt + weight | `row_strided` | 不带 centering 的 normalization |
| row logsumexp | `8192×8192 → 8192`, f32 | max/sum/log + scalar store | `row_strided` | scalar row result |
| GEMM | `4096×4096 × 4096×14336`, f16 | one contraction | `grouped_2d_tiles` | 基础 tiled contraction |
| gated dual GEMM | `2048×4096 × 2×(4096×4096)`, f16 | two contractions + gated epilogue | `grouped_2d_tiles` | 同 schedule 多 contract |
| attention | `4×32×4096×128`, f16 | Q tile + K state stream + two contracts | `multi_axis_stream` | contraction streaming |
| online softmax | `8192×8192`, f32 | two state streams, scalar carry | `row_stream` | 无 contraction 的 recurrence |
| MoE | `T=4096,D=4096,F=14336,E=8,top-k=2` | ragged + gather + two stages + atomic | `ragged_stages` | 多 stage irregular pipeline |
| grouped GEMM | `R=8192,K=N=4096,G=8` | ragged + gather + one stage + atomic | `ragged_stages` | 单 stage irregular pipeline |

### 9.3 各 Kernel 的 DSL 算法结构

#### Stable softmax

```python
maximum = I.reduce.max(values, axis=0, identity=-I.inf)
numerator = I.exp(values - maximum)
denominator = I.reduce.sum(numerator, axis=0, identity=0.0)
y[row, columns] = numerator / denominator
```

#### Weighted LayerNorm

```python
mean = I.reduce.sum(values, axis=0, identity=0.0) * inverse_features
second_moment = I.reduce.sum(values * values, axis=0, identity=0.0)
variance = second_moment * inverse_features - mean * mean
normalized = (values - mean) * I.rsqrt(variance + epsilon)
y[row, columns] = normalized * weight[columns] + bias[columns]
```

#### Weighted RMSNorm

```python
mean_square = I.reduce.sum(values * values, axis=0, identity=0.0)
mean_square = mean_square * inverse_features
y[row, columns] = values * I.rsqrt(mean_square + epsilon) * weight[columns]
```

#### Row logsumexp

```python
maximum = I.reduce.max(values, axis=0, identity=-I.inf)
denominator = I.reduce.sum(I.exp(values - maximum), axis=0, identity=0.0)
y[row] = maximum + I.log(denominator)
```

#### Gated dual GEMM

```python
gate = I.contract(x[mr, k], gate_weight[k, nr], reduce=((1, 0),))
value = I.contract(x[mr, k], value_weight[k, nr], reduce=((1, 0),))
y[mr, nr] = I.cast(I.maximum(gate, 0.0) * value, I.f16)
```

#### Streamed online softmax

```python
statistics = I.state_stream(columns, extent=I.auto("N_TILE"), init=(-inf, 0))
for region, (maximum, denominator) in statistics:
    local_maximum = reduce_max(x[row, region])
    next_maximum = maximum(maximum, local_maximum)
    next_denominator = exp(maximum - next_maximum) * denominator
    next_denominator += reduce_sum(exp(x[row, region] - next_maximum))
    statistics.yield_(next_maximum, next_denominator)

output = I.state_stream(columns, extent=I.auto("N_TILE"), init=statistics.result)
for region, (maximum, denominator) in output:
    y[row, region] = exp(x[row, region] - maximum) / denominator
    output.yield_(maximum, denominator)
```

#### Ragged grouped GEMM

```python
groups = I.ragged(outer=domain(0, G), offsets=group_offsets, indices=member_rows)
for group in I.parallel(groups.outer):
    for member_region in I.parallel(I.partition(groups[group], extent=I.auto(...))):
        rows = I.members(member_region)
        values = I.gather(x, index=(rows, slice(None)))
        result = I.contract(values, weight[group, :, :], reduce=((1, 0),))
        I.scatter_reduce(y, index=(rows, slice(None)), value=result, combine=I.add)
```

## 10. 目标代码实际长什么样

下面不是另一套算法模板，而是相同 Plan 经过不同 target spelling 后的结构摘要。

### 10.1 Row normalization

Triton：

```python
program_start = tl.program_id(0)
program_step = tl.num_programs(0)
for row in tl.range(program_start, n_rows, program_step, num_stages=num_stages):
    columns = tl.arange(0, BLOCK_SIZE)
    values = tl.load(..., mask=columns < n_cols, other=0.0)
    mean = tl.sum(values, axis=0) * inverse_features
    ...
    tl.store(..., result, mask=columns < n_cols)
```

cuTile 将同一个 row/vector Plan 投影为 `persistent_rows`、`ct.load`、`ct.sum` 和 `ct.store`；TileLang 则显式分配 fragment 并通过 `T.reduce_sum` 写 local/fragment result。

### 10.2 Dual contraction

一个 generated kernel 中出现两个独立 target contraction：

```text
gate  = target_matrix_primitive(x_tile, gate_weight_tile)
value = target_matrix_primitive(x_tile, value_weight_tile)
result = max(gate, 0) * value
```

Triton spelling 是两个 `tl.dot`，cuTile 是两个 `ct.mma`，TileLang 是两个 `T.gemm`。M/N ownership 和 K reduction role 只在 shared Plan 中决定一次。

### 10.3 Row stream

Triton / cuTile 生成两个 forward loop，carry 保存在 register value；TileLang 生成两个 `T.Pipelined` loop，scalar carry 放在 local buffer。

```text
program(row)
  stream pass 1:
    load tile
    update running maximum / denominator
  stream pass 2:
    load tile
    normalize and store
```

### 10.4 Ragged stages

MoE 的 physical plan 生成两个 staged target kernels；grouped GEMM 只生成一个 stage。每个 stage 都按 outer relation 与 member tile 组织 program grid，最后由 atomic add 完成 scatter merge。

这种多 target-kernel realization 是 DSL kernel 的物理 staging，不是 emitter 根据 `moe` 名称拆 kernel。

## 11. Repro 与上游 baseline 如何组织

### 11.1 单一公开入口

```bash
./examples/run/repro.sh \
  <triton|cutile|tilelang> \
  <softmax|layer_norm|rms_norm|logsumexp|gemm|dual_gemm|attention|online_softmax|moe|grouped_gemm>
```

脚本：

1. 选择仓库外 Python environment；
2. 在 `/tmp/intentdsl-build` 构建唯一 `intent-compile`；
3. 选择对应 source baseline；
4. 进入 `/tmp` 运行 target main；
5. 调用 DSL frontend、C++ compiler、生成 source、backend compile/JIT；
6. 对 generated / upstream / PyTorch reference；
7. 输出 generated MLIR、target source、数值与 benchmark。

### 11.2 运行环境

本轮实测环境：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D |
| compute capability | 12.0 |
| GPU memory | 32607 MiB |
| driver | 580.95.05 |
| Triton | 3.6.0 |
| cuda-tile | 1.5.0 |
| TileGym | 1.4.0 |
| TileLang | 0.1.13 |
| Triton / TileLang torch | 2.10.0+cu130 |
| cuTile torch | 2.13.0+cu130 |

所有环境均位于 `/home/kingdom/.venvs/`，项目目录中没有 virtualenv、build cache 或 runtime log。

### 11.3 Baseline 矩阵

| Kernel | Triton source | cuTile source | TileLang source |
|---|---|---|---|
| softmax | Triton fused softmax | TileGym softmax | TileLang online softmax |
| LayerNorm | FlashAttention Triton LayerNorm | NVIDIA cuTile LayerNorm | unavailable |
| RMSNorm | Liger Triton RMSNorm | unavailable | TileLang RMSNorm + weight composition |
| logsumexp | unavailable | unavailable | unavailable |
| GEMM | Triton matmul | TileGym matmul | TileLang matmul |
| dual GEMM | two upstream matmul calls + gated epilogue | same composition | same composition |
| attention | Triton fused attention | NVIDIA cuTile FMHA | TileLang MHA |
| online softmax | Triton fused softmax | TileGym softmax | TileLang online softmax |
| MoE | grouped GEMM composition | cuTile fused MoE | grouped GEMM composition |
| grouped GEMM | grouped GEMM + external gather/scatter composition | TileGym grouped GEMM composition | TileLang grouped GEMM composition |

Unavailable 的 5 条组合是：

```text
Triton  + logsumexp
cuTile  + RMSNorm
cuTile  + logsumexp
TileLang + LayerNorm
TileLang + logsumexp
```

这里的 unavailable 表示当前 source corpus 中没有等价 standalone 上游 baseline，不表示 Intent target 无法表达。对应 generated kernel 均已真实运行。

## 12. 30 条数值结果

下表为本轮实际运行的 generated vs PyTorch reference 最大绝对误差：

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | `2.7939677e-09` | `1.8626451e-09` | `2.7939677e-09` |
| LayerNorm | `1.9073486e-06` | `1.9073486e-06` | `2.8610229e-06` |
| RMSNorm | `2.8610229e-06` | `1.9073486e-06` | `1.9073486e-06` |
| logsumexp | `9.5367432e-07` | `9.5367432e-07` | `9.5367432e-07` |
| GEMM | `0.0` | `0.0` | `0.0` |
| dual GEMM | `7.8125e-03` | `7.8125e-03` | `7.8125e-03` |
| attention | `3.0517578e-05` | `3.0517578e-05` | `3.0517578e-05` |
| online softmax | `3.7252903e-09` | `1.8626451e-09` | `1.8626451e-09` |
| MoE | `1.9021332e-05` | `9.1064721e-06` | `1.9039959e-05` |
| grouped GEMM | `8.5830688e-05` | `8.5830688e-05` | `8.5830688e-05` |

30/30 数值通过。

dual GEMM 最终输出为 f16；grouped GEMM/MoE 包含 f16 contraction、f32 accumulation 与原子合并，因此误差尺度与 f32 row kernels 不同。

## 13. 性能结果与诚实边界

下表给出本轮一次实测中的 generated p50 / p95，以及存在上游 baseline 时的 generated/upstream p50。`—` 表示无等价上游 baseline。

### 13.1 Triton

| Kernel | generated p50 / p95 (ms) | generated / upstream p50 |
|---|---:|---:|
| softmax | `0.3550 / 0.3575` | `0.9980×` |
| LayerNorm | `0.1758 / 0.1778` | `1.0195×` |
| RMSNorm | `0.1764 / 0.1782` | `1.0118×` |
| logsumexp | `0.1635 / 0.1637` | — |
| GEMM | `2.1490 / 2.1523` | `1.0345×` |
| dual GEMM | `0.6980 / 0.6996` | `0.9021×` |
| attention | `4.9454 / 4.9496` | `0.9966×` |
| online softmax | `0.3796 / 0.3839` | `1.0640×` |
| MoE | `13.0976 / 13.1259` | `1.2806×` |
| grouped GEMM | `2.4658 / 2.4744` | `1.1876×` |

### 13.2 cuTile

| Kernel | generated p50 / p95 (ms) | generated / upstream p50 |
|---|---:|---:|
| softmax | `0.3579 / 0.3601` | `0.9996×` |
| LayerNorm | `0.1805 / 0.1838` | `0.5252×` |
| RMSNorm | `0.1799 / 0.1801` | — |
| logsumexp | `0.1696 / 0.1698` | — |
| GEMM | `2.2993 / 2.3037` | `1.0047×` |
| dual GEMM | `0.7200 / 0.7216` | `0.8925×` |
| attention | `4.9964 / 5.2644` | `0.0724×` |
| online softmax | `0.3499 / 0.3519` | `0.9772×` |
| MoE | `16.1568 / 16.2956` | `1.6842×` |
| grouped GEMM | `2.8907 / 2.9063` | `1.4288×` |

### 13.3 TileLang 0.1.13

| Kernel | generated p50 / p95 (ms) | generated / upstream p50 |
|---|---:|---:|
| softmax | `0.3493 / 0.3519` | `0.9599×` |
| LayerNorm | `0.1747 / 0.1758` | — |
| RMSNorm | `0.1739 / 0.1758` | `0.4973×` |
| logsumexp | `0.1614 / 0.1616` | — |
| GEMM | `2.1308 / 2.1365` | `0.9029×` |
| dual GEMM | `0.7339 / 0.7349` | `0.7960×` |
| attention | `4.8917 / 4.9047` | `0.7306×` |
| online softmax | `0.3811 / 0.3848` | `1.0523×` |
| MoE | `21.6398 / 21.7077` | `2.3092×` |
| grouped GEMM | `2.5876 / 2.6355` | `1.1251×` |

这些数字的解释边界：

- row reduction、normalization、dense GEMM 和 attention 已处于上游同一量级，部分 surface 更快、部分略慢；
- dual GEMM baseline 是两次上游 GEMM 加 gated epilogue，比较的是组合后端路径；
- grouped GEMM baseline 包含外部 gather/scatter composition，不是与 generated atomic staging 完全相同的单 kernel；
- MoE 与 ragged grouped GEMM 尚未达到所有上游路径的性能水平，尤其 TileLang MoE 和 cuTile MoE；
- cuTile attention baseline 的 source wrapper 开销很大，因此 `0.0724×` 不能解释为纯 kernel 级 13.8 倍加速；
- online softmax 是两遍 state-stream 算法，而 Triton/cuTile 的部分 baseline 使用不同的整行/chunked 算法；
- benchmark 是本轮实测观察，不当作跨运行稳定结论。

因此当前准确表述是：所有已有 repro 仍可运行；数值没有退化；dense/row/attention 路径性能健康；ragged/MoE 的性能 parity 仍是未完成工作，不能因为架构打通就宣称完成优化。

## 14. 为什么这次可以证明“编译器真的在发射”

### 14.1 没有 kernel 名称进入 compiler

对 `include/`、`lib/`、`python/intent/` 检索以下名称：

```text
layer_norm
rms_norm
logsumexp
dual_gemm
grouped_gemm
online_softmax
```

在 realization、projection、emission 中均无匹配。

### 14.2 新增 kernel 没有新增 realization/emission 文件

新增六个 kernel 时，compiler 侧真正发生的是：

- 通用 padded-value proof；
- scalar row result emission；
- row-stream mapping；
- multiple state-stream indexing；
- surface-specific scalar materialization；
- TileLang ragged ABI/dimension derivation。

这些都修改已有通用模块，没有产生：

```text
Realization/LayerNorm.cpp
Realization/DualGemm.cpp
Emission/OnlineSoftmax.cpp
Emitter/GroupedGemm.cpp
```

### 14.3 第五/第六种同类结构只组合已有 mechanism

- softmax、LayerNorm、RMSNorm、logsumexp 共用 row/vector mapping；
- GEMM、dual GEMM 共用 tiled contraction mapping；
- attention、online softmax 共用 state-stream concept，但分别投影为 multi-axis stream 和 row stream；
- MoE、grouped GEMM 共用 ragged stage mechanism，但 stage 数量不同。

这正是“kernel 类别是 mechanism 组合结果，而不是 compiler 入口分支”的实际证据。

## 15. 当前硬边界

### 15.1 已经成立

| 能力 | 状态 |
|---|---|
| Canonical Kernel MLIR | 10 个 kernel 共用，仍是唯一算法 IR |
| Common GPU schedule | 三后端共用一份 decision |
| Resolved vs search space | 已在 Plan IR 中分离 |
| Triton / cuTile / TileLang | 10 个 kernel 全部真实运行 |
| Generic traversal | analysis / Plan build / emission 共用 registry/traversal contract |
| Per-op failure | 未知 role/op/mapping 直接失败，无 fallback |
| Model-level input | 所有 repro 使用 4K/8K 级真实 shape |

### 15.2 尚未完成或明确受限

| 边界 | 当前准确状态 |
|---|---|
| 多 ragged relation | Plan/index 可保存多个；当前 SchedulePolicy 与 `ragged_stages` target mapping 都明确只接受一个 canonical relation |
| multi-axis 多 stream | shared model 可索引多个 `StreamOp`；当前 `multi_axis_stream` target emitter 要求恰好一个 `StreamOp`/ordered stream axis；`row_stream` 已验证同轴两个连续 state stream |
| ragged performance | 数值正确，但 MoE/grouped parity 尚未完成 |
| baseline corpus | 5 个 backend/kernel 组合没有等价 standalone source |
| CPU / RVV | 当前 GPU physical decision 与三 tile surface 已形成；CPU/RVV target 尚未接入本轮 repro |
| delegated compiler internals | layout、pipeline、register allocation、instruction selection 仍由下层工具负责，本项目不复制实现 |
| tuner | 下层 tuner 选择 winner；项目只提供合法 role 映射和 target configs，没有 cost model |

## 16. 本阶段提交脉络

| Commit | 作用 |
|---|---|
| `599d6b2` | 统一 compiler 与 repro 入口，形成四 kernel 基线骨架 |
| `79f9c45` | 将 schedule composition 上移到 Common realization |
| `c8d1678` | 建立共享 GPU machine decisions、Plan build 与 target projection |
| `033378e` | 补齐 TileLang MoE，完成四 kernel × 三后端基线 |
| `9726f81` | 将 padded row reduction 从 softmax 特化改为通用证明 |
| `ee4aed9` | 支持 scalar row result |
| `d97a529` | 组合式 streamed GPU schedule、row stream 与多 stream decision |
| `80f5131` | TileLang ragged binding 改为从 Plan/ABI 推导 |
| `6dfa9d6` | 扩展到 10 kernel，接入 30 条统一 repro 与 baseline matrix |

## 17. 当前最小复现方式

任意单条 repro：

```bash
./examples/run/repro.sh triton layer_norm
./examples/run/repro.sh cutile online_softmax
./examples/run/repro.sh tilelang grouped_gemm
```

每条命令都会从 DSL 重新 lower，调用同一个 `intent-compile`，生成并打印目标代码，然后真实 launch CUDA kernel 做数值与性能对照。项目没有 test 目录、pytest fixture 或另一条隐藏验证链。

## 18. 最终状态

这一阶段得到的核心结果不是“又支持了六个函数名”，而是把四个起始结构中暴露出的物理机制上移并组合：

```text
row ownership
tiled ownership
ordered stream
ragged ownership
boundary proof
scalar/tensor materialization
contraction stage
atomic merge
target-neutral tuner roles
```

10 个 kernel 的差异由这些机制的组合产生；三个 surface 只投影同一份 GPU decision，并补上各自真实需要的语法与物化。当前 30 条路径已经证明这套分层可以覆盖明显不同的 reduction、normalization、contraction、streaming 与 ragged 算法结构；同时，ragged/MoE 性能和缺失的 5 条上游 baseline 仍被明确保留为未完成边界，没有用“架构已打通”掩盖。
