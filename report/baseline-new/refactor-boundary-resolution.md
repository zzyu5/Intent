# baseline-new 前的重构边界关闭与接线审计

## 1. 本轮结论

本轮没有继续用 source adapter 的失败推动语言或共享编译器扩张，而是先按
`compiler-refactor-gate-before-baseline-new.md` 重新检查当前 V2 pass 结构。

结论分成四类：

1. V2 重构已经自然关闭的问题：旧的 Kernel IR 与旁表 Plan 双执行权威、Python/terminal
   emitter 的旧路径、state-stream contraction 的第二套本地 reduction loop、region argument
   与 ABI owner 的错误来源。
2. 本轮真正修改的问题：producer-chain replay 缺少显式 selected decision；`partition` 的
   public surface 比当前可 lowering Core 更宽；baseline runner 会把 adapter 错误误记成
   compiler failure。
3. 审计后保留的问题：TileLang 的 bulk-copy/逐元素 transfer、`T.copy_cast` 和 target-local
   candidate legality 是目标投影，不是共享物理决定；`floor`、`sin`、`cos` 是真实算法所需的
   canonical unary 能力。
4. 尚未完成但没有被伪装成能力边界的问题：source inventory 是 41/39/39，registry 仍是
   30/30/30；29 项只是尚未完成算法对齐或 adapter，并不等于“不需要处理”或
   “target unsupported”。

本轮没有新增 registry entry，也没有改动六张结果表。原因是编译器与接线的责任边界必须先
关闭；在这个边界关闭之前继续堆 entry，会重新制造“adapter 接不上就改语言/Plan”的混杂。

## 2. V2 后已经自然消失的旧问题

### 2.1 可执行权威已经唯一

当前主链是：

```text
canonical Kernel IR
  -> ConstructPhysicalProgramPass
  -> intent_plan.program 中唯一 physical function
  -> VerifyPhysicalProgramPass
  -> MaterializeTargetProgramPass
  -> intent_plan.target_program
  -> terminal translator
```

physical construction 会把 canonical function 原地移入 `intent_plan.program`；module 顶层不再
保留另一份仍可被执行的 Kernel IR。旧的 `RealizationOp`、`SurfacePlan`、Python emitter、
`emit*Source` 和旧 Emission 路径均已不存在。因此 gate 报告中最核心的“双份 executable
authority”没有换名字残留。

### 2.2 已有 correctness 修复的职责已收敛

以下历史修改在 V2 后都能沿唯一链解释，不需要为了本轮再造一份实现：

| 历史问题 | 当前权威来源 | target 的职责 |
|---|---|---|
| state-stream 内 contraction reduction range | `StreamAxisOp` 与 selected reduction range | 复用 active stream range，不再开第二个 reduction loop |
| scalar gather | canonical index relation 与 common role query | 拼写 extract/gather |
| routed staged RHS transpose | canonical contraction orientation | 拼写 target transpose/MMA 参数 |
| 多维 staged unique store | stage-axis、member、feature 与 validity bindings | 拼写 route/member/feature 地址与 store |
| constexpr shape | frontend signature/ABI metadata | 读取已绑定 shape，不从 output 反猜 |
| TileLang traversal region argument | `RegionBindingOp` 的 stable value ID | 使用 selected traversal range |
| cuTile dynamic shape owner | ABI input/inout owner | Out 不再成为自己的未分配 shape source |

这些路径没有按 kernel 名称、source 名称或 registry entry 分支。

## 3. producer-chain contraction replay：从隐式猜测改为显式选择

### 3.1 原来的边界问题

GPU physical construction 会依据 typed KIR facts 选择是否把 contraction operand producer chain
延迟到 contraction 内 replay。这个选择会改变生成源码结构，不能从 Kernel IR 唯一推出，属于
Physical Plan 应保存的结构性 selected decision。

原实现没有保存“选择了 replay”这个事实，只间接留下：

- operand residency 为 `shared`；
- 若干 transfer 的 materialization 为 `deferred_to_contract`。

common materialization 随后再次扫描 KIR，并用“是否存在 deferred transfer + shared operands”
反推 contraction 是否选择 replay。问题不是 producer list 被重算——producer list 本来就是可从
KIR def-use 唯一派生的索引——而是“是否选择 replay”也被当作派生事实猜了一次。

### 3.2 本轮修法

`intent_plan.contract` 新增 `producer_replay` boolean：

- GPU construction 是唯一写入者；
- `true` 时 verifier 要求 lhs/rhs residency 都是 `shared`；
- common lowering 只在该 selected bit 为真时，从 KIR 派生 producer/transfers/exclusive producer
  列表；
- selected replay 若无法解析出 canonical chain，或不是所有 source transfer 都被 deferred，
  直接在 Plan binding 上报错；
- 三个 target handler 继续共享同一份 derived index，只机械 dispatch producer op。

没有把 producer node list 写进 Plan。那份列表可由 KIR 唯一重算，写进去会制造第二份算法
真理。新增字段只保存不能从 KIR 推出的“多个合法实现中选了 replay”这一项。

这项改动没有增加 kernel matcher。replay eligibility 仍由受限 producer whitelist、一个共享
reduction domain、非 staged/non-stream 条件和 transfer 单 owner 约束组成；不读取 kernel 名、
Python 函数名或完整 op-count 模板。

## 4. `partition`：public Core 收敛到真实可 lowering 的两种 extent

### 4.1 真实使用面

当前 `examples/` 中有 117 个 `I.partition(...)` 调用：

- 109 个使用 `extent=I.auto(...)`；
- 8 个使用正的 compile-time fixed extent；
- 0 个使用 `count=`；
- 0 个使用 runtime scalar extent。

原前端接受 runtime scalar extent，但 KernelFacts 深层拒绝；`count=` 虽有 enum/type/iteration
分支，却从 frontend 第一天起就拒绝。它们是“表面看起来存在、实际不能 lowering”的假能力。

### 4.2 本轮收敛

当前 public lowering 合同变为：

```text
I.partition(axis, extent=<positive constexpr integer | I.auto(name)>)
```

具体变化：

- frontend 不再接受 `count` keyword；
- 删除不可达的 `PartitionMode.COUNT` 与 part-ordinal iteration argument 分支；
- runtime scalar extent 在 frontend、带源码位置地拒绝，不再生成 KIR 后到 KernelFacts 深层失败；
- fixed extent 同时要求正整数；
- `auto` 仍只授权 Realizer 选择 physical extent，不授权编译器插入 partition 或改变 body 从
  element 到 region 的算法语义。

没有把 `count=P` 静默改写成 extent。count 会暴露 part identity，并可能影响 partial ABI 与
多调用编排；它不是 extent 的语法糖。将来只有真实算法满足正式语言扩展门槛时，才可作为
独立能力重新讨论。

## 5. 语言能力复核：保留 `floor`、`sin`、`cos`

### 5.1 `sin` / `cos`

RoPE 与 Mamba3 rotation 都直接需要三角函数。现有 Core 不能在不改变作者算法和精度合同的
情况下用 exp/log 或多项式无损展开。三 target 都有原生 spelling：`tl.sin/cos`、
`ct.sin/cos`、`T.sin/cos`。

它们使用现有 canonical unary op，不新增 Plan fact。Plan 只负责已有 pointwise residency 和
axis binding，leaf 只查 unary role 并拼写目标调用。

### 5.2 `floor`

Mamba3 SISO step 的 upstream 明确执行：

```text
angle -= 2*pi*floor(angle/(2*pi))
```

当前 DSL 与 upstream 数值路径一致。旧的一次条件加/减只能处理一个周期，不能表达同一算法的
一般输入。三个 target 同样都有原生 floor。故 `floor` 是独立成立的 unary Core 能力，不是
producer replay 的附属能力，也不需要 Plan fact。

### 5.3 Mamba/selective-scan 改写的定性

逐项对照 vendored upstream 后，本轮确认：

- chunk-state 的 `group`/`last_position` 是作者显式给出的 bounds facts；
- chunk-scan 复用已有 rows domain 是等价 domain identity 收敛；
- `minimum(decay_delta, 0)` 与 upstream Triton 公式一致；
- Mamba3 floor wrap 与 upstream Triton 公式一致。

因此这些例子没有为了让编译器通过而换算法。历史上把 replay、fixed partition、语言 unary 和
算法文件混在同一提交是不合格的提交节奏，但当前每一项能力本身均有独立依据，不应整体回退。

## 6. target leaf 越权审计

### 6.1 没有发现的路径

对当前 `lib/Target/**` 与 Python frontend/compiler 的扫描没有发现：

- 按 kernel/source/example 名称分支；
- 根据完整 kernel op-count 选择模板；
- baseline registry 进入 compiler；
- Python adapter 直接调用 Plan 或 emitter 内部接口；
- terminal translator 绕过 `intent_plan.target_program` 再遍历 KIR 做决定。

### 6.2 TileLang 中保留的 target-local 逻辑

TileLang materialization 中仍有比简单字符串表更厚的代码，但本轮没有把它们误判为共享决策：

1. `bulk_copy` 与 `parallel_elements`：依据 Plan 已给出的 indexing class、validity、consumer
   neutralization 与目标 API 能力，选择同语义的目标搬运拼写。
2. `explicitBounds`：决定目标源码是否显式打印已经确定的 logical validity；不改变 validity。
3. `T.copy_cast`：TileLang 对 contract operand cast 的原生拼写；不改变 dtype 或 contraction。
4. symmetric program-tile candidate：只约束 TileLang tuner 的合法参数组合，没有选择共享
   ownership/tile winner，也没有进入 Kernel IR/共享 Plan。

这些属于 capability、spelling 或 target-local parameter legality。把它们抬进共享 Plan 会让
TileLang 当前 API 反向污染其它 target。若以后其中某项开始重选 ownership、stream end、
storage 或算法结构，才构成越权；当前代码没有跨过这条线。

### 6.3 仍需保持警惕的 derived index

stage operation inputs/outputs、replay producer list、scan producer owners 都是从
`Kernel IR + selected Plan bindings` 重算的 common index。它们不是第四层 IR，也不应被
序列化。三个 materializer 各调用同一个 common query，是每次 provider materialization 的
派生缓存，不是三套判断实现。

## 7. baseline 接线层的责任修正

### 7.1 compiler 与 adapter 已分开

当前 provider adapter 统一通过 `intent.compile(...)` 进入公开编译链，没有导入 C++ Plan、
provider materializer 或内部 compiler pass。source adapter 可以调用 vendored runtime 的
private kernel/executable helper，但这种脆弱性只属于 source runtime ABI，不构成扩语言或
修改 Plan 的依据。

### 7.2 不再把 adapter 错误记成 compile failure

原 runner 在 provider factory 阶段捕获所有 `Exception` 并写成 `compile_failed`。factory
同时包含 generated compile、source import、source JIT、shape/ABI adapter 与 workspace
准备，因此这个状态会把接线错误伪装成编译器缺口。

本轮把 generated compile/run/launcher preparation 包装为明确的
`GeneratedCompilationError`：只有这条错误写 `compile_failed`。数值树比较使用独立的
`NumericalComparisonError`，只写 `numerical_failed`。其它 adapter/source/runtime 异常不写
一行假状态，而是让命令直接失败并保留原始异常。

CSV schema 与 `construction.md` 保持不变，没有新增模糊的 `adapter_failed` 或
`runtime_failed` 状态。

## 8. 41/39/39 inventory 与当前 30/30/30 registry

### 8.1 当前数量

| Provider | README inventory | registry | 尚未进入 registry |
|---|---:|---:|---:|
| Triton | 41 | 30 | 11 |
| cuTile | 39 | 30 | 9 |
| TileLang | 39 | 30 | 9 |
| 合计 | 119 | 90 | 29 |

30 是每张最终表的最低数量，不是 inventory 截止线。本轮没有把缺失项登记为 unsupported，
因为它们尚未逐项完成相同算法、调用边界和计时 scope 的 adapter 验证。

### 8.2 Triton 的 11 项

| source entry | 当前分类 |
|---|---|
| FlashAttention fused cross entropy | 与现有 Liger CE 的 source/算法合同需逐项对齐，尚未 adapter |
| FlashAttention fused LayerNorm family | 与现有 LayerNorm 同算法族的另一高性能实现，先判定重复关系 |
| fused linear cross entropy | 不同 fused 算法，需要独立对齐 DSL/调用边界 |
| split-K paged attention | 不同多 kernel 算法，需要独立对齐 DSL/adapter |
| xFormers RMSNorm | 与现有 RMSNorm 同算法族的另一实现，先判定重复关系 |
| causal Conv1D forward | 与 varlen forward/decode update 不同，已有 Core 可表达但 adapter 未接 |
| causal Conv1D backward | 独立 backward 合同，adapter 未接 |
| modern FlashAttention forward | 与现有 dense causal attention 的 source 替代关系待确认 |
| MoE split-K expert projection | 与当前 routed projection 的分解方式不同，需要独立对齐 DSL |
| MoE column-major expert projection | 与当前 routed projection 的遍历/布局算法不同，需要独立对齐 DSL |
| Mamba3 SISO sequence forward | 与当前单 step 不同的 sequence 算法，需要独立 DSL/adapter |

### 8.3 cuTile 的 9 项

| source entry | 当前分类 |
|---|---|
| official fused MoE | 当前 entry 仅 expert projection；完整 fused MoE 需独立对齐 |
| TileGym dense attention forward | 与 official FMHA 的算法/source 重复关系待确认 |
| grouped flash decode | 独立 decode 算法，adapter 未接 |
| attention-sink decode | 与 sink prefill 不同，adapter 未接 |
| Gemma split-K decode | 独立 split-K pipeline，adapter 未接 |
| chunk gated delta rule | 独立 sequence/chunk 算法，adapter 未接 |
| fused linear cross entropy | 独立 fused loss，需对齐 DSL/adapter |
| NVFP4 quantization | 独立低精度量化合同，尚未 adapter |
| TileGym dense GEMM | 与 registry 中 official dense GEMM 是同算法的 provider 实现变体 |

### 8.4 TileLang 的 9 项

以下均尚无证据可归入“重复”或“target unsupported”，当前统一定性为尚未完成算法对齐/
adapter：

- persistent MLA decode；
- fused routed/shared MoE；
- DeepSeek V3.2 top-k selector；
- fused chunk linear-attention backward；
- attention-sink backward；
- sparse MLA backward；
- BitNet int2 decode；
- BF16 × FP4 dequant GEMM；
- block FP4 activation quantization。

因此，当前新 baseline 的诚实状态仍是“90 个第一批 adapter scaffold + 29 个待逐项处理的真实
source entry”，不是完整 119-entry baseline。

## 9. 定向验证

本轮没有做全量，只验证真正读取修改事实的路径。

### 9.1 producer replay + auto partition

命令：

```bash
examples/run/baseline-v2.sh cutile \
  /tmp/baseline-new-boundary-cutile.csv \
  splitk_attention_reduce
```

结果：数值通过；generated `0.005344 ms`，source `0.006144 ms`，ratio `0.869792`。

该 kernel 的 contraction 两侧包含 reshape/cast producer chain，直接消费
`producer_replay` selected decision；外层同时使用 `extent=I.auto("D_TILE")`。

### 9.2 fixed partition + `floor/sin/cos`

命令：

```bash
examples/run/baseline-v2.sh triton \
  /tmp/baseline-new-boundary-triton.csv \
  mamba3_siso_step
```

结果：数值通过；generated `0.106432 ms`，source `0.095824 ms`，ratio `1.110703`。

该 kernel 使用两个 fixed partition，并直接执行 floor wrap 与 sin/cos rotation，覆盖本轮保留的
语言能力和收敛后的 fixed-extent frontend path。

## 10. 当前停止线

本轮之后可以确认：

- baseline adapter 失败不再获得修改语言/KIR/Plan/leaf 的默认授权；
- replay 的 selected decision 与 derived producer index 已分开；
- `partition` 的可用表面与当前 realization 闭环一致；
- `floor/sin/cos` 与 Mamba 数值表达有独立真实算法依据；
- V2 当前未发现 kernel-name、source-name 或 op-count 模板路径；
- 新 baseline runner 不再把 source adapter 异常伪装成 compiler failure。

仍不能宣称：

- 41/39/39 source 已全部进入 registry；
- 90 个现有 registry entry 已在本轮全部复验；
- 六张 provider × device 表已经完成；
- 29 个遗漏项是 target unsupported；
- private upstream helper 已成为稳定 source runtime ABI。

这条停止线把能力工作与接线工作重新分开：后续若某个 source 不能接入，必须先证明是真实
算法无法用现有 Core 表达；否则它只属于 DSL 算法对齐或 adapter 工作，不能反向修改共享
编译器。
