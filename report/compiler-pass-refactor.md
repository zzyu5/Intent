# Compiler Pass 重构实施报告

## 1. 结论

这一轮已经把编译主链从：

```text
top-level Kernel IR
  + KIR-referenced realization records
  -> provider emitter 再遍历 Kernel IR 并直接写源码
```

切换为：

```text
canonical Kernel IR
  -> ConstructPhysicalProgramPass
  -> executable intent_plan.program
  -> VerifyPhysicalProgramPass
  -> provider MaterializeTargetProgramPass
  -> intent_plan.target_program
  -> terminal translator
  -> provider source
```

现在每个阶段只有一个 executable authority。进入 physical-program 阶段后，原顶层
Kernel function 被移入 `intent_plan.program`，标记为 `intent.kind = "physical"`；module
中不再同时保留一份可执行 canonical function。provider lowering 只能消费 program 内的
当前 physical function 和同一 program 中的 physical decisions。terminal translator 只读
`intent_plan.target_program.source`，不再分析 function、Plan 或 kernel 结构。

旧的 `intent_plan.realization`、`realizeKernel`、`emit*Source`、`TargetSourceEmitter`、
`SourceEmitter`、`SurfacePlan` 以及 `Emission/` 编译路径已经删除，没有 compatibility flag、
legacy fallback 或双路 driver。

## 2. 起点问题与本轮切面

重构前，同一 Kernel IR 被三次消费：

1. `KernelModel`/`KernelFacts` 分析 canonical KIR；
2. GPU builder 再遍历 KIR，构造零 operand/result 的 Plan records；
3. 三个 provider emitter 再遍历 KIR，结合 Plan 现场决定 execution skeleton、access form、
   temporary、workspace、replay 和目标源码。

因此旧 Plan 不是当前程序，只是旁边的 selected-decision skeleton。最根本的问题不是再加
Plan 字段，而是 leaf 仍然以旧 KIR 为 executable program。

本轮选择的迁移切面是“先改变 authority，再细分 passes”：沿用已有 Plan dialect 的物理词汇，
但让它真正拥有 physical function、launch/axis/range/value/access/structured decisions 和
provider materialization 结果。这样现有已经验证过的 policy 可以保留，同时切断旧 KIR +
旁表的执行合同。

## 3. Physical program IR

### 3.1 `intent_plan.program`

`intent_plan.program` 现在是 `IsolatedFromAbove + SymbolTable + SingleBlock` 的 executable
container，定义在 `include/Intent/Dialect/Plan/IR/PlanOps.td`。它拥有：

- 唯一 `intent.kind = "physical"` function；
- `device` 与 `launch`；
- axis/range/region binding；
- value residency、padding、transfer、reduce、scan、contract、stage decisions；
- 可选的一个 provider `target_program`；
- search space 仍是非 executable tuner metadata，不参与程序 authority。

原来表示 launch mapping 的 `intent_plan.program` record 改名为 `intent_plan.launch`，避免
“整个物理程序”和“某一条 launch mapping”继续共用一个名字。

`ProgramOp::verify()` 约束 program 必须直接位于 module、恰有一个 physical function、entry
名称匹配，并拒绝任何非 `func`/`intent_plan` 的未知子项。`verifyGpuProgram()` 继续验证 GPU
ownership、range、workspace、stream、stage 与 structured decisions 的组合合法性。

### 3.2 Partial conversion，而不是复制第二份程序

本轮没有发明一套与 Intent op 一一重复的新 physical op 集合。canonical function 被原地转移
为 physical function；仍具备合法物理语义的 Intent SSA/control/structured operations 被保留，
GPU decisions 作为同一 current program 的 typed extensions 存在。这与 Triton TTIR 到 TTGIR
允许 legal Triton/SCF/arith ops 留在同一 current IR 的 partial conversion 做法一致。

关键不变量不是 op 名字必须全部变化，而是旧 executable authority 已经消失：后续代码拿不到
module-level canonical kernel，只能通过 `getPhysicalEntry(program)` 得到当前 physical function。
node/value ID 只用于 provenance 和把 selected decision 绑定到当前 function；verifier 保证这些
引用闭合，它们不再指向另一份可执行程序。

## 4. Shared GPU pass pipeline

新的公共入口是 `runPhysicalProgramPipeline()`：

1. `ConstructPhysicalProgramPass`
   - 先验证 canonical Kernel IR；
   - 建立 pass-local `KernelModel`/`KernelFacts`；
   - 执行现有 axis/range/stage、residency、padding、transfer 和 structured-op policies；
   - 产出完整 program，并把 canonical entry 转移为 physical entry。
2. MLIR verifier 在第一项 pass 返回后立即运行。
3. `VerifyPhysicalProgramPass`
   - 验证完整 GPU physical contract；
   - 不接受 partial record、未绑定 range 或缺失 launch 的中间态。

这两项通过 `mlir::PassManager` 顺序执行，并显式启用 verifier。旧的直接
`realizeKernel(module, device)` 入口和文件已删除。

没有把 execution/value/access/structured 四组概念机械拆成四个空 pass。当前这些选择之间
共享同一次 axis/provenance/stage 分析；为了制造 pass 数量而发布不完整中间态，会违反
“每个 pass 边界都完整合法”的约束。四组仍然是审计每项 decision 的语义分类，不要求与
C++ pass class 一一对应。之后只有某一组能在完整合法 program 上独立替换 realization 时，
才有理由拆成单独 refinement pass。

所有 analysis 都在所属 pass 内创建并销毁。旧 `SurfacePlan` 改为 `ProgramAnalysis`，只作为
provider materialization pass 的派生索引，不被序列化，也不跨 IR rewrite 保存。

## 5. Provider lowering 与 terminal translation

### 5.1 Provider materialization pass

共同层新增 `MaterializeTargetProgramPass`。它：

1. 验证当前 physical program；
2. 取得 program 内唯一 physical function；
3. 构造 pass-local `PhysicalProgramIndex`；
4. 调用 Triton、cuTile 或 TileLang 自己的 `ProgramMaterializer`；
5. 在同一 program 中写入一个 `intent_plan.target_program {provider, source}`；
6. 再次经过 MLIR verifier 和 GPU program verifier。

三家不再被描述为同粒度的 printer。它们共享 physical obligations 和 pass contract，但拥有
各自的 operation handling、capability rejection、target forms、syntax 与 wrapper 接线。
TileLang 可以做比 Triton 更显式的 buffer/copy materialization；这不会要求 shared Plan
复制 TileLang 的字段表。

### 5.2 Terminal translator

`translateTargetProgram()` 只执行三件事：

- 找到唯一 physical program；
- 找到与请求 provider 匹配的唯一 `target_program`；
- 把已经 materialized 的 source 写到输出流。

它不再接受 `KernelModel`，不遍历 function，不索引 Plan，不调用 operation handlers，也不能
新增 tuner、workspace、mask 或 execution structure。任何 provider 能力缺失必须在前面的
materialization pass 报错。

目标端目录因此从 `Emission/` 改为 `Lowering/`：

```text
Target/<Provider>/Lowering/
  Driver/TargetProgram.cpp
  Handlers/Operations.cpp
  Materialization/Program.cpp
  Support/Model.h
  Syntax/Spelling.*
```

共同层相应为 `Target/Common/Lowering/`。目录现在表达的是 lowering/materialization 与
terminal translation 的真实边界，不再把全部工作模糊地称为 emission。

## 6. 参考编译器带来的具体取舍

### Triton

- 借鉴 backend-owned、顺序明确的 pipeline，而不是把 pass registration 当 pipeline；
- 借鉴 TTIR→TTGIR partial conversion：legal 高层 op 可以留在 current executable IR；
- 借鉴每个 conversion stage 后验证 legality；
- analysis 在改写后默认失效，不作为 serializer 的隐含第二份状态；
- translation 与 transformations 分离。

### TileLang

- 借鉴 planning/materialization/codegen 的硬边界；
- 借鉴 `InferLayout` 与 `Lower` 分开，而不是由最终 printer 重新决定 form；
- 借鉴未物化 carrier 必须 fail-closed；
- 没有把 TileLang 的 CUDA storage scope、TMA、barrier protocol 或 pipeline knob 抬进 shared
  physical program。

## 7. 删除的错误路径

本轮代码审计确认以下符号/路径已经不存在：

- `intent_plan.realization` / `RealizationOp`；
- 旧 mapping record `intent_plan.program`（已成为 `launch`）；
- `realizeKernel` 与 `Realization/Driver/Realize.cpp`；
- `emitTargetSource`、`emitTritonSource`、`emitCuTileSource`、`emitTileLangSource`；
- `TargetSourceEmitter` / provider `SourceEmitter`；
- `SurfacePlan`；
- 三家及共同层的 `Emission/` 目录；
- legacy/new pipeline 开关、fallback branch 或过渡期兼容入口。

`SearchSpaceOp` 没有删除，因为它只声明 provider tuner 的合法参数，不是 executable authority。
三份 provider operation projection 也没有误删：它们是三种不同目标 API 的必要 leaf code，
不是重复的 shared decision。

## 8. 验证

本轮没有做全量，按 execution/value/access/structured 的实际影响面选择现有手动 repro：

| repro | 覆盖 | 结果 | 本次观测 |
|---|---|---|---|
| `./examples/run/repro.sh triton value_select` | physical SSA/value + terminal translation | PASS | p50 约 0.0774 ms |
| `./examples/run/repro.sh cutile value_select` | cuTile provider materialization | PASS | p50 约 0.0895 ms |
| `./examples/run/repro.sh tilelang matrix_transpose` | access coverage + TileLang buffer/copy | PASS | p50 约 0.0916 ms |
| `./examples/run/repro.sh triton layer_norm_backward` | multi-stage/structured/workspace | PASS | generated p50 约 0.0761 ms |

最初选择的 `attention_backward` 编译和执行成功，但当前数值对照报错。为区分回归与既有状态，
使用同机旧提交 `5bfd69a`、独立 build root 做了 A/B；新旧两边稳定得到完全相同的误差：

```text
(dQ, dK, dV) = (0.01629638671875, 0.035400390625, 0.12646484375)
```

因此它不是本轮 V2 重构引入的回归。本轮没有改算法、容差或 reference 来制造 PASS；用已稳定
通过的 LayerNorm backward 覆盖 multi-stage 主链。临时 worktree 已删除。

## 9. 取舍与仍需诚实说明的边界

1. **共享 construction 当前是一个完整 pass，而不是四个空壳 pass。**这是有意选择；现有
   policies 先产出一份合法程序，再谈能保持 legality 的独立 refinement。
2. **provider form selection 与 source materialization 当前融合在一个 provider pass。**pass
   的输出已经是显式、可验证的 `target_program`，terminal translator 已纯化；如果以后确有
   provider-local form 需要被另一 pass 观察和替换，应在 materializer 前增加 provider IR
   extension，而不是把决定放回 translator。
3. **physical function 仍保留 legal Intent ops。**这是 partial conversion，不是双份 KIR。
   若后续某个 Intent op 的 physical semantics 不能由当前 operation + selected decisions 完整
   表达，应把该 op conversion 成新的 physical op；不能重新让 leaf 回看一份 canonical KIR。
4. **没有修改 `doc/`。**本轮状态与验证只记录在 report；正式规格何时同步由用户单独决定。
5. **未触碰工作区中已有的 `report/history/` 删除。**这些变化不属于本轮编译器重构。

## 10. 提交

本轮按可独立审阅的模块提交：

- `22b39d2 compiler: make physical program the executable authority`
- `458493d compiler: run physical construction as verified passes`
- `a19f911 compiler: separate provider materialization from translation`

