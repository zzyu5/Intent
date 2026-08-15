# 编译器模块架构

目录与模块边界应直接表达下面的数据流：

```text
Python wrapper / Host API
          ↓
Source Frontend
          ↓
Canonical Intent Kernel MLIR
          ↓
Realizer
          ↓
Physical Plan MLIR
          ↓
MLIR verifier + Backend Translator
          ↓
Generated target source
          ↓
Compiled Artifact
          ↓
Runtime Dispatch
```

## Host API 与 wrapper boundary

Host 侧接收 Python kernel object、runtime tensor/scalar、用户 specialization、target 与 compile options。Wrapper 负责分配和多-kernel orchestration，不进入 Kernel IR。

## Source Frontend

Frontend 读取受限 Python eDSL，解析：

- `@intent.kernel` 与 `@intent.fn`；
- signature、view kind、symbolic shape、dtype 与 specialization；
- domain/region、tensor expressions、控制流、structured primitives；
- logical buffers 与 effects。

Frontend 在 AST lowering 期间只维护 symbol、shape、region、constexpr 与源码位置等临时状态，并直接构造注册过的 canonical Intent Kernel MLIR。Python 不维护一套与 MLIR 平行的 typed Kernel IR；MLIR 进入 backend boundary 后，后端也不得绕回 Python object 重新解释算法。

## Kernel IR

Kernel IR 是 source-visible kernel algorithm 的权威表示。它保存 ABI、logical workset、tensor-flow、state、control、structured nodes、index relation 与 effects。Intent Kernel MLIR 保存稳定 operation/value node ID、结构化 region、类型和 metadata；Kernel IR verifier 集中核对 metadata 与真实 SSA function/result/block-argument schema，公共 KernelModel 再以这些稳定 ID 建立唯一索引。

详见 [Kernel IR](kernel-ir.md)。

## Realizer

Realizer 接收 Kernel IR、机器能力与 compile policy，只选择依赖算法结构才能确定的物理事实：逐轴角色与 range、program ownership、遍历关系、logical validity 的兑现方式、必要的 storage class、primitive 数值角色、execution-stage operation grouping/axis binding/synchronization，以及合法搜索轴。候选值由下层 tuner 选择；layout 推断、寄存器分配、指令选择和给定参数后的低层流水线继续交给下层。

Realizer 不修改 source algorithm，不执行 graph-level fusion/fission，也不改变 wrapper-visible ABI。

Backend boundary 从 Intent Kernel MLIR 开始。C++/MLIR compiler 解析并验证 Kernel MLIR，通过共享分析、per-op handler、合法性证明与 target policy 构造 `intent_plan` dialect。Python compiler 只负责把 Kernel MLIR 送入该工具，不保存 Physical Plan 对象，也不执行 tile、pipeline 或 launch 决策。

## Physical Plan

Physical Plan 是 realizer 的 target realization 结果，是独立于 source language 的 MLIR dialect。它通过稳定 node/value ID 把已选物理决定绑定到 Kernel IR，可以验证逐轴 range、operation binding、stage slice/synchronization 与 target capability，再交给 backend lowering。由 Kernel IR 与 stage slice 唯一得到的 dependency、boundary values、terminal、lifetime 和 visibility 只存在于可重算的公共 emission index，不进入 Plan schema。正式边界中只有 MLIR Plan；不存在 Python Plan、Python Plan serializer 或绕过 MLIR verifier 的旁路输入。

详见 [Physical Plan](physical-plan.md)。

## Backend Emitter

Target emitter 只接收经过 MLIR parser 与 verifier 的 `Kernel IR + Physical Plan MLIR`，通过共享遍历和 target spelling table 直接生成目标 program。每个 target family 提供自己的 realizer 与 emitter；不同 surface 只是同一 machine Plan 的投影，另一类机器则产生自己的 machine Plan。Region argument、row-vector extent、stream/ragged relation 与 stage-axis 的已选物理绑定都进入可验证的 Physical Plan；可重算的数据流边界由公共 KernelModel/SurfacePlan 建一次临时索引。Target leaf 只消费这些来源，不再重选。Kernel IR 与 Physical Plan 之外没有第三份 target IR。Realization 与 emission 可以在同一 compiler process 内连续完成，但仍以组合 MLIR 作为严格阶段边界。目标语言即使是可读的 Python source，也不表示后端决定由 Python 实现，更不要求先转换成该目标自己的 MLIR。

## 每一层的唯一权威来源

| 层 | 持有 | 不持有 |
|---|---|---|
| Kernel IR | ABI、logical workset、tensor-flow、typed combiner、state/control、index relation、effects | tile、worker、storage、target spelling |
| KernelModel / analysis index | 从 Kernel IR 重算的 provenance、def-use、shape、ragged/stream relation、stage boundary | 独立 schema、serializer、与 Kernel IR 一致性 verifier、物理选择 |
| Physical Plan | 多个合法机器方案中已经选定的 axis/range、ownership、tile、storage、operation slice、stage-axis binding、synchronization | 可从 Kernel IR 与所选决定唯一重算的第二份算法事实 |
| Target leaf | capability declaration、Plan concept 到目标 API/语法的映射、compile/run 接线 | kernel 分类、ownership/tile/流终点重选、算法改写 |
| 下层 compiler/tuner | layout、寄存器、指令、低层 pipeline，以及已委托候选的评测和赢家 | Intent source algorithm 与 wrapper orchestration |

详见 [后端 lowering](backend-lowering.md)。

## Compiled Artifact 与 runtime

Artifact 保存组合 MLIR、可读生成源码、可调用 entry，以及首次真实 JIT 后得到的后端/低层 IR。Launch 与 execution-stage policy 位于 Physical Plan 和生成源码中，不再以第二套 Python 配置对象保存。Runtime 只负责物化并提交这个 logical entry；完整图与多 source-kernel 调度仍在 Python wrapper。

详见 [编译产物与运行边界](compiled-artifact.md)。
