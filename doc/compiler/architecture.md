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

Kernel IR 是 source-visible kernel algorithm 的权威表示。它保存 ABI、logical workset、tensor-flow、state、control、structured nodes、index relation 与 effects。Intent Kernel MLIR 保存稳定 operation/value node ID、结构化 region、类型和 metadata，并由 MLIR parser 与 Kernel IR verifier 守住 backend boundary。

详见 [Kernel IR](kernel-ir.md)。

## Realizer

Realizer 接收 Kernel IR、机器能力与 compile policy，只选择依赖算法结构才能确定的物理事实：逐轴角色与 range、program ownership、遍历关系、logical validity 的兑现方式、必要的 storage class、primitive 数值角色以及合法搜索轴。候选值由下层 tuner 选择；layout 推断、寄存器分配、指令选择和给定参数后的低层流水线继续交给下层。

Realizer 不修改 source algorithm，不执行 graph-level fusion/fission，也不改变 wrapper-visible ABI。

Backend boundary 从 Intent Kernel MLIR 开始。C++/MLIR compiler 解析并验证 Kernel MLIR，通过共享分析、per-op handler、合法性证明与 target policy 构造 `intent_plan` dialect。Python compiler 只负责把 Kernel MLIR 送入该工具，不保存 Physical Plan 对象，也不执行 tile、pipeline 或 launch 决策。

## Physical Plan

Physical Plan 是 realizer 的 target realization 结果，是独立于 source language 的 MLIR dialect。它通过稳定 node ID 绑定 Kernel IR，可以被验证、比较、搜索与交给 backend lowering。正式边界中只有 MLIR Plan；不存在 Python Plan、Python Plan serializer 或绕过 MLIR verifier 的旁路输入。

详见 [Physical Plan](physical-plan.md)。

## Backend Emitter

Target emitter 只接收经过 MLIR parser 与 verifier 的 `Kernel IR + Physical Plan MLIR`，通过共享遍历和 target spelling table 直接生成 Triton、TileLang、cuTile、CPU SIMD 或 RVV program。Kernel IR 与 Physical Plan 之外没有第三份 target IR；target 侧的临时 binding 只是查找索引，不是可独立验证或持久化的表示。Realization 与 emission 在同一个 `intent-compile` 进程内连续完成，但仍以组合 MLIR 作为严格阶段边界。Triton backend 的目标语言恰好是可读的 Triton Python source，不等于后端决策在 Python 中实现，也不要求先转换成 Triton MLIR。

详见 [后端 lowering](backend-lowering.md)。

## Compiled Artifact 与 runtime

Artifact 保存组合 MLIR、可读生成源码、可调用 entry，以及首次真实 JIT 后得到的后端/低层 IR。Launch policy 位于 Physical Plan 和生成源码中，不再以第二套 Python 配置对象保存。Runtime 只负责物化并提交这个 entry；完整图与多-kernel 调度仍在 Python wrapper。

详见 [编译产物与运行边界](compiled-artifact.md)。
