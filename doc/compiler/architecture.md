# 编译器模块架构

目录与模块边界应直接表达下面的数据流：

```text
Python wrapper / Host API
          ↓
Source Frontend
          ↓
Canonical Intent Kernel MLIR
          ↓
Shared Physical Program construction/refinement
          ↓
Shared Physical Program MLIR
          ↓
Provider-local ProgramForms + materialization
          ↓
Provider-legal Program + verifier
          ↓
Terminal translator
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
- 完整 logical domain、作者可观察的 segment/region、tensor expressions、控制流与 structured primitives；
- logical buffers 与 effects。

Frontend 在 AST lowering 期间只维护 symbol、shape、region、constexpr 与源码位置等临时状态，并直接构造注册过的 canonical Intent Kernel MLIR。Python 不维护一套与 MLIR 平行的 typed Kernel IR；MLIR 进入 backend boundary 后，后端也不得绕回 Python object 重新解释算法。

## Kernel IR

Kernel IR 是 source-visible kernel algorithm 的权威表示。它保存 ABI、logical workset、tensor-flow、state、control、structured nodes、index relation 与 effects。Intent Kernel MLIR 保存稳定 operation/value node ID、结构化 region、类型和 metadata；Kernel IR verifier 集中核对 metadata 与真实 SSA function/result/block-argument schema，公共 KernelModel 再以这些稳定 ID 建立唯一索引。

详见 [Kernel IR](kernel-ir.md)。

## Realizer

Realizer 接收 Kernel IR、机器能力与 compile policy，从完整 logical domain、独立实例和 structured operation 构造 source 中不存在的 physical regions，再选择依赖算法结构才能确定的物理事实：逐轴角色与 range、program ownership、遍历关系、logical validity 的兑现方式、必要的 storage class、primitive 数值角色、execution-stage operation grouping/axis binding/synchronization，以及合法搜索轴。候选值由下层 tuner 选择；layout 推断、寄存器分配、指令选择和给定参数后的低层流水线继续交给下层。

Realizer 不修改 source algorithm，不执行 graph-level fusion/fission，也不改变 wrapper-visible ABI。

Backend boundary 从 Intent Kernel MLIR 开始。C++/MLIR compiler 解析并验证 Kernel MLIR，先由 shared passes 构造并逐步细化唯一的 `intent_plan.program` Physical Program；provider-local ProgramForms 与 materialization 随后把它变成 provider-legal Program，terminal translator 只序列化该结果。Python compiler 只负责把 Kernel MLIR 送入该工具，不保存 Physical Program 对象，也不执行 tile、pipeline 或 launch 决策。

## Physical Program 与 Plan decisions

Physical Program 是 compiler-owned、可由 passes 持续改写的 executable MLIR。Shared passes 通过稳定 node/value ID 把跨 provider 成立的已选物理决定绑定到 Kernel IR，并验证逐轴 range、operation binding、stage slice 与 synchronization；provider-local passes 再选择和物化各 surface 所需的合法 form。由 Kernel IR 与 stage slice 唯一得到的 dependency、boundary values、terminal、lifetime 和 visibility 只存在于可重算的公共 analysis index，不进入第二份 schema。不存在 Python Plan、Python Plan serializer 或绕过 MLIR verifier 的旁路输入。

详见 [Physical Program 与 Plan decisions](physical-plan.md)。

## Provider lowering 与 terminal translation

Provider leaf 由 provider-local ProgramForms/refinement、provider materialization 和 terminal translator 组成。前两者读取并改写同一 Physical Program，选择并物化该 surface 所需的合法 form；terminal translator 只把已经合法化的 provider Program 序列化为目标 source、entry 与 wrapper。不同 GPU surface 共享同一算法与 shared physical obligations，但不要求具有同样粒度的 provider forms；另一类机器则拥有自己的 physical pipeline。Region argument、row-vector extent、stream/ragged relation 与 stage-axis 的 shared 绑定进入可验证的 Physical Program，可重算的数据流边界由公共 KernelModel/analysis index 建一次临时索引。Provider-local passes 可以作有来源的 target-form 选择；terminal translator 不再作结构选择。目标语言即使是可读的 Python source，也不表示物理决定由 Python frontend 实现。

## 每一层的唯一权威来源

| 层 | 持有 | 不持有 |
|---|---|---|
| Kernel IR | ABI、logical workset、tensor-flow、typed combiner、state/control、index relation、effects | tile、worker、storage、target spelling |
| KernelModel / analysis index | 从 Kernel IR 重算的 provenance、def-use、shape、ragged/stream relation、stage boundary | 独立 schema、serializer、与 Kernel IR 一致性 verifier、物理选择 |
| Shared Physical Program | 多个合法机器方案中已经选定的 axis/range、ownership、granularity、storage obligation、operation slice、stage-axis binding、synchronization | 可从 Kernel IR 与所选决定唯一重算的第二份算法事实 |
| Provider-local ProgramForms | target capability 下的 access/storage/primitive form 与 materialization | kernel 分类、shared ownership/流终点重选、算法改写 |
| Terminal translator | provider-legal Program 到目标 API/语法的确定性序列化与 compile/run 接线 | 结构选择、shape/provenance 反推、算法改写 |
| 下层 compiler/tuner | layout、寄存器、指令、低层 pipeline，以及已委托候选的评测和赢家 | Intent source algorithm 与 wrapper orchestration |

详见 [后端 lowering](backend-lowering.md)。

## Compiled Artifact 与 runtime

Artifact 保存组合 MLIR、可读生成源码、可调用 entry，以及首次真实 JIT 后得到的后端/低层 IR。Launch 与 execution-stage policy 位于 Physical Program 和生成源码中，不再以第二套 Python 配置对象保存。Runtime 只负责物化并提交这个 logical entry；完整图与多 source-kernel 调度仍在 Python wrapper。

详见 [编译产物与运行边界](compiled-artifact.md)。
