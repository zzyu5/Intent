# Kernel IR

Kernel IR 表示一个完整、runtime-visible 的 source logical callable algorithm。它位于 Python frontend 与 realizer 之间。

每个 operation 与 SSA value 都有 module 内稳定的非负 node id。Physical Plan 只能通过这些 id 引用 Kernel IR，不复制或按 source 文本重新猜测算法节点。

Intent Kernel MLIR 是 Kernel IR 的正式 backend-boundary 表示。Function parameter、operation result 与 nested-region block argument 的 value ID，以及每个 operation 的 node ID，都显式进入 MLIR metadata；SSA 打印名称不承担 identity。MLIR consumer 集中验证 function ABI metadata、真实 SSA type/rank/shape/access mode、node/value ID 唯一性、result schema、region-argument schema、structured-region terminator、effect/index metadata 与 operation 所需属性。通过后，公共 KernelModel 对 parameter、operation result 和 nested-region block argument 提供同一套 value-ID 查询；target leaf 不再解析 metadata 建第二份身份表。

## 必须保存的内容

### Entry 与 ABI

- `I.In`、`I.Out`、`I.InOut` view；
- symbolic shape、dtype、stride/layout constraint；
- runtime scalar 与 `I.Constexpr`；
- alias、alignment 与 effects；
- wrapper-visible partition 与 workspace relation。

### Logical indexing

- domain、region 与 product/ragged membership；
- positional tensor shape；
- broadcasting、reshape、transpose；
- region 到原始 logical indices 的 relation；
- gather/scatter index relation 与 conflict semantics。

所有参与地址形成的 logical index 都遵守同一宽度不变量：在可达 shape、stride 和 offset 范围内不得溢出或窄化。具体 surface 无法表达所需宽度时由 target capability 明确拒绝，不能把宽度变成 Physical Plan 的可调选择。

### Tensor-flow

- pure tensor SSA；
- explicit dtype 与 `cast`；
- pointwise math 与 logical mask；
- `reduce`、`scan`、它们引用的 typed pure combiner helper，以及 `contract`；
- logical buffers；
- atomic、mutable load/store 与 RNG identity。当前语言没有 public fence 构造；没有 scope、ordering 与 participant 合同的同步不会以 no-op 进入 Kernel IR。

### Control 与 state

- runtime `if`、`for`、`while`；
- specialization-time branch；
- `parallel`、`ordered` 与 `state_stream`；
- rank-one domain/region 的 exclusive `end`，以及由它收紧的 stream logical stop；
- 支配后续精确 index/view/axis 访问的 unsafe `assume_in_bounds` 前置条件；
- carry schema、initial state、step 与 final projection；
- 普通 `@intent.fn` 展开的算法 helper relation，以及 structured combiner 保留的 typed、effect-free helper body。

普通 Python loop 的 `break`/`continue` 在 frontend 被改写成结构化 `if`、`for`/`while` 与 carried control state；Kernel IR 不保留独立终止类节点。是否在现有语料中出现不影响这条语言合同。

Python tuple state 在 Kernel IR 中正规化为有序的多 SSA carry/result schema；需要字段身份的复合值使用 `RecordType` 与 `make_record/extract`。Kernel IR 不保留一个无法被后端观察的 opaque tuple object。

## IR 不包含的内容

Kernel IR 不保存 Python wrapper、完整计算图、physical worker id、grid、internal tile、address arithmetic、physical mask、storage address space、fragment layout、pipeline 或 launch attributes。

这些信息属于 [Physical Plan](physical-plan.md) 或 generated backend program。

## 关键不变量

1. 一个 Kernel IR module entry 对应一个 source `@intent.kernel` 和一个 target callable entry；目标 entry 内可以包含多个 compiler-private execution stages。
2. `I.auto` 只能占据内部 region extent hole，不能成为普通 SSA value。
3. `partition(count=...)` 是保留但尚未 realization 的 Core 语义，当前 frontend 明确拒绝；启用后 count 必须是 source-visible runtime/shape/`Constexpr`/wrapper value。
4. Physical refinement 不得改变 logical workset、state、effect、ABI 或 wrapper-visible relation。
5. Pure SSA 可以安全地复制、删除、融合或重算；effectful node 必须保持依赖与执行语义。
6. `ordered` 与 `state_stream` 的 source 顺序不可降格为 unordered partial merge。
7. `reduce`、`scan` 与 `contract` 保持为 structured nodes，直到后端选择 physical implementation；generic reduce/scan combiner 是 typed、effect-free 的 Kernel IR helper，不是 opaque callable。
8. Logical identity 来自原始 domain index，不来自 physical worker 或 auto-region ordinal。
9. Address-forming index 的宽度必须覆盖已声明 shape/stride 的可达地址范围；不能依赖目标默认整数宽度静默回绕。

## 算法与物理 refinement

Kernel IR 固定 logical node，允许 realizer 为该 node 构造复合实现。例如一个 `contract` 可以变成多条 MMA 和补偿步骤，一个 `reduce` 可以变成多级 private partial；但不能把 ordinary GEMM 变成 Strassen，或把 stable softmax 变成 online recurrence。
