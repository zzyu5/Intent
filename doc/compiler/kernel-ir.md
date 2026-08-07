# Kernel IR

Kernel IR 表示一个完整、runtime-visible 的 source kernel algorithm。它位于 Python frontend 与 realizer 之间。

每个 operation 与 SSA value 都有 module 内稳定的非负 node id。Physical Plan 只能通过这些 id 引用 Kernel IR，不复制或按 source 文本重新猜测算法节点。

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

### Tensor-flow

- pure tensor SSA；
- explicit dtype 与 `cast`；
- pointwise math 与 logical mask；
- `reduce`、`scan`、`contract`；
- logical buffers；
- atomic、mutable load/store、fence 与 RNG identity。

### Control 与 state

- runtime `if`、`for`、`while`；
- specialization-time branch；
- `parallel`、`ordered` 与 `state_stream`；
- carry schema、initial state、step 与 final projection；
- `@intent.fn` 展开的算法 helper relation。

Python tuple state 在 Kernel IR 中正规化为有序的多 SSA carry/result schema；需要字段身份的复合值使用 `RecordType` 与 `make_record/extract`。Kernel IR 不保留一个无法被后端观察的 opaque tuple object。

## IR 不包含的内容

Kernel IR 不保存 Python wrapper、完整计算图、physical worker id、grid、internal tile、address arithmetic、physical mask、storage address space、fragment layout、pipeline 或 launch attributes。

这些信息属于 [Physical Plan](physical-plan.md) 或 generated backend program。

## 关键不变量

1. 一个 Kernel IR module entry 对应一个 source `@intent.kernel` 和一个 target entry。
2. `I.auto` 只能占据内部 region extent hole，不能成为普通 SSA value。
3. `partition(count=...)` 的 count 必须是 source-visible runtime/shape/`Constexpr`/wrapper value。
4. Physical refinement 不得改变 logical workset、state、effect、ABI 或 wrapper-visible relation。
5. Pure SSA 可以安全地复制、删除、融合或重算；effectful node 必须保持依赖与执行语义。
6. `ordered` 与 `state_stream` 的 source 顺序不可降格为 unordered partial merge。
7. `reduce`、`scan` 与 `contract` 保持为 structured nodes，直到后端选择 physical implementation。
8. Logical identity 来自原始 domain index，不来自 physical worker 或 auto-region ordinal。

## 算法与物理 refinement

Kernel IR 固定 logical node，允许 realizer 为该 node 构造复合实现。例如一个 `contract` 可以变成多条 MMA 和补偿步骤，一个 `reduce` 可以变成多级 private partial；但不能把 ordinary GEMM 变成 Strassen，或把 stable softmax 变成 online recurrence。
