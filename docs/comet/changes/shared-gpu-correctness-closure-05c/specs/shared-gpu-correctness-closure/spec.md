# Shared GPU 正确性、配置与 Triton 终端闭合

## 目标状态

Canonical KIR lowering 产生一份 provider-neutral、可执行、可独立验证的 shared GPU Program。所有影响执行的 traversal、ownership、range、access、effects、candidate legality 与 device binding 都由 current typed carriers 决定；provider legalization 补齐真实 local forms，serializer 只机械输出已闭合的 terminal program。

## Shared program relation 与 resource authority

- Physical relation identity 至少包含 concrete value occurrence、source identity、source axis、derived identity 与对应 logical/physical range。相同 logical dimension 不能单独证明两个 occurrences 具有相同 traversal、extent 或 fragment projection。
- Lockstep 只有在 logical start/stop/step 与 current physical start/extent/step 均相容时为 exact；无法证明时返回 unknown/inconsistent，并保留合法保守程序或由依赖该事实的 transformation 放弃，不借用另一 occurrence 的 authority。
- Relation、ownership、access、validity、effect、buffer lifetime 与 resource facts 从 current GPU IR 计算；任何影响这些 facts 的 rewrite 后旧分析失效。
- Resource legality 消费 complete candidate、current Physical Program、dtype/element width、liveness/interference 与 typed target limits。Intent 只删除已证明超过 grid/resource/provider-surface 约束的 candidate；layout、register allocation、pipeline 等未能在当前层证明的失败仍由 provider compiler 拥有。

## Config、resource 与 device binding

- Shared tuning table 产生少量完整、相关联的 provider-neutral tuples；provider legalization 将其与自身 local options 组合成非空 closed candidate set。普通默认运行从同一集合选择第一份合法 tuple，不存在第二条固定值或 fallback path。
- TileLang 的 provider-domain expansion、MMA warp partition legality 与非空检查在 provider legalization/verifier 中完成，并成为 current provider program 的 typed config carrier；serializer 不新增、删除或重新验证候选。
- 编译调用选择的 provider、hardware capability 与 concrete GPU device identity 进入 resolved target 和 compiled artifact。Runtime 对所有 external tensor arguments 验证 device type/id 一致且等于 artifact binding；无 tensor 输入时仍使用该显式 binding。
- 输出分配、provider compile/JIT context、stream 与 launch 都使用 artifact 绑定设备。device mismatch 在 generated launch 前给出明确错误，不把按一个设备能力编译的 artifact 静默运行到另一个设备。

## Public shorthand 与 provider forms

- `I.sparse_contract_2to4(compressed_lhs, metadata, rhs, acc_dtype=...)` 是 rank-2 surface shorthand。`metadata` 是与 compressed groups 对齐的 canonical record `{first: index, second: index}`；frontend 固定 `two_of_four(compression_axis=1, logical_extent=rhs.shape[0])`、`reduce=((1,0),)` 与 `batch=()`，并产生与直接 `I.sparse_contract` 相同的 canonical op。Opaque instruction-packed metadata 必须由作者先显式解码，shorthand 不建立第二种 metadata semantics。
- Shared `ScaledContractOp` 保留 closed positional scale-group schema。TileLang provider 对其 ref/native surface 能表达且 target capability 支持的 format、carrier、scale granularity 与 storage 组合形成 provider-local block-scaled GEMM；其它组合在 provider legality 阶段精确拒绝，不改写 shared language semantics，也不串行展开冒充支持。
- cuTile native rewrite 完成后，完整 MLIR module verification 必须执行所有 generated/cuTile op schema verifiers，随后再执行 provider closed-surface verifier。两者分别拥有 schema legality 与 provider policy。

## Triton 终端结果

- 当前 registry 的每个 entry 按其完整 Python orchestration 运行；multi-kernel entry 不拆成独立通过项。
- Generated artifact 必须真实完成 terminal serialization、Triton compile/JIT、launch 与 numerical comparison。Source baseline 必须真实运行其登记实现；compatibility、resource、adapter、timeout 或 measurement gap 均不能记为通过。
- 性能使用 tuning table 的静态默认 generated config，以 source 实际耗时为分母。RTX 5090D 与 H100 上每个 entry 的 `generated_p50_ms / source_p50_ms` 不超过 1.1。
- `report/baselinev2/triton-5090.csv` 与 `report/baselinev2/triton-h100.csv` 保存同一 current compiler state 的真实结果；不合并不同 compiler binary、重复运行挑最优值或修改 cuTile/TileLang 表。

## Acceptance scenarios

Scenario: Shared program 只从精确 occurrence relation 得到执行事实

Given 一个 canonical kernel 含相同 logical dimension 的多个 derived occurrences、不同 source ranges 或 atomic/ordinary effects

When 它经过 shared construction、blocking、relation refinement 与 shared verification

Then current GPU Program 的 traversal、extent、ownership、access、validity 与 effects 由完整 typed occurrence/range facts 唯一决定，dimension equality 不会把 unknown 提升为 exact，程序无需 KIR side record 即可验证与继续 lowering

Scenario: Candidate、resource 与 compile device 在 serializer 前闭合

Given 编译调用选择 concrete GPU device，shared table 提供 complete tuples，provider 声明 local option domains

When compiler 形成 terminal candidate set 并 materialize artifact

Then 只删除 current program 与 capability 能证明非法的 candidates，默认 config 来自同一非空集合，serializer 不作 legality 决策，artifact/runtime 校验所有 tensors，且无 tensor 输入时仍在显式绑定设备完成 allocation、JIT context、stream 与 launch

Scenario: 当前 public/provider 缺口走唯一 typed lowering path

Given 合法的 rank-2 two-of-four shorthand、TileLang 支持的 scaled-contract capability，或 cuTile rewrite 产生的新 provider ops

When 分别经过 frontend canonicalization 与对应 provider legalization

Then shorthand 形成 canonical sparse contract，scaled contract 形成 provider-native block-scaled form或在 unsupported capability 上精确拒绝，cuTile ops 先通过 MLIR schema verifier再通过 closed-surface verifier，TileLang serializer 仅打印已闭合 configs

Scenario: Triton registry 在 RTX 5090D 与 H100 上达到当前 05c 结果

Given 当前 54-entry Triton registry、同一 current compiler state 与各 entry 登记的 source baseline

When 两台设备分别执行完整 generated/source numerical 与 timing workflow

Then 54 个 entries 的全部 62 个 component references 均到达 terminal/JIT/launch，数值比较通过，每项具有真实 timing 且默认 generated ratio 不超过 1.1，两张 Triton CSV 完整记录这些结果
