# Physical Program 与 Plan decisions

Physical Program 是 compiler-owned、可由 pass 改写的 executable MLIR；其中的 Plan decisions 记录多个合法机器方案中已经选定的实现决定。二者都不是用户填写的 schedule DSL，也不复制 Kernel IR 中的算法。

## Executable program 与搜索空间

`intent_plan.program` 是 compiler-owned、唯一可持续改写的 executable Physical Program。Shared passes 在其中确定跨 provider 仍成立的 execution、value、access/validity 与 structured-operation obligations；provider-local ProgramForms 再在同一 program 中选择并物化 provider-specific forms。Ragged relation、state stream、def-use 和算法阶段仍以 Kernel IR 为唯一真理；公共 KernelModel 只派生语义索引，Physical Program 通过稳定 node/value/axis 引用保存已选物理关系，不复制第二份算法 schema。

Structured primitive 的算法语义只从 Kernel IR 读取，Physical Program 只补充 lowering 所需的已选物理角色与逻辑引用。例如 scan 的 combine/inclusive semantics 来自 canonical scan op，shared passes 绑定 logical axis/tensor axis、result/carry residency、owner 与 operation slice；provider-local passes 可以在这些绑定下选择 target materialization form，但不得根据周围结构重推 shared axis、owner 或算法语义。

`intent_plan.search_space` 保存尚未选择、明确委托给目标后端 tuner 的合法轴和参数角色。Realizer 负责证明候选的结构合法性并给出参数关系；候选值、排序和赢家由 Triton、cuTile 或 TileLang 自带 tuner 决定。源码结构选择不进入 search space。

两类对象不能混用：Physical Program 中不存在“运行时再猜”的字段，search space 也不能改变 ownership、遍历顺序、边界语义或数值语义。

## 组合式逐轴决策

Realizer 不先问“kernel 属于哪一类”。它从 Kernel IR 读取 parallel independence、sequential/ordered traversal、reduction 与 ragged membership 等算法事实，再逐个逻辑轴选择：哪一个 physical range 用于 program ownership、块内 lane、ordered traversal 或 reduction，哪一个 access range 描述某次读取的覆盖范围；多级 traversal 则在同一轴上保留不同 level。

因此不规则 membership 与 ordered stream、分阶段 contraction 与 ordered traversal、一个轴的外层块和内层顺序都由角色与 range 的组合得到，不需要新增互斥 mapping mode。Relation、def-use 与 provenance 从 Kernel IR 在公共 KernelModel 中派生一次；SurfacePlan 只能索引这份语义事实和 Plan 中已选的物理绑定，不能遍历周围 operation 再重建一份。

每条 range 绑定 canonical logical axis、用途/层级、target specialization extent spelling 与已选 tile。这里的 extent spelling 是 ABI/runtime specialization 的稳定引用，不是独立 shape schema；真实 shape 仍只由 Kernel IR/ABI 定义并集中校验。Region block argument 通过稳定 value ID 显式绑定到 axis、range purpose 和 level；state stream 显式绑定到 axis、range purpose/level 以及适用的 ragged relation。Row-vector 上界、stream 上界和 region 归属因此都由 leaf 直接读取，leaf 只负责目标符号和语法拼写。

## 稳定引用与验证

Physical Plan 使用 Kernel IR 的稳定 operation/value ID 引用逻辑节点，不依赖函数名、Python 变量名或整-kernel matcher。每个 binding 必须满足三层约束：

- 被引用的 Kernel IR 节点存在且种类相符；
- machine realization 保持 logical workset、def-use、effect、state 与 ABI；
- target projection 只使用该表面真实能表达的 realization 子集。

Shared Physical Program 由 C++ `intent-compile` 的 construction/refinement pipeline 构造并验证；provider-local passes 随后在同一 MLIR program 上完成 form selection/materialization。Python frontend 到 canonical Kernel MLIR 为止；项目中没有 Python `PhysicalPlan`、Python Plan verifier 或 Python Plan serializer。

## Execution stages

一个 logical callable 需要多个 machine stages 时，Plan 只保存真正选择出来的内容：stage identity、operation slice、`same_stream` synchronization，以及每个 stage 轴绑定到哪个 logical domain/value dimension 后选定的 tile 与 worker axis。Operation slice 是物理 grouping 决定；它可以在保持同一 Kernel IR 算法与 effects 的前提下选择 pure recomputation 或 intermediate materialization。

Dependencies、input/output values、effectful terminals、intermediate producer/consumers、lifetime 与 visibility 都可由 `Kernel IR + operation slice + synchronization` 唯一得到，因此不再作为 `StageOp` 字段，也没有独立 `StageBufferOp`。公共 emission index 每次从 def-use 与 memory effects 重算并验证：producer 必须在 consumer 之前、intermediate 只有一个来源、final/intermediate stage 形态不混用、effectful terminal不能被多个 stage 复制。这个 index 是缓存，不是第三层 IR，也没有 serializer 或一致性 verifier。

`same_stream` synchronization 表示 target 必须按 Plan 顺序提交 private launches，并以同一 execution stream 的 happens-before 兑现派生 dependency 与 visibility。Stage grouping 已由 operation slices 给出，surface 不得重新分组。

跨 compiler-private stage 或跨 source callable 的融合不属于这个算子编译器，Physical Plan 不提供 leaf 合并 stages 的入口。不同 target family 可以选择不同的初始 stage grouping，但 leaf 都不能改写已经选定的 Plan。

## Ownership 与 physical identity

Source 定义 logical instance space：

\[
L=\{\text{logical index / algorithm-visible segment instances}\}
\]

Plan 可以从 \(L\) 构造 source 中不存在的 physical regions，再构造 physical worker space：

\[
R_p=\{\text{physical regions over subsets of }L\}
\]

\[
W=\{\text{program / CTA / thread / task}\}
\]

并定义：

\[
\operatorname{own}:W\rightarrow\operatorname{Seq}(R_p)
\]

算法可见 segment 必须保持原 boundary/identity；普通 physical region 则只需完整、无非法重复地覆盖对应 logical instances。对 `partition(count=P)`，Plan 显式绑定 canonical partition、source axis、count SSA value、part block argument、region block argument和已选 ownership range；segment extent 只能是 canonical `ceil(N/P)` 的稳定派生引用，不能由 leaf 从 worker count 或相同 shape 反猜。空 part 没有 logical body execution，Plan 可以不为其启动 physical worker，但不能重编号非空 part，也不能删除 wrapper-visible 的 `P` 个 slots 或 identity-initialization relation。

`program_id`、`ct.bid` 或 `T.Kernel` block binding 是 target surface 对这份 ownership 的拼写，不是 portable source identity。Sequential/state-stream order 由 Kernel IR 固定，Plan 只选择 grouped ordering、persistent traversal 等物理兑现方式；这份选择只做一次，各 surface 只投影与渲染。

## 数值与实现边界

Kernel IR 保存数学角色、dtype、累加语义、logical validity、state transition 与 effect。Shared Physical Program 可以选择跨 provider 成立的 granularity、ownership、遍历、storage obligation、stage 和合法搜索轴；target primitive、target-specific access/storage form 等 provider choices 由 provider-local ProgramForms 选择并记录。任何一层都不能改变 tensor-flow、logical workset、wrapper-visible ABI 或数值角色。

Layout 推断、寄存器分配、指令选择以及给定参数后的低层流水线尽量委托给下层。某个 surface 中不存在的概念不会为“字段对齐”而被抬到共享 Plan；它要求显式打印的机器决定则必须来自同一份 realization，不能在 emitter 中重新选择。

Source partition 只在 part identity 或 boundary 被算法、effect、ABI 或 wrapper 观察时改变算法结构。`count=P` 的边界公式、空 part 无执行、part identity 和 wrapper slots 都属于这份 source 语义，不是 Plan 的候选决定。语法上让 body 拿到一个 region 并不足以证明 source partition：如果 region 只用于把若干独立实例凑成 tensor operand，换一种合法机器实现时它无需保留，就应由 Plan 从完整 logical domain 引入。

把多个 `parallel` 实例装进一个 physical program、lane 或 tensor primitive 属于 realization，即使每个 source 实例内部含 reduction、contract、scan、logical buffer 或 effect。合法性判据是实例间没有 source-defined happens-before、逐实例 value/state/effect identity 与冲突语义保持不变，且 source body 不观察新建 region 的 ordinal/boundary；不能以“body 含 structured op”为全局门槛，也不能借物理批处理引入跨实例 reduction、state 或 effect。目标无法保持 effect 合同时，该 batching realization 不合法。

地址索引宽度是正确性不变量，不是搜索参数。地址上界超过某个 surface 的可表达范围时，该 surface 必须明确拒绝；不能窄化、回绕，也不能把宽度放进候选空间。

卷积式重叠读取可以记录比写区域更大的 access footprint，但 Plan 不要求显式物化唯一 halo 覆盖。边界也不要求展开成逐元素搬运；收紧范围、整块守卫、目标原生 checked transfer 或 mask 都是保持同一 logical validity 的合法投影。Leaf 只能在目标能力允许且保持语义时选择等价拼写。

`private_workspace` 的 placement 是机器决定；owner 线性化与行主序偏移只有一份共享投影。Leaf 只拼写 Plan 选择的 storage 与地址关系，不得自行重选驻留位置。
