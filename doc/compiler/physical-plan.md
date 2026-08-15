# Physical Plan

Physical Plan 是 compiler-owned 的机器实现决定，不是用户填写的 schedule DSL，也不复制 Kernel IR 中的算法。

## 两类对象

`intent_plan.realization` 保存已经确定、发射器必须机械兑现的决定：逐轴角色与 range、program-space 映射、block extent、logical buffer residency、transfer/padding、structured primitive 的物理角色，以及确有需要的 execution-stage operation slice、stage-axis tile/worker binding 与 synchronization。Ragged relation、state stream、def-use 和算法阶段仍以 Kernel IR 为唯一真理；公共 KernelModel 只派生一次语义索引，Plan 通过稳定 node/value/axis 引用保存已选物理关系，不复制第二份算法 schema。

Structured primitive 的算法语义只从 Kernel IR 读取，Plan 只补充 emitter 机械投影所需的已选物理角色与逻辑引用。例如 scan 的 combine/inclusive semantics 来自 canonical scan op，Plan 绑定 logical axis/tensor axis、result/carry residency、owner 与 materialization slice；leaf 逐 op 同时读取这两个权威来源，但不得根据周围结构重推 axis、owner 或 materialization。Chunk/carry/materialization 尚未选择时则明确缺失，不能由某个 leaf 私自补成自己的实现策略。

`intent_plan.search_space` 保存尚未选择、明确委托给目标后端 tuner 的合法轴和参数角色。Realizer 负责证明候选的结构合法性并给出参数关系；候选值、排序和赢家由 Triton、cuTile 或 TileLang 自带 tuner 决定。源码结构选择不进入 search space。

两类对象不能混用：realization 中不存在“运行时再猜”的字段，search space 也不能改变 ownership、遍历顺序、边界语义或数值语义。

## 组合式逐轴决策

Realizer 不先问“kernel 属于哪一类”，而是逐个逻辑轴回答：是否 parallel、ordered、reduction、ragged member 或 lane；哪一个 range 用于 program ownership、块内 lane、ordered traversal、reduction，哪一个 access range 描述某次读取的覆盖范围；多级 traversal 则在同一轴上保留不同 level。

因此不规则 membership 与 ordered stream、分阶段 contraction 与 ordered traversal、一个轴的外层块和内层顺序都由角色与 range 的组合得到，不需要新增互斥 mapping mode。Relation、def-use 与 provenance 从 Kernel IR 在公共 KernelModel 中派生一次；SurfacePlan 只能索引这份语义事实和 Plan 中已选的物理绑定，不能遍历周围 operation 再重建一份。

每条 range 绑定 canonical logical axis、用途/层级、target specialization extent spelling 与已选 tile。这里的 extent spelling 是 ABI/runtime specialization 的稳定引用，不是独立 shape schema；真实 shape 仍只由 Kernel IR/ABI 定义并集中校验。Region block argument 通过稳定 value ID 显式绑定到 axis、range purpose 和 level；state stream 显式绑定到 axis、range purpose/level 以及适用的 ragged relation。Row-vector 上界、stream 上界和 region 归属因此都由 leaf 直接读取，leaf 只负责目标符号和语法拼写。

## 稳定引用与验证

Physical Plan 使用 Kernel IR 的稳定 operation/value ID 引用逻辑节点，不依赖函数名、Python 变量名或整-kernel matcher。每个 binding 必须满足三层约束：

- 被引用的 Kernel IR 节点存在且种类相符；
- machine realization 保持 logical workset、def-use、effect、state 与 ABI；
- target projection 只使用该表面真实能表达的 realization 子集。

Plan 只由 C++ `intent-compile` 的 realization 阶段构造。Python frontend 到 canonical Kernel MLIR 为止；项目中没有 Python `PhysicalPlan`、Python Plan verifier 或 Python Plan serializer。

## Execution stages

一个 logical callable 需要多个 machine stages 时，Plan 只保存真正选择出来的内容：stage identity、operation slice、`same_stream` synchronization，以及每个 stage 轴绑定到哪个 logical domain/value dimension 后选定的 tile 与 worker axis。Operation slice 是物理 grouping 决定；它可以在保持同一 Kernel IR 算法与 effects 的前提下选择 pure recomputation 或 intermediate materialization。

Dependencies、input/output values、effectful terminals、intermediate producer/consumers、lifetime 与 visibility 都可由 `Kernel IR + operation slice + synchronization` 唯一得到，因此不再作为 `StageOp` 字段，也没有独立 `StageBufferOp`。公共 emission index 每次从 def-use 与 memory effects 重算并验证：producer 必须在 consumer 之前、intermediate 只有一个来源、final/intermediate stage 形态不混用、effectful terminal不能被多个 stage 复制。这个 index 是缓存，不是第三层 IR，也没有 serializer 或一致性 verifier。

`same_stream` synchronization 表示 target 必须按 Plan 顺序提交 private launches，并以同一 execution stream 的 happens-before 兑现派生 dependency 与 visibility。Stage grouping 已由 operation slices 给出，surface 不得重新分组。

跨 compiler-private stage 或跨 source callable 的融合不属于这个算子编译器，Physical Plan 不提供让 leaf 合并 stages 的权限。不同 target family 可以选择不同的初始 stage grouping，但 leaf 都不能改写已经选定的 Plan。

## Ownership 与 physical identity

Source 定义 logical region space：

\[
R=\{\text{logical region instances}\}
\]

Plan 构造 physical worker space：

\[
W=\{\text{program / CTA / thread / task}\}
\]

并定义：

\[
\operatorname{own}:W\rightarrow\operatorname{Seq}(R)
\]

`program_id`、`ct.bid` 或 `T.Kernel` block binding 是 target surface 对这份 ownership 的拼写，不是 portable source identity。Grouped ordering、persistent traversal 和 ordered state stream 同理：决定在 machine plan 中只做一次，各 surface 只投影与渲染。

## 数值与实现边界

Kernel IR 保存数学角色、dtype、累加语义、logical validity、state transition 与 effect。Plan 可以选择 tile、ownership、遍历、storage、target primitive、stage 和合法搜索轴；不能改变 tensor-flow、logical workset、wrapper-visible ABI 或数值角色。

Layout 推断、寄存器分配、指令选择以及给定参数后的低层流水线尽量委托给下层。某个 surface 中不存在的概念不会为“字段对齐”而被抬到共享 Plan；它要求显式打印的机器决定则必须来自同一份 realization，不能在 emitter 中重新选择。

`I.partition(...)` 改变 source body 看见的对象：作者选择的是一个 region 而不是一个元素，因此属于算法结构。相反，当 body 只含逐点标量 SSA、没有 reduction、contract、scan、region mask、logical buffer、atomic/scatter 或 tensor-valued 中间量时，把多个独立标量实例装进一个 physical program 的 lane 只是 realization；body 仍只看见一个元素。Realizer 不得跨过这条判据把标量 contraction 自动升级成块 contraction。

地址索引宽度是正确性不变量，不是搜索参数。地址上界超过某个 surface 的可表达范围时，该 surface 必须明确拒绝；不能窄化、回绕，也不能把宽度放进候选空间。

卷积式重叠读取可以记录比写区域更大的 access footprint，但 Plan 不要求显式物化唯一 halo 覆盖。边界也不要求展开成逐元素搬运；收紧范围、整块守卫、目标原生 checked transfer 或 mask 都是保持同一 logical validity 的合法投影。Leaf 只能在目标能力允许且保持语义时选择等价拼写。

`private_workspace` 的 placement 是机器决定；owner 线性化与行主序偏移只有一份共享投影。Leaf 只拼写 Plan 选择的 storage 与地址关系，不得自行重选驻留位置。
