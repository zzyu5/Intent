# Physical Plan

Physical Plan 是 compiler-owned 的机器实现决定，不是用户填写的 schedule DSL，也不复制 Kernel IR 中的算法。

## 两类对象

`intent_plan.realization` 保存已经确定、发射器必须机械兑现的决定：逐轴角色与 range、program-space 映射、block extent、logical buffer residency、transfer/padding、structured primitive 的物理角色，以及确有需要的操作片段与临时值边界。Ragged relation、state stream、def-use 和算法阶段仍以 Kernel IR 为唯一真理；公共 KernelModel 只派生一次语义索引，Plan 通过稳定 node ID 引用它们并保存已选物理关系，不复制第二份算法 schema。

Structured primitive 的 binding 必须把 emitter 机械投影所需的规范角色与逻辑轴引用写进 Plan。例如 scan 保存 canonical semantics、logical axis node、tensor axis 与 result residency；target 不得回到 Kernel op 各自重推这些字段。Chunk/carry/materialization 尚未选择时则明确缺失，不能由某个 leaf 私自补成自己的实现策略。

`intent_plan.search_space` 保存尚未选择、明确委托给目标后端 tuner 的合法轴和参数角色。Realizer 负责证明候选的结构合法性并给出参数关系；候选值、排序和赢家由 Triton、cuTile 或 TileLang 自带 tuner 决定。源码结构选择不进入 search space。

两类对象不能混用：realization 中不存在“运行时再猜”的字段，search space 也不能改变 ownership、遍历顺序、边界语义或数值语义。

## 组合式逐轴决策

Realizer 不先问“kernel 属于哪一类”，而是逐个逻辑轴回答：是否 parallel、ordered、reduction、ragged member 或 lane；哪一个 range 用于 program ownership、块内 lane、ordered traversal、reduction，哪一个 access range 描述某次读取的覆盖范围；多级 traversal 则在同一轴上保留不同 level。

因此不规则 membership 与 ordered stream、分阶段 contraction 与 ordered traversal、一个轴的外层块和内层顺序都由角色与 range 的组合得到，不需要新增互斥 mapping mode。Relation、def-use 与 provenance 从 Kernel IR 在公共 KernelModel 中派生一次；SurfacePlan 只能索引这份语义事实和 Plan 中已选的物理绑定，不能遍历周围 operation 再重建一份。

每条 range 同时保存 canonical logical extent 与已选 tile。Region block argument 通过稳定 value ID 显式绑定到 axis、range purpose 和 level；state stream 显式绑定到 axis、range purpose/level 以及适用的 ragged relation。Row-vector 上界、stream 上界和 region 归属因此都由 leaf 直接读取，leaf 只负责目标符号和语法拼写。

## 稳定引用与验证

Physical Plan 使用 Kernel IR 的稳定 operation/value ID 引用逻辑节点，不依赖函数名、Python 变量名或整-kernel matcher。每个 binding 必须满足三层约束：

- 被引用的 Kernel IR 节点存在且种类相符；
- machine realization 保持 logical workset、def-use、effect、state 与 ABI；
- target projection 只使用该表面真实能表达的 realization 子集。

Plan 只由 C++ `intent-compile` 的 realization 阶段构造。Python frontend 到 canonical Kernel MLIR 为止；项目中没有 Python `PhysicalPlan`、Python Plan verifier 或 Python Plan serializer。

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

两个已否决的默认策略不重新引入：卷积式重叠读取可以记录比写区域更大的 access footprint，但显式物化唯一 halo 覆盖在现有目标上更慢；边界也不默认展开成逐元素搬运，优先使用收紧范围、整块守卫、目标原生 checked transfer 或 mask。只有目标能力要求且能保持性能语义时，leaf 才可选择自己的等价拼写。

当前 `private_workspace` 使用外层分配的全局设备内存，这是总能成立的驻留位置，不代表物理 placement 已经选优。Owner 线性化与行主序偏移只有一份共享投影；workspace 应驻留 global、shared 还是目标私有存储仍是明确未决的机器决定。
