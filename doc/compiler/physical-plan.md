# Physical Plan

Physical Plan 是 compiler-owned 的机器实现决定，不是用户填写的 schedule DSL，也不复制 Kernel IR 中的算法。

## 两类对象

`intent_plan.realization` 保存已经确定、发射器必须机械兑现的决定：

```text
device       算法层可用的机器能力与资源上限
axis         logical domain → physical role / tile role
program      ownership + traversals[]
storage      value → machine storage class
transfer     access / boundary fill / materialization / defer
reduction    target-independent reduction role与物理存储
pointwise    operation role、复用和物化位置
contract     primitive role、累加语义、转置与 operand storage
stream       ordered axis、tile、carry storage
ragged       relation、outer domain、member domains、ragged traversal
stage        跨 kernel 的物理阶段和 workspace 边界
atomic       terminal combine mechanism
```

`intent_plan.search_space` 保存尚未选择、明确委托给目标后端 tuner 的合法轴和参数角色。Realizer 负责证明候选的结构合法性并给出参数关系；候选值、排序和赢家由 Triton、cuTile 或 TileLang 自带 tuner 决定。源码结构选择不进入 search space。

两类对象不能混用：realization 中不存在“运行时再猜”的字段，search space 也不能改变 ownership、遍历顺序、边界语义或数值语义。

## 组合式 program 决策

`intent_plan.program` 把两件正交的事分开保存：

- `ownership`：哪类 physical program 拥有 logical work，当前 GPU machine schema 为 `row`、`tiled` 或 `ragged`；
- `traversals[]`：program 内部如何推进，可组合地记录 `persistent`、`grouped`、`ordered_stream`、`staged` 等机制。

因此 row + ordered stream、tiled + ordered stream、ragged + ordered stream 是同一组部件的不同组合，不是三个 kernel 类别。Verifier 检查组合关系：例如 `ordered_stream` 必须有对应 `stream`，`staged` 必须有 stages，ragged ownership 必须有 ragged relation；它不根据 kernel 名称选择模式。

`intent_plan.ragged.member_nodes` 是一个 domain 集合。同一 ragged relation 可以同时约束被 program 拥有的 member domain 与被 ordered stream 遍历的 member domain；是否 owned 或 streamed 由 axis role 和 program/stream binding 决定，而不是由 relation 本身硬编码。

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
