# 语言定位与边界

## 定义

Intent 是一门 Python-hosted、跨后端、面向 logical domain 与结构化 tensor-flow 的算子 kernel DSL。

用户在一个 `@intent.kernel` 中写出完整的 kernel 内算法：logical domain、算法可见的 segment/region、tensor-flow、顺序或独立工作、carry state、structured computation、控制流与 effects。用户不先替编译器搭建 blocking skeleton，也不写物理 worker identity、tile、grid、地址运算、storage placement、fragment layout 或 pipeline。

最短定义是：

> Intent 保留一个 logical callable 内的完整算法，抽掉它对具体机器 realization 的绑定。

形式上：

\[
\operatorname{compile}_{t}(K[s]) \rightarrow (E_t, L_t)
\]

- \(K\)：Intent source kernel；
- \(s\)：用户 specialization；
- \(E_t\)：目标上对调用方可见的一个 callable entry；
- \(L_t\)：该 entry 的单次 kernel launch Physical Program realization。

一次 source-kernel invocation 对应一个 target kernel 与一次 launch。需要 partial buffer、跨 launch workspace 或多遍调用的算法由作者在 wrapper 中写成多个 source kernels；realizer 不把一个 kernel 拆成多个 launches。

## 在完整程序中的位置

Intent 是嵌入普通 Python 模块的 eDSL，不是独立的 `.intent` 文件语言。

```text
Python / framework wrapper
        ↓
一个可 dispatch 的 Intent kernel entry
        ↓
设备执行
```

它替换 Triton、TileLang 或 cuTile kernel 所在的位置，不接管其上方的计算图。

普通 Python wrapper 负责：

- 检查 shape、dtype 和 device；
- 分配输出与跨 kernel workspace；
- 绑定用户 specialization；
- 按作者写定的顺序调用一个或多个 kernels；
- 根据 target、shape、dtype 或库策略选择 source variant；
- 注册 framework custom op 与 autograd 接口。

多 source-kernel 算法由 wrapper 明确表达，Intent 不自动融合或拆分 source callables。单个 callable 内部的数据依赖只能在同一次 launch 内通过重算、片上暂存或单-launch private workspace 兑现。

## 语义与实现分层

Compiler 的实现自由来自语言语义和可证明事实，不来自作者额外授予“可以分块、可以重排、可以向量化”的权限。一个 source construct 只有在改变 body 可观察的 logical workset、state、effect、数值合同或 ABI 时才属于语言；只决定 GPU program 怎样分块的内容属于 Physical Program。

Source 固定：

- kernel ABI、输入输出、alias 与 effects；
- 算法阶段、数据遍数、状态 schema 与更新；
- logical domain、作者可观察的 segment/region 与 indexing relation；
- 普通顺序 `for`、`parallel` 的实例独立性，以及 `state_stream` 的 segment/carry/stop 语义；
- `reduce`、`scan`、typed pure combiner，以及目标矩阵原语支持的 `contract` 语义；这些 structured op 自身不定义一棵 source 物理树；
- stable、online、multi-pass 等算法选择；
- 显式 dtype、数值 `cast`、等宽 `bitcast`、packed-format 索引/位运算与数学表达；
- gather/scatter 的索引和冲突语义；
- runtime control flow 与用户 specialization；
- wrapper、输出、RNG identity、effect 或其他 kernel 可见的 partition count/extent。

Realizer 决定：

- 从完整 logical domain 和 structured op 引入哪些 physical regions、它们如何嵌套，以及逐轴角色与合法 sub-tiling 关系；
- logical instance/segment 到 program、CTA、thread、task 的 ownership，以及不改变 source 实例语义的 batching、lane packing 与 tensorization；
- program folding、grid-stride、persistent traversal 与 swizzle；
- logical validity 的物理兑现、access footprint、tail 与 address formation；
- 算法结构要求的 storage level、片上复用边界与 target primitive 数值角色；
- 可交给下层 tuner 的合法参数轴与资源上界。

下层 target compiler 决定不依赖 Intent 独有算法信息的部分：layout 推断、寄存器分配、指令选择、给定候选后的低层 pipeline/prefetch/unroll，以及候选值、排序与赢家。Surface 变强时 Intent emitter 应变薄，不把这些决定重新搬进共享 Plan。

根规则是算法可观察性：

> Realizer 可以自由改变物理实现，但不能改变 source 的 tensor-flow、logical workset、state、effect、ABI 或 wrapper-visible 约定。

因此 pure expression 可以 CSE、融合、重算或 spill；`parallel` instances 可以被物理批处理；reduction/scan/contract 可以选择其语义允许的物理层次；f32 contraction 可以使用目标正常支持的机制。只有真正改变 logical workset、state transition、effect 或数值算法时，才需要不同 source。

这里的 pure-expression fusion 只发生在一个 source callable、一次 launch 的既定算法内部。跨 source callable 的融合、自动 kernel fission 与跨-launch workspace orchestration 不属于 Intent compiler。

## 明确不属于 Intent Core

Intent Core 不做：

- graph partition 或 framework operator fusion/fission；
- 自动决定一个 framework op 使用几个 kernels；
- 改变 kernel 调用顺序；
- stable softmax 与 online softmax之间的算法替换；
- ordinary GEMM 与 Strassen 之间的算法替换；
- atomic bucket 与 radix grouping 之间的算法替换。

Portable source 不暴露 `program_id`、block/thread/warp id、grid、`num_warps`、`num_stages`、物理 address space、MMA fragment layout、physical barrier 或 pipeline schedule。
