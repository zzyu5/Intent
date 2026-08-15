# Intent Kernel DSL 设计文档

这组文档是 Intent Kernel DSL 当前唯一的语言与编译器设计规范。它描述最终语义和模块边界，不记录实现进度、历史版本、测试清单或迁移过程。

Intent 是一门 **Python-hosted、跨后端、region-parametric 的结构化算子 kernel DSL**：用户写一个 logical callable 内的完整算法，编译器补全不可由 source 观察的机器 realization；一个 callable 的目标实现可以包含多个 compiler-private stages。

## 冻结状态

本文档组定义的编程模型、canonical Kernel IR 与分层边界自此冻结。冻结不是声称实现永远不变，而是停止把语言构造和表示层当作普通补丁扩张：目录、字段或 target API 可以演进，但不能制造第二份算法真理、改变 source algorithm，或把某个 target 的抽象抬进共享层。

冻结后新增语言构造必须同时满足两条：

1. 一个真实算法无法用现有 Core 表达，且问题不是缺少语法糖、library helper 或作者显式的多-kernel wrapper；
2. 所需能力不能机械委托给下层 target compiler 已有的原语或接口。

冻结后新增物理决定也必须同时满足：其正确取值依赖 Kernel IR 保留的算法结构；该值只有在具体机器上才有意义，作者没有写也不应写。否则需求应当被拒绝、作为 library policy 表达、成为 target capability subset，或交给下层 compiler/tuner，而不是扩展语言或 Plan。

## 阅读顺序

1. [语言定位与边界](dsl/model.md)
2. [Python eDSL](dsl/python-edsl.md)
3. [Domain、Region 与控制](dsl/domains-and-control.md)
4. [Tensor-flow 与 Core primitives](dsl/tensor-flow.md)
5. [Core 与算法库](dsl/core-and-algorithms.md)
6. [数值与确定性](dsl/numerics.md)
7. [编译器模块架构](compiler/architecture.md)
8. [Kernel IR](compiler/kernel-ir.md)
9. [Physical Plan](compiler/physical-plan.md)
10. [后端 lowering](compiler/backend-lowering.md)
11. [编译产物与运行边界](compiler/compiled-artifact.md)

典型 kernel 的 DSL 写法单独放在 `kernels/`：

- [GEMM](kernels/gemm.md)
- [Stable softmax](kernels/softmax.md)
- [FlashAttention forward](kernels/attention.md)
- [Host-visible two-pass reduction](kernels/reduction.md)
- [Ragged MoE expert kernel](kernels/moe.md)

## 文档分工

```text
doc/
├── dsl/       source language 的构造与语义
├── compiler/  编译器模块、IR、Plan、lowering 与产物
└── kernels/   按 kernel 类型组织的 canonical DSL 模板
```

每个概念只有一个权威落点。其他文档只引用该定义，不复制出第二套规则。
