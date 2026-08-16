# Intent Kernel DSL 设计文档

这组文档描述 Intent Kernel DSL 的语言语义、编译器模块边界与稳定使用方法，不记录实现进度、历史版本、测试清单、失败状态或迁移过程。运行与实现状态只进入 `report/`；设计文档只在明确修改规格时更新，环境文档只维护可复现的依赖与构建合同。

Intent 是一门 **Python-hosted、跨后端、region-parametric 的结构化算子 kernel DSL**：用户写一个 logical callable 内的完整算法，编译器补全不可由 source 观察的机器 realization；一个 callable 的目标实现可以包含多个 compiler-private stages。

## 设计变更门槛

语言构造和表示层不能作为普通实现补丁扩张：目录、字段或 target API 可以演进，但不能制造第二份算法真理、改变 source algorithm，或把某个 target 的抽象抬进共享层。

新增语言构造必须同时满足两条：

1. 一个真实算法无法用现有 Core 表达，且问题不是缺少语法糖、library helper 或作者显式的多-kernel wrapper；
2. 所需能力不能机械委托给下层 target compiler 已有的原语或接口。

新增物理决定也必须同时满足：其正确取值依赖 Kernel IR 保留的算法结构；该值只有在具体机器上才有意义，作者没有写也不应写。否则需求应当被拒绝、作为 library policy 表达、成为 target capability subset，或交给下层 compiler/tuner，而不是扩展语言或 Plan。

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
12. [环境、依赖与构建](setup/environment.md)

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
├── kernels/   按 kernel 类型组织的 canonical DSL 模板
└── setup/     可复现环境、依赖与构建入口
```

每个概念只有一个权威落点。其他文档只引用该定义，不复制出第二套规则。
