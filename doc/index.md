# Intent Kernel DSL 设计文档

这组文档是 Intent Kernel DSL 当前唯一的语言与编译器设计规范。它描述最终语义和模块边界，不记录实现进度、历史版本、测试清单或迁移过程。

Intent 是一门 **Python-hosted、单-kernel、跨后端、tile-parametric 的 Structured Tensor-Flow DSL**：用户写完整的 kernel 内算法，编译器补全不可由 source 观察的机器 realization。

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
