# 扩展 Pass

Pass 是优化知识的入口。它读取当前 typed semantics 与执行关系，改写真实 IR，再把完整程序交给后续 consumer。新增优化前，先阅读[Pass 与分析](../compiler/passes-and-analyses.md)、对应 [GPU](../compiler/gpu-program-ir.md)/[CPU](../compiler/cpu-program-ir.md) 程序合同和[目标 lowering](../compiler/target-lowering.md)。

## 找到负责层

| 改动 | 负责位置 |
|---|---|
| domain、dtype、effects 与作者算法的语义 | DSL / canonical KIR；设计变化需要单独讨论 |
| def-use、coordinate、alias、dependence、working-set 查询 | 对应 dialect 的 `Analysis/` |
| ownership、遍历、blocking、reuse、materialization | GPU/CPU/DSA 的 `Transforms/` 职责组 |
| 目标能力、特定实现、local forms 与 legality | `lib/Target/<Provider>/` |
| 源码 API 拼写 | 目标 serialization |
| ABI 参数绑定、工具链启动、调优调用 | `python/intent/runtime/<provider>/` |

GPU 与 CPU 可以复用分析思想，但不要求共享一套 physical topology。一个 pass 不必在所有后端运行；共享事实、独立决策、各自目标 consumer 是有效复用。

## 一个完整优化组件

1. 从当前 typed operands、regions、def-use、access relation、effects、lifetime 与 capability 定义合法性，不能以 kernel 名称、操作数量或字符串标签触发。
2. 用 analysis 提供可重算的事实。Unknown 保持 unknown，不能解释为 replay、alias 或初始化许可。
3. 实际改写 operations、types、regions、coordinates、carry 或 resource lifetime；不另造共同解释执行的 schema/plan。
4. 保持 logical members、dtype、数值许可、ordered control、effects、ABI 和作者 kernel 数量。
5. 声明失效的 analysis，在完整 transformation group 后检查后置条件。Verifier 不修程序，serializer 不补 loops、workspace 或参数。

Pass 声明、注册与 pipeline 使用 MLIR 的既有机制。先查看 `include/Intent/Dialect/<Family>/Transforms/Passes.td`、`lib/Dialect/<Family>/Transforms/` 和对应的 `Passes.cpp`；target-local transformations 从 `lib/Target/<Provider>/Transforms/` 进入。实际可调用 pass 及选项可通过 `intent-opt --help` 查阅。

只改完整经验配置时不创建新 pass。Provider 已负责的 collective tree、线程通信、distributed layout 和 machine pipeline，直接提供合法的输入，不在 Intent 重建。

## 使用已有程序观察

选择受影响的现有完整 program，检查前后 IR 结构、生成源码和实际调用结果，再观察同算法同输入的热调用；不要复制算法或扩建测试矩阵。若另一同类 program 受益，可以说明知识的复用范围。生成、native 编译、执行、数值检查与性能改善分别陈述。

仓库的[贡献指南](https://github.com/zzyu5/Intent/blob/main/CONTRIBUTING.zh-CN.md)说明开发入口；[详细编译器参考](compiler-reference.md)保留深入实现说明（中文）。
