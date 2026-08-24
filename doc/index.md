# IntentDSL 规格

本目录只保存稳定设计，不保存实现进度、测试结果、性能数字或历史决策过程。

当前规格分为两层：

- [`programming-model/`](programming-model/)：定义作者、kernel、host 与编译器之间的语义边界；
- [`dsl/`](dsl/)：定义作者可使用的语言表面、Core 构造及其理想化示例。

编译器 IR、pass、target lowering 和运行时合同将在编程模型与 DSL 通过审查后另行定义。现有实现和历史报告都不能反向修改这里的语言语义。
