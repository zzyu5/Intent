# IntentDSL 规格

本目录只保存稳定设计，不保存实现进度、测试结果、性能数字或历史决策过程。

当前规格分为两层：

- [`programming-model/`](programming-model/)：定义作者、kernel、host 与编译器之间的语义边界；
- [`dsl/`](dsl/)：定义作者可使用的语言表面、canonical semantics、surface shorthand 及其理想化示例。

target 由编译调用在 DSL 之外选择，不是 source value、constexpr、类型或控制条件。同一份 source 与 canonical Kernel IR 可以被编译到不同 target；target capability 只决定 lowering 是否成立，不改变这里定义的逻辑结果、控制、effects 和接口语义。

编译器 IR、pass、target lowering 和运行时语义在编程模型与 DSL 之后另行定义。现有实现、历史报告和任一 target API 都不能反向修改这里的语言语义。
