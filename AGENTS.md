# 协作规则

## 唯一目标
推进骨架。判据:本轮改动是否让某个语言构造/IR 节点
更接近"能 lowering 出后端代码"?不是,就不要做。

## 规格权威
`doc/` 是最终设计规格,实现必须向它收敛。动 DSL、canonical KIR、compiler IR/pass、
provider lowering 或 runtime 前,先从 `doc/index.md` 进入并完整阅读对应章节。

- `doc/programming-model/` 定义作者、kernel、host 与 compiler 的语义边界。
- `doc/dsl/` 定义 public surface 与唯一 canonical semantics。
- `doc/compiler/` 定义 KIR 之后的 executable physical programs、passes、target
  extensions 与外部 provider compiler 边界。
- `report/` 只记录讨论、现状和验证事实,不是规格权威。

当前实现、旧 IR、examples、历史报告或任一 target API 都不能反向定义规格。代码与 `doc/`
不一致时,按规格迁移代码并删除旧 executable path;不得为了保住现状而把 `doc/` 改成当前行为。
只有用户明确要求修改设计时才改 `doc/`;普通实现推进不把进度、失败、性能数字或临时字段同步进去。

## 目录结构纪律
目录结构就是架构,内部层级和顶层目录同样重要。动手前先从整体结构判断
文件归属,不能只找一个能放的位置。

- 同一层保持一致的抽象,用稳定的职责和模块边界组织文件。
- 模块应拆到职责清楚,但不为单次任务随手造层级;不要把独立模块平铺堆在一起。
- 强耦合、共同演进的文件应相邻;共享内容放在职责明确的共同边界。
- `source/` 按语言、上游来源和算子职责形成可读层级,源码与对应 runtime 相邻。
- 正式编译器也依靠目录表达阶段、模块、依赖和 lowering 边界。
- 落点或边界不明确时停下来问我;缓存、环境和临时产物不得进入项目。

## 卡住时
遇到设计歧义、信息缺失、方案分叉 —— 停下来问我。
禁止用"造一个可交付物"来填补不确定性。
宁可一轮只给出一个问题,也不要给我一堆自洽但没用的产物。

## 验证
唯一允许的验证:一条可手动执行的 repro 命令,
把 DSL 例子 emit 成后端代码并实际跑一次对数值。
不建 test 目录,不用 pytest,不留 fixture。

## 禁止
- 任何 hash / SHA / checksum 校验来源或产物
- 上述 repro 之外的一切测试:单测、边界测试、版本兼容测试、脚手架
- 版本号、CHANGELOG、迁移指南、deprecation 标记 —— 有 git 就够了
- 未经要求的重构、目录整理、注释批量补写、README 更新
- 兜底代码:try/except 吞异常、默认值兜底、"防御性"分支
  未实现就直接 raise NotImplementedError,不要假装能跑

## 交付形式
- 只改必要文件,不新建计划文档/进度文档
- 回复结构:改了什么(一句) → 关键设计取舍 → 卡住的地方
- 不要总结你干了什么,不要罗列"下一步建议"
