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

Comet 个人记忆和项目知识只帮助定位，应用前以当前 `AGENTS.md`、`doc/` 与所选 change
的 brief/spec 为准。归档 change 的验收门槛、暂停指令和运行结论属于历史上下文，
不得自动作为新 change 的约束；项目知识摘要不能代替当前原文。

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
自查必须对照 ref/triton 或 ref/tilelang 的同类实现，给出双方 file:line、具体差异与实际后果；找不出具体差异等于未完成。

日常实现验证使用一条可手动执行的 repro 命令,
把 DSL 例子 emit 成后端代码并实际跑一次对数值。用户明确要求全量时，
使用同一现有生产路径覆盖 registry；允许资源预算内的同机多进程并发，
功能运行与性能计时分阶段安排，不让每个 worker 全程串行等待。
不建 test 目录,不用 pytest,不留 fixture。

## Baseline 与性能调查
`source/` 是 provider source/runtime 参考 corpus，`examples/kernels/` 是 Intent 作者算法，
registry 只连接 runtime-visible entry 与完整 callable closure，CSV 只是一组运行观察；
它们都不能定义语言语义或 compiler policy。

比较 generated 与 source 的算子性能以同算法为前提，保持输入 shape、外部 dtype
及明确的调用和计时范围；ABI 表示、辅助输出与布局转换差异如实注明。
舍入、近似数学和中间精度的细微差异不一概阻断计时，不要求先对齐完整候选集合。
CSV 记录真实算子时间、source 时间、ratio 和必要失败说明，不承担运行历史审计，
候选调优耗时不是算子性能结果；这些比较口径不放松 compiler 对 Intent 语义的保持要求。
Baseline 用来暴露 compiler 缺口和验证改动归因，不能反向驱动 DSL、kernel-name matcher、
source template 或只对单条语料成立的规则。性能差距先从 current Physical Program 的 mapping、
blocking、ownership、traversal 和 materialization 调查，再看 provider-local form、serializer、
外部 compiler 与 measurement；没有证据时不能把差距归因给下层。

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

<comet-ambient-resume>
<!-- Managed by Comet. Edits inside this block may be replaced by comet init/update. -->
<!-- Contract: comet.resume_probe.v2 -->

## Comet Ambient Resume

在这个仓库中，开始处理需要改动或调查的任务前，如果可能存在活跃 Comet workflow，把当前用户请求传入只读探针：`comet resume-probe . --stdin --json`。

- 如果用户通过宿主明确调用任意 Comet Skill（例如 `@comet`、`/comet`、`@comet-native` 或 `/comet-hotfix`），显式调用优先于本恢复协议；不要运行 resume probe，直接进入被调用的 Skill。
- 如果用户通过宿主明确调用的是非 Comet 的 Skill 或斜杠命令，任务意图已由该调用明确：不要运行 resume probe，直接执行该 Skill。
- 如果你正在 Comet 流程内（包括正在等待用户回复你在流程中提出的问题），不要运行 resume probe；把这类回复（例如方案/选项选择）当作当前 change 的继续，直接按用户的选择推进。
- 只信任返回的 `workflow`、`skill` 和 `entrySource`；它们只由项目配置或无配置兼容回退决定。不得扫描或切换另一套 workflow。
- 如果 probe 返回 `auto_resume`，简短说明选中的 active change，并进入 `nextCommand` 指向的永久入口。不要把状态命令当作恢复入口直接推进。
- 如果 probe 返回 `ask_user`，只问一个简短问题并等待用户回复。
- 如果当前请求未明确调用 Comet Skill，且 probe 返回 `out_of_scope` 或 `none`，不要进入 Comet workflow。
- `out_of_scope` 或 `none` 只表示不要因为这个新请求进入 Comet workflow；它绝不表示要暂停或退出一个已在进行的 Comet 流程。
- 如果配置或状态无效且没有 `nextCommand`，停止并报告原因；不要猜测另一个 workflow。
- 不能只因为存在 active change 就把无关任务挂到该 change。Native 的未提交改动由 Native 入口检查，不由探针自动归因。
</comet-ambient-resume>
