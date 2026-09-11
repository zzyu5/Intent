# IntentDSL 协作原则

## 目标与责任

- 推进可运行的 lowering 骨架与真实算子性能；问答、调查、清理按用户指定范围完成，不用测试、报告或管理流程冒充进展。
- Intent 是可编程算子 DSL，不默认采用“高层算子图交给 compiler 决定全部算法”的模型。作者负责算法与显式程序组织，包括多个 kernels 和 host 编排；compiler 在既有语义下形成物理程序并优化。
- 合法的类型、shape、数值与 effect 表达 lowering 失败，是 compiler 问题；不要求作者换算法或反复拆 kernel 迁就实现缺陷。
- 明确语义与适用条件下，允许 lowering 到高性能 micro-kernel/目标构造，不必由通用 IR 重建全部底层细节；也不能让 leaf 猜测缺失语义或接管整个算子。

## 规格与参考

- `doc/` 是最终设计规格。动 DSL、KIR、compiler IR/pass、provider 或 runtime 前，从 `doc/index.md` 进入并完整阅读相关章节；实现向规格收敛，不为保住现状反改规格。
- 只有用户确认设计变化才改规格；进度、实验失败、临时字段和性能数字不写入 `doc/`。Memory、短期记录和 report 只能帮助定位，不能替代当前原文。
- 架构、pass 与性能决策先对照 `ref/triton`、`ref/tilelang` 或相应成熟目标实现：确认同类职责由哪层承担，关键结论给出双方 file:line、具体差异与后果。参考边界与机制，不照搬 surface，不凭框架名称作判断。
- `source/` 是 provider source/runtime corpus，`examples/kernels/` 是作者算法，registry 连接完整 callable，CSV 是运行观察；它们不定义语言语义或 compiler policy。

## IR、pass 与目录

- 分块、布局、ownership、遍历、复用、materialization 等执行决定必须存在于相应 IR 与 passes；shared 和 target-local 各守职责，serializer 只拼写已决定的结构。
- Policy 根据 current typed semantics、def-use、coordinate relations、effects、lifetime 与目标能力决策，不靠算子名、字符串标签或单条样本。正式 typed 参数、shape 与 capability 可以参与合法性判断。
- 复用已有 carrier；只有真正缺少执行事实才增加 IR 表达。目标 API 的拼写差异不自动要求修改 shared IR。
- 只保留一条有效执行路径。迁移后删除旧路径，不保留兼容开关、默认值或异常回退掩盖未实现。
- 目录表达稳定职责与 lowering 边界；同层同抽象，强耦合文件相邻，source 与 runtime 相邻。不随手造层级、平铺模块或放入缓存和环境。
- 可查清的信息自行调查；只有现有规格无法决定的语言语义、作者可观察行为或核心架构分叉才暂停问用户，不能用另造产物填补不确定性。

## Benchmark 与实验

- 运行只复用必要的生产算子 benchmark，同次做一次既定容差检查；容差内即可，不追求 bitwise 一致，不为通过放宽容差。
- 不额外设计独立数值、边界、回归、兼容、压力或组合测试；不建 test 目录，不用 pytest，不留 fixture。临时脚本、内联命令和跑完即删也不是例外。
- 全量指现有 registry 的性能运行，不扩矩阵。准备、编译与必要运行可按资源预算并发，同机性能计时避免干扰，不让所有工作全程串行。
- 同算法、相同输入规模与外部 dtype 下比较；ABI、辅助输出、布局转换与精度细节注明，不一概阻断计时。Compiler 仍必须保持 Intent 语义。
- 报真实完整算子时间、reference 时间及清楚的比值；编译/JIT/tuning 不是算子耗时，CSV 不承担历史审计。差距先查物理结构，再查 provider/外部 compiler/measurement，不无证据归因给下层。
- Agent 实验每题每语言组只交付一次完整程序，提交后不反馈错误或性能继续生成。Bench 自带 reference 保留；开发端修 compiler 不改写首次交付成绩。

## 禁止与交付

- 禁止任何 hash / SHA / checksum 校验来源或产物。
- 不增加版本号、CHANGELOG、迁移指南、deprecation 标记；用 Git 记录修改。
- 不做未经要求的重构、目录整理、注释批量补写或 README 更新。
- 不吞异常、不加默认值兜底或“防御性”假支持；未实现明确报 unsupported / NotImplementedError。
- 只改必要文件，除用户指定或 todoskill 的单份短期记录外，不新建计划/进度文档。完成一个连贯改动后提交自己的文件，保留他人修改，不自动 push。
- 简明报告实际结果、关键取舍和未完成项；未经运行不宣称数值正确或性能达标。
