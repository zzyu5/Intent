# 编译基础独立调查

## 目的与交付

本能力提供对当前 IntentDSL 语言表面和编译结构的独立调查结果，为恢复 cuTile 以及后续 TileLang、其他硬件工作提供可据以决策的事实。交付是一份调查报告与对应 Comet 验收结论，不是语言或编译器设计的替代规格，也不是生产代码修改。

调查基于已提交的 `a11b80a`。旧 cuTile change 停止执行并保留现场，其未提交修改与新调查基线明确区分。后续若比较其他提交或运行环境，必须说明该事实的适用范围。

## 语言表面与抽象

调查覆盖 contract family、reduce/scan、region fold/scan 及 helper 调用中的命名、参数结构、可发现性和使用负担。结论必须连接真实调用、frontend 行为与 canonical 语义，分别说明：已有设计为何成立、哪些表达可以局部简化、是否存在语义实现偏差、是否存在足以挑战核心抽象的反例。

surface 改善不得被默认升级成 canonical 重构。不同 transition/state schema、轴关系、effects、数值契约所要求的复杂性不能因写法冗长而删除。建议只作为后续选择，本 change 不实施 API 变化。

## 编译结构、权威与实际能力

调查还原 production 的 DSL → canonical KIR → executable GPU Program → provider program/source → 外部 compiler → runtime 路径，说明 construction、mapping、blocking、structured realization、关系维护、合法性检查与 serialization 的实际位置。

真实编译能力以当前 IR 中新增或改变的 control、ownership、fragment、access、accumulator、resource 和 parameter binding 为依据。仅有 pass 名称、注释、类型名称或最终源代码不足以证明能力。重复修补、隐含前置条件和职责交叉应检查其具体影响，不能直接等同错误。

provider-local grid、storage、copy、layout-related source form、precision 和 tuning 决定应对照真实 Triton/TileLang compiler 实现判断。不同编译层的输入抽象高度必须明确，不能把外部机器 lowering 的职责强行归入 shared，也不能把 shared 缺失事实留给 serializer 猜测。

## Config 组织与性能归因

调查区分参数 schema 与合法域、共享相关候选、provider 选项、candidate instantiation 和 runtime winner，说明它们的 producer、consumer、验证位置及跨 provider 影响。

维护性选择应具体比较内嵌 C++ 数值表、独立编译期声明表、编译调用读取的数据表等形式。每种形式必须说明易调部分、仍由代码负责的语义/合法性判断、是否需要重编译和增加的复杂度，不预设外置配置必然更优。

性能证据只在算法、数值契约、ABI、输入、调用次数、候选契约和计时范围清楚时支持比较。应分别说明 physical transformations、provider forms/config 与外部编译器带来的实际变化。未经受控比较的百分比、不同提交的混合数字、仅输出 dtype 相同的比较，不得作为确定归因；不能单独合法运行的必要 lowering 也不构造伪消融结果。

## 后续工作判断

报告明确划分已成立基础、真实阻塞、局部维护机会、证据不足和需要用户决定的核心问题，并说明对恢复 cuTile、接入 TileLang 及其他硬件的实际影响。

新硬件的判断区分 GPU provider/hardware 与 CPU/RVV 等 execution family，不要求把所有硬件塞入同一 GPU topology。本调查不实现新后端，不以缩小语言、禁止成熟 provider 能力或增加任意门禁获得“干净”结论。

## 验收与范围边界

验收唯一采用 brief 的 A1—A4 四项结果，不重复派生内部线索、单个文件、命令或实现步骤为验收项。报告中的关键结论应给出 current/ref 对应位置及后果，运行事实和未知项如实区分。

本 change 不修改生产代码或 `doc/`，不继续旧 cuTile 性能目标，不建设测试、兼容或长期 benchmark 框架。必要动态核验仅使用现有 production DSL emit/JIT/数值 repro，临时文件保存在工作区外。
