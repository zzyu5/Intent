# Outcome

对 IntentDSL 当前语言表面和编译实现重新进行独立调查，说明它实际承担的算子 kernel 编译工作、仍然薄弱的边界以及后续 cuTile、TileLang 和其他硬件接入的真实阻塞。结论应使维护者能够判断哪些设计应保留、哪些问题只需局部修正、哪些重大问题才值得重新讨论抽象。调查不预设实现正确，也不预设需要重构。

# Scope

- 以已提交的 `a11b80a` 为共同调查基线；暂停 `cutile-provider-closure-round-six` 的执行，在独立 worktree 开展普通调查 change。原工作区的两个未提交试验只作为待核实线索，不作为已完成修复。
- 审查 public DSL 的命名、调用结构、参数冗余、可发现性和语义表达，重点包括 contract family、reduce/scan、region fold/scan 和 helper 调用。区分 surface ergonomics、frontend 实现偏差与真正的抽象问题；不因个人审美直接提出核心改写。
- 还原真实 production 编译路径，解释 canonical KIR、GPU Program、construction、各 shared transformation、provider legalization、serialization、runtime 和外部 compiler 的输入、输出与职责；调查 pass 的前置条件、类型/关系维护和验证边界。
- 调查实际性能来源：shared mapping/blocking/traversal/materialization、provider form、candidate/config 与外部 compiler 各自改变了什么。核对已有比较的算法、数值契约、ABI、输入、调用次数和计时范围；只有受控事实支持时才给出定量归因。
- 审查 05c 之后 config 的真实分离程度，区分合法域、相关候选表、provider 选项和 runtime winner；评估把易调数值表与 C++ 逻辑分离的低成本组织方式，并明确是否仍需重编译及其跨 provider 影响。
- 对照 `/home/kingdom/phdworks/ref/triton` 与 `/home/kingdom/phdworks/ref/tilelang` 的真实编译器实现。provider corpus、安装包和历史报告用于补充现状，不替代编译器对照或语言规格。
- 以一份 `report/compiler-foundations-reassessment.md` 汇总调查结果；必要的代码位置、IR 片段和已有/新运行事实直接放在该报告中，不另建计划、进度或证据文件。

# Non-goals

- 本 change 不继续 cuTile 性能修复，不启动 TileLang 或新硬件后端实现，不迁移、重命名或重构生产代码。
- 不修改 `doc/` 的语言和架构设计。发现不可成立的核心假设时说明反例、影响和最小选择，留待用户决定。
- 不把本次具名 public op 的设计结论当作已经实施的 API 变更，不预设每个公开名字都要独立 canonical op；不预设所有策略必须外置文件，或所有 provider 决策都应上移 shared。
- 不以固定候选表、缺少 cost model、使用源码作为中间产物、存在 provider pass 本身判定设计失败；也不以已通过某批 kernel 判定全部语义已闭合。
- 不建设 pytest、fixture、benchmark/ablation 框架、兼容层或版本管理；不把双机全量达成 1.05 作为本调查的验收。

# Acceptance examples

- A1：维护者能够从真实 DSL 调用与对应 canonical 语义判断语言表面的合理性，获得有具体理由的保留项和局部改善候选；命名偏好、实现偏差与抽象缺陷有明确区分，重大抽象问题只在有可核验反例时提出。
- A2：维护者能够沿一条真实 kernel 编译过程理解各层和关键 pass 如何形成 executable GPU Program，并据实际 IR/代码变化判断 Intent 已承担的编译能力与尚未闭合的职责；shared、provider 和 serializer 的判断经过同类参考实现对照。
- A3：维护者能够区分性能变化中的 physical transformation、provider form/config 和外部编译贡献，知道当前哪些数字可比较、哪些归因尚无证据；同时获得 config 数据与策略分离的具体可维护性选择及其行为、重编译和跨 provider 影响。
- A4：后续 cuTile、TileLang 与其他硬件工作获得一份按实际影响整理的继续条件：已成立的基础、需要局部修复的阻塞、可延后的维护问题和必须讨论的核心分叉分别明确；结论保留未知项，不依赖新增限制或未经验证的“全部干净”承诺。

# Constraints and invariants

- `doc/` 仍是现行规格；真实 ref 用于核对成熟编译器如何处理同类问题。二者有张力时明确指出，而不是修改定义来迁就实现或凭空增加限制。
- 具体缺陷必须给出 current/ref 的对应位置、差异及实际后果；合理差异、可选优化和证据不足不能包装成 correctness bug。
- 调查对象是结构化算子 kernel 编译器，不将图级算法选择、隐藏多次 launch 或完整机器后端视为本项目必需能力。GPU provider 与 hardware target、CPU/RVV execution family 分别讨论。
- 不承诺任意 kernel 的 pass 性能贡献可以相加或从最终 ratio 分解；必要 lowering 无合法缺省程序时，不制造伪 ablation。
- 子代理只负责独立探索和只读核验；主代理亲自读取基础规格、点验关键结论，并统一作出判断。

# Decisions

- 用户已要求暂停 cuTile change，开启并行的调查 change；采用独立 worktree，保留旧工作区和未提交修改。
- 使用一个普通 Native Change。几个调查方向共同回答同一套语言与编译边界问题，探索可并行，最终结论需要统一，因此不拆 Supervisor。
- 先形成事实和有界建议，不自动实施语法调整、config 外置或核心重构。
- 用户特别要求避免过度收缩；参考编译器中的合法 provider 决策和可调 policy 必须作为对照，不能仅凭层名、字符串、shape 查询或 pass 数量判越界。
- 用户已确认本 Shape 的调查范围、四项结果型验收和非目标，授权进入调查 Build。
- 用户复核指出 A1 的作者体验调查过窄：不能把 contract 限定为 GEMM/matmul 命名问题；补全内积、GEMV、batch、多轴收缩、outer 及 reduce/scan/region 的实际使用与能力边界，在原验收范围内修订报告，不改生产实现。
- 用户进一步确认：公开 DSL 应以符合作者习惯的具名计算 op 为主要入口；语义等价时映射到现有统一轴/contract/reduce/scan 等，不因此删除用于新算法的通用构造。报告必须明确 region 等机制的表达条件、实现覆盖与真正语义扩展的区别，并给出下一轮行动依据。具体 API 契约及生产实现留待下一轮，不改变本 change 的 A1—A4 或调查性质。
- 用户最终同意以实际 kernel 为依据的结论：保留能承载复杂算法的可编程骨架，完善常用计算词汇与普通函数复用；作者决定算法，编译器决定语义允许的物理组织。最终落稿补足已有成功路径，并准确限定局部 provider/form 失败，不再复跑或新增验证记录，准备收尾本轮调查。

# Open questions

无。

# Verification expectations

最终落稿由主代理与独立只读复核核对四项结果、已有 kernel/CSV/调用链和 current/ref；不再启动运行、追加测试台账或大规模扫描。前期已有 production repro 只保留其具体适用范围，不能泛化为整门语言或其它 provider 的能力结论；未经支持的性能归因保持未知。
