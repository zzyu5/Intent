# Outcome

收尾前两轮 DSL/compiler 调查与实现留下的明确问题，使作者表面、canonical semantics、shared physical lowering 和 provider 数值契约在本轮涉及的语义类上形成完整可执行路径，为后续恢复 cuTile 后端工作提供可靠基础。不能再把 change 已归档、API 名字存在或相似程序通过解释为所有已知缺口关闭。

用户原始请求：`$comet 建立这轮的change吧，收尾这些问题，成为成熟compiler，为我们做cutile后端做好准备`。

本轮的“成熟”使用下述可观察完成条件，不承诺任意 kernel、任意 target 或任意性能目标均已完成。旧 `cutile-provider-closure-round-six` 继续暂停。

# Scope

## 当前事实与来源

基线为调查与上一轮实现均已合入的 `main`（`beeba27`）。本轮之前的只读复核完整阅读了 `report/compiler-foundations-reassessment.md`、两轮已归档 brief/Spec 和相关现行 `doc/`，并核对 current/ref 与原始 production repro。历史报告保留其当时含义，不改写成当前实现进度。

本轮需求来自用户对该次复核中未完成事项的收尾要求。下表区分明确要修的项、需要保持的成果和背景；不会把原调查中所有未知硬件或性能事项自动变成本轮目标。

## Source coverage

| 来源单元 | 读取/核对 | 本轮语义与分类 | 完整目标 Spec | 验收 | 状态 |
|---|---|---|---|---|---|
| 用户收尾请求；报告 §1、§2.6、§7.2 的具名作者入口方向 | complete | 完成作者表面迁移与编译路径，而非只改名字或模板 | 作者表面；统一 lowering | A1、A2、A3 | covered |
| 复核发现 examples/kernels 102 处 contract、doc/dsl/examples 10 处；6 处固定加法 scan | complete | 对完整作者目录逐调用判断，迁移普通具名计算；保留真正需要一般配轴/自定义 combine 的用法。数量是基线观察，不是零残留门槛 | 作者表面 | A1 | covered |
| 报告 §2.3、§8 的 multi_reduce；本次复核仍在 shared realize-contractions 报 physical extent 冲突 | complete | 关闭原始多 reduction-axis 反例及其 typed logical-to-physical 关系缺口，不在作者端强迫 flatten 绕过 | 多轴 contraction | A2 | covered |
| 报告 §2.3、§8 的 paired_batch；本次复核仍报 pointwise store validity 无法形成 | complete | 关闭原始固定 shape/direct paired-batch 反例，保留 source/axis/ownership/validity；动态 named batch 成功不替代该项 | 直接 batch contraction | A3 | covered |
| 报告 §2.5、§7.1；上一轮完整目标 Spec 的退化 region 归一项；现行 doc/dsl/core.md §9 | complete | 落实 element-summary fold 与 element-summary/output scan 的 canonicalization；保留真实 slice-level tensor algorithm | 退化 region 归一 | A4、A5 | covered |
| 报告 §4.1 的 TileLang f32 同类风险；本次 production 实测规范值 1.00146484375、实际 1.0009765625 | complete | 该风险已成为当前路径的实测数值错误；修复 full-f32 契约，不仅沿用 Triton 的成功记录 | Provider 数值契约 | A6 | covered |
| 复核确认已完成的具名接口、typed helper、builtin identity、dot/GEMV、四类既有语义修正、groups/JSON/occupancy 分离 | complete | 作为本轮必须保持的已有行为；与本轮改动相交的路径用 production repro 核验 | 已有能力与职责 | A7 | covered |
| 上一轮归档页全部 passed/未报告风险与交接 known_limits 的差异 | complete | 各项必须保留独立结论、原始失败的去向和证据适用范围，不能静默移入 known_limits | 验收与交付 | A1-A7 | covered |
| 报告 §6、§7 的性能归因、旧 cuTile 追平、所有 provider/硬件覆盖、标准 pass instrumentation | complete | 保留原调查边界；不自动恢复旧任务、不新增全量性能或通用 instrumentation 工程 | 非目标 | 无 | background/non-goal |

## 实现范围

- 完整检查 `examples/kernels/` 与 `doc/dsl/examples/` 的 common contraction/prefix 写法。普通 dot/matvec/vecmat/matmul、transpose/batch 矩阵计算、固定 builtin prefix 使用已有具名接口；不能只迁移 f16 版本而漏下语义同类的 bf16/helper/variant。真正一般配轴、自定义 summary 和严格 recurrence 保留原构造。
- 多轴 contraction 和直接 paired-batch 修复发生在其通用 typed relation、shape/axis 变换、ownership、blocking、access/validity 或 realization 所属层。公开具名入口与 generic 入口消费同一 canonical 语义；不增加按 API 名字分流的后端路径。
- 退化 region 在 canonical 路径归一回 ordinary reduce/scan。判断基于 typed helper body、combine、source/output relation 和 state action，不能依赖 helper 名字、Python 源码模板或数学猜测。
- TileLang 的 f32 contraction 在实际 provider lowering 中兑现现行 full-f32 输入语义。原始单非零乘积必须正确；不能仅改变容差、benchmark reference、默认精度或把新 unsupported 诊断当修复完成。
- 保留已成立的 transformation groups、postconditions、有限 JSON 数据/legality 分离与调优 effect 隔离；本轮只在实际影响的职责边界内补齐实现，不重新设计整个 IR。

# Non-goals

- 不恢复旧 cuTile 性能任务，不应用或删除 `paused-cutile-round-six-before-foundations-merge` stash。
- 不推翻 domain/KIR/GPU Program，不删除 generic contract/reduce/scan/region，不为公开名字复制 backend 或 whole-kernel 模板。
- 不增加新硬件后端，不要求所有 examples 在所有 provider 全量通过，不引入性能比例阈值或承诺性能归因百分比。
- 不以缩窄语言、放宽数值/effects、改算法或发射串行伪支持完成收尾。必要公开语义分叉先询问用户。
- 不创建 test 目录、pytest、fixture、测试台账、兼容层、版本/迁移体系或额外计划/进度文档；不改历史报告和 CSV 伪造当前完成状态。
- Shape 不修改生产实现或运行新增 repro。Git 环境升级属于创建工作区前置修复，不属于 compiler capability，也不写入 `doc/`。

# Acceptance examples

- A1：作者目录与规格示例中的普通具名 contraction/prefix 写法迁移完整，不能留下没有语义理由的旧轴对或固定 builtin combine/identity；剩余 generic 调用具有明确的一般配轴、自定义组合或顺序语义理由。迁移保持算法、dtype、结果轴序、ABI、读写和调用次数，并能通过受影响现有 kernel 的 production emit/运行体现。
- A2：原始 `multi_reduce`（`[32,2,32] × [2,32,32]`、`reduce=((1,0),(2,1))`）保留 generic 作者表达，经当前 compiler 生成可执行后端程序并数值正确；实现依据 typed paired-axis/provenance 形成等价 physical program，不依赖该组常量、函数名或作者预先 flatten。
- A3：原始 `paired_batch`（`[2,32,64] × [2,64,32]`、`reduce=((2,1),)`、`batch=((0,0),)`）原样生成可执行后端程序并数值正确；固定/dynamic extent identity、batch/free/reduction 和 store validity 在统一 lowering 中成立，既有 named batch 路径不退化。
- A4：仅以同一 pure combine/identity 折叠 slice elements 的退化 region_fold 在 canonical 输出中成为 ordinary reduce，保留 source-order、schema、captures、empty/identity 与数值契约并实际生成/运行正确；真正含 slice-level tensor algorithm 的 region_fold 仍保留 region 语义。
- A5：element-summary/element-output 的退化 region_scan 在等价条件可验证时归一为 ordinary scan及必要的显式 state 应用，保留 prefix/output 与 final state；不能丢失 initial state、apply/emit 或 source relation。相应程序实际生成/运行正确，非退化 region_scan 不被误抹除。
- A6：当前 TileLang f32 contraction 的原始单非零乘积例经 production emit/JIT/launch 得到 1.00146484375，而非 1.0009765625；lowering 对所覆盖 f32 语义类保留输入精度和显式 accumulator 契约，不借另一 public 名称、精度放宽、容差变化或特定输入匹配通过。
- A7：本轮触及路径保持此前具名接口/typed helper、builtin identity、dot/GEMV、Triton f32 与 signed floor/rem、cuTile signed64 index/scalar ABI、三 provider autotune effect/alias 隔离，以及 shared groups/postconditions/JSON/occupancy 的职责分离。验收逐项说明实际核对与运行范围，旧成功记录不能替代受影响路径当前候选的证据。

# Constraints and invariants

- `doc/` 为语义与架构权威；作者决定算法，compiler 决定语义允许的 physical realization。优先阅读真实 `ref/triton`、`ref/tilelang` 同类实现，给 current/ref 的具体位置、差异和后果。
- 本轮不改变已确认的 named API、batch/transpose、dtype/NaN/empty、captures、ordered effects 和 host-visible invocation 契约。
- generic 与 named 表面等价时进入同一 typed canonical/lowering；不能让 generic 旧反例留着失败而只证明 named 的另一个例子通过。
- 参数与策略仍写入 typed IR。保留合法默认搜索策略，不把 occupancy 默认 1、small-thread family 分类或有限表本身误认成 correctness bug。
- 子代理仅承担探索和独立只读复核；主线程负责方案、修改和最终验证。沿用现有目录职责，不按一次问题增设杂项层级。
- 仅修改当前 change 的正式产物与必要生产文件。旧 cuTile change 的正文、状态、验收和暂存试验保持不动。

# Decisions

- 用户要求新建收尾 change，不继续旧 cuTile change；本轮已确认问题沿用原始反例与 current/ref 事实，不重新做一次无实现交付的泛调查。
- 用户优先沿用干净主工作区，并明确授权必要时创建 worktree。Runtime 拒绝与旧 active change 共用当前目录，因此使用独立 worktree `comet/compiler-foundations-closure`，目标分支为 `main`。
- 维持一个普通 Native Change：作者迁移依赖统一 frontend/shared lowering，region 与 contraction 修复会交叉影响 schema、source relation、realization 和 provider 程序；本轮统一顺序实施与最终验收，不拆分会争用这些边界的子 change。
- 用户在完整范围、关键边界与逐项验收说明后回复“可以”，已确认本文 A1-A7 并授权进入实现。归档、merge/push/PR 与 worktree 清理由 Archive 阶段另按用户授权处理。

# Open questions

无。用户已确认完整范围；新发现的公开语义分叉仍须暂停并澄清。

# Verification expectations

仅使用项目允许、可手动执行的 production DSL -> 后端源码 -> provider JIT/launch -> 数值比较 repro；失败必须保留为失败，脚本退出 0 不等于数值相符。原始 multi/batch/TileLang f32 case 必须在当前候选上各自有结果；不能用动态 shape、新名称、全一输入的另一个例子或旧日志替代。受影响语义类的必要现有算法执行随改动范围选择，不建设新的测试体系。

原始脚本保存在工作区外：`/tmp/intentdsl-foundations-repro.h6uHVZ/contract_forms.py` 和 `numerics.py`；多轴、batch 和 f32 定义也可从旧调查报告 §8 恢复。旧构建仅作为已确认问题的基线证据，实现验收使用本轮源码构建的 compiler，不能拿旧 binary 宣称当前候选通过。

Builder 交接、只读复核与独立 Verifier 均逐项核对 A1-A7 和保留的 generic 原因。未完成项必须标记 failed/blocked，或在用户明确修改范围后更新 Spec；不得仅移入 known_limits 再给出“全部完成”。最终验收页明确区分新运行事实、源码/ref 核对、历史背景和仍不属于本轮的未知。
