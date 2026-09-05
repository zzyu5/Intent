# Outcome

将上一轮编译基础调查的结论落实为作者可直接使用、内部职责清楚的 DSL/compiler 实现。保留已有复杂 kernel 的可编程骨架，以具名计算 op 和 typed 函数复用改善作者表达；现有统一语义继续承接算法组合，编译器形成语义允许的物理程序。实现完成后归档、本地合并到 `main`，清理本 change 的干净 worktree/分支，不推送或创建 PR。

用户已确认下文三块完整范围、公开 API 与 JSON 配置契约，并明确授权进入实现。

# Scope

## 需求来源

唯一报告来源：`report/compiler-foundations-reassessment.md`，已完整阅读其正文、表格、代码片段、引用与范围说明。上一轮已归档在 `docs/comet/archive/2026-09-05-compiler-foundations-reassessment/` 并合入 `main`。报告是本 change 的需求来源和事实依据，不取代 `doc/`；涉及公开设计的确定变化先写入对应 `doc/`，再实现。

已确认的完整范围：

- 作者接口：为常用收缩/prefix 计算提供符合使用习惯的具名 op，统一公开签名、frontend binding 和诊断；复用已有规范化语义，不按每个公开名字复制 backend path。
- 算法复用：在已有算法职责边界中完善 typed helper 的正常复用，保留 generic contract、自定义 reduce/scan、region、普通 control/indexing/effects；不新建封闭的 whole-operator 模板库。
- 必要 lowering：从现有 kernel 与实际调用链出发，补齐选定 public op 需要的 typed form/relation/realization；局部 provider/form 失败不被泛化为算法或语言能力不足。
- 编译器组织：明确已有 transformation groups、顺序依赖和 postconditions；分离 config 有限数据与判断逻辑，并区分 provider legality 与搜索策略。外置为编译期读取的有限 JSON 数据表。
- 已知语义兑现：修复报告明确的 Triton f32 输入精度、signed floor division/remainder、cuTile logical index 宽度与 autotune 可观察 effects 问题；静态风险与已复现问题分别处理，不承诺本轮穷尽所有 provider 的未知行为。

## Source coverage

下表覆盖整份报告；`complete` 表示原文读取完整，`needs-clarification` 表示归属或公开契约尚待 Shape 选择，不表示遗漏原文。实现建议映射到本 change 的完整目标 Spec 章节和验收草案；源码引用、数值、旧运行命令和环境描述作为证据背景，不自动成为新增运行任务。

| 来源单元 | 读取 | 保留语义/分类 | Spec 对应 | 验收草案 | 覆盖状态 |
|---|---|---|---|---|---|
| §1、§1.1，含已有 kernel/CSV 表和引用 | complete | 已有可编程骨架、作者/语言/compiler 分工；成功与失败适用范围分别保留 | 作者接口；算法组合；不变量 | A1、A2 | covered |
| §2.1–2.2，含轴表和调用代码 | complete | 具名计算入口与统一 contraction 表示并存；名称/rank/dtype 契约不从 provider 倒推 | 作者接口与归一 | A1 | covered |
| §2.3，含五种 cuTile 形态与 current/ref | complete | 选定操作所需的实际 lowering 修复；不推广为整类算法不可表达 | 实现覆盖与语义兑现 | A2、A3 | covered |
| §2.4–2.5，含 helper/state/capture 与退化 region 说明 | complete | 保留普通控制和 summary 协议，完善复用及必要 canonicalization | 算法组合与函数复用 | A2 | covered |
| §2.6，含具名 op 映射表 | complete | 完整公开契约、统一 binding、等价映射、有限专用参数简化 | 作者接口与归一 | A1 | covered |
| §2.7，含新算法表与代数式 | complete | 区分作者组合、实现覆盖、真正语义扩展；不宣称任意算法可 region 化 | 算法组合；不变量 | A2 | covered |
| §3.1–3.2，含 pipeline/IR/改写表 | complete | 已有真实链路与实现背景，不另造中间层 | 编译阶段与策略组织 | A4 | background |
| §3.3，含 repair/analysis 生命周期与组织建议 | complete | 明确完整 transformation groups 与维护职责；无证据不作 stale-cache/误绑定修复 | 编译阶段与策略组织 | A4 | covered |
| §4.1–4.4，含反例和同类静态风险 | complete | 明确数值/effect 行为按现行语义兑现；未知保持限定 | 实现覆盖与语义兑现 | A3 | covered |
| §5.1–5.2，含配置组织选项和数据示例 | complete | typed IR 仍是参数权威，有限数据与逻辑分离，不增加策略语言 | 编译阶段与策略组织 | A4 | covered |
| §5.3，含 occupancy 对照 | complete | 区分合法值/结构约束与 mapping/search policy，不无条件扩大搜索 | 编译阶段与策略组织 | A4 | covered |
| §6.1–6.2，含 CSV 数字、closure 表 | complete | 性能与可比性背景，不作为本轮数值阈值或全量追平任务 | 非目标与不变量 | — | background |
| §6.3 | complete | 避免伪归因/新 profiling 框架，按当前物理程序定位实际问题 | 非目标与不变量 | A2、A4 | covered |
| §7.1–7.2，含继续条件和行动边界 | complete | 从现有 kernel 推进 public→canonical→lowering，维护工作按确认范围纳入；归档/合并属于 Archive 完成条件 | 全文；交付与收尾 | A1–A4 | covered |
| §8，含前期脚本、命令、表和完整 DSL | complete | 既有事实与实现参考；不把复跑或新增台账作为 Shape 工作 | 非目标与不变量 | — | background |

## 已确认的公开契约

第一批常用操作按计算职责命名，不让一个 `matmul` 名字承担向量升维和降维分派；也不要求普通作者手写 `reduce`/`batch` 轴对。

| 操作 | 作者输入与结果 | 自动产生的机械部分 |
|---|---|---|
| `dot` | `[K] × [K] → []` | 唯一 reduction pair；不暗含复数共轭 |
| `matvec` | `[..., M, K] × [..., K] → [..., M]` | 矩阵—向量 reduction/batch pairs |
| `vecmat` | `[..., K] × [..., K, N] → [..., N]` | 向量—矩阵 reduction/batch pairs |
| `matmul` | `[..., M, K] × [..., K, N] → [..., M, N]` | 矩阵 reduction/batch pairs |
| `outer` | `[M] × [N] → [M, N]` | size-one axes 和普通 broadcast multiply，无 accumulator |
| `cumsum` / `cummax` | 沿显式 axis 返回同 shape 的 prefix values | builtin combine/identity；默认 inclusive、forward |
| `scaled_matmul` | 现行 `[M,G,C]/[M,G] × [G,C,N]/[N,G] → [M,N]` | 固定轴对与空 batch；同一 `group_size` 不重复传两次 |
| `sparse_matmul` | 现行 logical sparse matrix × dense matrix → matrix | 固定矩阵轴对；保留显式 format/metadata interpretation |

- `matvec/vecmat/matmul` 的前导 batch 维度按现有 size-one broadcasting 规则右对齐；reduction extent 必须相等，不进行 K 轴 broadcast。矩阵末两轴可显式转置；具名操作不推断任意高阶 contraction。
- 乘加类仍显式给出 `acc_dtype`，不混用 provider 的输入精度或 dtype promotion；`outer` 保持普通乘法 dtype。具体签名与数值规则写在完整目标 Spec 中。
- 保留已有 `reduce.sum/max/any/all` 名字，不另造 `sum/max` 同义入口；由 builtin 自动产生 identity。需要自定义 combine/identity 时使用 generic reduce/scan。
- Generic contract、scaled/sparse contract、reduce/scan、region、普通 control 与 `@intent.fn` 均保留。具名操作在 kernel 和允许的 helper 调用点使用同一归一路径。

## 已确认的配置输入

在 `intent.compile` 和 `compile_shared_gpu` 上增加显式 `tuning_config=path`，对应 CLI `--tuning-config`。JSON 只覆盖已有 profile 家族中的有限数据行，区分 shared 与各 provider；不含算法、条件表达式或 kernel 名。未提供的家族使用随编译器分发的默认数据，显式提供的家族整组替换默认候选，不隐式追加候选。

调表不重编译 C++，但需重新编译受影响 kernel artifact。解析与读取发生在配置物化的编译阶段；shape/ownership/capability 分类、角色绑定与 legality 留在代码，实际选择的候选仍完整写入 typed IR。文件不可读、未知字段/家族、错误类型或空表直接诊断；合法数据行仍需通过当前程序与 provider 的约束过滤，无合法候选时明确失败，不退回默认表。默认数据与各自 shared/provider 职责相邻，不新增顶层策略系统，也不在 launch 时重读文件。

# Non-goals

- 不推翻已有 domain/structured KIR/physical program；不删除高级通用构造，也不将所有公开名字变成独立 canonical op。
- 不启动新硬件后端，不以所有 examples×所有 provider 全面覆盖、双机全量或统一性能阈值作为交付。
- 不恢复旧 cuTile 性能追平任务，不自动采用其未提交试验。
- 不建设 pytest、fixture、测试台账、benchmark/兼容/迁移体系，不做与当前目标无关的目录重排。
- 不按 kernel 名、source template、registry case 或 CSV 成绩定义编译策略；不通过放宽精度/effects 或改变算法获得“通过”。

# Acceptance examples

本轮的结果型验收：

- A1：作者能通过已确认的具名计算接口编写对应运算，签名、shape/dtype/default 和诊断可发现；操作机械归一到已有语义，不要求重复填写可由操作定义确定的轴/组合规则。
- A2：已有复杂算法仍能使用具名操作、typed helper、generic structured constructs 和普通 control/effects 组合；在本轮涉及的目标路径上形成可执行后端程序，不把局部形态差异误判为算法能力边界。
- A3：纳入本轮的已知数值/effect 问题兑现明确契约；f32 输入精度、signed floor/remainder、logical index 和一次调用的可观察 effect 不继承不等价的 provider 默认值。
- A4：纳入本轮的 transformation/config 重构具有清楚职责：完整阶段的 postconditions 可定位，有限候选数据与选择逻辑分离，合法性不被固定搜索预算冒充；保留既有 typed IR 权威。

# Constraints and invariants

- 作者决定算法和可观察语义；语言改善表达；compiler/provider 决定语义允许的物理组织。普通 loop、显式逻辑 chunk、region、typed record 的职责不混淆。
- `doc/` 是规格权威；实现前完整阅读对应章节。报告已确认的设计变化经 Shape 明确后写入 `doc/`，不反向改规格来迁就当前失败。
- 当前生产基线来自已合入调查报告的 `main`；主线程负责方案和代码修改，子代理仅作探索/只读复核。
- 原主工作区两处 cuTile 试验保存在具名 stash `paused-cutile-round-six-before-foundations-merge`，不自动应用到本 change，也不丢弃。
- Shape 只作必要的代码/ref 调研，不运行测试或新建进度记录。实施时只使用仓库允许、与实际改动相称的单条 production repro，不扩大验证体系。

# Decisions

- 上一轮 `compiler-foundations-reassessment` 已归档、本地合入 `main`，旧 worktree/分支已清理。
- 新 change 使用独立 worktree，分支 `comet/dsl-compiler-foundations-refactor`，目标 `main`；用户已要求完成后本地合并并保持干净，不推送、不创建 PR。
- 用户已在完整范围、公开 API 与 JSON 配置输入说明后回复“同意，可以进入实现”，确认本 Shape 并授权 Build。
- Q1 已确认：作者接口/函数复用、编译阶段与 config 组织、四类已知语义修复全部纳入本轮，不另拆一个仅作者接口的缩小范围。
- Q2 已确认：采用本文具名操作分工、矩阵类 batch broadcast、显式 accumulator dtype 与 builtin prefix/identity 契约，保留通用构造。
- Q3 已确认：采用显式 `tuning_config` JSON 输入与按家族替换规则，调表不重编译 C++。
- 拆分检测：公开归一后的 lowering、语义修复与 config/stage 重构会交叉修改 shared transformations 和 provider legalization/serialization；维持单个 Native Change、由主线程顺序实施，避免在相同核心边界做并行集成。子代理仅承担探索/只读复核，不创建子 change 或派发实现任务。
- 本地合并与清理仍是本 change 的必需交付，由验收通过后的 Archive 完成；不把它列成要求在 Archive 前已经完成的代码验收项。

# Open questions

无。用户已确认范围、公开契约与进入实现；新发现的语言语义或用户可见行为分叉仍须重新澄清。

# Verification expectations

本次 Shape 以已有代码、ref 和报告为依据，不重新跑 baseline。实现阶段按确认的操作与目标选择必要的单条 DSL→后端源码→实际数值 repro，并给 current/ref 的具体差异；不因框架或台账而扩张任务。所有新增运行只服务于已请求的实际实现，不能替代对语言与编译职责的思考。
