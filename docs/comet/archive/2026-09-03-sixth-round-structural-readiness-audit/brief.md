# 目标

闭合 05c/05d 后审查暴露的 shared indexed-access bounds correctness 阻塞，再次确认 canonical KIR→shared executable GPU Program→provider legalization→terminal serialization 的权威与验证边界是否已经干净，并对“第六轮是否可以在不再大改 shared/core 的前提下，主要推进 cuTile 与 TileLang leaf lowering”给出明确的 Ready 或 Not Ready 结论。

# 范围

- 以 `doc/` 为唯一语义与架构权威，审查 program mapping/traversal、ownership、coordinate/access/validity、resource/buffer、physical parameter/config 以及 provider-local extension 是否都由 current typed IR 表达并在正确层级验证。
- 沿唯一 production compiler path 核对 Triton、cuTile 和 TileLang legalizer/serializer/runtime 的责任，并与 `ref/triton`、`ref/tilelang` 或 cuTile 的当前 source/API 同类实现比较；只有会导致错误语义、无法 lowering 或迫使 provider 重建 shared 事实的差异才算阻断。
- 重新裁决历史自查线索与当前代码：已修复、已由 typed carrier/provider verifier 闭合或不再成立的项不重新打开。
- 将当前 cuTile/TileLang 剩余工作按实际后果分为：可由现有 shared program 直接承载的 provider leaf form/pass/verifier/runtime 工作、规格允许的明确 unsupported，或真正缺失 shared semantic carrier/invariant 的 core blocker。
- 当前已确认不需重开的基座包括 compile-device artifact binding、serialization 前的 TileLang config legality、cuTile rewrite 后的 MLIR/local surface verification、`sparse_contract_2to4` 的唯一 canonical path、TileLang scaled-contract provider form，以及 Triton serializer 不读 KIR/名称重建 shared mapping/access 的边界。
- 修复已确认可达的 shared access validity/bounds 缺口：动态或间接访问只有在 current typed program 中存在显式有效性、可验证的 relation 范围或匹配的 canonical in-bounds 前提时才合法，并在进入 provider 前闭合这一不变量。
- 修复后重新裁决：buffer initialization/lifetime、provider-local rewrite 产生的 resource/control/form，以及 cuTile/TileLang 剩余 native access、storage/copy/sync/pipeline 与 structured forms 是否仍能作为 leaf 扩展。

# 非目标

- 本 change 只修复已经确认的 shared indexed-access bounds correctness 阻塞并重新审查；不提前实现第六轮 cuTile/TileLang 功能。
- 不修改 DSL/canonical KIR/`doc/` 语义，不因 ref 拥有 TMA、persistent、occupancy 或特定 MMA 形式就把它们上移到 shared IR。
- 不把 provider 当前未支持的每个 dtype/primitive/form 都判为架构缺陷；有精确 provider rejection 且不缺 shared 语义的项不阻断第六轮。
- 不重跑 Triton 双机全量或进行性能优化，不建设 test/pytest/fixture/新门禁/兼容层/额外证据文档。

# 验收示例

- Scenario 1：给定包含动态或间接索引的 canonical kernel，shared construction 产生的每个可达访问都由 current typed validity、可验证 relation 范围或匹配的 canonical in-bounds 前提证明安全；无法证明的访问在 provider lowering 前被精确拒绝。
- Scenario 2：给定同一 shared GPU Program 分别进入 Triton、cuTile 与 TileLang，provider legalization 只从 current typed facts 选择 local form，改写后的 local ops/resources/control/config 由 MLIR op verifier 与 provider verifier 完整闭合，serializer 仅作确定性 spelling 与已声明 wrapper 绑定。
- Scenario 3：给定当前 cuTile/TileLang 未覆盖或低效的可达形态，审查结果证明它们要么能以 provider-local IR/pass/verifier/serializer/runtime 工作完成，要么在最早拥有足够信息的 provider/capability 层精确拒绝；任一必须新增 shared 语义或大改 common program 的反例都使 Ready 结论失败。
- Scenario 4：给定一条现有 production compile/emit/JIT/launch 命令，它实际运行一个由目标 provider 当前完整支持、同时覆盖 shared access 与 provider-local form 的代表性 kernel，并通过数值比较；审查结论不依赖新测试体系或重跑全量 registry。

# 约束与不变量

- `doc/` 是唯一设计权威；report、archive、CSV、source 与 ref 只是调查入口或 provider 实际 surface 证据。
- 审查必须给出 current/spec/ref 的 `file:line`、具体差异与实际后果；缺少可达反例的“验证不够多”或“ref 功能更多”不得单独判为 blocker。
- shared GPU IR 继续是唯一 provider-neutral executable authority；provider-local storage/copy/TMA/MMA/pipeline/layout 与 tuning form 不因审查上移到 shared。
- Ready 只表示当前第六轮目标可以主要在 cuTile/TileLang leaf 完成，不表示所有 provider surface 已经实现或未来任何新语言构造都永远不需 shared 扩展。

# 决策

- 使用一个普通 Native Change，不拆 Supervisor；本轮只生成一个结构就绪性结论。
- 不把挂在 current operation 上、并与显式 operands/results/control 一起被 verifier 约束的 typed mapping attributes 误判为 side decision table。
- provider rewrite 后的闭合标准是“MLIR schema + provider-local semantic verifier 足以验证新 current program”，而不是机械要求已不再是纯 shared surface 的 IR 再次通过 shared-only verifier。
- 若只发现可在 target dialect、provider pass/verifier/serializer 或对应 runtime adapter 内闭合的问题，结论为 Ready with bounded leaf gaps；只有现有 shared IR 无法表达所需语义，或存在已可达的 shared correctness 反例时才判定 Not Ready。
- 用户已明确把“修复审查阻塞并继续审查”纳入本轮；由于 Native Runtime 不允许在已有 current-isolation active change 的同一工作区再创建第二个 change，本轮使用该 change 的 repair iteration，保留首次 A1/A4 失败记录而不重写历史结论。

# 待解决问题

无需用户决策的语义分叉；Build 阶段依上述标准完成事实裁决。

# 验证预期

- 静态审查以 current/spec/ref 双向 `file:line` 对照为主，对争议点追到实际 executable consumer 和可达后果，不用文件宽度、缺少某个名为 verifier 的 pass 或 ref feature count 代替语义证据。
- 只执行一条已有 production 路径命令做代表性数值 repro；不新增测试、fixture、runner 或全量门禁。
