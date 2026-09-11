# Intent 单次生成与编译器问题

本文件只保留当前工作边界和 compiler 问题定位，不保存旧候选、旧正确率、性能数字或迭代结论。设计规格以 [doc/index.md](../doc/index.md) 为准。

## 单次生成口径

- 独立 Luna max 使用任务说明和对应语言资料，每题分别交付一份 Intent 程序和一份直接 Triton 程序。允许查手册、提交前编辑和显式多 kernel host 编排。
- 提交后由 evaluator 编译和 benchmark，不把错误或性能反馈给 agent 继续修复；没有重复候选取最好值或后续优化阶段。
- Bench 自带 reference 是双方共同的数值、性能参照。Intent/direct 和 candidate/reference 是不同的比较，不混用分母。
- Compiler/JIT、有限 tuning、预热和计时重复是执行准备，不是 agent 的迭代生成。
- 按用户要求删除此前全部实验数据及项目外旧候选副本，包括接通记录和 compiler 复测；不另建归档包或复现库，不再从旧档案提取实验结论。有效 compiler/runtime 修复不回滚。
- 当前单次入口、公开手册 MCP 和专用 provider 配置保留。删除历史产物不等于已经获得新的正确率或性能结果。

## 编译器责任

语法可解析之后仍须满足公开类型、shape、数值和 effect 语义。满足这些语义却 lowering 失败，是 compiler 实现问题；不能要求 agent 更换算法、不断拆 kernel 来迁就错误。

合法程序的失败应沿 canonical KIR、current Physical Program、shared passes、provider legalization 和 serializer 定位。修复执行结构必须落在 IR/pass，serializer 只拼写；不通过算子名称、旧候选模板或放宽 verifier 掩盖缺口。开发端修 compiler 与被评测 agent 一次交付是不同的活动。

当前代码保留了 tuple view-to-value、scalar/rank-0 schema、独立轴与 access relation、归约回调 legalization、循环挂接和公开 ABI 参数顺序等修复；这些实现不能被旧实验清理一并撤销，也不代表所有合法组合均已覆盖。

## 当前需要收束的实现边界

- **Contract、pointwise、reduce 的组合与独立轴。** 检查当前操作位置、axis relation 和实际数据流，不以相同 extent 合并坐标。更融合的合法表达应能正常 lowering；尚未重新确认的组合不得直接宣称成功或失败。对照 `ref/triton/python/tutorials/03-matrix-multiplication.py:308` 在 accumulator 写回前融合后续计算；本项目执行结构由 `lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp`、`RealizePointwiseBlocking.cpp` 与 `RealizeReductionBlocking.cpp` 负责，不交给 serializer。
- **多个 contraction / execution groups。** `lib/Dialect/GPU/Transforms/RefineProgramMapping.cpp:68` 的 grouped mapping 只接受一组 M/N；`:157` 在多个 delinearize mapping 时跳过 refinement。需按 `doc/compiler/kir-to-gpu.md:37` 的 group、依赖与 coverage 规则补齐，而不是删除错误检查或默认参数不 alias。Runtime 非重叠检查不是已确定方案。
- **已定义 Out 的读取。** `python/intent/frontend/lowering/ast/context.py:447` 仍按 view kind 无条件拒绝 Out read；`doc/dsl/authoring.md:31` 要求的是读取前已定义。需要表达并证明定义状态，不能让作者统一改为 InOut 掩盖缺口。
- **全局归约物理并行度。** `doc/compiler/kir-to-gpu.md:90` 的初始 mapping 把 reduction axes 留在 instance 内；`lib/Dialect/GPU/Transforms/RealizeReductionBlocking.cpp:2298` 将 runtime reduction 形成带 carries 的 chunk loop。对于 singleton workset，有限参数 tuning 不等于已形成跨 program partial/combine。它是待优化的物理结构边界，不是“所有 reduce 都错误”，也不授权隐藏新增 kernels。
- Safe indexed access 和 ordered control 内 parallel 仍属于后续实现核对范围；不从已删除的实验日志推断当前状态，不把其它后端的限制直接套到 GPU。

只有新的、按当前口径取得的生产 benchmark 才用于说明相应程序的数值与性能。本次清理不运行模型或算子，也不把未完成 compiler 工作标成验收通过。

## 当前入口与接续位置

- 工作目录：`/home/kingdom/phdworks/intentdsl`，分支 `main`。旧实验产物已删除；没有新的整体正确率或性能成绩。编译器未完成项仍见上节。
- 不建独立测试目录。一次生成入口是 `examples/repro/agent_study/__main__.py`，生产 benchmark 是同目录 `benchmark.py`；任务/profile 在 `suite.json`，作者提示在 `instructions.md`，编译接线在 `program.py`。
- 只读手册 MCP：`python/intent/tools/manual.py`，内容来自 `doc/` 与公开 API。
- 独立评测环境：`/home/kingdom/.local/share/intentdsl/agent-evaluation/`；认证仅留该环境，不抄入本报告。被评测 agent 继续禁用历史 memory，与开发 Codex 的记忆设置分离。
- 上游 Bench：`/home/kingdom/phdworks/ref/tritonbench`；Python 环境：`/home/kingdom/.venvs/intentdsl-mlir20`。当前可用编译器为 `/tmp/intentdsl-agent-build.tFX1vN/tools/intent-compile/intent-compile`，这是临时构建位置，不作为长期规范。

## 协作环境整理结果

- 已卸载 IntentDSL、TianchenRV 和全局 Comet 接入，移除项目 hooks/skills/state/档案、全局数据、用户级与系统级安装包、dashboard 及代理服务。两个已合入 main 的 CPU worktree 和已合入的旧 Comet 分支已清理；有效代码与正式 `doc/` 未改。
- 稳定原则已精简到 `/home/kingdom/.codex/AGENTS.md` 与项目 `AGENTS.md`，不搬运旧成绩或阶段门禁；CodeGraph/FastCtx 的命令与 MCP 保留，CodeGraph 索引已同步。
- 用户级 `/home/kingdom/.codex/skills/todoskill/` 只包含 `SKILL.md` 和 UI 元数据，用单份短期记录保存长任务的已确认决定与剩余节点，不创建管理流程。当前任务沿用本报告。
- 原生 memory 已通过 `memory/reset` 清空生成物、提取记录和生成任务。旧会话的记忆生成资格已关闭：当前存储格式用原生接口处理，旧格式只同步记忆元数据及索引标记；聊天内容、账号和其它配置保留。
- 开发 Codex 的 memory feature、读取和生成已开启；TOML 已解析，CLI 已加载 enabled 标志，skill 结构检查通过。未运行模型、独立算子测试或新 benchmark，也未宣称新会话的后台记忆提取已验证。
- 下一次开发工作应新开 Codex 任务，以两级 `AGENTS.md`、`doc/` 和本报告的入口接续；当前长会话的旧上下文不会因删除文件自动消失。被评测的 Luna 仍禁用 memory，不把开发经验注入一次生成实验。
