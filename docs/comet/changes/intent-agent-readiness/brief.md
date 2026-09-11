# 目标

建立一条可实际使用的本地 Intent 作者链路：独立 Luna Codex 查询使用手册 MCP、交付 Intent 程序，经正式 compiler 和既有 benchmark 获得正确性与算子性能；同时修复首批共性 compiler 阻塞。先把项目和工具做好，不恢复已废止的多轮 agent 实验框架。

# 范围

- 收口旧 `agent-tritonbench-evaluation`：清除 active 残留和无用过程产物，保留必要任务、原始失败/候选和测量依据；有效 compiler/runtime 修复不回滚。
- 建立只读 Intent 使用手册 MCP，提供概念/诊断检索、API 精确查询和章节/完整通用示例读取。覆盖 dtype/shape、scalar/rank-0、tuple、domain/indices、reduce/contract，以及显式多 kernel host 编排；来源以 `doc/` 为权威，API 信息复用公开声明。
- 复用本机 Codex CLI，建立项目外专用配置、状态和候选工作目录。使用用户指定的实验 provider、`gpt-5.6-luna` 与 `max`，不继承开发会话、Comet 记忆或旧答案，不改主开发 provider。凭据只放专用认证环境。
- 实际确认 provider 的必要通信与独立 agent 的 MCP 使用；以一个既有任务的一次完整候选交付接通 production benchmark。允许多个 kernels，不把读文档/编辑的模型回合算成多个候选，不把该连通运行当正式实验结论。
- Compiler 首批聚焦两个共性边界：重复 dimension/source 出现位置的独立 ownership；scalar/rank-0/tuple 的 schema 衔接。以已记录原始程序定位和修复，实现向既有规格收敛，不为了通过而放宽类型或物理合法性。
- Out 先写后读、safe indexed access、ordered control 内 parallel 及其它数值/性能异常继续保留在复盘报告，随后按根因推进；不将它们遗忘或称为已解决，也不把整个 compiler 的收尾塞进本轮环境验收。
- 后续正式评估沿 TritonBench-T 当前 50 题子集，每题两组各交付一次；本 change 准备所需精简入口，不执行全量评估或第二阶段优化。

# 非目标

- 不恢复三次重复、五次反馈修错、五次优化、预算轨迹及一批重复统计表；不部署 RAG 向量库、重排模型或自动题解服务。
- 不在本轮完成 50 题正式实验，不宣称复现原论文完整协议或预设 Intent 必胜。
- 不改语言语义、添加 whole-operator matcher、source template、隐藏 launch 或 emitter 内执行结构；核心设计分叉须另行确认。
- 不创建开发 worktree，不改 CPU/DSA 实现，不升级或修改外部 ref，不安装第二套 Codex，不推送或创建 PR。

# 验收示例

- A1：旧任务不再出现在 active 列表，实验目录只保留可解释结果所需内容；重复报表、无用过程日志和旧多轮调度不再作为活动入口，原始失败和必要复现未被选择性抹除。
- A2：独立 agent 能通过只读 MCP 查到真实 API 签名/返回类型、语义规则及完整多 kernel 通用示例；未知 API 明确返回不存在或未确认，工具不执行代码、不读取 oracle、不生成题解。
- A3：专用 provider 的 luna/max 经真实调用确认；独立 agent 使用 MCP 交付一个既有任务程序，正式 benchmark 得到容差检查和完整算子毫秒数，配置与模型不继承主会话，凭据不进入仓库。
- A4：首批重复轴 ownership 和 scalar/rank-0/tuple 问题被准确归因，确认合法的原始表达不再因这些 compiler 缺陷阻塞 lowering；修复有 ref 对照和受影响既有 benchmark 结果，错误作者程序保持准确诊断，未证实问题不冒充已修复。

# 约束与不变量

- `AGENTS.md`、`doc/` 是规格权威；[复盘报告](../../../../report/intent-tritonbench-reset-and-manual-mcp.md)记录事实和后续根因，不替代语言规格。
- 编译器负责 shared physical structure；provider 处理真实局部差异；serializer 仅拼写。作者可显式多 kernel，compiler 不隐藏拆分或跨 kernel workspace。
- MCP 仅暴露发布的手册、API 资料和通用示例，不访问个人/项目记忆、历史答案或任务 reference；文档与实现冲突明确显示，不能反向修改语言规则。
- 运行仅为配置接通所必需的模型/MCP 调用及既有算子 benchmark；同次一次容差检查，不扩大容差、不新增独立数值/边界/回归测试或临时测试脚手架。
- 编译、JIT、有限 tuning 和预热可以存在，但不计为算子时间。准备可并发，同 GPU 计时不重叠。
- 保留用户其它改动。已废止资料可删除，唯一测量/失败复现先明确保留位置；不整批 revert 混有有效 compiler 修复的提交。

# 决策

- 用户先要求报告、废止旧 change，随后要求先建 Shape 留档；本轮停在 Shape，未经最终确认不实现 MCP、配置 provider 或启动 benchmark。
- 旧 change 主目录已封存；经用户允许，CPU worktree 中继承的三个旧文档也已清除并单独提交，CPU 实现不变。新 change 使用 `main/current`。
- 环境隔离指被评测 Codex 的配置/会话/文件权限隔离，不是创建第二个开发 worktree，也不代表只换 CODEX_HOME 即完成隔离。
- 手册优先保证内容准确和可查询，以本地 STDIO MCP 提供小接口；不另建与 compiler 并行维护的语言定义。
- 暂不拆 Supervisor：本轮交付是一条本地作者入口链，手册、agent 配置、薄适配和首批 compiler 复现共同演进；完整实验与其余 compiler 攻坚不塞入本轮。
- provider 接口能力尚未实测是 Build 工作，不假定 `/v1` 即兼容；若实际不支持指定模型/协议，应报告缺口，不静默换模型或搭转换平台。

# 待解决问题

- [blocking] CONFIRM: 确认本轮仅完成旧产物收口、手册 MCP、独立 luna/max 环境及首批 ownership/schema 修复；一个既有任务用于接通链路，50 题正式评估和其它 compiler 根因留后续，不进入旧多轮实验。

# 验证预期

只做必要配置/查询可用性确认，并复用同一条 production benchmark 检查原始候选和受影响 compiler 路径；保留真实算子时间与同次容差结果。静态对照 ref/triton 或 ref/tilelang 的具体实现与职责差异，不为 Shape 增加独立测试命令或文件。本轮起草阶段没有模型调用或性能运行。
