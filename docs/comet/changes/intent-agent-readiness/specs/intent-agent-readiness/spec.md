# Intent 作者工具与本地评估准备

## 1. 结果与范围

本能力提供隔离的 Luna 作者环境、只读 Intent 手册 MCP 和正式 compiler/benchmark 接线，并收束首批 ownership/schema 缺陷。它服务于后续 TritonBench-T 子集评估，不是多轮 agent 研究框架；不以完成全量实验或 Intent 全面胜出作为本轮完成条件。

旧 `agent-tritonbench-evaluation` 已废止，其三次重复、每阶段五次候选、继续优化和预算轨迹不再是有效要求。清理保留解释结果与 compiler 缺口必需的源码、测量、输入合同和失败说明，删除无用副本、过程日志和重复汇总，不按性能好坏过滤任务。历史已提交内容由 Git 保存，不另造大归档包或转移到临时目录继续维护。

## 2. 使用手册 MCP

手册是公开语义的作者向入口，以 `doc/` 为权威并关联当前公开 API 声明，不建立第二套语义或手写导出清单。概念材料覆盖 dtype/literal、broadcast/index relation、domain/indices、scalar/rank-0、tuple/record、reduce/contract、helpers/control/effects 和 kernel/host 边界。

提供三个只读职责：检索概念/API/诊断/示例，精确查询 API 签名与返回 schema，读取章节及完整通用示例。返回来源定位和必要上下文；不存在的名字、合法但实现未支持的构造、尚未验证的行为分别说明，不能编造可用性或性能结论。

通用示例包含 imports、kernel/helper 和可运行的 host 编排；多 kernel 示例分别编译并显式传递中间 tensor。`build(context)` 是实验适配，不改变 public compile/artifact/launch 语义。示例不使用未定义的分配/launch 占位，不直接提供评测任务的优化答案。

MCP 初始采用本地 STDIO 和精确符号/全文检索，缓存不提交。服务仅读取发布手册、API 资料与通用示例；不执行候选、不调用 compiler、不读取 oracle 或历史答案、不调用另一个模型。文档与实现冲突应显式揭示，而非以实现覆盖规格。

MCP 属于 Python 作者工具，不进入 compiler IR/pass 或 runtime 执行语义；实验驱动只连接它，不复制另一套服务。同批材料与 compiler 对应同一明确快照，Git revision 和来源记录足够，不使用 hash/checksum 校验。

## 3. 独立 Luna 环境

复用现有 Codex CLI，配置、会话状态和候选目录放在项目外；固定使用用户指定实验 provider 的 `gpt-5.6-luna`、`max`。不得继承主开发 provider、开发对话、个人/项目 Comet 记忆、其它任务答案或默认辅助 agents。密钥由专用认证环境提供，不能进入 Git、prompt、报告或日志。

独立 agent 只能访问任务说明、对应语言手册工具和自己的候选目录；compiler、reference 与历史测量由评测端持有。文件权限、模型连接、MCP 进程/网络权限分开配置，独立状态目录不能替代实际访问边界。只开放本任务所需文档与编辑能力，不启用子代理、自动修错反馈或额外答案检索。

provider 的接口、流式/工具交互和指定模型/effort 必须通过真实调用确认；文件配置通过不等于运行可用。若不支持则准确报告，不静默更换模型或建立协议转换平台。

一个任务交付一份完整程序，可以包含多个 kernels 与正常 Python host wrapper。Codex 内部查文档/编辑的多个模型回合不计成多份候选，也不能称为裸模型单次请求。提交后由既有 production benchmark 编译和评估，agent 不接收 benchmark 反馈再自动重试。

## 4. 首批 compiler 收束

第一批范围是：重复 dimension/source occurrences 的独立 ownership，以及 scalar/rank-0/tuple 的 schema 衔接。使用此前保存的真实候选定位，不另写测试用例，不因表格状态或错误阶段标签直接判定责任。

相同 extent identity 不得合并独立 operand/result 坐标；同一 source 在 contraction 两侧也须保留各自角色。修复必须使 current physical program 的 mapping、fragment、access、validity 和 contraction relations 一致，不能只更换参数名或放宽 verifier。

Scalar、rank-0 tensor 和 structural products 保持各自类型语义，必要转换/广播在正式 lowering 中明确形成。合法作者表达不得因 compiler 产生错误 KIR 而被拒绝；确为作者错误则给准确诊断，不能靠统一强转或默认 dtype 掩盖。未定因的 `min` 等记录须先归因，不能承诺任何原候选修后必然正确。

已有等价 step、dot out_dtype 与输出分块修复保留。Out 已定义内容读取、safe indexed access、ordered control 中 parallel、其它数值和性能异常仍作为后续根因保存在复盘报告，不因本 change 完成而标为解决。第一批如果涉及新增 public semantics 或核心架构分叉，必须回到设计确认。

所有修复对照 ref/triton 或 ref/tilelang 的具体职责与实现，保留 file:line、差异和实际后果；shared pass 形成通用执行结构，provider 只补局部形式，serializer 不重建算法。不引入 task/kernel-name matcher、source template 或隐藏 kernel。

## 5. 最小运行与交付

用一个既有任务的一次候选交付确认“独立 agent 查询 MCP → Intent → 正式 compiler → production benchmark”可用；该运行是环境接通，不作为正式 50 题实验结果。首批 compiler 修复只重跑受影响的原始程序。

Benchmark 在同一输入下做一次既定容差检查并记录完整 CUDA Graph 算子时间，包含所有 kernels 和内部数据处理；编译、JIT、有限 tuning、预热与外部分配不冒充算子毫秒数。准备允许资源内并发，同设备计时避免重叠。没有数值结果不声称正确，没有实测不声称性能提升。

不新增独立数值、边界、回归、压力或组合测试，不新建 test 目录、fixture 或临时测试脚手架。配置/MCP 的必要真实调用不能扩展为另一套验证工程。项目只保留必要结果与说明，不继续发布多轮预算和重复统计表。

后续正式评估复用 TritonBench-T 当前 50 题子集及明确 profile，两组各一次完整交付，使用同一模型与相应语言资料。该全量评估、生成 Triton 的后续优化实验，以及完整 compiler backlog 均不属于本 change 的验收范围。
