# 目标

以 TritonBench-T 为共同任务来源，评价同一个 Codex/Luna agent 使用 Intent DSL 与直接编写 Triton 时，获得数值正确且高性能 GPU 程序的能力；进一步评价 Intent 生成的 Triton 作为 agent 后续优化起点的价值。实验接入与 compiler 攻坚共同推进，不把当前实现缺口当作不可修改的边界，也不预设实验必须得到正面结论。

# 范围

- 实验一：相同任务、模型、语言资料规则和预算下，比较 agent 编写 Triton 与 agent 编写 Intent 后编译到 Triton。记录首次成功、预算内成功、正确且快的任务比例和实际算子时间；不能以人工 DSL 代替 agent 输出。
- 实验二：在同一任务上比较直接生成的 Triton 继续优化、Intent 生成的 Triton 继续优化，以及 Intent 生成代码不再修改。优化阶段双方都编辑 Triton；区分起点质量与后续增益，以及优化阶段预算与完整工作流预算。
- 首轮 agent 为 Codex CLI，显式指定 `gpt-5.6-luna` 与 `model_reasoning_effort=max`，单 agent；各任务和独立重复使用新会话。保留真实编辑、编译、性能反馈和修改循环，不继承开发对话、Comet 记忆或已有同题答案。
- 每个任务的两条路径各做 3 次独立重复；每次重复的生成、Triton 优化阶段分别最多提交 5 个 agent 候选，初次生成计入生成阶段预算。静态起点对照复用该次生成结果，不另开 agent 运行；单独阅读参考代码的对照后补，不纳入本轮验收。
- 共用 T 复杂版中的任务语义、接口和公式；分别提供对应版本的 Triton/Intent 语言文档和通用示例。复杂版不是 Triton 教程，one-shot 检索是额外实验设置，本轮不暗中加载训练语料或优化答案。
- 以全部 166 个任务建立范围对应；首轮实测 50 个不同任务，横向覆盖 pointwise、reduction、contraction、融合与索引等计算结构，不以同一任务的多个 shape 凑数。纳入规则依据任务语义、输入合同与实验范围，不依据 Intent 是否已经能编译。范围内 compiler 不支持、agent 错误、超差、预算耗尽均保留结果并区分原因。
- 用户明确授权修复任务暴露的 compiler bug、缺失 lowering 和通用性能问题；保持 doc/ 的语言/IR/pass/provider 边界。正式结果对应一致的 compiler 版本，修复只重跑受影响 benchmark，不把人工修 compiler 算作被评测 agent 的贡献。
- 复用项目生产 benchmark 的输出比较、CUDA Graph 计时和准备/计时协调；必要任务与候选接线属于正式实验入口，不建立性能运行之外的测试体系。结果持续写入项目，不以 /tmp 产物或上游历史性能数字替代。

## Source coverage

下表保存已经明确的用户需求。TritonBench 仓库和论文是任务数据及实现参考，不是要求复制其全部框架或训练流程的规格文件；逐任务的语义适配在本 change 范围内完成，不把结构扫描误记为 166 项都已验证。

| 来源单元 | 读取 | 当前要求或归类 | Spec | 验收 | 状态 |
|---|---|---|---|---|---|
| 用户提出 TritonBench-T 同任务生成实验 | complete | 比较 agent 使用 Intent/直接 Triton 的正确率和性能 | §1、§3 | A1、A3、A6 | covered |
| 用户提出生成 Triton 后继续迭代 | complete | 相同优化预算下比较起点及优化轨迹；基准代码本身不再优化作对照 | §4 | A4、A6 | covered |
| 用户强调高性能 reference 的作用及 Q3 确认 | complete | 本轮完成 warm-start 主实验；只阅读参考的独立效应后补 | §4 | A4 | covered |
| 用户明确 compiler 可修 bug/补不支持 | complete | 开发与实验协同，修复真实 compiler 缺口；不永久冻结开发 | §6 | A5 | covered |
| 用户认可 Codex/Luna max 方案 | complete | 同一单 agent、显式模型/effort、新会话与统一权限/预算规则 | §2 | A2 | covered |
| 用户认可任务 prompt 与语言教学材料分离 | complete | 共用复杂版任务语义；两边提供语言文档和非同题示例 | §2 | A2 | covered |
| 用户要求增加到约 50 个任务 | complete | 首轮按 50 个不同任务确定，扩大横向覆盖，替代原 20 项提议 | §1 | A1、A3、A4 | covered |
| 用户回复“可以”确认 Q2/Q3 推荐设置 | complete | 每任务每组 3 次独立重复、每阶段最多 5 个候选；阅读参考对照后补 | §2、§4 | A2–A4 | covered |
| 用户要求收口旧任务并创建本 change | complete | CPU change 已接受并归档；本 change 使用 main/current，先 Shape | §8 | A1 | covered |
| TritonBench README 各节 | complete | G/T 数据、原环境、原评测、训练语料是参考背景，不直接照搬旧脚本 | §1、§5 | A1 | background |
| T 简单/复杂 JSON | partial | 两份各 166 项的结构已扫描，代表 prompt 已读；逐题合同与首轮名单尚待形成 | §1 | A1 | background：数据语料而非需求规格 |
| 上游 T 正确性/性能脚本与代表算子 | complete | 已核实 stdout 比较、极端 ratio 排除及任务边界问题；不沿用其错误判断 | §5 | A1、A6 | background |

# 非目标

- 不直接复现论文所有模型、SFT、BM25 RAG、CodeBLEU 或历史榜单；不使用 ARS。
- 不在本轮自动扩展多个 LLM、多个推理档位、全部 GPU/CPU/DSA 后端或新输入测试矩阵；完整 Shape 采用当前本地 RTX 5090 D 单 GPU。
- 不在本轮加入只阅读参考代码的独立对照；本轮通过静态起点与继续优化的主对照评价生成代码的起点价值。
- 不为数据集建立 kernel-name matcher、整算子 source template、runtime 算法替代品或 emitter 内隐式优化；不默默修改任务算法、关闭 dropout、忽略副作用或放宽容差。
- 不把第二阶段 agent 修改后的 Triton 宣称为 compiler 原生输出，也不把本轮开发中已见任务的结果宣称为未见任务零样本泛化。
- 不新增 worktree、不修改个人 Codex 默认配置、不推送、不创建 PR；新 Shape 确认前不发起被评测模型调用或 GPU benchmark。

# 验收示例

- A1：166 项具有稳定来源对应和明确的纳入/排除/待处理说明；首轮 50 个不同任务组成横向集合，有一致的任务语义、接口、输入和数值合同，使用正式生产 benchmark 适配，不以是否已能编译来过滤结果。
- A2：Codex/Luna max 以独立会话和固定配置真实运行；每任务每组 3 次独立重复，每阶段最多 5 个候选；两路径共用任务信息、预算和反馈规则，各自获得语言资料；实际模型、调用配置、候选提交和停止原因可记录，不继承开发历史或读取未授权同题答案。
- A3：确认集合上的直接 Triton 与 agent 编写 Intent 两条生成路径均获得实际运行结论；首次/预算内正确率及正确且快的比例使用固定分母，compiler 拒绝、超差与未完成不被当作成功或静默删除。
- A4：第二阶段比较直接 Triton 优化、Intent 生成 Triton 优化与不修改生成代码的对照；报告最好正确程序的绝对耗时轨迹和实际预算，区分 warm-start 与总流程收益；静态对照复用对应生成结果，不将阅读参考的独立效应混入本轮结论。
- A5：任务暴露的 compiler 问题有基于当前 typed IR 与 ref/triton 或 ref/tilelang 的具体归因；必要修复进入正式 construction/pass/provider 边界并重跑受影响性能项，不以 unsupported 记录代替应推进的 lowering，也不使用单题特判。
- A6：项目内正式结果包含每个 task/case、实验组、重复、候选阶段的状态、真实算子时间、reference 时间和清楚定义的比值；汇总包含失败，不按极端速度删除数据，不混入 JIT/agent 耗时，不要求预设 Intent 必胜。

# 约束与不变量

- `AGENTS.md` 与 `doc/` 是语言和 compiler 规格权威；基准、已有 example、CSV 和此实验不能反向定义语言语义。若需要新增 public semantics，先取得明确设计确认。
- correctness 来自同输入下的独立参考输出与预先确定的容差，不来自 LLM 自评、stdout 相等或仅编译通过；对 tensor/tuple、shape/dtype 及必要外部效果保持相同合同。
- 两组均允许明确的 host 包装与多 kernel 组合，GPU 计算不能通过调用 PyTorch reference 作弊；多 kernel 的完整执行时间和内部数据处理进入计时。
- Compiler 内部调参与 agent 提交程序不是同一种候选；分别记录实际搜索消耗，不能让一条路径获得未披露的额外搜索预算。算子时间和搜索成本分开。
- Agent 可修改候选代码，不能修改评测器、任务参考或 compiler；我们在开发侧继续修 compiler，并统一更新受影响比较，不将人工协助计入 agent 自主成功。
- 只运行为获取实验性能结果所必需的编译、预热、计时和同次容差比较；每个结果及时发布到项目，失败不清空已有结果。

# 决策

- `cpu-programmable-lowering` 已由用户接受 5/5 验收并归档，归档提交 `8d01146`；其 CPU/RVV 范围及历史性能门槛不自动进入新 change。
- 新 change 名为 `agent-tritonbench-evaluation`，沿用 main/current，当前只建立 Shape。
- Q1 已解决：用户要求增加任务规模，首轮确定为 50 个不同的 TritonBench-T 任务，替代原 20 项提议；不改变全部 166 项的范围对应与结果纳入原则。
- Q2 已解决：用户确认每任务每组 3 次独立重复、生成和优化阶段各最多 5 个 agent 候选；候选失败和预算耗尽保留，预算不授权无限调优。
- Q3 已解决：用户确认单独阅读参考代码的对照后补，本轮先完成生成、warm-start 与静态起点主对照。
- 用户明确回复“继续，按照你的要求，可以的话进入build吧”，接受当前完整 Shape、A1–A6、运行 profile 和非目标，授权进入 Build。
- 用户明确要求直接思考，不使用 ARS；Comet Native 负责本 change 流程。
- 首轮使用 Codex CLI + Luna max；这是 agent 系统评估，不是裸模型单次输出评估。已核实本机 CLI 为 0.154.0；个人默认模型不等于实验模型，实验必须显式绑定。
- 两个主实验的统计对象不同：生成实验衡量入口效果；优化实验衡量编译生成代码的起点价值。若仅给参考资料，则是第三个明确的处理条件，不能混在第一阶段。
- Compiler 在开发阶段可持续修改；正式比较固定的是被报告结果的版本与口径，不是禁止修复。未受影响数据可保留，受影响数据补跑。
- 原 T 的 `1_exe_acc.py:10-29` 只比较 stdout，`softmax_mul.py:59` 只保存结果；`2_efficiency.py:24-25,40-45` 会排除极端加速比。必须复用任务而非继承这些评测判断。
- 已核实本地 GPU 为 NVIDIA GeForce RTX 5090 D、32607 MiB；完整 Shape 采用该单设备，不以旧 A100/H100 数字代替实测。硬件识别不等于已完成实验或全部原始输入均能运行。
- 当前不拆 Supervisor：两项实验复用同一 task/候选/benchmark 接口、compiler 版本与第一阶段产物，紧密耦合；不是两个互不相关的实现工程。确认前不派发执行子任务。

# 待解决问题

无未解决的用户决定。Q1–Q3 和完整 Shape 均已确认，按当前范围进入 Build。

# 验证预期

只用正式生产 benchmark 获得需要的数值/性能反馈；复用相同输入和预定容差，不复用上游 stdout 判定，不新增单元/边界/回归/压力测试。计时采用统一 CUDA Graph 口径，覆盖完整算子及必要中间处理；编译、调优、模型调用、加载与外部分配不冒充 GPU 时间。准备与编译允许资源内并发，GPU 实测与内部 autotune 的计时不能相互干扰。首轮为 50 个不同任务，每任务每组 3 次独立重复、每阶段最多 5 个候选；阅读参考对照后补，原始结果和实际限制如实交付。
