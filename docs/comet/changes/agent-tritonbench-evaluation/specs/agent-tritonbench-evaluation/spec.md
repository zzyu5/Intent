# Agent 生成与优化实验

## 1. 目标、任务与范围

本能力评价两个问题：同等任务和预算下，agent 编写 Intent 后编译到 Triton 是否更容易获得正确且高性能的实现；Intent 生成的 Triton 是否能改善 agent 后续优化的起点和效率。完整对照、真实数据和 compiler 缺口的工程收束是交付目标，不预设正面结论。

任务来源为 `https://github.com/thunlp/TritonBench` 的 TritonBench-T。外部参考仓库已位于 `/home/kingdom/phdworks/ref/tritonbench`，路径作为环境事实而不是硬编码实验语义。复杂版任务 JSON 为共同任务描述基础，简单版不是另一批任务；发布数据各含 166 条。训练语料、LLM_generated 和原论文的模型结果不自动成为 agent 可访问资料。

为全部 166 个任务保存稳定的来源和范围对应，并在获取比较结果前确定首轮 50 个不同任务的主实验集合。横向覆盖 pointwise、reduction、contraction、融合与索引等计算结构，不以同一任务的多个 shape 替代任务数量；不能以 Intent 已编译成功或某组性能好为纳入条件。语义范围外、参考实现/适配存在问题、compiler 尚不支持与 agent 写错分别记录，不通过静默删除失败改变分母。

每项明确 callable signature、输入 shape/dtype、标量/constexpr、输出结构及必要的 out/in-place/alias 等可观察效果。Forward tensor kernel、随机状态、autograd 或模型对象变换不可混为同一种任务；不默默关闭 dropout、改变 p、丢弃 backward 或将框架模型操作替换成一个算子后仍宣称原任务完成。必要 specialization 必须对两组一致、明确命名并披露与原任务的差异。

参考是独立的原任务 PyTorch 计算，不是我们的生成结果。T 的 PyTorch reference 提供基本性能锚点，不自动等于高性能 Triton 上限；核心比较包括同任务直接 Triton 组。新增强参考只在语义一致且有实际实现时单独标明，不由旧 CSV 或理论吞吐量替代。

## 2. Agent、资料与预算

首轮 agent 为 Codex CLI，显式绑定 `gpt-5.6-luna` 与 `model_reasoning_effort=max`。两条路径使用同一 CLI、模型、推理档位、工具能力和预算策略；不依赖个人默认模型，不修改个人配置。实际调用不可用时明确报告，不静默更换模型/推理强度或把未运行记为成功。

各任务和独立重复启动新会话；同一重复内的迭代保留其必要反馈。实验 agent 不继承当前开发对话、Comet 个人/项目记忆、其它任务的答案或跨实验组历史。初始主实验使用单 agent，不自动启用子代理、跨模型审查、RAG 或其它隐藏辅助流程。

任务的数学语义、接口与输入约束对两组相同，只有实现语言指令和对应语言资料不同。Triton 组获得相应版本文档与通用示例，Intent 组获得现有 DSL 文档与通用示例；资料选择规则预先固定，不能只给一组同题优化解。T 复杂版中增加的是数学和语义说明，不是 Triton API 教程；原论文 one-shot 的检索样例属于另加处理，不默默混入本实验。

Agent 可以读取允许的资料、编辑自己的程序、调用正式编译和 benchmark、读取错误与性能反馈，并据此修改。不能修改任务、数值容差、reference、评测器或 compiler，也不能读取用于判定的参考输出/未授权同题优化实现。独立参考的可执行代码由评测端持有，不通过整个项目文件系统无限暴露。

重复与候选预算待 brief Q2。一个 agent 候选是提交的一份完整程序及其 wrapper，不是 compiler/autotuner 内部的每个参数实例。初次输出也是候选；编译失败、数值失败和重复提交均不能得到无限次免费重试。固定停止规则与资源上限，保留成功/失败/耗尽/环境中断的实际结论，不由模型声称成功决定结果。

记录实际 agent token/耗时与 benchmark/调参搜索消耗，用来解释工作流效率，不能写成算子毫秒数。两边可使用合法 tuning，但遵守同一有限搜索规则；Intent 的内部调优不是不可见的额外预算。不必强制两种语言的候选集合逐项相同。

## 3. 实验一：编程入口

两组分别生成直接 Triton 与 Intent DSL。Intent 必须由本次 agent 从任务描述编写，经正式 frontend、canonical KIR、shared GPU passes、Triton legalization/serialization 和外部 Triton compiler 执行；不得拿人工 DSL 或已有项目同题 kernel 替换失败候选。

两组都可以使用任务所需的普通 host wrapper、外部分配和显式多个 kernels。GPU 数值计算主体使用对应路径，不能调用 PyTorch reference 代替生成 kernel。一个 Intent kernel 仍遵循 doc 的单 kernel/host 边界，不为实验添加隐藏 kernel 或跨 kernel workspace。

分别记录首次提交是否正确、固定预算内是否得到正确程序、当前最好正确程序的性能。编译成功与数值正确是不同状态；compiler 拒绝非法程序不能计为任务成功。输出比较直接消费实际 tensor/结果树，不比较 stdout，也不使用 LLM 自评分数。

失败种类至少可区分 agent 语法/算法错误、compiler 错误/缺失 lowering、外部编译失败、数值超差、预算耗尽与运行环境问题。分类只解释原因，不把范围内未完成的任务改成成功或移出端到端统计。

## 4. 实验二：生成代码作为优化起点

主实验有三个条件：

| 条件 | 起点 | 操作 |
|---|---|---|
| 直接 Triton 优化 | 实验一直接路径保留下来的程序 | agent 继续修改 Triton |
| Intent 起点优化 | 实验一 Intent 路径实际编译生成的 Triton | agent 继续修改 Triton |
| Intent 静态起点 | 与上一条件同一生成程序 | 不继续修改 |

起点按预定选择规则取得，不由人工挑选最好个案。若生成阶段没有合法起点，保留未获得起点的结论，不由我们手写补齐；分析双方有合法起点的配对子集时同时报告覆盖范围，不能代替全部任务的端到端结果。

为隔离起点作用，优化 agent 使用新的会话，得到相同任务信息、对应起点、起点测量与相同反馈规则，不继承两条生成路径中不等量的历史解释。之后双方都可编辑 Triton 程序及合法 wrapper，而不是一组仍被限制只能修改 Intent。

记录每次预算 checkpoint 的最好且正确的绝对耗时、达到共同性能目标的预算，以及最终失败。只比较相对起点的提升倍数不足以说明哪条路径更好；静态起点对照区分 compiler 原始质量与 agent 新增收益。

同时区分优化阶段预算和完整流程预算。后者包含生成 DSL/直接 Triton 起点所用资源；不能把 Intent 起点制作过程算成免费。经过 agent 修改的 Triton 明确标记为后续优化结果，不再称为未经修改的 compiler 输出，也不再承诺继承 compiler 的语义保持性质；仍由同一独立 reference 判定数值。

若 brief Q3 确认加入阅读参考对照，取同一份直接 Triton 初稿，分别允许/禁止阅读 Intent 生成的 Triton，其余输入和预算保持一致。该条件单独回答“参考资料”的作用，不与替换起点混合；未选择时不作为本轮验收门槛。

## 5. 数值与性能测量

复用项目正式 benchmark 的输出比较、CUDA Graph 计时与准备/计时协调。TritonBench 只提供任务、原参考计算及可复用输入，不能照搬 stdout 相等判断、删除失败文件或按极端比值排除结果的逻辑。

同一个 benchmark 对候选与 reference 使用相同实际输入，并做一次预定容差的输出检查。数值口径在结果产生前确定；上游没有明确容差时根据现有项目数值合同确定并记录，不为通过而放宽。保留输出 shape/dtype/结构与必要外部效果，随机行为不能仅凭“相同 seed”假设不同实现逐元素一致；未闭合的随机比较不可混入主统计。

优先采用统一 CUDA Graph replay 的完整算子执行时间，包含该调用所需的全部 GPU kernels、内部 materialization、转换与工作区处理。外部输入/输出分配、编译、JIT、tuning、加载、agent 推理及反馈通信不计入算子时间。若某任务不能使用同一计时方式，明确记录限制，不静默换成 Python wall time 或单个 kernel 的局部耗时。

准备与编译允许资源预算内并发；实际 GPU 计时，包括内部 autotune 的计时，要避免同设备其它任务干扰，不把所有 worker 全流程串行。首轮硬件提议为当前本地 RTX 5090 D，最终执行 profile 在 Shape 确认中固定；不把旧 A100/H100 数字作为当前 reference。

每得到一项结果就发布到项目表格，不等待全量完成，不因某项失败清空已有数据。当前任务集合、原有输入与确认的 agent 重复/迭代就是本轮测量范围，不扩展独立数值、边界、回归、兼容或压力测试，也不在 /tmp 新建另一套测试逃避限制。

## 6. Compiler 攻坚与归因

发现任务的合法 DSL 无法 lowering、输出超差或存在明显性能差距时，先定位 current physical program 的 mapping、blocking、ownership、traversal、reuse 与 materialization，再调查 provider form、serializer、外部 compiler 和测量。对照 ref/triton 或 ref/tilelang 的具体 file:line、差异与后果，不能只登记 unsupported 后停止推进。

用户已授权必要 compiler bug 修复、通用缺失 lowering 和性能优化。遵循现有 doc，复用 typed IR/carrier；shared transformations 形成共同执行结构，provider 补真实局部差异，terminal 只拼写。禁止 task/kernel 名称匹配、单题模板、旁路 recipe、改变算法或把高性能结构藏进 emitter。新增 public semantics 或核心架构分叉仍需明确设计确认；外部仓库改动不在本授权中。

开发阶段允许持续改 compiler，固定的是正式结果所对应的编译器与实验配置，不是永久停止开发。改动影响旧结果时补跑相关 benchmark，未受影响结果可复用；不能每题使用未披露的不同临时 compiler，也不能将人工开发动作计为 agent 自主修复。

开发任务参与 compiler 改进属于披露事实，不自动构成未见任务泛化证据。需要研究泛化时使用未参与开发的任务另行明确范围，不临时扩大本轮输入矩阵或制造额外测试门槛。

## 7. 项目结果与统计

实验驱动、任务适配与候选执行留在现有 `examples/repro/` 的职责边界，复用正式测量实现；compiler/runtime 不读取 benchmark registry 决定算法。必要 reference source/runtime 按 `source/` 现有语言、上游与算子职责组织，不将外部训练库和全部生成答案复制进项目。

正式结果保存在 `report/` 下本实验的表格与必要机器可读记录。保存 task/case、组别、独立重复、预算 checkpoint、编译器与实际模型配置、状态、候选/参考的实际时间、明确方向的 ratio 和必要失败归因。仅保留支撑结果所需的程序与运行记录；环境、缓存、完整推理轨迹和临时编译输出不是项目交付物。

沿项目习惯记录 `ratio = candidate_ms / reference_ms`，越小越好；展示 speedup 时显式取倒数并命名，不能把 G/S 当吞吐量。正确率和“正确且达到某性能阈值”的比例使用预定集合及重复为固定分母，不能只汇总成功任务。共同成功子集的性能统计独立呈现并标出覆盖范围；多 shape 的任务不能靠多记几行获得额外任务权重。

预算轨迹只更新最好正确候选，错误程序的快时间不能计为有效性能。性能阈值属于可读统计口径，不导入旧 cuTile 的 1.05/1.1 门槛，也不以 Intent 必须全面胜出作为数据有效的前提。

## 8. 当前边界

当前处于 Shape，首轮 50 个不同任务的规模已确定，尚未授权启动实验。brief Q2/Q3 解决后，汇总实际运行 profile、范围、预算、A1–A6 与非目标请用户最终确认，再进入 Build。

本 change 使用 main/current，不新建 worktree，不推送或创建 PR，不使用 ARS，不修改个人 Codex 设置。默认不扩展其它模型、推理档位、GPU 设备、CPU/DSA 后端或完整论文复现实验。
