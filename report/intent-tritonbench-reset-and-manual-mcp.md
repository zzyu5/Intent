# Intent 编译器收束、使用手册 MCP 与 TritonBench 评估重整

2026-09-11。本文记录用户要求停止旧实验方案后的事实、责任边界和后续方案；不是新的语言规格，也不代表下述能力已经实现。语言与 compiler 仍以 [doc/index.md](../doc/index.md) 为权威。

## 1. 收束结论

目标是使用 TritonBench-T 检验 Intent 的表达能力、agent 使用难度和生成性能，不是另建一个多轮 agent 研究框架。顺序为：清理过度扩展的实验 → 修复已暴露的 compiler 共性问题、建立可查询的作者手册 → 用隔离的 Luna 实例完成精简评估。

- 主开发 agent 修 compiler 和作者工具；被评测 agent 只生成候选，不能修改 compiler、任务或 reference。
- Intent 允许作者显式定义多个 kernels，并由 Python host 分配中间 tensor、依次调用；不要求一个 kernel 承担整个算子。完整算子时间包含这些 kernels 和必要中间处理。
- 使用手册 MCP 是必须建设的作者工具，不只是一句“给模型更多文档”；但它只检索、解释已有公开能力，不替 agent 写题解或自动选择执行算法。
- 暂停第二项“生成 Triton 作为后续优化起点”的实验。其研究问题保留，不继续消耗当前工程预算。
- 当前 change `agent-tritonbench-evaluation` 的旧范围已被用户停止。未完成的验收不变成通过，旧 Spec 不发布为新有效要求。

本文最初形成于旧 change 封存阶段；用户现已确认 `intent-agent-readiness` 进入 Build，授权精简旧数据、部署专用环境/手册 MCP 和首批 compiler 修复。

## 2. TritonBench 原本做什么，我们额外加了什么

[论文 §4.3、§5.1](https://arxiv.org/pdf/2502.14752) 的主评测是代码生成后的调用成功率、执行正确率与性能；zero-shot/one-shot 指提示中是否提供检索示例。one-shot 使用 BM25 选取相关示例，不是一次修错。公开的 DeepSeek-R1 普通/RAG 两份 T 输出各有 166 条不同任务、每题一个 `predict`；发布的评测脚本消费这些输出，未提供我们的多轮反馈优化流程。

三次独立重复、最多五次修错、再五次 Triton 优化、阶段预算轨迹、恢复调度和大量统计表，是旧方案额外加入的，不是 TritonBench 的要求。不能把这些工程工作称为复现上游协议。

适合本项目的表述是：**在 TritonBench-T 的明确子集和运行配置上，比较同一模型直接编写 Triton 与编写 Intent 后编译到 Triton。** 当前 50 个任务、每题一个既有 invocation profile 是子集评估，不冒充原论文完整 166 题、全部输入分支或原硬件结果。

复用上游任务、接口和独立参考计算，不照搬存在问题的判断：`ref/tritonbench/EVAL/eval_T/1_exe_acc.py:10` 使用 stdout 比较；`2_efficiency.py:24` 排除极端速度比。项目继续用实际输出、既定容差和完整 CUDA Graph 算子计时；不因很快、很慢或失败而删除任务。

## 3. 已有结果说明什么

### 3.1 不能把旧结果等同于当前 compiler 的最终能力

`refined/environment.json` 的 compiler 为 `e28e8e72`。该首轮 50 题中，直接 Triton 首次正确 43、预算内正确 50；Intent 首次正确 29、预算内正确 45。后续 compiler 修复和人工复测没有回填为 agent 自主成功。重复轮次未补齐，优化阶段已停止。

共同成功的 45 题，静态 Intent/direct 耗时比几何均值约 2.20，但中位数约 1.047，11 题超过 10 倍。15 个逐元素任务整体接近，严重差距集中在部分归约、融合和求解结构。不能据此说“Intent 全面很慢”，也不能只报局部胜例宣称目标成立。旧批次观察统一保存在 [observations.csv](agent-tritonbench/observations.csv)，`batch/stage/candidate` 保留原始含义，不能把第五次提交当成首次生成。

### 3.2 多 kernel 已有实际证据，不是未来设想

旧 `expanded` 的 `exp_mean/intent/repeat-0/generation-5/candidate.py:5` 定义 partial/final 两个 Intent kernels，`:26` 分别编译，`:30` 由 host 串联；同目录 `measurement.json` 为 pass、0.007968 ms。同批直接 Triton 为 0.007840 ms。

这证明该显式多 kernel 路径可以接近直接 Triton；它是第 5 个候选，不是首次生成结果，也不能移作 `refined` 的结果。后者另一个正确的单 kernel 候选为 0.120104 ms。

旧生成阶段“首次正确即停止”，会保留正确但低效的起点。这影响它所测量的内容，却不应因此继续扩大为性能反馈修错框架。新的单次交付评估明确测“当前模型、手册和 compiler 组合能一次交出什么”；开发侧可分析和修复，正式样本不接受人工补写。

### 3.3 首次失败的归因边界

以下按首轮 **21 个首次失败任务的直接阻塞** 分类，基于候选、诊断和规格的静态核对；不是独立 bug 数量，也不是每份程序完整语义/性能的验收。后续候选成功不能证明早期缺陷已消失。

| 归类 | 数量 | 任务及直接原因 |
|---|---:|---|
| 确定实现 bug | 3 | `tensordot` 等价 step 误拒绝；`matmul` 累加 dtype 未正确传递；`fused_qr_solve` 的 Gram 独立轴合并 |
| 合法构造支持缺口/规格不一致 | 3 | `conv2d` safe-gather lowering；`solve` 有序控制中的 parallel construction；`fused_cholesky_solve` 已定义的 Out 内容仍被禁止读取 |
| 作者接口/表达不符合当前公开规则 | 14 | `argmax` tuple 当 index；`mean` wrapper 缺省参数；`log1p`、`asin` 不存在的 API；`logsumexp` Python math 调用；`add_mean` domain 当 axis；`symmetric_mm_and_abs_sum` Python abs 调用；`ifftshift` domain 算术；`gelu_conv2d`、`relu_conv2d` dtype；`grid_sample`、`fused_layer_norm_relu_linear`、`fused_gather_masked_fill`、`permute_copy` shape/index/broadcast 表达 |
| 未定因 | 1 | `min` tuple component 的 KIR 类型不一致 |

证据位置统一为 `report/agent-tritonbench/refined/programs/<task>/intent/repeat-0/generation-1/{candidate.py,measurement.json}`。这 14 项也包含手册可发现性和 API 易用性问题，不能简单推给模型。涉及 indexed relation 的判断只限当前失败表达，不能反向把某种实现限制提升为新语言规则。

例如 `fused_qr_solve` 的 Gram 轴合并是确定的 compiler 阻塞，但该候选使用 normal equations 而非参考的 QR，后续算法/稳定性仍需单独判断；不能由这项阻塞归因推导整份候选修完就会通过。

Out 需要特别纠正：`doc/dsl/core.md:34` 规定进入 kernel 时不可读、读取前必须定义；Cholesky 候选 `candidate.py:35` 的 ordered loop 先写早期行，再读取这些行，而 `python/intent/frontend/lowering/ast/context.py:445` 无条件拒绝所有 Out read。不能教 agent 把一切 Out 都改成 InOut 来掩盖这一缺口。修复仍须保证先定义后读取，不是取消读写检查。

其它已记录但未完成根因确认的异常包括：`gelu_conv2d` 后续 binary schema、`solve` 后续生成 Python 空分支语法、`fused_layer_norm_relu_linear` 非法访存和 `permute_copy` 数值超差。保留相关原始候选，不能在未核实前全部算 compiler bug，或因换一个候选通过就说它们已解决。

## 4. Compiler 攻坚的范围与顺序

先修共同表示和合法构造，再看 provider 形式；不用任务名称、单题模板或 serializer 内算法弥补缺口。

1. **独立轴与 ownership。** 相同 extent/dimension 名不等于相同坐标，同一个 source 在 contraction 两侧也可能产生两个 free-axis occurrences。`RealizePointwiseBlocking.cpp:4398` 创建 dimension-only 参数，`Utilities.cpp:2743` 按 dimension 扩散，可能合并独立轴。应沿当前 operand/result axis relations 修 mapping、blocking、access 和 validity；只添加 source binding 不足以处理 `A.T @ A`。对照 `ref/triton/lib/Dialect/Triton/IR/Ops.cpp:249` 的 operand 0/1 encoding 与 `OpInterfaces.cpp:37` 的位置关系。后果是保留二维输出所有权，而不是放宽 contraction 的独立性检查。
2. **Scalar/rank-0/tuple 的一致 lowering。** 保留语义区分并正确转换，不把所有空 shape 都当一个类型。先定位 `min` 的 tuple 和融合程序的 binary schema；未知根因不预设成大重构。Triton 对 scalar/block 的处理可对照 `ref/triton/python/triton/language/core.py`，具体修复前必须定位同类语义与实现差异。
3. **Access/effect 语义。** 补 Out 已定义内容的合法读取，以及 indexed read 的 safe-coordinate/validity/fill lowering。不能让无效 lane 先产生错误访存，也不能以“当前 tl.gather 不接受”为语言限制。卷积对照 `ref/tilelang/examples/convolution/example_convolution.py:32` 的显式索引/计算结构。
4. **Ordered control 中的合法 parallel。** `solve` 的首次阻塞是 shared construction 未实现；保留 pivot 顺序与唯一行写入，形成合法物理执行。不能引入无同步的跨 program 依赖，不能隐藏拆 kernel。
5. **正确但明显慢的程序。** 检查独立输出的异常标量化、冗余物化、遍历与复用，再检查 target-local form。多 kernel 的算法分解由作者显式表达；同一 kernel 的 physical blocking 由 compiler 决定。两者都要能用，不能互相替代。

已完成且应保留：`7232b125` 的等价 step、dot out_dtype、纯 reduce 输出分块；`31f6aadc`/`95bdc2cb` 的 ordered-loop 独立输出 rank lifting；`e28e8e72` 的可见 SSA launch cardinality。原始 DSL 的 `conv2d_add` 从 1.702560 ms 降至 0.065184 ms 是 compiler 改善证据，不是 agent 自动优化结果。具体双方 file:line 和限制见 [compiler-rechecks/index.json](agent-tritonbench/refined/compiler-rechecks/index.json)。

只用受影响原始程序的既有 production benchmark 确认修复，在同一次运行做一次约定容差检查；不为此次收束新建测试体系。上述顺序是工程优先级，不要求修完整个语言才允许评估，也不设“先全面赢过 Triton”门槛。

## 5. Intent 使用手册 MCP

### 5.1 内容先行，MCP 提供可靠入口

当前 `examples/repro/agent_study/agent.py:28` 只提供六份 DSL/编程模型文档和单 kernel vector-add；未提供 `doc/dsl/examples/`。`split_k_pipeline.py:45` 的 host 部分又是伪代码。提示虽在 `instructions.md:9` 允许多 kernel，但没有等价的可运行教学入口。

手册需要覆盖四类问题：

- **如何写：** definitions/helpers、domain 与 indices、索引/广播/转置、scalar 与 rank-0、tuple/record 解构、dtype/literal、reduce/contract 返回值。
- **如何组合：** 单 kernel、按行归约、显式 partial/final 两阶段、host 中间 tensor 分配与传递；logical 分组不是 warp/tile 参数，允许作者表达算法分解。
- **如何调用：** public compile/artifact/launch 的区分，显式 output 与 `.run()`；实验的 `build(context)` 是薄适配接口，不是 Intent 语言语义。
- **如何读诊断：** 将常见 dtype、axis、shape、effect 诊断链接到规则和最小正例，明确“不存在 API”“语言合法但实现未支持”“尚未验证”是不同情况。

手册是 `doc/` 的作者向说明，不另写一套语义。稳定说明与示例沿现有 DSL/编程模型分层组织；当前实现支持情况单独来自已核实实现事实，不把进度写入语言规格。通用示例必须可运行，不使用 undefined `launch/allocate_tensor` 占位；不能复制评测题的优化解作为“示例”。

### 5.2 最小但完整的查询接口

建议三个只读工具；名称是方案，不表示已注册：

| 工具 | 职责 | 返回的关键内容 |
|---|---|---|
| `intent_manual.search(query, kind)` | 按 API、概念、诊断、通用示例找材料 | 标题、摘要、文档 ID、相关 API |
| `intent_manual.api(name)` | 精确查询公开入口 | 签名、参数、返回 schema、dtype/shape/effect 规则、最小示例、来源 |
| `intent_manual.read(id, section)` | 读取规则或完整通用示例 | 原文/代码、必要上下文、关联章节、源码位置 |

要求：精确符号优先，初始采用全文/BM25 类检索即可；不先建向量数据库、重排模型或自动回答服务。查不到就明确 not-found，不编造 API、题解或性能结论。API 信息复用实际公开声明并关联 `doc/`；不维护第二份手写导出清单。文档与实现不一致时显式暴露冲突，不能以当前代码覆盖规格。

示例返回要包含 imports、完整 kernel/helper 和必要 host wrapper。归约必须说清返回值和 axis，不能只有一段缺少类型上下文的正文。没有 benchmark 证据的示例不宣称性能已验证。

服务首选本地 STDIO，内容限定为发布的手册、API 资料与通用示例。MCP 只承载文档查询，不接受候选代码执行、不调用 compiler、不读取测试 oracle，也不调用另一个 LLM。OpenAI 的 [Docs MCP](https://developers.openai.com/learn/docs-mcp) 同样采用只读 search+文档内容、与模型 API 执行分离的边界；[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) 支持本地 STDIO。

### 5.3 项目归属与维护

MCP 属于作者工具，不进入 `lib/` 的 IR/pass、runtime 的执行语义或 `examples/repro/` 的 benchmark 逻辑。后续实现时在 Python 作者工具职责下建立一个小模块；索引与环境缓存不提交。实验只连接该工具，不另复制一套手册服务。

同一批评估绑定同一 compiler 与手册快照；用 Git revision 和材料来源记录即可，不使用 hash/checksum 校验。手册改动后自然更新该批输入；不能在个别题失败后偷偷补同题提示。检索命中原文由 agent 自己消费，服务不持有跨题答案记忆。

## 6. 独立 Luna Codex 的隔离

复用现有 Codex CLI（已核实 0.154.0），建立项目外专用配置/状态目录与候选工作目录，不安装第二份二进制，不新增开发 worktree。

- **配置和历史：** 专用 Codex state root，不继承开发会话、个人/项目 Comet 记忆、其它题答案、默认 skills/hooks；每题每组新会话。
- **模型连接：** 两组均固定 `gpt-5.6-luna`、`max`，使用用户指定的实验 provider；不修改主开发会话。凭据只通过专用认证环境提供，密钥及认证头不写入项目、prompt、日志或报告。
- **文件权限：** 仅任务、手册工具和自己的候选目录；compiler、reference、历史结果由评测侧持有。独立配置目录本身不是文件权限隔离。
- **工具权限：** 可查手册、编辑候选；不能跑 benchmark 获取试错反馈、修改 compiler/reference 或派生子代理。模型连接与只读 MCP 的网络/进程权限单独管理，不能把 shell 禁网误称为所有通道均被隔离。
- **提交边界：** 一次完整程序交付，可以含多个 kernels。Codex 内部读文档/编辑可能有多个模型回合，不等于多份评测候选，也不应冒称裸模型只调用一次。

provider 的 Responses/流式/工具交互以及该服务的 luna/max 映射尚未实测。连接配置完成不等于运行行为已验证，不因为 URL 以 `/v1` 结尾就假定兼容，不静默换模型或加代理转换层。参考 [Codex 配置](https://learn.chatgpt.com/docs/config-file/config-advanced) 与 [状态目录说明](https://learn.chatgpt.com/docs/config-file/environment-variables)。

## 7. 精简评估与数据清理

### 7.1 新评估只回答必要问题

沿当前 50 题范围，同一模型/任务分别交付直接 Triton 和 Intent，各一份；提供相应语言资料与同等可用的文档检索方式。Intent 经正式 compiler；提交后由 production benchmark 记录可调用、容差内正确与完整算子时间。失败留在分母，性能只对数值通过的程序报告。

报告“有文档工具支持的单次交付”，不是无工具 zero-shot，也不是原论文 one-shot 的原样复现。模型能力、资料与接口支持属于本实验条件，不宣称只测到了 DSL 抽象本身的因果效果。50 题范围和 profile 不因结果不好而替换；上游有歧义的任务/reference 先明确处理，不通过修改容差或丢弃输出掩盖。

编译、JIT、正常有限 tuning、预热属于获得性能所需步骤，不取消；它们不是算子时间。准备可并发、同设备计时避免重叠。外部分配不计入 GPU 算子时间，跨 kernel 中间计算/转换/物化计入。不再新增阶段预算轨迹、自动五轮优化或独立三次重复。

统一结果保留 task/profile、两组状态与算子 ms、reference ms、含义明确的 ratio、简短失败说明；必要源码和原始 measurement 支撑这些行。按现有项目习惯用 generated/reference、越小越好，若展示上游 speedup 则明确取倒数。开发修复复测与正式 agent 结果不能互相替代，但无需为此建多套表格。

### 7.2 清理按职责，不按结果好坏

盘点时 `report/agent-tritonbench` 有 3,274 个文件，其中优化候选目录内 1,494 个；Git 跟踪 2,908 个。问题是把过程日志、多轮候选和多份可派生表格变成长期维护对象，而非磁盘容量不足。

后续清理保留任务/参考薄适配、生产 benchmark、有效 compiler/runtime 修复、一份必要结果数据与少量直接支撑的源码/测量。删除旧多轮调度/恢复/预算发布逻辑、重复汇总、无用 agent 活动记录与被取代的候选副本。原有失败及其原因不从分母消失；尚未解决缺口所需的独特原始候选保留。

已提交冗余由 Git 历史留痕，不再创建另一个归档包或把维护负担转移到 `/tmp`。未跟踪文件先判断是否含唯一结果/复现，再删除。清理前后核对引用，不能留下指向已删程序的“证据”。不整批 revert 混有 compiler 修复的提交，不碰无关 CPU/worktree、外部 ref 或个人设置。

本文是用户明确要求的单份复盘报告；不为本报告再衍生计划、进度、核对清单和多份统计文件。

## 8. 旧 change 的处置与当前未完成项

用户要求停止旧 `agent-tritonbench-evaluation` 并准备重新设计，旧三次重复、生成与优化各至多五次提交的范围不再执行。实际后台实验已停止。

封存前 Native 归档预检返回 `ready=false`：处于 Build，0/6 验收通过且缺少 `verification.md`。公开 CLI 和文档只提供验收完成式 Archive，没有未完成 change 的 cancel/abandon 入口。不能补造 verification 或把旧 Spec 发布到主规格以满足工具。

用户随后明确回复“是的，允许”，授权按“废止方案”手动封存。三个正式文件已从 active 目录移至 [docs/comet/archive/2026-09-11-agent-tritonbench-evaluation](../docs/comet/archive/2026-09-11-agent-tritonbench-evaluation/brief.md)。这不是 Native 的验收完成事务：`comet-state.yaml` 原样保留 Build、pending、archived=false 的历史快照，brief/Spec 明确标记废止；未生成验收报告，未发布旧 Spec，未修改 Runtime 机器文件。

主工作区封存后，Comet 曾从 `.worktrees/cpu-region-programs` 发现继承的同名旧副本，显示 `bindingState=mismatch`。用户随后明确允许清理旧任务残留；确认副本无自身修改后，删除了该副本的三个文档，单独提交 `e628976c`，CPU 实现和 worktree 均保留。原文可由 Git 历史和主目录归档恢复。再次查询 Native 的旧 active 列表为 0。

按用户要求，新建 [intent-agent-readiness](../docs/comet/changes/intent-agent-readiness/brief.md)，绑定 main/current，用户已确认进入 Build。其第一批为旧产物收口、手册 MCP、专用 Luna 环境及 ownership/schema 修复；只用一个既有任务接通链路，不把 50 题正式实验、第二阶段优化或全部 compiler backlog 塞入本轮验收。其它根因仍保留在本文，未宣称解决。

旧数据已收口为一份 811 行观察 CSV，保留 refined 生成程序及其原测量、早期多 kernel 实例和必要 compiler 复测；删除 2,839 个重复表格、逐轮活动/停止日志、优化副本和 Python 缓存。已知 evaluator 修复仅对原来对应的失败行应用，`original_status` 保留；未重跑或补造任何数值。CSV 的 `original_program` 是原始位置，`retained_program` 非空才表示当前仍保留的源码；其它已提交材料由 Git 历史保存，不另建归档包。专用环境/MCP 与 compiler 首批修复仍在实现中。
