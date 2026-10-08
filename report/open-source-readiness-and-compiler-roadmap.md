# IntentDSL 成熟化路线图：性能量级、Pass 资产与多后端产品

更新：2026-10-08。依据当前 main（0f7f193c）、正式规格、本地成熟实现、既定 30 个产品 program 和已有运行产物原位更新。本轮只调查和修改路线图，没有重新编译、运行或测量。已完成工作从待办移除；历史成绩与当前实现分开说明。本文安排产品演进，不修改 doc/ 的语义合同。

## 1. 产品目标与当前判断

**现有编译器骨架可以继续发展成产品，接下来需要同时解决“结构是否正确形成”和“形成的程序是否有合理性能”。** 只证明比自己的旧版本快，不足以说明性能已经合理；没有统一硬 baseline，也不等于没有外部尺度。

产品的三个入口是：作者算法 DSL、开发者可组合的 pass、使用者可安装并真实调用的程序。作者负责算法、数值、状态与显式 kernel/host 编排；compiler 在这些语义下形成适合执行模型的物理程序。优化知识通过分析、合法性证明、成本判断和 IR 改写沉淀，不能只存在于某个例子的补丁或 serializer 中。

性能目标是：在维护的目标和适用算法范围内，形成有解释的、接近同类成熟实现合理水平的程序；明显偏慢的程序必须有具体成本归因和解决任务。某些 DSL 组织能够减少中间物化、重复供数或调用次数，超过已有实现也完全合理。外部实现是校准工具，不是性能上限，更不是要求作者换算法的理由。

当前已有真实的 GPU、CPU、DSA 物理 IR、分析、pass、目标实现、配置和运行入口。还没有证据要求推倒骨架、重造调优器或优化框架。剩余工作集中在条件物理程序的配置/资源一致性、跨消费者的物理程序质量、性能量级校准，以及 BANG C 的真实设备闭环。

v1 不依赖 agent 服务。实验性的“激进生成器 + agent 落地”保留为未来方向，不用它掩盖确定性 lowering 缺失。工具链和 ABI 适配仍属于目标实现与 runtime；性能轮次以分析、pass 和目标消费为主要修改范围。

## 2. 已完成的地基与当前覆盖

| 已完成部分 | 当前事实及边界 |
|---|---|
| 标准 pass、阶段合同和独立 IR 工具 | [GPU Passes.td](../include/Intent/Dialect/GPU/Transforms/Passes.td)、[Registration](../lib/Compiler/Registration.cpp) 已连接标准注册、pipeline、intent-opt、optimize_ir 和 generate_from_ir。必要 lowering 与可选优化分责，不再建设第二套编排框架。 |
| GPU/CPU/DSA 的职责组织 | GPU cleanup 与 hoist 已分开；CPU 已有 task/block、供数、存储与实现选择；BANG C 已有 LocalComposition、StorageReuse、SupplySynchronization、StorageBinding。目录和组件拆分已经有真实消费者，不再把整理文件本身列为里程碑。 |
| 完成值与版本知识 | GPU [FragmentSnapshot](../include/Intent/Dialect/GPU/Transforms/Storage/FragmentSnapshot.h)、CPU [ProducerVersions](../include/Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h)、BANG C [CompletedSupply](../lib/Target/BangC/Transforms/CompletedSupply.cpp) 已形成各执行模型的复用路径。 |
| 产品入口与完整配置 | [examples/use.py](../examples/use.py)、[examples/programs](../examples/programs)、[ConfigurationProfiles](../lib/Dialect/GPU/Transforms/Configuration/ConfigurationProfiles.cpp) 已有完整 program、host 编排、prepared 调用和完整经验行。Triton 原生 Config/autotune 及其它后端的候选选择已经存在。 |
| GPU 遍历与访问合法性 | [IterationDependencies](../include/Intent/Dialect/GPU/Analysis/IterationDependencies.h)、ownership 容量、TMA 对齐和快照消费已经贯通；证明依据当前 IR，而非算子名。 |
| 普通 prefix 消费的流式组织 | 原 Nonzero 的逐元素串行消费问题已解决。独立效果、typed count-prefix 活跃写入、无溢出、别名和读取 epoch 共同证明；不把这种证明推广成任意浮点 scan 可重排。 |
| 归约 summary 与输出消费者共同组织 | [ReductionConsumers](../lib/Dialect/GPU/Analysis/ReductionConsumers.cpp)、[RealizeReductionConsumers](../lib/Dialect/GPU/Transforms/Reduction/RealizeReductionConsumers.cpp) 已连接共享 SSA 工作集分析、完整/有界遍历和紧凑 summary。旧 W2 的核心不再列为未实现；跨 region 和更多消费者组合仍需推进。 |

已有 canonical 产品记录中，Triton、H100 cuTile、Mojo、Weft 各有 30 个 executed program、35 个 artifacts；BANG C 有 30 个 generated program、35 个 artifacts。这是已有记录覆盖，不是本轮重新执行，也不是任意 shape、dtype 或硬件均已覆盖。BANG C 的 source 覆盖不能代替 CNCC、CNRT、设备数值和完整调用性能。

近两轮已有真实收益：同一轮 prefix 改写的 Nonzero 完整调用，Triton 241.008 → 17.472 μs，cuTile 153.024 → 18.752 μs；它与更早快照阶段的结果分别保留，不混成一次配对比较。最近 GroupNorm backward 在 cuTile/H100 上为 53.952 → 45.056 μs，约降低 16.5%；Triton 最终 66.272 → 65.888 μs，native 指令和资源没有变化，只能认为持平。初版 Triton 88.096 μs 的回退促成成本策略修正，不能只报告最终局部收益。

这些证据说明 pass 能产生跨程序、跨 provider 的真实作用，但还不能证明全部 30 个程序达到成熟性能水平。已经解决“有无真正编译器骨架”的问题；正在解决“优化覆盖是否广、程序质量是否好”的问题。

论文 experiments 保留历史材料。产品唯一算法入口是 examples/kernels，完整调用入口是既定 examples/programs 和 examples/use.py。后续产品推进不改写、扩建论文实验；产物、缓存和运行观察在仓库外，沿用现有结果输出原位更新。

## 3. 性能如何校准：没有硬 baseline，仍有清楚尺度

### 3.1 三类证据一起使用

1. **自身变化**：同一 program、输入、设备和调用口径下观察 pass 前后，说明改写是否实际受益、是否回退。它证明进步，不证明起点或终点已经合理。
2. **同类成熟实现**：优先使用已有历史 reference、外部本地源码和成熟库的经验，找相近算法、规模、精度和输出职责。重要差距需要时才通过既有入口观察，不为每个 program 建一套外部 baseline。
3. **物理成本尺度**：逻辑读写量、实际重复访问、有效计算量、准备/同步、存活工作集、并行度和完整调用次数。用它判断为何偏慢、该形成什么 pass，而不是只看比例。

对照有三个强度：同算法同口径可直接比较；相近算法但数值/布局/辅助输出有差异可判断合理区间；设备或规模不同只能校准数量级。差异明确标注，不把所有差异都当成禁止观察性能的理由，也不把弱对照包装成精确倍率。

例如，LayerNorm 不同方差公式仍能帮助判断同类归约和内存访问的成本，但不能直接宣称完全等价。ordered recurrent 更新与 chunked 算法的差距能说明算法上限不同，不能据此让 compiler 默默改写作者的算法。量化、转置、workspace 初始化、部分结果合并等若属于完整 callable 的职责，就计入并说明。

### 3.2 性能状态与优先级

每个算法族需要有清楚判断：

| 状态 | 含义与后续动作 |
|---|---|
| 明显结构性低效 | 重复完整物化、逐元素串行读取、缺失 native primitive、双份长期存活、准备或同步过多。直接安排共同分析/pass 任务。 |
| 有显著质量差距 | 可比或相近成熟实现持续明显更快，差距无法由算法和调用职责解释。优先检查物理结构、下层生成和测量口径。 |
| 位于合理区间 | 完整调用与同类实现接近，或有效吞吐与该规模的并行度相符。保留现有结构，转向其它长尾。 |
| 有可解释优势 | DSL 显式组织或 pass 减少真实工作，收益在完整调用中出现。说明原因及适用条件，成为可复用知识。 |
| 尚未校准 | 只有生成、运行正确或自身计时，缺少可信的外部尺度/成本解释。明确记录缺口，不能直接称性能成熟。 |

强可比条件下持续超过约 2 倍的差距，是优先调查的信号；这不是硬性合格线或编译规则。更小但覆盖广的差距也值得解决；5 倍、10 倍不能用“同量级”含糊过去。参考实现本身明显低效时不能拿它宣布胜利，合理的 DSL 优势也不需要人为压回参考水平。

有效带宽按必要逻辑读写量除以时间，有效矩阵吞吐按实际算法计算量除以时间；两者不是硬件计数器测到的 DRAM 流量或全部机器指令吞吐。理论下界帮助识别冗余工作，但小程序、依赖链、cache 和调用成本会影响可达到的水平。[NVIDIA 有效带宽说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation)提供了这种读写量与时间的计算口径。

### 3.3 当前测量的盲区必须先分清

[use.py](../examples/use.py):71–106 的 GPU 完整 prepared 调用，以 CUDA events 包围 Python enqueue。它排除 InOut reset 和下载，包含作者 program 的 workspace 初始化；设备事件之间仍可能存在 host 提交造成的设备空档。旧实验的 [support.py](../experiments/_common/support.py):48–85 可选择 CUDA Graph，且每次 GPU sample 前显式执行 L2 flush，不能把两种结果直接算作 pass 倍率。

当前许多小程序完整调用约 16–20 μs。ReLU 只有 64×1024 个 F16 元素，必要读写共 256 KiB；GEMM 是 M128、N192、K256，约 12.58 MFLOP。这类规模先区分调用底噪、设备利用率不足和 kernel 内部低效。不能按大型 GEMM 峰值要求小 GEMM，也不能看到微秒级就认为小程序已经优秀。

后续仍报告真实完整调用；分析时区分已有 native 结构与调用成本。若确认瓶颈属于产品提交/绑定路径，就列为独立产品任务；不能在一次性能 pass 轮次里顺手改 ABI、runtime 或计时方式制造提速。未经分解的 16–20 μs 目前只是观察，不是已经证实的 Python 成本。

### 3.4 已有历史材料提供的尺度

以下仅从现有 H100 CSV 的 source 时间和算法工作量换算，未重新测量，也不代表当前 HEAD。它们说明某类规模下成熟实现大致能到哪里，不是新的 baseline 表或 v1 数字门槛。

| 历史成熟实现与输入 | 已有完整时间 | 逻辑工作量换算 |
|---|---:|---:|
| Softmax，8192×8192，F16 | 119.608 μs | 按一次读入、一次写出约 2.24 TB/s |
| SwiGLU，8192×14336，BF16 | 239.104 μs | 按两次读入、一次写出约 2.95 TB/s |
| ReLU，8192×4096，F16 | 49.624 μs | 按一次读入、一次写出约 2.70 TB/s |
| GEMM，M4096/N14336/K4096，F16 | 1.451800 ms | 按 2MNK 约 331 TFLOP/s |

来源：[triton-h100.csv](../experiments/gpu/results/triton-h100.csv):2、3、10；[cutile-h100.csv](../experiments/gpu/results/cutile-h100.csv):23。逻辑带宽不等于实际 DRAM 带宽，不把大尺寸成绩按元素数线性缩放到当前小输入。

GroupNorm 的旧 reference 还有 CUDA Graph 口径差异；MoE alignment 的旧 reference 有 workspace 初始化职责差异。这类材料仍可提示方向，但当前完整调用是否达标需要结合这些差异判断。不同设备的 Triton/cuTile 产品时间也不能直接排列 provider 优劣。

## 4. 以 30 个产品程序形成算法族性能画像

分组用于产品观察和推进，不进入 compiler policy，不允许按 program 名选择特判。

| 算法族 | 既定程序 | 主要尺度与共同问题 |
|---|---|---|
| 内存、pointwise、布局 | relu、swiglu、rope、transpose、dropout、adamw | 必要字节量、连续访问、额外 materialization、多输出和完整调用成本；小输入先看调用底噪。 |
| Collective 与统计 | softmax、layer_norm、batch_norm、group_norm_backward、cumsum、histogram | native reduce/scan、summary carry、工作集、输入重读、输出遍历及多 consumer；不能只看 nominal words 降低。 |
| Contraction 与量化 | gemm、batched_gemm、ragged_gemm、int8_gemm、fp8_gemm、q4_projection | 有效 FLOP、tile 利用率、native matrix 构造、供数/量化成本、初始化和 epilogue。 |
| Attention、recurrence、结构化计算 | attention、paged_decode、causal_linear_attention、mamba_chunk_state、gated_delta、causal_convolution、cholesky | 作者算法的真实工作量、依赖链、状态 carry、版本准备、contraction 衔接、部分结果合并和多 kernel 调用。 |
| 不规则访问与写入 | embedding、csr_spmv、jagged_mean、nonzero、moe_alignment | 有效成员、间接访问、负载分布、活跃写入唯一性、scan/scatter 消费、清零和辅助输出成本。 |

近期重点不是重新围着 GEMM 或 GroupNorm 打转：先补齐 collective、不规则和 recurrent 长尾的画像，找覆盖多个程序的共同结构问题。矩阵族需要判断 native 路径和准备成本；已经合理的简单程序不反复追几个百分点。

有些机会来自作者更好的显式 DSL 组织，如原生分块算法、减少跨 kernel 中间物化、共享可复用准备和更合适的 state 更新。区分两种情况：同语义物理冗余由 compiler/pass 消除；改变算法、数值顺序或 host 编排的机会说明给作者，由作者选择。未来新算法仍走同一入口，不增加一套“高性能专用解法库”。

## 5. 接下来完成的里程碑

### M0：条件物理程序的配置与资源义务一致化

这是有限的基础收口任务，独立于性能 pass。目标是优化产生的合法分支不会错误改变整份经验配置，资源约束也不会错误混合互斥执行路径。

当前明确缺口：

- [ConfigurationProfiles.cpp](../lib/Dialect/GPU/Transforms/Configuration/ConfigurationProfiles.cpp):32 的 familyFor 从整个 kernel 的参数选择最高优先级 family；分支新增的 R/P 参数可能把原完整遍历的配置归类改变。
- [Resources.cpp](../lib/Dialect/GPU/Analysis/Resources.cpp):740 收集的归约义务没有对应控制域；[ConfigurationConstraints.cpp](../lib/Dialect/GPU/Transforms/Configuration/ConfigurationConstraints.cpp):37 将义务共同约束。[UniformBranches.cpp](../lib/Dialect/GPU/Transforms/Mapping/UniformBranches.cpp):109 因缺少相应关联，保留了限制。
- 现有 requirement.activation 只允许声明的 provider boolean，[PhysicalParameters.cpp](../lib/Dialect/GPU/Analysis/PhysicalParameters.cpp):116–122；不能把任意 scf.if 条件塞入它。
- 现有 PhysicalExpr Select 的 C++/Python checked evaluator 会先计算两臂；未执行臂的缺值或溢出可能使整体 Unknown。launch 条件提取也只支持有限关系。不能只给收集器套一个 Select 就宣称问题闭合。

连贯修改范围：

1. 从当前 SCF、def-use 和绑定阶段查询条件与可达性。区分 native specialization 可消去的分支、uniform runtime 分支和未知路径；uniform 不等于 constexpr。
2. 同一逻辑轴和归约结果的完整/分块表示保持同一计算域的配置分类；新增分支局部参数不自动升级整个 kernel family。真正不同计算域仍使用既有明确策略，不自动合并 family。
3. 继续从一个既有 TuningProfileTableAttr 投影完整经验行，保留参数、warps/stages 等关联。每条原经验行最多产生一个候选；不增加行、补端点、生成轴组合、拼接分支候选或额外搜索。R=1 仍表示真实粒度 1。
4. 资源峰值和义务依据实际存活及可达路径。已证明消去的分支不误过滤；两臂都保留的 native 必须承担对应义务。Unknown 保持未知，不能当 false 放行。
5. 候选过滤、静态 invocation specialization、IR verifier 和 runtime 对声明规则的求值使用一致事实。复用 Select 时统一条件判定、未选臂的求值和验证规则，不预设全局改成惰性；若涉及正式求值合同变化，先明确并确认设计。不伪造 provider 参数。
6. serializer 只拼写已决定的行和表达式；runtime 绑定事实、派生参数并执行规则，不重新猜 family、算法或控制流。

优先复用已有 typed carrier 和临时分析。若确实缺少必须持久化的执行事实，先说明现有表示为何不足及哪些消费者需要它，作为独立基础修改处理；不能在 pass 外维护私有 schema/plan。涉及正式合同变化时另行确认设计，不能为保住现状反改规格。

完成后的行为是：同一计算域的条件结构保留正确的完整经验行；被消去的路径不误拒绝候选；保留路径不遗漏义务；正常 pipeline、独立 IR 入口和实际绑定一致。只用受影响的原产品调用确认，不扩大矩阵。跨 region 的消费者重写属于 M2，不夹带进 M0。

### M1：建立能指导优化的算法族性能画像

这是本轮提出的新主线。目标是知道哪些程序已经合理、哪些明显偏慢、哪些尺度还未知，以及下一包 pass 为什么值得做。

- 为原 30 个 program 整理现有完整调用、设备、dtype、规模、输出职责和 native 结构，各维护后端分别判断性能状态；信息沿用现有产物和输出，不建立第二套结果表。不同设备、时间和采样次数的 CPU 记录不直接排列 provider 优劣，BANG C 尚无 device 结果的项明确区分。
- 每个算法族选择少量可信尺度：已有论文 reference、外部本地成熟实现或合理工作量模型。优先解释重要长尾，不要求 30 个新的硬对手。
- 对差距标注比较强度和具体成本：调用/准备、串行遍历、重复读取、双份存活、缺失 native 构造、供数、同步或下层生成。缺乏证据时写未知，不凭 backend 名称归因。
- 明确小输入的调用底噪与设备程序成本。若需要真实观察，继续使用原 program、原输入、原容差和既有入口；不改计时口径后直接与旧结果计算提速。
- 输出下一批共同知识任务及适用程序群，而不是按 CSV 最慢的一行反复修复。

此步骤只做必要观察，不是反复试配置的反馈循环。画像服务于分析与 pass 设计；物理结构清楚后再运行确认收益。正确但明显差的性能仍要处理，不能以没有硬 baseline 为由结束。

### M2：Collective、间接访问与状态消费者的共同物理组织

目标是 producer、summary、consumer 在当前语义下形成合理遍历、存活和供数，扩大已经成立的 prefix/归约知识，而非再造单项优化。

- 从同 block 扩到可证明的 region/yield/live-out 组合，闭合路径、读取 epoch、别名、effect、执行次数和 lifetime。未知关系保留原合法结构，不把分析事实直接当 replay permission。
- 对多个 consumer 一起判断完整表示、紧凑 summary、有界输出遍历和准备成本；避免优化了一个 consumer，却让其它 consumer 继续保留原大表示。
- 对间接读取与写入，复用稳定版本、活跃成员和坐标关系证明。整数 count-prefix 的唯一性不是通用 scan 单调性；ordered state 的依赖必须保持。
- 成本考虑整个活跃阶段的 SSA 工作集、重复 collective 和供数、carry 以及可利用并行度。nominal footprint 是决策信息，不等于寄存器或 shared allocation；native 结果检验实际采用质量。
- Triton/cuTile 已有的 reduce、scan、布局和线程通信直接消费；不在 shared IR 复制 provider 的树形归约或机器 layout。
- 输出多个程序可消费的分析与变换，标准 pass 可独立启闭可选采用策略。关闭优化仍合法；关系闭合、bufferization 和 legalization 继续是必要阶段。

由 M1 决定优先覆盖哪些 collective、不规则或 recurrent 长尾。可复用性以共同事实和多个消费者成立，不能用 operator name、字符串标签或一个特殊尺寸证明。

### M3：Contraction、量化与 state 的供数—初始化—输出知识

目标是使 contraction 相邻结构得到高质量 native 程序，而不仅是单个 dot 已生成。

- 复用已有 free/reduction blocking、能力约束和完整经验行，统一 start、extent、grid、fragment 及目标 form 的消费。
- 同一完成值在多个 contraction、窗口或迭代中，判断完整准备一次、按窗口准备和直接消费的总成本。区分 source 存储、准备表示和 accumulator 类型，按所选实现需求及总体流量判断转换位置，保持原数值合同、layout、初始化范围和最后使用。
- 区分 accumulator 和下一次 contraction 输入，明确 carry 的表示与转换位置；避免状态每轮重复写回或双份长期存活，也不改变作者 recurrence。
- 初始化、量化/scale、descriptor、搬运和 epilogue 纳入完整成本。Q4 的当前零权重生产输入只说明该输入已运行，不冒充量化路径的普遍正确性或竞争力。
- 原生 dot/scaled-dot、MMA、descriptor 和 pipeline 交下层；真实缺少的供数事实在所属 IR/目标 pass 解决，API 拼写差异不自动扩 shared IR。
- 小 GEMM 的调用/利用率与大 GEMM 的计算吞吐分别判断，不能只挑易接近峰值的矩阵证明产品性能。

允许专家 native micro-kernel 在明确 typed 条件下作为目标实现；leaf 不能接管缺失的算法、整段控制流或算子知识。收益应来自 reusable supply/consumer/capability 规律。

### M4：同一知识在 CPU 与 DSA 上形成适合模型的实现

GPU 和 CPU/DSA 不必共用一个改写函数；复用的是完成版本、工作量、存活、消费者和能力这些知识。各后端用自己的物理 IR 和目标构造消费，不硬套 GPU topology。

| 知识包 | CPU/Mojo/Weft | DSA/BANG C |
|---|---|---|
| 完成版本的准备粒度与共享寿命 | 使用 ProducerVersions、ReusePreparedInputs、task/cohort、packed view 和 SIMD；比较完整准备、窗口准备和直接读取。Weft 按已有结构化 panel/SSA 和 native 能力消费。 | completed supply、transfer completion、首次写入、final binding 与 NRAM/WRAM 生存期共同判断；不能只因指针相同就认为版本相同。 |
| producer—summary—output 的共同工作集 | 使用已有 ReductionSupply、TraversalFusion、scratch budget 和向量 carry；不恢复不适合模型的长跨行 SIMD carry。 | 利用已有局部组合和 native collective，减少重复 transfer、准备及过长存活；保持作者 effect 和输出职责。 |
| 能力约束下的 native 组合 | 使用已有 implementation registry、Weft i8/RVV/矩阵能力及 retained LHS 窗口。区分外部存储、准备表示与 accumulator 类型，按实现需求和总成本决定转换位置，保持 source 数值合同。 | 已有双缓冲与 supply pipeline，推进真实 transfer—compute 关键路径、完成关系和 buffer lifetime，不另写一个空泛“加双缓冲”任务。 |

CPU 的现有 30 program 运行覆盖可以支持下一轮性能画像，但不等于各类 consumer 已高效。CPU 校准同时看 cache/内存流量、SIMD/矩阵有效工作以及 task/worker 利用率；DSA 看 GDRAM 与 NRAM/WRAM 之间的传输、同步及 compute 关键路径，不能只套 GPU FLOP/带宽尺度。必要时修 Weft 主线已有能力缺口；不能要求作者迁就合法表达的 lowering 失败。

BANG C 先完成当下 CNCC → CNRT → 原 30 program 的真实设备闭环，再判断性能长尾。工具链、设备状态、数值、完整调用分开报告；source 结构减少工作只能报告为结构改善，不能先宣称设备提速。配置中的 local_bytes 是预算，不自动等于硬件物理容量。

CPU/MLU 不同设备上的工作可独立推进，无需等待全部 GPU 优化完成；同机性能计时避免干扰。M4 不意味着每次 GPU 修改都重跑所有后端。

### M5：收束可维护的 v1 产品

目标是安装、调用、扩展和维护具有一致路径，优化资产能继续增加，而非宣布未来所有 SDK/硬件都已解决。

- 确认维护范围内的安装/工具链、冷编译、prepared 调用、独立 IR 续生成和必要诊断可用，修实际产品阻碍，不重建已有 JIT/MCP/调优器。
- 新开发者可沿当前目录找到算法语义、分析、pass 合同、IR 决策、目标消费和运行绑定。新增 pass 使用标准注册和 pipeline；共享分析在 mutation 后重建，不能复用失效证明。
- 工具链 API/ABI 变化集中于目标 legalization、adapter 和 binding；算法和性能知识尽量保留。能力或数值语义发生变化时明确处理，不能以默认值或回退假装支持。
- M1 若证明小程序调用路径有显著冗余，作为独立产品改动解决，并保持可解释的完整调用口径；不混入纯性能 pass 的收益。
- 对每个维护 provider 清楚区分生成、native 编译、设备运行、数值和性能范围。BANG C 的设备完成情况单独列明。
- v1 收束时，重要算法族应有性能尺度和主要长尾归因，且具备广泛消费的 pass 资产。无需宣称所有程序领先，也不能只有“编译成功”和几个快例子。

## 6. Pass 如何成为资产，而非复合补丁

一个优化组件应能回答：读取什么事实、合法性由谁证明、决策写入什么 IR、哪些目标消费、关闭时保留什么合法程序，以及在哪些结构上值得采用。这些职责可以由分析、builder、策略 pass 和目标 pass 分担，不要求一个函数承担全部。

| 可复用资产 | 稳定知识 | 模型相关消费 |
|---|---|---|
| Producer/version 与 effect 分析 | 当前完成版本、读取 epoch、别名、稳定性、首次写入及最后使用 | GPU fragment snapshot；CPU packed/prepared input；DSA completed transfer。 |
| Collective consumer 与工作集分析 | def-use closure、summary/live-out、执行次数、重复工作与活跃峰值 | GPU full/bounded traversal；CPU task/vector carry；DSA local summary/supply。 |
| 坐标和遍历依赖证明 | 活跃成员、独立效果、坐标关系、容量和有效范围 | GPU ownership/gather/scatter；CPU task partition；DSA block/transfer。 |
| 能力与完整配置合同 | 合法 dtype/axis/form、真实粒度、相关经验参数和资源义务 | 各 provider 的原生配置、实现选择和资源消费。 |

已有 carrier、公共分析和构造优先复用。分析是对当前 IR 的查询，不是永久 side schema；决定性能结构的事实写入相应 IR，serializer 只拼写。共享思想不等于硬共用所有源码，也不意味着放一个公共 helper 就已复用。

正常性能轮次可以修改分析、优化 pass、目标消费和必要 pipeline 接线，这是编译器优化的正常范围。若必须改 frontend、公共 ABI、runtime 或通用 carrier，要指出当前缺失的真实事实，先完成独立基础任务，再进入性能轮次。这样逐渐实现“基础稳定，性能知识持续增加”，而非要求所有未来优化只能编辑一个名为 Pass.cpp 的文件。

## 7. 成熟实现对照与当前差距

| 问题 | 本地成熟实现 | Intent 当前事实与行动 |
|---|---|---|
| 条件程序与 specialization | [Triton code_generator](../../ref/triton/python/triton/compiler/code_generator.py):1026–1055 区分 constexpr 分支与生成 SCF 的 tensor 条件；[TileLang simplify](../../ref/tilelang/src/transform/simplify.cc):454–465 只在条件可证明时消去分支。 | M0 区分绑定阶段、可达路径和义务。uniform runtime 条件不自动成为 specialization；当前 family/资源收集仍缺完整关联。 |
| 完整配置与剪枝 | [Triton autotuner](../../ref/triton/python/triton/runtime/autotuner.py):284–313 对既有 configs 剪枝，:328–380 保留 kwargs、warps、stages 的关联。 | 继续既有完整经验行；不生成分支候选乘积。剪枝依据实际合法性，不能让已证明在 native 特化中消去的臂误过滤。 |
| nominal 与机器资源 | [Triton compiler](../../ref/triton/python/triton/compiler/compiler.py):469–484 消费 native binary/shared metadata；[fused softmax](../../ref/triton/python/tutorials/02-fused-softmax.py):142 起利用 native 资源判断驻留。 | shared working set 是结构决策模型，native allocation 由下层决定。减少 nominal carry 不保证寄存器或完整调用改善。 |
| Collective 与内存成本 | [Triton fused softmax](../../ref/triton/python/tutorials/02-fused-softmax.py):42 说明减少中间读写；[layer norm](../../ref/triton/python/tutorials/05-layer-norm.py):198 起形成块内累积和后续归约。 | M2 扩共同消费者，不重建 native collective；比较公式差异但仍校准同类成本。 |
| Contraction 组织 | [Triton matmul](../../ref/triton/python/tutorials/03-matrix-multiplication.py):223、252 的经验配置和分组遍历；[block-scaled matmul](../../ref/triton/python/tutorials/10-block-scaled-matmul.py):258 起消费 scale/descriptor/native 构造。 | M3 优化供数与相邻生命周期，消费原生能力；不靠新增候选数量代替物理程序质量。 |
| CPU 准备与计算边界 | [MAX packing](../../ref/modular/max/kernels/src/linalg/packing.mojo):193、1119，及 [accumulate](../../ref/modular/max/kernels/src/linalg/accumulate.mojo):122。 | MAX 提供保留存储类型、消费 packed view 的参照；Intent 的 [ImplementationInputs](../lib/Dialect/CPU/Transforms/Implementation/ImplementationInputs.cpp):121、134 和 [InputCopies](../lib/Dialect/CPU/Transforms/Implementation/InputCopies.cpp):31–38 也支持按实现需求准备及转换。M4 比较两种位置的总成本，不一律禁止预转换，不照搬 GPU carry。 |
| Pipeline 与存储寿命 | [TileLang pipeline planning](../../ref/tilelang/src/transform/pipeline_planning.cc):406、702；[storage rewrite](../../ref/tilelang/src/transform/storage_rewrite.cc):1947。 | DSA 已有对应机制，下一步检查真实完成、关键路径和存活，不重复造框架。 |

公开说明用于交叉核对：[Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)、[Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。关键职责以上述本地源码为依据；参考的 surface、precision 和调用职责不能凭框架名称视为相同。

main GPU 产品只维护 Triton/cuTile；TileLang 后端继续保存在 archive/tilelang-backend，外部 TileLang 仅作为机制参照，不恢复到主线。

## 8. 推进顺序与每轮产出

**先完成 M0，同时利用已有材料准备 M1；随后根据 M1 的质量差距推进 M2/M3，M4 在独立设备上并行，M5 收束产品。** 不预先决定每轮只优化一个固定算子，也不要求全部后端全量重测后才能继续。

一个里程碑是一组连贯任务，解决一类横向成本：事实/证明、采用策略、IR 改写、多个消费者、目标消费及重复旧路径删除。几百行补丁、一个 helper、更多验证数量或漂亮报告本身都不代表完成；代码量也不能代替真实作用。

运行只使用必要的既有生产 program/benchmark，保持原输入和原容差，同次确认一次数值。完整调用是结果，native 和资源是解释；编译/JIT/tuning 不混入算子时间。不建新的 test/fixture、边界矩阵、30 个硬 baseline 或平行结果表，不改写论文实验。

同配置 native 未变时，不把完整调用噪声归因给 pass；资源超限、编译失败、数值失败、持平和实际提速分别说明。遇到显著差距先查物理结构，再查下层和测量，不以“可能是 provider”代替调查，也不为抹掉回退反复择优测量。

下一阶段的实际目标是：性能尺度清楚，重要长尾有共同知识任务，已有基础缺口有限收口，多后端持续消费相同思想。骨架已经成立，继续推进的价值应体现在程序质量和可复用优化资产上。
