# IntentDSL 成熟编译器与产品化路线图

调研日期：2026-10-02。当前实现基线：main / 7769c7ec。本文在原报告上完整重写，结算旧工作包，吸收公共接口、算法重组与优化权限的自查结果。目标保持为：**在 Intent 承诺的语言、执行模型和平台范围内，达到 Triton/TileLang 同等级的编译器工程质量、使用体验、扩展能力和持续维护能力。**

本轮交付是调查与路线图，不修改编译器、正式规格、作者算法或实验 CSV。里程碑是多项相关任务和一系列提交共同完成的能力，不是一个 commit、一个 pass 或一个测试通过。

## 1. 核心判断与本次纠正

**已有基础足以进入成熟产品建设，不需要再次推倒，也不需要不断重做统一接口。** Intent 已经有真实的 typed KIR、分 execution family 的 physical program、共享分析、provider lowering、JIT/runtime、独立安装包、IR 工具和 agent 工具。近期重构确实消除了多份参数表、ABI 推导、生成式 host wrapper 和不明确的变换边界。

剩余工作可以收束为六组明确交付：可控且合同明确的优化；标准 MLIR 工具可持续处理的 IR；独立用户与框架使用闭环；DSA 和 CPU 剩余组合性问题；可解释的物理质量与资源决策；可重复分发和后端演进维护。它们可以并行，不要求一切后端、硬件、精度和框架模式同时完美后才成为产品。

前两次简要回答有两处需要修正：

1. **不能把“作者没有显式写出该循环或状态”当作优化不合法的充分理由。** 成熟编译器会合并运算、重组循环、引入中间状态和改变存储。关键是当前语言/编译选项允许什么、变换保持什么、实现条件是否成立。
2. **不能把“未验证所有平台或所有组合”说成“现有工具仍然不成熟到不能使用”。** 已完成的独立安装、调用、MCP、IR 续编译和有限框架接入应正式结算；新增支持面和当前实现缺陷分别安排。

同样，乐观推进不等于用容差测试替代语义判断。合法优化积极保留，条件与收益策略适合时默认采用；需要额外数值自由的优化给出清楚的编译合同；实现错误修复；不因一次未命中或暂时缺少证明就删除整项能力。

### 1.1 “达到 Triton 成熟度”的可操作含义

成熟度按下列结果判断，不以代码量、pass 数、目录层数或论文成绩替代：

| 维度 | 最终应具备的能力 |
|---|---|
| 作者使用 | 合法算法在承诺支持的范围内可编译、调用、诊断；无需知道 compiler 内部命中什么图形 |
| 编译正确性 | 数值、类型、坐标、effects、alias 和调用合同明确；各层消费相同事实 |
| 优化质量 | 常见程序有高质量默认实现；性能问题可以定位并用通用变换改善 |
| IR 工程 | 当前 IR 独立表达执行事实；标准解析、打印、验证、优化和续编译可组合 |
| 扩展 | 新 primitive、优化规则和 provider 能在明确模块完成，不需在多层添加同一特例 |
| 产品交付 | 可取得、安装、运行和诊断；文档、工具、依赖与实际能力一致 |
| 持续维护 | 构建、原生产验收、产物和 SDK 升级有稳定流程；贡献者能复现并提交改动 |

不把 Triton 的社区人数、全部 GPU 历史支持或全部底层机器优化复制过来作为验收门槛。也不把“我们发射下层 DSL”当成无需做好上述工程的理由。

### 1.2 为什么不负责 lane layout，工作仍然不少

Triton 的作者输入已经包含 program id、tile shape、访问和部分遍历；TileLang 作者通常还给出 buffers、tile operations 与循环组织。Intent 输入更接近逻辑算法，必须先把它变成完整 block/task/local-storage program。

我们省去了下层大量指令、寄存器分配、lane layout 和 pipeline 实现，但增加了三项真实职责：

- 从逻辑成员、坐标、依赖和 effects 建立跨执行模型的 ownership 与遍历；
- 保持张量值、视图、物理 fragment、loop carry 和存储之间的一致关系；
- 让同一语义证明在 GPU、CPU、DSA 的不同改写中复用。

这能解释复杂性，不能作为无限建设基础设施的借口。对应的 IR、pipeline、analysis 与 runtime 基础现在已经存在；下一步应完成具体缺口并交付使用能力。

## 2. 调研方法、依据与覆盖范围

### 2.1 证据顺序

1. 当前 Intent 代码与正式 doc；
2. 本地 ref 的实际实现；
3. 当前编译观察、上一轮明确记录的运行结果；
4. 官方网络资料，用于补充背景和核对版本，不覆盖本地实现事实。

本地参考工作树在本次检查时干净：

- Triton：ef08c7f，2026-08-19；
- TileLang：f354430，2026-08-19。

这是本地快照，不声称等于某个实验环境安装包或网站当前主分支。网络资料与本地快照不一致时，以对应版本的代码说明行为；不从一个教程的写法推导“整个编译器绝不会做某种优化”。

### 2.2 本次完成的全链路自查

| 范围 | 核查内容 | 本次依据 |
|---|---|---|
| DSL/数值 | reduce、scan、ordered loop、region summary、cast、FMA、近似/FTZ、alias | 正式规格与当前 matcher/realizer |
| 前端与 KIR | 参数归属、typed graph、结构化与访问接口、规范化 | 当前 frontend、Intent IR 与公共分析 |
| 编译调度 | KIR/shared/provider 阶段、标准 PM、错误阶段、续编译 | Compiler、Backend、intent-opt 与 Python pipeline |
| GPU | 关系维护、mapping owner、局部 fold/CSE、online、参数与候选 | 当前 IR/Analysis/Transforms，含原生产源码编译观察 |
| CPU | 共同 pipeline、实现需求、ABI、布局/alias 限制、Mojo/Weft | 当前 CPU IR/passes、provider 与 runtime |
| DSA | construction 分支、matrix、shared passes、BANG 分层 | KIRToDSA、DSA transforms 与 BANG pipeline |
| Provider | 原语映射、target-local facts、数值选项、source/runtime 连接 | Triton/cuTile/TileLang/Mojo/Weft/BANG 代码 |
| 产品 | 包、依赖安装、CLI、JIT/cache、离线产物、PyTorch、MCP | 当前 Python/CMake/environment/examples |
| 维护 | 贡献导航、分发材料、许可证、自动交付、SDK/ABI 演进 | 根文件、构建目录、ref 的分发与运行代码 |
| 旧计划 | A1–A4、B1–B6、框架/教程/发布项 | 原报告各工作包与当前实现逐项对照 |

“完整自查”在这里指覆盖上述责任链，不把所有源码逐行读一遍或做全库形式化证明当作本轮完成条件。发现必须落到具体代码、后果和任务；没有发现问题的既有边界不因本次审查而重做。

### 2.3 运行证据怎样使用

- 上一轮已有：MLIR20 compiler/optimizer 与 wheel 构建；5090D 上四个原生产算子经 Triton/cuTile 共八项通过；Mojo 两项通过；无 SDK 环境的 shared IR 优化、续编译和导出；安装包的原 softmax + torch.compile 调用。
- 上一轮 Weft、TileLang、BANG C 的部分新结果只有 source generation，和更早 registry 的设备结果分开。
- 本次新增的是原 sparse MLA 的编译/IR 观察，以及原 TileLang RMSNorm 的当前编译复核；未启动 GPU benchmark，未产生新的数值或性能通过结论。
- 历史 CSV 是已有覆盖和问题的定位入口，不是当前 HEAD 的全量新验收。既有通过不抹掉，也不把未重测写成已失败。

## 3. 最近多轮到底完成了什么

### 3.1 旧 roadmap 逐项结算

| 旧工作包 | 当前结算 | 剩余工作归属 |
|---|---|---|
| A1 标准安装与工具定位 | 基础完成：pyproject、包内 compiler/optimizer/profiles/runtime 依赖闭包；前端不再依赖 Python MLIR bindings | M3/M6：声明平台基线、重复构建、发布与安装诊断 |
| A2 普通调用、README、工作流图 | 基础完成：公开 compile/run、原算法 quickstart、后端入口与 README | M3：补完整使用旅程及公共返回约定，不重写一遍首页 |
| A3 独立 manual MCP | 基础完成：包内材料、独立入口和客户端配置；compiler MCP 也已有 | M3：实际 agent 发现/诊断旅程、文档 API 同步 |
| A4/B3 阶段诊断与产物 | 主干完成：stage/location/log/source/IR、标准 PM、CLI/Python/MCP 同路 | M1/M5：优化采用/拒绝原因、编译/JIT/tuning 成本解释 |
| B1 目标能力合同 | 已补集中 provider eligibility 与候选验证，不能重复沿用旧 major>=9 漏口结论 | M5：按当前实际消费者继续补 feature/资源解释 |
| B2 写法与物理关系稳定性 | 部分完成：operation-owned schema、access/region 接口、关系维护与参数重构 | M2/M5：mapping 存活、fold/CSE、原程序的结构质量 |
| B4 共享语义证明 | 有真实共享成果：contraction/product axes、整数关系、online combine 公共核心等 | M2/M4：只提取新的实际重复，不造万能 scheduler |
| B5 DSA 分层 | 部分完成：已有 DSA shared passes、BANG 分阶段 lowering；旧“无 shared pipeline”已过时 | M4：消除剩余 whole-source matrix construction 分叉 |
| B6 资源决策改进 | capabilities、预算、参数、候选和 provider 过滤已有 | M5：候选剔除理由、事实来源、结构浪费与成本 |
| PyTorch adapter | GPU functional/fake 范围完成；原 softmax fullgraph 路径已运行 | M3：作者 backward 注册、公共调用语义、实际模型使用 |
| 教程和贡献入口 | CONTRIBUTING 与基本 examples 已有 | M3/M6：补能力旅程、保持入口与实现同步 |
| 公开分发/许可证 | 仍未闭合 | M6；许可证是维护者选择，不阻断其它任务 |

对应实现：[pyproject](../pyproject.toml):20，[公共 pipeline](../python/intent/compiler/pipeline.py):18–108，[编译驱动](../lib/Compiler/Compiler.cpp):58–118，[开发导航](../CONTRIBUTING.md):18–45、155–214，[DSA shared pipeline](../lib/Dialect/DSA/Transforms/Passes.cpp):63–71。

### 3.2 不应再次立项重做的基础

- 公共参数合同与实际物理 ABI 已分开，且 source signature 与 native slots 同源。
- GPU 参数声明、稳定引用、候选绑定与 host producer 依赖已经进入 typed IR。
- 编译事实与本机 device/SDK binding 已分开；GeneratedProgram 与 CompiledArtifact 职责明确。
- CLI、Python、compiler MCP 已调用同一编译入口；无需另造 agent compiler。
- native library/cache handle 已有复用机制；无需再次用缓存局部优化充当宏观里程碑。
- CPU/GPU 不同物理 IR 是合理分工，不能为了“统一”强行融合它们。
- 普通 reduce、scan 和 matrix primitives 应继续利用 provider 原生能力，不重建下层 layout/通信算法。

### 3.3 为什么会感觉在原地打转

首先，确有一部分工作在偿还早期“多轮追加功能先跑通”留下的结构债务。参数表、候选、ABI、生成 host wrapper 和关系维护先后收口，修改量大，但外部用户不直接看见。

其次，旧 roadmap 没有及时结算，已完成的安装/MCP/调用仍出现在“未来缺口”的叙事中；若每次都以尚未覆盖的更大集合评价成熟度，终点会不断移动。

第三，阶段完成曾被表述成宏观里程碑，而产品结果没有与其一起收束。一个 typed interface 迁移可以是重要提交，但单独不足以代表“用户现在能完成一个新的完整任务”。

第四，对语义保持和新优化的讨论有时走向两个极端：先凭性能推动特殊结构，后又因不是作者字面结构而过度否定。应改为固定的合法性、数值许可与收益判断。

因此后续每组任务必须同时回答：已完成的能力是什么；删掉了哪条冗余路径；外部用户或下一位开发者现在能独立做什么；还剩哪一项明确工作。代码量记录实际工作规模，不能代替这些答案。

## 4. 对照 Triton/TileLang：该学什么，不该照搬什么

### 4.1 责任层的对照

| 责任 | Triton/TileLang 的本地事实 | Intent 应承担的相应责任 |
|---|---|---|
| 作者输入 | Triton TTIR 对应 block 程序；TileLang 输入带 tile/buffer/loop 结构 | KIR 更高，需要先形成完整 GPU/CPU/DSA 程序 |
| 局部正规化 | Triton op 自带 fold/canonicalizer；Combine 做真实图改写 | 等价的 shape/value 规则归 op；跨 ownership 的变化归完整 transformation |
| 执行优化 | 共同 IR 上做 target-aware pipeline、warp specialization、存储改写 | 优化自己的 program mapping、blocking、reuse、materialization；不复制 lane/ISA |
| 局部专家实现 | TileLang GEMM 经 target/layout/thread 条件展开局部实现 | typed primitive/microkernel 可以复用；外围程序由 compiler 组织 |
| 编译选项 | 数值、优化选择、资源与 native compile 选项有具体消费者 | 给出同样清楚的类别与传播路径，不把所有选择都叫 tuning |
| JIT/产物 | compiler facts、source/IR/native code、runtime handle 有阶段边界 | 继续复用已有 GeneratedProgram/materialize；补兼容诊断和维护合同 |
| 分发 | 二进制、源码、构建基线和 CI 作为产品代码维护 | 建立我们承诺平台上的同等完整交付链 |

参考：[Triton NVIDIA pipeline](../../ref/triton/third_party/nvidia/backend/compiler.py):273–369；[Triton shape ops](../../ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td):469–531；[TileLang CUDA pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py):68–163、166–249；[TileLang GEMM](../../ref/tilelang/src/op/gemm.cc):188 起。Intent 对应：[compiler 边界](../doc/compiler/README.md):35–48，[CPU implementation 合同](../doc/compiler/cpu-program-ir.md):34–62。

### 4.2 成熟编译器确实会主动改变计算结构

本地 Triton 的 Combine pass：

- 对满足 rank、axis、broadcast 与 combine 条件的乘法归约，构造 IEEE precision DotOp：[Combine.cpp](../../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp):135–192。
- 对只供 reduce/histogram 使用的一维 reshape，允许元素重排：同文件 196–215。
- 在 zero accumulator、single-use 等条件下把 dot+add 合并：同文件 249–282。
- 这些规则被加入普通 Combine pass：同文件 297–314。

因此“作者没有写 dot / 新 accumulator，所以 compiler 不应生成”显然不是合理标准。Intent 已有相应数值合同的局部融合，也应积极做。

TileLang 的 CUDA pipeline 自动形成 producer/consumer warp specialization、software pipeline、buffer allocation 与 storage rewrite：[pipeline.py](../../ref/tilelang/tilelang/cuda/pipeline.py):100–140、173–178、210–217。改变循环、缓冲和同步的具体结构本来就是 compiler 工作；是否改变作者可观察语义必须另查。

### 4.3 选项不是保守的替代品，而是成熟控制面

本地 Triton 的 CUDAOptions 把 enable_fp_fusion、enable_reflect_ftz、dot precision 等分开：[compiler.py](../../ref/triton/third_party/nvidia/backend/compiler.py):120–129。Intent 当前 Triton runtime 显式允许后端 FMA 融合、关闭普通 libdevice FTZ：[runtime/triton.py](../python/intent/runtime/triton.py):181–186；这已经说明默认合同可以主动允许合理优化，而不是逐操作 bitwise 保守。

TileLang 的 PassConfigKey 包含：

- Simplify 内部策略；
- warp specialization 等优化选择；
- TL_ENABLE_FAST_MATH 对应 native fast-math 编译；
- reducer baseline 选择与计划解释。

依据：[pass_config.py](../../ref/tilelang/tilelang/transform/pass_config.py):15–50、73–80、110–130。它们并非同一语义：关闭某项优化不等于更换数值合同，启用 fast math 也不等于任意错误均可接受。

GCC 式思路可用于组织 Intent 的公开选项，但不照搬旗标数量。官方 GCC 文档也区分普通优化级别、浮点重结合及相关前提；这里只作背景对照，实际方案以本地 provider 实现及 Intent 合同为准。[官方优化选项说明](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

### 4.4 必须固定的判断顺序

1. 是否保持调用、shape/dtype、索引成员、effects、必要顺序与别名关系？
2. 改变的浮点求值方式，是否已经被该 operation 或选定数值模式允许？
3. matcher 是否足以证明该规则的适用条件，而非只认一个像某算法的外形？
4. 不采用此优化时，正常编译路径是否完整？
5. 采用后是否改善完整算子的资源、编译或运行成本？

前两项定义合法性，第三项负责实例化正确，第四项保证可组合，最后一项决定收益。正确性不要求每次编译形式化证明所有数学定律；维护者应给出明确规则、可执行条件、实现审查和原生产证据。

## 5. Online 重组：保留能力，补齐合同与编译路径

### 5.1 当前代码和实际触发

默认 GPU shared pipeline 在 [Passes.cpp](../lib/Dialect/GPU/Transforms/Passes.cpp):262 调用 OnlineReductionsPass。

[RealizeOnlineReduction.cpp](../lib/Dialect/GPU/Transforms/RealizeOnlineReduction.cpp):79–120 从顶层四字段 summary 图匹配；123–169 建立 chunk 参数和四个 carry；299–341 合成 max-rescale、mass 与 moment 合并。它不要求源程序已经使用 region_fold/combine。

本次使用既有生产算法 token_sparse_mla_value_prefill，原源码位于 [mla.py](../examples/kernels/streaming/mla.py):249–294、403–411，入口由 [GPU registry](../experiments/gpu/registry.py):94 连接。先生成 target 无关 KIR，再指定现有 5090D 编译事实生成 shared IR，观察到了 before/after online group：

- before：原来的四字段 summary；
- after：新增 REDUCE_CHUNK_12_A0 与四 carry 的 scf.for；
- 对应观察文件：/tmp/intentdsl-online-review-sparse-mla.log 与 /tmp/intentdsl-online-review-sparse-mla-shared.mlir。

这些是当前生产源码的编译证据，不是新增测试算法，也不是设备数值验证。

### 5.2 问题不在“合成了状态”，而在许可和适用条件

实数下，用局部最大值归一化，再通过比例合并 summary，可以是合理的流式实现。若源 operation 或数值模式允许相应有限精度变化，compiler 完全可以实现这一优化。

当前具体需要解决的是：

- 原路径对 global maximum 归一化后的概率进行低精度 cast；
- 新路径对 chunk maximum 归一化后的概率先 cast，再对局部 moment 重缩放；
- 一般不能把 cast(exp(x-global_max)) 与 cast(exp(x-chunk_max)) 后缩放视为逐操作等价；
- FMA 或普通 reduction 的重结合许可，并未自动说明这种跨操作、含低精度转换的重写允许什么。

同时需要核对 NaN/Inf、空集合 identity、validity guarding、不同 dtype 与额外消费者，不只看“有 max、exp、sum、matmul”这几个节点。当前 matcher 与生成代码的依据见 [OnlineSummary.cpp](../lib/Dialect/GPU/Transforms/OnlineSummary.cpp):127–262 和 RealizeOnlineReduction 的概率 cast/contract、合并部分。

**结论：不应整体删除 online 优化，也不能直接认定它在所有匹配输入上已经被当前数值合同覆盖。** 应形成默认合同可证明的适用域，以及显式声明额外数值自由后的适用域；实现错误在两种模式下都必须修复。

### 5.3 对照作者显式 region 路径

作者通过 region_fold/region_scan 给出的 summarize/combine 合同已经允许指定的连续分段与有限精度差异：[数值规格](../doc/dsl/types-numerics-and-effects.md):165–176。对应合法 lowering 应继续保留并优化。

Triton attention 教程和 TileLang attention 示例中作者显式写 online recurrence，只能说明这些程序怎样表达算法，不能推出编译器永远不能从另一表达中得到它。应该从默认图重写、数值选项和具体合法性检查共同判断。

### 5.4 关闭优化暴露出的真实实现缺口

本次从同一生产程序的 before-online IR，用标准 intent-opt 执行其余 shared groups，跳过该 group：

- 剩余 shared pipeline 可输出 /tmp/intentdsl-online-review-sparse-mla-without-online.mlir；
- 从该产物继续 provider 编译时，解析第 193 行失败；
- fragment extent 含 runtime Dimension 的 min/next_power_of_two 组合，违反当前 fragment 类型只允许常量/physical-parameter extent 的合同。

对应构造入口是 [Traversal.cpp](../lib/Dialect/GPU/Transforms/Traversal.cpp):33–59；类型约束是 [GPUDialect.cpp](../lib/Dialect/GPU/IR/GPUDialect.cpp):710–725。这不是已经证明默认 pipeline 的所有程序都会失败，而是这个原生产输入的优化关闭路径存在实际缺口。

因此新路线应同时完成：

1. 合法的 online 路径；
2. 原合同下保留 global max、再分块 exp/contract 的正常路径，优先复用现有 reduction/traversal；
3. 开关关闭后的完整 pipeline 与可重读 IR；
4. 不符合某种优化条件时的清楚原因。

不把“写成 region_fold 才行”甩给作者，也不在 serializer 或异常处理里暗中选择另一算法。

## 6. 当前结构与产品缺口清单

优先级含义：P0 是已有正确性、可重放或公开交付的直接缺口；P1 是下一阶段的主要能力；P2 是依附相关任务清理或后续扩展。优先级不等同于“必须串行等它完成才能做其它工作”。

| 编号 | 级别 | 当前事实与后果 | 收束目标 |
|---|---|---|---|
| F01 | P0 | Online 数值自由与 matcher 资格未闭合；已在原 sparse MLA 编译中触发 | M1：明确数值许可、保留优化、闭合正常路径 |
| F02 | P0 | 变换能产出内存中可流转、文本却不可重读的 fragment type | M2，前置修复与 M1 联动 |
| F03 | P1 | Pure Delinearize 承担隐藏的持久 mapping 事实，需要特殊保活 | M2：owner 表达持久事实，坐标计算正常 DCE/CSE |
| F04 | P1 | 局部 folds、手写 CSE 与跨 use-def 关系修复耦合 | M2：局部规则进 op，完整关系变换保留 |
| F05 | P1 | DSA whole-source matrix 与 structured/localMatMul 是两套入口合同 | M4：单一 construction/realization 主路径 |
| F06 | P1 | 公共 schema 已统一，但返回约定、named alias 及部分 view 支持仍不一致 | M3/M4：行为合同和 family 能力收口 |
| F07 | P1 | 资源过滤已有，精确事实、名义预算、偏好及下层拒绝缺统一解释 | M5：可追踪的合法性/收益决策 |
| F08 | P0 公开发布 | 根许可证、可重复二进制生产和自动交付入口未闭合 | M6，与 M3 并行开始 |
| F09 | P1 | 产物能保存恢复，但兼容范围及外部 SDK 改动的处理入口未形成完整产品合同 | M6 |
| F10 | P1 | 框架/MCP 基础完成，但能力较分散，作者训练集成和完整使用旅程不足 | M3 |
| F11 | P2 | 少量 canonical product schema 和可派生 buffer 字段仍重复 | 随 M4/M5 删除，不独立宣称里程碑 |
| F12 | P2 扩展 | provider/hardware 概念已分开，GPU capability 实现仍主要 NVIDIA | 主产品稳定后完成第二 GPU vendor 的真实纵向路线 |
| F13 | P1 | 原 TileLang RMSNorm 在当前 provider form-memory 阶段仍因非单位轴关系缺失而失败 | M2.7–2.8：修关系产生/维护与目标消费链，见 §11.1 |

### 6.1 IR 可组合性：优先修 owner 和不变量，而不是继续加 repair

DelinearizeOp 声明为 Pure：[GPUOps.td](../include/Intent/Dialect/GPU/IR/GPUOps.td):76–82。但 [EliminateCommonValues.cpp](../lib/Dialect/GPU/Transforms/EliminateCommonValues.cpp):187 与 [ValueRelations.cpp](../lib/Dialect/GPU/Transforms/ValueRelations.cpp):2056 特别排除其删除；其它变换按该 op 的位置、数量或附属属性读取 segment/mapping。

具体消费者：[ContractionAnalysis.cpp](../lib/Dialect/GPU/Transforms/ContractionAnalysis.cpp):58–84；[RefineProgramMapping.cpp](../lib/Dialect/GPU/Transforms/RefineProgramMapping.cpp):156–168；[PointwiseAnalysis.cpp](../lib/Dialect/GPU/Transforms/PointwiseAnalysis.cpp):1182–1187。

这说明持久执行事实还部分依赖“不应被删掉的纯计算节点”。应利用已有 program/segment/ownership owner 保存事实；Delinearize 只表示坐标求值。不能只是给它伪造 side effect，让所有 optimizer 都不敢动。

GPU 的普通 shape/value operations 尚缺合适的局部 fold/canonicalizer，而 [EliminateCommonValues.cpp](../lib/Dialect/GPU/Transforms/EliminateCommonValues.cpp):170–208 自行维护 block-local CSE。对照 Triton 的 op-local canonicalization，可以迁移明确纯局部的等价规则。涉及 coordinate identity、owner、captures、accumulator 和 access 的成组改写仍归现有 transformation/worklist，不新建另一套全局修复系统。

接入标准 CSE 前，先区分仅用于诊断的 origin 与有语义作用的 source-axis、owner、effect identity、numeric attributes。当前 OperationEquivalence 使用 exactValueMatch、忽略 location，但仍比较 attributes。不能为提高命中率统一忽略 origin/axisMap/owner 等字段；仅诊断信息的处理也须保留可用的 source 映射。同 shape 不等于同 value。

F02 的处理也不能止于“打印后再读一遍”：应修产生非法类型的构造和 staging，再让类型构造、完整变换后置与文本入口消费同一个约束。无需为每个 helper 重复全 module 验证。

### 6.2 DSA：已有进步，剩余两路线必须真正合并

[KIRToDSA.cpp](../lib/Conversion/KIRToDSA/KIRToDSA.cpp):146–185 根据整份 source 中 contract 的位置和 structured operations 选择路线。

- lowerMatrix 在 3578–3612 限定 direct full rank-two views、特定轴配对和共享 LHS；
- 3615–3640 又限定单 output 和 pointwise epilogue；
- structured localMatMul 在 675–742 另行处理 local shape、reduction slicing、projection 与 replay；
- 两路分别形成 loops、local allocations、loads、matrix 与 stores。

问题是合同和 realization 的重复，不是文件长。新任务应统一语义查询与当前 DSA program 的矩阵实现，再删除 whole-function shortcut；保留已经存在的 [DSA shared passes](../lib/Dialect/DSA/Transforms/Passes.cpp):63–66 和 BANG target-local realization。

TileLang 的 [CUDA pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py):91–142 与 [CPU pipeline](../../ref/tilelang/tilelang/cpu/pipeline.py):21–66 表明：共同 tile/operation 语义可以保留，不同 execution model 各自实现，不需要统一硬件拓扑。

### 6.3 公共调用：schema 统一以后还要收口行为

当前 .run()/prepared .result() 的实际约定：

| 路径 | 返回内容 | 没有输出时 |
|---|---|---|
| GPU | Out，声明顺序 | None |
| Mojo / Weft | Out + InOut，声明顺序 | 空 tuple |
| BANG C | Out，声明顺序 | 空 tuple |

依据：[native.py](../python/intent/runtime/native.py):215；[GPU interface](../python/intent/runtime/gpu/interface.py):67–70、272–273；[Mojo](../python/intent/runtime/mojo/program.py):150–157；[Weft](../python/intent/runtime/weft/program.py):226–236；[BANG C](../python/intent/runtime/bangc/program.py):138–141。区别已写在 CONTRIBUTING，但正式 public authoring 没有完整定义零/单/多输出及 InOut 回传。

建议统一为：run/result 只返回 Out；零个返回 None，一个返回该值，多个按声明顺序返回 tuple；InOut 通过原对象观察。公开 launch/显式输出调用负责执行，返回 None；内部 provider 的 LaunchResult/compiled kernel 仍供 backend IR 收集和诊断使用，不随公开返回约定删除。**这是待确认的公开行为提案，不是本轮已修改的规格。** CPU join、GPU stream、MLU queue 继续各守执行语义。

named alias group 同样需要一句完整合同。建议同组表示同一底层 allocation，允许不同 offset、shape 和不重叠子视图；不同组不推出 disjoint；noalias 维持既有 allocation 不重叠前置条件。若作者实际需要的不是这个含义，应在语言合同处一次解决，不能让各 runtime 自己解释。

另外，CPU 当前 verifier 对 view layout、stride constraints、scalar carrier 有明确限制：[PhysicalProgram.cpp](../lib/Dialect/CPU/Analysis/PhysicalProgram.cpp):199–238。readonly/Out/InOut 的别名和布局支持要按实际实现资格处理；不把 provider 某个 microkernel 的限制自动提升为整个语言的限制。

### 6.4 资源和硬件：强化已有机制，不重新实现下层编译器

已有能力包括 shared fragment footprint、provider collective 预算、TileLang 明确 allocation lifetime 的容量分析：

- [ConfigurationConstraints.cpp](../lib/Dialect/GPU/Transforms/ConfigurationConstraints.cpp):12–32；
- [Triton Configurations.cpp](../lib/Target/Triton/Transforms/Configurations.cpp):436–466；
- [TileLang Legalize.cpp](../lib/Target/TileLang/Transforms/Legalize.cpp):179–218。

需要区分“准确证明非法”和“名义 payload 预算不看好”。例如 warps × 32 × 255 不是实际 register allocation；当前审查没有证明它已经错误剔除了某个生产候选，不能直接删除或宣称发现性能 bug。

Triton 的真实 allocation analysis 处理 buffers、offsets、intervals 和 aliases：[Allocation.h](../../ref/triton/include/triton/Analysis/Allocation.h):70–129；这是更低层的职责。Intent 应解释自己的 footprint/lifetime，并利用下层实际反馈，不复制该 allocator。

当前 NVIDIA 的多 CTA 限制已排除 SM12x：[Configuration.cpp](../lib/Target/Triton/IR/Configuration.cpp):13–27。下一硬件扩展再引入实际需要的 vendor/wave/feature facts，参考 [Triton AMD compiler](../../ref/triton/third_party/amd/backend/compiler.py):106–108、154–174、260–273。不为证明“泛用”而提前加入无人消费的硬件字段。

### 6.5 可删除的重复，必须附着于真实任务

- CPU/DSA 的 canonical tuple/record 叶类型、字段顺序和 offsets 可共享；各自的 storage、lazy materialization、slicing、copy 与 carries 不硬合并。入口：[KIRToCPU.cpp](../lib/Conversion/KIRToCPU/KIRToCPU.cpp):253–278；[KIRToDSA.cpp](../lib/Conversion/KIRToDSA/KIRToDSA.cpp):2895–2928。
- 当前 BufferType 的 workspace 与 lifetime 由 scope 唯一确定：[GPUDialect.cpp](../lib/Dialect/GPU/IR/GPUDialect.cpp):745–768。核清消费者后可以派生查询替代重复字段。
- initialization、owner、instance、visibility 有独立意义，不能因为“字段少更漂亮”一起删除。
- serializer 从已经确定的 typed shapes/axes 拼写目标 API，不等于偷偷做算法决策。本轮没有新发现整算子 serializer 旁路，不以怀疑制造另一轮整理。

## 7. 编译优化控制面的具体设计方向

本节是下一阶段设计提案。正式 doc 的数值变化必须在提案确定后同步；不让 report 暗中覆盖现有规格。

### 7.1 四类输入分开，不再混进 tuning_config

| 类别 | 内容 | 作用 |
|---|---|---|
| 语言/数值许可 | source 已有 FMA/reduce/region 权利，及将来明确的额外重组权限 | 决定哪些变换合法 |
| 可选优化策略 | 选择/关闭某个完整可省略优化 group | 决定是否采用合法优化，便于复现与排查 |
| 目标能力 | dtype、primitive、grid、资源、架构 feature | 决定目标能否实现 |
| 候选参数 | 已声明结构的 tile/grain/provider options | 在合法空间中选择性能 |

tuning_config 继续提供有限候选数据，不放布尔算法规则、错误绕过或设备名分支。必要 construction、ABI legalization、验证不属于用户可以随意关闭的普通优化。

### 7.2 推荐的默认与额外数值模式

- 默认名称和说明应是“遵循 source 合同”。它本来就允许规定范围内的并行归约、FMA、局部 contraction 融合，不应叫“逐操作严格模式”。
- 合同已覆盖、条件充分的优化保留为合法候选，并在收益策略适合时默认采用，不要求作者重复提示。
- 对 normalized reduction 等额外重组，给出具体许可：允许改变的归一化参考、部分累加/舍入关系、适用 dtype，以及必须保持的特殊值、effects 和 ABI。
- 不直接提供一个含义模糊的 fast_math=True，让所有 pass 自行扩大解释。
- 显式 approximation/FTZ、dot 输入精度、外部格式和量化操作的独立舍入条款分别处理；不能由一个后端全局旗标悄悄改变。
- 若证明某类 online 变换已经由现有合同覆盖，直接纳入默认；只有实际需要新增数值自由的范围才使用新模式。

### 7.3 贯通路径

公开 Python/CLI/compiler MCP
→ 一份 compile request
→ typed IR 中的有效数值/优化事实
→ family/provider eligibility
→ 实际 source/native 编译选项
→ artifact metadata 和缓存依赖。

现有入口：[pipeline.py](../python/intent/compiler/pipeline.py):18–54、130–152；[Compiler::Request](../include/Intent/Compiler/Compiler.h):36–46。扩展这些入口，不另造“优化编译器”或 agent 特殊通道。

shared IR 续编译沿用产物已经绑定的许可；不能在加载时无声换一种数值模式。缓存必须区分实际影响生成/执行的选项；继续使用现有完整输入与文件依赖机制，不新增 hash/checksum 机制。

### 7.4 优化开关的准入

一个 group 可被关闭，须满足：

1. 前后的程序都拥有完整语义和类型；
2. 后续必需 passes 不依赖它附带修复本不属于它的不变量；
3. 不执行该优化仍能得到合法 provider program；
4. 开关改变能从诊断、IR 和产物信息中看到；
5. 对应原生产程序可以通过现有编译与数值入口验证。

这是编译器设计要求，不是额外的人为审批。M1/M2 要解决现有 online 关闭路径，而不是先公布一个实际无法工作的开关。

### 7.5 诊断输出的最小内容

复用 MLIR remarks 与现有阶段诊断，逐步让关键规则能说明：

- 规则名称及 source location；
- 已采用、未命中、缺数值许可、目标不支持或预算淘汰；
- 依赖的 dtype/axes/alias/capability 事实；
- 原因来自 Intent 还是下层 compiler；
- 对应 IR/source 与日志位置。

这些是观察结果，不是第二份 executable plan。完整日志继续留缓存目录，普通用户默认看到简洁结论。

## 8. 后续设计与改动的共同标准

1. **先复用当前 owner。** 参数、访问、结构化 regions、ABI 已有承载者；只有真实缺少执行事实才扩展 IR。
2. **一个语义事实一个权威。** 类型、操作或 region 保存正式关系，analysis 读取当前 IR；缓存不承担语义。
3. **模块边界跟职责走。** 公共证明、family rewrite、provider realization、runtime 相互清楚；不强求 CPU/GPU 同目录同形态。
4. **正向构造与检查一致。** 不先造不合法 IR，指望后面某个优化顺带修复。
5. **优化有前提但不为所有前提造新 hint。** 从 dtype、def-use、effects、坐标与已有 operation 合同直接取得的事实，compiler 自己使用。
6. **标准能力优先。** 使用 MLIR 的 op interfaces、fold/pattern、PassManager、diagnostics；使用 provider 的 reduce/scan/layout/pipeline。
7. **一个有效实现路径。** 迁移完成即删除同责旧路径；显式优化控制不等于保留两套编译器。
8. **慢与错分开。** 正确但慢的实现可以是明确的基础 lowering；不能把另一算法或隐式 host 替代称作兼容 fallback，也不能仅因暂时慢就拒绝所有合法程序。
9. **删除冗余随任务完成。** 不按文件长度拆模块，不按删除行数挑目标，不保留无人使用的“未来扩展层”。
10. **公开能力有有限而真实的支持范围。** 范围内的缺陷要修；范围扩展单独推进；不随每轮调研扩大成熟度门槛。

## 9. 六个成组里程碑

下面每个里程碑都允许由多项任务、多次实现与若干连贯提交完成。实现、删除旧路径、用户入口和必要生产验证一起收束；不能只完成表中的一个节点就宣布整个里程碑完成。

源码增删按整组实际工作统计，延续用户对实质改动规模的要求，可以累计多个提交达到 5,000+；不靠重排、无关注释、生成代码、文档或结果表凑数。任务自然形成的规模和能力结果同时报告。

### 总览与依赖

| 里程碑 | 用户或开发者得到的结果 | 主要依赖 |
|---|---|---|
| M1 优化合同与控制闭环 | 能积极使用、关闭和解释合法优化，online 的正常/放宽数值路径明确 | 现有编译链；与 M2 的类型修复共同推进 |
| M2 可组合的 MLIR 与共享变换 | IR 能在标准工具中优化、保存、重读、续编译；不靠纯 op 保活 | 已完成的 typed 参数/访问/ABI，不重做 |
| M3 统一调用与完整使用旅程 | 外部用户和 agent 能独立安装、编译、调用、集成、诊断 | 现有产品入口；公共行为选择一次收口 |
| M4 跨执行模型单一路线 | DSA 去掉 whole-source matrix 分叉，CPU/DSA 共享正确层的知识 | M2 的改写原则；现有 DSA/CPU 骨架 |
| M5 物理质量与资源解释 | 性能问题能沿结构和真实资源定位，优化可以复用 | M1/M2；可与 M4 的特定分析共享并行 |
| M6 可持续分发与演进 | 可重复取得产品、升级 provider、接受外部贡献和持续验证 | 构建/许可证工作立即开始，最终汇集其它里程碑 |

建议推进顺序：M1/M2 是近期编译器主线；M3 与 M6 的构建/分发准备同时推进。之后 M4 与 M5 并行，最后以 M6 的产品交付收束。不是等 M1–M5 全部结束才让别人使用。

### M1：优化合同、控制和 online 路径成为一个完整能力

**问题：** 当前正确优化与额外数值自由未形成统一控制面；online 关闭会暴露普通路径缺口。单独加一个 bool 开关或删除 pass 都没有完成目标。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M1.1 明确数值权限 | 对照 types-numerics、provider 选项与真实算子；形成少量明确条款 | 默认继续允许已有 FMA/reduce/contract 融合；额外 online 许可具体化 |
| M1.2 闭合正常 lowering | Traversal、RealizeReductionBlocking、retained values、ContractionValues | 保留 global max 与原 cast 关系的分块实现；修已定位 fragment extent 问题 |
| M1.3 收口 online 资格 | OnlineSummary 公共证明、GPU matcher/realizer、region 路径 | 普通图使用 typed 归一化关系与数值许可；已有 region/combine 的路径复用 combine 分析；dtype/特殊值、replay/effects 各有实际检查 |
| M1.4 统一编译输入 | Python pipeline、Compiler::Request、CLI、compiler MCP | 一个调用参数模型贯通各入口，不把数值政策放进 tuning JSON |
| M1.5 绑定产物与下层 | typed module/program facts、provider lowering/runtime、artifact/cache | 默认与额外模式可区分，实际 native options 与声明一致，续编译不变更许可 |
| M1.6 提供合法的优化控制 | 注册的完整可选 group、默认 pipeline | 关闭 group 不损坏必要不变量，未知选项明确报错 |
| M1.7 暴露采用原因 | 现有 MLIR diagnostics、toolchain、CLI/MCP | 区分缺权限、未匹配、目标限制和无收益，不在日志之外另造决策权威 |
| M1.8 原生产收束 | sparse_mla_prefill 及确实受影响的既有归约/region/contraction case | 原输入/容差，生成、编译、数值、耗时分别记录；不新造边界矩阵 |

**应删除的东西：** 散落的 policy 默认值、仅凭形状外观获得数值权限的隐式路径、为关闭优化临时补的 exception fallback。保留同一变换中的合法模式，不建立两份长期并存的整套 compiler。

**完成标准：**

- 同一个原生产程序有明确默认合同和优化资格；作者无需为了绕过 compiler 缺口改算法。
- 已选择的优化可以关闭并得到可解析、可续编译、可运行的完整程序。
- 选项在 API、IR、metadata、实际下层编译及缓存中一致。
- 原生产数值比较通过；性能数据只反映实际完整算子，不将 JIT/tuning 混入。
- online 能力保留并有明确适用域，不能以“保守”为由一删了之，也不能用 fast-math 名称覆盖实现错误。

### M2：让 MLIR 本身成为可组合、可维护的编译接口

**问题：** typed IR 已经有实际作用，但 fragment staging、mapping 生命周期和局部正规化仍存在自定义工具链假设。这是一组横向 IR 问题，应一起解决。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M2.1 统一 extent staging | GPU PhysicalExpr/FragmentType、Traversal、Deferred 参数 | 优先把 host-derived capacity 表达为已有 Deferred 符号，fragment type 消费合法 compile-time 表达式 |
| M2.2 统一构造和验证 | GPU IR builders/mutation、GPUDialect verifier、完整 transformation 后置 | release 构造与文本解析遵守同一约束；不靠 Debug assert 才发现 |
| M2.3 给 mapping 正确 owner | Program、program space/segments、Delinearize consumers | 先区分持久执行决定与派生坐标值，复用现有 owner |
| M2.4 删除 keepalive 例外 | EliminateCommonValues、ValueRelations、ContractionAnalysis、RefineProgramMapping | 无用坐标计算可被标准 DCE 删除，后续不再靠扫描纯 op 找语义 |
| M2.5 落实局部 folds | GPUOps 定义及相邻 canonicalization 实现 | identity projection、select、record extract、合法 transpose composition 等纯局部规则有唯一实现 |
| M2.6 收口 CSE 职责 | 自定义 common-values 与标准 canonicalizer/CSE | 保留真正需要专门分析的变换，删与标准能力等价的重复逻辑 |
| M2.7 修复并收口完整关系变换 | 当前 ValueRelations worklist、access/structured/ownership 查询 | 修复普通 reduce→broadcast/pointwise→store 的 source-axis 关系产生/维护，追到 RMSNorm derived 22 与 source 6 分歧的责任层；跨图修复不塞进 op fold |
| M2.8 完成工具链消费 | intent-opt、generate_from_ir、三个 GPU provider 与公共工具 | 原 shared IR 保存、标准优化、重新读取、继续生成走同一链路；原 rms_norm_f32 的 TileLang form-memory 正常路径闭合 |

**应删除的东西：** 纯 mapping op 保活白名单、同一局部规则的多处实现、没有独立语义的重复类型字段、用文本/旁表补回 owner 的逻辑。

**完成标准：**

- 已定位的 sparse MLA shared IR 重读问题在产生不合法类型的层解决。
- 原 rms_norm_f32 无需改写作者算法即可完成 TileLang 生成与原生产运行；不在 scalarizer 中按相同 shape 猜测缺失轴关系。
- 普通标准 canonicalize/CSE 不破坏后续所需 program facts。
- 局部等价规则在 op/pattern 层可复用；跨结构改写仍有完整后置合同。
- 至少通过原有归约、contraction、结构化 region、带 compiler-private resource 的实际编译路径；运行只选择真实受影响的原 production cases。
- 不需要重新运行一整套论文实验，也不以新建 fixtures 替代真实程序。

### M3：统一公共行为，完成用户、框架与 agent 使用旅程

**问题：** 包和工具已经可用，但返回行为、别名约束和较完整使用方式仍分散在 provider 或开发文档中。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M3.1 明确公开返回合同 | authoring 提案、PublicInterface、GPU/native/BANG result | 零/单/多输出、Out/InOut、run/launch/result 语义一次定义 |
| M3.2 统一 alias 声明检查 | PublicInterface、各 provider observer/binder | named group 与 noalias 分清；family 不支持 overlap 时给正确性质的错误 |
| M3.3 把调用要求归实际能力 | CPU/DSA EntryRequirements、native observers、provider legality | 不以共同 binder 名称掩盖全局 contiguity/disjointness 限制；可以共享的检查只保一份 |
| M3.4 完成框架作者流程 | 现有 torch adapter、softmax forward/backward 算法与公开示例 | 作者注册 backward 的完整实际用法；明确保存张量、fake/output schema 和调用目标 |
| M3.5 组织产品教程 | 现有 examples/README、README 深入入口、CONTRIBUTING | compile/run、显式输出/prepared、同 source 换 target、离线产物、IR 优化形成少量完整旅程 |
| M3.6 完成 agent 发现流程 | manual 与 compiler MCP、CLI API 声明 | 查询语义→编译→读取真实错误/IR→修正的日常流程独立可用 |
| M3.7 解释首次使用成本 | 当前 toolchain/materialize/provider 边界 | Intent compile、native JIT、tuning、load、热调用分别说明，默认输出不淹没用户 |
| M3.8 交付环境定位 | doctor、backends、environment 安装入口 | 说明缺的是包、compiler provider、SDK、设备还是语言能力，不要求维护者私有路径 |

**应删除的东西：** 公共接口统一后残留的重复返回筛选、dtype/shape/alias 推导，以及教程中已经失效的实验输出路径。手写 baseline 的声明 helper 已迁到 experiments/cpu/baselines/mojo/source.py，仍有原实验消费者，不重复立项迁移或当作无用代码删除。不会借此复制一套算法或把 baseline 带入 manual。

**完成标准：**

- 新用户取得 Intent 后，按文档在声明环境完成现有算法调用，而不依赖 experiments driver。
- 原有 forward/backward 能作为作者定义的训练操作集成，不自动发明 backward。
- 同一算法换 target 时，参数和公开结果规则一致；执行同步仍遵循各目标实际模型。
- 真实 MCP 客户端可发现公开 API、编译原程序并取得可行动诊断。
- readonly manual 仍只提供通用语法、类型、接口、语义和最小片段；完整算法保留在普通教程。
- 固定 agent 实验继续一次交付，不以产品交互流程回写历史成绩。

### M4：跨执行模型的组合能力与单一路线

**问题：** GPU/CPU shared 骨架已成立，DSA 剩余 whole-source matrix 分支及 CPU/DSA 的一部分 schema 重复影响真实扩展。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M4.1 列清共同语义与目标事实 | ContractionAxes、StructuredOpInterface、canonical product schema | 轴、字段路径、访问窗口、result mapping 共用；storage/task 独立 |
| M4.2 统一 DSA 初始构造 | KIRToDSA 普通 structured/workset 路线 | 每个合法局部 operation 有初始完整表示，不按 whole-function 图形选择另一编译器 |
| M4.3 统一矩阵 realization | lowerMatrix/localMatMul 与 DSA MatrixSupply | 一份 typed contract 资格和数据供应/结果映射接口 |
| M4.4 迁移执行策略 | task traversal、tiling、local storage、协作与资源分析 | 在当前 DSA program 的完整变换中产生，不回读 KIR 重建 |
| M4.5 删除旧分叉 | whole-source matrices classification、epilogue 白名单、重复 loops/alloc/load/store | 移植完成即删，不留“旧路径模式” |
| M4.6 闭合 CPU 表达能力差异 | CPU view/stride/alias eligibility、Mojo/Weft 接口 | 可由当前 IR 表达的形式正常 lowering，只有真实 provider/hardware 缺能力才拒绝 |
| M4.7 扩展一个真实局部能力 | 从原 registry 的现有 primitive/form 缺口选择 | 验证新增规则无需在 construction、serializer、runtime 多次重复算法判断 |
| M4.8 设备闭环 | 原 MLU/CPU registry 及已有远端入口 | source→native compile→runtime→原数值比较；硬件不可用时明确未完成设备部分 |

**应删除的东西：** DSA 两条矩阵 realization 的同责代码、CPU/DSA 纯 product schema 重复、已经被共同查询替代的局部白名单。不删除 DSA local-memory/DMA/engine 模型，也不强迫 Weft 经过 Mojo SIMD。

**完成标准：**

- plain matrix、matrix+epilogue 与结构化 region 中的 matrix 共用同一局部实现合同。
- 后续变换只消费当前 DSA/CPU IR；原程序不需要为命中某个入口改写算法。
- 共享知识至少有两个真实消费者，并说明各自不同的物理改写。
- 原 MLU/CPU 受影响生产程序完成实际链路；生成成功单独报告，不提前把设备里程碑结案。

### M5：物理质量、资源决策与优化解释

**问题：** 已有能力和约束还缺可追踪的解释；部分写法敏感性仍可能来自 facts 丢失或早期结构固定。此里程碑不以单个算子刷新耗时为终点。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M5.1 给决策分类 | 当前 capability、resource、profile、candidate consumers | 硬非法、名义预算、偏好、未知、下层失败分别说明 |
| M5.2 统一资源查询 | GPU Resources、storage/lifetime、CPU input supply | 当前 IR facts 在短生命周期 analysis 中复用，不跨改写保留失效结果 |
| M5.3 修已观察结构问题 | 既有 producer projection、reduction/free-axis ownership、reshape/broadcast 关系 | 从实际失败/掉速程序选一条完整关系链，修共同机制 |
| M5.4 提升候选质量 | 现有 correlated profiles 与 parameter domains | 保持有限候选和合法性；不靠放大笛卡尔积掩盖结构问题 |
| M5.5 暴露下层事实 | provider 返回的 register/shared-memory 等已有信息 | 来源、阶段、测量身份明确；不反向猜成上层真值 |
| M5.6 验证跨架构职责 | 原 5090D/H100 入口与同算法程序 | 解释 Intent 结构差异、provider config 差异及下层处理的差异 |
| M5.7 清理确证冗余 | buffer scope 派生 workspace/lifetime 等 | 删除重复 carrier，保留独立 owner/instance/initialization 事实 |
| M5.8 形成优化扩展范式 | 现有贡献指南、当前 IR query/rewrite/provider 边界 | 新优化有明确 facts、legality、rewrite、收益和实际复用，不写任务名规则 |

**应删除的东西：** 同一事实的多套估计/解释、已被证明冗余的 repair 或字段、按样例身份分支。没有错误证据的预算启发式先解释再改，不因“不是精确 occupancy”就全部丢弃。

**完成标准：**

- 对选定原生产问题，能回答“当前结构为何慢、哪个事实丢失、规则为何适用、改了哪个 IR”。
- 至少一个结构规则服务不同原有程序，至少一项共同分析被两个执行模型或两个 provider 的真实变换消费。
- 完整算子性能有原 reference、输入、dtype、容差和比值；不把编译/JIT/tuning 当算子时间。
- H100 与 5090D 产生相同 source 也可以是正确结果；不把“源码必须不同”作为资源感知验收。

### M6：可重复分发、SDK 演进和长期维护成为产品的一部分

**问题：** 本地 wheel 已有真实价值，但公开产品还需要可重复生产、明确支持合同以及接手和升级流程。它不要求先支持所有平台。

| 任务 | 实现与主要位置 | 必须完成的收口 |
|---|---|---|
| M6.1 选择许可证和材料范围 | 根许可证、package/sdist/wheel 内容、third-party notices | 维护者决定许可证；实验 baselines 与产品材料分发边界明确 |
| M6.2 固定二进制构建基线 | 现有 CMake/scikit-build/bundle runtime、构建自动化 | 选定 Linux 系统 ABI、Python 与工具链组合；不把本地包误标为任意系统通用 |
| M6.3 自动生成可安装产物 | wheel、compiler/optimizer/profiles/manual data、notices | 构建可重复执行，安装离开 checkout 后不需手工库路径 |
| M6.4 自动复用现有验收入口 | 构建、安装、doctor、离线 IR、现有 production registry | 不新造 test/fixture/矩阵；按变更范围调度既有入口 |
| M6.5 定义产物兼容行为 | GeneratedProgram、target matching、provider SDK 绑定 | 能解释、能加载，或要求使用调用方保留的原 Intent definition/KIR 重编；不默默猜旧字段 |
| M6.6 SDK/ABI 升级入口 | provider legality、source spelling、native ABI/runtime、environment | 一次真实升级有清楚修改层，不跨后端扩散 |
| M6.7 外部贡献闭环 | CONTRIBUTING、构建/问题定位/提交说明、原 production 入口 | 开发者能在自己的环境修改一个实际局部能力并给出相关证据 |
| M6.8 产品交付与维护节奏 | 选定支持域、实际获取方式、已有结果更新 | 每次交付说明已验证组合、已知缺口与解决入口；实际发布按维护者授权执行 |

本地成熟参考：[Triton installation](../../ref/triton/docs/getting-started/installation.rst):11、27；[Triton wheels workflow](../../ref/triton/.github/workflows/wheels.yml):72；[TileLang dist workflow](../../ref/tilelang/.github/workflows/dist.yml):76–93。学习它们的可重复链路，不照搬其全部平台矩阵或缓存来源校验机制。

**应删除的东西：** 产品路径中的维护者私有目录、过时安装说明、重复 SDK 检测、将“包能 import”冒充“后端可运行”的模糊状态。无需新增 CHANGELOG、版本号或迁移文档；修改用 Git 记录，支持说明放已有用户入口。

**完成标准：**

- 用户取得声明支持的产物后可以独立安装并运行原有完整程序。
- 构建、依赖定位、原生产验收和产物获取可以再次执行。
- 一个 provider 的实际 API/ABI 变化能在所属层完成适配，并通过原程序验证。
- 旧产物不兼容时说明是否有调用方保留的原 Intent definition/KIR 可供重编，不承诺仅凭旧生成产物恢复原算法；不维护无边界的 schema 猜测兼容层。
- 编译器的使用、调试、优化控制和贡献已形成稳定产品流程。

## 10. 后端与 ABI 改变时怎样跟进

统一接口的目标是让变化局部化，并让影响范围可以判断，不是让未来所有版本自动兼容。

| 变化类型 | 应修改的层 | 应保持的边界 |
|---|---|---|
| 用户算法新增输入、shape/dtype/外部布局改变 | 作者声明/host wrapper 与现有 frontend/public interface | compiler 不猜新算法；同一声明仍供所有后端消费 |
| 物理参数重排、增加 workspace/stride/extent | family/provider lowering、typed argument binding 或 native slots | 公共参数不跟着物理位置变化；runtime 读取当前 ABI |
| descriptor、allocator、launcher API 改变 | 对应 provider legality、source/runtime adapter | 不修改无关 family 的算法和公共合同 |
| native 标量或结构体 carrier 改变 | NativeABI query、provider signature/runtime marshal | 源码签名与调用参数仍同源；缺少必要表示时再扩展 |
| SDK 只改变内部 LLVM/机器 ABI | 外部 provider 自己的 compiler/runtime，Intent 核对实际使用接口 | 不复制下层内部 ABI；必要时在适配层跟进 |
| 新 GPU vendor 或真实执行模型变化 | target facts、family/provider-specific lowering | 不以配置名称冒充已实现的新硬件/执行模型 |
| 保存产物的 schema 不能被当前实现解释 | 产物加载诊断和原 Intent definition/KIR 重编路径 | 不猜字段、不给缺失语义塞默认值；重编输入需由调用方保留 |

当前共同 ABI 已显著改善维护成本：[NativeABI.cpp](../lib/Serialization/NativeABI.cpp):33–64；[GPU ProgramInterface](../include/Intent/Dialect/GPU/Analysis/ProgramInterface.h):14–26；[GeneratedProgram](../python/intent/compiler/artifact.py):62–78、108–155。

GeneratedProgram.save 当前只保存 provider source、最终 IR 和 metadata，不保存原 Intent Python definition 或原始 KIR（同文件 98–101）。因此这里的“重编”明确要求调用方仍持有原 definition/KIR；program.source 是生成的后端源码，不能当作原算法输入。仅有旧 GeneratedProgram 时，不承诺恢复原算法或跨硬件 retarget；本路线不为此另造一个隐含归档格式。

成熟参考也保留具体产物边界：[Triton CompiledKernel](../../ref/triton/python/triton/compiler/compiler.py):407–488 从 metadata 恢复具体 target 并在加载设备时检查资源；[TileLang kernel](../../ref/tilelang/tilelang/jit/kernel.py):151 起的恢复和 737–762 的共享库导出均受实际 backend/runtime 条件约束。

因此 M6 的验收是：对一个真实升级，知道该改哪里、能解释哪些产物需重编、能使用原程序验证升级。不是承诺所有旧 MLIR、所有 SDK 组合永远互通。

### 10.1 第二 GPU vendor 的后续纵向任务

在当前产品主线稳定后，按真实设备条件推进：

1. 从现有 NVIDIA-only capability consumers 抽出实际 architecture family/wave width/feature facts；
2. 同一 Triton provider 接另一 hardware target，保留 NVIDIA-local descriptor 等判断；
3. 使用已有原算法和对应设备环境贯通 compile→native→run→原数值比较；
4. 证明共享 KIR/分析与 GPU program 的价值，实际消除硬编码 32/SM 的错误假设；
5. 不复制 Triton AMD/NVIDIA 下层的布局、MMA 或寄存器分配。

这是新的支持能力里程碑，不把尚未做完的多 vendor 覆盖作为当前 Linux/NVIDIA 产品永远不能交付的理由。

## 11. 未完成事项承接表

这张表明确承接此前留下的工作，避免下一轮再次只选一个容易的局部点。状态在实施时依据实际代码和原运行入口更新。

| 事项 | 当前证据/状态 | 归属与下一动作 |
|---|---|---|
| named alias group 含义 | doc 未明确；CPU/MLU 同 allocation 检查，GPU 未检查 | M3.1–3.2：确定一条公共合同并统一；不把历史差异永久化 |
| 公共返回约定 | Out/InOut、None/tuple 存在当前差异 | M3.1：按明确零/单/多输出合同收口 |
| online 重组适用性 | 本次原 sparse MLA 编译已真实触发，lowp cast/rescale 需合同判定 | M1：保留优化并建立数值资格与可关闭的完整路径 |
| 关闭 online 后 shared IR 不能重读 | 本次原生产输入编译已复现 | M2.1–2.2，作为 M1 正常路径的前置修复 |
| GPU mapping Pure op 保活 | 静态代码确认隐藏存活依赖 | M2.3–2.4：归实际 owner，删除例外 |
| TileLang 普通 RMSNorm 关系失败 | 本次重新编译原 rms_norm_f32，仍在 form-memory 失败，见下文 | M2.7–2.8：追到产生错误/缺失轴关系的变换，不在 serializer 猜补 |
| GPU workspace host 依赖 | IR/runtime 主干已改，上一轮无新的完整设备覆盖 | 在 M2/M3 中使用确实保留 workspace 的既有 production case 补必要运行 |
| Weft host view descriptors | 上一轮已修 base/offset，matvec source 可生成；不等于新增 native 运行通过 | M4/M6：用原 Weft 环境与 case 完成 native/runtime 验收 |
| BANG C/MLU 更新后的链路 | 早期有设备结果，最近 ABI 变更的新增证据主要是 source generation | M4：当前代码在可用 MLU 上跑原受影响 case，先核真实设备/SDK |
| TileLang 其它历史失败 | 既有 CSV 有多个 stage 的失败，不能假定已全部修复，也不能假定全部仍存在 | 按共同原因和原 case 收口，M2/M5/M6，不另造平行矩阵 |
| CPU FP8 numerical_failed | 既有 mojo-x86.csv 保留该结果，未放宽容差 | M4/M5：按原输入和容差查清数值合同/实现，不被重构成绩覆盖 |
| CPU run_only 项 | 部分原条目缺同合同 reference 或已有算法差异 | 保留真实性质；不自动升级为 pass，不通过改容差或参考算法制造成绩 |
| 优化资格/未采用原因 | 现有阶段/IR 诊断已可用，统一解释层未完成 | M1.7/M5 |
| 作者 backward 框架注册 | 独立 backward kernel 与 forward/backward 示例存在，正式框架旅程还需完成 | M3.4；不等同于新增自动求导编译器 |
| 根许可证/可重复分发 | 源码包和 wheel 基础完成，维护者许可证及分发入口未闭合 | M6.1–6.4，并行推进 |
| 新 SDK/ABI 维护合同 | 当前各 adapter 边界清楚，升级旅程未形成完整产品规范 | M6.5–6.6 |

### 11.1 本次补核的 TileLang RMSNorm

为避免把历史错误直接当作当前缺陷，本次通过现有公开 CLI 编译 [rms_norm_f32](../examples/kernels/normalization/rms_norm.py):28–43。该算法正是 [TileLang production adapter](../experiments/gpu/providers/tilelang/normalization.py):35–54 使用的定义；未修改算法、输入合同或容差，未运行设备。

结果仍为 provider_pipeline 失败：

~~~
'intent_gpu.store' op
TileLang scalarization cannot recover a missing non-unit physical axis
TileLang transformation failed: intent-tilelang-form-memory
~~~

source/target 的相应 axis map 分别涉及 derived source 22 与原 source 6。完整当前 diagnostic 在 /tmp/intentdsl-roadmap-tilelang-rms.json；编译目录为 ~/.cache/intentdsl/roadmap-review/compiler/9df31b11d4884faab437a207b1837318/。

这是一个普通归约加 pointwise 输出的当前编译问题。后续要找到关系在哪个变换中丢失或没有被正确表达，不能要求作者改写 RMSNorm，也不能由终端 scalarizer 仅凭相同 shape 猜等价。它同时让 M2 的“IR 真正有作用”具备具体的生产验收对象。

## 12. 验证与交付方式

### 12.1 每组任务采用相同的完成逻辑

1. 依据原生产程序、当前 IR 和 ref 代码确认问题，不先写抽象框架。
2. 在正确层实现；相关旧路径、重复 authority 和临时兼容代码一并删除。
3. 构建必要 compiler/provider；用原输入观察有关 IR、source 和诊断。
4. 运行真正受影响的既有生产入口，保留原算法、规模、dtype 和容差。
5. 同一机器的性能计时避开相互干扰；准备与编译可以合理并发。
6. 更新原实验组既有结果，简明记录实际成功层次和剩余项。
7. 提交一组连贯改动，直到整个里程碑的用户/开发者结果成立。

不新增测试目录、pytest、fixtures、独立边界/数值/压力/组合矩阵；不编写临时脚本绕过这一约束。自动化复用已有构建、公开示例、CLI/MCP 和 production registry。若未来另行扩大验证体系，应单独决定，不能在本 roadmap 中偷偷加入。

### 12.2 验证量与变化范围匹配

| 改动 | 合理的必要验证 |
|---|---|
| 公共 ABI、返回/alias 合同 | 对受影响 family 选择已有输入/输出/可变参数 case，检查实际 native 调用 |
| IR owner、type staging、局部 canonicalization | 已有 production 的 shared 保存、标准优化、重读与续编译，必要原运行 |
| 数值模式或 online | 原生产输入下的前后结构与既定容差；明确当前模式改变的数值关系 |
| provider API/SDK 升级 | 已有 provider registry 的相关原程序，按实际依赖验证 JIT/launch |
| 二进制/安装 | 选定平台的真实 wheel 安装与原公开调用，离开 checkout |
| 完整产品交付 | 已承诺支持范围对应的现有 registry 性能运行；不扩大矩阵 |

不是每次内部修改都全量重跑；也不把少数用例通过扩大成所有模式通过。原数值失败、run_only、SDK 不可用和性能差距分开处理。

### 12.3 里程碑交付应报告什么

- 用户现在能完成的完整任务或开发者能够复用的实际能力；
- 解决的横向问题、删除的旧路径及责任边界；
- 该组提交列表和实际代码增删，文档/CSV/生成产物不冒充实现规模；
- 生成、原生编译、运行、数值、性能分别有哪些证据；
- 仍未完成的具体任务及下一位置；
- 不自动 push，不自动对外发布，不新增版本号或 CHANGELOG。

普通节点完成只报告节点完成。一次失败调查、一个新类型、一个 matcher 命中、几百行移动或几十次运行，都不单独定义为宏观里程碑。

## 13. 建议立即开始的执行批次

### 第一批：M1/M2 编译闭环与 M3/M6 产品任务并行

编译器线程：

1. 修复 boundedTraversalChunk 产生非法 fragment extent 的 staging；
2. 收口 online 数值资格和正常路径，随后提供实际可用的优化控制；
3. 处理 mapping owner/Delinearize 保活及对应普通正规化；
4. 解决原 TileLang RMSNorm 的完整轴关系链。

产品线程：

1. 把公共返回与 alias 合同提案一次明确；
2. 完成作者 backward、prepared/offline/IR 工具的用户旅程；
3. 选定二进制构建基线，准备现有入口的自动交付；
4. 维护者决定许可证和外部分发材料范围，其余工作继续进行。

不等待上述全做完才提交；按依赖形成可审查、可验证的连贯提交，最终以完整里程碑条件收束。

### 第二批：M4/M5 组合能力和默认程序质量

- DSA 矩阵两路线合并是主要横向架构任务；
- CPU/DSA 共同 product/axis/访问知识随之收口；
- 资源决策和优化 remarks 接入实际消费者；
- 从既有程序选择已经暴露的结构问题，改一套可复用机制，而不是继续逐题微调。

### 第三批：M6 产品维护基线与后续扩展

- 汇集已完成用户/API、compiler、provider、安装和原运行证据；
- 固化实际支持的交付组合及升级处理方式；
- 用外部开发者的真实扩展任务验证模块化；
- 然后按需求推进第二 GPU vendor、更多 provider 的分发/框架支持与新的优化能力。

六组任务完成后的目标是**可持续使用、扩展和维护的成熟产品基线**。这与 Triton/TileLang 的工程成熟度对齐，同时保留 Intent 跨执行模型、较高层算法表达的独立价值。后续新增硬件或算子不应再反复推倒 pipeline、ABI、owner 和调用接口。

## 14. 需要明确的设计选择与本轮边界

以下只有真正改变公开合同的部分需要明确，普通模块划分、验证方式和实现细节不逐项请用户审批：

| 选择 | 本报告建议 | 实施影响 |
|---|---|---|
| 数值优化策略 | 默认遵循现有 source 合同；额外 normalized/online 重组提供明确许可，优化开关与许可分开 | 统一 request/IR/provider/cache；正式数值规格随已确认设计更新 |
| run/result 返回 | 只返回 Out；零 None、单值、多 tuple；InOut 通过原对象观察 | 统一四类 runtime 的公开行为和调用文档 |
| named alias group | 同组共享 allocation，不要求访问重叠；不同组不推断 disjoint | 各 runtime 统一检查，避免隐含优化假设 |
| 开源许可证与分发边界 | 由维护者选择；第三方 notices 与实验基线另清范围 | 影响公开发布，不阻断其它实现 |

这些提案作为下一轮具体实施的输入；本次不擅改 doc，也不把尚未确认项包装成已实现合同。

本次已经完成：当前代码全链路分工审查；本地 Triton/TileLang 具体机制对照；旧工作包结算；online 默认/跳过路径和 TileLang 原 RMSNorm 的编译观察；分组任务、依赖、删除路径、验证与产品完成标准的重写。没有修改实现、运行设备或刷新性能结果。

## 附录 A：本轮可复查的编译观察

临时目录仅用于复查本次调查，不成为新产品接口或实验数据权威。持久复现入口是原作者源码、现有 CLI、指定 target facts 与标准 pass 工具。

### A.1 Online 触发

源：examples/kernels/streaming/mla.py 的 token_sparse_mla_value_prefill。先用现有 intent compile --stage kir 生成 KIR，再用 intent-compile --target=cutile --stop-after-shared 和标准 IR 打印选项观察 intent-gpu-co-realize-online-reductions 前后。

本次指定目标事实：

~~~
compute-units=170
shared-memory-per-unit=102400
max-dynamic-shared-memory-per-block=101376
registers-per-unit=65536
max-threads-per-block=1024
compute-capability-major=12
compute-capability-minor=0
single-to-double-precision-perf-ratio=64
matrix-units=true
dynamic-vector-width=false
~~~

这些是指定的编译输入，不表示本轮检测了当前设备状态。输入产物：~/.cache/intentdsl/compiler/e039dccf5d414c298d19d57987a86e56/input.mlir。

IR 日志 /tmp/intentdsl-online-review-sparse-mla.log：296 行原 record；372 行 after 起点；374–375 行 chunk 参数；566 行四 carry loop。

### A.2 跳过 online 的剩余 shared pipeline

输入为标准 IR tree dump 中该 group 的 before 快照。使用现有 intent-opt，运行：

~~~
builtin.module(
  intent-gpu-realize-region-folds,
  intent-gpu-realize-region-scans,
  intent-gpu-realize-reductions,
  intent-gpu-realize-contractions,
  intent-gpu-compose-realized-accesses,
  intent-gpu-schedule-private-stores,
  intent-gpu-vectorize-buffer-loops,
  intent-gpu-promote-buffer-values,
  intent-gpu-refine-program-mapping,
  intent-gpu-simplify-range-predicates,
  intent-gpu-eliminate-common-values,
  intent-gpu-materialize-configurations,
  intent-gpu-fuse-independent-traversals
)
~~~

输出 /tmp/intentdsl-online-review-sparse-mla-without-online.mlir 后，以相同 target facts 和 --input-stage=shared 继续生成 cuTile。解析失败位置 193:69；日志 /tmp/intentdsl-online-review-sparse-mla-provider.log。没有修改 pipeline 实现、作者源码或数值数据。

### A.3 TileLang RMSNorm 当前编译

现有 CLI compile 命令使用 examples/kernels/normalization/rms_norm.py:rms_norm_f32、--target tilelang、相同指定 GPU facts、当前 intent-compile，输出 JSON。未 materialize、launch 或测量。当前错误、源码位置和缓存入口见 §11.1。
