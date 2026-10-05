# IntentDSL 产品形态、Pass 资产与成熟化路线图

更新：2026-10-05。本轮依据当前源码、正式规格、本地 Triton/MLIR/目标实现、现有 examples 与既有运行记录重新制定。本文原位替换上一轮 P1–P4 工作清单；已经完成的基础不再列为待办。它是产品决策与实施路线，不替代 doc/ 的语言和 IR 规格，也不宣布所有后端已经成熟。

## 1. 判断：产品可以做成，但必须改变推进单位

**现有架构足以继续做成一个有用、可维护的多后端编译器产品，没有证据要求推倒重来。** KIR、分 execution family 的物理程序、目标实现、标准 MLIR pass manager、产物和 runtime 边界都已有真实实现；多个后端已有完整运行及性能成果。问题集中在优化组件的组合合同、覆盖与收益判断、开发者可发现性，以及公共案例到完整调用之间的产品体验。

目前也不能宣称已经达到 Triton 的成熟度。已有几个高速程序不代表新的程序组合都能获得相同质量；已有工具入口不代表陌生开发者能顺利贡献优化；已有后端注册不代表任意合法程序都完成 lowering。后续以这些实际能力收口，停止把一条更快 CSV、一个 helper、一次目录拆分称为完整里程碑。

目标需要有边界：**在明确维护的 provider、工具链版本和能力范围内，提供可靠的作者入口、可组合的优化开发入口，以及持续扩大的高质量程序覆盖。** “支持所有未来硬件和 SDK，并让任意算法自动达到专家上限”没有有限完成条件，也不是 Triton 已经兑现的保证。不把这个无限目标当作 IntentDSL 是否可行的判断标准。

对当前框架的判断可以被推翻。如果后续仍持续出现以下情况，应停止补局部 matcher，修改相应 IR 或阶段边界：

- 同一语义规律换一个普通写法就必须新增算子专用路径。
- 新优化必须让 serializer 或 runtime 重新猜测算法、ownership、mask、carry 或工作区。
- 共享分析只能回答最初样例的问题，新的消费者不断复制并略改同一套证明。
- 一个完整 pass 的正确性依赖调用者额外记住未声明的 repair 顺序。
- 新后端只能接受针对它重写的作者算法，且差异不是实际 capability 或已声明数值合同造成的。

这五项是架构反证，不是要求每次小改动都启动全库重构。

## 2. 理想产品：算法入口与优化入口共同成立

### 2.1 面向算法作者

作者用 Intent DSL 表达算法、逻辑访问、数值合同、状态和显式多 kernel 编排；在 host 选择目标，通过普通 Python API 得到可调用 artifact。编译器负责形成该 execution family 的执行结构，目标编译器继续完成自己的机器布局、指令和调度。

作者无需为某个 pass 改成专用模板，也无需在 kernel 内判断 Triton、CPU 或 MLU。作者可以检查 IR、源码、选中配置和失败阶段，但普通使用不要求先学编译器内部层次。

这条入口已经存在：[compile / generate / compile_ir](../python/intent/compiler/pipeline.py)、[公开调用示例](../examples/README.md)、[安装入口](../environment/install.py)。下一步扩大完整场景覆盖和诊断质量，不重新造 JIT、Torch adapter 或安装系统。

### 2.2 面向优化开发者

优化开发者应能完成以下闭环：

1. 找到当前算法对应的 IR 阶段和可用事实。
2. 选择一个现有优化组件，或按同一接口贡献新组件。
3. 使用标准 pipeline 组合、运行、替换适用的可选优化，并检查前后 IR。
4. 将闭合的优化后程序交回同一 provider 生成与运行路径。
5. 在既有案例及一个未为该改动定制的相近案例上，观察同一规律是否成立。

现有 [intent-opt](../tools/intent-opt/intent-opt.cpp)、[optimize_ir / generate_from_ir](../python/intent/compiler/pipeline.py)、[compiler MCP](../python/intent/tools/compiler_mcp.py) 已经提供机械入口；[CONTRIBUTING 的编译调度部分](../CONTRIBUTING.md#编译调度与独立-ir-工具) 也已有说明。**本轮审查纠正上一轮解释中不充分的地方：pass 开发入口不是空白，不能再次作为从零建设项目。**

真正要补齐的是：用户能否看懂某个组件接收什么程序、依赖哪些事实、产出什么程序、是否可选、适用范围在哪里，以及该从哪个公共接口复用它。现有命令能执行某个 pass，不自动说明这些产品问题已经解决。

### 2.3 多后端的共同价值

共同价值是作者语义、可复用证明、成体系的优化规律、统一产物与调用体验；GPU/CPU/DSA 分别形成适合自己的物理程序。

例如“同一存储版本的稳定输入只准备一次”是共同知识：

- GPU 可以保留 fragment SSA，并根据实际 alias 条件守护读取。
- CPU 可以形成共享 panel、向量值或 task 内的完整遍历。
- DSA 可以复用 NRAM/WRAM 内容，并保持 transfer completion 和资源生命周期。

三者不必执行同一段 C++ 改写。Triton/cuTile 应消费共同 GPU 程序；Mojo/Weft 应消费共同 CPU 程序，但具体微实现不强求相同。某个能力已由下层 compiler 完成时直接使用，不为体现 Intent 的工作量复制它。

## 3. 当前真正的产品缺口

| 已有基础，直接复用 | 尚需完成的产品问题 |
|---|---|
| 标准 MLIR pass 注册、PassManager、IR 打印和计时 | 完整组的输入/输出合同与可发现性不一致；部分依赖仍只能读实现得知 |
| KIR/shared 导出、独立 optimize、IR 续生成、MCP | 把这些入口串成清楚的优化贡献流程；中途 dump 与闭合的可续编译 IR 必须分清 |
| GPU/CPU/DSA 分层与 provider 路径 | 消除跨阶段重复证明、规范化时点遗漏和孤立的 target 特例 |
| alias、坐标、存储版本、replay、资源查询 | 扩展实际共同消费域，并保证 mutation 后失效；不能只增加无人消费的分析接口 |
| CPU 目标 implementation registry、GPU autotune | 完善结构与供数质量；不重建微实现框架或 winner 选择器 |
| examples 中大量目标无关算法 | 精选约 30 个完整产品场景，补齐 host 编排与使用导航，避免用户必须读实验 adapter 才能调用 |
| 原生产入口、数值检查与 CSV | 改为产品能力驱动的必要观察和自比较，减少追单个 reference 的连续调参 |

三个需要实际修改的组合问题：

1. **GPU：可选的存储/作用域优化与一般值清理的边界。** [CommonValues](../lib/Dialect/GPU/Transforms/Value/EliminateCommonValues.cpp) 已同时承接轻量规范化与稳定读取移动。需要明确哪些工作属于必要的关系闭合，哪些是可选收益决策；有独立闭合合同的组才考虑拆成具名优化 pass。
2. **CPU：规范化后的普通计算必须进入同一批消费者。** [NormalizeReductions](../lib/Dialect/CPU/Transforms/Collective/NormalizeReductions.cpp)、[ProducerVersions](../include/Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h)、遍历融合与目标实现已经能组合；仍需减少因表示、buffer 版本或阶段时点导致的断点，不能为每种 normalization 再写实现。
3. **DSA/BANG C：局部存储组合依赖顺序过强。** [composeLocalProgram](../lib/Target/BangC/Transforms/Supply.cpp) 串接规范化、供数、存储复用和收益判断。需要明确每组产生的事实和完成边界，局部 fixed point 由责任组拥有，避免每个新机会都靠往大函数中插一次调用才能生效。

现有骨架不需要再造一份统一执行计划。改进应落在当前 IR、标准接口和真实消费点。

## 4. Pass 作为核心资产，应怎样模块化

### 4.1 一项资产包含规律、证明、改写和消费

一个可复用优化组件需要回答：

- **规律**：减少哪一种真实工作，或形成哪一种可由目标高效消费的结构。
- **合同**：输入 IR 阶段、数值许可、effects、坐标、alias、lifetime、目标能力等前置条件。
- **改写**：真正改变哪些 loop、buffer、access、carry、fragment 或 task。
- **结果**：完成哪些关系闭合；哪些分析失效；后续哪些消费者能直接使用。
- **收益边界**：增加的存活值、准备成本、同步、工作集和并行度可能带来什么代价。
- **复用范围**：适用于哪类语义结构；哪些相近结构尚未覆盖，以及原因。

这些内容分别归入现有 ODS description/options、公共头接口、实现和贡献说明。不得另造一份同时解释程序的规则数据库、schema 或 side plan。

### 4.2 三种组件，三种公开方式

| 组件 | 入口与边界 | 是否独立作为 pass |
|---|---|---|
| 分析/证明与机械构造 | 公共 include 中同责查询接口；读取当前 IR，或按明确绑定机械构造 | 查询本身不冒充 pass |
| 完整语义保持的优化组 | 注册的 MLIR pass，有明确前置与后置条件，独立完成改写与 closure | 可以在满足合同的阶段组合、运行和替换 |
| 目标局部实现 | 既有 implementation/provider 接口；承接明确 block、dtype、stride、accumulator 与 effects | 按实际责任作为实现或 legalization，不强包装成公开优化 pass |

局部 matcher、首个 rewrite、repair helper、预算事务可以继续留在 lib 的私有头。跨库或明确供开发者复用的接口进入 include。文件夹表达职责；不是所有函数公开、所有头文件移动到 include，才算模块化。

### 4.3 插拔的准确含义

- 必要 construction、bufferization、legalization 与关系闭合承担可执行性责任，不能作为普通性能开关任意删除。
- 可选优化关闭后，应仍保留同语义的合法程序；若不存在合法未优化实现，就应说明它是必要 lowering。
- 两个 pass 的组合取决于输入/输出合同，不承诺任意顺序都合法。
- 先使用已有 intent-opt、标准 pipeline 语法和 ODS options。只有多个真实调用者需要同一闭合组合时，才增加相应命名 pipeline。
- 不另造 Python pipeline 编排框架或 Intent scheduling DSL。进程内注册与源码级扩展先做好；跨版本动态插件 ABI 不作为本阶段前置条件。
- 普通用户默认调用仍简单；专家和 agent 可进入 IR/pass 层。优化开发入口不变成作者必须手写 schedule 的义务。

### 4.4 资源与目标知识也要有明确归属

能力事实决定“能不能表达”；effects/lifetime/shape 决定“是否合法”；收益判断决定“值不值得采用”；最终布局与分配由对应的 target passes 或 native compiler 承担，已有候选机制负责选择 winner。这四者不能混为一个万能 resource pass。BANG C 的 NRAM/WRAM/SRAM arena 和 WRAM 布局由 Intent target passes 绑定，当前 runtime 执行固定 entry，没有候选 autotuner；不能把 Triton 的分工套在它身上。

GPU nominal payload 不等于 occupancy；CPU private_bytes 不等于全部 cache；MLU arena-fit 不等于更快。结构选择要使用实际工作量、存活范围和准备/复用成本。结果仍写进当前 IR，估算和诊断不成为第二份程序。

## 5. 本轮选定的 30 个核心产品场景

这里选的是 **30 个场景，不是 30 个内置算子实现**。多 kernel 场景保留作者显式编排，因此实际 kernel 定义数大于 30。选择依据是用户价值与不同语义/执行结构，全部来自现有 examples；没有生成新算法来迎合当前 matcher。

| # | 场景 | 已有算法入口 | 主要展示的能力 |
|---:|---|---|---|
| 1 | ReLU | [relu_forward](../examples/kernels/activation/pointwise.py) | 基础逐元素、读写合并、尾部 |
| 2 | SwiGLU | [swiglu_forward](../examples/kernels/activation/swiglu.py) | 多输入 producer、数学函数与融合 |
| 3 | Q/K RoPE 原地更新 | [rotary_qk_inplace](../examples/kernels/position/rope.py) | 坐标变换、共享值、多个 InOut |
| 4 | 矩阵转置 | [matrix_transpose](../examples/kernels/layout/transpose.py) | 访问方向、局部布局与搬运 |
| 5 | Embedding lookup | [embedding_forward_lookup_bf16](../examples/kernels/backward/embedding.py) | 数据相关 gather、行供数 |
| 6 | CSR SpMV | [csr_spmv](../examples/kernels/sparse/csr_spmv.py) | 稀疏访问、变长归约 |
| 7 | Jagged mean | [jagged_mean](../examples/kernels/ragged/jagged_mean.py) | 分段范围、free/reduction 轴与 SIMD |
| 8 | FP16 softmax | [stable_softmax_f16](../examples/kernels/normalization/softmax.py) | producer→max/sum→consumer 的整链 |
| 9 | 加权 LayerNorm | [weighted_layer_norm](../examples/kernels/normalization/layer_norm.py) | 共享统计与 epilogue；保留原二阶矩算法 |
| 10 | 训练态 BatchNorm | [batch_norm_training](../examples/kernels/normalization/batch_norm.py) | Welford/tuple、辅助输出、状态更新 |
| 11 | GroupNorm backward | [dx + weight_bias](../examples/kernels/backward/group_norm.py) | 嵌套归约、跨 consumer 复用、显式双 kernel |
| 12 | 行 cumsum | [row_cumsum_f32](../examples/kernels/streaming/ordered_prefix.py) | 保成员顺序的 prefix |
| 13 | 因果线性 attention | [causal_linear_attention_f32](../examples/kernels/streaming/attention_f32.py) | region_scan、矩阵状态与 emit |
| 14 | Nonzero compaction | [compact_nonzero_rows](../examples/kernels/compaction/nonzero.py) | prefix→索引→唯一写入 |
| 15 | MoE alignment | [count/prefix/scatter/mark](../examples/kernels/routing/moe_align.py) | 整数计数、路由及显式四阶段调用 |
| 16 | FP16 GEMM | [gemm](../examples/kernels/contraction/gemm.py) | 双侧供数、分块、accumulator |
| 17 | Batched GEMM | [batched_gemm_nn](../examples/kernels/contraction/batched_gemm.py) | batch/free/reduction 轴组织 |
| 18 | Ragged grouped GEMM | [ragged_grouped_gemm](../examples/kernels/ragged/grouped_gemm.py) | 不等工作量、分组与 panel 复用 |
| 19 | INT8 GEMM + bias | [gemm_i8](../examples/kernels/contraction/gemm.py) | 整数 accumulator、矩阵能力与融合 |
| 20 | Q4_K projection | [quantized_projection](../examples/kernels/quantization/quantized_projection.py) | 量化格式合同、decode 与专家微实现 |
| 21 | Block-scaled FP8 GEMM | [block_scaled_matmul](../examples/kernels/contraction/block_scaled.py) | scale/group、目标原生能力与明确支持边界 |
| 22 | BF16 attention | [flash_attention_bf16_fwd](../examples/kernels/streaming/attention.py) | 收缩与统计组合、分块与存活范围 |
| 23 | Paged GQA decode | [partials](../examples/kernels/streaming/paged_attention.py) + [f32-to-f16 merge](../examples/kernels/streaming/splitk_reduce.py) | page gather、分段状态与作者显式 merge |
| 24 | Mamba chunk state | [mamba_chunk_state_bf16_fwd](../examples/kernels/streaming/mamba.py) | 数学 producer、矩阵供数、局部状态 |
| 25 | Gated delta recurrence | [recurrent_gated_delta_fwd](../examples/kernels/streaming/gated_delta.py) | 有序 tensor state、读写版本与驻留 |
| 26 | 因果 depthwise convolution | [causal_depthwise_conv1d](../examples/kernels/convolution/direct.py) | 滑动窗口、邻域复用与边界 |
| 27 | 小批量 Cholesky | [batched_cholesky_lower](../examples/kernels/factorization/cholesky.py) | 有序依赖、变长内循环、原地状态 |
| 28 | Dropout | [xor_shift_dropout](../examples/kernels/regularization/dropout.py) | 原作者整数混合/位运算、mask、cast |
| 29 | AdamW | [adamw_update](../examples/kernels/optimization/adamw.py) | 多数组 mutable state、共享 producer |
| 30 | Histogram | [histogram_256](../examples/kernels/statistics/histogram.py) | 重复地址、统计更新与冲突语义 |

### 5.1 这 30 个场景怎样成为产品入口

算法定义继续唯一保存在 examples/kernels。公开 host 用法从现有 API 和已有完整 callable 提炼，保留 multi-kernel 编排、Out/InOut、constexpr、dtype 和输入约束；不再复制一份算法，也不让用户从 reference 文件拼调用。

普通调用示范归 examples；性能/数值运行继续复用所属 experiments 的现有入口。选择列表只参与产品导航与运行调度，不能传入 compiler 作为优化分派键。不在 examples 中再造一套 benchmark runner 或 reference 副本。

每个场景应能说明：做什么、输入输出、作者控制的组织、如何选目标、如何查看产物，以及哪些目标组合已运行。小型使用示范与性能输入明确区分，不用缩小问题来宣称性能完成。

### 5.2 后端覆盖不能写成假对称

当前 GPU 产品线是 Triton/cuTile，CPU 是 Mojo/Weft，DSA 是 BANG C/MLU370。TileLang main 后端不恢复。

30 个场景不自动意味着本轮做 30×5 的新实验矩阵，也不声称目前每项都能在所有目标运行。后续在原入口中逐项分清：

1. 已有完整运行证据。
2. 已明确查证目标能力不满足。
3. 语言合法，但现 lowering/目标实现尚缺，属于 compiler 欠账。
4. 尚未验证。

未实现不能冒充硬件不支持；没有性能测量不能冒充性能良好。比如当前 MLU 的 13 个实验入口只是已观察范围，mtp_372 是当前实现 profile，不是所有其它 MLU 都不可能；Weft 的已有六项成功也不代表一般 BF16 view/task 都已闭合。

阶段收口要求适用组合能正常使用，并明确剩余 compiler 缺口；不能把困难案例从这 30 项悄悄删除以制造完成率。

## 6. 产品驱动的观察与自比较

### 6.1 主比较对象改为自己的程序质量

每项优化先提出可解释预期：哪些重复读取/准备消失，哪个中间物化不再需要，哪些独立状态共享遍历，或哪个串行依赖被合法组织。随后只运行实际受影响的既有生产入口，比较同一算法、输入、dtype、数值选项与计时边界的前后版本。

可以用同类已有程序的有效带宽、计算量、并行度和工作集判断某个实现可能很差；这属于选题依据。估算明确标为估算，不写成测量值，不拿不同规模的原始毫秒直接比较。

外部 reference 继续有价值，但不再成为每次推进的必需性能对手。需要校准数量级、辨别算法组织差异或解释异常时再查。**数值 reference 与性能 reference 分开：不要求每轮竞争外部性能，不等于放弃原有正确性检查。**

自比较保存一个当前结果输出；前一有效版本由 Git 中的原表、原运行日志和实际产物定位。既有 CSV 的 source/ratio 若表示外部 reference，不偷偷改作自身前一版本的含义，也不再建立平行结果库。

### 6.2 泛化观察使用 30 项之外的现有近邻

选择依据是同一语义合同或同一物理规律，而不是名字相似。下面是候选近邻，使用前仍核数值、成员域和 effects，不预先宣称它们都已受益：

| 核心场景/知识 | 现有集合之外的近邻 | 要观察的泛化 |
|---|---|---|
| LayerNorm/softmax 的共享 producer | RMSNorm、fused-add RMSNorm | 共享读取与保留/重放规律，不要求统计公式相同 |
| SwiGLU producer 融合 | SwiGLU helper 写法、GEGLU | helper/表达式组织变化后能否采用同一证明 |
| 普通 GEMM 的供数与 epilogue | QKV projection、gated dual GEMM | 多 consumer 的输入准备是否复用 |
| Jagged mean 的 free/reduction 轴 | nested jagged mean、CSR SpMM | 层级/稀疏访问变化后的合法范围 |
| RoPE 坐标关系 | 等价索引写法、partial RoPE | 坐标组合不依赖原 AST 写法 |
| 状态/矩阵首次写入 | chunk gated delta、Mamba state passing | 当前版本、初始化与完成事实能否复用 |

每个知识包挑必要近邻即可；不对 30 项和所有近邻反复全量重测，不修改近邻算法来迎合 pass。原 fixed agent 实验的首次提交成绩仍独立保留，产品开发的修复循环不回写其成绩。

### 6.3 性能与复用都需要诚实归因

- IR 更紧凑不自动等于更快；检查是否转移了主成本或增加了存活值。
- 一个目标上有效、另一个没有盈利，先解释执行模型差异，不强求相同物理改写。
- 有意义的可复用结构可以在某个案例持平；不能据此宣传稳定加速。
- 明确退化的默认选择要修正或撤回，不能以“模块化更好”为由忽略用户性能。
- 编译/JIT/调优、热 kernel、host/框架调用分别归因；不在性能 pass 任务中无声修改 runtime 来改变结果。

## 7. 下一阶段的六个连贯工作包

每个工作包可以有多个提交。完成标准是完整能力与使用路径，不是代码行数、pass 数量或测试数量。

### M1：把现有 pass 入口收成可使用的开发者产品

**结果：陌生开发者能找到正确阶段、复用已有事实、增加一个闭合优化，并通过同一工具链运行它。**

- 沿 CPU 已有 ODS description 补齐 GPU/DSA/target 完整组的前置、后置、依赖事实和失败边界；声明与代码同源，避免再维护一份独立 catalog。
- 梳理必要 lowering、可选优化和组内 helper。只有具有独立完整合同的组才公开；可选决策与必要 closure 混在一起的地方做实际分离。
- 给出利用现有 compile_ir、intent-opt/optimize_ir、generate_from_ir 的完整贡献路径；中间快照必须带上需要继续运行的现有组，不假装所有 dump 都可直接 materialize。
- 公共查询与构造接口进入相应 include，private implementation 留在 lib。新增普通优化应主要落在该组件、现有 pipeline 接线及必要分析消费者中。
- 使用现有 IR 打印、remarks 和错误阶段说明“不适用、资源不合适、目标不支持、native 编译失败”，不重建诊断框架。

落点：[GPU Passes.td](../include/Intent/Dialect/GPU/Transforms/Passes.td)、[CPU Passes.td](../include/Intent/Dialect/CPU/Transforms/Passes.td)、[DSA passes](../lib/Dialect/DSA/Transforms/Passes.cpp)、[BANG C groups](../lib/Target/BangC/Transforms/Legalize.cpp)、[现有开发指南](../CONTRIBUTING.md)。

结束时应能展示一个新增优化通过现有入口完成贡献与组合，不需要新增 frontend/runtime 分派。不是仅补完一张说明表。

### M2：收束三组可复用优化知识

**结果：以完整知识域组织优化，并删除被替代的局部路径。**

| 知识域 | GPU | CPU | DSA |
|---|---|---|---|
| 稳定版本、供数与作用域 | alias-aware 读取、SSA snapshot、多 consumer 与 live scope | producer/current buffer version、task/cohort 供数 | completed local version、显式 transfer lifetime、原生准备复用 |
| producer→collective→consumer | typed tuple/轴规范化、native reduce/scan 消费 | 嵌套归约规范化、共同遍历、SSA/SIMD carries | typed state/identity、原生 collective 与局部消费者 |
| contraction 初始化与输入/输出流 | 原有 blocking、accumulator/epilogue、provider native form | 双侧准备、完整 C 工作集、微实现需求 | 首次写入、NRAM/WRAM supply、后续累积与 completion |

每组明确合法性查询、改写 owner、收益选择、后续消费者与分析失效。一个 family 已经完成的部分直接复用；不为了凑三条线制造改动。增加的条件必须来自 current typed semantics、effects、坐标和能力，不能来自 kernel 名、模板节点数量或某个测量阈值。

结束时，核心场景和必要近邻实际采用同一规律；既有模块之间不再各自维护同责证明；不能只交付“公共 helper 已抽出来”。

### M3：交付 30 个场景的完整使用体验

**结果：用户以 example 为入口使用产品，实验脚本不再承担主要使用文档的职责。**

- 按 §5 固定首批场景及真实源码入口；多 kernel 场景保留完整 host 编排。
- 复用现有 artifact/target API 补齐可运行 host 用法和简洁导航。输入格式、Out/InOut、准备与热调用说明清楚。
- 实验 adapter 与公开调用共享应共享的 callable，reference 与 timer 仍在 experiments；不复制 algorithm、runner 或后端调用机制。
- 明确每个适用目标的依赖与状态，使用现有 doctor/安装入口。ABI/toolchain 问题按所属 provider 解决，不让用户改 kernel 迁就。

不要求本轮创建 30 份新测试或立刻全测全部目标。实施期间沿既有生产输入完成必要运行，逐步形成真实支持状态。

### M4：让默认结构具有可解释的性能质量

**结果：产品默认程序质量持续提升，而非对个别 CSV 反复试错。**

- 使用核心场景的结构性工作量和自比较选择主问题，优先处理重复全域遍历、重复准备、无解释的串行慢路径与过大存活工作集。
- 合法性、profitability 与 native resource 分配分层。保留有意义的候选，不通过扩大搜索掩盖缺少的程序结构。
- 每个知识包只补足必要的实际收益证据；数值失败、编译失败、波动和持平都如实保留。
- 外部 reference 做抽样校准与根因分析；明显长尾不能永久用“算法组织不同”搁置，也不能为了追 reference 换掉作者算法。

结束时能解释主要性能差距由哪些程序结构造成、哪些已解决、哪些受目标能力限制。没有固定“所有案例统一 1.2×”作为成熟度替代品。

### M5：闭合已暴露的支持与交付欠账

**结果：v1 的承诺范围准确，合法语言缺口有清楚归属。**

- CPU/Weft 已发现的 BF16 view、task/structured consumer 等缺口，先区分硬件事实与 compiler 实现；不能用六个已成功入口覆盖全部承诺。
- 原 Mojo FP8 数值失败、cuTile softmax-backward 历史失败仍需在相应原入口处理或明确状态；不改容差，也不拿历史失败直接推断当前根因。
- 为 DSA 现有 task/group、NRAM/WRAM/SRAM、completion 与异步 lifetime 补足正式合同。先描述并审定现有职责，不能把文档欠账扩成另一套 IR；新跨 completion 调度有明确合同后再做。
- 冷编译、错误可读性、安装与公开分发作为独立产品质量工作。复用已有打包/CI/doctor；许可证与发布决定仍由维护者决定，本路线不代为发布。

这一包不把所有基础设施重新列为待办；只解决实际阻碍产品承诺的具体缺口。

### M6：以贡献与迁移能力收束 v1

**结果：新增算法、优化组件和目标适配有清楚路径，现有用户能持续使用。**

- 新算法复用同一 DSL/compiler，不要求专用 leaf 或改写为已知 example。
- 新优化能复用 proof/IR/implementation 接口，并通过现有 pass 工具进入默认或显式选择的 pipeline。
- 目标 API/ABI/工具链变更主要在对应 adapter、legalization、runtime binding 中处理，已有 family 优化继续有效；不承诺任何版本变化都零成本。
- 30 个场景提供完整产品入口；适用组合、已运行状态、compiler 欠账与硬件限制分清。
- 新开发者或 agent 能根据现有入口和合同完成一次真实贡献，而非依赖维护者口头补充隐藏顺序。

这才是本路线的完成状态。它比“框架已搭好”更具体，也不等同于立即拥有 Triton 多年积累的全部架构、生态和覆盖。

## 8. 对 agent 落地底牌的判断

这个方向有价值，但应作为 **确定性产品之外的实验扩展**，不能成为掩盖缺失 lowering 的默认路径。

### 8.1 可以保留的分工

Pass 负责形成明确的执行结构：ownership、blocking、逻辑布局/供数、融合、状态、生命周期与依赖。已有 provider compiler 负责的 warp/lane layout、机器流水线、指令和寄存器分配继续交给它，不在 Intent 重建。

Agent 可以帮助安装与选择环境、解释诊断、编写 host glue，以及提出和实现工具链/目标 adapter 的修复。更积极的实验可以让 agent 生成或修复目标适配代码，再通过编译、结构检查和真实运行淘汰错误候选。已经确认的通用修复应沉淀回确定性 adapter，而不是让每次 JIT 重新付出推理成本。

### 8.2 需要修正的边界

“理想 kernel”不能是一张只有 tiling/fusion 意图的表。主产品的交付必须是完整、可验证的相应 family 物理程序，包含实际访问、状态、effects 与调用事实。CPU、GPU、MLU 可以实现同一算法，不必共用一份假想物理 kernel。

ABI、dtype 和 launch 配置也不全是无关性能的胶水：stride/alias 会影响访问合法性，accumulator dtype 会影响数值，grid/warps/stages 会影响覆盖和性能。Agent 可以在已声明合同与合法配置范围内适配，不能为了编译通过擅改这些事实。

实际运行是必要裁决证据，但一次数值通过不能证明任意输入上的等价。实验模式应同时使用 IR/ABI/数值合同约束、可审改动和原运行观察，不把 agent 的自我判断当作证明，也不对传统手写后端声称已经具有形式化全证明。

### 8.3 两个可研究的形态与退出条件

1. **维护型 agent**：修复一个 provider adapter/工具链适配，重新运行同一确定性编译链；经过审查的规则成为普通代码。这最容易帮助主产品，也最能摊薄后续维护成本。
2. **实验型目标落地 agent**：从完整物理程序尝试生成目标实现，明确保存候选、改动与运行证据；它属于研究路径，不静默替代失败的主编译器，不改原 agent 实验首次交付成绩。

若大量例子仍需 agent 猜测缺失的 mask、carry、同步、精度或多 kernel 组织，说明输入合同或后端抽象有缺口；agent 并没有消除复杂度。若修复能够跨多个程序复用、正确性边界明确，计入重复生成/编译、人工审查和维护后的总成本下降，且性能结构得到保持，这条实验才值得扩大。

**主产品可以在没有 LLM 服务时编译和运行。** 当前先完成 M1–M3 的独立产品能力，再决定是否开展实验型落地，不立即新建 agent backend 或 runtime fallback。

## 9. 成熟实现的具体参照

| 问题 | 本地参照 | 对 Intent 的要求 |
|---|---|---|
| pass 调度与真实依赖 | [Triton NVIDIA compiler](../../ref/triton/third_party/nvidia/backend/compiler.py):297 起按阶段、架构组织 passes，反复规范化与 CSE | 保留明确 pipeline 责任；不把任意顺序插拔作为成熟标准 |
| 稳定读取移动 | [Triton LICM](../../ref/triton/lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp):22、48、61 对全 loop 只读及零次循环分别处理 | Intent 可在已有 alias/effect 事实下扩适用域；必须保持零次无访问和原读取版本 |
| 循环组合 | 本地 LLVM 20 mlir/lib/Dialect/SCF/Utils/Utils.cpp:1417 起提供 sibling fusion；Utils.h:195 起明确调用者承担合法性 | 复用机械 IR 改写，依赖与存储证明属于 Intent 对应 family |
| 更细粒度 schedule | 本地 LLVM 20 mlir/docs/Dialects/Transform.md:14 起说明 Transform dialect 与 pass 框架的关系 | 不因要“pass 为入口”就新造 scheduling DSL；真实需求出现再选择已有机制 |
| 目标完成与首次写入 | 本地 NeuWare bang_device_functions_decls.h:63395 的 conv、:61259 的 conv_partial；[MLU-OPS msda](../../ref/mlu-ops/kernels/ms_deform_attn/ms_deform_attn_forward/msda_forward_fast_union1.mlu):419；Intent [现有 native 拼写](../lib/Target/BangC/Runtime/TileImplementations.inc):1455 | 准确表达 primitive 的读写合同，让公共存储优化消费，不在 leaf 私下猜初始化 |
| 作者与目标的义务 | [TileLang kernel](../../ref/tilelang/tilelang/language/kernel.py)、[allocation](../../ref/tilelang/tilelang/language/allocate.py)、[CUDA pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py) | 借鉴职责分层；不把其低层 schedule/存储义务强加给 Intent 作者，不恢复 main TileLang 后端 |

本地实现用于确认本项目参考快照的真实行为；网络资料用于后续核对公开版本与 API，不用外部宣传代替实际代码，也不把未安装版本能力当成本机能力。

## 10. 本轮收尾与下一步顺序

当前 GPU 稳定读取、CPU 归约组合、MLU 首次矩阵写入及必要存储证明已完成原入口运行并提交；迁移中发现的 MLU GEMM 冗余填充回退已消除。这些是旧工作包的收尾，不继续列入新路线的待办，也不据此宣布整个产品成熟。

1. 先做 M1，并以 M2 中已有真实组件作为贡献路径的载体；不空建 pass 框架。
2. 并行推进 M3 的 30 个完整产品场景；从实际使用暴露的缺口驱动 M2/M4/M5。
3. 以 M6 收束可维护的 v1；agent 实验单列，不作为收束前提。

本轮收尾结果与提交在 Git 中记录；既有 CSV 保当前观察。后续实施不续写流水账式报告、不新增平行性能表，不再以不断增加的小节点代替上述完整能力。
