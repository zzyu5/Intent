# IntentDSL 可复用优化与多后端性能路线图

更新日期：2026-10-04。调查基线：`main / 2dff8fe4`。本轮原位更新路线图；只调查当前源码、本地成熟实现、已有结果和提交记录，没有修改 compiler、正式规格、作者算法、baseline 或测量程序，也没有运行新 benchmark。本文替换旧的基础设施收口待办，旧调查保留在 Git。

## 1. 阶段切换：从骨架建设转向成体系的性能优化

**下一阶段的主线是：把可复用的优化知识落实为真正改写程序的 passes，使同一份作者算法在 GPU、CPU、MLU 各自的执行模型上获得有竞争力的性能。** 编译时间、文件拆分、分析缓存和局部清理是配套工作，不能继续占据主要里程碑。

之前的问题不在于这些局部工作没有价值，而在于任务选择失去了整体优先级：基础设施已经建立，后续却仍容易沿一个热点连续深入，没有同时回答“这一类优化覆盖了哪些程序、复用了哪些机制、其它后端如何受益、完整算子性能是否改善”。本路线图以优化能力组织工作，不以问题数量、pass 数量、提交数量或修改行数组织工作。

当前可以进入性能主线。旧路线图的五个明确结构任务已经完成对应迁移；不能继续把它们当作开始优化的前置工程。不过，**基础骨架可用不等于所有后端性能成熟，也不等于已经没有任何欠账**。现存数值失败、DSA 规格覆盖和公开发布决定分别保留在 §8，不能反复包装成新的全项目重构。

### 1.1 本阶段的后端范围

| 执行模型 | 产品 provider 与当前硬件线 | 本阶段要求 |
|---|---|---|
| GPU | Triton、cuTile；现有 H100、RTX 5090D 生产入口 | 共同 GPU 优化同时服务两个 provider；分别解决各架构上真实存在的物理程序差距 |
| CPU | Mojo/x86；Weft/RVV 与已有 IME 实现 | 共同 CPU 优化下沉到两条目标链；向量、矩阵和供数实现各按目标能力选择 |
| DSA | BANG C/MLU370 | 在已有 CNCC/CNRT 真实运行链上推进流量、协作、供数和资源优化，并补足同设备性能比较 |

“同时推进”是每个横向里程碑都有三类执行模型的工作与结果，不要求同一天提交、使用同一份物理变换代码或在不支持某 primitive 的硬件上伪造支持。某项规则在一个目标上已由下层实现，应确认直接消费该能力；不能为了凑后端覆盖再写一遍。

TileLang 后端仍保存在 `archive/tilelang-backend`，不恢复到 main。本地 `../ref/tilelang` 继续作为成熟实现参考。新增 provider、新硬件产品线、公共 ABI 变更不混进本阶段。

### 1.2 保持稳定的边界

正常性能工作修改 family passes、pass 私有收益分析、既有参数角色的候选数据，以及现有 target lowering/微实现中的优化；DSL/KIR 语义、公共调用、产物合同、runtime 和安装链保持稳定。目标微实现的改进可以属于性能工作，不意味着引入整算子库旁路。

Pass 必须基于 current typed IR 的 shape、坐标、def-use、数值许可、effects、alias、lifetime 和 capability 决策；执行结果写回真实 types、ops、regions、operands 和 def-use。不能以算子名称、benchmark 身份、设备名称或隐藏 side plan 决定实现。

性能工作发现确实缺少的公共合同或执行事实时，先定位具体缺口，单独完成必要的基础修改，再恢复优化。不得用 serializer 特例、runtime 分支或私有属性绕过，也不因一个缺口暂停其余有完整依据的优化。

依据：[编译边界](../doc/compiler/README.md):20、35–48、60–74；[pass 合同](../doc/compiler/passes-and-analyses.md):5–22、145–167、201–235；[目标 lowering](../doc/compiler/target-lowering.md):25–39、113–140。

## 2. 已完成的基础：移出待办，作为优化入口

| 旧任务或基础 | 当前可复用入口 | 本阶段处理 |
|---|---|---|
| S1 GPU helper 资格与重建同源 | [Helpers 分析](../lib/Dialect/GPU/Analysis/Helpers.cpp):57–131；[PhysicalCloning](../lib/Dialect/GPU/Transforms/Value/PhysicalCloning.cpp)、[ValueMaterialization](../lib/Dialect/GPU/Transforms/Value/ValueMaterialization.cpp)；`0b405b87`、`55675c21` | 直接使用共同 proof/clone/replay，不再为新优化复制 helper 白名单 |
| S2 CPU producer proof 与重放统一 | [ProducerReplay](../lib/Dialect/CPU/Analysis/ProducerReplay.cpp):102、121；[重建接口](../include/Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h):9–20；`e431b692`、`7c13b793` | 扩大有收益的融合域，基础 replay 不重新建设 |
| S3 DSA 执行关系供 pass/verifier 共用 | [ExecutionRelations](../lib/Dialect/DSA/IR/ExecutionRelations.cpp)；`605125cc` | 复用一致性和坐标依赖证明，调度策略仍归 DSA |
| S4 Weft descriptor/capture 来源统一 | [ViewRelations](../lib/Dialect/CPU/Analysis/ViewRelations.cpp):199–227；[Weft Views](../lib/Target/Weft/Transforms/Views.cpp):49–92；`c911fce0` | 在现有目标可表达布局内优化，不把一般 stride 扩展冒充遗留迁移 |
| S5 DSA entry extent 义务进入 IR 验证 | [DSA IR](../lib/Dialect/DSA/IR/DSAOps.cpp)、[DSA backend](../lib/Compiler/DSABackends.cpp):115–139；`6cc104a6` | 使用现有入口和资源验证，不再另立 schema 工程 |
| family pipeline、provider、公共参数与调用 | [GPUBackends](../lib/Compiler/GPUBackends.cpp):20–59；[CPUBackends](../lib/Compiler/CPUBackends.cpp):22–63；[DSABackends](../lib/Compiler/DSABackends.cpp):86–112 | 在既有链路内优化，不另开执行路径 |
| 安装、Torch/MCP 与开发者入口 | [README](../README.md)、[CONTRIBUTING](../CONTRIBUTING.md)；`a5586d15` 记录已有打包安装及公开调用 | 已有入口继续使用；许可证与分发决定单列，不重做工具体系 |

已经存在的性能能力同样不从零立项：GPU producer 共享、作用域放置、共同 traversal 与 epilogue DAG；CPU producer 融合、共同遍历、输入 cohort、scratch 复用、连续访问、task reduction、native vector reduction 和双块 scan carry；DSA local value、collective、matrix supply、显式异步传输与 storage 分配。后续任务必须明确**在现有能力上扩展哪一段真实支持域或改进哪一种收益选择**。

最近的分析按需构造和空段消除已改善 JIT 可用性。这些成果保留，但“不再浪费编译时间”不能替代“生成更高效的程序”。

## 3. 当前性能证据：用于选题，不冒充当前 HEAD 全量成绩

现有 CSV 混合了不同轮次的原生产观察，本轮没有重测。它们足够定位优先级，不足以计算“全项目完成百分比”或证明所有目标已稳定达标。

| 当前观察 | 对计划的影响 |
|---|---|
| GPU 的大部分已有组合接近或优于各自 source，但 H100 仍存在明显尾部；例如 Triton corpus 中 causal-conv update、变长卷积存在大差距 | 优先分析当前访问、状态读取、物化、遍历与供数，而不是继续盯住一个 attention 的编译耗时 |
| batch-norm 在多个组合中很慢，但其原入口包含 mutable custom op、状态处理及 `torch.compile` | 先分清设备程序和 host 调用职责；不能只凭 CSV 比值把全部差距归给 GPU pass |
| [Mojo 结果](../experiments/cpu/results/mojo-x86.csv)有 199 条：187 pass、1 numerical_failed、11 run_only；小型归约、dot/matvec、路由等仍有尾部 | 普通 pointwise/collective/contraction 的横向质量与完整调用成本都需要面对；run_only 不算数值通过 |
| [Weft 结果](../experiments/cpu/results/weft-rvv.csv)有 6 条 pass，覆盖 RVV attention、IME i8 和 Q4_K；Q4_K 仍有差距 | 现有成功不能代表整个 CPU 支持域；每个 CPU 能力包要让 Weft 同步受益，先使用这些已有入口 |
| [MLU 主表](../experiments/mlu/results/bangc-mlu370.csv)有 13 条 pass，并保存真实设备耗时；source 时间和 ratio 为空 | MLU 运行链已经闭合；“性能好”尚缺同设备比较，不能拿其它设备的 reference 时间填入，也不能继续忽略 MLU |

GPU 定位入口是 [Triton/H100 表](../experiments/gpu/results/triton-h100.csv)、[cuTile/H100 表](../experiments/gpu/results/cutile-h100.csv)及同组既有 5090 与交叉 provider 表。不要把两种 corpus 中同名 kernel 自动合并为同一算法、dtype 或计时合同。

最终目标按每个 provider/硬件组合分别看：主要算法族接近或优于同设备、同算法的成熟 source；先消除缺失物理优化导致的数量级差距，再收敛稳定的中等差距。可把 **2 倍以上慢项优先消除、主要可比程序向 1.2 倍以内收敛**作为选题目标，但不能把数字写成语言支持条件，也不能以极少数弱 reference 上的大收益掩盖尾部。不同计时合同、硬件能力限制和数值合同差异逐项说明。

仅快于 PyTorch eager 或某个弱 source 不足以证明接近成熟编译器性能。已有原生目标 source 优先作为质量参照；缺少专业同合同对照时，明确该证据的范围，不为了好看换算法或放宽数值合同。

## 4. 怎样沉淀优化，怎样证明复用

### 4.1 三层复用，而不是一套万能物理 pass

| 层次 | 应沉淀什么 | 应保留什么差异 |
|---|---|---|
| canonical/shared 语义与分析 | 轴/坐标组合、整数范围、product/record、identity、数值许可、effects、alias、逻辑依赖；现有 [lib/Analysis](../lib/Analysis/) 与 KIR analyses | 不在这一层决定线程、CPU worker、SRAM 或 provider 指令 |
| execution family 的物理优化 | GPU 的 fragments/ownership/traversal/replay；CPU 的 tile/task/storage/producer；DSA 的 local/cooperative/transfer/completion | 三类执行模型分别形成真正的物理程序，不硬复用互不兼容的调度 |
| target 实现与原生 compiler | Triton/cuTile primitive 和 local forms；Mojo SIMD/微实现；Weft RVV/IME；BANG C 搬运和计算原语 | dtype、向量/矩阵能力、传输/同步、布局与原生 API 的真实差异 |

共享分析也必须按数据表示适用：CPU/DSA 的 memref 存储事实可以复用 `BufferStorage`，不能强迫 GPU resource 模型改用它。复用是同责实现被多处实际消费，不是所有目录都依赖一个巨型模块。

Triton/cuTile 应共同受益于同一个 GPU family 改写；Mojo/Weft 应共同受益于同一个 CPU family 改写。GPU 与 CPU/DSA 的收益来自同一语义规律在各自物理模型中的实现，不能用“共用了前端”代替这一价值。

### 4.2 每项性能知识必须形成完整工作包

一个工作包应能回答以下问题，并落实在原有代码与提交说明中，不新增计划文件或规则数据库：

1. **规律**：什么数据流或执行结构造成重复计算、访存、通信、串行依赖或资源浪费？
2. **合法性**：哪些现有类型、坐标、effects、lifetime、identity、顺序与数值许可保证可改写？
3. **收益选择**：保存、重算、融合、分块、并行化之间如何取舍？必须考虑仍存活的工作、数据量和资源代价，不能只数 op 或只看单条样本。
4. **真实改写**：哪些 loop、access、buffer、carry、fragment 或 task 会改变？serializer 能否只消费结果？
5. **组合与复用**：后续 pass 如何继续利用结果；哪些已存在的不同作者程序满足同一条件；哪些现有消费者需要共同迁移？
6. **实际收益**：原生产入口的数值和完整算子性能怎样变化，慢项是否转移到了其它阶段？

同一 producer 被多个 consumer 使用时，需要比较“保留一次共享计算”和“各处重算”的总体成本；不能把局部融合数量最大化当作目标。临时收益事实可以在 pass 内推导，选定结构必须进入当前 IR，改写后失效的分析不能继续使用。

已有 native reduce/scan/MMA 应按合同直接映射。微内核可接受明确的 block、dtype、stride、accumulator 和 effect 合同，不能接收整个 attention/normalization 算法再自行猜测语义。普通 reduce、scan、ordered loop 各守自己的合同；允许的重结合、融合和物理并行应积极实施，不额外发明严格保序限制。

## 5. 四个横向性能里程碑

每个里程碑可以由多轮、多个连贯提交构成。一次 helper 提取、一个缓存优化或一个算子加速是子任务。以下范围是需要完整形成的能力，不要求每个实现文件从头重写。

### P1：数据流、访问与中间物化的整体优化

**结果：让 producer 的计算与数据尽量在合适的 tile/scope 内被复用，减少无收益的中间张量、重复读取和遍历，同时保留值得保存的共享值。** 覆盖普通逐元素链、masked/gather producer、多个 consumer、buffer 版本与不变输入，不围绕某个算子名称匹配。

当前缺口有具体边界：GPU replay 已考虑未绑定的存活 slice，但部分 slice 绑定只处理单轴/无 region；稳定读取复用还要求同 block、坐标/valid/fill 的 SSA 相等。CPU 已有统一重放与遍历融合，但部分 producer 资格限于 same-block/single-output，遍历匹配要求完整等 bounds。DSA 已有 fill/copy/offset 消除，主要仍在局部执行域中生效。这些是拓展起点，不是未实现整套 fusion 的证据。

| 分线 | 连贯实现范围 | 原有落点 |
|---|---|---|
| GPU → Triton/cuTile | 在既有 scope/replay 证明上组合多消费者需求；利用坐标等价和稳定 snapshot 共享真实读取，扩大有收益的保留范围与消费 scope；限制重复重算和过长 live ranges，落实为访问、traversal 与 def-use 的改写 | [ReplayPolicy](../lib/Dialect/GPU/Transforms/Value/ReplayPolicy.cpp):244–299、393–423；[ScopePlacement](../lib/Dialect/GPU/Transforms/Value/ScopePlacement.cpp)；[AccessLoads](../lib/Dialect/GPU/Transforms/Access/AccessLoads.cpp):502–582 |
| CPU → Mojo/Weft | 从相邻、等域融合扩展到可证明的共同 tile 和多个 consumer；连接 tensor producer 与显式 loop/access 优化；按实际 buffer 版本和读取区间删除物化，兼顾 SIMD 可行性与重用 | [FuseStructuredComputations](../lib/Dialect/CPU/Transforms/Structure/FuseStructuredComputations.cpp):177–303；[TraversalFusion](../lib/Dialect/CPU/Transforms/Control/TraversalFusion.cpp):37–52、157–199；[FuseIntermediateBuffers](../lib/Dialect/CPU/Transforms/Storage/FuseIntermediateBuffers.cpp) |
| DSA → BANG C | 把现有局部消除组合成完整搬运/计算链的复用：消除重复 global→local 读取、可证明被覆盖的填充与拷贝；保留异步消费者所需 lifetime 和 collective 边界 | [LocalValues](../lib/Dialect/DSA/Transforms/LocalValues.cpp):55、98、153；[BANG Supply](../lib/Target/BangC/Transforms/Supply.cpp):835–862；[DSA Storage](../lib/Dialect/DSA/Analysis/Storage.cpp) |

优先使用现有逐元素、RMS/normalization、masked/变长访问、路由及 MLU relu/jagged/状态更新程序确认规则是否跨程序生效。例如 CPU 的 swiglu backward、fused-add RMS 已有优化成果，下一轮必须从残余 IR 证明新收益；DSA 的 recurrent delta/state passing 优化存储，不改为并行 scan。具体病例由实际 IR 命中决定，不能为“覆盖 P1”新增算法或矩阵。

**一个完整 P1 应得到：** 三类模型均有可复用的数据流优化改进；GPU 两 provider、CPU 两 provider 消费各自 family 的改写；冗余旧局部逻辑被迁移或删除；受影响程序的实际读取、物化或重算减少，并反映到完整调用性能。仅增加分析 API、更多 matcher 或一条更快 CSV 不算完成。

### P2：归约、扫描和分段状态的高效实现

**结果：在既有算法语义下，普通归约、tuple/record 统计、prefix 和 region state 能形成适合各执行模型的分块、局部汇总、carry 与原生 primitive。** 处理的是整条 producer→collective→consumer 链，而不是继续只微调一次 shuffle 或一个 scan 宽度。

1. 扩大闭合 typed combine、identity、free axes 与多个 consumer 的优化覆盖。共用已有 product/数值/坐标分析，保持每个字段的实际 dtype、empty、NaN/tie 和捕获值。
2. 在语义允许的循环和 task 内，把可并行局部汇总与最终合并明确分开，减少重复同步、遍历和无收益的中间写回。任意 accumulator update 不自动具有可合并的 partial 合同。
3. 对 prefix/region-scan 保留成员顺序、incoming state、最终 state 与输出关系；优化 carry 依赖和驻留，不把 ordered recurrence 当作 associative scan。
4. 对 masked/region 程序组合已有 range 与 identity 证明，减少无效遍历、冗余 predicate 和可以证明不需要的 summary 状态；不改变作者可观察的 page/window/chunk。

| 分线 | 在已有能力上继续实现的部分 |
|---|---|
| GPU | [ReductionTraversal](../lib/Dialect/GPU/Transforms/Reduction/ReductionTraversal.cpp):684–806 已会把循环中多次 reduce 变为 tile carry 与循环后 native reduce，但有 combine/identity/capture 资格边界；[RealizeScanConsumers](../lib/Dialect/GPU/Transforms/Reduction/RealizeScanConsumers.cpp):382–389 的 snapshot 路径也有单 source、rank 和方向限制，这不是整个 scan 的支持范围。按真实 producer/state 图扩大适用域，向 Triton/cuTile 输出各自可消费的 collective；block 内通信树仍归外部 compiler |
| CPU | [PartitionTasks](../lib/Dialect/CPU/Transforms/Task/PartitionTasks.cpp):37–51 的 partial 目前限于顶层 f32 Add/+0；[NormalizeReductions](../lib/Dialect/CPU/Transforms/Collective/NormalizeReductions.cpp):19–32 有表示约束。继续形成合法的多分量 partial/merge，改进 free-axis 连续 SIMD 与 task 粒度；已完成的双块 scan carry 和 native vector reduction 不重做 |
| Weft 目标消费 | CPU 共同 partial/state 结构直接进入 Weft 可用的 reduce/scan/loop，不先强制转换为 Mojo 专用向量树；[Reductions](../lib/Target/Weft/Transforms/Reductions.cpp):10–43 保留目标 kind/order 资格检查 |
| DSA | [Collective Realize](../lib/Dialect/DSA/Transforms/Collective/Realize.cpp):90–168 已实现通用 state/items/next；[NativeReduction](../lib/Dialect/DSA/Transforms/Collective/NativeReduction.cpp) 已有原生归约。扩大适合的局部/协作分块与 state 驻留，减少重复传输；coupled 字段的两阶段 snapshot 不能因“减少 copy”而被破坏 |

原生产入口包括现有 sum/max/product、Welford、softmax/RMS、cumsum/prefix、causal/linear attention 与 Mamba/state；它们是复用效果的不同使用者，不是 pass 的分派键。现有外部算法拆成几个 kernel 就保留几个 kernel，不引入隐藏 launch/global barrier，不自动跨 kernel 融合。合法的 compiler-private invocation workspace 继续沿已有 allocation、ownership、lifetime 和 ABI 合同形成，不因它不由作者显式分配就禁止使用。

**一个完整 P2 应得到：** 归约与状态优化从若干特定标量/形态扩展到明确的可复用语义域；在现有不同算法组织中实际生效；普通 collective 在有 native primitive 的目标上没有重复实现下层机制，分段 state 在各目标上没有无解释的串行慢路径。

### P3：收缩计算、供数、输出融合与工作映射

**结果：让 dot/matvec/GEMM、批量或分组收缩、attention 中显式收缩都能通过共同的轴、分块、供数与 accumulator 规则获得高质量实现。** 不给每个算法添加独立的整算子 leaf。

| 工作 | GPU → Triton/cuTile | CPU → Mojo/Weft | DSA → BANG C |
|---|---|---|---|
| 收缩形态与工作映射 | 根据 free/reduction/batch axes、并行工作量与 reuse 选择 ownership、loop order、grouping；小/窄收缩和大矩阵各有收益判断 | 在既有 task/block 与 implementation registry 内选择标量、向量、矩阵实现；避免低工作量的并行与供数开销吞噬收益 | 在现有 task/group/local 工作中组合 matrix 与 vector 计算，不把同一 workset 只含一次 MatMul 当作长期组织边界 |
| 输入供应与复用 | 暴露便于下层优化的访问、fragment 与 contract 关系；pointer/descriptor form 保持同一访问合同 | 按实际 consumer cohort 组合双侧输入准备、packing、向量载入和局部缓冲复用；完整计算 preparation 成本 | 组合连续 copy、gather、WRAM/SRAM/NRAM supply 和显式 completion；扩大可证明的 producer/consumer 链，保留同步和容量约束 |
| accumulator 与 epilogue | 基于已有 epilogue DAG 和数值许可融合合法 consumer，避免重复取数、转换和不必要的跨 scope 放置 | 把合适的逐元素后处理接入已有输出流，避免中间结果完整落地再遍历；显式 cast 与 source initial 保留 | 让 local matrix/vector 输出在合法作用域内直接被下一段消费，避免每一步经全局存储往返 |

三个需要完整拓展的现有边界：

- **GPU**：[ContractionEpilogue](../lib/Dialect/GPU/Transforms/Contraction/ContractionEpilogue.cpp):253–331 已共用 DAG，不重建。Triton [AccessForms](../lib/Target/Triton/Transforms/Access/AccessForms.cpp):48–144 的 descriptor 资格依赖特定 Cartesian range；[Supply](../lib/Target/Triton/Transforms/Supply/Supply.cpp):367–450 的普通供数主要匹配直接循环体；[RefineProgramMapping](../lib/Dialect/GPU/Transforms/Mapping/RefineProgramMapping.cpp):150–245 的 persistent 路径还有单顶层 group/traversal-worker 限制。根据真实访问几何、scope 和工作量拓展，cuTile 通过自身 load/gather/MMA 形式消费共同优化；不复制 Triton 的机器软件流水。
- **CPU**：[BlockContractions](../lib/Dialect/CPU/Transforms/Contraction/BlockContractions.cpp):34–77 的有界输入 cohort 只协调一个 Consumers 输入，:140–146 已优先保持现有跨消费者复用；Mojo [MaterializeRegisterContractions](../lib/Target/Mojo/Transforms/MaterializeRegisterContractions.cpp):28–55 已有 epilogue，但限相邻、相同 shape/dtype 与单 consumer。后续协调双方输入、多个消费者、完整输出 tile 的 K 轨迹及最终输出；不能简单强选同一 loop order，也不能让任务共享 mutable readiness。使用既有 [Implementation](../include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h) 与 [ImplementationInputs](../include/Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h)，不再立项建设 portfolio/微实现框架。
- **DSA**：[MatrixSupply](../lib/Dialect/DSA/Transforms/MatrixSupply.cpp):40–54、93 对 MatMul 组织和 load 来源有窄资格；[BANG Supply](../lib/Target/BangC/Transforms/Supply.cpp):320–378、433、580 已分别处理简单流、row 与 matrix pipeline。把共享的访问/完成证明复用到更完整供数链，减少逐案例增加 whole-loop matcher 的趋势；原 native primitive 的能力限制保留。

原生产覆盖沿既有 dense/grouped/batched GEMM、dot/matvec、Q4_K/IME、attention/MLA 和 MLU dense/state contraction。QKV 等多 projection 共用输入的作者程序可用于检验重复供数；不预先宣称已有代码已经命中。Micro-kernel 优化以 block 的具体数值/存储合同为边界，不改作者算法、输入精度或计时范围。

**一个完整 P3 应得到：** 供数、计算与输出被作为一个真实执行片段优化；收益不仅出现在大 GEMM，也能服务满足同一规则的窄矩阵、嵌入收缩和量化计算。各目标能使用自身已有高性能构造，公共层不会积累一份按算子名选择的实现目录。

### P4：资源感知、候选质量与多后端性能收口

**结果：前面形成的程序在不同设备资源下得到合理结构与有效候选，性能收益能跨 provider 和设备保持。** 本包的资源判断从 P1 起就随改写使用；P4 是整体收敛，不是等所有 pass 写完才考虑资源。

资源不是一项万能 pass，应分开四种责任：

1. **能力与资源事实**：读取已有 typed capability、当前存活值/缓冲区、fragment/workset 和供数要求。
2. **结构选择**：在 family pass 中选择 blocking、ownership、遍历、重算/保留、task 或 transfer 组织，真正改写 IR。
3. **候选合法性与质量**：在既有参数角色和候选机制中去除可证明非法或明显无效的组合，区分硬约束与收益启发式，保持有意义的备选；不要用扩大搜索空间掩盖差的程序结构，也不要为降低编译时间随意删掉有效候选。
4. **原生编译与选优**：让下层处理其实际 layout、register、pipeline 和指令限制；现有 tuner 测量候选并选择 winner，不把结果硬编码回 KIR。

| 分线 | 本包重点与真实边界 |
|---|---|
| GPU | [Resources](../lib/Dialect/GPU/Analysis/Resources.cpp):243–275 当前按不同 FragmentType 收集 nominal payload，不等于同时存活峰值；[ReplayPolicy](../lib/Dialect/GPU/Transforms/Value/ReplayPolicy.cpp):294–299 的最低 footprint 也不是 occupancy 保证。联合实际 live payload、reuse scope、并行工作量和能力选择结构/候选。H100 与 5090D 可以得到不同结构，但不为了“体现架构差异”强行生成不同 IR |
| CPU | [Implementation](../lib/Dialect/CPU/Transforms/Implementation/Implementation.cpp):338–373 已有有限关联 portfolio，明确不是完整代价模型。结合 vector/worker/private-memory、实际 packing/reuse、并行度、尾部和同时存活的 accumulator/scratch 改进选择；不能把单份输入准备容量当全部工作集，也不能把 private_bytes 当全部硬件 cache |
| DSA | 结合现有 tasks/tile 与 local storage/completion 事实改善工作分配和供数选择；[BANG Storage](../lib/Target/BangC/Transforms/Storage.cpp):200–218 已有 NRAM/WRAM/SRAM 最终预算检查，继续使用。当前 [BANG runtime](../python/intent/runtime/bangc/program.py):37–44 是固定 entry，不能宣称已有与 Triton 相同的候选 autotuner；普通 pass 优化不以新建 tuner 为前提 |

**Triton/cuTile autotune 已经存在，不是待开发项。** [Triton runtime](../python/intent/runtime/triton/program.py):95–115 使用原生 autotuner；[cuTile runtime](../python/intent/runtime/cutile/program.py):185–217 维护试跑状态并消费实际最佳配置。`num_warps/num_stages/num_ctas` 留在 Triton provider 配置；cuTile 使用自身合法参数。改进的是结构和候选质量，不是另造 winner 选择器。cuTile SDK 当前不提供的寄存器/shared 字段继续明确不可得，不能拿 Intent 估算冒充 native observation。

MLU 同设备性能比较属于必要配套，但与 compiler 修改分开提交：只针对现有 registry 的相同程序、规模和 dtype，在既有 `experiments/mlu/baselines/` 中接入合适的 BANG/SDK callable，结果回写原 CSV 的 source/ratio 栏。双方使用原 CNRT notifier 设备计时边界，包含该原多 kernel 序列所需设备工作，不混入 host、主机传输或 CNCC 时间；其它后端也各守其原设备/native/host 计时合同。原 reference 继续负责既定数值比较；不扩 case，不拿跨设备时间作比值，没有可比 source 的条目保持说明。可在 P1 开始时先做这一独立子任务，避免到收尾才发现无法评价收益。

**一个完整 P4 应得到：** GPU 两 provider、CPU 两 provider、MLU 分别有可信的完整调用结果和可解释的剩余差距；不能只用 H100 或 Mojo 的局部提升宣布整项路线图完成。使用原 registry 做阶段收束，保留硬件限制、数值未通过和无同合同 reference 的真实状态。

## 6. 下一轮从哪里开始，如何避免再次偏离

**下一实施里程碑选择 P1。** GPU、CPU、DSA 三条线从已有实际 IR 中各选一组同类数据流，先确定重复计算/访问/物化的共同原因，再完成规律对应的消费者和目标消费路径。P4 的既有资源事实用于限制融合和放置；MLU 同设备 baseline 子任务可并行准备。

推荐顺序是 **P1 → P2 → P3 → P4 收口**，独立部分可以并行：不需要等待 GPU 全部优化完才让 CPU/MLU 开始，也不需要把 P1 整体结束作为使用 P3 中现成供数规律的流程门禁。每次范围调整都围绕上述能力包，避免按“眼下哪个函数最耗时”无限扩展。

| 下一轮必须完成的工作面 | 实际产出 |
|---|---|
| 选定跨程序的数据流规律 | 指出原 IR 中重复读取、物化或重算的位置与原因；沿用已有 proof 接口 |
| 实现三类模型的对应改写 | GPU 两 provider 共用 family 结果；CPU 两 provider 共用 family 结果；DSA 在自身存储/协作模型下兑现同一优化规律 |
| 形成收益判断并迁移消费者 | 同责策略有清楚归属，删除被替代路径；不能只新增公共接口、保留旧局部名单 |
| 在原生产入口确认效果 | 必要数值检查与完整算子时间；改写是否生效和性能是否获益分开说明 |
| 结算能力包而非单点 | 已适用、尚未覆盖、由目标能力限制的部分明确；不能只报告“某例快了” |

若一个后端已有该能力，则保留并检查组合效果，不做无收益改动凑覆盖。设备暂不可用时，相关实现可以推进，运行项保持未完成；其它后端继续工作，最终不把源码发射成功计作该后端性能完成。

编译缓存、分析构造、文件长度、某个 JIT 慢例仅在阻止上述工作时处理到必要程度。runtime 测量若暴露独立 host 瓶颈，应如实归因并单列任务，不能在纯 pass 轮偷偷改 runtime，也不能把这类时间差全部说成 IR 问题。

## 7. 成熟编译器参照：借用机制与职责，不复制全部底层

本地 `../ref/triton`、`../ref/tilelang`、Modular、MLU-OPS 和 LLVM/MLIR 实现是主要依据；官网用于核对公开行为。本地快照与已安装 SDK 不保证逐行相同，不把外部最新文档中的能力直接视作本机可用。

| 对照事项 | 本地成熟实现 | Intent 的对应责任与下一步 |
|---|---|---|
| 合法代数与结构规范化 | Triton [Combine.cpp](../../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp):135–192、249–282 对 multiply/reduce、dot/add 做真实 IR 改写 | 已有局部融合合同允许积极优化；不因担心“替作者改算法”删除正确融合，也不扩大到整算法替换。对照 [pass 合同](../doc/compiler/passes-and-analyses.md):109–113、145–167 |
| 共享值与 live range | Triton [RemoveLayoutConversions](../../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp):895–959 根据实际 use dominance 重用结果，验证 backward slice 后重物化；[ReorderInstructions](../../ref/triton/lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp):104–123 缩短特定低层值的存活期 | P1/P4 的 [ReplayPolicy](../lib/Dialect/GPU/Transforms/Value/ReplayPolicy.cpp):244–299、[Resources](../lib/Dialect/GPU/Analysis/Resources.cpp):243–275 应联合 scope 与真实存活范围；不照搬 Triton 的 layout 语义 |
| producer/consumer 融合 | LLVM 20.1.8 `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`:1524–1539、1568–1611 分开 consumer tiling、实际 slice producer fusion 与控制选择 | P1 在各 family 的实际表示上组合坐标与 producer；原生接口的 tensor 前提不自动适用于 CPU memref，更不直接覆盖 GPU/DSA |
| partial reduction | 同一 MLIR 文件 :585–603、620–637 分开 partial 初始化与局部 tile | P2 从 [PartitionTasks](../lib/Dialect/CPU/Transforms/Task/PartitionTasks.cpp):37–51 拓展符合 dtype/combine/order 的 partial/state，而不是把任意操作套成 f32 Add 或强制一种目标树 |
| GPU 下层职责 | Triton [Coalesce](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp):77–118 生成 distributed encoding；[NVIDIA compiler](../../ref/triton/third_party/nvidia/backend/compiler.py):297–360 组织 layout、MMA、pipeline、TMA 和重排 | Intent 改善 block program、访问几何及参数，继续把 lane/layout/ISA 留给外部 compiler；P3 的目标 [Supply](../lib/Target/Triton/Transforms/Supply/Supply.cpp) 只形成适合其消费的合法 source |
| TileLang 的层次与作者责任 | [kernel.py](../../ref/tilelang/tilelang/language/kernel.py):277–300、[allocate.py](../../ref/tilelang/tilelang/language/allocate.py):49–92、[loop.py](../../ref/tilelang/tilelang/language/loop.py):112–169 暴露 threads/shared/pipeline；[CUDA pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py):100–140、245–264 分阶段 lowering | 借鉴 tile、storage、pipeline 的职责划分，不把 TileLang 的低层作者义务搬给 Intent，不恢复 TileLang 后端。Intent 的义务边界见 [编译规格](../doc/compiler/README.md):35–48 |
| CPU 供数与微内核 | Modular [impl.mojo](../../ref/modular/max/kernels/src/linalg/matmul/cpu/impl.mojo):338–395 一次准备 B 供多个 M 子块，并在 last-K 做 epilogue；[utils.mojo](../../ref/modular/max/kernels/src/linalg/utils.mojo):493–544 联合 N/K、packing 容量与微块选 tile | P3/P4 从 [BlockContractions](../lib/Dialect/CPU/Transforms/Contraction/BlockContractions.cpp):34–77 的局部 cohort 扩到真实复用组件；专家实现负责 FMA/prefetch，不照抄其它 compiler 的容量常数 |
| 显式搬运与流水依赖 | TileLang [pipeline_planning.cc](../../ref/tilelang/src/transform/pipeline_planning.cc):36 起按真实访问域看依赖；MLU-OPS [binary_op_3pipeline.h](../../ref/mlu-ops/kernels/binary_op/binary_op_3pipeline.h):82 起共用搬运/计算/同步/尾部组织 | DSA P3 改进 [Supply](../lib/Target/BangC/Transforms/Supply.cpp):320–378 的组成能力；primitive 可以专家实现，整条供数链不按算子名各写一次 |

MLIR 本地源码从 `/tmp/intentdsl-llvm-20.1.8-src/mlir/` 定位。在线 [Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html) 与 [autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) 同样区分参数配置、候选裁剪和实测 winner；[TileLang autotuning](https://www.tilelang.com/programming_guides/autotuning.html) 也提供配置与 pass 控制。它们支持“优化 passes 与配置选优协作”的方向，不证明 Intent 已具备相同覆盖或性能。

## 8. 保留但不支配性能主线的欠账

### 8.1 已有数值状态

- Mojo `fp8_gemm` 仍是 `numerical_failed`。`a5586d15` 记录原入口中一个 f32 ULP 跨 FP8 舍入中点的定位，没有证明 accumulator/cast 实现错误，也没有使原容差检查通过。保留原算法、reference 和容差，不能改写为已通过；进一步处理必须区分数值合同、参考差异和 compiler 缺陷。
- Triton corpus/H100 的 cuTile `flaggems_softmax_backward` 仍记录数值失败，见 [原表](../experiments/gpu/results/triton-h100.csv):48。该历史观察不能直接定为当前 HEAD 同一根因，也不能从路线图删除；相关 collective/数值工作沿原入口处理，不额外造测试。
- 硬件不支持的 scaled primitive、source JIT 失败和 run_only 分开保留，不能为了性能统计调整支持范围。

### 8.2 DSA 正式规格覆盖

现有 [compiler 总览](../doc/compiler/README.md):9–20 和相关正式章节主要描述 GPU/CPU。DSA 已有实现、verifier 与运行，但 task/group 参与者、local/workunit/group completion、NRAM/WRAM/SRAM ownership 及异步 lifetime 尚缺相应完整正式说明。

这是一个**范围有限、独立处理的合同文档欠账**：对照当前 DSA IR、Storage、BANG Supply 及 NeuWare 真实语义，补清现有职责；不因文档缺口重新发明一套 DSA IR。普通基于明确 effects/completion 的消除与复用继续推进；新增跨 completion 边界的异步重排必须先有明确合同。真正改变设计或作者可见行为时仍需用户确认，报告不代替正式规格。

### 8.3 JIT、公开交付与运行环境

已有安装/Torch/MCP 调用继续沿原路线维护。Mojo 冷编译、重复生成代码和 CPU 短算子的 host 成本保留为独立配套问题，不能再次变成连续多个主要性能里程碑。许可证与实际公开分发方式仍需维护者决定，不在本报告代选、不发布、不增加版本号或迁移文档。

## 9. 实施纪律与本路线图的结束状态

实施以 [doc/index](../doc/index.md) 为准，本报告不修改语言和 IR 合同。每项改写都应兼顾收益、可复用性和目标消费，而不只是增加 pattern 数量。目录沿既有稳定职责扩展；只在同责实现确实需要共同归属时整理，不按行数移动文件。

验证只使用必要的原生产入口、输入规模、dtype、reference、容差和计时合同。不建立 test 目录、pytest、fixture 或额外数值/边界/组合脚本，不扩 registry 矩阵，不要求每个内部提交全量重测。准备和编译可以并发，同机性能计时避免相互干扰。固定版本测评与性能修改分开组织。

结果回写各实验组既有 CSV；IR/source/cache 留在仓库外。不新增平行结果表，不以新报告、验证数量或流程节点代替实现。报告完整算子时间和 reference 比值，并区分 Intent 编译、native 编译、调优、运行、数值与实际性能。存在波动时说明，不挑最好一轮证明提升。

**本路线图的结束状态是四种优化能力形成可复用的实现，并在 GPU、CPU、MLU 的现有产品范围内带来可信的性能质量：** 新程序满足同一语义/物理条件即可受益；同 family 的 providers 共享优化；不同 family 各有适合的实现；主要性能差距通过 program/pass 改进解决，剩余限制明确。不能仅凭骨架完整、一个后端领先、某次编译变快或几个算子通过宣布完成。
