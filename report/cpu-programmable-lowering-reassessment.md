# CPU 编程模型与可编程目标实现：现状、职责与修改建议

本文回答：CPU 是否需要改变；共同 CPU IR 应处于什么抽象层；Intent compiler、目标实现作者和 Mojo/Weft compiler 分别负责什么；当前代码具体应怎样调整。

这是设计讨论报告，不是已经生效的新规格。`doc/` 仍为当前规格权威；本文将当前事实、参考机制和建议变更分开。此次没有修改实现、设计规格、外部 TianchenRV 或 Comet 状态，没有编译、生成或运行测试。当前 `cpu-pointwise-reductions` 的已有验收不等于下述新能力已完成。

## 1. 结论：改变分层与实现机制，保留 CPU IR

需要改变，但不是取消共同 CPU 程序，也不是转向完整算子库调用。

建议目标是：

> Intent 建立并优化结构化的 CPU task/block program；针对其中适合专业实现的计算块，由 target lowering 选择、参数化并实例化专家编写的微程序；生成的程序继续交给 Mojo/Weft compiler。

共同 CPU IR 不应穷尽 AVX/RVV 寄存器、量化解码序列、Weft Level 或 IME fragment 的全部细节。它必须完整定义本层的计算、访问、依赖和生命周期，但“本层完整”不等于“已经展开到指令或标量循环”。一个数值合同完整的 structured operation 本身就是完整语义，不是待猜测的洞。

三个结论需要同时成立：

1. **保留 compiler 主干。** KIR→CPU construction、task partition、跨操作 fusion/reuse、访问分析、ABI 与当前程序 verifier 都有真实作用，不能被整算子模板代替。
2. **允许专家编程。** 复杂数值实现和目标组织可以由专家写成 lowering implementation，不要求通用优化器从零推导，也不要求普通 Intent 作者写同样低层的程序。
3. **实现必须成为编译过程的一部分。** 选择依据、参数、接口和展开结果明确；不能在最后打印字符串时临时创造结构，也不能拿整个 source baseline 作为隐藏执行路径。

此前讨论的两种偏差都应纠正：只允许“shared 自动推导、provider 机械映射”过窄；把专业实现一概理解成“运行时调用整算子库”也不准确。这里需要的是**编译期的可编程实现选择与展开**。

## 2. 当前 CPU 到底是什么，而不是什么

### 2.1 真实编译链

```text
Intent source → canonical KIR
    → KIRToCPU
    → runCPUPasses
        fusion、blocking、部分 packing/microtile、task partition
        ├─ Mojo legalization
        │    task loops / structured computation materialization
        │    → SCF + memref + vector + math
        │    → Mojo source → Mojo/LLVM → native artifact
        └─ Weft legalization
             structured CPU task → Canonical Weft IR
             → 当前只生成，不接设备执行
```

直接入口是 [intent-compile.cpp](/home/kingdom/phdworks/intentdsl/tools/intent-compile/intent-compile.cpp:154)。Mojo 才调用 [materializeCPUProgram](/home/kingdom/phdworks/intentdsl/lib/Target/Mojo/Transforms/Legalize.cpp:67)；Weft 直接消费未走这一步的 CPU task，把 contraction 转为 Weft `OuterContractOp`，而不是先强制转换成 Mojo 式 SIMD。[Weft 转换](/home/kingdom/phdworks/intentdsl/lib/Target/Weft/Transforms/Legalize.cpp:354)

因此，“现在 CPU 没有 IR”“Weft 已被迫消费完全标量化程序”“所有优化都写在 serializer”都不是当前事实。

### 2.2 当前表示和已有能力

CPU dialect 使用自有 ABI、capability、configuration、reduction-order、microtile attributes，以及 `TasksOp`、`ReduceOp`；其余复用 `func/linalg/scf/memref/vector/arith/math`。没有独立 CPU value type 不等于没有 CPU IR，也不构成必须另造一套 type 的理由。[属性定义](/home/kingdom/phdworks/intentdsl/include/Intent/Dialect/CPU/IR/CPUAttrs.td:13)、[operation 定义](/home/kingdom/phdworks/intentdsl/include/Intent/Dialect/CPU/IR/CPUOps.td:11)

| 当前能力 | 真实作用 | 应如何看待 |
|---|---|---|
| task partition/isolation | 把逻辑 workset 变成显式任务、captures 和坐标映射 | 应保留，不是 provider 可随意补写的组织 |
| axis/access、effect、storage-root 分析 | 支撑 fusion、读写顺序、alias 与中间值复用 | 应继续成为共同分析 |
| structured computation/reduction | 保留 contraction、combine、identity、访问关系 | 是较高层 CPU 程序的基础 |
| contraction blocking | 建立 M/N/K 遍历、输出 ownership 与数据复用 | 保留通用部分，拆开与特定微核绑定的部分 |
| register contraction materialization | 形成 f32 vector loads、广播、FMA、循环、prefetch | 已有一种真实微核实现，不是只有空接口 |
| finite tuning | 编译多个 config，实测并缓存 winner | 有效，但还不是专业实现族之间的选择 |

一维 task coordinate 也不等于不能表达二维 GEMM ownership。当前 partition 已把多维矩形 workset 线性化，再用除余恢复各坐标；真正限制是零起点、unit step、矩形域，以及尚未实现的嵌套调度/任务归约。不能仅为“看起来像多维 IR”新增另一套 task 模型。[PartitionTasks.cpp](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/PartitionTasks.cpp:34)

### 2.3 真正需要调整的地方

**第一，共同分块提前绑定了当前 f32 向量微核的偏好。** `BlockContractions` 不只创建外层 cache blocks：它按 `vectorWidth × microN` 决定 B micro-panel，按固定行/列序列拆尾部，分配 f32 partial，并把对齐设为 `width × 4`。这些选择同时进入发给 Weft 的 shared 程序。[BlockContractions.cpp](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/BlockContractions.cpp:69)

这不证明已有 f32 程序错误；它意味着下一种实现尚未提出要求，共同层就已替它决定了一部分内部结构。对更不同的量化、RVV 或 matrix implementation，这可能限制选择空间。

**第二，配置与能力混合了共同职责和局部职责。** 同一 `ConfigurationAttr` 包含 task grain、cache tiles、vector width、microtiles、register/reduction replicas；shared legality 使用 `vectorBits / 32`、每元素 4 bytes 和固定微核范围。[Passes.cpp](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/Passes.cpp:78) 这些是当前实现条件，不能成为整个 CPU family 的长期定义。

**第三，数值覆盖仍很窄。** KIRToCPU 拒绝非 f32 view；矩阵识别依赖二维 indexing maps 和单个 FMA body；Weft external capture 固定为 `dense.f32`。只增加一个“量化实现名称”无法让数据和语义穿过这些入口。[KIRToCPU.cpp](/home/kingdom/phdworks/intentdsl/lib/Conversion/KIRToCPU/KIRToCPU.cpp:39)、[contraction 识别](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Analysis/PhysicalProgram.cpp:15)、[Weft encoding](/home/kingdom/phdworks/intentdsl/lib/Target/Weft/Transforms/Legalize.cpp:112)

**第四，专业实现的编译接口尚不存在。** 当前选择基本是 `vector/contraction` 两类数值配置；Mojo 有一份 f32 register-contraction builder，而没有各实现分别提供 applicability、接口需求和 expansion 的机制。当前 tuner 测的是已生成 config functions，不能凭 tuning 补出另一套数值微程序。[微核实现](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/MaterializeRegisterContractions.cpp:51)、[winner 选择](/home/kingdom/phdworks/intentdsl/python/intent/runtime/mojo/program.py:33)

这些是结构与覆盖限制，不是已经证实的性能归因。本文不把静态审查升级成“某性能差距已经解释”。

## 3. Reference 支持什么：TileLang 的可编程 GEMM lowering

### 3.1 不只是 operation→API，也不是完整 GEMM 外部调用

本地 TileLang 的链路是：

```text
Gemm node：regions、dtype、M/N/K、accumulator、storage 等
    → instruction/implementation selection
    → implementation.infer_layout
    → implementation.lower
    → 专门编写的 PrimFunc body
    → 插回当前 TIR，继续访问/布局/资源/codegen 处理
```

- [实现注册](/home/kingdom/phdworks/ref/tilelang/tilelang/cuda/op/gemm/__init__.py:45)分别登记 MMA、WGMMA、TCGEN05、block-scaled 等类；[Gemm.lower](/home/kingdom/phdworks/ref/tilelang/tilelang/tileop/gemm/__init__.py:127)选择实现并调用其 lowering。
- [GemmMMA.lower](/home/kingdom/phdworks/ref/tilelang/tilelang/cuda/op/gemm/gemm_mma.py:109)用 `T.prim_func` 写出 local buffers、K-loop、fragment loads 和 MMA；并不是从通用乘加 IR 自动搜索出这些组织。
- [GemmNode::Lower](/home/kingdom/phdworks/ref/tilelang/src/op/gemm.cc:198)取回 PrimFunc body，带 lexical allocation scope 插入 TIR；[LowerTileOp](/home/kingdom/phdworks/ref/tilelang/src/transform/lower_tile_op.cc:1134)提供 buffer/layout/workspace 等上下文，并继续处理展开的子树。

因此，专家编写复杂实现并不违背 compiler lowering。实现产出真实 IR，才是判据；类名叫 `Emitter` 并不能说明它是错误的终端发射器。官方 [GemmBase 接口](https://tilelang.com/autoapi/tilelang/tileop/gemm/gemm_base/index.html)也把 `infer_layout` 和 `lower` 分别列为实现接口。

### 3.2 专业实现还需要与外围协调

TileLang 的实现会提出布局要求，外围处理已经存在的共享布局与新要求的关系；某些 instruction 的严格布局不能被忽略。[GemmNode::InferLayout](/home/kingdom/phdworks/ref/tilelang/src/op/gemm.cc:243)

给 CPU 的启示是：不能先把全部 blocking/packing 固定，再在最后选一个微核。实现需求至少要在相关结构冻结前参与选择。

这不要求做任意布局全局搜索。可以先采用有界的“枚举适用实现→检查需求→形成合法 binding→展开”流程；只有真实的多个消费者冲突才协调共享表示或显式转换。

### 3.3 专门量化实现的证据与边界

SM120 block-scaled GEMM 有独立实现，内部可以组织 fragments、scale loads、unroll 和 ping-pong；它的 applicability 同时约束 scale operands、target、storage 与格式。[选择条件](/home/kingdom/phdworks/ref/tilelang/src/cuda/op/gemm.cc:363)、[专门微程序](/home/kingdom/phdworks/ref/tilelang/tilelang/cuda/op/gemm/gemm_mma_sm120.py:130)

这支持“难以通用推导的实现由专家编写”的机制，但不证明它能处理任意 Q4_K/Q8_K 或所有量化 GEMM。专门实现仍需要明确的适用语义。

CUTLASS/CuTe 在当前 ref 中确实参与部分局部指令与类型封装，例如 [mma.h](/home/kingdom/phdworks/ref/tilelang/src/tl_templates/cuda/instruction/mma.h:4)；不能把这种组件复用笼统说成每个 `T.gemm` 都调用完整 CUTLASS GEMM。

### 3.4 与当前 Intent CPU 的具体差异和后果

| TileLang 机制 | 当前 Intent CPU | 实际后果 |
|---|---|---|
| 多个专业实现类，分别有 layout/lower | 一份主要的 f32 register microkernel builder | 新实现需要继续修改通用路径，缺少独立扩展边界 |
| 实现需求参与布局推断 | shared blocking 已绑定 width/micro-panel | 实现尚未选择时，一部分数据供应被提前固定 |
| 专业程序展开成真实 TIR | 当前 C++ builder 同样能生成真实 MLIR | 这项原则已有基础，应扩展机制，不是否定 IRBuilder |
| target-specific warp/shared/fragment | CPU tasks、顺序控制、普通存储与非 SIMT 执行 | 借鉴接口与展开方式，不复制 CUDA execution model |

## 4. 建议的 CPU IR 编程模型

### 4.1 根模型：普通调用中的 task/block program

CPU IR 是 canonical KIR 的一种 executable realization：一次调用包含显式任务域；任务拥有 workset、captures 和输出责任；任务内部执行顺序控制与 shaped computations；返回前完成所需任务与同步。

它不是以 AVX register、RVV lane、matrix tile 或某个 worker 为根对象。它与 GPU IR 的相似处是“完整的 block program＋保留 structured compute”；不同处是没有隐式 SIMT 实例、warp/lane ownership，也不要求 shaped value 的大小等于硬件寄存器宽度。

### 4.2 本层必须表达的内容

| 实体 | 共同 CPU IR 保存什么 | 不在这一层穷尽什么 |
|---|---|---|
| Program/ABI | views、scalars、shape identity、访问方向、alias、调用完成边界 | provider 函数拼写、机器寄存器 |
| Tasks/worksets | 任务域、坐标映射、captures、唯一写入与依赖 | 将 task coordinate 当固定 hart/thread ID |
| Shaped values/computations | dtype、逻辑轴、局部计算范围、广播与结果关系 | AVX/RVV lane、LMUL、物理 register tuple |
| Access/resources | 坐标、有效范围、效果、共享范围、初始化和 lifetime | 所有局部 pack 格式、固定指令布局 |
| Control | 作者 ordered control、compiler blocking、carry、完成关系 | 微程序内每条指令的排程 |
| Structured compute | contraction/reduction 等完整语义及与外围的连接 | 每种格式的全部高性能解码/累加实现树 |
| Numerical/format contract | 解释输入所需的格式、分组、转换和累加语义 | Weft Encoding/Level 的完整可编程 surface |

实现上继续复用合适的 MLIR 标准 dialect。建议先保持现有 carrier，补齐 structured-operation 的语义查询和适用性接口；确有标准表示无法完整保存的 format/scale/accumulator schema 时，再增加 CPU 自有 structured op。不要为了证明“有 dialect”重新包装每个 add/load/loop。

当前 `isMatrixContraction` 的二维单 FMA 识别可以作为已支持 dense 形式的实现，不应继续作为所有 contraction 的唯一语义接口。新的格式不能靠不断扩大这个 body matcher 来恢复已经丢失的高层合同。

### 4.3 程序形态示意

以下只表示层次，不提议新的 DSL 或 MLIR 拼写：

```text
CPU program(A, W, Y, shapes, logical-format information)
  tasks over output worksets
    task-owned output block
    accumulator for this block
    for each chosen outer K block
      input/weight regions with explicit bounds and dependencies
      structured contraction block → updated accumulator
    pointwise consumers / output write
  complete tasks before invocation returns
```

外层遍历已经明确，内部 contraction 仍可足够高层。lowering 可将该计算块替换为一个专家编写的 Weft/Mojo 程序，而不用先消解成通用 scalar multiply/add 再反向识别。

## 5. 谁负责编程、谁负责优化

### 5.1 三种作者身份

1. **Intent kernel 作者**写算法及调用方必须知道的数值/格式关系，不写 AVX/RVV/Weft 的完整实现组织。
2. **CPU compiler 开发者**实现 task/block program 的 construction、共同分析、跨操作优化与实现选择/集成机制。
3. **目标实现作者**为某类明确计算块写高性能微程序，使用 Mojo SIMD 或 Weft 的 Encoding、Level、有限位宽与复用表达；不要求该程序可由普通 Intent DSL 逐句复述。

Weft 是面向更具体数值 realization 的语言。把它用作专业实现的编写工具，正是利用它的编程性，而不是否认它或逼共同 CPU IR 成为另一份 Weft。

### 5.2 按作用范围分配职责，而不是按优化名字分配

| 决定 | 责任 |
|---|---|
| 整个 kernel 的任务覆盖、并行安全、跨计算块的数据依赖 | 共同 CPU compiler |
| 多个操作共同使用一份值，是否跨块物化/复用 | 共同分析与相应 CPU transformation |
| task/cache block 大小与 traversal | CPU passes，在需要时消费候选实现的约束 |
| 微程序内部的子块、局部 decode、有限位宽 partial、内部阶段组织 | 专家 implementation，在所承接 operation 的数值合同内 |
| 微程序要求的输入表示如何由外围供应 | implementation 提需求；CPU/provider integration 显式形成供应与 lifetime |
| RVV layout、LMUL、IME pack/fragment、机器 schedule | 外部 Weft compiler，除非该信息确实必须由 adapter 给出 |
| Mojo SIMD 表达、合法 intrinsic 与 API types | Mojo implementation/local lowering；机器指令、寄存器分配继续交 Mojo/LLVM |
| host/runtime 分配、调用、join、产物与 winner 复用 | 执行已经声明的 ABI/资源要求，不另造算法 |

例如“packing”不能统一归给某一层：同一 task 中多个消费者共享 panel 的存在与 lifetime，需要外围协调；仅供某个局部 matrix atom 使用的内部 packing，属于其实现或下层 compiler；跨调用 persistent repack 则改变 artifact/ABI，不能藏进每次 invocation 的微程序。

专家微程序也可以包括局部循环、scratch、转换和一组操作，不必限制成固定几条指令。关键边界是它承接的计算范围和外围可见 effects，而不是代码行数。

## 6. 可编程 implementation 机制应长什么样

### 6.1 接口不是一张字符串映射表

建议一个 implementation 至少提供以下信息与操作；这些是内部编译接口职责，不是要求马上新增同名 classes：

| 接口职责 | 内容 |
|---|---|
| 描述所实现的计算 | operation/region 的语义、operand/result relation、format、数值模式 |
| 判断适用性 | dtype、shape/尾部、访问关系、target capability、必要资源条件 |
| 提出实现需求 | block 约束、输入供应/布局要求、输出形式、scratch 与可共享资源需求 |
| 声明有限参数 | 只暴露该实现真正消费的参数，不把整个实现树变成开放式搜索语言 |
| 实例化/展开 | 将当前 operands、regions、parameters 绑定到专家程序，产出真实 IR |

专业实现可以由 C++ IRBuilder、结构化宏或目标 DSL helper 编写。它们都是实现手段，不应限制为 C++ 硬编码所有数值实现。对 Weft，优先复用其现有作者语言与 frontend；对 Mojo，现有 MLIR→Mojo 实现可先作为一种正式 implementation 保留下来。

### 6.2 选择、约束协调、展开的顺序

```text
完整 structured CPU program
    → 从当前 operation facts 找到有限适用 implementations
    → 查询需求，协调外层 block / supply / resource
    → 固定本候选的 implementation 与参数
    → 实例化局部微程序，显式连接输入、结果、effects 和 lifetime
    → 验证并继续 provider / external compiler lowering
```

布局推断与 lowering 必须消费同一个候选选择，不能前一阶段按实现 A 组织数据、后一阶段又自行选择 B。若某候选不合法，应在候选构造阶段拒绝或按明确策略考虑其它已声明候选，不得在 emitter/JIT 失败后隐式切换程序。

选择前的 registry 是编译规则集合；选择后的 executable authority 是实例化的当前程序。不能长期保留“一个原 operation＋旁边一份决定记录”，让 runtime 或 serializer 再解释如何执行。

### 6.3 为什么这仍然是 pass 化

选择/实例化由正式 transformation 驱动，读取 current IR，检查适用性，改变 operation/type/region/def-use，更新 effects 和资源，并验证后置条件。专家提供的是这个 transformation 可调用的具体实现，不是另一个绕过 compiler 的 runtime。

“通用 pass”也不意味着 pass 能自动发明所有算法。复杂实现可以由专家编写一次，之后由编译器按参数实例化；它的语义保持是实现作者承担的正确性义务，编译器检查其声明和当前调用是否匹配，不要求每次编译重新证明任意量化代数恒等式。

终端 serializer 的限制仍保留：只打印已合法化程序；不得匹配 kernel 名、临时决定 scratch/loops 或选择另一套实现。给文件改名为 `Pass` 不算满足，反过来将 IRBuilder 命名为 `Emitter` 也不自动构成违规。

## 7. 量化：哪些信息必须上来，哪些细节应封装

### 7.1 共同接口不等于暴露全部 Weft Encoding

“W 使用某个量化格式、对应逻辑 shape、scale/zero-point 解释和输出数值关系”是调用的语义。格式的专业实现如何用 Encoding fields、Level、局部 partial 和 IME/RVV 组织出来，可以封装在 implementation 中。

因此，输入格式可以通过一个闭合的、硬件无关的 schema 被传递；不需要让普通 Intent 作者写 Weft 的完整 bit-layout builder 或 stage tree。但是只给普通 f32 `matmul`，没有任何量化输入语义，也不能凭空选择 Q4_K 实现。

不同层面的选择必须分开：

- **格式/算法选择**：Q4_K、其它 INT4 scheme、FP4 microscaling 不是同一输入的可随意替换实现；输入表示和运算约定先确定。
- **实现选择**：同一已确定语义可以有不同合法解码、供应、局部累加或向量/matrix realization，由 lowering 选择。
- **参数选择**：在该实现的有限域内绑定 block、replica、unroll 等，再实测选优。

“一个 matmul 下有几十种量化乘”不要求通用 compiler 推出几十棵专家程序；但这些程序必须知道自己各自实现哪种合同。

### 7.2 当前规格有一个必须正面处理的边界

当前 [DSL core](/home/kingdom/phdworks/intentdsl/doc/dsl/core.md:322)的 scaled contract 是特定 microscaling schema；普通 packed INT4/INT2 则要求作者显式 decode 后调用 ordinary contract。不能把 Q4_K 的 scale/min/bsum 程序直接冒充为现有 FP4 scaled contract。

建议采用两种明确处理，而不把所有量化都推回作者的逐位代码：

1. 已有 structured schema 能完整定义的格式，直接保留语义并补 CPU/provider lowering。
2. 真正需要专家级整数/量化 realization、而现有 schema 未定义其输入与数值关系的格式，先增加最小的硬件无关格式/计算合同，再交给专业实现。具体形式必须由选中的真实格式驱动，不预建任意字符串格式或全能 encoding DSL。

这属于明确的语言能力设计，不是以现有实现反向修改规格，也不是只给 CPU attribute 加一个 format 名就算完成。若要让 Q4_K 用户不再手写完整 decode graph，这一项应在下一 Shape 中明确，而不能暗中完成。

### 7.3 一个必须保留的实际边界：量化准备不能被重复藏入微核

外部 Weft 的 [Q4_K mul_mat 示例](/home/kingdom/phdworks/TianchenRV/examples/kernels/mul_mat/q4_k.py:10)先将 f32 activation 量化到 Q8_K，再遍历输出调用量化点积 helper；[点积 helper](/home/kingdom/phdworks/TianchenRV/examples/kernels/vec_dot/q4_k_q8_k.py:8)内部包含 block/subtile、整数统计和 scale/min correction。

这说明可以复用专家程序，但不能把完整示例 kernel 原封不动塞进每个 output microtile：那会重复量化 activation、改变 scratch lifetime 或数据复用。应承接明确的计算块，并让一次性的准备、跨块共享与输出接口在外围可见。

同理，persistent weight interleave 的存在属于输入 artifact/ABI；实现不能默认为输入已经具有该布局，更不能在第一次调用偷偷创建永久缓存。

## 8. Mojo 与 Weft 如何各自承接，不强迫对称

### 8.1 Mojo：保留已有实现，将其变成可选择的 realization

Mojo 是完整系统语言，SIMD 只是其低层值与计算工具，不是 CPU IR 必须采用的统一抽象。

当前 f32 microkernel 已由 [MaterializeRegisterContractions](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/MaterializeRegisterContractions.cpp:51)形成；固定 f32 vector、prefetch 和四步 unroll 是这一实现的具体策略。建议把这份策略及其参数、合法性归到 Mojo 对应的 vector realization，而不是删除后重写已有性能结构。

通用的逐点/归约 loop materialization 可以继续复用；但它应是所选 realization 的步骤，不是所有 provider 无条件经过的最低公共层。普通算术语义仍由 Intent 保持，显式 FMA 与普通 add/mul 分开；当前 `--fp-mode=contract=off` 不应因为加入实现选择就被去掉。[native 编译选项](/home/kingdom/phdworks/intentdsl/python/intent/targets/mojo.py:30)

### 8.2 Weft：复用专家 source 编程，但必须补结构化实例化接口

Weft 普通 helper 当前可以真实 inline：frontend 绑定 typed values 与静态参数，在宿主上下文中编译 helper body。[现有 inline 实现](/home/kingdom/phdworks/TianchenRV/python/weft/frontend/compiler.py:2963)

然而，这不等于 Intent 已能调用这个入口。当前 Intent 链接的是外部 Canonical dialect，用 C++ 构造新的 Weft task module；没有已提供的 helper importer 或 microprogram splice API。

推荐接口边界是：**专家 helper/source 是实现定义；经 Weft frontend 形成可实例化的 canonical 程序；Intent adapter 把它结构化地绑定到当前 task。** 不另写 Weft parser，也不让 runtime 执行专家 Python 函数作为算子计算。

接口至少要处理：

- 输入/输出、view 与 scalar captures 的绑定，以及必要的 encoding declarations；
- shape symbols、axis identities 与动态 extents；
- domain/Level 的宿主关系和内部标识；
- 局部资源、输出写入与外围 lifetime/effect；
- 专业程序本身的数值义务及允许的参数域。

不能简单拼接两个完整 kernel body。现有 Weft `KernelOp` 是 isolated single-block，root domain、entry arguments 和 terminal return 都有约束；kernel return 当前不能携带 SSA results。[KernelOp](/home/kingdom/phdworks/TianchenRV/include/Weft/Dialect/Kernel/IR/KernelOps.td:134)、[Return verifier](/home/kingdom/phdworks/TianchenRV/lib/Dialect/Kernel/IR/KernelDialect.cpp:954)

第一版应明确一种足够支撑实际微程序的输入/输出形式，不能承诺任意 standalone module 都可自动内联。源级 fragment export 与 canonical adapter 的具体实现需要结合现有 frontend 落地；本文确认了必要边界和现有缺口，不宣称这套接口已经存在。

Weft 的 RVV/IME physical layout、machine resources 和 intrinsic lowering 继续归外部 compiler。不要在 Intent 中再实现一套 RVV/IME 指令选择器。Mojo 和 Weft 的停止层次可以不同，只要都消费完整的 CPU 程序及明确的实现边界。

## 9. 当前代码应怎样修改

以下是建议变更范围，不是已执行的修改，也不是要求一次重构整个 CPU 后端。

| 当前位置 | 保留 | 建议修改 |
|---|---|---|
| `KIRToCPU.cpp` | canonical→CPU 的唯一 construction | 让所选新数值/format schema 完整进入 CPU；移除仅因当前实现为 f32 而产生的公共入口限制，支持范围逐项明确 |
| `CPUAttrs.td`、`CPUOps.td` | CPU ABI、Tasks、Reduce 等真实 carrier | 拆分共同绑定与实现绑定；补实际所需的 structured numeric/format 信息，不造空的 matrix/engine 字段 |
| `PhysicalProgram.cpp`、`AxisRelations.cpp` | current-program facts、访问/effect/reuse 分析 | 为 structured compute 提供稳定语义查询；不能把精确二维单 FMA matcher 当成所有 contraction 的定义 |
| `BlockContractions.cpp` | task/cache blocking、输出 coverage、外围复用 | 把 width-dependent micro-panel、具体行/列尾部策略与局部 partial 供应从无条件 shared 决定中分离，由实现需求约束后形成 |
| `Passes.cpp` | transformation 顺序、候选实例化、verifier | 增加需求查询/选择/展开边界；不再以 `vector/contraction` 两张固定行表表达全部实现空间 |
| `MaterializeRegisterContractions.cpp` | 当前 f32 FMA 微核的有效结构 | 作为 Mojo vector implementation 的展开逻辑复用；固定 unroll/prefetch 等策略及合法参数与实现相邻 |
| `VectorizeLoops.cpp` 等 | 可复用的通用 vector realization 工具 | 由选择该 realization 的 provider 调用，区分通用算法和当前 f32/固定宽度约束 |
| `Target/Mojo/Transforms` | native ABI、FP environment、surface legalization | 接入明确的 implementation 选择与展开；serializer 继续只拼写 |
| `Target/Weft/Transforms` | CPU task→Canonical Weft 的正式转换 | 增加 format-aware 接口及微程序实例化/集成；保留简单 structured op 的直接转换，不把它降格为异常 fallback |
| tuning 与 runtime | 有限候选、真实测量、artifact/winner 分离 | 实现身份和实际参数进入候选；仅搜索适用实现，不混入格式/模型量化算法自动选择 |

目录责任建议：共同 IR/analysis/transformation 契约继续在 `Dialect/CPU`；Mojo/Weft 专业实现、参数和适用性分别与各自 `Target/.../Transforms` 相邻，数量增长时再按稳定的实现职责成组。外部 Weft helper 属于其 source/std 实现；Intent 只持有集成契约和 adapter，不复制整套 Weft 到项目内。不要新增顶层通用 `microkernels/` 杂物目录或另一份 planning IR。

### 9.1 配置必须从“所有 CPU 一张向量表”改为分层所有权

当前可编辑配置在 [CPU TuningProfiles.json](/home/kingdom/phdworks/intentdsl/lib/Dialect/CPU/Transforms/TuningProfiles.json:1)。它不是无效 tuning，但不同职责现在被放进同一表和同一 Configuration。

建议区分：

- shared：task grain、外层 block 与跨块组织参数；
- implementation：该微程序真正消费的局部 block、vector/replica、unroll 等；
- external compiler：外部 provider 自己负责并公开的机器参数。

若某参数影响 shared block 与 implementation 两边，必须有一个 binding owner 和显式约束关系，不各自选择两个值。目前 `WeftTarget` 把 `cpu-vector-bits` 明确作为 construction budget；它会影响 shared 微块，但并不证明或直接指定 Weft 的实际 RVV 宽度/LMUL。应拆开这个 f32 导向的构造预算、实现需求和外部 compiler 的机器参数，避免把三者当成同一件事。[WeftTarget](/home/kingdom/phdworks/intentdsl/python/intent/targets/weft.py:21)

### 9.2 修改重点不是批量搬文件

成功的变化应体现在程序：同一个 structured CPU operation 能根据适用条件实例化不同的合法目标程序；外层任务和共享值不被重建或丢失；实现要求参与数据供应；专业量化程序能通过正式接口进入 lowering。

仅新建 registry、改 pass 名、移动现有文件，或增加从未消费的配置字段，都不能算完成。已有 f32 路径应迁到唯一的新职责边界，不保留旧 pipeline 与新 implementation pipeline 两套等价执行入口。

## 10. 哪些规格需要重新澄清，而不是推翻

1. **`doc/compiler/cpu-program-ir.md`**：保留 CPU execution family、tasks、数值和 effects；明确 common structured program 与 implementation realization 的阶段边界；将 §5 的“vector、accumulator graph、packing”按作用范围拆分，避免被解读为全部必须先由共同层完成。
2. **`doc/compiler/README.md` 与 target lowering 说明**：明确 provider legalization 可以实例化专家编写的程序，不只做 API spelling。禁止的是 kernel-name 整算子替换和 emitter 重建，不是有语义边界的 programmable lowering。
3. **CPU 的 capability/physical binding 说明**：将资源事实、共享参数、实现参数分别绑定到真实消费者；不把固定 f32 向量配置当 CPU 定义。
4. **DSL 数值/格式规格**：只有选定的量化计算确实缺少上层数值合同才扩展。不要因为要引入 implementation registry，就扩大普通算术的 fast-math 权限或复制 Weft source model。

当前规格已经允许 structured operation 有不同合法 realization，也允许 provider-local lowering。不是所有建议都与现有设计冲突；需要明确调整的主要是职责分配、可编程实现接口，以及真实缺失的量化合同。

## 11. 收束到什么结果，哪些决定尚未授权

建议下一项实现工作围绕**结构化 CPU 程序＋可编程目标 realization**闭合，而不是再新增一个纯 mapping 后端，也不是先做全量量化格式。

最小有意义的完成结果应同时包含：

- 当前 f32 dense microkernel 被接入正式实现机制，现有任务/复用与 native 路径不丢失；
- 一个数值和格式合同明确的实际专业计算块，能由专家实现实例化到 CPU→Weft 或 CPU→Mojo 路径，证明机制不只是把原 f32 builder 包装了一层；
- 生成产物能看见被展开的结构及完整接口，不能靠 kernel 名、side recipe 或 runtime 算法替代；
- 对具有执行环境且受改动影响的项，复用现有生产 benchmark，报告真实算子时间并在同次运行做一次原容差检查；不增设独立数值/回归测试，不用“必须全量重跑”推迟实现。

本报告不替用户决定首个量化格式、是否连带 activation quantization/persistent repack、设备运行边界，以及新性能输入和门槛。这些会改变用户可见语义与实现范围，应在后续 Shape 中明确；当前报告没有创建新 change，也没有将已有验收自动视为接受归档。

最终建议可以概括为：**CPU IR 留在 task/block 与 structured computation 这一层；Intent compiler 负责从算法形成和优化这个程序，并协调专业实现；难以通用推导的内部结构交给专家可编程 realization；Mojo/Weft compiler 再完成其机器层工作。** 这不是放弃编译器职责，而是把职责放在有依据、可实现和可扩展的位置。
