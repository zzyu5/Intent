# 开发者指南

本页帮助开发者和 agent 找到应修改的模块，以及可以复用的能力。语言和编译器合同以 [doc/index.md](doc/index.md) 为入口；本页不另定义语义。安装与工具链准备见 [安装说明](environment/README.md)，工作约定见 [AGENTS.md](AGENTS.md)。

## 先选择正确的职责层

Intent 作者定义算法、逻辑成员、数值合同和显式 kernel/host 编排。编译器形成保持这些事实的物理程序；provider 和下层编译器继续完成其负责的布局、指令与调度。

GPU 与 CPU 有独立的 physical IR 和 passes，这是执行模型的边界。它们可以共享坐标关系、常量求值、identity 和代数合法性分析，但不必使用相同的 loop、ownership 或 storage 变换。Mojo 与 Weft 则复用同一 CPU family pipeline。

开始修改前，先确定问题属于作者算法、语义分析、physical construction、family transformation、provider realization，还是运行时。某个合法程序 lowering 失败，不自动意味着作者必须改写算法。

- DSL 或 frontend：读 [作者指南](doc/dsl/authoring.md) 和涉及的语言章节。
- 编译器：读 [编译边界](doc/compiler/README.md) 与 [analysis/pass 合同](doc/compiler/passes-and-analyses.md)。
- Provider：再读 [target lowering](doc/compiler/target-lowering.md)；CPU implementation 还需读 [CPU program](doc/compiler/cpu-program-ir.md)。
- 改动涉及成熟目标已有能力时，对照 `../ref/triton`、`../ref/tilelang` 或对应目标实现的具体代码，核对操作合同和职责归属。

## 从一次调用定位代码

```text
Python definition
  → frontend：语义检查、specialization、canonical KIR
  → compiler driver：目标解析与 intent-compile 调用
    ├─ KIRToGPU → GPU transformations → GPU provider
    ├─ KIRToCPU → CPU transformations → Mojo / Weft
    └─ KIRToDSA → DSA transformations → BANG C legalization
  → provider serialization
  → runtime materialization → provider 编译与 launch
```

这条路径的主要入口如下：

| 环节 | 代码入口 | 负责什么 |
|---|---|---|
| Kernel/helper 定义 | [python/intent/api/](python/intent/api/) | `@intent.kernel`、`@intent.fn` 及 source definition |
| Python frontend | [frontend/compilation/compiler.py](python/intent/frontend/compilation/compiler.py)、[frontend/lowering/](python/intent/frontend/lowering/) | 类型化捕获、helper、控制与 intrinsic lowering |
| 编译调用 | [compiler/pipeline.py](python/intent/compiler/pipeline.py)、[compiler/toolchain.py](python/intent/compiler/toolchain.py) | 目标解析、compiler 定位、产物缓存与失败阶段 |
| C++ 编译入口 | [intent-compile.cpp](tools/intent-compile/intent-compile.cpp) | 选择 execution family 与 provider pipeline |
| 运行时 | [runtime/artifact.py](python/intent/runtime/artifact.py)、[runtime/](python/intent/runtime/) | 绑定 generated source/native program，管理调用与输出 |

已有 `.codegraph/` 时，可用 Codegraph 查询上面的具体文件或符号，再补读未覆盖部分；没有索引不必先创建索引才能工作。

## 按贡献类型选择模块

| 要做的修改 | 首先看哪里 | 复用与边界 |
|---|---|---|
| 新增算法 | [examples/kernels/](examples/kernels/)；普通 host 调用参考 [examples/softmax.py](examples/softmax.py) | 保留目标无关算法；多 kernel 编排由作者显式写出 |
| 接入生产运行 | [GPU providers](experiments/gpu/providers/)、[CPU providers](experiments/cpu/providers/)、[MLU providers](experiments/mlu/providers/) | 在已有 registry 接入完整 callable；公共运行机制在 [experiments/_common/](experiments/_common/) |
| 修复公开 API 或 frontend 语义 | [language/](python/intent/language/)、[frontend/semantics/](python/intent/frontend/semantics/)、[lowering/intrinsics/](python/intent/frontend/lowering/intrinsics/) | 优先修复合法表达的处理；不把实现限制反写成语言规则 |
| 共享 analysis/证明 | [include/Intent/Analysis/](include/Intent/Analysis/)、[lib/Analysis/](lib/Analysis/) | 只读查询与证明；由消费者改写各自的 current IR |
| 改初始 physical construction | [KIRToGPU](lib/Conversion/KIRToGPU/)、[KIRToCPU](lib/Conversion/KIRToCPU/)、[KIRToDSA](lib/Conversion/KIRToDSA/) | 从 immutable KIR 建立完整 family program，不留给 serializer 猜 |
| GPU family 优化 | [GPU/Transforms/](lib/Dialect/GPU/Transforms/)、[Passes.cpp](lib/Dialect/GPU/Transforms/Passes.cpp) | 变换当前 GPU program；coherent group 负责自身的关系维护 |
| CPU family 优化 | [CPU/Transforms/](lib/Dialect/CPU/Transforms/)、[Passes.cpp](lib/Dialect/CPU/Transforms/Passes.cpp) | 修改 task/block、供数、存储和计算组织，复用 CPU implementation 接口 |
| DSA family 优化 | [DSA/Transforms/](lib/Dialect/DSA/Transforms/)、[Passes.cpp](lib/Dialect/DSA/Transforms/Passes.cpp) | 从当前 local-memory program 形成矩阵供数、索引与局部值复用；不回读 KIR 选择整算子路径 |
| 修改 IR 与 verifier | [include/Intent/Dialect/](include/Intent/Dialect/)、[lib/Dialect/](lib/Dialect/) 中相应 `IR/` | 前者声明 types/ops/attributes，后者实现与验证；优先使用已有 carrier |
| Provider primitive/form | [lib/Target/](lib/Target/) 中相应 `Transforms/` | 处理真实目标能力、合法性和必要 local structure |
| CPU micro-kernel | [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation.h)、[Mojo implementations](lib/Target/Mojo/Transforms/Implementations.cpp)、[Weft implementations](lib/Target/Weft/Transforms/Implementations.cpp) | 按当前 operation、dtype 和 capability 选择局部实现；外层分块/供数仍由 CPU passes 负责 |
| 设备与运行时接入 | [targets/](python/intent/targets/)、[targets/base.py](python/intent/targets/base.py)、[runtime/](python/intent/runtime/) | Host 解析目标、绑定产物与 launch；不把设备分支加入 KIR |

DSA 的 [MatrixSupply.cpp](lib/Dialect/DSA/Transforms/MatrixSupply.cpp) 消费已有 LoadTile、MatMul、Store 和循环关系，形成协作或常驻供数；[CollectiveGather.cpp](lib/Dialect/DSA/Transforms/CollectiveGather.cpp) 从当前 offsets 写入、task 坐标、只读视图与 stride 关系证明相邻参与者可以共享 gather 供数。两个变换都消费完整的普通 DSA program，不通过 construction 候选侧表选择路径。[BANG C driver](lib/Target/BangC/Transforms/Legalize.cpp) 再依次完成原生计算、workspace、实现选择、局部组合、同步和最终存储绑定。当前 DSA 的矩阵与 group 合同、BANG C 实现仍以 MLU370 为已实现边界，目录分层不代表已经支持其他 DSA 设备。

## 共享分析与完整变换

分析回答当前程序满足什么条件；变换根据这些条件改变程序。不要为了共享代码，让 analysis 创建新 operation、替换 uses，或者替某个 family 决定具体循环与存储结构。

选择共享位置时，可以依次问：

1. 这是多个消费者都需要的语义关系或合法性规则吗？共享规则与查询接口。
2. 它依赖某种 physical topology、fragment representation 或 storage 吗？保留在相应 family。
3. 它只是 provider API 拼写差异吗？直接序列化已有事实，不增加 shared IR。
4. 它确实改变当前 program，并被后续分析或变换使用吗？在正确层形成显式 IR，再交给 serializer。

一个实际的共享入口是 [matchOnlineSummaryCombine](include/Intent/Analysis/OnlineSummaryCombine.h)：它只读 combine graph，识别 validity、guarded maximum、指数缩放和 weighted sum 关系，返回指向当前 SSA 的 `OnlineSummaryCombineRelations`。调用方提供投影与常量查询规则；它不创建 IR，也不选择 tile、storage 或 loops。

[Canonical matcher](lib/Analysis/OnlineSummary.cpp) 的 `matchOnlineSummary` 和 [GPU matcher](lib/Dialect/GPU/Transforms/OnlineSummary.cpp) 的 `matchOnlineSummaryMerge` 共用这个核心，同时保留各自的四字段 summary 识别、axis/shape/cast 检查、候选枚举及 GPU physical schema 检查。扩展共同 combine 规则时从共享核心开始；调整 fragment 投影时改 GPU 适配层。CPU 当前没有接入这个 matcher，不能把这项复用描述成所有 family 已共用。

[IntegerRelations.h](include/Intent/Analysis/IntegerRelations.h) 的 `foldIntegerDifference` 复用 `UniformExpression` 描述，只读折叠加减、常数乘法及等宽整数/index cast 的变化系数。[CPU VectorizeLoops](lib/Dialect/CPU/Transforms/VectorizeLoops.cpp) 用它判断循环坐标差值，再检查连续 stride、别名与依赖；[DSA CollectiveGather](lib/Dialect/DSA/Transforms/CollectiveGather.cpp) 用它判断四个参与者之间的地址差值，保留自己的 task 商余关系、只读视图、局部 buffer 写入和控制一致性证明。CPU 的这个 vectorizer 当前由 Mojo legalization 调用，共享 CPU family 不意味着所有 provider 都调用它。

该核心只接受 i64 或调用方明确绑定为 64 位的 index；两个适配层依据 Intent 的逻辑 index 合同传入位宽，不假定任意 MLIR index 都是 64 位。值运算仍遵守模整数语义，系数的加减乘另外检查是否能用 `int64_t` 表示；失败返回 `Unknown`，不能当作系数零。窄整数回绕后的扩宽需要独立范围证明，地址有效性、memory effects 与拓扑也不由系数证明。GPU 的按位宽模运算规范化和 source-axis 关系分析有不同合同，不应仅因都有 Add/Mul 就接到这一接口。

公开的 transformation 入口必须完成自身改写所需的 relation closure，使调用方得到满足 postcondition 的 current program。中间 repair helper 不因可以被调用就成为独立 pass；pipeline 负责次序，不应成为调用者必须记忆的隐式修复配方。

例如，[realizeRegionFolds / realizeRegionScans](lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp) 在完成 region 改写后，自身调用 [closeValueRelations](lib/Dialect/GPU/Transforms/ValueRelations.cpp)，通过工作队列闭合受影响的 value/access/aggregate 关系。这两个 region 阶段在 [GPU pipeline](lib/Dialect/GPU/Transforms/Passes.cpp) 中只调度完整入口，随后验证 postcondition。调用者不需要再附加一串 repair 调用；这也不要求 CPU 使用相同的关系维护方式。

### GPU 中直接可复用的接口

[GPU Passes.h](include/Intent/Dialect/GPU/Transforms/Passes.h) 只暴露完整变换与验证入口。实现内部需要的查询与改写按下表包含具体头文件，不通过一个通用 Utilities 模块取得所有能力。

| 需要的能力 | 接口 | 使用方式 |
|---|---|---|
| 当前 value/access 的坐标、范围和复用事实 | [Analysis/PhysicalProgram.h](include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h) | 只读 current IR；相关 def-use、类型或范围改变后重算 |
| scalar/fragment schema与投影轴 | [Analysis/ValueSchema.h](include/Intent/Dialect/GPU/Analysis/ValueSchema.h) | 只读查询当前类型与轴关系，不创建值、不选择 blocking |
| 物理整数表达式求值 | [Analysis/UniformValues.h](include/Intent/Dialect/GPU/Analysis/UniformValues.h) | `evaluatePhysicalExpression` 接受 symbolic-leaf binding；算术和溢出检查共用一份实现 |
| range/loop 中的整数比较与完整 tile 界限 | [Analysis/IndexPredicates.h](include/Intent/Dialect/GPU/Analysis/IndexPredicates.h) | `proveRangeComparison`、`queryCompleteTileLimit` 与 `queryIndexComparisonBound` 只读当前范围；区分已证明的真值、条件蕴含和未知 |
| 常量、大小关系与访问对齐 | [Analysis/IndexRelations.h](include/Intent/Dialect/GPU/Analysis/IndexRelations.h) | `IndexRelations` 共用于范围谓词、Triton descriptor 与 cuTile tile access；按 typed index 与回绕合同证明，不创建 guard 或选择原生 form |
| 参数声明与完整候选绑定检查 | [Analysis/PhysicalParameters.h](include/Intent/Dialect/GPU/Analysis/PhysicalParameters.h) | `PhysicalParameterSpace::read` 建只读快照；改变声明后重读；候选仍保存在 IR |
| fragment 结构资源估计 | [Analysis/Resources.h](include/Intent/Dialect/GPU/Analysis/Resources.h) | `FragmentResourceAnalysis` 缓存稳定 IR 的类型与参数使用关系；类型或 IR 改写后重建。估计不代替下层布局、寄存器分配和 occupancy |
| specialization 后才能判定的资源约束 | [Transforms/Resources.h](include/Intent/Dialect/GPU/Transforms/Resources.h) | 将 deferred reduction bounds 写成当前 IR 的断言，供 Triton/cuTile 兑现；不是 analysis 中的隐藏改写 |
| value projection、replay、validity 与显式常量 | [Transforms/ValueMaterialization.h](include/Intent/Dialect/GPU/Transforms/ValueMaterialization.h) | 传入当前 schema、source-axis 与 replay scope；由调用者决定合法的变换范围 |
| 改写后的 value/access/aggregate 关系闭合 | [Transforms/ValueRelations.h](include/Intent/Dialect/GPU/Transforms/ValueRelations.h) | 在完整 transformation 内调用，随后验证，不能让 serializer 补修 |
| coverage traversal、参数生命周期 | [Traversal.h](include/Intent/Dialect/GPU/Transforms/Traversal.h)、[PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/PhysicalParameters.h) | 分别改变当前 ranges/access 与参数引用；参数替换同时覆盖 SSA、types 和 attributes |
| predication、workspace 与 retained slice | [Predication.h](include/Intent/Dialect/GPU/Transforms/Predication.h)、[Storage.h](include/Intent/Dialect/GPU/Transforms/Storage.h) | 保持 effects、allocation ownership 与 lifetime；不由 provider 字符串猜测 |

收缩计算的完整入口在 [RealizeContractionBlocking.cpp](lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp)。同目录下 `ContractionSources` 负责合法的 source 规范化，`ContractionAnalysis` 负责轴与范围查询，`ContractionValues` 负责 replay，`ContractionProjection` 负责结果关系，`ContractionTraversal` 与 `ContractionBlocking` 形成具体循环与 ownership。普通与 scaled contraction 共用能成立的判定和构造机制，各自的 dtype、scale 与 packing 条件留在相应实现。Provider 只通过 [Contraction.h](include/Intent/Dialect/GPU/Transforms/Contraction.h) 调用必要的形状规范化与查询，不接管 shared blocking。

[ContractionTraversal](lib/Dialect/GPU/Transforms/ContractionTraversal.cpp) 保留完整 retained result 的原始 contraction：完整物理 extent 属于已有 tile 候选域且不超过所选 tile，reduction 也已具备原生执行条件时，直接使用原 shape、读快照和 accumulator，避免分片与拼回；其它情况保留分片与 padding 路径。这是 current IR 的完整分支，不由 serializer 根据运行时 shape 猜测。

Pointwise 的两个完整入口也在同一 driver 文件 [RealizePointwiseBlocking.cpp](lib/Dialect/GPU/Transforms/RealizePointwiseBlocking.cpp)：`realizePointwiseOwnership` 形成 ownership 与 program mapping；`realizePointwiseBlocking` 在已有 mapping 上形成局部 blocking、写回和复用 traversal。两者有各自明确的依赖次序，通过相邻私有头 [Pointwise.h](lib/Dialect/GPU/Transforms/Pointwise.h) 使用以下机制：

| 私有模块 | 修改入口与职责 |
|---|---|
| [PointwiseAnalysis.cpp](lib/Dialect/GPU/Transforms/PointwiseAnalysis.cpp) | 查询 source-axis、结构化范围用途、写入 effect 与现有 mapping 坐标；同一个 dimension 不自动代表同一个 Cartesian occurrence |
| [PointwiseCoverage.cpp](lib/Dialect/GPU/Transforms/PointwiseCoverage.cpp) | 兑现 scan/reduction 的完整 coverage，形成固定或局部范围、tail validity，保留不能安全 replay 的 gather source，并完成值关系闭合 |
| [PointwiseOwnership.cpp](lib/Dialect/GPU/Transforms/PointwiseOwnership.cpp) | 提升 workset，处理 axis occurrence 与 ownership 依赖，选择 ownership 并写入 program mapping |
| [PointwiseTraversal.cpp](lib/Dialect/GPU/Transforms/PointwiseTraversal.cpp) | 重放合法 value graph，选择并形成写回/复用 traversal，兑现已确定 ownership 的 histogram |

`PointwiseRewrite` 只保存一次完整变换期间的工作状态；driver 在相关改写后重新读取 current-IR facts，执行决定写入当前 IR，不跨两个入口保留第二份 plan。新增局部机制放入对应私有模块；需要多个 GPU 变换复用的只读关系才进入公开 Analysis。

[SimplifyRangePredicates.cpp](lib/Dialect/GPU/Transforms/SimplifyRangePredicates.cpp) 是 `IndexPredicates` 的改写消费者：全体物理 lane 上已证明的比较可替换为布尔常量；只有条件蕴含时，写入 specialization guard 与原谓词的逻辑或，未满足 guard 时仍保留原判定。它不把未证明的 shape 关系变成输入要求，也不负责 provider 的 native-load 分支或 MMA 循环组织。

新增一个 physical rewrite 时，先确定它读取的 current-IR facts，从上表选择查询或 materialization 接口；将 rewrite 和必要 relation closure 放进一个完整入口；在 family pipeline 中安排依赖位置与 postcondition 验证。新增只读查询应放 Analysis，只有本模块用的算法细节留在相邻私有实现，不扩大 Passes.h。CPU 或 DSA 的类似优化先复用它们自己的 analysis 和 storage/control 合同，只有与执行拓扑无关的规则才上提到公共 Analysis。

## Provider 与 runtime 扩展

先比较 Intent operation 与目标原语的合同，包括 dtype、accumulator、NaN/tie、顺序、effects 和 ABI。合同吻合时优先直接映射；例如 provider 已有 reduce/scan，就不在 Intent 再实现其线程通信与归约树。

- 同一 execution family 的新 provider，先复用已有 physical program 和 family passes。
- 新设备代际优先通过已有 capability consumers 和 legality predicates 表达；不因设备代号不同就建立一套 dialect。
- 只有共同 IR 确实缺少、且多个 consumers 或独立 verifier 需要的目标结构，才增加 local extension。
- `lib/Target/<provider>/Serialization/` 打印已经决定的当前程序，不重新选择 ownership、workspace、pipeline 或调优参数。
- `python/intent/runtime/` 绑定目标要求并执行；无法兑现的能力应明确报错，不改变算法或隐藏失败。

CPU 的 implementation registry 是明确的局部扩展点。GPU provider 通常复用下层 compiler 的 primitives 与布局机制，不需要为了目录形式对称再建一套同名 leaf 系统。

Triton 的 [Passes.cpp](lib/Target/Triton/Transforms/Passes.cpp) 调度 grid、prepare-memory、native-forms 和 finalize；[Legalize.cpp](lib/Target/Triton/Transforms/Legalize.cpp) 通过私有 [Legalization.h](lib/Target/Triton/Transforms/Legalization.h) 组合完整阶段。cuTile 的 [Passes.cpp](lib/Target/CuTile/Transforms/Passes.cpp) 调度 prepare、native-program 和 finalize，私有入口在 [Legalize.h](lib/Target/CuTile/Transforms/Legalize.h)。按实际职责选择相邻模块，不把新增规则继续堆入 driver：

| Provider | 模块 | 职责 |
|---|---|---|
| Triton | [AccessForms.cpp](lib/Target/Triton/Transforms/AccessForms.cpp) | 从当前 access facts 形成 descriptor、block pointer 与 pointer 表示；对齐关系复用 GPU 分析 |
| Triton | [Collectives.cpp](lib/Target/Triton/Transforms/Collectives.cpp) | gather/scatter 的目标表达、scan tail 与原生 reduce/scan callback |
| Triton | [Supply.cpp](lib/Target/Triton/Transforms/Supply.cpp) | ordered access dependencies、CTA 同步与 load-loop policy |
| Triton | [Values.cpp](lib/Target/Triton/Transforms/Values.cpp)、[Verify.cpp](lib/Target/Triton/Transforms/Verify.cpp) | 前者形成 contract/value 表示，后者验证完整 Triton surface |
| cuTile | [NativeProgram.cpp](lib/Target/CuTile/Transforms/NativeProgram.cpp)、[NativeAccess.h](lib/Target/CuTile/Transforms/NativeAccess.h) | 收集本次输入、声明 provider 参数并统一提交 native replacements；原 GPU SSA 保留到相关 facts 消费完成 |
| cuTile | [NativeAccessAnalysis.cpp](lib/Target/CuTile/Transforms/NativeAccessAnalysis.cpp) | 只读坐标、shape 与原生访问资格 |
| cuTile | [NativeAccess.cpp](lib/Target/CuTile/Transforms/NativeAccess.cpp) | 构造原生 memory/extraction forms 及保持语义的 guards |
| cuTile | [ComputeForms.cpp](lib/Target/CuTile/Transforms/ComputeForms.cpp) | 构造 reduce、scan、histogram 和 MMA primitives |
| cuTile | [Legalize.cpp](lib/Target/CuTile/Transforms/Legalize.cpp) | 共享输入准备、宽索引与循环收尾、完整 surface 验证 |

这些私有 facts 和待提交 replacements 只服务一次变换；阶段之间传递当前 IR 与不可变 profiles，不保留另一份执行计划。Triton 的局部候选在 native-forms 阶段内闭合为 IR configs；cuTile 提交替换后才进入后续循环与配置变换。

对齐推断中的参数域必须是当前证明可依赖的域。`ResidentWorkers` 会由 provider 配置重绑定，公共关系查询不把它的临时候选当作常量或整除事实；coverage capacity 也不等于 logical extent。分支内额外对齐条件由调用方提供局部叶证明，不能传播成其它分支的全局性质。新增整数规则先核对位宽、回绕与除法合同，再接入共同查询，避免在各 provider 重写递归证明。

Triton/cuTile 的 `Transforms/Configurations.cpp` 负责各自的候选策略与资源合法性，使用共同的参数绑定分析。Triton 的 tensor/descriptor/collective 约束从当前 IR 一次收集后逐候选求值；cuTile 保留 launch 与 memory hints 的相关候选及 resident-capacity 绑定。新增设备约束时在对应模块处理，不复制参数解析器，也不把 Triton TTGIR 的布局、MMA 或 pipeline 再实现一遍。

资源查询的 `Unknown` 表示当前求值无法证明，可能来自未绑定维度，也可能来自表达式求值失败；不能据此宣称候选合法或已精确证明资源不足。Shared 候选策略只按可得事实筛选和绑定，保留需要 specialization 或下层 compiler 判断的约束；局部候选 matcher 也不等同于完整 coverage 证明。

BANG C 的 [Storage.cpp](lib/Target/BangC/Transforms/Storage.cpp) 分开只读 `measureStorage` 和最终 `bindStorage`。前者可供局部复用与供数变换比较资源需求，后者才写入目标偏移；公共 alias/lifetime 查询在 [DSA Analysis](include/Intent/Dialect/DSA/Analysis/PhysicalProgram.h)。新增目标实现需要的 workspace 在目标变换中形成显式 operand，最终由目标 verifier 检查，不能在资源查询或 serializer 中补写。

DSA 的 `isSumOfIntegerProducts` 使用 MLIR 的整数表达式规范化证明地址关系，允许单位 stride 消除、常数结合和交换后的等价索引；供数 matcher 不应依赖某一种 Add/Mul 树形。`BangCTarget.shapes/strides` 是编译变体的 ABI 约束，运行时会核对实参。已有 [MLU 调用适配](experiments/mlu/providers/bangc/common.py) 从实际 tensor 绑定这些事实，并原样传输其 stride/offset；尚未分配的输出不猜布局。不要用忽略已知布局来规避 matcher 缺陷，也不要从 shape 猜连续布局。

## 构建、定位和完成改动

使用 [安装说明](environment/README.md) 中已配置的工具链。C++ 构建目录放在仓库外，例如：

```bash
cmake --build /path/to/intent-build --target intent-compile --parallel 4
INTENT_COMPILER=/path/to/intent-build/tools/intent-compile/intent-compile \
  python examples/softmax.py
```

`INTENT_COMPILER` 只切换 C++ 可执行文件。若修改 `python/intent/`，按安装说明重新安装当前 checkout，并核对验证解释器的 `intent.__file__`，使运行使用本次修改的代码。

这个调用示例用于理解公开入口；修改具体能力时，选择实际受影响的既有生产程序及其所属实验组入口，保留原输入、容差和完整 callable 计时合同。

定位失败时先看 `CompilationStageError.stage`、`cache_directory`、`artifact.source` 和 `artifact.mlir`。区分 frontend、physical program、provider source、下层编译及实际 launch，不把所有失败归成“后端不支持”。

作者位置沿 [SourceUnit.location](python/intent/frontend/source/unit.py)、[canonical KIR 打印](python/intent/frontend/mlir/builder.py) 和 [compiler IR 输出](tools/intent-compile/intent-compile.cpp) 保存在标准 MLIR location 中。缓存的 `input.mlir`、`kernel.mlir` 与 operation 诊断使用这条位置链；新增 rewrite 创建或克隆 operation 时保留相应 source location，不用旁表替代。编译日志位于同一 `cache_directory` 的 `compiler.log`。

### 查看完整 transformation group 的 IR 与编译时间

GPU shared、DSA、BANG C 及 Triton/cuTile provider 的完整 groups 是具名、注册的 MLIR module passes；标准 PassManager 负责调度、IR 打印和计时。Shared group 完成自己的 rewrite 与 relation closure，再检查 family postcondition；provider native forms 出现后检查结构和 operation 合同，finalize 才完成完整 surface verifier，不能再用拒绝 provider dialect 的 shared GPU verifier。内部 repair helper 不注册成可独立运行的 pass。公共 [PassManager.h](include/Intent/Transforms/PassManager.h) 只应用 MLIR 标准 instrumentation 选项；它不选择优化或管理另一份 program。

复用已有生产 case 的 `input.mlir` 和原编译参数。下面 `production_args` 是该 GPU 调用已有的目标、设备能力与 profile 参数组成的 Bash 数组，`dsa_bindings` 则是已有 DSA 调用的 shape、stride 和 block 参数；不要用另一台机器的参数替换它们。这些命令只重新编译已有输入，不启动 benchmark。

```bash
compiler=/path/to/intent-build/tools/intent-compile/intent-compile
input=/path/to/existing-production-cache/input.mlir
dump_root=/tmp/intentdsl-pass-ir
mkdir -p "$dump_root"

"$compiler" "${production_args[@]}" "$input" \
  --stop-after-shared --ir-output="$dump_root/shared.mlir" \
  --mlir-print-ir-before=intent-gpu-form-pointwise-blocking \
  --mlir-print-ir-after=intent-gpu-form-pointwise-blocking \
  --mlir-print-ir-tree-dir="$dump_root/gpu" --mlir-timing

# 失败时输出该 pass 留下的 current IR；与普通 after 打印分开使用。
"$compiler" "${production_args[@]}" "$input" \
  --stop-after-shared --ir-output="$dump_root/shared.mlir" \
  --mlir-print-ir-after-failure \
  --mlir-print-ir-tree-dir="$dump_root/failure" --mlir-timing

dsa_input=/path/to/existing-dsa-production-cache/input.mlir
"$compiler" --target=bangc "${dsa_bindings[@]}" "$dsa_input" \
  --stop-after-shared --ir-output="$dump_root/dsa.mlir" \
  --mlir-print-ir-before=intent-dsa-matrix-supply \
  --mlir-print-ir-after=intent-dsa-matrix-supply \
  --mlir-print-ir-tree-dir="$dump_root/dsa" --mlir-timing
```

去掉 `--mlir-print-ir-tree-dir` 会把快照写到 stderr；`--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` 可查看已接入 instrumentation 的所有阶段。若增加 `--mlir-print-ir-module-scope`，同时传 `--mlir-disable-threading`。`--mlir-timing` 是编译 pass 时间，不能当作 kernel 执行时间。CPU 当前只有既有 canonicalizer/CSE 调用接入这些 instrumentation，尚未把候选构造与完整 CPU pipeline 改成具名 groups。

查看 provider 阶段时，沿用同一输入、目标 options 和 tuning profile 的完整编译命令，去掉 `--stop-after-shared`，同时指定 `--ir-output` 与 `--source-output`。例如既有 [cuTile official_fmha](experiments/gpu/providers/cutile/attention.py) 编译 [flash_gqa_attention_fwd](examples/kernels/streaming/attention.py) 时，在原命令追加 `--mlir-print-ir-before=intent-cutile-native-program --mlir-print-ir-after=intent-cutile-native-program --mlir-print-debuginfo`，即可对照原生 form 形成前后的 IR 并显示作者位置；对应 Triton 阶段名是 `intent-triton-native-forms`。最终合法化分别看 `intent-cutile-finalize-program` 与 `intent-triton-finalize-program`。这些阶段名用于同一完整 pipeline 的诊断，不表示可以跳过其输入依赖和 tuning profiles 单独调用。

提交一个连贯变换前，说明它读取的 facts、合法条件、实际改写的 IR、保持的语义、失效或重算的分析，以及在哪个边界验证 postcondition。用必要的既有生产运行确认影响；不另建测试目录、平行结果表或额外评测矩阵。

实验运行与数据继续归入对应的 `experiments/{gpu,cpu,mlu,agent_tritonbench}/`；`examples/kernels/` 不承载 benchmark runner。报告结果时分别说明生成成功、编译成功、运行正确与已测性能，不用 pass 数量或文件拆分数量代表能力。

## Agent 的两个入口

编写 Intent 算法时，使用 README 中的 [manual MCP](README.md#use-with-an-agent) 查询公开声明、语言合同和必要最小片段；实现入口是 [manual.py](python/intent/tools/manual.py)。完整算法示例在 `examples/`，manual 不提供题解或执行结论。

修改编译器时，使用本页的模块导航、正式规格和当前源码；公开语言手册不承担 compiler implementation guide。合法性或职责不清楚时，先查原合同与真实消费者，再选择改动层次。
