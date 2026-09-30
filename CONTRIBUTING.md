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

公开的 transformation 入口必须完成自身改写所需的 relation closure，使调用方得到满足 postcondition 的 current program。中间 repair helper 不因可以被调用就成为独立 pass；pipeline 负责次序，不应成为调用者必须记忆的隐式修复配方。

例如，[realizeRegionFolds / realizeRegionScans](lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp) 在完成 region 改写后，自身调用 [closeValueAccessRelations](lib/Dialect/GPU/Transforms/ValueRelations.cpp)，维护 access-result、pointwise、access-value、reshape 和 contract 关系。这两个 region 阶段在 [GPU pipeline](lib/Dialect/GPU/Transforms/Passes.cpp) 中只调度完整入口，随后验证 postcondition。调用者不需要再附加一串 repair 调用；这也不要求 CPU 使用相同的关系维护方式。

### GPU 中直接可复用的接口

[GPU Passes.h](include/Intent/Dialect/GPU/Transforms/Passes.h) 只暴露完整变换与验证入口。实现内部需要的查询与改写按下表包含具体头文件，不通过一个通用 Utilities 模块取得所有能力。

| 需要的能力 | 接口 | 使用方式 |
|---|---|---|
| 当前 value/access 的坐标、范围和复用事实 | [Analysis/PhysicalProgram.h](include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h) | 只读 current IR；相关 def-use、类型或范围改变后重算 |
| scalar/fragment schema、投影轴、寄存器 footprint | [Analysis/ValueSchema.h](include/Intent/Dialect/GPU/Analysis/ValueSchema.h) | 返回类型或估算，不创建值、不选择 blocking |
| 物理整数表达式求值 | [Analysis/UniformValues.h](include/Intent/Dialect/GPU/Analysis/UniformValues.h) | `evaluatePhysicalExpression` 接受 symbolic-leaf binding；算术和溢出检查共用一份实现 |
| 参数声明与完整候选绑定检查 | [Analysis/PhysicalParameters.h](include/Intent/Dialect/GPU/Analysis/PhysicalParameters.h) | `PhysicalParameterSpace::read` 建只读快照；改变声明后重读；候选仍保存在 IR |
| value projection、replay、validity 与显式常量 | [Transforms/ValueMaterialization.h](include/Intent/Dialect/GPU/Transforms/ValueMaterialization.h) | 传入当前 schema、source-axis 与 replay scope；由调用者决定合法的变换范围 |
| 改写后的 value/access/aggregate 关系闭合 | [Transforms/ValueRelations.h](include/Intent/Dialect/GPU/Transforms/ValueRelations.h) | 在完整 transformation 内调用，随后验证，不能让 serializer 补修 |
| coverage traversal、参数生命周期 | [Traversal.h](include/Intent/Dialect/GPU/Transforms/Traversal.h)、[PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/PhysicalParameters.h) | 分别改变当前 ranges/access 与参数引用；参数替换同时覆盖 SSA、types 和 attributes |
| predication、workspace 与 retained slice | [Predication.h](include/Intent/Dialect/GPU/Transforms/Predication.h)、[Storage.h](include/Intent/Dialect/GPU/Transforms/Storage.h) | 保持 effects、allocation ownership 与 lifetime；不由 provider 字符串猜测 |

收缩计算的完整入口在 [RealizeContractionBlocking.cpp](lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp)。同目录下 `ContractionSources` 负责合法的 source 规范化，`ContractionAnalysis` 负责轴与范围查询，`ContractionValues` 负责 replay，`ContractionProjection` 负责结果关系，`ContractionTraversal` 与 `ContractionBlocking` 形成具体循环与 ownership。普通与 scaled contraction 共用能成立的判定和构造机制，各自的 dtype、scale 与 packing 条件留在相应实现。Provider 只通过 [Contraction.h](include/Intent/Dialect/GPU/Transforms/Contraction.h) 调用必要的形状规范化与查询，不接管 shared blocking。

新增一个 physical rewrite 时，先确定它读取的 current-IR facts，从上表选择查询或 materialization 接口；将 rewrite 和必要 relation closure 放进一个完整入口；在 family pipeline 中安排依赖位置与 postcondition 验证。新增只读查询应放 Analysis，只有本模块用的算法细节留在相邻私有实现，不扩大 Passes.h。CPU 或 DSA 的类似优化先复用它们自己的 analysis 和 storage/control 合同，只有与执行拓扑无关的规则才上提到公共 Analysis。

## Provider 与 runtime 扩展

先比较 Intent operation 与目标原语的合同，包括 dtype、accumulator、NaN/tie、顺序、effects 和 ABI。合同吻合时优先直接映射；例如 provider 已有 reduce/scan，就不在 Intent 再实现其线程通信与归约树。

- 同一 execution family 的新 provider，先复用已有 physical program 和 family passes。
- 新设备代际优先通过已有 capability consumers 和 legality predicates 表达；不因设备代号不同就建立一套 dialect。
- 只有共同 IR 确实缺少、且多个 consumers 或独立 verifier 需要的目标结构，才增加 local extension。
- `lib/Target/<provider>/Serialization/` 打印已经决定的当前程序，不重新选择 ownership、workspace、pipeline 或调优参数。
- `python/intent/runtime/` 绑定目标要求并执行；无法兑现的能力应明确报错，不改变算法或隐藏失败。

CPU 的 implementation registry 是明确的局部扩展点。GPU provider 通常复用下层 compiler 的 primitives 与布局机制，不需要为了目录形式对称再建一套同名 leaf 系统。

Triton/cuTile 的 `Transforms/Configurations.cpp` 负责各自的候选策略与资源合法性，使用共同的参数绑定分析。Triton 的 tensor/descriptor/collective 约束从当前 IR 一次收集后逐候选求值；cuTile 保留 launch 与 memory hints 的相关候选及 resident-capacity 绑定。新增设备约束时在对应模块处理，不复制参数解析器，也不把 Triton TTGIR 的布局、MMA 或 pipeline 再实现一遍。

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

提交一个连贯变换前，说明它读取的 facts、合法条件、实际改写的 IR、保持的语义、失效或重算的分析，以及在哪个边界验证 postcondition。用必要的既有生产运行确认影响；不另建测试目录、平行结果表或额外评测矩阵。

实验运行与数据继续归入对应的 `experiments/{gpu,cpu,mlu,agent_tritonbench}/`；`examples/kernels/` 不承载 benchmark runner。报告结果时分别说明生成成功、编译成功、运行正确与已测性能，不用 pass 数量或文件拆分数量代表能力。

## Agent 的两个入口

编写 Intent 算法时，使用 README 中的 [manual MCP](README.md#use-with-an-agent) 查询公开声明、语言合同和必要最小片段；实现入口是 [manual.py](python/intent/tools/manual.py)。完整算法示例在 `examples/`，manual 不提供题解或执行结论。

修改编译器时，使用本页的模块导航、正式规格和当前源码；公开语言手册不承担 compiler implementation guide。合法性或职责不清楚时，先查原合同与真实消费者，再选择改动层次。
