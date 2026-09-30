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
    └─ KIRToDSA → BANG C legalization
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
| 修改 IR 与 verifier | [include/Intent/Dialect/](include/Intent/Dialect/)、[lib/Dialect/](lib/Dialect/) 中相应 `IR/` | 前者声明 types/ops/attributes，后者实现与验证；优先使用已有 carrier |
| Provider primitive/form | [lib/Target/](lib/Target/) 中相应 `Transforms/` | 处理真实目标能力、合法性和必要 local structure |
| CPU micro-kernel | [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation.h)、[Mojo implementations](lib/Target/Mojo/Transforms/Implementations.cpp)、[Weft implementations](lib/Target/Weft/Transforms/Implementations.cpp) | 按当前 operation、dtype 和 capability 选择局部实现；外层分块/供数仍由 CPU passes 负责 |
| 设备与运行时接入 | [targets/](python/intent/targets/)、[targets/base.py](python/intent/targets/base.py)、[runtime/](python/intent/runtime/) | Host 解析目标、绑定产物与 launch；不把设备分支加入 KIR |

DSA 当前的 construction 和部分目标实现仍在 [KIRToDSA.cpp](lib/Conversion/KIRToDSA/KIRToDSA.cpp) 中耦合。修改这条路径时应看真实代码边界，不能假定它已经拥有与 GPU/CPU 对称的独立 transformation pipeline。

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

例如，[realizeRegionFolds / realizeRegionScans](lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp) 在完成 region 改写后，自身调用内部的 [closeValueAccessRelations](lib/Dialect/GPU/Transforms/Utilities.cpp)，维护 access-result、pointwise、access-value、reshape 和 contract 关系。这两个 region 阶段在 [GPU pipeline](lib/Dialect/GPU/Transforms/Passes.cpp) 中只调度完整入口，随后验证 postcondition。调用者不需要再附加一串 repair 调用；这也不要求 CPU 使用相同的关系维护方式。

## Provider 与 runtime 扩展

先比较 Intent operation 与目标原语的合同，包括 dtype、accumulator、NaN/tie、顺序、effects 和 ABI。合同吻合时优先直接映射；例如 provider 已有 reduce/scan，就不在 Intent 再实现其线程通信与归约树。

- 同一 execution family 的新 provider，先复用已有 physical program 和 family passes。
- 新设备代际优先通过已有 capability consumers 和 legality predicates 表达；不因设备代号不同就建立一套 dialect。
- 只有共同 IR 确实缺少、且多个 consumers 或独立 verifier 需要的目标结构，才增加 local extension。
- `lib/Target/<provider>/Serialization/` 打印已经决定的当前程序，不重新选择 ownership、workspace、pipeline 或调优参数。
- `python/intent/runtime/` 绑定目标要求并执行；无法兑现的能力应明确报错，不改变算法或隐藏失败。

CPU 的 implementation registry 是明确的局部扩展点。GPU provider 通常复用下层 compiler 的 primitives 与布局机制，不需要为了目录形式对称再建一套同名 leaf 系统。

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
