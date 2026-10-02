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
  → Python frontend：类型化构造、specialization、单次 MLIR 序列化
  → intent-compile：typed KIR 验证 → KIR 规范化 → canonical KIR 验证
  → compiler driver：目标解析与 family/provider 编译
    ├─ KIRToGPU → GPU transformations → GPU provider
    ├─ KIRToCPU → CPU transformations → Mojo / Weft
    └─ KIRToDSA → DSA transformations → BANG C legalization
  → provider serialization：source + 当前 IR 的 interface 与 compilation target
  → GeneratedProgram：可保存、恢复的 source / IR / metadata
  → materialize(target=...)：检查目标事实，再绑定本机 runtime
  → CompiledArtifact.runtime → provider 编译、绑定与 launch
```

这条路径的主要入口如下：

| 环节 | 代码入口 | 负责什么 |
|---|---|---|
| Kernel/helper 定义 | [python/intent/api/](python/intent/api/) | `@intent.kernel`、`@intent.fn` 及 source definition |
| Python frontend | [frontend/compilation/compiler.py](python/intent/frontend/compilation/compiler.py)、[frontend/lowering/](python/intent/frontend/lowering/) | 类型化捕获、helper、控制与 intrinsic lowering |
| 编译调用 | [compiler/pipeline.py](python/intent/compiler/pipeline.py)、[compiler/artifact.py](python/intent/compiler/artifact.py)、[compiler/toolchain.py](python/intent/compiler/toolchain.py) | `generate` 返回独立 source/IR/metadata；`save/load` 不加载 SDK；`materialize` 绑定运行环境；`compile` 组合生成与绑定 |
| C++ 编译入口 | [Compiler.cpp](lib/Compiler/Compiler.cpp)、[Backend.h](include/Intent/Compiler/Backend.h)、[intent-compile.cpp](tools/intent-compile/intent-compile.cpp) | 库拥有请求、阶段调度与当前模块；CLI 只解析参数和读写文件；backend 声明 family 与 provider 的连接 |
| 公共产物与调用 | [runtime/artifact.py](python/intent/runtime/artifact.py)、[GPU program](python/intent/runtime/gpu/program.py)、[native ABI](python/intent/runtime/native.py) | `ArtifactRuntime.run/launch` 是显式调用接口；`prepare` 通过能力协议绑定一次调用，provider 完成自身的 JIT/tuning/launch |
| 用户与 agent 工具 | [tools/compilation.py](python/intent/tools/compilation.py)、[tools/cli.py](python/intent/tools/cli.py)、[tools/compiler_mcp.py](python/intent/tools/compiler_mcp.py) | CLI 与显式启用的 compiler MCP 共用公开编译 API 和错误阶段，不建立第二条 compiler 路径 |

已有 `.codegraph/` 时，可用 Codegraph 查询上面的具体文件或符号，再补读未覆盖部分；没有索引不必先创建索引才能工作。

### 前端构造与 KIR 边界

`lower_to_mlir` 捕获作者程序，不加载 MLIR Python bindings。前端的 [state.py](python/intent/frontend/mlir/state.py) 保存 value 的定义、operation 的 operands/results/regions，以及 block/region 的词法关系；[builder.py](python/intent/frontend/mlir/builder.py) 负责构造并分配 provenance 和逻辑维度身份。读取 owner、effects 或可见性时查询当前结构，不新增另一份 value→block 或 operation→block 表。

[serialization.py](python/intent/frontend/mlir/serialization.py) 在构造结束后一次遍历图，输出类型、属性、最终 source names 与 source locations。不要在这里匹配算法、选择目标或做 region 改写；也不要在 AST lowering 中先输出嵌套文本再重新解析来获得语义关系。

Pure helper 的隔离边界在构造 region 时就存在：`RegionState.isolated_from_above`
不移除结构 parent，但 `BlockState.visible_parent` 在该边界停止 SSA 可见性。
Shape 查询与 builder operand 校验使用相同边界；helper 的动态维度从自己的参数取得，
不能为了取得相同 dimension ID 而引用外层 kernel view。普通控制 region 保留合法祖先访问。

Lowering 的共同构造能力有明确入口：

| 需要的能力 | 模块 | 责任边界 |
|---|---|---|
| tuple/record schema、类型递归、取字段和重建 | [products.py](python/intent/frontend/lowering/products.py) | 控制合流、structured identity 和普通表达式共用；leaf 的 dtype/shape 合同仍由调用方决定 |
| helper、分支、循环的临时环境 | [scope.py](python/intent/frontend/lowering/scope.py) | 异常时也恢复词法环境和插入位置，不跨作用域保留另一份 owner 表 |
| pure structured region | [regions.py](python/intent/frontend/lowering/regions.py) | 一次完成 region 构造、作用域、effect 检查和 yield；reduce/scan 的顺序与数值资格继续由各操作检查 |
| dimension 来源、整数 shape 关系与 domain bounds | [shapes.py](python/intent/frontend/lowering/shapes.py) | 查询当前 typed graph；只 intern 确切整数表达式与维度身份，不决定物理 blocking |
| shape 类型、extent operands 与属性构造 | [shape_construction.py](python/intent/frontend/lowering/shape_construction.py) | 广播、显式 shape 与 outer 共用同一维度对象和操作数位置；调用方决定 SSA 来源，不让 serializer 补齐关系 |

职责参考是 Triton 的 `python/triton/compiler/code_generator.py:129–148`（子 region 的作用域恢复）和 `:300–322`（typed builder 与 semantic 层）。Intent 对应上表中的 scope 和 [MlirBuilder](python/intent/frontend/mlir/builder.py)，但把 MLIR 原生操作放在独立 compiler 进程中，因此 Python 安装不需要匹配 ABI 的 MLIR bindings。

Native `intent-normalize-kernel` 是 KIR 规范化的完整入口：输入结构验证、合法 region 归一、输出 canonical 验证在同一 pass 中闭合，之后才建立 canonical analyses。它同时服务 `intent-compile` 和 `intent-opt`。新的跨目标 KIR 规范化放在 [lib/Transforms/](lib/Transforms/)，需要满足相应语言合同；GPU/CPU 的物理变换继续留在各自 family。

只检查既有作者程序时，可用 `intent.compile_ir(definition)` 或 `intent compile path/to/program.py:kernel --stage kir --json`；这个阶段无需 target、后端 SDK 或设备。`--stage shared --target …` 输出共享物理 IR，默认 `provider` 阶段输出 provider IR、source 和 metadata。`intent-compile --compiler-info` 查询当前二进制实际编入的 providers；它不证明外部 provider 编译器或设备可用。

### 编译许可与优化开关

`compile`、`generate`、`compile_ir` 和 `compile_shared_gpu` 接受同一个
[`intent.CompileOptions`](python/intent/compiler/options.py)：

```python
options = intent.CompileOptions(
    numerics="source", online_reduction=True, optimization_remarks=False,
)
program = intent.generate(definition, target=target, options=options)
```

`source` 是默认数值合同，已包含语言允许的 FMA、局部融合和普通并行 reduce，
不承诺 bitwise 一致。`relaxed_normalization` 额外允许特定 normalized-summary
分段重标定及低精度 weight cast 参考变化；调用方必须保证 valid score 有限，
以及所有实际参与加权 contraction 的 value 有限，包括乘以零权重的 value。
具体许可、masked access 和空域边界见[数值规格 §5.6](doc/dsl/types-numerics-and-effects.md#56-编译调用的数值许可)。
`online_reduction=False` 禁用这一可选改写；设为 `True` 不授予额外数值权限，
也不保证采用该改写。`optimization_remarks=True` 输出编译决策诊断，不改变数值许可。
CLI 的对应参数是 `--numerics`、`--online-reduction true|false`、
`--optimization-remarks true|false`；compiler MCP 的 `compile` 使用同名 `options` 字段。

[Compiler.cpp](lib/Compiler/Compiler.cpp) 将选项绑定为模块的 typed
`intent.compile_options`；[CompileOptions.h](include/Intent/Dialect/Intent/IR/CompileOptions.h)
提供统一读取和 metadata 导出。Pass 从当前 IR 读取权限，provider serializer
导出 `metadata.compile_options`，`GeneratedProgram.compile_options` 读取并验证它。
选项进入原编译调用和缓存身份，保存、恢复与 materialize 保留原 policy。

[OnlineSummary.cpp](lib/Dialect/GPU/Transforms/OnlineSummary.cpp) 从当前
contraction、权重、max/sum 和坐标投影识别 normalized summary；record helper
通过字段适配复用这份证明。新增规则不要依赖可被合法 fold 消除的 record、extract
或同类型 cast。相同 SSA 来源也不够：reshape/broadcast 后的 reference 必须仍对应
每个归约行的保留轴。[RealizeOnlineReduction.cpp](lib/Dialect/GPU/Transforms/RealizeOnlineReduction.cpp)
再检查数值权限、重放与 dominance，并完成实际改写。
Shared IR 续编译沿用已绑定的 policy，`generate_from_ir` 不提供重选入口；
需要改变许可时，从原作者程序重新编译，不能由 runtime fast-math 覆盖。

### 编译调度与独立 IR 工具

[Compiler.h](include/Intent/Compiler/Compiler.h) 的请求和结果是原生编译入口。
[Backend.cpp](lib/Compiler/Backend.cpp) 汇集编入的 adapter；
[GPUBackends.cpp](lib/Compiler/GPUBackends.cpp)、[CPUBackends.cpp](lib/Compiler/CPUBackends.cpp)
与 [DSABackends.cpp](lib/Compiler/DSABackends.cpp) 分别连接自己的 construction、shared pipeline
和 provider。新增 provider 时，在所属 family 声明注册、pipeline 和 serialization，
不要在 CLI、错误映射和 profile reader 中各加一套分派。

Pipeline 用 `OpPassManager` 构造；pass 的名字、options 和 dependent dialects 在相邻
`Passes.td` 声明。组内需要标准 cleanup 或对子模块运行变换时，用当前 pass 的
`runPipeline`，使诊断、IR 打印和计时保留在同一次编译中。内部修复函数不因此成为
独立 pass。GPU 的分支 realization、候选绑定与 traversal 合并也属于具名的完整组，
不在 compiler driver 中追加另一条变换路径。

CPU 通过 [ImplementationProviderInterface](include/Intent/Dialect/CPU/IR/ImplementationProvider.h)
从当前 MLIR context 取得已编入 provider 的 registry。需要 implementation 的 pass
使用显式 `provider` option，不捕获进程外或调用栈上的 registry 指针。
GPU 的 resolved profiles 则存于当前模块的 typed attributes：每张表携带列声明和完整候选，
消费者使用自己唯一的 schema 校验查询。默认 JSON 和 override 只在 profile 解析阶段读取，
后续 pass 不再依赖最初调用者持有的对象。CPU 与 GPU 不共用不相同的候选组织方式。

安装产物同时提供 `intent-compile` 和 `intent-opt`。后者注册同一套 dialect、provider
接口和 pass，可使用标准 MLIR pipeline 语法：

```bash
intent optimize /path/to/current.mlir \
  --pipeline 'builtin.module(canonicalize,cse)' --json
```

Python 对应 `intent.optimize_ir(ir, pipeline=...) -> OptimizedIR`；显式 compiler MCP
也提供 `optimize`。每次请求都实际执行并保留输入、工作目录、命令、输出和诊断。
任意 pipeline 可能读取外部文件，因此优化工具不猜测其依赖并复用旧输出。
这些入口处理调用者明确提供的 IR，不导入算法模块，也不启动 kernel。

Shared IR 优化后，用 `intent.generate_from_ir(optimized.ir, input_stage="shared",
name=..., target=...)` 继续生成 `GeneratedProgram`，再按需调用 `materialize()`。
命令行对应 `intent generate-ir current.mlir --name NAME --target PROVIDER --json`；
原生入口为 `intent-compile --input-stage=shared`，继续传入原调用的目标能力参数。
续编译只验证当前 shared program 并运行 provider pipeline，不重复 construction、
family passes 或 profile 导入，也不接受新的 tuning override。GPU/CPU 的硬件能力
必须与 IR 已绑定的事实一致；实现、候选和 ABI 都读取当前 IR。
`input_stage="kir"` 则走完整编译。输入阶段由调用者明确给出；`name` 只是生成程序的
诊断标识，实际入口和候选由当前 IR 决定，不按名称筛选 kernel，也不从文本猜测。

职责参考：Triton 的 `third_party/nvidia/backend/compiler.py:273–369` 用 pass manager
构造配置明确的 lowering 阶段，`:638–647` 声明各阶段；
`python/triton/compiler/compiler.py:289–350` 顺序传递各阶段的实际产物。
Intent 由同一个编译库连接 GPU、CPU 和 DSA 的不同 IR，保留各 family 的物理决策边界。
标准工具可重放这些阶段；只有相应输入合同成立，单个 pass 才可独立使用。

### 编译目标与本机运行绑定

[targets/specification.py](python/intent/targets/specification.py) 的
`GPUCompilationTarget`、`CPUCompilationTarget`、`DSACompilationTarget` 只提供
编译所需的能力与预算。显式传入这些对象时，生成 source/shared IR 不查询设备，
也不查找 provider SDK。`TritonTarget`、`MojoTarget` 等本机入口仍可探测环境，
但解析结果中的 `compilation` 与 device、SDK、native compiler options 分别保存。

编译器从最终 typed IR 导出 `metadata.target` 和 `metadata.provider`；
`GeneratedProgram.target` 读取这份事实。GPU target 不包含设备序号；CPU 的
vector、workers 和 private-memory 预算不包含 Mojo executable 路径。
公共编译入口核对请求与结果，不从 MLIR 文本或本机设备补出缺失 metadata。

`program.save(directory)` 保存 `kernel.source`、`kernel.mlir`、`artifact.json`；
`intent.GeneratedProgram.load(directory)` 恢复同一产物，包括诊断入口名字。
新目录完整写出后才发布，已有目录不会被覆盖。加载过程不执行生成源码。
恢复的程序调用 `materialize(target=...)` 时必须显式选择本机运行环境；
生成时使用了本机 target 的程序可无参沿用其进程内绑定。
运行绑定检查编译事实一致，GPU 还重新查询所选设备，不能把另一架构的产物
默默当作当前设备生成的程序。需要改变编译目标时重新运行相应编译阶段。

Weft 的 AOT lowering、系统编译和 native artifact 仍有自己的产物合同；
公共生成产物的保存不能替代这些步骤。BANG C 的导出和加载使用同一公共生成载体，
NeuWare 编译、CNRT queue 与实际设备绑定继续属于其 runtime。

职责对照：Triton `python/triton/compiler/compiler.py:226–233` 接受显式 target，
`:413–438` 从 metadata 恢复目标并延迟 native handles，`:452–488` 才加载设备代码。
Intent 的显式目标描述对应其 `python/triton/backends/compiler.py:8–14` 的 `GPUTarget`，
同时保留 GPU、CPU、DSA 各自实际需要的构造能力。

### 公共接口与物理调用绑定

公共参数的唯一声明是 [InterfaceAttr](include/Intent/Dialect/Intent/IR/IntentAttrs.td)：
按作者 runtime 参数顺序保存 `PublicParameterAttr(name, type)`。View 直接使用 canonical
`intent::ViewType`，因此 dtype、shape identity、访问方向和约束不再复制成各 family 的
另一套 ViewArgument。`buildPublicInterface` 在 construction 时移除已完成 specialization
的 constexpr 参数；它们不占 runtime 槽位。

进入 physical construction 之前，KIR 把作者名字与 source origin 保存在对应函数参数的
`intent.parameter` 属性中；类型角色直接读取当前 signature。不维护平行的参数名、role
或 provenance 位置表。标准函数参数改写会携带自己的属性，source interface verifier
检查其完整性；module provenance 和坐标来源仍由 KIR 验证阶段闭合。

| 需要修改的事实 | Owner 与复用入口 |
|---|---|
| 公共 scalar/view 合同及 metadata | [Intent/IR/Interface](include/Intent/Dialect/Intent/IR/Interface.h)：构造、验证和 `serializePublicInterface` |
| GPU 参数身份、metadata producer 与公共参数对应关系 | [GPU/IR/ProgramInterface](include/Intent/Dialect/GPU/IR/ProgramInterface.h)：typed binding、轻量查询和 intrinsic verifier |
| 一次 GPU analysis 中的参数查询 | [GPU/Analysis/ProgramInterface](include/Intent/Dialect/GPU/Analysis/ProgramInterface.h)：读取当前 signature；改动后重建 snapshot |
| 追加 workspace/metadata 或转换 workspace 类型 | [GPU/Transforms/ProgramInterface](include/Intent/Dialect/GPU/Transforms/ProgramInterface.h)：同步维护 function type 与参数属性 |
| CPU/DSA 的外部原生参数槽 | [Serialization/NativeABI](include/Intent/Serialization/NativeABI.h)：从最终 physical entry 展开参数，供源码签名和 metadata 共用 |
| Python 公共参数与调用关系 | [runtime/interface.py](python/intent/runtime/interface.py)：唯一声明解析及 shape/stride 关系；不观察或缓存实际 tensor |
| 原生调用绑定 | [runtime/native.py](python/intent/runtime/native.py)：消费编译器导出的 native slots，生成绑定函数；observer 与设备调用留在各 runtime |

GPU 的每个物理参数携带 `ArgumentBindingAttr`。其稳定 `ArgumentRefAttr` 与当前
`BlockArgument` 位置、作者参数名、生成源码名相互独立。Public binding 指向公共参数
序号；Dimension/Stride binding 指向当前 view 的 reference 和 axis；Workspace
binding 表示 compiler-private allocation，不进入作者接口。Launch expressions
直接引用这些 bindings。Verifier 同时检查 reference、类型、逻辑 dimension 和 stride
所属 view/axis，不能只验证一个字符串存在。

Host 求值也依据这些引用。Public 参数绑定完成后，metadata 读取、deferred coverage
选择和 workspace 分配按实际依赖执行，不按函数参数槽位或声明顺序推断先后。
Intrinsic verifier 拒绝依赖环、缺失 producer 和 allocation 对尚未选择的 candidate
参数的依赖；canonical Dimension 的 owner 必须是公共 view。Coverage bound 保留
不依赖 physical parameter 的已有边界。Python 在加载接口时从相同声明计算一次执行
顺序，调用时读取本次实参；不输出另一张有独立语义的计划表。

Pass 查询 `getArgumentBinding`、`queryArgumentExpression`、`getPublicView` 等入口，
不要扫描 generated names 或把槽位再存回 view type。新的参数通过 mutation owner
进入 signature；不能只修改 entry block 而遗漏 function type。Provider 的附加参数事实
应依附参数本身或 typed reference，例如 cuTile 的索引范围保存在相应 argument attr，
不另建按旧 slot 排列的平行数组。

Metadata 的 `interface.parameters` 只保存公共合同；`gpu.arguments` 保存最终 GPU
bindings，`native.slots` 保存 CPU/DSA 实际调用参数。Native pointer 的 `element`
描述物理 pointee，scalar 的 `carrier` 描述调用载体，它们可能与公共 dtype 的 signedness
或 bool 表示不同。Runtime 不再根据公共 dtype 猜测 native carrier。Family 的
contiguous、alignment、disjoint-output 等执行要求仍由 family 声明和检查。

GPU source serializer 用 [PythonSignature](include/Intent/Dialect/GPU/Serialization/Python.h)
为已验证参数分组并分配源码名字。这个投影不解释公共语义；runtime 仅在 provider
调用边界把这些名字转换为稳定 argument IDs。Physical parameter 使用独立命名空间，
作者 scalar 名称不会覆盖 shape metadata 或 tuning 参数。

职责参考：Triton `lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp:12–25,106–143`
在 lowering 中形成额外 scratch/descriptor ABI；
`third_party/nvidia/backend/driver.py:274–298` 消费 signature 和 metadata 构造 launcher。
Intent 另外保留 logical view 的公共合同，但同样由编译结果决定目标调用结构。

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

### 运算自身提供结构关系

KIR 和共享 GPU 的 reduce、scan、region fold/scan 通过
[StructuredOpInterface](include/Intent/Interfaces/StructuredOpInterface.h)
提供 sources、identities、captures、initial states，以及各 helper 的参数和 yield。
ODS 的命名 operand/result segments 是分组的唯一来源；创建或更换一组操作数时使用
typed builder 与 `get…Mutable()`，不要另外存 count 属性、计算跨组偏移或修改分组长度副本。
Region scan 的 emitted results 与 final states 也有独立的 result segments。

构造 helper 时，用 `getCombineArgumentTypes` 等查询取得参数 schema。
Source slice 的类型由当前 KIR 或 GPU lowering 决定，再传给
`getSummarizeArgumentTypes` / `getEmitArgumentTypes`；接口不决定分段大小。
已有 helper 的 formal 参数查询返回真实的 `BlockArgument` ranges，
可以直接用于 `IRMapping`，不需要重新拼接参数顺序。

`getValueRelations()` 描述当前 SSA 上的关系，不证明两值相等。
Capture 是直接转发；SourceSlice 保留切片关系；Reduction、Prefix、Emission
分别保留归约、前缀和输出拼接语义。Accumulator 允许该 IR 层规定的
scalar/slice 到完整结果的提升，不能在 KIR 中误作类型完全相同。
同一个 identity 或 capture 可以占多个 formal 位置，查询某一组件时保留它的位置，
不能仅按 SSA value 反查组件。

GPU 的 collective 结果推导在 [Program.h](include/Intent/Dialect/GPU/IR/Program.h)：
shared/native 的 `InferTypeOpInterface`、verifier 和关系维护复用相同轴规则。
保留的 source 轴决定结果 shape/coordinate，声明的 accumulator/result 决定元素类型。
CPU 的 [RegionOpInterface](include/Intent/Dialect/CPU/IR/CPUOpInterfaces.h)
独立表达目标缓冲区、只读输入及 helper 参数角色；分区分析、验证和展开共用这份 schema，
不把这些缓冲区冒充 GPU 的 SSA fragment。

KIR 普通控制在 [ControlFlow.cpp](lib/Dialect/Intent/IR/ControlFlow.cpp)
实现标准 MLIR `RegionBranchOpInterface`：For 的 source coordinates 与 carries 分组，
While 的 condition 与 forwarded arguments 分组，MLIR 检查 region 间类型传递。
共同接口处理结构关系，各 family 继续选择自己的物理循环与存储。
显式 capture 的 helper 使用 `IsolatedFromAbove`，不可从外层暗捕获 SSA 常量或 runtime 值。

职责对照：Triton 的 `include/triton/Dialect/Triton/IR/TritonOps.td:761–818`
在 reduce/scan 运算上声明 `InferTypeOpInterface` 与 region 验证；
MLIR SCF 的 `SCFOps.td:136–148,953–958` 在循环上声明 region branch 接口。
Intent 同样让运算提供自身结构，但保留显式 identity/capture、逻辑坐标与 region segmentation 合同；
没有将这些语义交给 provider serializer 或框架名称推断。

### 分析与改写的职责

访问操作的 operand schema 也由 IR 自己提供。KIR 的
[IndexedAccessOpInterface](include/Intent/Dialect/Intent/IR/IndexedAccessOpInterface.h)
区分 source、indices、写入值、读的 validity/fill，以及 CAS 的 expected/desired。
`IndexRelation` 的 operand positions 只引用 indices 分组，不是整个 operation 的位置。
新增或修改访问时，用 ODS 的命名 operands；不要再添加 `value_operand_index` 一类旁路字段。
逻辑 buffer 的动态 extents 与 optional initializer 同样独立，shape relation 只引用 extents。

[CanonicalKernelAnalysis::indexRelation](lib/Analysis/CanonicalKernel.cpp) 将当前分组解析成
SSA values 和源轴关系，GPU、CPU、DSA construction 共用这份结果。
Slice 的 start/stop/step 保留三个位置：静态或缺省位置的 Value 为空，不能压缩后改变槽位。
这个分析不选择物理 tile、内存布局或读取实现；这些仍由各 family 完成。

GPU 的 [AccessOpInterface](include/Intent/Dialect/GPU/IR/AccessOpInterface.h) 独立描述
物理 resource、coordinates/source axes、payload、结果及 validity/fill。
Footprint、关系闭合、predication、workspace 与 provider 的访问分析读取这份合同。
`updateAccessOperands` 保留 operation 和结果 SSA 身份，维护 ODS operand segments；
修改 source axes 时须显式更新轴映射，不能在 serializer 中补修。
CAS 的两个 payload 依次是 expected、desired，谓词 shape 取 payload schema，
不能把 `{old_value, success}` record 当作单个数据值。

访问接口不是优化许可：Gather 仍是纯 SSA 读取；atomic 的 ordering、sharing 和目标能力
独立验证；普通 store 的 payload 投影也不自动适用于 atomic 或 scatter。
读取共同字段以后，消费者仍需保留自己原有的别名、effect、重放与 predication 资格。
这种边界对应 Triton `TritonOpInterfaces.td:130–173` 中分别声明 predicate 和 atomic
语义的做法；Intent 的 KIR 逻辑索引与 GPU 物理访问继续使用各自的接口。

KIR 的验证入口直接属于 operation，按合同分布在
[Access.cpp](lib/Dialect/Intent/IR/Access.cpp)、
[ValueOps.cpp](lib/Dialect/Intent/IR/ValueOps.cpp)、
[ShapeOps.cpp](lib/Dialect/Intent/IR/ShapeOps.cpp) 和
[StructuredOps.cpp](lib/Dialect/Intent/IR/StructuredOps.cpp)。修改某类运算时，在其
`Op::verify()` 及所属模块完成验证，不追加全局 operation-name 分派。
相邻私有 [TypeSchema.h](lib/Dialect/Intent/IR/TypeSchema.h) 共用元素类型、shape 与
维度关系查询；[RegionVerification.h](lib/Dialect/Intent/IR/RegionVerification.h)
共用 helper 参数、yield 与纯度检查。它们验证当前 IR，不提供 lowering policy。
这与 Triton `lib/Dialect/Triton/IR/Ops.cpp:273,647–650,701–704` 中由 Dot、Reduce、Scan
各自实现 verifier 并复用 helper 的边界一致。

分析回答当前程序满足什么条件；变换根据这些条件改变程序。不要为了共享代码，让 analysis 创建新 operation、替换 uses，或者替某个 family 决定具体循环与存储结构。

选择共享位置时，可以依次问：

1. 这是多个消费者都需要的语义关系或合法性规则吗？共享规则与查询接口。
2. 它依赖某种 physical topology、fragment representation 或 storage 吗？保留在相应 family。
3. 它只是 provider API 拼写差异吗？直接序列化已有事实，不增加 shared IR。
4. 它确实改变当前 program，并被后续分析或变换使用吗？在正确层形成显式 IR，再交给 serializer。

一个实际的共享入口是 [matchOnlineSummaryCombine](include/Intent/Analysis/OnlineSummaryCombine.h)：它只读 combine graph，识别 validity、guarded maximum、指数缩放和 weighted sum 关系，返回指向当前 SSA 的 `OnlineSummaryCombineRelations`。调用方提供投影与常量查询规则；它不创建 IR，也不选择 tile、storage 或 loops。

[Canonical matcher](lib/Analysis/OnlineSummary.cpp) 的 `matchOnlineSummary` 和 [GPU matcher](lib/Dialect/GPU/Transforms/OnlineSummary.cpp) 的 `matchOnlineSummaryMerge` 共用这个核心，同时保留各自的四字段 summary 识别、axis/shape/cast 检查、候选枚举及 GPU physical schema 检查。扩展共同 combine 规则时从共享核心开始；调整 fragment 投影时改 GPU 适配层。CPU 当前没有接入这个 matcher，不能把这项复用描述成所有 family 已共用。

[ContractionAxes.h](include/Intent/Analysis/ContractionAxes.h) 统一 contraction 的 reduction/batch 配对、free axes 与操作数轴到结果轴的位置关系。CPU construction 和 GPU 当前 IR 查询共用这份纯轴关系；GPU 的 projection、vector realization 与 provider 原生矩阵检查消费相同结果。结果位置按正式 operand axis 推导，同一 source 或 dimension 在两边出现不代表同一个结果轴。该分析不读取 SSA、不选择 packing/tile，也不替各 provider 扩大原生 rank 或 dtype 支持。

同一接口中的 `ProductContractionAxes::get` 从两个操作数到乘积公共域的投影、已证明的逻辑 unit 轴及归约轴，推导 contraction 配对、保留的原操作数轴和结果排列。CPU 的 [ContractionSources](lib/Dialect/CPU/Transforms/ContractionSources.cpp) 与 GPU 的 [ContractionSources](lib/Dialect/GPU/Transforms/ContractionSources.cpp) 共用它识别乘法后求和的轴关系。CPU 的显式 contraction 查询也使用这个核心，但不将合法的 `K=1` 配对当成广播消除。调用方分别证明数值合同、唯一数值消费者和存储或 SSA 关系；物理 tile 大小为 1 不构成逻辑 singleton 的证明。新增共同轴规则改此分析；存储快照、方向选择、blocking 与目标支持改各自消费者。

[IntegerRelations.h](include/Intent/Analysis/IntegerRelations.h) 的 `foldIntegerDifference` 复用 `UniformExpression` 描述，只读折叠加减、常数乘法及等宽整数/index cast 的变化系数。[CPU VectorizeLoops](lib/Dialect/CPU/Transforms/VectorizeLoops.cpp) 用它判断循环坐标差值，再检查连续 stride、别名与依赖；[DSA CollectiveGather](lib/Dialect/DSA/Transforms/CollectiveGather.cpp) 用它判断四个参与者之间的地址差值，保留自己的 task 商余关系、只读视图、局部 buffer 写入和控制一致性证明。CPU 的这个 vectorizer 当前由 Mojo legalization 调用，共享 CPU family 不意味着所有 provider 都调用它。

该核心只接受 i64 或调用方明确绑定为 64 位的 index；两个适配层依据 Intent 的逻辑 index 合同传入位宽，不假定任意 MLIR index 都是 64 位。值运算仍遵守模整数语义，系数的加减乘另外检查是否能用 `int64_t` 表示；失败返回 `Unknown`，不能当作系数零。窄整数回绕后的扩宽需要独立范围证明，地址有效性、memory effects 与拓扑也不由系数证明。GPU 的按位宽模运算规范化和 source-axis 关系分析有不同合同，不应仅因都有 Add/Mul 就接到这一接口。

公开的 transformation 入口必须完成自身改写所需的 relation closure，使调用方得到满足 postcondition 的 current program。中间 repair helper 不因可以被调用就成为独立 pass；pipeline 负责次序，不应成为调用者必须记忆的隐式修复配方。

例如，[realizeRegionFolds / realizeRegionScans](lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp) 在完成 region 改写后，自身调用 [closeValueRelations](lib/Dialect/GPU/Transforms/ValueRelations.cpp)，通过工作队列闭合受影响的 value/access/aggregate 关系。这两个 region 阶段在 [GPU pipeline](lib/Dialect/GPU/Transforms/Passes.cpp) 中只调度完整入口，随后验证 postcondition。调用者不需要再附加一串 repair 调用；这也不要求 CPU 使用相同的关系维护方式。

### GPU 中直接可复用的接口

[GPU Passes.h](include/Intent/Dialect/GPU/Transforms/Passes.h) 只暴露完整变换与验证入口。实现内部需要的查询与改写按下表包含具体头文件，不通过一个通用 Utilities 模块取得所有能力。

| 需要的能力 | 接口 | 使用方式 |
|---|---|---|
| 当前 value/access 的坐标、范围和复用事实 | [Analysis/PhysicalProgram.h](include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h) | 只读 current IR；相关 def-use、类型或范围改变后重算 |
| GPU 类型与形状属性自身的不变量 | [IR/TypeVerification.h](include/Intent/Dialect/GPU/IR/TypeVerification.h) | `verifyGPUTypeInvariants` 用 MLIR `AttrTypeWalker` 复用各类型/属性的 `verify`；完整 GPU verifier 在操作验证前调用，避免 release 构造绕过 checked constructor 后漏检 |
| 执行组构造、重建与 provider 展开 | [Transforms/ExecutionGroups.h](include/Intent/Dialect/GPU/Transforms/ExecutionGroups.h) | shared 变换维护真实 body 与坐标参数；`lowerExecutionGroups` 在 provider 准备入口统一展开 |
| scalar/fragment schema与投影轴 | [Analysis/ValueSchema.h](include/Intent/Dialect/GPU/Analysis/ValueSchema.h) | 只读查询当前类型与轴关系，不创建值、不选择 blocking |
| 物理整数表达式求值 | [Analysis/UniformValues.h](include/Intent/Dialect/GPU/Analysis/UniformValues.h) | `evaluatePhysicalExpression` 接受 symbolic-leaf binding；算术和溢出检查共用一份实现 |
| range/loop 中的整数比较与完整 tile 界限 | [Analysis/IndexPredicates.h](include/Intent/Dialect/GPU/Analysis/IndexPredicates.h) | `proveRangeComparison`、`queryCompleteTileLimit` 与 `queryIndexComparisonBound` 只读当前范围；区分已证明的真值、条件蕴含和未知 |
| 常量、大小关系与访问对齐 | [Analysis/IndexRelations.h](include/Intent/Dialect/GPU/Analysis/IndexRelations.h) | `IndexRelations` 共用于范围谓词、Triton descriptor 与 cuTile tile access；按 typed index 与回绕合同证明，不创建 guard 或选择原生 form |
| 参数声明与完整候选绑定检查 | [Analysis/PhysicalParameters.h](include/Intent/Dialect/GPU/Analysis/PhysicalParameters.h) | `ParameterSpace::read` 读取 kernel 声明；不依赖 SSA 读取是否存在；改变声明后重读 |
| fragment 结构资源估计 | [Analysis/Resources.h](include/Intent/Dialect/GPU/Analysis/Resources.h) | `FragmentResourceAnalysis` 缓存稳定 IR 的类型与参数使用关系；类型或 IR 改写后重建。估计不代替下层布局、寄存器分配和 occupancy |
| specialization 后才能判定的资源约束 | [Transforms/Resources.h](include/Intent/Dialect/GPU/Transforms/Resources.h) | 将 deferred reduction bounds 写成当前 IR 的断言，供 Triton/cuTile 兑现；不是 analysis 中的隐藏改写 |
| value projection、replay、validity 与显式常量 | [Transforms/ValueMaterialization.h](include/Intent/Dialect/GPU/Transforms/ValueMaterialization.h) | 传入当前 schema、source-axis 与 replay scope；由调用者决定合法的变换范围 |
| 改写后的 value/access/aggregate 关系闭合 | [Transforms/ValueRelations.h](include/Intent/Dialect/GPU/Transforms/ValueRelations.h) | 在完整 transformation 内调用，随后验证，不能让 serializer 补修 |
| coverage traversal、参数生命周期 | [Traversal.h](include/Intent/Dialect/GPU/Transforms/Traversal.h)、[PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/PhysicalParameters.h) | 分别改变当前 ranges/access 与参数引用；参数替换同时覆盖 SSA、types 和 attributes |
| predication、workspace 与 retained slice | [Predication.h](include/Intent/Dialect/GPU/Transforms/Predication.h)、[Storage.h](include/Intent/Dialect/GPU/Transforms/Storage.h) | 保持 effects、allocation ownership 与 lifetime；不由 provider 字符串猜测 |

共享 GPU 的 `ExecutionGroupOp`（[GPUOps.td](include/Intent/Dialect/GPU/IR/GPUOps.td)）
拥有实际执行 body、坐标 block arguments、runtime/launch extents、coordinate roles
及 segment 范围；mapping 和 traversal 改写维护这一个 owner。
Triton [ProgramGrid.cpp](lib/Target/Triton/Transforms/ProgramGrid.cpp) 先读取它调整网格，
再由 [Triton prepareTritonMemory](lib/Target/Triton/Transforms/Legalize.cpp)、
[cuTile prepareProgram](lib/Target/CuTile/Transforms/Legalize.cpp)、
[TileLang formNativeMemory](lib/Target/TileLang/Transforms/Legalize.cpp) 各自调用共同的
`lowerExecutionGroups`，生成纯 `DelinearizeOp` 坐标计算并展开 body。
正常编译与 shared IR 续编译使用同一入口；serializer 不保留或解释执行组。

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

### GPU 候选声明、形成与消费

当前函数只有一份 `intent_gpu.configurations`，类型为 `ConfigurationSetAttr`。
`shared` 阶段绑定共同的静态参数；provider 完成合法性筛选后，以 `complete` 表替换它。
每行给出该阶段全部必要符号的具体值，顺序是候选枚举顺序。Coverage 的运行期 extent
通过独立的 deferred 声明绑定，不写成静态候选值；没有候选时直接失败。

Kernel 的 `intent_gpu.parameters` 保存唯一有序 `ParameterAttr` 声明表，包含名字、
类型、有限 domain、角色、绑定阶段与 typed source/coverage binding。
`ParameterRefAttr` 引用它；`ParameterOp` 是无副作用的 index 读取，
其 `getDeclaration()` 查询当前 owner，不保存另一份 domain。
Fragment 类型、physical expressions、region segment 和 provider loop-stage
属性也使用引用，声明不依赖某个 SSA operation 存活。

Index 声明严格为正整数；Triton descriptor choice 引用独立的 `i1` 声明，domain 为
`{0, 1}`。Native warps/CTAs 等仅编译器消费的选项直接声明，不制造无用途的 SSA 值。
普通 MLIR verifier 检查声明唯一性及 operation attributes、result types、block-argument
types 中的引用。标准 DCE/CSE 可以删除或合并读取，无需保活声明的特殊名单。

| 需要修改的职责 | 入口 |
|---|---|
| 当前声明、完整绑定与阶段验证 | [Analysis/PhysicalParameters.h](include/Intent/Dialect/GPU/Analysis/PhysicalParameters.h) 的只读 `ParameterSpace` |
| 创建声明、按需读取、改域、替换与改名 | [Transforms/PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/PhysicalParameters.h) 的 `declareParameter`、`materializeParameter`、`updateParameter`、`replaceParameter`、`renameParameters` |
| 校验并发布候选表 | [Transforms/PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/PhysicalParameters.h) 的 `writeConfigurations` |
| 当前图的分类、关联参数及完整结果机会 | [ConfigurationAnalysis.cpp](lib/Dialect/GPU/Transforms/ConfigurationAnalysis.cpp) |
| 有限 profile 解码、family 选择与 role 投影 | [ConfigurationProfiles.cpp](lib/Dialect/GPU/Transforms/ConfigurationProfiles.cpp) |
| 候选 extent 与资源约束 | [ConfigurationConstraints.cpp](lib/Dialect/GPU/Transforms/ConfigurationConstraints.cpp) |
| 共同候选形成的完整入口 | [MaterializeConfigTuples.cpp](lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp) |

这些私有 policy facts 只在一次不变的 current program 上使用，不跨改写缓存，不拥有
第二张执行表。修改 shared/deferred 声明使候选表失效；仅修改 provider 声明时，
已有 shared 表仍有效，complete 表失效。完整变换在结束前重新形成最终表。
改名与替换通过统一 owner 同时修改 attributes、result types 和 region argument types；
不能只做 SSA RAUW，也不能把名称改写分散到 serializer。
跨类型参数引用不是 MLIR 标准 SymbolTable 的完整遍历合同，因此这里使用 kernel-owned
typed references 和显式 owner API，不把自有参数冒充通用 module symbols。

Triton 的 [ConfigurationSchema](include/Intent/Target/Triton/IR/Configuration.h)
统一查询 kernel constexpr 顺序以及 `num_warps/stages/ctas` 对应的参数符号，
不保存候选值。Serializer 机械导出最终表与符号映射；
[gpu/configurations.py](python/intent/runtime/gpu/configurations.py) 负责一次解析、deferred
绑定和已声明资源条件的求值。Triton 的真实 `Config` 从这张表投影，pruning 读取原行；
cuTile/TileLang 保留自己的 JIT 与调优入口。CPU 仍使用独立函数候选和 implementation
binding，不套用 GPU 的 block/config 表。

职责参考：Triton `python/triton/runtime/autotuner.py:140–147,276–279` 在试跑与最终调用中
消费同一 `Config.all_kwargs()`；`:328–380` 区分 kernel kwargs 与 native 编译选项。
Intent 先由 IR 验证完整绑定，再在 provider adapter 中投影这两类参数；runtime 不补默认候选。

### CPU 中直接可复用的接口

CPU 的候选绑定、存储证明与执行变换有各自的入口。[共享 pipeline](lib/Dialect/CPU/Transforms/Passes.cpp) 依次完成 source 规范化、候选形成、region 实现、供数与分块、task 形成；每个完整组包含所需规范化并验证当前 CPU program。Mojo 和 Weft 共用这些 family 机制，provider 的微程序及机器表示仍各自实现。

普通 contraction 的 construction 只形成完整 `linalg.generic` 索引映射、显式零初始化及原数值运算，不选择 dot、batch 循环或 packing。[Contractions analysis](include/Intent/Dialect/CPU/Analysis/Contractions.h) 从当前索引图和乘加 body 查询共享轴语义；[NormalizeContractions.cpp](lib/Dialect/CPU/Transforms/NormalizeContractions.cpp) 在候选选择前形成 dot、矩阵和 batch 程序，并按实际 strides 决定能否使用视图。转置或 unit 轴投影的输入快照稳定时，矩阵可直接消费派生视图；非 unit 广播保留显式计算，无法通过视图表达的轴合并仍形成显式 pack 与 lifetime。实现所需的 panel 准备继续由 implementation requirements 与 input supply 负责，不能把整块转置重新藏进 construction。

同一 source 规范化阶段先调用私有 `normalizeContractionSources`，将满足条件的 f32 乘法与零初始化普通求和组合为显式乘加 contraction，再由上述 normalizer 和 implementation registry 处理。它仅穿过纯轴投影与 unit views，用 [Storage analysis](include/Intent/Dialect/CPU/Analysis/Storage.h) 证明读取快照稳定，不跨越数值 cast 或其它计算；没有独立 free 轴的逐行点积仍交给普通归约路径。只有一侧 free 轴时，还要求能够形成连续矩阵列，否则保留原 producer-fused reduction，避免为了单个向量结果物化和打包矩阵。源识别、投影视图折叠和零初始化证明集中在相邻 `ContractionSources.cpp`，矩阵展平、batch 循环与 pack 保留在 `NormalizeContractions.cpp`。修改其中一个阶段不需要在 provider serializer 新增算子分支。

Region 展开后，模板参数变成具体 views，可以用同一存储证明再次折叠矩阵输入；私有 `foldContractionInputs` 保持已绑定 implementation、计算和配置。`ContractionRequirements::unitInnerStride` 显式保存所选实现的输入布局条件，候选选择、绑定、lookup 与该改写共用检查；例如 Mojo direct 的 RHS 必须保持单位内层 stride。条件不成立时保留原物化，不重新选实现，也不把 Region 参数假定为 noalias。

| 需要的能力 | 模块 | 使用方式 |
|---|---|---|
| 外层与局部参数 | [Configuration.h](include/Intent/Dialect/CPU/Transforms/Configuration.h) | 外层 task/block 参数与 implementation 的 local binding 分开，不通过完整 Passes.h 获取配置类型 |
| 有限 profile 数据 | [TuningProfiles.cpp](lib/Dialect/CPU/Transforms/TuningProfiles.cpp) | 读取后形成 typed rows；family 与 local 参数由 provider registry 声明，未知或缺失参数明确诊断，override 整族替换 |
| 完整候选形成 | [Configurations.cpp](lib/Dialect/CPU/Transforms/Configurations.cpp) | 从当前 computations 枚举有限 implementation portfolio；保留合法性筛选、顺序与去重，候选成为独立的完整函数 |
| 实现绑定与展开接口 | [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation.h)、[Implementation.cpp](lib/Dialect/CPU/Transforms/Implementation.cpp) | `bind` 一次提交 operation binding、函数配置与实现摘要；供数与展开消费同一个选择 |
| 存储别名、生命周期与读快照 | [Analysis/Storage.h](include/Intent/Dialect/CPU/Analysis/Storage.h) | `queryStorageAliases`、`queryStorageLifetime` 查询 views、captures、uses 与 lexical end；`areDisjointStorage`、`preservesStorage`、`isStorageReadStable` 结合当前 effects、alias analysis 与显式 ABI 证明能否重放读取，不移动 allocation 或决定 packing |
| 当前描述符的维度上界 | [Analysis/Storage.h](include/Intent/Dialect/CPU/Analysis/Storage.h) | `constantDimensionUpperBound` 通过 MLIR ValueBounds 查询闭合常数上界；未知界不作为收缩依据，不使用观察到的运行时尺寸，也不将未经溢出证明的 index 算术当作数学整数等式 |
| 供数与私有计算复用 | [ReusePreparedInputs.cpp](lib/Dialect/CPU/Transforms/ReusePreparedInputs.cpp)、[FuseIntermediateBuffers.cpp](lib/Dialect/CPU/Transforms/FuseIntermediateBuffers.cpp) | 在共同存储证明之外，分别检查坐标、effect、读取稳定性与计算可重放性，实际改写 current IR |

扩展 CPU implementation 时，`applicable` 描述它承接的计算语义，`check` 查询当前 capability 与 configuration，合法时返回 `std::nullopt`，否则返回具体拒绝原因。`candidates` 与 `bind` 共用布局、provider 条件、参数和供数检查；无合法候选时，诊断定位阻断的 computation，并列出 profile 行的实际参数与原因。`lookup` 服务于已绑定且经过变换的程序，核对实现身份、绑定参数及当前计算和输入布局，不重新选择实现或用原始配置要求检查已经缩小的微块。

绑定参数使用同一 `ImplementationParameter` 声明生成与验证：本地 profile 参数、别名、
常量和已有 shared extent 的有限派生分别声明来源，合法域也放在该声明中。
不再写一份生成字典的 callback，再在后续 pass 中假定所有键存在。
`verifyBinding` / `verifyBindings` 在继续编译当前 IR 或独立运行 CPU/provider pass 时
检查完整绑定；它们不重放依赖原始计算形状的候选筛选，也不重新选择实现。

`inputRequirements` 是只读查询，候选期与后续供数变换都可以调用。`checkInputRequirement` / `checkInputRequirements` 共享 operand、panel、alignment 与显式 widening 的证明，实际物化时依据当前 IR 重查。跨阶段只传递正式 binding，不缓存另一份供数计划。输入已经满足实现要求、无需额外准备时可以返回空需求；空需求只表示不需要外围 preparation，不说明它一定更快。

[ImplementationInputs.cpp](lib/Dialect/CPU/Transforms/ImplementationInputs.cpp) 的 group supply 将配置容量与当前 source 维度的已证明上界取小，只收缩未拆成 panel 的维度；panel 宽度、对齐、有效写入窗口及生命周期保持原合同。Mojo 的 group panel 预算检查使用同一上界查询。配置容量是分块上限，不能代替当前 IR 已有的更紧界；有效窗口宽度也不能代替实现要求的固定 panel pitch。Weft 当前不请求这类 group preparation，不因此宣称它使用了同一 packing 路径。

候选组合先为每个 contraction 找到合法且无需外围 preparation 的基准，再将每种注册实现应用于它能服务的计算，其余计算保持各自基准，按完整 bindings 去重。这样同一函数中的低精度 contraction 可以选择 widened 供数，另一个连续 f32 contraction 同时选择直接读取；不会因二者实现名不同而把后者改回默认 packing。需要准备供数的实现仍参与有限 portfolio，最终 winner 由实际调优决定，不展开每个 computation 的笛卡尔积，也不把这个基准当成布局或复用代价模型。

别名集合的 `complete=false` 表示仍有未知的内存值传播。验证器可以检查已知 uses 是否越过 lifetime end，但改变存储或重放读取的优化还必须证明其需要的完整性与 effect 条件。查询结果只服务当前图，移动、替换或删除相关 operations 后重算；不能把某次查询结果跨变换保存为另一份存储计划。

[IntegerSources.cpp](lib/Dialect/CPU/Transforms/IntegerSources.cpp) 在破坏性存储复用之前，将完整 pointwise 整数 producer 的读取替换为当前位置上的标量计算，保留位宽并证明输入快照稳定。[ContiguousAccesses.cpp](lib/Dialect/CPU/Transforms/ContiguousAccesses.cpp) 随后组合实际坐标与静态 strides：完整遍历的地址若等于同形状连续成员加固定基址，就形成标准 memref view/copy，交给既有输出转发与扫描实现。仿射证明同时检查原表达式及重排后算术的范围；未知 stride、无法证明的溢出或读写干扰保留原程序。两者是 `fuseStructuredComputations` 的相邻私有机制，不是新 scan 算法，也不让调用方手工拼装 pass 次序。

输出转发也使用这份存储查询，并保留目标的 disjoint、dominance 和 effect 检查。identity layout 与显式静态 strides 若具有相同 shape、元素类型、memory space、offset 和 strides，可通过标准 `memref.cast` 保持派生 view 的输入类型；两个未知动态 strides 不构成等价证明。这样，unit-axis 视图等正常 lowering 结构不会仅因类型拼写不同而强制保留中间结果拷贝。

Mojo 的 [Passes.cpp](lib/Target/Mojo/Transforms/Passes.cpp) 调度实现展开、私有计算融合、向量化和最终原生合法化，具体阶段在相邻 [Legalize.cpp](lib/Target/Mojo/Transforms/Legalize.cpp)。向量宽度来自已绑定 implementation；scratch 提升复用 CPU 的存储证明；算术、原子更新和浮点环境在最终 surface 验证前闭合。Weft 保留 Canonical Weft IR 的 structured 输入边界，不经过 Mojo 的 SIMD 展开。

Mojo 最终合法化完成后通过 [FinalizedCandidates.h](include/Intent/Dialect/CPU/Transforms/FinalizedCandidates.h) 删除结构完全相同的候选。比较保留完整 ABI、类型、SSA、嵌套任务、effects 和数值属性，仅忽略位置、顶层 entry 名字及已经消费完的配置/实现摘要；保留 profile 顺序中的第一个代表，serializer 和 runtime 从剩余函数形成源码与候选集合。这个入口不能用于尚未消费向量化或分块参数的程序，也不按生成源码文本或算子名字合并。Weft 已将 task 分离到另一个模块，不能只比较 host、忽略 callee 名字后套用此入口。

CPU 归约的相邻重结合与元素重排许可统一保存于 `ReductionOrderAttr`。Construction 从源操作合同建立许可，fusion 取参与计算的许可交集，partition 与 materialization 保留它；不能由末尾恰好有一个 add 推断整个计算可重排。[VectorizeLoops.cpp](lib/Dialect/CPU/Transforms/VectorizeLoops.cpp) 在访问独立且允许重排时跨块保留向量累加器，最后才做横向归约；初始 accumulator 只合入一次。私有 [VectorReductions.h](lib/Dialect/CPU/Transforms/VectorReductions.h) 为该路径和多输出归约提供同一套 tuple 横向树，调用者负责初始化、captures 与顺序合法性。Weft 直接消费自己的原生归约能力，不经过这一 SIMD 展开。

Mojo 的矩阵 `formTile` 同样把该许可传给微块，后续实现不能仅因识别到 FMA 就扩大重排许可。矩阵寄存器组织属于 Mojo 原生实现，CPU source 识别和 GPU provider 不承担该 SIMD 决策。

Weft 的私有 [Views.h](lib/Target/Weft/Transforms/Views.h) 证明标准 memref 描述符是否仅做轴置换或 unit 轴插删，并将纯 view capture 的定义链显式放回 task 内。原存储及所需标量进入 task ABI；[TaskLowering.cpp](lib/Target/Weft/Transforms/TaskLowering.cpp) 将逻辑访问反投影到原 Slice/Subview，缓存原存储顺序的 Admit 快照。矩阵消费者保留该顺序，将轴重命名为当前循环轴，直接交给按命名轴归约的 OuterContract；位置相关的普通读写则显式投影到对应逻辑顺序。不能把非连续 capture 直接标成连续，也不能只改 shape 冒充转置。当前 Weft RISC-V 不能实现一般置换 Reshape；动态轴合并、非矩形 flatten 和任意 strided reinterpretation 也不在该桥接能力内，失败明确报告，不插入隐藏 copy。

Host 已计算的 size、stride 等标量直接作为 capture，不为取得一个 shape 值将整块无数据用途的 storage 带入 task。生成完整 Weft body 后，[TaskInterface.h](lib/Target/Weft/Transforms/TaskInterface.h) 的 `finalizeTaskInterface` 统一清理可删除的无用值、收缩 kernel 参数及其属性并验证；host 调用与 scalar box 只根据该入口返回的参数位置生成。形状符号和 domain 还绑定类型中的身份，不能只按 SSA use 数删除。修改 task capture 或目标 lowering 时复用这一完整入口，不能只裁剪 kernel 签名而保留旧 host 参数或另让 serializer 修补接口。

Weft 的完整 provider pass 保留一个包含 `@host`、`@device` 子模块的当前程序。
前者是已形成调用与同步的 CPU host IR，后者是 Canonical Weft kernels；
[Program.h](include/Intent/Target/Weft/IR/Program.h) 提供共同查询与验证入口。
跨模块 task binding 使用 typed symbol references，公共参数对齐也保留于 IR。
编译 driver 不再换出并丢弃 host 模块。终端 [Serializer.cpp](lib/Target/Weft/Serialization/Serializer.cpp)
导出 host source、device source 和原有 runtime metadata；
[TaskABI.cpp](lib/Target/Weft/Serialization/TaskABI.cpp) 只读最终 kernel 的参数、encoding、
access、alias 和 shape symbols，生成 native ABI，不保存一份需与 IR 同步的 JSON 配方。

这个容器边界对应 MLIR `GPUOps.td:607–638,1390–1407` 中独立 kernel module 与
跨模块 symbol reference 的职责：保存可单独编译的程序及其真实调用连接。
Intent 的 Weft 路径继续使用 CPU tasks、普通 host 调用和原有同步合同，不采用 GPU launch/grid。

修改 Weft host 的 task 派发、capture 装箱或跨模块连接时，进入
[Legalize.cpp](lib/Target/Weft/Transforms/Legalize.cpp)；修改 task 内部的原生操作转换时，
进入相邻 `TaskLowering`。后者的只读分析和转换状态以单个 CPU function 为生命周期，
不由 serializer 重建，也不跨程序保留。

## Provider 与 runtime 扩展

先比较 Intent operation 与目标原语的合同，包括 dtype、accumulator、NaN/tie、顺序、effects 和 ABI。合同吻合时优先直接映射；例如 provider 已有 reduce/scan，就不在 Intent 再实现其线程通信与归约树。

- 同一 execution family 的新 provider，先复用已有 physical program 和 family passes。
- 新设备代际优先通过已有 capability consumers 和 legality predicates 表达；不因设备代号不同就建立一套 dialect。
- 只有共同 IR 确实缺少、且多个 consumers 或独立 verifier 需要的目标结构，才增加 local extension。
- `lib/Target/<provider>/Serialization/` 打印已经决定的 kernel 程序和接口 facts，不重新选择 ownership、workspace、pipeline 或调优参数，也不生成另一套通用 host 参数检查和输出分配代码。
- `python/intent/runtime/` 消费这些 facts 并执行；provider 适配实际原生调用，公共 binder 处理参数、输出与工作区。无法兑现的能力明确报错，不改变算法或隐藏失败。

CPU 的 implementation registry 是明确的局部扩展点。GPU provider 通常复用下层 compiler 的 primitives 与布局机制，不需要为了目录形式对称再建一套同名 leaf 系统。

### GPU 接口与 Python 产品边界

共同的 [Serialization/Interface.h](include/Intent/Dialect/GPU/Serialization/Interface.h) / [Interface.cpp](lib/Dialect/GPU/Serialization/Interface.cpp) 从最终 GPU IR 读取 ABI 顺序、view/scalar、shape/stride、工作区、overlap、grid、候选及资源约束。Provider 只补充自己的原生参数顺序、descriptor/array 形式和编译选项。Metadata 是当前程序的序列化结果，不是 runtime 再选物理结构的计划；新增执行事实仍应先在 IR 与相应变换中成立。

| 修改目标 | 入口 | 应保持的边界 |
|---|---|---|
| GPU 公共参数绑定、输出与 workspace | [gpu/interface.py](python/intent/runtime/gpu/interface.py) | `GPUInterface` 解析一次声明，每次 `bind` 读取真实实参并检查 dtype/shape/stride/alias；不缓存可变 tensor facts |
| 已导出的整数表达式 | [gpu/expressions.py](python/intent/runtime/gpu/expressions.py) | 只求值 compiler 已声明的表达式，不按算法名或观察到的 shape 发明策略 |
| 候选、deferred coverage 与资源条件 | [gpu/configurations.py](python/intent/runtime/gpu/configurations.py) | 唯一解析已导出的候选表；provider 明确选择用于执行或展示的现有行，不再重建第二份配置 |
| 调用生命周期与原生结果 | [gpu/program.py](python/intent/runtime/gpu/program.py) | `GPUProgram` 共用 run/launch/prepare；`PreparedCall` 属于已绑定的实参和 workspace，改变参数或元数据时重新 prepare |
| Provider 的 JIT、调优和发射 | [runtime/triton.py](python/intent/runtime/triton.py)、[runtime/cutile.py](python/intent/runtime/cutile.py)、[runtime/tilelang.py](python/intent/runtime/tilelang.py) | 消费 `BoundInvocation`，返回 `LaunchResult`；保留各下层 compiler/tuner 的职责，复用公共 trial-state 规则 |
| PyTorch operator 注册 | [runtime/torch.py](python/intent/runtime/torch.py) | `as_torch_op` 注册 opaque 调用；fake 只消费相同接口。当前限 GPU 只读 In/scalar 和 fresh Out，拒绝 InOut，backward 由作者注册 |
| 安装与依赖说明 | [tools/backends.py](python/intent/tools/backends.py)、[environment/install.py](environment/install.py) | 新安装路线声明实际依赖和外部工具链要求；不把实验私有环境或 baseline 包当作公共 runtime 依赖 |

新增 GPU provider 时，先让 legalization 交付可独立验证的当前程序，再导出共同 interface 与必要 provider facts，实现上述 provider 调用接口，并由 `ResolvedTarget.materialize` 接入。不要复制 serializer 中的 Python host 模板，也不要让 framework adapter 自己猜输出或解析生成源码。CPU、DSA 可以保留自己的 ABI/buffer 类型；共同 `ArtifactRuntime` 协议不要求它们采用 GPU 的 grid、workspace 或 tensor binder。

GPU 的纯编译事实位于 [targets/specification.py](python/intent/targets/specification.py)，设备观察位于 [gpu/device.py](python/intent/targets/gpu/device.py)，本机绑定与 materialization 共用 [gpu/target.py](python/intent/targets/gpu/target.py)。三个公开本机 Target 只声明 provider；`ResolvedGPUTarget` 分别保存 compilation 与 device，provider 表只连接实际 materializer。显式 compilation target 不经过设备观察或 SDK import。

实验适配若需要准备候选、观察调优或绑定调用，使用 `artifact.runtime` 的明确对象与 provider 扩展点。Compiler 生成的 source 不再承担 host `launch/run` 协议；只有显式作者提供的 Python source 由 [runtime/source.py](python/intent/runtime/source.py) 的独立 source loader 承接其已有 host callable。不要通过生成模块的私有字典改写编译器产物的执行语义。

Mojo、Weft、BANG C 的共同 native ABI 绑定在 [runtime/native.py](python/intent/runtime/native.py)。`NativeABI.read` 消费公共 `PublicInterface` 和编译器导出的 `native.slots`，为隐式分配与显式传入输出生成绑定函数；原生参数顺序、scalar carrier 和 pointer pointee 不在 Python 中重新推导。每次调用重新观察实参；allocator 返回新输出和本次 `ViewFacts`。CPU 的别名拒绝规则位于 [cpu.py](python/intent/runtime/cpu.py)，BANG C 的设备、队列、tile 资格与分配规则留在自己的 `program.py`；Mojo 的 Torch 规则、Weft 的 Buffer/alignment 规则和各自 tuning key 也留在 provider。Weft 的 `_ExecutionContract` 每次执行继续检查当前线程的 affinity、stack、RVV 状态和 VLEN。不要把一次实参观察或线程状态存入不可变 ABI schema，也不要在绑定函数中重建算法或 task 调度。

公共 `artifact.prepare(*inputs, outputs=...)` 通过 `PreparedRuntime.prepare_call` 接入这些能力，返回 `intent.PreparedCall`，仅共同保证 `launch()` 和 `result()`。GPU launch 使用当前 stream，CPU launch 等待本次任务，BANG C launch 同步自己的 CNRT queue；`result()` 只返回输出容器，不隐含同步。CPU 既有返回值包含 InOut，GPU/BANG C 仅返回 Out，这个差异没有被 ABI 复用改写。Provider 特有的 enqueue、benchmark 或调优观察仍属于其具体调用对象。

原生编译与产物寿命由 provider runtime 负责。Mojo 的 [compilation.py](python/intent/runtime/mojo/compilation.py) 组织候选和加载，[toolchain.py](python/intent/runtime/mojo/toolchain.py) 查询已支持工具链的依赖身份，公共 [compiler/cache.py](python/intent/compiler/cache.py) 提供输入比较、锁与发布机制。新增 SDK import 时同步 serializer 的 `native_dependencies`；无法证明依赖闭合时继续原编译并说明缓存不可复用原因。失败不发布成功产物，已加载的库不原位覆写；这些机制不改变算法、候选或算子计时范围。

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

这些私有 facts 和待提交 replacements 只服务一次变换；阶段之间传递当前 IR 与其携带的 resolved profiles，不保留另一份执行计划。Triton 的局部候选在 native-forms 阶段内闭合为 IR configs；cuTile 提交替换后才进入后续循环与配置变换。

对齐推断中的参数域必须是当前证明可依赖的域。`ResidentWorkers` 会由 provider 配置重绑定，公共关系查询不把它的临时候选当作常量或整除事实；coverage capacity 也不等于 logical extent。分支内额外对齐条件由调用方提供局部叶证明，不能传播成其它分支的全局性质。新增整数规则先核对位宽、回绕与除法合同，再接入共同查询，避免在各 provider 重写递归证明。

Triton/cuTile 的 `Transforms/Configurations.cpp` 负责各自的候选策略与资源合法性，使用共同的参数绑定分析。Triton 的 tensor/descriptor/collective 约束从当前 IR 一次收集后逐候选求值；cuTile 保留 launch 与 memory hints 的相关候选及 resident-capacity 绑定。新增设备约束时在对应模块处理，不复制参数解析器，也不把 Triton TTGIR 的布局、MMA 或 pipeline 再实现一遍。

cuTile 的 [Analysis/Tuning.h](include/Intent/Target/CuTile/Analysis/Tuning.h) 从最终 provider IR 查询哪些 runtime scalar 必须按值区分调优结果。证明覆盖 SSA、类型/属性中的 ScalarABI 以及潜在的写后读依赖；索引、控制、形状、资源和未知用途保持区分，只有完整证明为数据用途时才移除其值。Serializer 消费这份只读结果，并保留 view、overlap、完整覆盖和 array-view eligibility 的实际事实；它不改变 scalar 的原生传参或候选执行。

资源查询的 `Unknown` 表示当前求值无法证明，可能来自未绑定维度，也可能来自表达式求值失败；不能据此宣称候选合法或已精确证明资源不足。Shared 候选策略只按可得事实筛选和绑定，保留需要 specialization 或下层 compiler 判断的约束；局部候选 matcher 也不等同于完整 coverage 证明。

BANG C 的 [Storage.cpp](lib/Target/BangC/Transforms/Storage.cpp) 分开只读 `measureStorage` 和最终 `bindStorage`。前者可供局部复用与供数变换比较资源需求，后者才写入目标偏移；公共 alias/lifetime 查询在 [DSA Analysis](include/Intent/Dialect/DSA/Analysis/PhysicalProgram.h)。新增目标实现需要的 workspace 在目标变换中形成显式 operand，最终由目标 verifier 检查，不能在资源查询或 serializer 中补写。

DSA 的 `isSumOfIntegerProducts` 使用 MLIR 的整数表达式规范化证明地址关系，允许单位 stride 消除、常数结合和交换后的等价索引；供数 matcher 不应依赖某一种 Add/Mul 树形。`BangCTarget.shapes/strides` 是编译变体的 ABI 约束，运行时会核对实参。已有 [MLU 调用适配](experiments/mlu/providers/bangc/common.py) 从实际 tensor 绑定这些事实，并原样传输其 stride/offset；尚未分配的输出不猜布局。不要用忽略已知布局来规避 matcher 缺陷，也不要从 shape 猜连续布局。

## 构建、定位和完成改动

使用 [安装说明](environment/README.md) 中已配置的工具链。C++ 构建目录放在仓库外，例如：

```bash
cmake --build /path/to/intent-build --target intent-compile intent-opt --parallel 4
INTENT_COMPILER=/path/to/intent-build/tools/intent-compile/intent-compile \
  python examples/softmax.py
```

`INTENT_COMPILER` 只切换 C++ 可执行文件。若修改 `python/intent/`，按安装说明重新安装当前 checkout，并核对验证解释器的 `intent.__file__`，使运行使用本次修改的代码。

这个调用示例用于理解公开入口；修改具体能力时，选择实际受影响的既有生产程序及其所属实验组入口，保留原输入、容差和完整 callable 计时合同。

普通用户可先运行 `intent doctor --target triton --json` 检查所选环境，再用 `intent compile path/to/program.py:kernel --target triton --json` 取得真实阶段与编译产物。`--materialize` 调用同一个 `GeneratedProgram.materialize()`，不会重新 lowering。工具不主动 launch 所选 kernel，但加载模块仍执行其顶层 Python；依赖检查、编译成功与运行正确必须分开说明。

定位失败时看 `CompilationStageError.stage`、`cache_directory`、`artifact.source` 和 `artifact.mlir`。区分 frontend、physical program、provider source、下层编译及实际 launch，不把所有失败归成“后端不支持”。CLI/MCP 的 [compilation.py](python/intent/tools/compilation.py) 只转换现有诊断与路径，不维护另一份错误知识库或 compiler policy。

作者位置沿 [SourceUnit.location](python/intent/frontend/source/unit.py)、[KIR 序列化](python/intent/frontend/mlir/serialization.py) 和 [compiler IR 输出](tools/intent-compile/intent-compile.cpp) 保存在标准 MLIR location 中。缓存的 `input.mlir`、`kernel.mlir` 与 operation 诊断使用这条位置链；新增 rewrite 创建或克隆 operation 时保留相应 source location，不用旁表替代。编译日志位于同一 `cache_directory` 的 `compiler.log`。

Mojo 的 [native compilation](python/intent/runtime/mojo/compilation.py) 失败会指出具体 candidate、native 阶段和保留目录，目录中包含 bindings、命令、编译器 stdout/stderr 和阶段耗时，`request.json` 指向实际 source 与 FP object。`NativeLibrary.directory/cache_hit/cache_reason` 提供本次加载的产物及复用状态；既有 CPU runner 在准备阶段记录物化耗时和命中数量，二者不计为算子执行时间。Weft 的 [Canonical IR serializer](lib/Target/Weft/Serialization/Serializer.cpp) 同样保留标准 location，使下层编译诊断可以追到作者源码。

### 查看完整 transformation group 的 IR 与编译时间

GPU/CPU shared、DSA、BANG C 及 Triton/cuTile/Mojo provider 的完整 groups 是具名、注册的 MLIR module passes；标准 PassManager 负责调度、IR 打印和计时。Shared group 完成自己的 rewrite 与 relation closure，再检查 family postcondition；GPU provider native forms 出现后检查结构和 operation 合同，finalize 才完成完整 surface verifier，不能再用拒绝 provider dialect 的 shared GPU verifier。Mojo 中间组检查 CPU program，最终组还要求 structured computations 全部 materialize 且符合原生 surface。内部 repair helper 不注册成可独立运行的 pass。公共 [PassManager.h](include/Intent/Transforms/PassManager.h) 只应用 MLIR 标准 instrumentation 选项；它不选择优化或管理另一份 program。

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

去掉 `--mlir-print-ir-tree-dir` 会把快照写到 stderr；`--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` 可查看已接入 instrumentation 的所有阶段。若增加 `--mlir-print-ir-module-scope`，同时传 `--mlir-disable-threading`。`--mlir-timing` 是编译 pass 时间，不能当作 kernel 执行时间。

CPU 同样复用已有生产 input 与原 target/capability/profile 参数。共享阶段名为 `intent-cpu-normalize-source`、`intent-cpu-materialize-configurations`、`intent-cpu-realize-regions`、`intent-cpu-form-input-supply`、`intent-cpu-form-tasks`；例如对 `intent-cpu-materialize-configurations` 打印前后 IR，可看到单个未绑定函数变为具有完整 binding 的候选函数。Mojo 阶段名为 `intent-mojo-materialize-program`、`intent-mojo-fuse-private-computations`、`intent-mojo-vectorize-program`、`intent-mojo-finalize-program`。独立使用时须传入所需 pass options，并由相同 dialect registration 安装 context 内的 provider interface；前置 IR 合同仍需成立。

查看 provider 阶段时，沿用同一输入、目标 options 和 tuning profile 的完整编译命令，去掉 `--stop-after-shared`，同时指定 `--ir-output` 与 `--source-output`。例如既有 [cuTile official_fmha](experiments/gpu/providers/cutile/attention.py) 编译 [flash_gqa_attention_fwd](examples/kernels/streaming/attention.py) 时，在原命令追加 `--mlir-print-ir-before=intent-cutile-native-program --mlir-print-ir-after=intent-cutile-native-program --mlir-print-debuginfo`，即可对照原生 form 形成前后的 IR 并显示作者位置；对应 Triton 阶段名是 `intent-triton-native-forms`。最终合法化分别看 `intent-cutile-finalize-program` 与 `intent-triton-finalize-program`。这些阶段名用于同一完整 pipeline 的诊断，不表示可以跳过其输入依赖和 tuning profiles 单独调用。

提交一个连贯变换前，说明它读取的 facts、合法条件、实际改写的 IR、保持的语义、失效或重算的分析，以及在哪个边界验证 postcondition。用必要的既有生产运行确认影响；不另建测试目录、平行结果表或额外评测矩阵。

实验运行与数据继续归入对应的 `experiments/{gpu,cpu,mlu,agent_tritonbench}/`；`examples/kernels/` 不承载 benchmark runner。报告结果时分别说明生成成功、编译成功、运行正确与已测性能，不用 pass 数量或文件拆分数量代表能力。

## Agent 的职责边界

编写 Intent 算法时，使用 README 中的 [manual MCP](README.md#use-with-an-agent) 查询公开声明、语言合同和必要最小片段；实现入口是 [manual.py](python/intent/tools/manual.py)。完整算法示例在 `examples/`，manual 不提供题解或执行结论。

需要编译用户明确提供的程序时，显式启用独立 [compiler_mcp.py](python/intent/tools/compiler_mcp.py)。它要求已有程序路径，并复用 CLI 的同一实现；不会把完整算法加入 manual corpus，也不把动态编译诊断当作数值或性能通过。固定 agent 实验的一次正式提交规则继续由实验入口执行，日常产品工具不复用它的成绩协议。

修改编译器时，使用本页的模块导航、正式规格和当前源码；公开语言手册不承担 compiler implementation guide。合法性或职责不清楚时，先查原合同与真实消费者，再选择改动层次。
