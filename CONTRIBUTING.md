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

Structured intrinsic 的分派在 [structured.py](python/intent/frontend/lowering/intrinsics/structured.py)，具体构造按语义族组织：

| 私有模块 | 修改入口与职责 |
|---|---|
| [callbacks.py](python/intent/frontend/lowering/intrinsics/callbacks.py) | pure helper/combine 的参数、显式 capture、identity 与 product 重建；collectives、region 操作和内存归约直接复用 |
| [collectives.py](python/intent/frontend/lowering/intrinsics/collectives.py) | reduce、argmax、scan、histogram 的类型检查与 KIR 构造 |
| [region_operations.py](python/intent/frontend/lowering/intrinsics/region_operations.py) | region fold/scan 的 slice schema、summary 和结果 extent |
| [contractions.py](python/intent/frontend/lowering/intrinsics/contractions.py) | dense/scaled/sparse contraction 的轴配对、格式与 KIR 构造；matrix shorthand 直接复用同一 emitter |

新增语义规则放入对应族；共用的 callback 构造放 `callbacks.py`，而非通过 dispatcher 导入 helper。这些模块都消费已有 lowering context，不另存类型、作用域或编译配置。

职责参考是 Triton 的 `python/triton/compiler/code_generator.py:129–148`（子 region 的作用域恢复）和 `:300–322`（typed builder 与 semantic 层）。Intent 对应上表中的 scope 和 [MlirBuilder](python/intent/frontend/mlir/builder.py)，但把 MLIR 原生操作放在独立 compiler 进程中，因此 Python 安装不需要匹配 ABI 的 MLIR bindings。

Native `intent-normalize-kernel` 是 KIR 规范化的完整入口：输入结构验证、合法 region 归一、输出 canonical 验证在同一 pass 中闭合，之后才建立 canonical analyses。它同时服务 `intent-compile` 和 `intent-opt`。新的跨目标 KIR 规范化放在 [lib/Transforms/](lib/Transforms/)，需要满足相应语言合同；GPU/CPU 的物理变换继续留在各自 family。

只检查既有作者程序时，可用 `intent.compile_ir(definition)` 或 `intent compile path/to/program.py:kernel --stage kir --json`；这个阶段无需 target、后端 SDK 或设备。`--stage shared --target …` 输出共享物理 IR，默认 `provider` 阶段输出 provider IR、source 和 metadata。`intent-compile --compiler-info` 的 provider catalog 描述每个后端的 family、实际编入状态和编译所读的 profile 路径；它不证明外部 provider 编译器或设备可用。资源由 [Backend::profilePaths](lib/Compiler/Backend.cpp) 按真实 pipeline 声明，doctor 与缓存直接消费，不再分别推导文件名。CLI 拒绝当前 family 或阶段不消费的显式参数；shared IR 续编译核对已有目标事实，不重新解释构造参数。

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

[OnlineSummary.cpp](lib/Dialect/GPU/Transforms/Reduction/OnlineSummary.cpp) 从当前
contraction、权重、max/sum 和坐标投影识别 normalized summary；record helper
通过字段适配复用这份证明。新增规则不要依赖可被合法 fold 消除的 record、extract
或同类型 cast。相同 SSA 来源也不够：reshape/broadcast 后的 reference 必须仍对应
每个归约行的保留轴。[RealizeOnlineReduction.cpp](lib/Dialect/GPU/Transforms/Reduction/RealizeOnlineReduction.cpp)
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
创建与加载都经过 [ProgramContract](python/intent/runtime/contract.py)：先解析公共
target、数值策略和 GPU/Native ABI，再调用对应 provider 目录的 `contract.py`
解析其执行事实。Triton descriptor 参数、cuTile 编译配置、Mojo 候选源码区间、
Weft host/task 绑定及 BANG C extent 绑定由其实际消费者拥有，无需 SDK 即可读取。
Runtime 接收这一份已解析合同，不再各自从原始 metadata 重建参数和候选。
每个 [targets/](python/intent/targets/) 模块的 `PROVIDER` 连接自己的 family、目标类、
产物 reader、运行绑定与环境检查；公共 `ProgramContract`、目标构造和 doctor 消费
这一个 adapter，不各自维护后端分派表。安装所需的纯标准库声明仍只在
[tools/backends.py](python/intent/tools/backends.py)，供安装脚本与已安装 CLI 共用。
Adapter 导入不加载 SDK，实际 binding/probe 才导入其依赖。
这个职责对应本地 Triton `python/triton/backends/__init__.py:32–62` 的 backend/driver
连接；Intent 目前显式注册内置 provider，不因此声称已经支持外部插件发现。
`program.metadata` 返回用于检查的副本，修改它不改变产物或已经绑定的运行时。
不兼容的字段应明确报错；重新生成需要调用方保留原 Intent definition 或 KIR，
保存的 provider source 不承担恢复原算法或跨目标重新编译的职责。
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
| GPU allocation 转为 workspace ABI | [GPU/Transforms/Storage/Workspace](include/Intent/Dialect/GPU/Transforms/Storage/Workspace.h)：唯一终端 lowering；通过 [ProgramInterface](include/Intent/Dialect/GPU/Transforms/Mapping/ProgramInterface.h) 同步维护 function type、参数属性与 stride slots |
| CPU/DSA 的外部原生参数槽 | [Serialization/NativeABI](include/Intent/Serialization/NativeABI.h)：从最终 physical entry 展开参数，供源码签名和 metadata 共用 |
| Python 公共参数与调用关系 | [runtime/interface.py](python/intent/runtime/interface.py)：唯一声明解析、shape/stride 关系及 `BindingRelations` 的自动输出构造资格；不观察或缓存实际 tensor |
| 公共实参绑定 | [runtime/invocation.py](python/intent/runtime/invocation.py)：按同一声明生成 allocating/explicit binders，统一 shape/stride、Out 分配与作者 alias 检查 |
| 原生调用绑定 | [runtime/native.py](python/intent/runtime/native.py)：消费公共绑定结果与 compiler 导出的 native slots，不重推参数顺序、返回值或入口资格 |

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
最终 Workspace binding 只绑定完整的 `ViewType`；变换中的 `BufferOp` 不提前成为
函数参数。合法 Deferred coverage 已由 host 确定，可参与 allocation shape，不能用
它的 coverage bound 代替实际选出的容量，也不能把未选择的 Shared/Provider 参数交给分配器。

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
| 源码翻译 | [Source.h](include/Intent/Serialization/Source.h)、[GPU PythonEmitter](include/Intent/Dialect/GPU/Serialization/PythonEmitter.h)、各目标 `Serialization/` | 共用 SSA 作用域和 typed operation 注册；目标模块只拼写自身语言与原生能力 |
| CPU micro-kernel | [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h)、[Mojo implementations](lib/Target/Mojo/Transforms/Implementations.cpp)、[Weft implementations](lib/Target/Weft/Transforms/Implementations.cpp) | 按当前 operation、dtype 和 capability 选择局部实现；外层分块/供数仍由 CPU passes 负责 |
| 设备与运行时接入 | [targets/](python/intent/targets/)、[targets/base.py](python/intent/targets/base.py)、[runtime/](python/intent/runtime/) | Host 解析目标、绑定产物与 launch；不把设备分支加入 KIR |

DSA 的 [MatrixSupply.cpp](lib/Dialect/DSA/Transforms/MatrixSupply.cpp) 消费已有 LoadTile、MatMul、Store 和循环关系，形成协作或常驻供数；[CollectiveGather.cpp](lib/Dialect/DSA/Transforms/CollectiveGather.cpp) 从当前 offsets 写入、task 坐标、只读视图与 stride 关系证明相邻参与者可以共享 gather 供数。两个变换都消费完整的普通 DSA program，不通过 construction 候选侧表选择路径。[BANG C driver](lib/Target/BangC/Transforms/Legalize.cpp) 再依次完成原生计算、workspace、实现选择、局部组合、同步和最终存储绑定。当前 DSA 的矩阵与 group 合同、BANG C 实现仍以 MLU370 为已实现边界，目录分层不代表已经支持其他 DSA 设备。

### DSA 的单一构造入口与局部实现

[KIRToDSA.cpp](lib/Conversion/KIRToDSA/KIRToDSA.cpp) 只建立目标模块并调用一次
`Construction`。普通矩阵、矩阵后的逐元素计算、region 中的矩阵与一维运算都由
同一组操作实现构造；没有整函数 matrix 模式或另一套 eager tensor lowering。
新能力按下面的职责进入相邻私有模块，不另建算子入口：

| 模块 | 负责的事实与构造 |
|---|---|
| [Construction.cpp](lib/Conversion/KIRToDSA/Construction.cpp) | 公共参数到原生入口、操作分派、作者控制流与 carry 的生命周期 |
| [Values.cpp](lib/Conversion/KIRToDSA/Values.cpp) | SSA 物化、标量数值运算、product 字段绑定与控制状态槽 |
| [Tensors.cpp](lib/Conversion/KIRToDSA/Tensors.cpp) | 局部 shape、storage、已选窗口的投影及逐元素操作 |
| [Access.cpp](lib/Conversion/KIRToDSA/Access.cpp) | 消费公共 IndexRelation，形成当前窗口的地址、validity 和 load/store |
| [Contractions.cpp](lib/Conversion/KIRToDSA/Contractions.cpp) | 消费公共 ContractionAxes，形成局部矩阵、K 遍历和独立 accumulator |
| [Collectives.cpp](lib/Conversion/KIRToDSA/Collectives.cpp)、[CollectiveConstruction.cpp](lib/Conversion/KIRToDSA/CollectiveConstruction.cpp) | 前者形成 reduce/scan 的 source staging 与局部窗口；后者把 sources、initials、captures、active extents 和完整 helper 写入当前 DSA collective |
| [Regions.cpp](lib/Conversion/KIRToDSA/Regions.cpp) | region fold/scan 的 source 遍历、summary 和消费者投影 |
| [Worksets.cpp](lib/Conversion/KIRToDSA/Worksets.cpp) | 输出域与 task 分配、def-use 切片传播、独立性证明和有序行分段 |

[Construction.h](lib/Conversion/KIRToDSA/Construction.h) 是这一次 construction 的
私有状态声明。`LocalAxis` 分开源 extent、当前 begin/count 和物理 capacity；
`valueSlices` 保存已证明的 value/axis 窗口，未决定的 K 轴不必提前拥有完整局部容量。
真正物化张量时才要求完整 `LocalShape`。这些映射不交给后续 pass 保存或重放；
construction 的输出必须是含实际循环、存储和访问的完整 DSA program。

普通 collective 使用 `dsa.slice_reduce` 与 `dsa.scan` 保存完整 combine region。
成员轴、实际有效长度、输入、初值、捕获和输出都是当前操作的属性或 SSA operands；
helper 统一接收两组逐字段参数及显式 captures，通过 `dsa.collective_yield` 返回
各字段结果。只读输入可以直接返回，局部 scratch 也可以产生结果；realization 先将
全部结果写入独立 next 状态，再同时提交，保持多字段之间的数据依赖。

Local allocation 继续使用原二维 NRAM 存储。Collective 的标准 `memref.reinterpret_cast`
view 恢复逐轴 capacity，实际 active counts 另以 SSA 传递，不能把 padding capacity
当成源 `.shape`。完整 view 不改变元素顺序、offset 或 owner；它与 allocation 的
关系由公共 [Views.h](include/Intent/Dialect/DSA/IR/Views.h) 从当前 IR 查询，构造器与
pass 共用该 builder 和完整存储证明。Whole-slice helper 的不同字段可以有各自的 rank 和保留
形状；只有逐元素提升 scalar helper 时，才需要证明各字段共享同一个 free-axis 域。

[DSA collective transforms](lib/Dialect/DSA/Transforms/Collective/) 消费这些操作，
选择已有原生归约、可提升的 scalar combine 或完整 slice 实现；普通 reduce 保留
多 member axes，scan 保留方向、inclusive/exclusive、逐 prefix 输出与 final state。
构造器只形成实际 source staging 与 bounded collective，不再另行解释一套受限的
product combine。后续 pass 不访问 KIR 或 construction 的 `localShapes`。

新增 collective 优化从公共 [Collectives.h](include/Intent/Dialect/DSA/Transforms/Collective/Collectives.h)
及该目录进入；语义与借用检查归 [IR/Collectives.cpp](lib/Dialect/DSA/IR/Collectives.cpp)。
原低层 `dsa.reduce` 仍表示已经选定的本地原语，BANG C 继续处理其目标 workspace、
同步和源码拼写。高层 helper 必须在共享 collective pass 中闭合，serializer 不解释
combine 或补齐遍历。

`intent-dsa-realize-collectives` 是可以单独调度的 MLIR pass。它接收完整的 collective
IR，输出实际循环、局部存储和计算操作；后续供数、资源与 BANG C passes 使用
`verifyRealizedProgram` 检查这个边界。IR 保存后重新读取仍足以完成 lowering，
不依赖 construction 对象或额外的 stage 属性。

公共入口的 full-extent 义务只引用真实参数维度。静态或 compile-call shape binding
已证明的内部域在构造阶段兑现；不能把展平域的内部 identity 导出给调用方，或把
一个乘积范围要求拆成较弱的逐因子条件。需要完整局部行而尚未证明容量的派生域，
当前要求已有 shape binding 提供上界。

职责对照：本地 Triton `include/triton/Dialect/Triton/IR/TritonOps.td:761–802`
将 reduce/scan 的 combine 保存为 region；
`lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h:26–105` 共享真实 region 的参数
绑定、克隆与内联。Intent 的 DSA 也让 pass 消费完整 combine，但还需承载显式
local-memory view、active counts 和 whole-slice 状态，因此保留借用验证与同步
状态提交；不照搬 Triton 的同形标量参数限制或 GPU layout 实现。

`localMatMul` 只处理当前 typed contraction。共享左操作数时，还需证明实际输出
窗口一致、accumulator 独立、右操作数可重放，且中间没有冲突 effects。
输出 M/N task 遍历和 K 分段复用普通 workset 切片机制；epilogue 继续走普通操作分派。
对带有序控制的一维程序，只有逐元素独立、控制不依赖被切片数据且外部访问满足
同坐标关系时才分段；保留每个元素的原 carry 顺序，也保留空行中的标量控制。

形成协作供数、改变 lifetime 或选择局部资源策略时，修改 DSA transforms；
修改 MLU 指令和 micro-kernel 时，进入 [BANG C Matrix](lib/Target/BangC/Transforms/Matrix.cpp)。
`MatrixSupply` 从当前 Alloca/Fill/LoadTile/MatMul 的作用域、完整覆盖和 users 证明
供数资格，不要求输入 allocation 恰好出现在某一层循环。
Serializer 只输出已决定的结构，不能补 task 遍历或按算法身份选择实现。

职责对照：TileLang `src/transform/lower_tile_op.cc:1073–1156` 将当前 buffer、layout、
thread bounds 和 workspace 接口交给局部操作；`src/op/gemm.cc:198–236` 返回当前
GEMM 的 lowering 子树。Intent 的局部矩阵也消费已形成的 operand 和窗口，DSA
额外承担显式 local-memory/task 构造；BANG C 局部实现不接管作者的整函数算法。

## 共享分析与完整变换

### Canonical 关系与 physical construction

[CanonicalKernelAnalysis](include/Intent/Analysis/CanonicalKernel.h) 查询 immutable
KIR 的实际 def-use 与类型关系。[Kernel.cpp](lib/Analysis/Canonical/Kernel.cpp)
中的 `indexRelation` 同时返回每项索引贡献的
`resultAxes` 和 tensor index 的 `indexAxes`；GPU、CPU、DSA construction
直接消费这份映射，不各自计算 advanced-index block 的位置。
`operandProjections` 描述操作数轴到结果轴的投影，`axisProvenance` 跟踪坐标来源。
相同 extent 不证明相同坐标；
row-major reshape 也不自动成为 transpose。

[Shapes.cpp](lib/Analysis/Canonical/Shapes.cpp) 的
`tensorExtent(value, axis, fieldPath)` 按实际 SSA、product 字段路径与轴查询常数、
尺寸 SSA、domain、shaped-value dimension 或 inferred reshape 关系。
`reshapeGroups` 查询明确的元素重组关系，`equalTensorExtents` 证明逻辑尺寸相等，
`emissionAxis` 从 RegionScan 的实际 emit 字段定位恢复完整 source extent 的轴。
维度 identity 可用于证明相等，不能据此任意替换尺寸来源或统一 product 各字段的 shape。

[LogicalShape.h](include/Intent/Conversion/LogicalShape.h) 的 `reifyLogicalExtent`
以 `OpFoldResult` 接收 family 已绑定的 SSA 或 typed expression；
`materializeLogicalExtent` 是它的 `Value` 适配入口，两者共用
[LogicalShape.cpp](lib/Conversion/LogicalShape.cpp) 中同一份 inferred reshape
乘积／商解释。新增逻辑 shape 规则进入 canonical 查询，family 只实现实际绑定、
整数运算拼写与物化，不各自解释逻辑尺寸公式或维护全局 dimension 到值的替代表。
GPU construction 仍可读取显式 `Dim(source)` 的来源关系来选择该 source 当前的
fragment capacity；这属于物理映射，不将 capacity 返回成作者可观察的逻辑尺寸。

物化沿关系查询时，在当前作用域已有的 actual value/formal 处停止，保留具体字段和轴。
CPU 用当前 tensor/memref 的 dimension，且检查 dominance 与隔离 region 边界；
DSA 的逻辑 extent 与局部 storage capacity、GPU 的完整逻辑 extent 与 fragment
extent 各守自己的含义，不能因数值相同而互换。SourceSlice 的 member extent
属于当前 slice；它既不能被外层完整 source 尺寸覆盖，也不能自动成为 host launch 参数。

职责对照：本机 MLIR 20 `include/mlir/Interfaces/InferTypeOpInterface.td:344–370`
按实际 operands 为每个 result/axis reify `OpFoldResult`；本地 Triton
`lib/Dialect/Triton/IR/Ops.cpp:221–244,531–544` 分别按 source permutation 和每个
reduce operand 推导结果形状。Intent 保持 canonical KIR 只读，将关系查询与各
family 的实际物化分开，不复制目标 layout 或 local-memory 表示。

GPU 的私有 [ConstructionSchema](lib/Conversion/KIRToGPU/ConstructionSchema.h)
把这些 canonical 关系接到现有 physical value schema：构造实际 projection 后，
复用 GPU 的 current-IR schema 查询。这里是 KIR 到 GPU 的单次边界，
后续 GPU passes 只读当前 GPU IR。CPU 的 memref/linalg 构造和 DSA 的
local-memory/workset 构造保留各自实现；共同轴映射不决定它们的 storage 或 task。

GPU construction 的公开入口在 [KIRToGPU.cpp](lib/Conversion/KIRToGPU/KIRToGPU.cpp)。实现文件位于同一目录，按构造对象分工：

| 私有模块 | 修改入口与职责 |
|---|---|
| [KernelConstruction.cpp](lib/Conversion/KIRToGPU/KernelConstruction.cpp)、[KernelABI.cpp](lib/Conversion/KIRToGPU/KernelABI.cpp) | 完整 kernel/workset/launch 构造与公共参数到物理 ABI 的连接 |
| [Types.cpp](lib/Conversion/KIRToGPU/Types.cpp) | scalar、fragment 与 product 的类型、轴身份及 extent 投影 |
| [RegionLowering.cpp](lib/Conversion/KIRToGPU/RegionLowering.cpp) | 单一 typed dispatch、region 递归、value 映射与 origin 传递 |
| [Coordinates.cpp](lib/Conversion/KIRToGPU/Coordinates.cpp) | domain、subregion、range 与 index SSA 构造 |
| [AccessCoordinates.cpp](lib/Conversion/KIRToGPU/AccessCoordinates.cpp)、[AccessRelations.cpp](lib/Conversion/KIRToGPU/AccessRelations.cpp) | 访问坐标、结果 schema、操作数投影及 validity |
| [Memory.cpp](lib/Conversion/KIRToGPU/Memory.cpp) | buffer、读写、gather/scatter 与 atomic 操作构造 |
| [Values.cpp](lib/Conversion/KIRToGPU/Values.cpp) | pointwise、reshape/broadcast 与 product 值构造 |
| [Structured.cpp](lib/Conversion/KIRToGPU/Structured.cpp)、[Control.cpp](lib/Conversion/KIRToGPU/Control.cpp) | structured helper/collective/contraction 与显式控制 region |

私有 [Construction.h](lib/Conversion/KIRToGPU/Construction.h) 只声明跨文件接口和 `ScalarRegionLowering` 的词法状态。各操作构造共享同一个 builder、value/dimension/parameter 映射与 canonical analysis；子 region 仅在原词法边界派生上下文。新增操作在对应实现文件中处理，并接入唯一 dispatch，不创建平行的 lowering 路径。

构造 region 时用 `StructuredOpInterface::getRegionArgumentRelations` 查询
当前 block signature 的输入来源；`getValueRelations` 还包含 yield/result 边，
要求整个 operation 构造完成。两者共用入边定义，不能为了提前分析而制造占位 yield。

职责对照：本地 Triton `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:33–44`
用 typed adaptor/type converter 建目标操作，`:377–409` 保留 reduce/scan 的 helper；
TileLang `src/transform/lower_tile_op.cc:410–445` 先检查 shape/layout 合同，再映射
原多维索引。Intent 的 `ConstructionSchema.cpp:24` 同样消费明确关系，但 workset
可能给同一 KIR 类型附加不同执行前缀，因此桥接以当前 value 和作用域为单位。

阶段之间转交分块责任时，应使用接收方的实际能力查询。
[PointwiseAnalysis.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseAnalysis.cpp) 只有在
`hasRangeContractForm` 成立时才把输出 ranges 交给 contraction realization；
坐标可重放本身不证明后续阶段能够接管，否则仍由普通 pointwise ownership 负责。

### 同源的源码支持与发射

[OperationEmitters](include/Intent/Serialization/Source.h) 用 typed operation
注册 legality check 和 emission handler。最终支持检查与实际发射查询同一个表；
新增操作时在所属表登记，不再另加一份“允许操作”的白名单。
这只决定终端能否表达当前操作，不能替代 provider 对资源、effects 和物理结构的完整验证。

Triton/cuTile 共用 [PythonEmitter.cpp](lib/Dialect/GPU/Serialization/PythonEmitter.cpp)
中的 SSA、helper、record、结构化控制流与普通 GPU 操作遍历。
各自 serializer 注册 memory、collective、descriptor、MMA 等目标操作，并通过
明确的语言钩子拼写 range、cast 和 scalar/fragment 表达；发射层不选择新的算法或物理形式。

Mojo、BANG C、Weft host 与 Weft device 通过 [ScalarOps.def](include/Intent/Serialization/ScalarOps.def)
共用标准 scalar operation 的登记和语义解码。新增标准标量能力先补这份 catalog
及 [Scalar.cpp](lib/Serialization/Scalar.cpp)，再实现实际目标 renderer；
renderer 同时用于无输出的支持检查。C 家族共用整数宽度、除法、比较和转换表达，
Mojo 保留自己的 SIMD 语法，BANG C 保留 bf16 存储载体转换。
合法的未实现表达明确报错，不通过默认表达式继续生成源码。

Weft device 的 [ScalarValues.cpp](lib/Target/Weft/Transforms/ScalarValues.cpp)
消费同一个 `ScalarOperation`，将其转换成 Canonical Weft 原生操作；host renderer
负责 C 源码拼写，两者不各自重解 `arith`/`math`。整数运算的 signedness 由标准
operation 决定，不能从 Weft 的 index 存储载体推断；floor/ceil division 用商、余数
和符号形成精确结果。原生 cast、widen/narrow 与逐位转换分开，不能以数值 cast
替代浮点 bitcast。新增能力时先确认 native primitive 的类型和数值合同，再扩展此转换器。

标准原生内存与控制流由
[NativeSourceEmitter](include/Intent/Serialization/NativeSource.h) 共同翻译，
Mojo、Weft host 和 BANG C 直接消费当前 `memref` 与 `scf`：

- `metadataEmitters` 处理 dim、cast、subview、reinterpret 和 metadata/pointer extraction。
  Subview 相对当前 view 合成 offset/strides；reinterpret 相对 storage base 替换元数据。
- `controlEmitters` 处理 for、if、while 的 scalar 与完整 descriptor 传递。
  多值赋值先保存全部源分量，再写目的分量，保留交换状态和循环退出值。
- `memoryPointer` 使用 descriptor 的 offset、索引和真实 strides；释放仍使用 base。
  `bindEntryMemory` 读取当前类型，动态尺寸与 strides 从现有 NativeABI slots 绑定。

这些对象只保存源码表达式，不选择 tiling、bufferization、layout 或 lifetime。
Target 保留指针语法、分配、SIMD/atomic、DMA 与任务派发。
Mojo 的 [serializer](lib/Target/Mojo/Serialization/Serializer.cpp)、Weft 的
[HostSource](lib/Target/Weft/Serialization/HostSource.cpp) 和 BANG C 的
[serializer](lib/Target/BangC/Serialization/Serializer.cpp) 为原生内存操作登记实际 check+emit；
对应的 terminal 支持检查消费同一张表。新增内存形式应同时给出受支持的 layout、dtype
和 memory space，不能只把操作名加进 verifier，也不能在发射时把任意 memref 当二维连续 NRAM。

DSA 公共 memref 的 layout 同样表达真实 stride 约束；未知 strides 保持动态，
不以 identity layout 代替 ABI 事实。共同 emitter 支持某种 view/control，不代表 DSA
的 alias/lifetime 分析已经支持它；family verifier 继续负责自己的执行合法性。
这与 LLVM 20 的 `MemRefToLLVM.cpp::ReinterpretCastOpLowering` 和
`LLVMCommon/Pattern.cpp::getStridedElementPtr` 将 descriptor 语义集中处理的边界一致。
Intent 保留当前动态 strided source 能力，不引入只适用于静态 identity memref 的终端路线。

职责参考：本地 TileLang `src/cuda/codegen/codegen_cuda.h:23–65` 与
`src/backend/common/codegen/codegen_c_host.h:45–86` 复用公共 codegen 并覆写目标能力；
Intent 在相同职责边界共享遍历与语言状态，GPU 原生 collective/layout 仍交给
Triton/cuTile，CPU/DSA 的 physical program 仍由各自 transforms 形成。

### Canonical product 的结构查询

[ProductSchema.h](include/Intent/Analysis/ProductSchema.h) 从已验证的 canonical
tuple/record 类型读取字段，并按声明顺序递归访问 leaves。`walkProductLeaves`
同时给出结构字段路径；`getProductComponentType` 查询该路径的类型，
`getProductLeafRange` 和 `getProductLeafRanges` 返回展平
SSA components 的范围，不能把它们当内存字节偏移。字段名只用于诊断。

[KIRToCPU](lib/Conversion/KIRToCPU/KIRToCPU.cpp) 的 tuple/record 构造、extract、
helper 和 carry 使用这份查询；DSA 的 [Values](lib/Conversion/KIRToDSA/Values.cpp)
与 [Collectives](lib/Conversion/KIRToDSA/Collectives.cpp) 也消费同一顺序和范围。
CPU 保留自己的 buffer/state 表示，DSA 保留局部内存与惰性 SSA 物化。
KIR normalize/verifier 复用同一类型查询，不各自递归解释 schema。

参考 Triton `python/triton/language/core.py:771–804` 的 tuple 类型和递归
flatten/unflatten：共享的是类型结构和 component 顺序，物理存储与执行组织仍由
消费者决定。新增 product 操作先复用此入口；新的 storage、task 或 provider ABI
规则进入相应 family，不扩展为公共 product 类型的隐藏目标策略。

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

[Analysis/ControlFlow.h](include/Intent/Analysis/ControlFlow.h) 的
`queryControlFlowIncoming` / `queryControlFlowOutgoing` 读取标准 region branch、
terminator 与 CFG branch 接口，以真实 `OpOperand` 槽连接 `BlockArgument` 或
`OpResult`。Entry、region transfer、exit、bypass 和 CFG branch 分别保留边的种类
及来源/目标 region；同一 SSA 占两个初值或 yield 槽时不合并它们。查询返回结构边，
不替消费者决定可达性、重放资格或循环归约语义；produced 值、未知转发和尚未完成的
terminator 通过 `complete=false` 表达。结果只在当前 IR 未改动期间有效。
[ResourceAlias](lib/Dialect/GPU/Analysis/ResourceAlias.cpp) 与
[范围来源](lib/Dialect/GPU/Analysis/RangeProvenance.cpp)、[重放分析](lib/Dialect/GPU/Analysis/Replay.cpp) 共用这些边，
不各自计算 For/While 的 operand 偏移。

职责对照：Triton 的 `include/triton/Dialect/Triton/IR/TritonOps.td:761–818`
在 reduce/scan 运算上声明 `InferTypeOpInterface` 与 region 验证；
MLIR SCF 的 `SCFOps.td:136–148,953–958` 在循环上声明 region branch 接口。
Intent 同样让运算提供自身结构，但保留显式 identity/capture、逻辑坐标与 region segmentation 合同；
没有将这些语义交给 provider serializer 或框架名称推断。
本地 Triton 的 `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:1093–1114`
按具体 use 槽查询控制后继，`:1127–1136` 反查目标的 incoming operand 槽。
该 snapshot 使用更新的 MLIR mapping API；Intent 的共同查询在 MLIR 20 的标准
forwarding ranges 上保持相同的位置语义。

普通 GPU 值运算的坐标关系由
[FragmentOpInterface](include/Intent/Dialect/GPU/IR/FragmentOpInterface.h) 提供。
Unary、binary、compare、select、cast、broadcast、transpose 和 reshape 的
`queryFragmentOperandRelations` 按 **operand slot** 返回有序轴组；同一 SSA 值
占据两个操作数位置时仍是两条关系。Reshape 的 execution prefix 和 reassociation
在 [IR 实现](lib/Dialect/GPU/IR/FragmentOpInterface.cpp) 中解释一次。
新增同类运算在 ODS 声明接口并实现关系，不在每个分析和 provider 中各加分派。

`transportFragmentResultType` / `transportFragmentOperandType` 消费改写前的关系
和改写后的类型。Unary/cast 转发输入的完整 lane schema，保留声明的结果 dtype；
投影运算则保留目标的逻辑轴身份、validity 和 owner，传递确定的物理 extents。
需要分解一个多轴乘积而没有足够关系时，查询失败，不猜测某个维度的除法。
关系只在当前改写内使用；修改 operands、types 或 reassociation 后重新查询。

增加独立执行轴时，使用 [ExecutionSchema](include/Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h)。
调用方先决定执行域，再由 `lift`/`project` 构造 scalar、fragment、record 的目标类型，
并取得原轴和执行轴到目标轴的实际位置。已存在的轴不再插入；操作中的 reduction、
gather、transpose 和 contraction 轴属性按实际映射修改，不能统一加上请求的轴数。
这类 rank 变化和上面的同 rank extent 传递有不同前提，共用当前 IR 的类型与关系，
不增加另一份持久执行计划。

`cloneWithSchema` 保留原操作的属性与 properties，按真实映射后的 operands 和选定结果
完成类型投影；普通 elementwise/product 可以从操作数推导，具有独立轴选择的操作必须
给出结果 schema。Ownership、归约 combine 和带谓词的控制流共同消费这份机制。
原位改写先保存本次操作数的类型，因为 producer 修改后不能再用新类型解释旧轴编号。
新优化需要改变执行域时在这里扩展通用机制；是否允许重排、读取、复制、屏蔽 effects
以及怎样处理 inactive carry，继续由所属 transformation 决定。

职责参考是本地 Triton `lib/Dialect/TritonGPU/Transforms/Utility.cpp:840–872` 的
`cloneWithInferType`：先用 MLIR 克隆保留操作，再根据实际 operands 推导结果。
它主要传递 encoding；Intent 的 workset rank 变化还必须明确维护轴位置，不能直接
套用 encoding 传递。Triton 的 `TritonToTritonGPUPass.cpp:134–188` 同样在
expand-dims 转换中同步处理 shape、axis 与 encoding，并保留其它原操作属性。

[RangeProvenance](lib/Dialect/GPU/Analysis/RangeProvenance.cpp) 使用轴组追踪范围；
[Relations/Worklist](lib/Dialect/GPU/Transforms/Value/Relations/Worklist.cpp) 负责唯一关系工作队列。
[SchemaMutation](include/Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h)
提供共享改写：关系闭合使用 `closeSchemaBoundary` 从当前 producer 闭合 product、控制和 structured
helper 的 schema；`projectSchemaBoundary` 将已选 schema 投影到对应 incoming 槽，
不反向改写被多个组件共享的 seed SSA。克隆消费者使用 `rewriteClonedPhysicalTypes`，先保存未改写源
operation 的轴关系，统一改写 clone 的结果与 region formal 类型，再用实际 operands
传递结果的轴关系。
已知某个 fragment 轴的改写使用 `retargetFragmentAxisExtent(value, axis, extent)`；
`retargetSourceExtent` / `retargetDimensionExtent` 只负责从调用方给定的语义定位起点。
传递过程中保留值、record 字段路径及实际轴位置，沿 operation interface 的 operand
槽位关系传播，不在每个相邻值中按 source ID 或同 dimension 重新猜轴。所有入口返回
`LogicalResult`，调用方须将边界投影失败传回完整 transformation，不能继续报告成功。
结构化状态边界使用 `retargetValueExtents` 按 producer 的对应字段与轴传递 extent；record
只是分组，不能因两个字段带有同一 logical dimension 就强制它们采用相同物理宽度。
字段之间真正的对齐要求由 combine 内的逐元素、归约等操作关系表达。
逐元素闭合只把已有、可证明已物理化的结果执行域要求传给尚未独立物理化的结构字段，
并要求来源范围已知；不会仅因某个 extent 是 parameter 就让它覆盖常量或独立遍历。
[ValueMaterialization](lib/Dialect/GPU/Transforms/Value/ValueMaterialization.cpp)
与 region helper 展开复用这些入口传递分段形状。Pointwise coverage、访问组合和
online-summary 查询也消费相同的物理轴组，不能把逻辑 reshape 轴直接用作物理下标。
Triton/cuTile 的局部资格判断也读同一关系。
具体规则相邻放在 `Value/Relations/`：`Pointwise`、`Access`、`Contraction` 和
`Structured` 各自处理操作关系，`ExtentPropagation` 负责 extent 穿过聚合值与控制流
边界的传播。规则共用私有 `Worklist.h` 的类型更新与通知机制，不各建队列；对外仍使用
[ValueRelations.h](include/Intent/Dialect/GPU/Transforms/Value/ValueRelations.h)。
`FragmentOpInterface` 的关系还区分唯一 schema 输入、完整逐元素映射和 operation 自有的
固定轴。消费者按这些事实传递类型：RandomBits 的 counter 提供 shape，seed 保持 scalar；
Join 的输入 rank 保持不变，新增尾轴仍为 2。关系快照仅服务一次分析或改写，不是另一份 IR。
访问改写从 `queryAccessCoordinateAxes(access, slot)` 取得 coordinate 与 payload 的对应，
contraction 的成对轴及 reshape/transpose 的投影也保留具体 occurrence。
Region scan 的输出组装保持 slice/result 轴序；成员轴通过
`queryRegionScanEmissionAxis` 从当前 helper source-slice 关系取得，分段 extent 不沿组装边
传播成完整输出的 extent。
克隆默认保留原广播限制；需要将旧物理 singleton 与对应结果共同细化时，调用方通过
`ClonedAxisRefinement` 证明两端属于同一个已选坐标域。Region 切片与 helper 实参绑定
复用各自已有的切片/位置关系提供证明，单凭两个新 extent 相等不能获得这项许可。
坐标对应不证明数值相等：cast 舍入、逻辑 singleton、effects 和可重放资格仍由
各自分析或变换检查，不能仅因物理 extent 为 1 就消除整条逻辑轴。

变换中的 extent authority 可由一个已知输入建立，而其它输入仍待闭合；这种单条边的
合法投影继续使用 IR 的 `queryAxisProjection` / `queryBroadcastProjection`，不要求
整个运算已经具备最终 schema，也不在消费者重写这两个查询的规则。

职责参考：Triton `lib/Dialect/Triton/IR/Ops.cpp:221–244,800–825` 由运算推导
shape/layout，`lib/Dialect/TritonGPU/Transforms/Utility.cpp:503–641` 共享前后向
layout 推导。Intent 在 GPU IR 内共享自身的 fragment 轴合同，provider 继续负责
目标布局；不把 Triton 的 layout 表示移植为另一套共享执行计划。

### 跨执行模型的标量数值 lowering

[Conversion/ScalarLowering](include/Intent/Conversion/ScalarLowering.h) 接受当前 KIR
运算与 family 已物化的 scalar operands，统一生成 arith/math 操作。CPU 与 DSA
construction 复用普通算术、比较、转换和选择；CPU 的循环/向量、DSA 的 local-memory
遍历及原生近似 primitive 仍由各自模块形成。它属于 conversion，不放入只读 Analysis。

先根据 KIR 的逻辑 signedness 选择运算，再用 `realizeIntegerStorage` 将执行载体
归一为 signless integer；公共 `InterfaceAttr` 保留作者 dtype。新的 unsigned
lowering 必须同时接通 provider 对应的原语和发射，不能只让上游产生新 arith op。
若原生 tile primitive 无法表达 signedness，DSA 使用已有逐元素构造消费共同标量
lowering，而不把 unsigned storage 当作 signed 数值运算。

### 分析与改写的职责

访问操作的 operand schema 也由 IR 自己提供。KIR 的
[IndexedAccessOpInterface](include/Intent/Dialect/Intent/IR/IndexedAccessOpInterface.h)
区分 source、indices、写入值、读的 validity/fill，以及 CAS 的 expected/desired。
`IndexRelation` 的 operand positions 只引用 indices 分组，不是整个 operation 的位置。
新增或修改访问时，用 ODS 的命名 operands；不要再添加 `value_operand_index` 一类旁路字段。
逻辑 buffer 的动态 extents 与 optional initializer 同样独立，shape relation 只引用 extents。

[CanonicalKernelAnalysis::indexRelation](lib/Analysis/Canonical/Kernel.cpp) 将当前分组解析成
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

坐标到访问结果或 payload 的投影统一查询 `queryAccessCoordinateProjection(access, slot)`。
`slot` 是原 coordinate operand 的位置，`source_axes` 另行说明它访问 resource 的哪一轴。
同一个 SSA range 在两个位置出现，可以构成两个独立的 Cartesian 轴；不能按 SSA 去重，
也不能先按 resource 轴排序再重推投影。Provider 必须携带原投影完成重排、扩轴与广播。
Schema 闭合使用 `queryAccessCoordinateAxes` 的已知轴对；它可以返回部分关系，
只有完整且 extent 相容的 `Exact` projection 才能用于实际访问 lowering。
Verifier、value relations、Triton pointer/descriptor 与 cuTile native access 共用这两个查询，
新增访问规则不再各自实现 Cartesian、重复 provenance 或 singleton 轴匹配。
需要展开实际 coordinate SSA 时，两个 provider 共用
[ValueMaterialization](include/Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h) 的
`materializeAccessCoordinate`，由它按同一投影形成 transpose、显式扩轴和 broadcast。
它不重放 producer 或重新分块；目标访问选择和 load orientation 仍由 provider 决定。

访问接口不是优化许可：Gather 仍是纯 SSA 读取；atomic 的 ordering、sharing 和目标能力
独立验证；普通 store 的 payload 投影也不自动适用于 atomic 或 scatter。
读取共同字段以后，消费者仍需保留自己原有的别名、effect、重放与 predication 资格。
这种边界对应 Triton `TritonOpInterfaces.td:130–173` 中分别声明 predicate 和 atomic
语义的做法；Intent 的 KIR 逻辑索引与 GPU 物理访问继续使用各自的接口。

访问组合的完整入口仍是 [RealizeAccessComposition.cpp](lib/Dialect/GPU/Transforms/Access/RealizeAccessComposition.cpp)，
它保持规则次序、工作队列和最终关系闭合；相邻实现按改写对象组织：

| 私有模块 | 职责 |
|---|---|
| [AccessCoordinates.cpp](lib/Dialect/GPU/Transforms/Access/AccessCoordinates.cpp) | 消费明确绑定，重建 scalar/fragment 坐标、validity 与 fill；不重读未授权的 load |
| [AccessExpressions.cpp](lib/Dialect/GPU/Transforms/Access/AccessExpressions.cpp) | 整数索引重组和已被 mask 蕴含的坐标简化 |
| [AccessLoads.cpp](lib/Dialect/GPU/Transforms/Access/AccessLoads.cpp) | select/load、load/gather 组合，以及共享读取快照证明下的 load 复用与移动 |
| [AccessGathers.cpp](lib/Dialect/GPU/Transforms/Access/AccessGathers.cpp)、[AccessGatherProjection.cpp](lib/Dialect/GPU/Transforms/Access/AccessGatherProjection.cpp) | gather 穿过值计算、reshape、broadcast，及已有片段和 identity 访问复用 |
| [AccessReductions.cpp](lib/Dialect/GPU/Transforms/Access/AccessReductions.cpp) | gather/reduce 组合，保留普通归约与 scan 的不同顺序合同 |
| [AccessReshapes.cpp](lib/Dialect/GPU/Transforms/Access/AccessReshapes.cpp) | 多轴 reshape 与实际读写坐标的组合 |

这些实现通过私有 `AccessComposition.h` 连接，不能被当作任意次序执行的新 pass。

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

[Canonical matcher](lib/Analysis/OnlineSummary.cpp) 的 `matchOnlineSummary` 和 [GPU matcher](lib/Dialect/GPU/Transforms/Reduction/OnlineSummary.cpp) 的 `matchOnlineSummaryMerge` 共用这个核心，同时保留各自的四字段 summary 识别、axis/shape/cast 检查、候选枚举及 GPU physical schema 检查。扩展共同 combine 规则时从共享核心开始；调整 fragment 投影时改 GPU 适配层。CPU 当前没有接入这个 matcher，不能把这项复用描述成所有 family 已共用。

[ContractionAxes.h](include/Intent/Analysis/ContractionAxes.h) 统一 contraction 的 reduction/batch 配对、free axes 与操作数轴到结果轴的位置关系。CPU construction 和 GPU 当前 IR 查询共用这份纯轴关系；GPU 的 projection、vector realization 与 provider 原生矩阵检查消费相同结果。结果位置按正式 operand axis 推导，同一 source 或 dimension 在两边出现不代表同一个结果轴。该分析不读取 SSA、不选择 packing/tile，也不替各 provider 扩大原生 rank 或 dtype 支持。

同一接口中的 `ProductContractionAxes::get` 从两个操作数到乘积公共域的投影、已证明的逻辑 unit 轴及归约轴，推导 contraction 配对、保留的原操作数轴和结果排列。CPU 的 [ContractionSources](lib/Dialect/CPU/Transforms/Contraction/ContractionSources.cpp) 与 GPU 的 [ContractionSources](lib/Dialect/GPU/Transforms/Contraction/ContractionSources.cpp) 共用它识别乘法后求和的轴关系。CPU 的显式 contraction 查询也使用这个核心，但不将合法的 `K=1` 配对当成广播消除。调用方分别证明数值合同、唯一数值消费者和存储或 SSA 关系；物理 tile 大小为 1 不构成逻辑 singleton 的证明。新增共同轴规则改此分析；存储快照、方向选择、blocking 与目标支持改各自消费者。

[IntegerRelations.h](include/Intent/Analysis/IntegerRelations.h) 的 `foldIntegerDifference` 复用 `UniformExpression` 描述，只读折叠加减、常数乘法及等宽整数/index cast 的变化系数。[CPU VectorizeLoops](lib/Dialect/CPU/Transforms/Vector/VectorizeLoops.cpp) 用它判断循环坐标差值，再检查连续 stride、别名与依赖；[DSA CollectiveGather](lib/Dialect/DSA/Transforms/CollectiveGather.cpp) 用它判断四个参与者之间的地址差值，保留自己的 task 商余关系、只读视图、局部 buffer 写入和控制一致性证明。CPU 的这个 vectorizer 当前由 Mojo legalization 调用，共享 CPU family 不意味着所有 provider 都调用它。

该核心只接受 i64 或调用方明确绑定为 64 位的 index；两个适配层依据 Intent 的逻辑 index 合同传入位宽，不假定任意 MLIR index 都是 64 位。值运算仍遵守模整数语义，系数的加减乘另外检查是否能用 `int64_t` 表示；失败返回 `Unknown`，不能当作系数零。窄整数回绕后的扩宽需要独立范围证明，地址有效性、memory effects 与拓扑也不由系数证明。GPU 的按位宽模运算规范化和 source-axis 关系分析有不同合同，不应仅因都有 Add/Mul 就接到这一接口。

需要整数值范围时，复用 [IntegerRangeAnalysis](include/Intent/Analysis/IntegerRanges.h)：它按实际位宽消费 MLIR `ConstantIntRanges` / `InferIntRangeInterface`，并查询标准控制流和 shaped-value dimensions。CPU、DSA、GPU 适配层只补自己的参数、执行域与类型事实；未知整数保留完整位宽范围，IR 改写后重建查询。`provesSignedNoWrap` 与 `isValuePreservingIntegerCast` 分别证明有符号数学运算不回绕、cast 保持数值，不能以结果非负或两个整数类型代替这些条件，也不向 IR 添加 `nsw/nuw` 假设。

职责对照：本地 Triton `lib/Dialect/Triton/IR/Utility.cpp:55–84` 按比较的 signedness 和 operand bitwidth 建立范围；LLVM 20 `mlir/lib/Dialect/Arith/IR/InferIntRangeInterfaceImpls.cpp:67–90,223–252` 将算术及扩宽/截断交给共用的整数范围原语。Intent 复用这些位宽语义，额外固定自己的 logical index 为 64 位。

公开的 transformation 入口必须完成自身改写所需的 relation closure，使调用方得到满足 postcondition 的 current program。中间 repair helper 不因可以被调用就成为独立 pass；pipeline 负责次序，不应成为调用者必须记忆的隐式修复配方。

例如，[Region/Realization.h](include/Intent/Dialect/GPU/Transforms/Region/Realization.h) 的 fold/scan 两个入口在完成 region 改写后，自身调用 [closeValueRelations](lib/Dialect/GPU/Transforms/Value/Relations/Worklist.cpp)，通过工作队列闭合受影响的 value/access/aggregate 关系。这两个 region 阶段在 [GPU pipeline](lib/Dialect/GPU/Transforms/Passes.cpp) 中只调度完整入口，随后验证 postcondition。调用者不需要再附加一串 repair 调用；这也不要求 CPU 使用相同的关系维护方式。

[RealizeRegionFold.cpp](lib/Dialect/GPU/Transforms/Region/RealizeRegionFold.cpp) 与
[RealizeRegionScan.cpp](lib/Dialect/GPU/Transforms/Region/RealizeRegionScan.cpp) 分别拥有真实
遍历和状态构造。相邻的 `RegionPredicates` 负责范围分区与 identity 证明，
`RegionSummary` 负责 summary/carry 的字段处理，`RegionCoRealization` 负责已存在的
online/additive 合流。它们是组内机制，不是新增独立 pass。

Region 的私有 [RegionCloning.cpp](lib/Dialect/GPU/Transforms/Region/RegionCloning.cpp) 负责
helper 参数与 source slice 的 extent 绑定及内联；
[RegionScanOutputs.cpp](lib/Dialect/GPU/Transforms/Region/RegionScanOutputs.cpp) 负责 scan
输出消费者的坐标、分段和 tail 克隆。同 rank 的切片变换使用
`rewriteClonedPhysicalTypes` 保存源操作关系、更新 clone results/formals；不再只改
顶层 result 类型。新增独立执行轴则使用 `ExecutionSchema`，两种变换的选择由调用方明确。

带谓词的控制流由 [PredicateScalarControl.cpp](lib/Dialect/GPU/Transforms/Control/PredicateScalarControl.cpp)
检查资格和组织 pass，[PredicatedCloning.cpp](lib/Dialect/GPU/Transforms/Control/PredicatedCloning.cpp)
负责实际 active mask、读写 guard、SCF 与 inactive carry。后者复用共同 schema 克隆，
但保留 predication 所需的惰性求值、无效输入保护及循环终止语义；克隆失败必须由
ownership、buffer loop 和 scan consumer 传回完整变换。

### GPU 中直接可复用的接口

GPU 操作定义仍由 [GPUOps.h](include/Intent/Dialect/GPU/IR/GPUOps.h) 公开，具体验证与
类型推导位于 [IR/Operations/](lib/Dialect/GPU/IR/Operations/)：`Program`、`Values`、
`Shape`、`Access`、`Contraction`、`Collective` 按操作族归属。`GPUOps.cpp` 只编译
TableGen 生成定义；少量跨操作族的共同验证放在私有 `Verification`，不复制 shape
或 pure-helper 检查。修改某类操作时，其 infer/verify 实现与相关辅助函数相邻。

[GPU Passes.h](include/Intent/Dialect/GPU/Transforms/Passes.h) 只暴露完整变换与验证入口。实现内部需要的查询与改写按下表包含具体头文件，不通过一个通用 Utilities 模块取得所有能力。

[GPU Transforms](lib/Dialect/GPU/Transforms/) 按稳定职责组织子目录；顶层只保留
`Passes.cpp`、`VerifyGPUProgram.cpp` 和单一构建库的 CMake 声明。

| 子目录 | 放入的实现 |
|---|---|
| `Access/` | 访问坐标组合、load/gather/reshape/归约访问的实现 |
| `Contraction/` | contraction 分析、供数、分块、结果投影与遍历 |
| `Pointwise/` | workset 选择、执行域提升、coverage、ownership 与写回 |
| `Reduction/` | collective 分解、参数、combine、归约/scan 消费者及 online reduction |
| `Region/` | region fold/scan、source 准备、helper 绑定及输出克隆 |
| `Control/` | predication、遍历构造与融合、buffer loop 向量化 |
| `Value/` | 多组变换共用的 schema、投影、重放、关系闭合与局部值正规化 |
| `Storage/` | 私有值提升、保留值物化、store 调度与 workspace lowering |
| `Mapping/` | execution group、program mapping 和接口改写 |
| `Configuration/` | profiles、参数、候选及资源约束 |

新增 pass 先按实际职责选择组，再接入顶层 pipeline；组内多个 helper 不因此成为独立
pass。跨组需要稳定复用的构造接口在 `include/Intent/Dialect/GPU/Transforms/` 的对应
子目录，只有同一实现组或同一库内部使用的头留在相邻 `lib` 子目录。例如
`Pointwise.h`、`RegionCloning.h` 是私有实现接口；`ExecutionSchema.h` 是多个组的共享接口。
这与本地 Triton 的 `Transforms/Pipeliner/`、`Transforms/WarpSpecialization/` 分组及
`WarpSpecialization/PartitionAttrs.h` 私有头的组织方式一致；子目录不必各建一份库。

[Mapping/UniformBranches.h](include/Intent/Dialect/GPU/Transforms/Mapping/UniformBranches.h)
拥有 launch-uniform 分支的资格与重组入口，复用调用方传入的现有 shared transformations，
再合回一次 launch 和原 ABI。实际克隆与 mapping 改写在 Mapping 组，顶层 `Passes.cpp`
只负责调度这项已有策略。

GPU Transforms 的实现按 `Access`、`Contraction`、`Pointwise`、`Reduction`、
`Region`、`Control`、`Value`、`Storage`、`Mapping`、`Configuration` 分组，
仍由同一个 `MLIRIntentGPUTransforms` 库构建；顶层保留 pipeline 和完整程序验证入口。
公共头按相同职责放在 `include/Intent/Dialect/GPU/Transforms/` 下，组内实现头留在
`lib/`。`Reduction/OnlineSummary.h` 由 reduction 与 region 的实现共同使用，仍是
同库私有接口；这类依赖用明确的相对路径表达，不为目录整理扩大公共 API。

| 需要的能力 | 接口 | 使用方式 |
|---|---|---|
| 当前 value/access 的坐标、范围和复用事实 | [Analysis/PhysicalProgram.h](include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h) | 只读 current IR；相关 def-use、类型或范围改变后重算 |
| 资源 allocation 身份与别名 | [Analysis/ResourceAlias.h](include/Intent/Dialect/GPU/Analysis/ResourceAlias.h) | 从真实 allocation、公共参数和标准 control-flow/view forwarding 查询；不同 SSA 不代表不重叠，未知返回 MayAlias；资源定义或控制流改写后重建 |
| GPU 类型与形状属性自身的不变量 | [IR/TypeVerification.h](include/Intent/Dialect/GPU/IR/TypeVerification.h) | `verifyGPUTypeInvariants` 用 MLIR `AttrTypeWalker` 复用各类型/属性的 `verify`；完整 GPU verifier 在操作验证前调用，避免 release 构造绕过 checked constructor 后漏检 |
| 执行组构造、重建与 provider 展开 | [Transforms/Mapping/ExecutionGroups.h](include/Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h) | shared 变换维护真实 body 与坐标参数；`lowerExecutionGroups` 在 provider 准备入口统一展开 |
| scalar/fragment schema与投影轴 | [Analysis/ValueSchema.h](include/Intent/Dialect/GPU/Analysis/ValueSchema.h) | 只读查询当前类型与轴关系，不创建值、不选择 blocking |
| 已选执行域的类型提升、轴重映射和克隆 | [Transforms/Value/ExecutionSchema.h](include/Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h) | 实际 `oldToNew`/`executionToNew` 映射与 `cloneWithSchema`；不选择执行轴，不接管 effects 或控制策略 |
| 普通值运算的 operand/result 轴关系与形状传递 | [IR/FragmentOpInterface.h](include/Intent/Dialect/GPU/IR/FragmentOpInterface.h) | 按操作数位置查询；改写前取得关系，传递 extents，数值与重放资格由调用方证明 |
| 物理整数表达式求值 | [Analysis/UniformValues.h](include/Intent/Dialect/GPU/Analysis/UniformValues.h) | `evaluatePhysicalExpression` 接受 symbolic-leaf binding；算术和溢出检查共用一份实现 |
| scalar/fragment 整数值范围与数值保持 cast | [Analysis/IntegerRanges.h](include/Intent/Dialect/GPU/Analysis/IntegerRanges.h) | `queryIntegerRange` 给共同分析绑定 fragment element type、参数和坐标事实；设备算术保持定宽回绕语义 |
| host/resource 表达式的 checked 范围 | [Analysis/PhysicalExpressionBounds.h](include/Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h) | `queryPhysicalExpressionRange` 验证当前参数域上每个中间运算均可用有符号 64 位表示；可能溢出时返回未知，不把设备模运算结果用于 launch/resource 证明 |
| range/loop 中的整数比较与完整 tile 界限 | [Analysis/IndexPredicates.h](include/Intent/Dialect/GPU/Analysis/IndexPredicates.h) | `proveRangeComparison`、`queryCompleteTileLimit` 与 `queryIndexComparisonBound` 只读当前范围；区分已证明的真值、条件蕴含和未知 |
| 常量、大小关系与访问对齐 | [Analysis/IndexRelations.h](include/Intent/Dialect/GPU/Analysis/IndexRelations.h) | `IndexRelations` 共用于范围谓词、Triton descriptor 与 cuTile tile access；按 typed index 与回绕合同证明，不创建 guard 或选择原生 form |
| 参数声明与完整候选绑定检查 | [Analysis/PhysicalParameters.h](include/Intent/Dialect/GPU/Analysis/PhysicalParameters.h) | `ParameterSpace::read` 读取 kernel 声明；不依赖 SSA 读取是否存在；改变声明后重读 |
| fragment 结构资源估计 | [Analysis/Resources.h](include/Intent/Dialect/GPU/Analysis/Resources.h) | `FragmentResourceAnalysis` 缓存稳定 IR 的类型与参数使用关系；类型或 IR 改写后重建。估计不代替下层布局、寄存器分配和 occupancy |
| 静态与 specialization 后的候选条件 | [Transforms/Configuration/Resources.h](include/Intent/Dialect/GPU/Transforms/Configuration/Resources.h) | 从当前 IR 收集 typed requirements，静态筛选与 runtime 绑定求值使用同一条件；analysis 不隐藏改写 |
| value projection、replay、validity 与显式常量 | [Transforms/Value/ValueMaterialization.h](include/Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h) | replay 必须给出原语义位置和已绑定 SSA frontier；生成位置由 builder 表达，不能用空 anchor 跳过读取证明 |
| 改写后的 value/access/aggregate 关系闭合 | [Transforms/Value/ValueRelations.h](include/Intent/Dialect/GPU/Transforms/Value/ValueRelations.h) | 在完整 transformation 内调用，随后验证，不能让 serializer 补修 |
| coverage traversal、参数生命周期 | [Traversal.h](include/Intent/Dialect/GPU/Transforms/Control/Traversal.h)、[PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h) | 分别改变当前 ranges/access 与参数引用；参数替换同时覆盖 SSA、types 和 attributes |
| predication 与 retained slice | [Predication.h](include/Intent/Dialect/GPU/Transforms/Control/Predication.h)、[Storage.h](include/Intent/Dialect/GPU/Transforms/Storage/Storage.h) | 保持 effects、坐标、读取快照与 lifetime；不由 provider 字符串猜测 |
| allocation 声明与终端 workspace lowering | [Transforms/Storage/Workspace.h](include/Intent/Dialect/GPU/Transforms/Storage/Workspace.h) | `createProgramBuffer` / `createInvocationBuffer` 形成当前 IR 资源；`lowerWorkspaceAllocations` 唯一地形成 host ABI |
| workspace 分配容量 | [Analysis/Workspace.h](include/Intent/Dialect/GPU/Analysis/Workspace.h) | `workspaceAllocationShape` 按完整 Shared rows 替换参数，再形成逐轴矩形包络；只读查询不分配资源或选择候选 |

GPU 的作者 buffer、retained value 和 provider gather scratch 都先成为显式 `BufferOp`，
scope、初始化、实际读写与 lifetime 留在当前程序。Program/iteration-private allocation
保持词法访问顺序；invocation allocation 放在 kernel entry，并在多 program 时证明访问
切片互不冲突。Provider 只声明所需存储及访问，不自行追加 workspace ABI 或计算一套
host allocation shape。

[LowerWorkspace.cpp](lib/Dialect/GPU/Transforms/Storage/LowerWorkspace.cpp) 在参数绑定
稳定后统一消费这些 allocation。私有资源以实际 `ProgramId` 作为地址前缀，形成互不
重叠的 workspace slices；invocation 资源保留原坐标。终端产生实际分配形状的 `ViewType`
和 stride 参数，原 Buffer 的 `Dim` 仍读取当前候选 extent。较大的分配容量不改变逻辑
coverage、初始化位置或访问的 validity；atomic 的 physical sharing 随 backing resource
更新，原 memory order 保留。纯或 isolated helper 不能隐式捕获新 ABI 参数。

分配包络先将每个完整 Shared tuple 代入原表达式，再对同一结果轴取 `Maximum`，
不能先对各参数 domain 独立取最大值再组合。它是同 rank 的矩形容量，可能大于任一
单独候选的体积，不是新增候选或 winner。ABI leaves 与有正式 coverage binding 的
Deferred 参数保留，由现有 host 依赖顺序求值；缺失候选、未选 Provider 参数和非法
具体算术在此边界诊断。
表达式替换由 [ConfigurationExpressions.h](include/Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h)
的 `instantiateConfigurationExpression` 统一完成。Workspace 包络和访问上界证明
消费同一 checked 算术；`configurationExpressionAtMost` 按当前完整配置行分别比较，
保留同一行的参数相关性和 ABI leaves。它不以参数独立最大值替代组合关系；缺行或
无法证明时返回未知，Shared 阶段尚可能被 provider 扩展的 resident 初值不参与有限界证明。

职责对照：本地 Triton `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:571–587`
以 `GlobalScratchAllocOp` 声明每 program 的存储需求；
`lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp:74–104,118–143`
统一分配 offset 并发布总 size/alignment；`third_party/nvidia/backend/driver.py:304–319`
按 metadata、实际 grid 与 CTA 数调用 allocator。Intent 的 Shared tuple 包络服务于
候选选择前的 invocation 分配，具体候选的执行结构仍在 IR 中。

GPU 算术、cast、select 与纯 shape projection 的范围传递由 [IR/IntegerRanges.cpp](lib/Dialect/GPU/IR/IntegerRanges.cpp) 注册标准 `InferIntRangeInterface` external models，再调用共同的 `inferInteger*` 原语。新增运算在所属 operation 的接口模型中接线；[Analysis/IntegerRanges.cpp](lib/Dialect/GPU/Analysis/IntegerRanges.cpp) 只提供 GPU 固有事实，`IndexBounds` 保留坐标和控制关系证明，不再维护另一份算术范围计算。

访问构造保留完整的资源上下界；分块改变 range 的起点或容量后重新保留逻辑上下界，再由范围简化消除可证明的比较。不能把原 range 的非负结论沿用到新的坐标。Launch-visible 的 workset 分派和 pointwise tile 数通过 `PhysicalExprOp` 读取同一份 typed launch expression，不再单独构造另一套可能回绕的 SSA 除法与前缀长度计算。

`PhysicalProgramAnalysis` 的公开查询仍通过 [PhysicalProgram.h](include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h) 使用；维护查询算法时进入以下实现文件：

| 分析实现 | 职责 |
|---|---|
| [PhysicalProgram.cpp](lib/Dialect/GPU/Analysis/PhysicalProgram.cpp) | 分析上下文、program/resource ownership 与 structured/control source 查询 |
| [ScalarExpressions.cpp](lib/Dialect/GPU/Analysis/ScalarExpressions.cpp)、[IndexBounds.cpp](lib/Dialect/GPU/Analysis/IndexBounds.cpp) | physical parameter/launch 表达式、scalar 表达式归一、相等关系与 index 上下界 |
| [CoordinateRanges.cpp](lib/Dialect/GPU/Analysis/CoordinateRanges.cpp)、[RangeProvenance.cpp](lib/Dialect/GPU/Analysis/RangeProvenance.cpp) | 坐标范围及跨 value/control-flow 的范围来源 |
| [AxisRealization.cpp](lib/Dialect/GPU/Analysis/AxisRealization.cpp) | 物理轴实现、来源轴与派生 extent |
| [Replay.cpp](lib/Dialect/GPU/Analysis/Replay.cpp) | 坐标和值的可重放性、读取快照与 effect 资格 |
| [AccessRelations.cpp](lib/Dialect/GPU/Analysis/AccessRelations.cpp)、[AccessBounds.cpp](lib/Dialect/GPU/Analysis/AccessBounds.cpp) | 访问轴关系、有效性与资源边界证明 |

这些文件实现同一个分析对象，`unrestrictedRangeCache` 仍随该对象生存和失效；拆分不产生新的缓存 owner。相邻私有头只连接实际共用的 helper，不供 transforms/provider 绕过公开查询接口。

重放的结构查询与实际移动分开：`replayability` 可查询当前值图的结构资格；
`replayAt` 必须给出真实原语义位置以及本次改写的 `IRMapping`，已绑定值是保存的
SSA 快照，不沿其旧定义再次要求读取。没有 source selector 时检查整个未绑定 shaped
graph；指定 source 时，独立且支配原位置的值可复用。能够复用原值不代表允许克隆它的读取。

[MemoryEffects.h](include/Intent/Dialect/GPU/Analysis/MemoryEffects.h) 将读取稳定性和整个
区域的只读资格分开查询，消费 MLIR recursive effects、实际资源值与共同 alias 分析。
同一 effect resource 上可能别名的 write/free 会阻断移动，未知 effect 与 atomic 顺序也
保留为屏障。`assume_in_bounds` 写入独立的不可寻址 assumption-state resource：它保留
控制域中的约束，但不声称修改 tensor 数据。读取可以跨越这项独立 effect，含 assumption
的整个 region 却不能因此被当成纯值复制。本地 Triton `python/src/ir.cc:1859–1861`
同样创建 LLVM Assume；LLVM 将 assumption 的控制相关性建模为 inaccessible-memory
effect，而不是可由 DCE 丢弃的普通零结果纯操作。

[ReplayMaterialization.cpp](lib/Dialect/GPU/Transforms/Value/ReplayMaterialization.cpp) 集中执行按
source occurrence 或按 ranges 的重放，复用上述证明并检查替代值在实际 builder 插入点
可用；不同轴投影保留各自规则。Contraction 的 `ContractionValues` 保留重算或保留快照的
选择，不再另持一套克隆器。`createTraversalLoop`（[Traversal.h](include/Intent/Dialect/GPU/Transforms/Control/Traversal.h)）
先将循环接入当前 IR 再构造 body，供 contraction、region 与 scan 的位置分析使用。
这个边界可对照本地 Triton `FuseNestedLoops.cpp:246–274` 中分开的节点资格、dominance
和 hoist 集合；Intent 还需要证明逻辑 source 分块后的读取快照，不能仅以 pure 或只读 view 名称代替。

Fold/scan 共用私有 [RegionSources.cpp](lib/Dialect/GPU/Transforms/Region/RegionSources.cpp)：先保存
无法在新位置重新读取的 source，再从当前 IR 重算 source facts，最后形成片段和 tail。
普通与多轴归约共用 [ReductionReads.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionReads.cpp) 的相同
准备/物化边界。保留片段按遍历坐标索引原 SSA 快照，资源地址上的基址偏移不能再用作快照 ordinal。

资源查询的职责可对照本地 Triton `lib/Analysis/Alias.cpp:36–45`：真实 allocation
建立根，view 与 select 传播可能来源。Intent 的
[ResourceAlias.cpp](lib/Dialect/GPU/Analysis/ResourceAlias.cpp) 面向共同 GPU 的
buffer/public view/workspace，沿标准 region、branch、view 接口传播来源，供读取重放、
私有写入调度与 Triton 供数共用。未知来源不能证明不别名；不同 allocation 的证明也不能
替代同一 allocation 内的坐标不重叠证明。该查询不选择目标存储空间或建立另一份 lifetime 计划。

共享 GPU 的 `ExecutionGroupOp`（[GPUOps.td](include/Intent/Dialect/GPU/IR/GPUOps.td)）
拥有实际执行 body、坐标 block arguments、runtime/launch extents、coordinate roles
及 segment 范围；mapping 和 traversal 改写维护这一个 owner。
Triton [ProgramGrid.cpp](lib/Target/Triton/Transforms/Mapping/ProgramGrid.cpp) 先读取它调整网格，
再由 [Triton prepareTritonMemory](lib/Target/Triton/Transforms/Legalize.cpp)、
[cuTile prepareProgram](lib/Target/CuTile/Transforms/Legalize.cpp) 各自调用共同的
`lowerExecutionGroups`，生成纯 `DelinearizeOp` 坐标计算并展开 body。
正常编译与 shared IR 续编译使用同一入口；serializer 不保留或解释执行组。

收缩计算的完整入口在 [RealizeContractionBlocking.cpp](lib/Dialect/GPU/Transforms/Contraction/RealizeContractionBlocking.cpp)。同目录下 `ContractionSources` 负责合法的 source 规范化，`ContractionAnalysis` 负责轴与范围查询，`ContractionValues` 负责重算与保留快照的选择，`ContractionProjection` 负责结果关系，`ContractionTraversal` 与 `ContractionBlocking` 形成具体循环与 ownership。普通与 scaled contraction 共用能成立的判定和构造机制，各自的 dtype、scale 与 packing 条件留在相应实现。Provider 只通过 [Contraction.h](include/Intent/Dialect/GPU/Transforms/Contraction/Contraction.h) 调用必要的形状规范化与查询，不接管 shared blocking。

[ContractionTraversal](lib/Dialect/GPU/Transforms/Contraction/ContractionTraversal.cpp) 保留完整 retained result 的原始 contraction：完整物理 extent 属于已有 tile 候选域且不超过所选 tile，reduction 也已具备原生执行条件时，直接使用原 shape、读快照和 accumulator，避免分片与拼回；其它情况保留分片与 padding 路径。这是 current IR 的完整分支，不由 serializer 根据运行时 shape 猜测。

普通归约的完整入口和策略次序在 [RealizeReductionBlocking.cpp](lib/Dialect/GPU/Transforms/Reduction/RealizeReductionBlocking.cpp)。相邻私有模块分别承担具体机制：

| 私有模块 | 修改入口与职责 |
|---|---|
| [ReductionAnalysis.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionAnalysis.cpp) | source/root/extent 与 retained source 的只读资格查询 |
| [ReductionParameters.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionParameters.cpp) | reduction 参数选择与 free-axis 绑定 |
| [ReductionValues.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionValues.cpp) | identity、typed combine 克隆与 value schema 投影 |
| [ReductionCoverage.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionCoverage.cpp) | static padding、完整 coverage 与 tail neutralization |
| [ReductionDecomposition.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionDecomposition.cpp) | 多轴归约的分解和关系维护 |
| [ReductionTraversal.cpp](lib/Dialect/GPU/Transforms/Reduction/ReductionTraversal.cpp) | runtime chunk loop、nested hoist 与条件内归约 |

这些机制由同一公开 driver 调用，不是新 pass；driver 保留策略选择和改写后的 worklist 刷新。`SourcePlan` 只是一次改写读取的当前 SSA 事实，不能成为独立持久计划。跨变换需要复用的范围证明、replay 和参数生命周期仍使用上表中的共同接口。

Pointwise 的两个完整入口也在同一 driver 文件 [RealizePointwiseBlocking.cpp](lib/Dialect/GPU/Transforms/Pointwise/RealizePointwiseBlocking.cpp)：`realizePointwiseOwnership` 形成 ownership 与 program mapping；`realizePointwiseBlocking` 在已有 mapping 上形成局部 blocking、写回和复用 traversal。两者有各自明确的依赖次序，通过相邻私有头 [Pointwise.h](lib/Dialect/GPU/Transforms/Pointwise/Pointwise.h) 使用以下机制：

| 私有模块 | 修改入口与职责 |
|---|---|
| [PointwiseAnalysis.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseAnalysis.cpp) | 查询 source-axis、结构化范围用途、写入 effect 与现有 mapping 坐标；同一个 dimension 不自动代表同一个 Cartesian occurrence |
| [PointwiseCoverage.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseCoverage.cpp) | 兑现 scan/reduction 的完整 coverage，形成固定或局部范围、tail validity，保留不能安全 replay 的 gather source，并完成值关系闭合 |
| [PointwiseWorksets.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseWorksets.cpp) | workset 提升的依赖、合法性、选择与 range 构造 |
| [PointwiseLifting.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseLifting.cpp) | 消费已选执行轴，更新实际 SSA、structured operands 与控制边界；复用共同 schema 机制 |
| [PointwiseOwnership.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseOwnership.cpp) | axis occurrence 与 ownership 依赖、ownership 选择及 program mapping |
| [PointwiseTraversal.cpp](lib/Dialect/GPU/Transforms/Pointwise/PointwiseTraversal.cpp) | 重放合法 value graph，选择并形成写回/复用 traversal，兑现已确定 ownership 的 histogram |

`PointwiseRewrite` 只保存一次完整变换期间的工作状态；driver 在相关改写后重新读取 current-IR facts，执行决定写入当前 IR，不跨两个入口保留第二份 plan。新增局部机制放入对应私有模块；需要多个 GPU 变换复用的只读关系才进入公开 Analysis。

[SimplifyRangePredicates.cpp](lib/Dialect/GPU/Transforms/Value/SimplifyRangePredicates.cpp) 是 `IndexPredicates` 的改写消费者：全体物理 lane 上已证明的比较可替换为布尔常量；只有条件蕴含时，写入 specialization guard 与原谓词的逻辑或，未满足 guard 时仍保留原判定。它不把未证明的 shape 关系变成输入要求，也不负责 provider 的 native-load 分支或 MMA 循环组织。

新增一个 physical rewrite 时，先确定它读取的 current-IR facts，从上表选择查询或 materialization 接口；将 rewrite 和必要 relation closure 放进一个完整入口；在 family pipeline 中安排依赖位置与 postcondition 验证。新增只读查询应放 Analysis，只有本模块用的算法细节留在相邻私有实现，不扩大 Passes.h。CPU 或 DSA 的类似优化先复用它们自己的 analysis 和 storage/control 合同，只有与执行拓扑无关的规则才上提到公共 Analysis。

### GPU 候选声明、形成与消费

当前函数只有一份 `intent_gpu.configurations`，类型为 `ConfigurationSetAttr`。
`shared` 阶段绑定共同的静态参数；provider 完成合法性筛选后，以 `complete` 表替换它。
每行给出该阶段全部必要符号的具体值，顺序是候选枚举顺序。Coverage 的运行期 extent
通过独立的 deferred 声明绑定，不写成静态候选值。改写使候选失效时保留空 rows 与
已有 requirements；完整候选的消费者拒绝空表，不能将其当作默认配置。

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
| 创建声明、按需读取、改域、替换与改名 | [Transforms/Configuration/PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h) 的 `declareParameter`、`materializeParameter`、`updateParameter`、`replaceParameter`、`renameParameters` |
| 校验并发布候选表 | [Transforms/Configuration/PhysicalParameters.h](include/Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h) 的 `writeConfigurations` |
| 当前图的分类、关联参数及完整结果机会 | [ConfigurationAnalysis.cpp](lib/Dialect/GPU/Transforms/Configuration/ConfigurationAnalysis.cpp) |
| 有限 profile 解码、family 选择与 role 投影 | [ConfigurationProfiles.cpp](lib/Dialect/GPU/Transforms/Configuration/ConfigurationProfiles.cpp) |
| 候选 extent 与资源约束 | [ConfigurationConstraints.cpp](lib/Dialect/GPU/Transforms/Configuration/ConfigurationConstraints.cpp) |
| 共同候选形成的完整入口 | [MaterializeConfigTuples.cpp](lib/Dialect/GPU/Transforms/Configuration/MaterializeConfigTuples.cpp) |

这些私有 policy facts 只在一次不变的 current program 上使用，不跨改写缓存，不拥有
第二张执行表。修改 shared/deferred 声明使候选表失效；仅修改 provider 声明时，
已有 shared 表仍有效，complete 表失效。完整变换在结束前重新形成最终表。
改名与替换通过统一 owner 同时修改 attributes、result types 和 region argument types；
不能只做 SSA RAUW，也不能把名称改写分散到 serializer。
跨类型参数引用不是 MLIR 标准 SymbolTable 的完整遍历合同，因此这里使用 kernel-owned
typed references 和显式 owner API，不把自有参数冒充通用 module symbols。

`ConfigurationSetAttr.requirements` 保存 `ConfigurationRequirementAttr`：种类、单位、
有限 predicate、数量表达式、可选 activation 和诊断原因。Predicate 包括上界、相等、
正值、2 的幂和整除；上界/相等/整除有两侧表达式，正值/2 的幂只消费 usage。
Activation 只引用 provider 阶段的 `i1` 参数；未选择的形式不求值其数量条件，
不能把任意数据相关分支提升为 kernel 约束。`legality` 表示 provider 硬性限制，
`nominal_budget` 表示按当前物理结构使用的预算策略；`fragment_register_words` 的
单位是结构上的 32-bit words，不是机器码实际分配的寄存器。表达式引用当前参数或
host metadata binding，不依赖某个比较 SSA 值仍然存活。

新增条件时，先在所属 family/provider 从当前 typed facts 构造一次 requirement，
再交给 [Resources](lib/Dialect/GPU/Transforms/Configuration/Resources.cpp) 筛选和发布。
[Triton Configuration](lib/Target/Triton/Analysis/Configuration.cpp)
从当前 range、fragment 和 descriptor 取得条件，包括真实 block shape、元素上限、
连续字节与多 stage 对齐；descriptor 条件由当前 choice 参数激活。
[cuTile Configuration](lib/Target/CuTile/Analysis/Configuration.cpp) 保留共同归约预算
的适用范围，并表达 resident workers 与 CTA/occupancy 的派生等式。
候选形成与最终发布调用同一收集器；最终 verifier 重新读取当前 IR，核对条件集合及
候选绑定。条件集合不依赖遍历顺序，候选行仍保留原顺序。普通分支内 primitive assertion
留在原分支。Serializer 只导出当前 attributes，不识别 `cf.assert` 的比较形状。

Triton contraction 的展开元素上限由 [Contractions.cpp](lib/Target/Triton/Analysis/Contractions.cpp)
中的 `contractionExpansionRequirement` 与 form 选择共用当前 operand shapes，进入同一
候选条件收集器。已选展开形式读取实际 constexpr 分支条件；数据相关分支不能免除
原生编译合法性。未选分支逐个条件化 shape 因子，避免无用乘积先溢出；native dot
不会无条件承担展开张量的预算。扩展这类 provider form 时，在这里闭合条件，不向
serializer 或 runtime 另加 assertion 判定路径。

参数改名、替换和改域通过现有 mutation owner 同时维护 requirements；失效的是候选行，
不能顺便丢弃条件。Verifier 检查参数和 metadata 引用；改变控制域的变换必须证明条件
仍适用，不能把分支约束无条件合并。分析的 `Unknown` 保留给 invocation 绑定，
不是静态拒绝理由；实际 launch 时尚未绑定则明确报 `candidate_binding`。
共同 runtime 先检查 activation，再按 native evaluator 的逐节点有符号 64 位范围求值；
未溢出的正数量才参与 predicate。正 Add/Multiply 超过有限预算时可直接证明拒绝，
不把溢出解释为合法的小值。任一已知违反条件即可淘汰该行；剩余未知条件不能默认为通过。
全部候选被拒绝时，`candidate_selection` 给出类别、实际 usage/limit 和候选值。
已有 `optimization_remarks` 输出静态拒绝与待绑定条件，不另造决策日志路径。

Triton 的 [ConfigurationSchema](include/Intent/Target/Triton/IR/Configuration.h)
统一查询 kernel constexpr 顺序以及 `num_warps/stages/ctas` 对应的参数符号，
不保存候选值。Serializer 机械导出最终表与符号映射；
[gpu/configurations.py](python/intent/runtime/gpu/configurations.py) 负责一次解析、deferred
绑定和已声明条件的求值。参数的 `value_type` 从 IR 声明导出，不根据候选恰好是 0/1
猜测 bool。Triton 的真实 `Config` 从这张表投影，pruning、公开候选查询和 launch 前检查
共享筛选入口；单配置和缓存 winner 同样必须满足当前条件。实际 tensor 的指针、shape、
stride 及 allocator 仍由 descriptor runtime 检查，不能伪装成编译期数量。
Descriptor allocator 在本次调用的独立 Python context 内绑定，不覆盖宿主程序的设置。
cuTile 保留自己的 JIT 与调优入口。CPU 仍使用独立函数候选和 implementation
binding，不套用 GPU 的 block/config 表。

职责参考：Triton `python/triton/runtime/autotuner.py:140–147,276–279` 在试跑与最终调用中
消费同一 `Config.all_kwargs()`；`:328–380` 区分 kernel kwargs 与 native 编译选项。
Intent 先由 IR 验证完整绑定，再在 provider adapter 中投影这两类参数；runtime 不补默认候选。
Triton `python/triton/_utils.py:63–74` 核对 block 的 2 次幂与元素上限，
`python/triton/tools/tensor_descriptor.py:15–30` 区分 block 约束与实际 base/stride；
Intent 将可参数化的 block 条件写入当前 IR，保留真实 view 检查在 host adapter。

### CPU 中直接可复用的接口

CPU 的值构造、存储物化、候选绑定与执行变换有各自的入口。[共享 pipeline](lib/Dialect/CPU/Transforms/Passes.cpp) 依次完成 target 绑定、tensor SSA 优化与 bufferization、buffer source 规范化、候选形成、region 实现、供数与分块、task 形成。`CPUProgramStage` 区分值程序、buffer 程序和已实现程序；后两者不接受残留 tensor，最后一层也不接受尚未实现的 structured computation。Mojo 和 Weft 共用这些 CPU 阶段，provider 的微程序及机器表示仍各自实现。

[CPU Transforms](lib/Dialect/CPU/Transforms/) 的目录同时表达实现归属与 pass 边界：

| 目录 | 负责的当前 IR 改写或复用接口 |
|---|---|
| `Configuration/` | target facts、有限 profile、完整候选绑定与已实现候选去重 |
| `Implementation/` | 微程序注册与需求接口、函数级输入供应、共享准备和量化计算分组 |
| `Contraction/` | 乘法归约规范化、矩阵输入视图、contraction 分块 |
| `Collective/` | slice reduce/scan、标量归约、histogram 与归约遍历融合 |
| `Region/` | 分段计算分组与 region realization |
| `Storage/` | 值到 buffer 转换、ownership lowering、私有存储复用与中间 buffer 融合 |
| `Structure/` | 普通 structured computation 的融合、恒值折叠、实现展开及 loop 构造 |
| `Task/` | workset 分组、通用分块、task partition 与 captures 隔离 |
| `Vector/` | provider 可复用的循环向量化与横向归约机制 |

跨模块接口位于 `include/Intent/Dialect/CPU/Transforms/` 的对应子目录；调用方包含
所需的具体头文件。[Passes.h](include/Intent/Dialect/CPU/Transforms/Passes.h) 只提供
标准 pass factories 与 pipeline 入口，不再间接提供全部转换函数和分析类型。
`lib/` 中的窗口、整数来源和连续访问头文件是所属实现的私有
协作接口，provider 不通过相对路径包含它们。Vector 机制按 provider 需要调用，
不为了目录对称把 Mojo SIMD 展开加入共同 CPU pipeline。

各组的 `Passes.cpp` 将完整变换接到标准 MLIR PassManager；顶层 `Passes.cpp`
只组合这些入口及原生 `canonicalize`、`cse`。候选选择前的 source 规范化与选择后的
供数、分块、task 形成分别有可独立调度的入口；后者直接读取当前 IR 的 configuration
与 implementation binding。新增优化先判断它能否独立保持完整程序：能闭合的变换在
所属组注册 pass，依赖同一改写过程的内部修补仍留在该组，不要求调用方补齐隐含次序。

职责参考是本地 Triton `third_party/nvidia/backend/compiler.py:273–285` 对 TTIR
passes 的显式组合，以及 `lib/Dialect/Triton/Transforms/Combine.cpp:297–315` 中
变换自身的完整实现。TileLang `tilelang/cpu/pipeline.py:40–67` 同样在 CPU pipeline
中分开 reducer、tile lowering、allocation 与 vectorization。Intent 的 CPU 初始
输入还需要形成 task/block 程序；这里复用组织方式，不采用它们的 GPU layout 或
thread-binding 合同。

[KIRToCPU](lib/Conversion/KIRToCPU/KIRToCPU.cpp) 将普通值构造成标准 tensor/linalg SSA。Tuple/record 展平为逐分量的 scalar/tensor；`if/for/while` 直接携带这些值，helper 用真实返回值表达计算结果，不提前分配普通结果槽或两套循环状态槽。External view 和作者显式的 mutable buffer 保留 memref；[Read/Write](include/Intent/Dialect/CPU/IR/CPUValueOps.td) 明确表达内存与不可变值之间的 effect 边界。输出动态 shape 由 `tensor.empty` init 保存，不在旁表中恢复。

Construction driver 验证 immutable KIR，并把同一个 `CanonicalKernelAnalysis`
交给私有 [Construction.h](lib/Conversion/KIRToCPU/Construction.h) 上下文。转换期间的
SSA、product、domain 与 dimension 映射只有一份，完成后由独立 physical module
承接全部执行事实；后续 passes 不持有这个上下文。新增 canonical operation 的 CPU
转换在 [Construction.cpp](lib/Conversion/KIRToCPU/Construction.cpp) 的 typed dispatch
接入，并实现于对应模块：

| 模块 | 构造责任 |
|---|---|
| `Values.cpp` | 实际 SSA 绑定、product 展平、domain 与共享 logical extent 的 CPU 物化 |
| `Access.cpp` | indexed read/write、作者 mutable buffer、原子与 scatter effects |
| `Arithmetic.cpp`、`Tensor.cpp` | 数值合同、pointwise 和显式 tensor 形状运算 |
| `Collectives.cpp` | helper region、reduce/scan、分段 region 与 histogram |
| `Contractions.cpp` | 普通、稀疏、scaled 与量化 contraction |
| `ControlFlow.cpp` | ordered if/for/while 与 parallel workset |

这些构造模块不选择 provider 实现、分块或存储复用；这类改变当前程序的决定进入
上面的 CPU transform 组。

[Bufferization](lib/Dialect/CPU/Transforms/Storage/Bufferization.cpp) 是唯一值到存储边界，复用 MLIR One-Shot、tensor/linalg/SCF 的原生模型与 ownership deallocation。只读输入可以互相 alias，不能给每次读取附加 `to_tensor restrict`；只有当前 ABI 证明可写参数与输入分离时才借用只读存储，其 tensor 不允许被原地覆盖。可变读取在原 effect 位置形成快照。值融合只能移动满足 effect 条件的计算，不能把捕获 mutable load 的 tensor body 当成纯计算。

[CollectiveBufferization](lib/Dialect/CPU/Transforms/Storage/CollectiveBufferization.cpp) 负责 CPU collective 的 SSA 结果与 helper 签名转换；buffer 形式随后由现有 region、scan、reduction 与量化实现消费。新增 CPU structured operation 时，先明确 source/result、shape init、helper 参数与 effects，再实现 bufferizable 接口，不在 construction、Mojo 和 Weft 各写一套结果分配。已有标准 tensor 或 SCF 能表达的值关系直接复用标准操作；provider 专属 packing 继续归实现与 input supply。

值融合保留 MLIR 的 single-use profitability 条件，避免把共享 producer 复制到每个消费者；额外检查 payload effects，并将归约融合留给保存 CPU 数值许可的变换。不能在添加自定义 legality callback 时意外丢失原生代价条件。[RegionPredicates](lib/Dialect/CPU/Analysis/RegionPredicates.cpp) 按实际 source/capture 坐标关系收集比较，不依赖一个独立的 mask buffer 或单输出 generic；实例化与恒值分析消费同一组 predicate SSA。

普通 reduce 在 construction 中统一形成 `cpu.slice_reduce`，保存 sources、逐分量 identities、captures、shape init、SSA results、归约轴和完整 combine。标量 helper 保留多分量 SSA 计算；完整 slice helper 也先返回真实值，在 bufferization 时形成显式借用的 state/member/capture/destination 参数，不在 construction 提前生成归约循环。作者 product 的各分量可以有不同 rank 和保留形状，但归约 member axes 必须一致；负 axis 在每个分量上归一后必须指向同一组 canonical 位置。

[SliceCollectives.cpp](lib/Dialect/CPU/Transforms/Collective/SliceCollectives.cpp) 在候选选择前消费当前 CPU IR：同 free-axis 域、可逐元素提升的 scalar combine 形成原 `linalg.generic`；完整 slice combine 形成多归约轴遍历及逐分量私有 state/next slots，全部分量更新完成后再提交下一状态，最终写回 outputs。空域保留 identities；scalar helper 的不同 free-axis 域未获得明确提升关系时拒绝，不把未知动态 extent 当成相等。该模块和 slice scan 共用成员 subview、scratch、copy 与 helper 实例化，scan 仍独立保持 prefix、方向和 inclusive 合同。[CollectiveHelpers.h](include/Intent/Dialect/CPU/IR/CollectiveHelpers.h) 为 scan、slice reduction 和分段 region 提供 helper 的输入只读、局部 scratch 与调用方 destination 验证；嵌套 helper 按 operation 的实际 operands/effects 检查，不把其 formal arguments 当成外部存储。

[NormalizeReductions.cpp](lib/Dialect/CPU/Transforms/Collective/NormalizeReductions.cpp) 随后从当前单轴 generic 和私有 rank-zero 结果槽形成标量 SSA `cpu.reduce`：初始化、全部使用与生命周期必须闭合，才删除结果槽。Weft 的 [Reductions.cpp](lib/Target/Weft/Transforms/Reductions.cpp) 为 scalar SSA 与 shaped DPS 提供同一 native-combine 资格查询，NaN 规则、原初值与重排许可由当前运算确定；外层 scalar SSA 快照直接绑定，不重放其来源读取。

[ExtentRelations.h](include/Intent/Dialect/CPU/Analysis/ExtentRelations.h) 统一当前 extent 的判等、表达式和常数上界查询；[IR ShapeRelations](include/Intent/Dialect/CPU/IR/ShapeRelations.h) 从公共接口取得同一 dimension 的代表及静态约束，也供入口 `memref.dim` 规范化使用。[registerExtentRelations](lib/Dialect/CPU/Analysis/ExtentRelations.cpp) 由 [Compiler Registration](lib/Compiler/Registration.cpp) 注册 CPU helper、ABI 与缺失的描述符 ValueBounds 模型，复用 MLIR 已有的 memref/SCF 模型。未知关系保留为 unknown，不把任意可能 wrap 的 index 算术当作无界整数；extent 相等不代表 offset、stride 或访问坐标相同。

只读证明可以沿 helper formal 查询实际参数的尺寸，不能据此把外部 SSA 插入隔离 region。用于改写的 `queryExtentValue` 停在当前 block argument 和尺寸 SSA，只返回已有值或常量；消费者仍检查 dominance。共同查询在 IR 改写后重新执行，不维护独立 shape 表。

普通 contraction 的 construction 只形成完整 `linalg.generic` 索引映射、显式零初始化及原数值运算，不选择 dot、batch 循环或 packing。[Contractions analysis](include/Intent/Dialect/CPU/Analysis/Contractions.h) 从当前索引图和乘加 body 查询共享轴语义；[NormalizeContractions.cpp](lib/Dialect/CPU/Transforms/Contraction/NormalizeContractions.cpp) 在候选选择前形成 dot、矩阵和 batch 程序，并按实际 strides 决定能否使用视图。转置或 unit 轴投影的输入快照稳定时，矩阵可直接消费派生视图；非 unit 广播保留显式计算，无法通过视图表达的轴合并仍形成显式 pack 与 lifetime。实现所需的 panel 准备继续由 implementation requirements 与 input supply 负责，不能把整块转置重新藏进 construction。

`intent-cpu-normalize-contractions` 先调用 `normalizeContractionSources`，将满足条件的 f32 乘法与零初始化普通求和组合为显式乘加 contraction，再由上述 normalizer 和 implementation registry 处理。它仅穿过纯轴投影与 unit views，用 [Storage analysis](include/Intent/Dialect/CPU/Analysis/Storage.h) 证明读取快照稳定，不跨越数值 cast 或其它计算；没有独立 free 轴的逐行点积仍交给普通归约路径。只有一侧 free 轴时，还要求能够形成连续矩阵列，否则保留原 producer-fused reduction，避免为了单个向量结果物化和打包矩阵。源识别、投影视图折叠和零初始化证明集中在相邻 `ContractionSources.cpp`，矩阵展平、batch 循环与 pack 保留在 `NormalizeContractions.cpp`。修改其中一个阶段不需要在 provider serializer 新增算子分支。

Region 展开后，模板参数变成具体 views，可以用同一存储证明再次折叠矩阵输入；`intent-cpu-fold-contraction-inputs` 调用 [Contraction.h](include/Intent/Dialect/CPU/Transforms/Contraction/Contraction.h) 中的 `foldContractionInputs`，保持已绑定 implementation、计算和配置。`ContractionRequirements::unitInnerStride` 显式保存所选实现的输入布局条件，候选选择、绑定、lookup 与该改写共用检查；例如 Mojo direct 的 RHS 必须保持单位内层 stride。条件不成立时保留原物化，不重新选实现，也不把 Region 参数假定为 noalias。

| 需要的能力 | 模块 | 使用方式 |
|---|---|---|
| 值优化与存储物化 | [Bufferization.h](include/Intent/Dialect/CPU/Transforms/Storage/Bufferization.h) | `bufferizeValues` 完整关闭值程序；原生接口负责 alias、in-place、copy 与 ownership，不用未建模操作兜底 |
| 均匀存储内容 | [UniformValues.h](include/Intent/Dialect/CPU/Analysis/UniformValues.h) | `UniformMemoryAnalysis` 为普通折叠与 region predicate 共享读取、view 归一和 effect 失效；多输出先在同一输入快照求值，再同时发布结果 |
| 原生内存与控制流翻译 | [NativeSource.h](include/Intent/Serialization/NativeSource.h) | Mojo、Weft host 与 BANG C 共同处理标准 descriptor、视图、地址和 SCF 传递；目标保留分配与访存 API，serializer 不决定生命周期 |
| Contraction 初值 | [Contractions.h](lib/Dialect/CPU/Transforms/Contraction/Contractions.h) | 同一查询沿当前完整 `Copy → Fill` 取得初值及可删除性；初始化被其它消费者读取时保留，不在 blocking 中再写一套 Fill-only 扫描 |
| Weft task 局部存储 | [TaskStorage.cpp](lib/Target/Weft/Transforms/TaskStorage.cpp) | 均匀值保留 scalar 与尺寸，按实际读取窗口形成值；窗口几何保存标量 offset，向量访问才形成 gather 坐标，标量填充直接更新选中的坐标；已物化状态的覆盖与控制流交接保留 native owner，分支完整写入后才发布结果 |
| 外层与局部参数 | [Configuration.h](include/Intent/Dialect/CPU/Transforms/Configuration/Configuration.h) | 外层 task/block 参数与 implementation 的 local binding 分开，不通过完整 Passes.h 获取配置类型 |
| 有限 profile 数据 | [TuningProfiles.cpp](lib/Dialect/CPU/Transforms/Configuration/TuningProfiles.cpp) | 读取后形成 typed rows；family 与 local 参数由 provider registry 声明，未知或缺失参数明确诊断，override 整族替换 |
| 完整候选形成 | [Configurations.cpp](lib/Dialect/CPU/Transforms/Configuration/Configurations.cpp) | 从当前 computations 枚举有限 implementation portfolio；保留合法性筛选、顺序与去重，候选成为独立的完整函数 |
| 实现绑定与展开接口 | [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h)、[Implementation.cpp](lib/Dialect/CPU/Transforms/Implementation/Implementation.cpp) | `bind` 一次提交 operation binding、函数配置与实现摘要；供数与展开消费同一个选择 |
| 存储来源、别名、生命周期与读快照 | [Analysis/Storage.h](include/Intent/Dialect/CPU/Analysis/Storage.h) | `StorageAnalysis` 统一查询 `origins`、`aliases`、`lifetime`、`effects`、`disjoint` 与读取稳定性；消费标准 MLIR flow/effect 接口及显式 ABI，不移动 allocation 或决定 packing |
| 当前描述符的维度上界 | [Analysis/Storage.h](include/Intent/Dialect/CPU/Analysis/Storage.h) | `constantDimensionUpperBound` 委托共同 ExtentRelations 查询闭合常数上界；未知界不作为收缩依据，不使用观察到的运行时尺寸 |
| 供数与私有计算复用 | [ReusePreparedInputs.cpp](lib/Dialect/CPU/Transforms/Implementation/ReusePreparedInputs.cpp)、[FuseIntermediateBuffers.cpp](lib/Dialect/CPU/Transforms/Storage/FuseIntermediateBuffers.cpp) | 在共同存储证明之外，分别检查坐标、effect、读取稳定性与计算可重放性，实际改写 current IR |

扩展 CPU implementation 时，通过 `registry.add<Op...>(implementation)` 声明操作族；共同 registry 不再维护一份需要绑定的操作名单。可选 `applicable` 进一步限定当前实例的计算语义，`check` 查询当前 capability 与 configuration，合法时返回 `std::nullopt`，否则返回具体拒绝原因。`candidates` 与 `bind` 共用布局、provider 条件、参数和供数检查；无合法候选时，诊断定位阻断的 computation，并列出 profile 行的实际参数与原因。`lookup` 服务于已绑定且经过变换的程序，核对实现身份、绑定参数及当前计算和输入布局，不重新选择实现或用原始配置要求检查已经缩小的微块。

实际展开也由该实现注册：`materialize` 解释自己的参数后调用共同结构化构造；`expand` 通过 `ImplementationExpansion` 取得当前 native builder、借用视图或读取值，并明确写入 destination 或绑定 SSA result。共同 driver 不按计算名称裁剪参数、猜测输出位置或解释 provider 参数。后端原语的选型和编码仍属于 provider，不把整个算子组织移入局部实现。

绑定参数使用同一 `ImplementationParameter` 声明生成与验证：本地 profile 参数、别名、
常量和已有 shared extent 的有限派生分别声明来源，合法域也放在该声明中。
不再写一份生成字典的 callback，再在后续 pass 中假定所有键存在。
`verifyBinding` / `verifyBindings` 在继续编译当前 IR 或独立运行 CPU/provider pass 时
检查完整绑定；它们不重放依赖原始计算形状的候选筛选，也不重新选择实现。

`inputRequirements` 是只读查询，候选期与后续供数变换都可以调用。`checkInputRequirement` / `checkInputRequirements` 共享 operand、panel、alignment 与显式 widening 的证明，实际物化时依据当前 IR 重查。跨阶段只传递正式 binding，不缓存另一份供数计划。输入已经满足实现要求、无需额外准备时可以返回空需求；空需求只表示不需要外围 preparation，不说明它一定更快。

[ImplementationInputs.h](include/Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h)
公开函数级输入供应构造的三个入口：`prepare`、`prepareCaptured`、`prepareGroup`。
CPU passes 与 Mojo 共用同一上下文的准备和复用状态；内部 prepared windows、guards
及缓存结构由 `.cpp` 中的单一实现持有。仅供 CPU 内部使用的窗口分析留在
[InputWindows.h](lib/Dialect/CPU/Transforms/Implementation/InputWindows.h)。Mojo/Weft 共用的基础
index、算术与 loop 构造放在 [LoopBuilders.h](include/Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h)，
provider 通过公开 include 使用它们，不跨层包含 CPU 的私有源码路径。

[ImplementationInputs.cpp](lib/Dialect/CPU/Transforms/Implementation/ImplementationInputs.cpp) 的 group supply 将配置容量与当前 source 维度的已证明上界取小，只收缩未拆成 panel 的维度；panel 宽度、对齐、有效写入窗口及生命周期保持原合同。Mojo 的 group panel 预算检查使用同一上界查询。配置容量是分块上限，不能代替当前 IR 已有的更紧界；有效窗口宽度也不能代替实现要求的固定 panel pitch。Weft 当前不请求这类 group preparation，不因此宣称它使用了同一 packing 路径。

候选组合先为每个 contraction 找到合法且无需外围 preparation 的基准，再将每种注册实现应用于它能服务的计算，其余计算保持各自基准，按完整 bindings 去重。这样同一函数中的低精度 contraction 可以选择 widened 供数，另一个连续 f32 contraction 同时选择直接读取；不会因二者实现名不同而把后者改回默认 packing。需要准备供数的实现仍参与有限 portfolio，最终 winner 由实际调优决定，不展开每个 computation 的笛卡尔积，也不把这个基准当成布局或复用代价模型。

共同 [BufferStorageAnalysis](include/Intent/Analysis/BufferStorage.h) 直接使用 MLIR 的
`BufferViewFlowAnalysis`、`BufferOriginAnalysis`、AliasAnalysis 和 dominance。
CPU [StorageAnalysis](include/Intent/Dialect/CPU/Analysis/Storage.h) 只补充 task、
collective 与公共 ABI 的合同；DSA [StorageAnalysis](include/Intent/Dialect/DSA/Analysis/Storage.h)
补充借用参数和异步完成查询。标准 views、SCF、CFG 和 select 的来源由其原生模型解释；
CPU Tasks 和 collective 在 [CPUOpInterfaces.cpp](lib/Dialect/CPU/IR/CPUOpInterfaces.cpp)
及 [CollectiveHelpers.cpp](lib/Dialect/CPU/IR/CollectiveHelpers.cpp) 实现
`BufferViewFlowOpInterface`，声明真实 source/capture 借用关系。
Helper 的 summary/state/destination 类型原型不构成存储连边；已知 formal 可以是当前
region 内的来源边界，但不因此成为 fresh allocation，也不保证不同 formal 互不 alias。
`uniqueOrigin` 无法证明单一来源时返回空值，不能用两个空值相等推断共享存储。

`effects` 保留产生 effect 的 operation、实际目标和完整性，遵循标准
`MemoryEffectOpInterface` 与 recursive trait。Linalg 的 payload effects 也参与分析；
collective 在实际 operands 边界描述 effects，内部 scratch 不冒充外部访问。
原子操作同时保留目标读写与 ordering 屏障。新增 operation 应完善接口，避免在 fusion、
任务分区和 provider 中分别增加访问枚举。`PhysicalProgramAnalysis` 保留 allocation
预算与 verifier，写入来源同样从共同 effects 查询取得。
`registerBufferStorageInterfaces` 为 MLIR 20 的 `memref.prefetch` 补原生 effects 模型：
目标只读，独立 cache resource 保留提示的存在性；cache 写意向不作为 buffer 内容写入。
Storage 摘要仅按明确的 resource 合同分离非内容 effects，不静默丢弃未知 effects。
DSA 的 transfer/order resource 保留为移动屏障；它不冒充所有 buffer 的内容写入。

跨原生入口的同步 buffer 借用使用 [CPU `InvokeOp`](include/Intent/Dialect/CPU/IR/CPUOps.td)：
callee、实际参数及逐参数读写是当前 IR 的一部分，调用通过显式 storage 返回结果，
不保留或释放调用方的指针。SymbolUser 校验外部声明的完整签名；MemoryEffects 和
BufferViewFlow 接口使普通 lifetime 与 effect 查询直接理解这个边界。未知 `func.call`
仍不能被当成不逃逸的调用，声明上的 `cpu.external_runtime` 标记也不提供借用保证。
Weft 从最终 kernel 接口通过 `taskCallAccesses` 形成调用，并由 typed task binding
同时核对 callee、参数与 access；host serializer 只拼写调用。Mojo 使用同一操作的
外部 C ABI 拼写，普通浮点环境管理调用保持自身的 effects 和返回约定。

`unchangedBetween` 检查同 block 两个操作之间的开区间，用于读取 producer 完成后的
buffer 内容；`readStable` 还检查 producer 与 consumer 的执行范围，用于移动或重放读取。
两者都检查 dominance、lifetime 和可能 alias 的写入。`isReadOnly` 表示参数访问角色，
不能替代存储稳定性证明。常量传播按可能 alias 的 writes/free 使旧事实失效，不能把
不同 SSA 或不同来源名字当作不相交。

`accesses(memory)` 返回当前 alias 闭包对应、按真实操作去重的 effect 集合。
`preservesContents` 只证明内容不被写入或释放，`preserves` 还拒绝跨 ordering 移动；
根据变换实际需要选择，不能用“没有写入”代替读取移动的完整资格。
DSA 的 `writers`、`lastWriterBefore`、`allUsesCompleteBefore` 从这些事实派生。
最近 writer 查询跨循环时检查 backedge 内容稳定性，存储消费还区分静态末次使用和
循环中的动态末次使用。需要整块覆盖时继续查询 [Views.h](include/Intent/Dialect/DSA/IR/Views.h)，
单一 allocation 来源本身不证明两个 view 的坐标或覆盖范围相同。

来源或 alias 集合的 `complete=false` 保留未知转发、调用逃逸和地址身份观察。
`lifetime` 是供局部优化使用的严格查询：要求一个明确 lexical end 覆盖完整 alias 使用。
整个程序允许标准 SCF 传递存储及原生 ownership 条件，不强迫它们退化为词法结果槽。
[Ownership.cpp](lib/Dialect/CPU/Analysis/Ownership.cpp) 检查当前释放操作的来源、资源连接和
确定的词法 use-after-free；复杂控制流的 RAII 变换由 MLIR ownership pass 负责，
这个结构检查不另行声称证明每条动态路径。简单释放尽早成为 `memref.dealloc`，
其余 `bufferization.dealloc` 保留条件、别名去重及 retained buffers，直到 provider 的
`lowerOwnership` 用标准 conversion/inliner 消费。释放的 effects 通过统一接口可见，
不在每个优化消费者中分别加白名单。
统一 [Registration.cpp](lib/Compiler/Registration.cpp) 注册 MLIR 的 Func inliner
extension，使 ownership conversion 产生的局部 helper 能由原生 inliner 展开到
当前 entry。MLIR 20 的 `Dialect/Func/Extensions/InlinerExtension.h:8–22` 将这项
接口与 dialect 注册分开；仅安装 `FuncDialect` 和 inliner pass 不足以启用它。
移动、替换或删除相关 operations 后重建分析。多个只读查询可复用同一 snapshot，
不能跨实际 rewrite 缓存它。Read 的借用决定在完整值程序上形成，再交原生
bufferization 消费，避免在构造到一半的控制流中启动整个函数的存储分析。
对应成熟机制见 MLIR 20 的 `Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h:17–111`；
本地 Triton `lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp:22–52` 同样从
operation effects 证明读取移动。Intent CPU 还要证明显式 allocation/free 和 helper
调用边界，实际循环与物化改写继续由各 CPU pass 决定。

[IntegerSources.cpp](lib/Dialect/CPU/Transforms/Structure/IntegerSources.cpp) 在破坏性存储复用之前，将完整 pointwise 整数 producer 的读取替换为当前位置上的标量计算，保留位宽并证明输入快照稳定。[ContiguousAccesses.cpp](lib/Dialect/CPU/Transforms/Structure/ContiguousAccesses.cpp) 随后组合实际坐标与静态 strides：完整遍历的地址若等于同形状连续成员加固定基址，就形成标准 memref view/copy，交给既有输出转发与扫描实现。仿射证明同时检查原表达式及重排后算术的范围；未知 stride、无法证明的溢出或读写干扰保留原程序。两者是 `fuseStructuredComputations` 的相邻私有机制，不是新 scan 算法，也不让调用方手工拼装 pass 次序。

输出转发也使用这份存储查询，并保留目标的 disjoint、dominance 和 effect 检查。identity layout 与显式静态 strides 若具有相同 shape、元素类型、memory space、offset 和 strides，可通过标准 `memref.cast` 保持派生 view 的输入类型；两个未知动态 strides 不构成等价证明。这样，unit-axis 视图等正常 lowering 结构不会仅因类型拼写不同而强制保留中间结果拷贝。

Mojo 的 [Passes.cpp](lib/Target/Mojo/Transforms/Passes.cpp) 调度实现展开、私有计算融合、向量化和最终原生合法化，具体阶段在相邻 [Legalize.cpp](lib/Target/Mojo/Transforms/Legalize.cpp)。[Implementations.cpp](lib/Target/Mojo/Transforms/Implementations.cpp) 解释向量与 scan 参数，并将后续循环需要的 `mojo.vector` binding 写到新建循环；`vectorize` 回调消费当前循环的 binding。辅助 copy/fill 和普通作者循环使用已正式选择的函数 implementation，不从 `intent_cpu.implementations` 摘要的首项推断执行策略。循环融合要求 binding 相容并保留它；共同 materializer 只接收明确的宽度，Weft host 显式请求标量展开。Scratch 提升复用 CPU 的存储证明；算术、原子更新和浮点环境在最终 surface 验证前闭合。Weft device 保留 Canonical Weft IR 的 structured 输入边界，不经过 Mojo 的 SIMD 展开。

Mojo 并行任务的 [serializer](lib/Target/Mojo/Serialization/Serializer.cpp) 从当前 task region 的实际外部 SSA 使用形成闭包捕获；memref 捕获完整 descriptor。它不维护另一份“当前可见变量”名单，避免捕获无关或尚未赋值的控制流结果。闭包只改变源码绑定，不改变 task 的执行范围与数据依赖。

职责对照：本地 Triton `include/triton/Dialect/Triton/IR/TritonOps.td:214–239` 将 load 的 memory effect 与 tensor result 同时保存在 IR；`third_party/nvidia/backend/compiler.py:273–285` 在 TTIR 进行 combine/canonicalize，`:415–422` 才进入目标资源分配。Intent CPU 同样先保存值依赖，再决定存储；具体复用的是 MLIR 的 CPU bufferization，而非套用 Triton 的 GPU layout。MLIR 20 `BufferizationOps.td:414–423` 的 restrict 合同不能从“输入只读”推出，故输入读取由上述明确边界建模。

Mojo 最终合法化先验证完整 native surface，再清除已由具体循环、向量和资源兑现的逐操作 implementation binding；实现摘要继续保留供产物检查。随后通过 [FinalizedCandidates.h](include/Intent/Dialect/CPU/Transforms/Configuration/FinalizedCandidates.h) 删除结构完全相同的候选。比较保留完整 ABI、类型、SSA、嵌套任务、effects 和数值属性，仅忽略位置、顶层 entry 名字及已经消费完的配置/实现摘要；保留 profile 顺序中的第一个代表，serializer 和 runtime 从剩余函数形成源码与候选集合。这个入口不能用于尚未消费向量化或分块参数的程序，也不按生成源码文本或算子名字合并。Weft 已将 task 分离到另一个模块，不能只比较 host、忽略 callee 名字后套用此入口。

CPU 归约的相邻重结合与元素重排许可统一保存于 `ReductionOrderAttr`。Construction 从源操作合同建立许可，fusion 取参与计算的许可交集，partition 与 materialization 保留它；不能由末尾恰好有一个 add 推断整个计算可重排。[VectorizeLoops.cpp](lib/Dialect/CPU/Transforms/Vector/VectorizeLoops.cpp) 在访问独立且允许重排时跨块保留向量累加器，最后才做横向归约；初始 accumulator 只合入一次。公共 [VectorReductions.h](include/Intent/Dialect/CPU/Transforms/Vector/VectorReductions.h) 为该路径和多输出归约提供同一套 tuple 横向树，调用者负责初始化、captures 与顺序合法性。Weft 直接消费自己的原生归约能力，不经过这一 SIMD 展开。

Mojo 的矩阵 `formTile` 同样把该许可传给微块，后续实现不能仅因识别到 FMA 就扩大重排许可。矩阵寄存器组织属于 Mojo 原生实现，CPU source 识别和 GPU provider 不承担该 SIMD 决策。

Weft 的私有 [Views.h](lib/Target/Weft/Transforms/Views.h) 证明标准 memref 描述符是否仅做轴置换或 unit 轴插删，并将纯 view capture 的定义链显式放回 task 内。原存储及所需标量进入 task ABI；[TaskViews.cpp](lib/Target/Weft/Transforms/TaskViews.cpp) 将逻辑访问反投影到原 Slice/Subview，[TaskStorage.cpp](lib/Target/Weft/Transforms/TaskStorage.cpp) 缓存原存储顺序的 Admit 快照。矩阵消费者保留该顺序，将轴重命名为当前循环轴，直接交给按命名轴归约的 OuterContract；位置相关的普通读写则显式投影到对应逻辑顺序。不能把非连续 capture 直接标成连续，也不能只改 shape 冒充转置。当前 Weft RISC-V 不能实现一般置换 Reshape；动态轴合并、非矩形 flatten 和任意 strided reinterpretation 也不在该桥接能力内，失败明确报告，不插入隐藏 copy。

View 大小相等与动态 shape 来源统一查询 [ExtentRelations.h](include/Intent/Dialect/CPU/Analysis/ExtentRelations.h)，不在 Weft 再递归解读 Dim、Subview、allocation 和 task captures。Weft 仍负责 offset/stride 与轴投影资格，以及将查询结果映射到真实 native shape：本地 shape 可使用已证明常量，动态 host descriptor 则仍绑定已有公共 shape 参数，其余表达式不能凭相等证明获得新的运行时符号。存储轴来源相同不代表切片长度相等，不能以 `AxisRelations` 代替 extent 查询。

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
[Legalize.cpp](lib/Target/Weft/Transforms/Legalize.cpp)。Task 内转换使用相邻私有模块：

| 模块 | 职责 |
|---|---|
| [TaskLowering.cpp](lib/Target/Weft/Transforms/TaskLowering.cpp) | task/kernel 构造、operation driver 与最终接口提交 |
| [TaskViews.cpp](lib/Target/Weft/Transforms/TaskViews.cpp) | 原存储 view、命名轴与 extent 投影 |
| [TaskStorage.cpp](lib/Target/Weft/Transforms/TaskStorage.cpp) | 局部值、读写、窗口物化与输出提交 |
| [TaskControl.cpp](lib/Target/Weft/Transforms/TaskControl.cpp) | if/for/while 的 aliases、迭代状态和结果交接 |
| [TaskComputations.cpp](lib/Target/Weft/Transforms/TaskComputations.cpp) | generic/reduce/scalar 转换及已选 implementation 的 expansion context |

这些文件共享 [TaskConversion.h](lib/Target/Weft/Transforms/TaskConversion.h) 声明的一份状态，
没有第二份值映射或独立调度顺序。标准标量运算及命名轴值域转换在相邻 `ScalarValues`。
量化实现从 [Quantization.cpp](lib/Target/Weft/Transforms/Quantization.cpp) 自己取得所需
view 并写回 destination；新增 native implementation 不再修改 driver 的量化类型分派。
转换状态属于单个 CPU function；存储分析在每个 task lowering
开始时重建，因为此前的 task 可能已经移除。Task ABI 的读写与 encoded 格式查询共同
存储事实；只有当前 task 内的 allocation 才能成为其局部 SSA，捕获的外层 allocation
仍通过 view ABI 访问。Admit 快照复用需要证明整个 task 保持该存储，不仅检查只读角色。
这些决定不由 serializer 重建，也不跨程序保留。

## Provider 与 runtime 扩展

先比较 Intent operation 与目标原语的合同，包括 dtype、accumulator、NaN/tie、顺序、effects 和 ABI。合同吻合时优先直接映射；例如 provider 已有 reduce/scan，就不在 Intent 再实现其线程通信与归约树。

- 同一 execution family 的新 provider，先复用已有 physical program 和 family passes。
- 新设备代际优先通过已有 capability consumers 和 legality predicates 表达；不因设备代号不同就建立一套 dialect。
- 只有共同 IR 确实缺少、且多个 consumers 或独立 verifier 需要的目标结构，才增加 local extension。
- `lib/Target/<provider>/Serialization/` 打印已经决定的 kernel 程序和接口 facts，不重新选择 ownership、workspace、pipeline 或调优参数，也不生成另一套通用 host 参数检查和输出分配代码。
- `python/intent/runtime/` 消费这些 facts 并执行；provider 适配实际原生调用，公共 binder 处理参数、输出与工作区。无法兑现的能力明确报错，不改变算法或隐藏失败。

CPU 的 implementation registry 是明确的局部扩展点。职责参考是本地 Triton `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:212–243`：driver 注册对应操作的 lowering，具体实现消费转换上下文。Intent 的 [Implementation.h](include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h) 还负责 CPU 有限候选、供数要求及已选绑定的后续展开；这不是让 leaf 重新选择完整算法。GPU provider 通常复用下层 compiler 的 primitives 与布局机制，不需要为了目录形式对称再建一套同名 leaf 系统。

### GPU 接口与 Python 产品边界

共同的 [Serialization/Interface.h](include/Intent/Dialect/GPU/Serialization/Interface.h) / [Interface.cpp](lib/Dialect/GPU/Serialization/Interface.cpp) 从最终 GPU IR 读取 ABI 顺序、view/scalar、shape/stride、工作区、overlap、grid、候选及资源约束。Provider 只补充自己的原生参数顺序、descriptor/array 形式和编译选项。Metadata 是当前程序的序列化结果，不是 runtime 再选物理结构的计划；新增执行事实仍应先在 IR 与相应变换中成立。

| 修改目标 | 入口 | 应保持的边界 |
|---|---|---|
| 公共参数、输出与 alias | [invocation.py](python/intent/runtime/invocation.py)、[torch_views.py](python/intent/runtime/torch_views.py) | GPU、native 和 fake 消费同一 public binder；Torch observer 供 GPU/Mojo 复用，不缓存可变 tensor facts |
| GPU 参数与 workspace | [gpu/interface.py](python/intent/runtime/gpu/interface.py) | 将公共实参绑定到当前 GPU 参数身份，按已导出的 host dependencies 形成 metadata、coverage 和 workspace |
| 已导出的整数表达式 | [gpu/expressions.py](python/intent/runtime/gpu/expressions.py) | 只求值 compiler 已声明的表达式，不按算法名或观察到的 shape 发明策略 |
| 候选、deferred coverage 与资源条件 | [gpu/configurations.py](python/intent/runtime/gpu/configurations.py) | 唯一解析已导出的候选表；provider 明确选择用于执行或展示的现有行，不再重建第二份配置 |
| 调用生命周期与原生结果 | [gpu/program.py](python/intent/runtime/gpu/program.py) | `GPUProgram` 共用 run/launch/prepare；`PreparedCall` 属于已绑定的实参和 workspace，改变参数或元数据时重新 prepare |
| Provider 的 JIT、调优和发射 | [runtime/triton/program.py](python/intent/runtime/triton/program.py)、[runtime/cutile/program.py](python/intent/runtime/cutile/program.py) | 消费 `BoundInvocation` 与本 provider 的已解析合同，返回 `LaunchResult`；保留各下层 compiler/tuner 的职责，复用公共 trial-state 规则 |
| 原生资源与候选观察 | [runtime/diagnostics.py](python/intent/runtime/diagnostics.py) | 不可变 `NativeObservation` 保存实际调用、已选配置、SDK 返回值与失败；不持有 tensor，不参与候选策略 |
| PyTorch operator 注册 | [runtime/torch.py](python/intent/runtime/torch.py) | `as_torch_op` 按 `PublicInterface` 与 `TorchOutputInference` 注册 CPU/CUDA opaque 调用；InOut 进入 mutation schema，Out 为 fresh 返回；functional backward 由作者注册 |
| 安装与依赖说明 | [tools/backends.py](python/intent/tools/backends.py)、[environment/install.py](environment/install.py) | 新安装路线声明实际依赖和外部工具链要求；不把实验私有环境或 baseline 包当作公共 runtime 依赖 |

新增 GPU provider 时，先让 legalization 交付可独立验证的当前程序，再导出共同 interface 与必要 provider facts，实现上述 provider 调用接口，并由 `ResolvedTarget.materialize` 接入。不要复制 serializer 中的 Python host 模板，也不要让 framework adapter 自己猜输出或解析生成源码。CPU、DSA 可以保留自己的 ABI/buffer 类型；共同 `ArtifactRuntime` 协议不要求它们采用 GPU 的 grid、workspace 或 tensor binder。

GPU 的纯编译事实位于 [targets/specification.py](python/intent/targets/specification.py)，设备观察位于 [gpu/device.py](python/intent/targets/gpu/device.py)，本机绑定与 materialization 共用 [gpu/target.py](python/intent/targets/gpu/target.py)。两个公开本机 Target 只声明 provider；`ResolvedGPUTarget` 分别保存 compilation 与 device，provider 表只连接实际 materializer。显式 compilation target 不经过设备观察或 SDK import。

GPU provider 的 `contract.py`、`program.py`、`math.py` 在同一目录；cuTile 的
`compilation.py` 也归该 provider。源码所引用的辅助函数与 native SDK 调用随
provider 维护，共同 `runtime/gpu/` 只负责 host 参数依赖、候选和调用生命周期。
SDK 更换时先定位源码拼写、纯产物字段或 native 调用哪一层变化，修改所属模块；
不要在共同 ABI 或无关 provider 中加入版本猜测、默认字段或失败回退。

实验适配若需要准备候选、观察调优或绑定调用，使用 `artifact.runtime` 的明确对象与 provider 扩展点。Compiler 生成的 source 不再承担 host `launch/run` 协议；只有显式作者提供的 Python source 由 [runtime/source.py](python/intent/runtime/source.py) 的独立 source loader 承接其已有 host callable。不要通过生成模块的私有字典改写编译器产物的执行语义。

Mojo、Weft、BANG C 的 [NativeABI](python/intent/runtime/native.py) 先消费公共绑定结果，
再按 `native.slots` 形成 pointer、extent、stride 和 scalar carrier。公共 binder 与
native packing 都按一次声明生成，prepared launch 不重新遍历公共 schema。
`ViewFacts` 只属于本次观察：shape/stride、访问字节范围与 allocation 身份是不同事实。
同名 alias group 要求同 allocation，允许不同 offset/shape 和不重叠子视图；
`noalias` 检查 allocation 范围，不能与后端的写入重叠限制混为一条规则。

`BindingRelations.require_output_allocation` 统一判断省略 `Out` 的调用是否成立：
shape 必须由输入或公共声明中的静态维度确定；涉及 `Out` 的同 allocation 关系
需要调用方显式提供 buffers。普通 run、prepare、fake 与 PyTorch 注册消费同一判断，
不因此拒绝只能显式调用的产物。默认输出仍使用 runtime 的独立分配布局；stride
关系依据真实输出检查，不能满足时提供显式 outputs，不猜测共享 storage 或 offset。

原生入口要求由 [queryNativeEntryRequirements](include/Intent/Serialization/NativeABI.h)
读取当前 family 的 typed IR，导出 `native.requirements` 中逐 view 的 layout/alignment
及 disjoint 参数对。[NativeRequirements](python/intent/runtime/native_requirements.py)
是 Python 的唯一解释器，`NativeABI.read` 同时检查 slots 和 requirements，保存、加载
与 materialize 都消费这份事实。修改后端支持范围时先改 IR 的合法性与导出，不能只删
runtime 检查，也不能从 provider 名字推断所有输入必须 contiguous。

Native 调优保存范围由 [NativeABI.trial_regions](python/intent/runtime/native.py) 从
InOut 的实际 pointer、extent slots 和物理 pointee 字节宽度导出。Mojo 与 Weft 的
native benchmark wrappers 共用这份范围事实，保留各自语言的计时与内存拼写；每次
repetition 在计时前恢复输入，计时一次完整入口，完成后恢复调用方状态。当前 CPU
入口只允许只读 In 使用 strided layout，InOut 的连续性来自已声明的入口合同。

Weft 的 `prepare` 只绑定 storage，不缓存其中的数据；`choose` 和 `benchmark`
开始时才捕获当前 InOut 内容，并在正常或异常退出时恢复。这样，同一 prepared call
之前的真实更新或 prepare 后的调用方更新不会被旧快照覆盖。只读 view 不清零，
accepted alias 关系和原始 pointers 保持不变；显式 `prepare` 测量回调也在这次状态
恢复边界内。候选集合、试跑次数与选优策略仍归各 provider，不能用 snapshot helper
改变它们。
入口要求的 `check_geometry` 只检查 shape/stride，Mojo fake 通过 `torch._check` 复用；
`check_storage` 与 `check_pair` 检查真实地址及跨度，只用于 concrete binding。
Mojo 的空指针 ABI 限制、BANG C 设备/队列/整行 tile 资格和 Weft 的调用线程
affinity、stack、RVV/VLEN 检查继续留在各自 runtime。

`artifact.run(*inputs)` 与 prepared `result()` 只返回 `Out`：零个为 `None`，一个为
单值，多个为声明顺序的 tuple；`InOut` 通过原对象观察。`artifact.launch(...)`、
`artifact(...)` 与 prepared `launch()` 执行后返回 `None`。内部 `LaunchResult` 的
native kernel 仍用于后端 IR 收集，不成为公共调用结果。
`artifact.prepare(*inputs, outputs=(...))` 可显式绑定 Out 并复用调用。GPU 使用当前
stream，CPU 等待本次任务，BANG C 同步自己的 CNRT queue；`result()` 不隐含同步。
参数或其 shape/stride 改变时重新 prepare。enqueue、benchmark 等扩展仍归具体 provider。

`prepared.compile()` 编译本次绑定下具备资格的 native 候选并返回 `None`，不调优、
不执行 kernel，也不建立 prepared replay。GPU 将逐候选编译结果记入 observation；
个别候选编译失败可保留为失败记录，全部失败则报告 native compilation 错误。
后续 `launch()` 仍执行原候选选择与调用路径。Mojo 与 BANG C 的 materialization
只绑定程序；`prepare` 绑定实参，`compile` 编译 portfolio 或固定入口，首次调用
才加载库、绑定 symbols 并建立实际执行资源。Weft 的系统编译由显式 AOT 步骤完成，
prepared `compile` 验证已有原生产物，加载同样延迟到调用。编译本身不创建 CNRT queue，
也不运行 RVV 设备指令探测；参数分配和绑定仍遵守实际设备要求。
编译成功不代表数值正确或性能达标，`result()` 此时也只返回已绑定的输出容器。

Mojo、Weft 和 BANG C 复用 [native_artifact.py](python/intent/runtime/native_artifact.py)
的 `NativeArtifact`、`build_native_artifact`、`run_native_command` 与
`LoadedNativeLibrary`。Provider driver 只交付真实编译命令、文件与依赖；公共层复用
已有 cache entry/attempt 机制，编译完整后发布不可变产物，失败保留输入和日志。
`NativeArtifact` 不持有动态库或设备句柄；只有 loader 才加载库并按已声明 ABI 绑定
symbols。同一二进制的 process image 可以共享，每个 runtime 有独立使用权；
关闭一个 runtime 不卸载其它 runtime 的库，也不改写正在使用的二进制文件。

SDK dependency closure 由实际 driver 声明。Mojo 保留工具链检查、linker inputs
与实际动态库解析路径核对；CNCC 和任意 Weft C 编译命令尚未提供完整闭包，
因此每次新编译请求真实调用 compiler，同一程序正常复用自己的已编译产物。
不把有限几个 SDK 文件的 stat 冒充完整缓存身份，也不把磁盘产物读回冒充观察到 cache hit。

Weft 的 [WeftArtifact](python/intent/runtime/weft/compilation.py) 在一次流程中携带
已解析的 source、公共合同、native task facts 与目标 profile，贯通 export、system
compile 和 `NativeProgram`。`save/read` 是跨进程或跨机器的目录边界；系统编译返回
携带本机 `NativeArtifact` 的对象，不再覆盖导出目录中的 `kernel.so`。源码目录可以
搬到目标机器重新编译，已加载的本机产物始终留在独立 attempt 中。

职责对照：本地 Triton `python/triton/compiler/compiler.py:407–438` 先保存编译产物和
metadata，`:452–488` 再创建实际 runtime handles 并检查设备资源。Intent 的共同
native owner 负责文件和库生命周期，CNRT queue、RVV/VLEN 资格和各 provider 的
tuner 仍由相邻 runtime 负责，不把这些执行模型差异塞进公共 loader。

Triton 的 [program.py](python/intent/runtime/triton/program.py) 对各 eligible config
调用底层 JIT function 的 `warmup`，保留本次参数、coverage、grid 与 native options；
正式 launch 继续使用 SDK autotuner 和原试跑状态边界。本地参考
`python/triton/runtime/jit.py:359–371` 明确区分 `warmup=True` 与下标调用的实际运行。
cuTile 的 [CuTileCompilation](python/intent/runtime/cutile/compilation.py) 由 program
拥有 kernel 变体及 native 编译结果；显式 compile 根据实际 signature、hints、架构和
context 调用 SDK compiler，普通 JIT 的 kernel `_compile` 回调消费同一份结果。
这个接入仍依赖 SDK 的私有编译接口；它不替换全局 `ct.launch` 或 `ct.kernel._compile`。
SDK `exhaustive_search`、试跑和 winner cache 仍由原
[CuTileProgram.launch](python/intent/runtime/cutile/program.py) 路径管理，
不会把编译候选当作已选 winner。

`prepared.inspect_configurations()` 直接使用本次绑定，返回每个声明候选的
`ConfigurationAssessment`，不重新绑定、分配、编译、选优或计时。GPU 的 requirements
及 Triton descriptor 实参资格与实际候选筛选共用一条判定路径；已知拒绝优先于未知
条件，状态为 `eligible`、`rejected` 或 `unknown`，不把资格判断当作原生编译成功。
Mojo/Weft 返回真实 entry、配置值与 implementation portfolio；BANG C 没有运行时
候选搜索，返回空 tuple。`artifact.inspect_configurations(*inputs, outputs=(...))`
是先 prepare 的便利入口，因而可能分配 Out；现有
`artifact.tuning_configurations(*all_arguments)` 仍要求完整实参（含 Out），用于读取
可用 GPU 配置投影。

实际调用后读取 `artifact.observation` 或 `prepared.observation`，得到同一份
`NativeObservation`；读取不触发 JIT 或 launch。快照包含各 family 的真实 target
facts、本次实参 shape/dtype/stride、已选配置和已有 tuner 返回的候选状态。候选
`elapsed_ms` 只取现有 `provider_tuning` 测量，不作为完整算子 benchmark 时间。
GPU 候选历史、资格查询与已选配置均包含本次调用的 coverage 绑定；这不改变 SDK
内部的候选 identity 或调优 key。目标和 native portfolio 的固定描述在程序创建时冻结，
每次调用只补实际实参、选择与执行事实。
CPU 的配置保留原生 portfolio 结构，不套用 GPU 参数角色，也不声明 GPU 寄存器或
shared-memory 计数。Mojo/Weft 的选择、launch 和 benchmark 分别记录
`selected`、`launched`、`benchmarked` 阶段；BANG C enqueue 记录 `submitted`。

`CacheObservation` 按 layer、scope、stage 标明事实来源：产物加载与候选选择的记录
保留其原阶段，Prepared 首次重放增加本次调用复用的 `prepared_call`/`dispatch`
记录，后续重放复用该快照。该记录不推断 SDK 是否命中缓存；Triton 重放仍可能进入
SDK dispatcher。快照不持有 tensor，可用 `dataclasses.asdict` 保存 JSON；保存和
加载 GeneratedProgram 只处理编译产物，不序列化这些运行观察或选优缓存。

Triton 读取 loaded kernel 的寄存器、local-memory words、shared memory 与线程限制。
`n_spills` 在当前 NVIDIA driver 中是每线程 local-memory bytes 除以四，不能称为
“溢出的寄存器数”。cuTile 当前公开编译结果不提供这些资源字段，观察保留
unavailable reason。Mojo native artifact 命中取实际库加载结果，Weft 外部 AOT
构建和 BANG C 编译若没有公开命中记录则保持未知。SDK 未返回完整失败历史或无法
区分磁盘缓存命中时明确保留未知，不重跑 tuner 补造记录。失败快照附在
`CompilationStageError.observation`，CLI/MCP 错误响应导出为 `native_observation`。

职责参考：本地 Triton `python/triton/compiler/compiler.py:468–484` 在 native loading
检查真实 shared-memory 上限并接收 driver 的资源数；Intent 的结构预算发生在 GPU IR
候选形成阶段。两者依据与时机不同。SDK 的缓存属于下层，Intent 自己没有缓存记录
不能证明 SDK 未命中。

PyTorch fake 通过 runtime 的 `infer_outputs` 调用同一个公共 binder，使用 symbolic
关系断言，既不读取 data pointer，也不创建 GPU workspace 或执行 native code。
`as_torch_op` 从实际输入位置生成 `Tensor(aN!)` 与 `mutates_args`，保持 `run` 的
返回合同：只有 `InOut` 的 kernel 返回 `None`，混合 `InOut/Out` 只返回新分配的
`Out`。该 adapter 的 mutable 调用要求被写入输入与其它输入不共享 Torch storage；
concrete 与 fake 使用同一 storage-identity 查询检查，不要求 source 注解
`noalias=True`，只读输入之间仍可 alias。该限制属于 PyTorch functionalization
接入；普通 artifact 调用的 alias 合同不变。Torch 本身无法识别的、由不同外部
StorageImpl 包装的同一地址不在此桥接保证内。
PyTorch 自动处理声明 mutation 的版本计数与图内 functionalization，Intent 不重建
这一过程。Mutable custom operator 不支持 `register_autograd`；优化器/显式状态更新
与需要作者 backward 的 functional operator 分开注册，遵守 PyTorch 的 grad-mode
规则，不由 adapter 隐式切换 `no_grad`。
[softmax_forward_backward.py](examples/softmax_forward_backward.py) 展示两个已编译
artifacts 注册为 forward/backward operators，作者通过 `register_autograd` 保存并
传递张量；同一份 host 组织可选择 Triton、cuTile 或 Mojo CPU。
这对应本地 Triton `python/tutorials/05-layer-norm.py:233–294` 的职责：host wrapper
保存上下文并调用作者 backward，compiler 不自动发明梯度算法。Intent 的 adapter
负责 dispatcher/fake 接口，不承担这份算法与多 kernel 编排。

原生编译与产物寿命由 provider runtime 负责。Mojo 的 [compilation.py](python/intent/runtime/mojo/compilation.py) 组织候选和加载，[toolchain.py](python/intent/runtime/mojo/toolchain.py) 查询已支持工具链的依赖身份，公共 [compiler/cache.py](python/intent/compiler/cache.py) 提供输入比较、锁与发布机制。新增 SDK import 时同步 serializer 的 `native_dependencies`；无法证明依赖闭合时继续原编译并说明缓存不可复用原因。失败不发布成功产物，已加载的库不原位覆写；这些机制不改变算法、候选或算子计时范围。

Triton 的 [Passes.cpp](lib/Target/Triton/Transforms/Passes.cpp) 调度 grid、prepare-memory、native-forms 和 finalize；[Legalize.cpp](lib/Target/Triton/Transforms/Legalize.cpp) 通过私有 [Program.h](lib/Target/Triton/Transforms/Program.h) 组合完整阶段。cuTile 的 [Passes.cpp](lib/Target/CuTile/Transforms/Passes.cpp) 调度 prepare、native-program 和 finalize，私有入口在 [Program.h](lib/Target/CuTile/Transforms/Program.h)。按实际职责选择相邻模块，不把新增规则继续堆入 driver：

| Provider | 模块 | 职责 |
|---|---|---|
| Triton | [AccessForms.cpp](lib/Target/Triton/Transforms/Access/AccessForms.cpp) | 从当前 access facts 形成 tensor descriptor 或普通 pointer 访问；对齐与坐标投影复用 GPU 分析 |
| Triton | [Access/](lib/Target/Triton/Transforms/Access/) | gather 的规范化、实现选择、临时存储及 scatter；共用组内的资格与构造接口 |
| Triton | [Collective/](lib/Target/Triton/Transforms/Collective/) | 原生 reduce/scan callback 与 scan tail，各自保留完整物化过程 |
| Triton | [Supply.cpp](lib/Target/Triton/Transforms/Supply/Supply.cpp) | ordered access dependencies、CTA 同步与 load-loop policy |
| Triton | [Values.cpp](lib/Target/Triton/Transforms/Value/Values.cpp) | 形成 contract/value 表示；只读 contraction 约束位于 [Analysis/Contractions.h](include/Intent/Target/Triton/Analysis/Contractions.h) |
| cuTile | [NativeProgram.cpp](lib/Target/CuTile/Transforms/NativeProgram.cpp)、[NativeRewrite.h](lib/Target/CuTile/Transforms/NativeRewrite.h) | 收集本次输入、按实际访问声明 access/load 参数并统一提交 native replacements；原 GPU SSA 保留到相关 facts 消费完成 |
| cuTile | [Configuration/Configurations.cpp](lib/Target/CuTile/Transforms/Configuration/Configurations.cpp) | 在 workspace lowering 前声明 launch 参数并稳定 resident bindings；同一 launch tuple 枚举服务准备和最终候选形成 |
| cuTile | [Access/TilePlan.cpp](lib/Target/CuTile/Transforms/Access/TilePlan.cpp)、同组 `Bounds` / `Coordinates` | 只读 tile 资格、对齐与坐标分解 |
| cuTile | [Access/Accesses.cpp](lib/Target/CuTile/Transforms/Access/Accesses.cpp) 与 `Loads` / `Extraction` / `Atomics` / `Stores` | 按原次序构造各类访问；`TileIndices` 形成真实坐标与 guard，`AccessFormSelection` 共用一次变换的配置物化状态 |
| cuTile | [CollapseArrayViews.cpp](lib/Target/CuTile/Transforms/Access/CollapseArrayViews.cpp) | 从已有原生 tile load 形成具有真实 collapsed layout 的条件视图，保留原访问语义 |
| cuTile | [ComputeForms.cpp](lib/Target/CuTile/Transforms/Compute/ComputeForms.cpp) | 构造 reduce、scan、histogram 和 MMA primitives |
| cuTile | [Legalize.cpp](lib/Target/CuTile/Transforms/Legalize.cpp)、[Control/Loops.h](include/Intent/Target/CuTile/Transforms/Control/Loops.h) | 前者组合准备和收尾；后者形成有范围证明的原生循环或显式宽整数循环 |
| Triton / cuTile | 各自的 [Triton Program.h](include/Intent/Target/Triton/Analysis/Program.h)、[cuTile Program.h](include/Intent/Target/CuTile/Analysis/Program.h) | 验证当前 provider IR 的类型、形式、候选与资源要求；由调用者传入同一终端 operation registry |
| Triton / cuTile | 各自的 [Triton Numerical.h](include/Intent/Target/Triton/Serialization/Numerical.h)、[cuTile Numerical.h](include/Intent/Target/CuTile/Serialization/Numerical.h) | 同一 typed translation 负责数值合法性、源码表达式和所需依赖，保留 SDK 各自的转换与数学接口 |

这些私有 facts 和待提交 replacements 只服务一次变换；阶段之间传递当前 IR 与其携带的 resolved profiles，不保留另一份执行计划。Triton 的局部候选在 native-forms 阶段内闭合为 IR configs；cuTile 在 prepare 中稳定 allocation 所需的 launch/resident 绑定，native replacements 提交后再完成循环与完整候选验证。

最终 pass 和 serializer 都调用同一个只读 program verifier，验证当前 IR 与终端
operation registry，不把 `legalized` 标记当作验证证书。Analysis 只依赖 IR 与分析库；
registry 由调用者传入，因此检查可以复用而不造成 Serialization/Transforms 循环依赖。
Triton 的 contract form、loop unroll 等真实执行属性仍保存在 IR，公共声明与合法性在
[IR/Program.h](include/Intent/Target/Triton/IR/Program.h)，不能用阶段标记代替。

新增 GPU 数值操作时，修改相应 `Serialization/Numerical.cpp` 的 typed translation，
通过 [Source.h](include/Intent/Serialization/Source.h) 登记检查、发射和依赖。依赖从当前
操作查询后统一去重，不能另在 preamble 维护数学操作名单。转换记录只是一次源码拼写的
结果，不保存执行计划、不选择分块。库精度、NaN、逐操作 approximate/FTZ 和 dtype
转换必须兑现 Intent 合同；例如本地 Triton
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp:245–246`
使用饱和 E5M2 指令，Intent 的普通 E5M2 转换则按 RN-even 溢出到无穷，因此目标转换
显式补齐结果编码，不能直接继承 SDK 默认。普通 cast 与 bitcast 保持独立。

物理表达式的动态整数除法也由该 provider 的 `integerDivision` 拼写：Triton 与普通数值操作共用商/余数修正，cuTile 直接使用原生 floor/ceil division；constexpr 子表达式按 Python 数学除法计算。不能把 Triton 的截零 `//` 当作数学 floor，或用可能中间溢出的 `n + d - 1` 替代 ceil。本地 Triton `python/triton/language/semantic.py:312–322` 与 `standard.py:34–43` 展示了这两个需要区分的合同。

原生坐标构造也须兑现 IR 位宽：Triton 的 `program_id` 和 `arange` 默认产生 i32（本地 `semantic.py:39–42,578–594`），serializer 在后续乘加之前转为当前 IR 的 index/element type；cuTile 的 block id 转换与 arange dtype 同样显式表达该类型。不能用 SDK 对小整数的默认推导代替 Intent 的 index64 合同。

Triton descriptor 的 offsets 在 pass 中显式形成 `i32` SSA，serializer 只拼写已决定的
操作数；shape、stride 和 block shape 通过当前 view layout 与 launch-expression 查询取得。
普通 pointer 的地址仍由 coordinate projection 和实际 resource stride 组成。
两者不再经过 `BlockLoad`/`BlockStore` 或 `make_block_ptr` 路线。职责对照是本地 Triton
`lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp:90–211`：descriptor 与 pointer
之间的转换在 IR/pass 中表达，而不是留给源码打印时猜测。该 ref 的
`python/triton/language/core.py:2444–2464` 已移除 block-pointer API；公开安装的 Triton 3.6 仍支持它，
删除本项目旧路径不等于已完成所有新 SDK 版本的适配。

cuTile `ArrayViewOp` 复用 GPU `ViewType/ViewLayout`，结果直接携带折叠后的 rank、extent
乘积和 stride；单轴组保留 dimension identity，合并轴不伪造逻辑 dimension。
`InferTypeOpInterface` 与 verifier 使用同一类型推导；`ViewLikeOpInterface` 将它和
Triton descriptor 连接到真实 base，使共同 alias analysis 能沿标准接口查询。
Native load/store/atomic 的 memory effects 指向实际 resource，非 relaxed atomic 另外
保留跨资源的 ordering effect。新增 provider handle 应提供相应接口，而不是让共同分析
按 provider 名字或操作名猜别名。

条件 array view 的访问受其 eligibility 控制，full-tile 判断读取当前视图尺寸。
合法性不再依赖 if 内有几个操作、相邻 reshape 或 else 分支里的另一条 load。
索引宽度的 tile bounds 分别附在实际参数和 `ArrayViewOp` 上；serializer 导出当前 native
array、bounds 与 eligibility，[cuTile contract](python/intent/runtime/cutile/contract.py)
解析后由 [program.py](python/intent/runtime/cutile/program.py) 检查实际绑定的数组。
未启用的 alias 不参与该数组的范围检查，原始 base 仍保留自己的检查。
这与 cuTile SDK 的 `_ir/type.py:524–572` 中 `ArrayTy` 用实际 shape、strides 和 index dtype 描述数组一致；
无需从已改变 rank 的访问反推一份原始参数轴表。

对齐推断中的参数域必须是当前证明可依赖的域。Provider preparation 之前的 `ResidentWorkers` 初值不是最终配置事实，公共关系查询不能据此证明常量或整除；coverage capacity 也不等于 logical extent。分支内额外对齐条件由调用方提供局部叶证明，不能传播成其它分支的全局性质。新增整数规则先核对位宽、回绕与除法合同，再接入共同查询，避免在各 provider 重写递归证明。

Triton/cuTile 的 `Transforms/Configuration/Configurations.cpp` 负责候选选择与写回；各自的
`Analysis/Configuration.cpp` 收集和验证当前 IR 的目标约束。共同归约资源查询及要求
验证位于 [Analysis/Resources.h](include/Intent/Dialect/GPU/Analysis/Resources.h)，候选
筛选仍在 Transforms。Triton 的 tensor/descriptor/collective 约束与 cuTile 的
resident-capacity 关系由同一查询同时服务变换和终端验证。新增设备约束进入对应
只读模块，不复制参数解析器或下层的布局、MMA、资源分配器。
cuTile 的 `prepareLaunchConfigurations` 使用唯一的相关 launch tuple 枚举，根据
CTA/occupancy 得到所有最终 resident 值，并将其投影回既有 Shared 行；Provider
bindings 不写进 Shared 行。Workspace 随后才求包络。最终 complete candidates 复用
同一 launch 枚举和原 memory-form 相关性，通过 resident 等式及其它 requirements
筛选，不再改写 resident domain 或扩大此前分配容量所依赖的参数取值。

矩阵 primitive 需要的二维物理投影由现有
[ContractionProjection.cpp](lib/Dialect/GPU/Transforms/Contraction/ContractionProjection.cpp) 依据
typed free/batch/reduction axes 形成，并恢复结果的原坐标映射。Triton 和 cuTile
在各自准备边界共用 `normalizeMatrixContractShapes`；serializer 只输出
已决定的 transpose/reshape。新增 provider 不应重新限制作者只能声明二维矩阵。

Value schema 的闭合先查询 [ValueSchema.h](include/Intent/Dialect/GPU/Analysis/ValueSchema.h)：
`queryElementwiseShapeSource` 描述 unary/cast/bitcast 的逐 lane 等形关系；
`queryStructuredSchemaGroups` 按位置连接 region summary/state 的 producer、seed
operand、helper formals、yield 与 result。`ValueRelations` 的正向刷新与反向 extent
传播、`SchemaMutation` 的边界改写和 value materialization 使用这些关系；普通控制
边使用共同 `Analysis/ControlFlow` 查询。不同 state 分量可复用同一个零值 SSA，仍是
不同 operand slot，不能仅按 SSA 相等合并 schema。这里闭合的是物理表示，不改变
作者的 combine/apply/emit、迭代次序或数值运算。

cuTile 的 [Analysis/Tuning.h](include/Intent/Target/CuTile/Analysis/Tuning.h) 从最终 provider IR 查询哪些 runtime scalar 必须按值区分调优结果。证明覆盖 SSA、类型/属性中的 ScalarABI 以及潜在的写后读依赖；索引、控制、形状、资源和未知用途保持区分，只有完整证明为数据用途时才移除其值。Serializer 消费这份只读结果，并保留 view、overlap、完整覆盖和 array-view eligibility 的实际事实；它不改变 scalar 的原生传参或候选执行。

资源查询的 `Unknown` 表示当前求值无法证明，可能来自未绑定维度，也可能来自表达式求值失败；不能据此宣称候选合法或已精确证明资源不足。Shared 候选策略只按可得事实筛选和绑定，保留需要 specialization 或下层 compiler 判断的约束；局部候选 matcher 也不等同于完整 coverage 证明。

Pointwise blocking、retained gather 和 reduction blocking 共用
`minimumFragmentRegisterFootprint`，统一 dtype word 数、当前参数域最小值和饱和乘积。
调用方明确选择当前 physical shape，或包含 construction scalar seed 的完整逻辑容量，
再按自身策略判断预算；不能把任意非单调表达式的各叶最小值当作表达式下界。
Retained loop/scan 和 provider program scratch 通过 `isProgramAllocationContext`
检查可声明私有 allocation 的控制作用域，终端再形成 program-private slices。
需要单 program 的 retained-store 路径继续用 `isSingletonExecutionGroup` 检查当前
space、segment 与 extents；作用域扩展不能把私有状态变成未经证明的 invocation 共享状态。

BANG C 的 [Storage.cpp](lib/Target/BangC/Transforms/Storage.cpp) 分开只读 `measureStorage` 和最终 `bindStorage`。前者可供局部复用与供数变换比较资源需求，后者才写入目标偏移；别名与 effects 复用共同 BufferStorageAnalysis，DSA 的 [Storage](include/Intent/Dialect/DSA/Analysis/Storage.h) 与 [PhysicalProgram](include/Intent/Dialect/DSA/Analysis/PhysicalProgram.h) 负责异步使用完成和局部存储区间。操作的完成要求统一定义在 [MemoryEffects.h](include/Intent/Dialect/DSA/IR/MemoryEffects.h)：StageTile 等异步操作不能在提交时就结束其 buffer 存活期，缺少可证明完成点时不做破坏性复用。新增目标实现需要的 workspace 在目标变换中形成显式 operand，最终由目标 verifier 检查，不能在资源查询或 serializer 中补写。

这与本地 Triton `lib/Analysis/Alias.cpp:23–61` 通过 view/控制流传播别名、
`lib/Analysis/Allocation.cpp:303–378` 合并别名存活期的职责一致。
TileLang `src/transform/storage_rewrite.cc:105–295` 同样把 allocation scope 和最后访问
作为存储复用依据。Intent 的 DSA 另外保留已有 DMA/group 完成合同，不把 CPU 的同步
生命周期或 GPU provider 的内部 shared-memory 分配规则套进来。

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

可安装分发使用 [environment/build.py](environment/build.py)，基线为 Ubuntu 22.04
x86-64、CPython 3.10–3.12 和 LLVM/MLIR 20。它先生成 sdist，再从该归档构建 wheel，
沿现有 CMake `IntentRuntime` 安装规则收集编译器、优化器、profiles、手册和依赖 notices：

```bash
python3 environment/build.py --output-dir /path/to/distributions \
  --work-dir /path/to/new-build-workspace --jobs 8
```

输出和工作目录必须在 checkout 外；工作目录为本次新建，已有分发文件不覆盖。
可显式选择 `--mlir-dir`、`--llvm-dir`、`--runtime-notices`，或同时提供 Weft 的
`--weft-source-dir` 与 `--weft-binary-dir`。构建后在隔离环境安装 wheel，清除源码和
SDK loader 路径，调用已有 doctor、公开声明、MCP 服务启动与 EOF 退出，以及归档中原 softmax
定义的 KIR 编译和标准 MLIR 优化入口；失败保留工作目录与诊断。它不启动 kernel，
也不证明 provider 的数值与性能。此 recipe 不承诺 manylinux 或逐字节一致；
其他平台仍可手工源构建，实际生产运行继续使用对应实验组的原入口。

[Distribution workflow](.github/workflows/distribution.yml) 在产品源码、构建输入和文档的
PR 变更及手动 dispatch 中调用同一 recipe，使用 hosted Ubuntu 22.04、Python 3.10 和
LLVM/MLIR 20，不连接私有或 self-hosted 设备。Checkout 只读且不保留凭据；同一 PR
的新运行取消旧运行。成功产物为可下载的 sdist/wheel，失败诊断包含 build.py 保存的
逐命令日志、退出码和现有工具输出。下载与安装方式见[安装说明](environment/README.md#download-a-ci-build)。
CI 不安装 GPU SDK、不代替原 production 数值与计时，也不执行公开发布；新增设备验收
仍先使用真实存在的环境和对应 registry，不在分发 workflow 中假定设备 runner。

安装 wheel 后，`intent setup --target BACKEND` 使用包内的依赖声明配置当前 Python
环境；[environment/install.py](environment/install.py) 也调用这个入口。它只安装
Python 依赖，外部编译器和设备由实际 target/doctor 检查。普通用户可先运行
`intent describe --json` 查询实际公开 API，`intent doctor --json`
检查基础 compiler/KIR；选择后端后再用 `--target triton` 等检查相应包、SDK 与设备。
`intent compile path/to/program.py:kernel --target triton --json` 返回真实阶段与编译产物，
`--materialize` 调用同一个 `GeneratedProgram.materialize()`，不会重新 lowering。
工具不主动 launch 所选 kernel，但加载模块仍执行其顶层 Python。
`intent read-artifact <返回的文件路径> --offset 0 --limit 16000 --json` 可分页读取
真实 IR、源码或日志，不执行文件内容。compiler MCP 提供相同 describe/environment/
read_artifact 入口，使仅使用 MCP 的客户端也能从编译诊断继续读取服务器上的产物。

CLI 的源码编译与 compiler MCP 的编译、续编译、物化、优化和环境检查通过
[requests.py](python/intent/tools/requests.py) 调用同一个单请求
[worker.py](python/intent/tools/worker.py)。源码及其普通 Python 依赖在请求进程中导入，
进程结束后不把 `sys.modules` 或 SDK 状态留给下一次 MCP 请求。MCP 异步等待 worker，
取消请求时回收该请求及其 native compiler 子进程；独立结果文件将工具响应与源码的
stdout/stderr 分开。编译缓存、导出目录与诊断文件仍由原公开编译 API 持有，
不会随临时通信目录删除。直接调用 `intent.compile/generate` 仍是进程内 Python API。

定位失败时看 `CompilationStageError.stage`、`cache_directory`、`candidate` 与
`artifacts`。Native 编译失败保留实际 attempt 目录及异常 cause，不被 materialize
覆盖成只有 Intent 编译目录的错误。CLI/MCP 的 [compilation.py](python/intent/tools/compilation.py)
转换同一阶段链与文件路径，不维护另一份错误知识库或 compiler policy。
Intent lowering、provider native compilation、首次 tuning/load 与热 launch 是不同成本；
`generate` 不运行 kernel，materialize 是否立即 native compile 由 provider 决定，
首次 launch 仍可能触发 JIT/tuning。普通 PyTorch graph capture 前先完成需要的首次调用。

作者位置沿 [SourceUnit.location](python/intent/frontend/source/unit.py)、[KIR 序列化](python/intent/frontend/mlir/serialization.py) 和 [compiler IR 输出](tools/intent-compile/intent-compile.cpp) 保存在标准 MLIR location 中。缓存的 `input.mlir`、`kernel.mlir` 与 operation 诊断使用这条位置链；新增 rewrite 创建或克隆 operation 时保留相应 source location，不用旁表替代。编译日志位于同一 `cache_directory` 的 `compiler.log`。

各 native driver 通过共同 command runner 保留命令、编译器 stdout/stderr、状态与阶段耗时。
Mojo 的 [native compilation](python/intent/runtime/mojo/compilation.py) 另外保留具体
candidate bindings，`request.json` 指向实际 source 与 FP object。
`NativeArtifact.directory/cache_hit/cache_reason` 提供本次产物及复用状态；既有 CPU
runner 在准备阶段记录编译耗时和命中数量，编译、加载与调优不计为算子执行时间。
Weft 的 [Canonical IR serializer](lib/Target/Weft/Serialization/Serializer.cpp) 保留标准
location，使下层编译诊断可以追到作者源码。库加载失败保留 `native_loading` 与实际
native 目录，不被调用入口覆盖成只有顶层 Intent cache 的错误。

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

CPU 同样复用已有生产 input 与原 target/capability/profile 参数。完整共享 pass 列表及输入条件见 [CPU Passes.td](include/Intent/Dialect/CPU/Transforms/Passes.td)。例如 `intent-cpu-materialize-configurations` 将单个未绑定函数变为具有完整 binding 的候选函数；`intent-cpu-prepare-inputs`、`intent-cpu-block-computations`、`intent-cpu-partition-tasks` 可分别打印供数、分块与任务形成的前后 IR，`intent-cpu-isolate-tasks` 闭合显式 captures。Mojo 阶段名为 `intent-mojo-materialize-program`、`intent-mojo-fuse-private-computations`、`intent-mojo-vectorize-program`、`intent-mojo-finalize-program`。独立使用时须传入所需 pass options，并由相同 dialect registration 安装 context 内的 provider interface；前置 IR 合同仍需成立。`intent-cpu-pipeline` 仍是完整默认入口，其展开顺序直接由标准 PassManager 执行。

查看 provider 阶段时，沿用同一输入、目标 options 和 tuning profile 的完整编译命令，去掉 `--stop-after-shared`，同时指定 `--ir-output` 与 `--source-output`。例如既有 [cuTile official_fmha](experiments/gpu/providers/cutile/attention.py) 编译 [flash_gqa_attention_fwd](examples/kernels/streaming/attention.py) 时，在原命令追加 `--mlir-print-ir-before=intent-cutile-native-program --mlir-print-ir-after=intent-cutile-native-program --mlir-print-debuginfo`，即可对照原生 form 形成前后的 IR 并显示作者位置；对应 Triton 阶段名是 `intent-triton-native-forms`。最终合法化分别看 `intent-cutile-finalize-program` 与 `intent-triton-finalize-program`。这些阶段名用于同一完整 pipeline 的诊断，不表示可以跳过其输入依赖和 tuning profiles 单独调用。

提交一个连贯变换前，说明它读取的 facts、合法条件、实际改写的 IR、保持的语义、失效或重算的分析，以及在哪个边界验证 postcondition。用必要的既有生产运行确认影响；不另建测试目录、平行结果表或额外评测矩阵。

实验运行与数据继续归入对应的 `experiments/{gpu,cpu,mlu,agent_tritonbench}/`；`examples/kernels/` 不承载 benchmark runner。报告结果时分别说明生成成功、编译成功、运行正确与已测性能，不用 pass 数量或文件拆分数量代表能力。

## Agent 的职责边界

编写 Intent 算法时，使用 README 中的 [manual MCP](README.md#use-with-an-agent) 查询公开声明、语言合同和必要最小片段；实现入口是 [manual.py](python/intent/tools/manual.py)。完整算法示例在 `examples/`，manual 不提供题解或执行结论。

需要编译用户明确提供的程序时，显式启用独立 [compiler_mcp.py](python/intent/tools/compiler_mcp.py)。它要求已有程序路径，并复用 CLI 的同一实现；不会把完整算法加入 manual corpus，也不把动态编译诊断当作数值或性能通过。固定 agent 实验的一次正式提交规则继续由实验入口执行，日常产品工具不复用它的成绩协议。

修改编译器时，使用本页的模块导航、正式规格和当前源码；公开语言手册不承担 compiler implementation guide。合法性或职责不清楚时，先查原合同与真实消费者，再选择改动层次。
