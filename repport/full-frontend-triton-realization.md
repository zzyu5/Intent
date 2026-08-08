# Intent DSL 前端、C++ Physical Plan 与 Triton Lowering 实现报告

## 结论

当前唯一可执行编译链是：

```text
受限 Python eDSL
  │  FrontendCompiler / FunctionLowerer
  ▼
typed Python Kernel IR
  │  Python semantic verifier
  ▼
Intent Kernel MLIR
  │  C++ intent-realize
  │  Kernel boundary verifier + stable-softmax matcher
  ▼
Intent Kernel MLIR + intent_plan MLIR
  │  MLIR verifier + PlanOp verifier
  │  C++ intent-translate
  ▼
Triton Python source
  │  Python host compile/exec + Triton JIT
  ▼
callable kernel + TTIR/TTGIR/LLVM IR/PTX
```

后端语义已经收敛到 C++/MLIR：Python 中不存在 Physical Plan 数据模型、Plan verifier、Plan serializer、stable-softmax realizer 或 Triton source emitter。Python 仍然存在，因为当前 source language 是 Python eDSL，而且 Triton 的目标语言和 JIT host API 本身也是 Python；这两件事不等于“用 Python 决定后端 realization”。

当前只有 canonical stable softmax 从 DSL 真正走到 GPU 并与原始高性能实现比较。其余 frontend 构造只证明能够形成并验证 Kernel MLIR，不代表已有 Triton lowering。

## 冗余代码审计

### 原有重复不是什么

旧代码中并没有第二份 Python Triton source emitter。原 `python/intent/backend/triton/translator.py` 只是调用 C++ `intent-translate` 的 subprocess wrapper，原 runtime 只是执行生成源码。

真正重复的是 stable-softmax 后端语义被分散在两种语言中：

- Python `realizer/triton/softmax.py` 匹配算法并构造 `PhysicalPlan`；
- Python `realizer/verify.py` 验证同一 Plan；
- Python `mlir/plan.py` 把 Plan 再序列化为 MLIR；
- C++ `PlanOp::verify()` 和旧 `Translate.cpp` 又重新检查同一算法、ABI 与 Plan；
- 旧 `Translate.cpp` 同时承担 matcher、Plan legality 与源码发射，形成单文件多职责。

### 删除的 Python 后端语义

下列实现已经删除，没有保留兼容 shim：

| 删除路径 | 原职责 | 删除原因 |
|---|---|---|
| `python/intent/realizer/model.py` | `PhysicalPlan`、launch、tile、storage 等 Python 数据模型 | Plan 的唯一正式表示改为 MLIR dialect |
| `python/intent/realizer/triton/softmax.py` | Python stable-softmax matcher 与固定 Plan 构造 | realization 迁入 C++ |
| `python/intent/realizer/verify.py` | Python Plan verifier | Plan legality 由 C++ MLIR verifier 守住 |
| `python/intent/mlir/plan.py` | Python Plan → MLIR serializer | C++ realizer直接构造 MLIR operation |
| `python/intent/backend/triton/translator.py` | C++ translator 进程桥接 | 统一迁入 `compiler/toolchain.py`，不再伪装成 backend implementation |
| `python/intent/backend/triton/target.py` | target host object | 移到中立的 `targets/` 层 |
| `python/intent/backend/triton/runtime.py` | 目标源码物化 | 移到明确的 `runtime/` 层 |
| `python/intent/backend/artifact.py` | artifact 数据结构 | 移到 `runtime/`，并去掉 Python Plan/launch 对象依赖 |
| `python/intent/driver.py` | 混放在包根的端到端编排 | 移到 `compiler/pipeline.py` |
| `python/intent/definitions.py` | DSL public definitions | 移到 `api/definitions.py` |
| `python/intent/errors.py` | 跨层错误类型 | 移到 `diagnostics/errors.py` |

全仓库当前没有旧路径 `intent.backend`、`intent.realizer`、`intent.driver`、`intent.definitions`、`intent.errors`、`intent.mlir.plan` 的导入，也没有 `PhysicalPlan`、`LaunchSpec` 或 Python Plan serializer 的实现引用。

### C++ 中保留的重复调用与重复实现之别

`matchStableSoftmax()` 会在 realization、`PlanOp::verify()` 和 translation 边界被调用。这是同一个 `lib/Analysis/StableSoftmax.cpp` 实现被多个不可信 IR 边界复用，不是三份 matcher 代码：

- realizer 在生成 Plan 前确认 Kernel IR 是支持的算法；
- Plan verifier 在任意组合 MLIR 被解析时确认 Plan 没有改变算法；
- translator 需要 matcher 返回的 operation 指针与 value/node 信息才能发射。

`intent-translate` CLI 原来在调用 library emitter 前额外执行一次 `mlir::verify()`；library 本身已经完成该验证，这个重复执行已经删除。当前 `lib/Target/Triton/Translate.cpp` 只保留薄入口，算法分析只有一份，目标源码发射只有一份。

## 目录边界

### Python

```text
python/intent/
├── __init__.py              # 仅 public façade
├── api/
│   └── definitions.py       # @intent.kernel、@intent.fn、Definition
├── compiler/
│   ├── pipeline.py          # 编译阶段编排
│   └── toolchain.py         # C++ executable stdin/stdout 边界
├── diagnostics/
│   └── errors.py            # 跨阶段异常类型
├── frontend/
│   ├── compiler.py          # source function/module lowering
│   ├── lowering.py          # FunctionLowerer 与环境/SSA 操作
│   ├── expressions.py       # Python AST expression lowering
│   ├── statements.py        # Python AST statement lowering
│   ├── signature.py         # kernel/helper ABI lowering
│   └── intrinsics/          # control/tensor/structured/memory 分类
├── ir/
│   ├── module.py ops.py types.py values.py effects.py
│   ├── builder.py
│   └── verifier.py          # typed Kernel IR semantic verifier
├── language/                # 用户可见 annotation、dtype、builtins
├── mlir/
│   ├── emitter.py           # 只发射 Intent Kernel MLIR
│   ├── attributes.py
│   └── types.py
├── targets/
│   └── triton.py            # host device capability → target facts
└── runtime/
    ├── artifact.py          # source/MLIR/callable/backend IR
    └── triton.py            # 执行已生成源码并连接 Triton JIT
```

根包只有 `__init__.py`，其余实现按职责进入目录。`frontend/` 与 `ir/` 没有为了追求目录外观而继续拆碎：两者分别是 AST lowering 和 typed logical IR，内部文件已经按表达式、语句、签名、intrinsic、type、value、effect、verifier 分工。

### C++/MLIR

```text
include/Intent/
├── Analysis/StableSoftmax.h
├── Dialect/
│   ├── Intent/IR/           # Kernel dialect ODS/type/op public boundary
│   └── Plan/IR/             # Physical Plan dialect ODS public boundary
├── Target/Triton/
│   ├── Target.h             # target facts
│   └── Translate.h          # translator public API
└── Transforms/
    ├── Passes.h
    └── RealizeStableSoftmax.h

lib/
├── Analysis/
│   └── StableSoftmax.cpp    # 唯一 canonical algorithm matcher
├── Dialect/
│   ├── Intent/IR/           # Kernel dialect implementation
│   └── Plan/IR/             # Plan dialect + PlanOp verifier
├── Transforms/
│   ├── VerifyKernelIR.cpp   # Kernel MLIR boundary verifier
│   └── RealizeStableSoftmax.cpp
└── Target/Triton/
    ├── Translate.cpp        # verify/match/dispatch 薄入口
    └── StableSoftmax.cpp    # 唯一 Triton source emitter

tools/
├── intent-opt/              # Kernel verifier CLI
├── intent-realize/          # Kernel MLIR → Kernel+Plan MLIR
└── intent-translate/        # Kernel+Plan MLIR → Triton source
```

旧的空 `include/Intent/Conversion`、`lib/Conversion/IntentToSCF` 与无实际构建作用的 header-only Target CMake 空壳已经删除。构建目录仍在 `/tmp/intentdsl-build`，没有写入仓库。

## Python DSL 到 typed Kernel IR

### Definition 与源码身份

`api/definitions.py` 中：

- `@intent.kernel` 构造 `KernelDefinition`；
- `@intent.fn` 构造 `HelperDefinition`；
- Definition 保存原始 Python function、源码文件和首行；
- Definition 对象不能像普通 Python function 一样直接执行，必须经过 frontend lowering。

这使 Python 只承担 source carrier，而不是在运行时解释 kernel 算法。

### Signature lowering

`frontend/signature.py` 将 Python annotation 转成 ABI：

- kernel 参数只能是 view、runtime scalar 或 constexpr；
- view 保存 `In`/`Out`/`InOut`、dtype、symbolic/static shape 与 constraints；
- kernel 不能返回内部 SSA value，结果通过 output view 表达；
- helper 参数统一为 SSA value，当前禁止 default、keyword-only 和 variadic 参数。

Stable-softmax source 的 ABI 是两个 symbolic `(M, N)` f32 view：input 为 `In`，output 为 `Out`，两者要求 row-major 和 noalias。

### Function 与 helper lowering

`FrontendCompiler.lower()` 的实际流程是：

1. 从 Definition 建立 `SourceUnit`；
2. lower kernel signature；
3. 通过 `IRBuilder` 创建唯一 kernel function 和 entry block；
4. 使用 `FunctionLowerer` 递归 lower AST；
5. 构造 module 后执行 typed Python IR verifier。

Helper 不是运行时 Python 调用。它以 `(HelperDefinition, 实参 IRType tuple)` 为 specialization key，形成独立 helper function；相同 specialization 复用 cache。递归 helper 当前没有 recursive IR contract，因此直接报错，不伪造递归 lowering。

### AST 分派与 SSA

`FunctionLowerer` 把职责分给：

- `statements.lower_statement()`：assignment、expression statement、return、if/for/while、break/continue 等；
- `expressions.lower_expression()`：name、constant、unary/binary/compare、subscript、call、intrinsic 等；
- `intrinsics/control.py`：logical control primitives；
- `intrinsics/tensor.py`：broadcast/reshape/transpose/reduce/scan/contract 等；
- `intrinsics/structured.py`：domain、parallel、ordered、state stream 等；
- `intrinsics/memory.py`：view/buffer/gather/scatter/atomic/fence。

Kernel 正常走到函数尾会产生 `intent.return`。Helper 必须显式 return，并形成稳定 result schema。Region operation 产生 block argument 与显式 terminator，不能用 Python 控制流对象绕过 IR。

### View 与 effect lowering

读取 external view 时，frontend 生成 `VIEW_LOAD` 和 `READ/EXTERNAL_VIEW` effect；写入生成 `VIEW_STORE` 和 write effect。ABI 权限在 lowering 时就被检查：

- `Out`-only view 不能读取；
- `In`-only view 不能 store/scatter；
- external atomic target 必须为 `InOut`；
- gather/scatter、buffer load/store、atomic 和 fence 都保留 resource、target 与 effect kind。

Stable softmax 的一行逻辑算法在 Kernel IR 中保持为：

```text
columns = domain(0, dim(x, 1))
rows    = domain(0, dim(x, 0))
parallel row in rows:
    values      = view_load(x, row, columns)
    maximum     = reduce.maximum(values, axis=0, identity=-inf)
    numerator   = exp(values - broadcast(maximum))
    denominator = reduce.add(numerator, axis=0, identity=0)
    view_store(y, numerator / broadcast(denominator), row, columns)
```

这里没有 `program_id`、BLOCK_SIZE、warp、stage、shared memory 或 launch grid。

## Kernel IR 与两级 verifier

### 50 个 Core opcode 的含义

`ir/ops.py` 当前定义 50 个 logical opcode，覆盖：

- domain/product/partition/indices；
- parallel/ordered/state stream 与 if/for/while；
- view、buffer、gather/scatter、atomic、fence、random；
- reshape/transpose/broadcast、unary/binary/select/cast/mask；
- reduce/scan/contract 与 ragged 构造；
- helper call/return。

repro 中 11 个代表 Definition 合并覆盖这 50 个 opcode，并逐个通过 Kernel MLIR parser/verifier：

```text
vector_add, tensor_showcase, control_showcase, stream_showcase,
memory_showcase, gemm, stable_softmax, flash_attention_fwd,
reduction_pass1, reduction_pass2, moe_expert_ffn
```

这只是 frontend representation coverage：它证明 source AST 能形成 typed Kernel IR、序列化为 MLIR，并通过两层 verifier。11 个 module 没有进入 realizer/translator，只有 `stable_softmax` 进入 backend。因此“50 opcode 存在”不能解释成“50 opcode 已有 Triton lowering”。

### Python semantic verifier

`ir/verifier.py` 检查的是 typed Kernel IR 语义：

- module 恰好一个 kernel，函数名唯一；
- SSA value 的 owner、定义可见性、result type 与 block scope；
- region 数量、block argument schema 和 terminator 位置；
- if/while/for/state stream 的 carry/yield/condition 类型；
- dtype、shape、axis、broadcast、reduce、scan、contract、ragged 等 op 约束；
- memory op 的 index relation、ABI access 与 effects；
- 禁止 `program_id`、block/thread identity、num_warps、num_stages、launch 等 physical 属性进入 logical IR。

任何诊断最终抛 `VerificationError`；没有 fallback Kernel IR。

### Intent Kernel MLIR metadata

`mlir/emitter.py` 在 generic Intent operation 上保存 backend boundary 所需身份：

| 层次 | metadata |
|---|---|
| module | `intent.source_module` 与 module attrs |
| function | `intent.kind`、`intent.parameters`、`intent.parameter_nodes`、`intent.results`、`intent.source` |
| parameter | name、kind/view_kind、type、shape、strides、layout、alignment、alias、noalias |
| operation | `intent.node`、`intent.result_nodes`、`intent.result_names`、`intent.result_types`、可选 result shapes |
| region | nested block argument node IDs 与 source names |
| effectful op | effects、resource、target |
| indexed op | structured `intent.index` relation |

Operation node ID 与 value node ID 是不同命名空间。Plan 用 operation node 绑定 extent/primitive/loop，用 value node 绑定 ABI storage/layout。Source name 用于生成代码可读性，不能用来识别算法。

### C++ Kernel boundary verifier

`lib/Transforms/VerifyKernelIR.cpp` 验证序列化后的 MLIR，而不是重新执行 Python verifier：

- function kind、唯一 kernel、ABI metadata 数量与 MLIR signature 对齐；
- view kind/shape/strides/layout/alignment/alias/noalias schema；
- operation node ID 非负且唯一；
- result IDs/names/types/shapes 与 MLIR results 对齐；
- nested region argument IDs/names、single-block schema 与 terminator；
- semantic attribute、effect 与 index metadata 的基本结构。

两级 verifier 的原因是信任边界不同：Python verifier守住 typed builder 语义，C++ verifier守住任意文本 MLIR 输入。ODS 当前仍有较宽的 `AnyType + metadata` 表达，因此不能只依赖自动生成的 ODS verifier。

## C++ stable-softmax analysis

`lib/Analysis/StableSoftmax.cpp` 是 realizer、Plan verifier 和 translator 共享的唯一算法 matcher。它不检查函数名，而是检查下面的完整结构。

### ABI

- module 恰好一个 `func.func`，且 `intent.kind = kernel`；
- 恰好两个参数；
- 两者都是 dynamic rank-2 f32 `!intent.view`；
- access 分别为 `in`、`out`；
- symbolic shape 两维非空且完全相同；
- strides 为 `[unknown-row-stride, 1]`；
- layout 为 row-major，noalias 为 true；
- parameter value node IDs 存在。

### Top-level workset

函数顶层只能且必须包含：

- 两个 zero constant；
- 两个 `intent.dim`；
- 两个 `intent.domain`；
- 一个 `intent.parallel` row loop；
- 一个 `intent.return`。

Row domain 必须是 `0 .. dim(x, 0)`，column domain 必须是 `0 .. dim(x, 1)`。Row loop 必须只有一个 domain operand、一个 single-block region 和一个 logical row index。

### Loop body operation multiset

Loop body 数量被精确限制为：

| operation | 数量 |
|---|---:|
| `view_load` | 1 |
| `view_store` | 1 |
| `constant` | 2 |
| `reduce` | 2 |
| `broadcast` | 2 |
| `binary` | 2 |
| `unary` | 1 |
| `yield` | 1 |

不能夹带其他 operation。两个 reduction 必须分别是 maximum/`-inf` 和 add/zero，axis 都是 0；unary 必须是 exp，binary 必须是 subtract 与 true_divide。

### Def-use 与地址关系

Matcher 继续检查唯一链：

```text
load ───────────────┬─> reduce maximum -> broadcast ─┐
                    └────────────────────────────────> subtract
                                                        │
                                                        v
                                                       exp
                                                        ├─> reduce add -> broadcast ─┐
                                                        └────────────────────────────> divide -> store
```

Load 必须来自 input ABI value，store 必须写 output ABI value。两者 index relation 都必须使用同一个 row block argument 和 column-domain result；store 的 value operand 必须正是 divide result。

Matcher 成功后返回源 operation 指针、input/output value ID、`M/N` symbolic extent。它不生成 Plan，也不发射目标代码。

## Physical Plan MLIR 与 C++ realization

### Dialect schema

`intent_plan.plan` 保存 entry/backend/architecture/device/warp_size，并包含下面的 typed child operations：

| Plan op | 字段 | 绑定对象 |
|---|---|---|
| `extent` | node、axis、logical、tile | source loop/domain operation |
| `ownership` | loop_node、worker、worker_axis、traversal、mapping | logical loop |
| `storage` | value、space、access | ABI value |
| `layout` | value、kind、axis order | ABI value |
| `primitive` | node、kind、operator、axis、identity | source compute operation |
| `boundary` | node、logical、tail、predicate、load_fill | source extent/domain |
| `pipeline` | low/high stages、smem threshold、prefetch、async copy | target pipeline policy |
| `launch` | loop_node、grid policy、num_warps | realized ownership |
| `yield` | terminator | Plan region |

Plan 是 MLIR operation，不是字符串字典，也不是 Python dataclass。

### `intent-realize` 的实际工作

Python `compiler/toolchain.py` 把 Kernel MLIR 写入 `intent-realize` stdin，并传入：

```text
--architecture=sm_xy --device=N --warp-size=32 -
```

CLI 加载 Intent/Plan dialect，解析 module，调用 `realizeStableSoftmax()`，验证生成后的完整 module，再用 debug-info printing 保留 source locations 输出到 stdout。

`RealizeStableSoftmax.cpp` 当前不是搜索器。它调用共享 matcher，然后构造一个确定 Plan：

| 决策 | 当前值 | 来源/性质 |
|---|---|---|
| row extent | logical `M`, tile `one` | matcher 的 row loop 与 ABI shape |
| column extent | logical `N`, `next_power_of_two` | matcher 的 column domain 与 ABI shape |
| ownership | program axis 0, persistent, grid-stride | 固定 policy |
| input/output storage | global read / global write | ABI value ID 与 access |
| input/output layout | row-major `[0, 1]` | ABI constraint |
| max primitive | reduction/maximum/axis 0/negative infinity | source reduce node |
| numerator path | broadcast/subtract/exp | source node IDs |
| sum primitive | reduction/add/axis 0/zero | source reduce node |
| output path | broadcast/true_divide | source node IDs |
| boundary | masked/index_lt_extent/negative infinity | column domain |
| pipeline | stages 2/4, threshold 200000, no prefetch/async | 固定 policy |
| launch | persistent occupancy, 8 warps | 固定 policy |
| target | architecture/device/warp | resolved target + CLI |

这里的“自动”是从算法 IR 自动识别并构造 Plan，不是已经实现自动搜索。没有候选生成、cost model、benchmark search 或 autotuner。

### Plan verifier

`PlanOp::verify()` 再调用同一个 stable-softmax matcher，并验证：

- backend 目前只能是 Triton；target 字段完整；
- Plan entry 指向 matcher 找到的 kernel；
- Plan body 只能包含声明过的 Plan child op；
- row/column extent 精确回指 source node、`M/N` 和 tile policy；
- ownership 精确绑定 row loop，且为 program-axis-0 persistent grid-stride；
- storage/layout 恰好覆盖两个 ABI value，不能改变访问方向或 row-major axes；
- 七个 primitive binding 恰好覆盖 matcher 捕获的七个 compute node，并保持 operator/axis/identity；
- boundary 恰好绑定 column domain，保持 mask、predicate 与 fill；
- pipeline 恰好一个，stages/threshold 为正，当前禁止 prefetch 和 async copy；
- launch 恰好一个，绑定 row loop，grid policy 为 persistent occupancy，warps 为正。

Plan verifier 目前是 stable-softmax 专用 legality，不是通用 Physical Plan 证明器。它没有验证 ordered/state-stream effect reorder、通用 scratch lifetime、任意 layout legality或跨 kernel dependency，因为这些 Plan 目前根本不会被构造和接受。

## C++ Triton source lowering

### Translator 入口

`lib/Target/Triton/Translate.cpp` 只做四件事：

1. 执行 C++ Kernel boundary verifier；
2. 执行 MLIR verifier，因此触发 `PlanOp::verify()`；
3. 要求 module 恰好一个 Plan，并调用共享 stable-softmax matcher取得源 op；
4. 分派到 `emitStableSoftmaxSource()`。

它不接收 Python AST、typed Python IR 或 Python Plan。

### 命名

Emitter 从 MLIR metadata 读取 parameter、region argument 与 result source names。若同一作用域名字冲突，在名字后附 operation node ID。名字只影响可读性，不参与算法识别。

### Operation 到 Triton 的映射

| Kernel/Plan 信息 | 生成 Triton |
|---|---|
| ownership worker axis | `tl.program_id(axis)`、`tl.num_programs(axis)` |
| persistent grid-stride row ownership | `tl.range(row_start, n_rows, row_step)` |
| column tile | `tl.arange(0, BLOCK_SIZE)` |
| column boundary | `valid = columns < n_cols` |
| input `view_load` | row-stride offsets + masked `tl.load(..., other=-inf)` |
| max reduction primitive axis | `tl.max(value, axis=axis)` |
| max broadcast | scalar名直接进入 tensor expression |
| subtract node | Python `-` expression |
| exp node | `tl.exp` |
| sum reduction primitive axis | `tl.sum(value, axis=axis)` |
| sum broadcast | scalar名直接进入 tensor expression |
| divide node | Python `/` expression |
| output `view_store` | row-stride offsets + masked `tl.store` |
| Plan pipeline | runtime shared-memory threshold 选择 low/high stages |
| Plan launch warps | warmup/compile 的 `num_warps` |
| target device/warp | device guard 与 occupancy 公式 |

Value name map 以 MLIR SSA `Value` 为键，因此每一个 generated expression 都沿 Kernel IR def-use 取输入，不重新解析 source Python。

### 当前不是通用 Plan 解释器

当前 emitter 直接读取：ownership axis、reduction axis、pipeline 值、launch warps、target/device/warp 与 ABI storage direction。

下面这些 Plan 字段由 verifier 先限制为唯一支持值，emitter 随后发射固定实现，而不是动态 switch：

- extent tile 只能 `one/next_power_of_two`；
- ownership 只能 persistent grid-stride；
- storage/layout 只能 global row-major ABI；
- boundary 只能 masked、`index_lt_extent`、negative-infinity fill；
- pointwise primitive 只能当前七节点集合；
- launch grid policy 只能 persistent occupancy。

因此添加第二个合法值时，必须同时扩展 analysis/Plan legality/emitter。当前没有 `else` 默认成另一种 tile、boundary 或算法；unsupported structure 在 matcher 或 verifier 中失败。

Architecture 当前写入 Plan，但尚未显式传给 Triton JIT target selector；实际 JIT 使用 active CUDA device。Device 与 warp size 进入实际 runtime guard/occupancy。这个差异是当前实现边界，不能把 architecture 字段描述成已经强制控制 codegen target。

## Runtime 与 artifact

### Host target resolution

`targets/triton.py` 只查询 CUDA device 是否存在及其 compute capability，并把当前 Triton CUDA target 的 warp size 明确设为 32，形成 `ResolvedTritonTarget`。它不选择 tile、pipeline、primitive 或 launch policy。

### Toolchain 进程边界

`compiler/pipeline.py` 的实际调用是：

```text
lower_to_kernel_ir(definition)
emit_mlir(kernel_ir)
realize_mlir(kernel_mlir, intent-realize, resolved_target)
translate_mlir(combined_mlir, intent-translate)
materialize_triton_artifact(source, combined_mlir)
```

`compiler/toolchain.py` 对两个 C++ executable 使用 stdin/stdout。Executable 不存在、返回非零或 stdout 为空都会直接抛异常，并保留 stderr；没有 Python fallback。

### Generated source materialization

`runtime/triton.py` 对 C++ 生成的 source 执行 Python `compile(..., "exec")` 和 `exec`，要求命名空间提供 callable `launch()` 与 `run()`。这一步是加载目标语言，不是重新发射目标代码。

`CompiledArtifact` 当前保存：

- `source`：C++ translator 输出的 Triton source；
- `mlir`：Kernel IR + Physical Plan 组合 module；
- `_launcher`：生成 source 的 `launch`；
- `_runner`：生成 source 的 allocation wrapper `run`；
- `backend_ir`：第一次显式 `artifact(input, output)` 后从 Triton compiled kernel `.asm` 收集的文本 IR。

Artifact 不保存 Python Plan 或独立 launch configuration。Launch policy 只存在于 Plan MLIR 与生成 source。

### Generated launch wrapper

生成的 `launch(x, y)` 在 JIT 前检查：

- tensor 位于 Plan device；
- input/output 都是 f32；
- 两者 rank-2 且 shape 相同；
- innermost stride 为 1；
- 两个 allocation 地址区间不重叠，满足 noalias。

随后计算：

```text
BLOCK_SIZE = next_power_of_2(n_cols)
num_stages = high_stages if device_smem > threshold else low_stages

register_occupancy = max_num_regs /
                     (kernel.n_regs * warp_size * num_warps)
smem_occupancy     = max_shared_mem / kernel.metadata.shared
occupancy          = min(register_occupancy, smem_occupancy)
num_programs       = min(num_sms * occupancy, n_rows)
```

`kernel.n_regs` 和 `kernel.metadata.shared` 只有 Triton warmup/JIT 后才知道，所以动态 program count 不伪装成静态 Plan integer。最终一次 target kernel invocation 使用 persistent grid-stride loop覆盖全部 rows。

## Baseline 与唯一 repro

### Baseline 来源

比较对象是未修改的上游源码：

```text
source/triton/triton/normalization/softmax/02-fused-softmax.py
```

repro 用 Python AST 只加载该文件 `end_lineno <= 175` 的原始定义区，从 namespace 取得原始 `softmax` wrapper；这样不会执行文件后部自带测试/绘图 benchmark，也没有复制或改写 kernel implementation。

### 比较条件

- device：`cuda:0`，打印的 target architecture 为 `sm_120`；
- stream：两边使用同一显式 CUDA stream；
- input：`(8192, 8192)`，f32；
- source algorithm：full-row max → subtract → exp → sum → divide；
- wrapper：两边都包含 `empty_like` output allocation；
- 数值比较在计时外；
- benchmark：`triton.testing.do_bench`，`warmup=100`、`rep=500`、p50/p95；
- backend IR 来自生成 source 的真实 JIT。

### 最终实际结果

```text
frontend MLIR modules: 11/11 PASS
frontend opcode representation coverage: 50/50 PASS
本次运行观察到的 backend IR: llir, ptx, source, ttgir, ttir
```

| 数值 | 结果 |
|---|---:|
| generated vs `torch.softmax` max abs error | `1.862645149230957e-09` |
| original vs `torch.softmax` max abs error | `1.862645149230957e-09` |
| generated vs original max abs error | `1.862645149230957e-09` |

| wrapper latency | 原始上游 | 自动生成 | generated/original |
|---|---:|---:|---:|
| p50 | `0.3686 ms` | `0.3686 ms` | `0.9999x` |
| p95 | `0.3707 ms` | `0.3707 ms` | `1.0000x` |

这些键和数值是这一次 repro stdout 的观测结果，不是 artifact 对所有 Triton 版本承诺的固定 IR 集合，也没有另外保存 benchmark 日志。这次运行中二者性能实质相同。千分之一量级差异不能解释为稳定加速或回退；当前只做了一次 run，没有跨运行方差、置信区间、cache 清理对照，也没有跨 shape/dtype/GPU 矩阵。

唯一手动命令是：

```bash
bash examples/repro/run_frontend_softmax.sh
```

它构建 `intent-opt`、`intent-realize`、`intent-translate`，验证 11 个 frontend module，打印带 source location 的 Kernel+Plan MLIR与生成 Triton source，执行真实 GPU 数值比较，并测量原始/生成 wrapper。

## 当前硬边界

| 能力 | 当前状态 | 不支持时发生什么 |
|---|---|---|
| Python eDSL → typed Kernel IR | 50 个 logical opcode 有表示与 verifier | 未支持 AST/组合直接 frontend error/NotImplementedError |
| typed Kernel IR → Intent MLIR | 50-op representation path 已走通 | metadata/schema 不合法由 Python/C++ verifier 拒绝 |
| Kernel MLIR → Physical Plan | 仅 canonical rank-2 f32 stable softmax | matcher 失败，不生成 Plan |
| Physical Plan legality | 仅首个 Triton stable-softmax schema | `PlanOp::verify()` 失败 |
| Triton source emission | 仅 stable-softmax 专用 emitter | 未支持 op 直接 emit error |
| 搜索/cost model/autotune | 未实现 | 没有伪造候选或默认搜索结果 |
| GEMM/attention/MoE backend | 只有 frontend representation | 没有 target code，不声称可运行 |
| TileLang/cuTile/CPU/RVV | 未实现 backend | `compile` 当前只接受 TritonTarget |
| architecture-directed JIT | 只记录 architecture；JIT 随 active device | 尚未显式控制 Triton target |
| strict numerics/determinism policy | API 未实现 | 不接受一个假 `options` 字典 |

所以当前准确表述不是“完整算子编译器已经完成”，而是：完整前端表示骨架已经建立并穿过 MLIR 边界；后端已经形成干净的 C++/MLIR 单路径；第一条 stable-softmax realization 从 DSL、Kernel IR、Physical Plan、C++ source lowering 到真实 GPU baseline 全链走通。搜索与更多 kernel/backend realization 仍是后续主体工作。
