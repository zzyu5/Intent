# Intent Kernel 编译器前端、Realization 与 Triton Emission 实现报告

## 1. 当前结论

当前仓库中已经形成一条单一、真实执行的编译链：

```text
Python Intent DSL
  │
  │  Python frontend
  │  AST / constexpr / symbol / shape / region state
  ▼
Canonical Intent Kernel MLIR
  │
  │  C++ intent-realize
  ▼
Intent Kernel MLIR
+ intent_plan.realization
+ intent_triton target choices
  │
  │  C++ intent-translate
  │  generic per-op emission
  ▼
Triton Python source
  │
  │  Triton JIT on CUDA
  ▼
Callable kernel + TTIR/TTGIR/LLVM IR/PTX
```

Python frontend、Kernel IR、realization、target emission 四层职责已经分开：

- Python 不再维护独立 typed Kernel IR，也不再有 Python IR verifier 或二次 MLIR emitter；
- Kernel MLIR 只保存目标无关的算法、逻辑 domain、结构化 region、SSA def-use、ABI 和 effect；
- `intent_plan` 只定义 realization/search-space 的跨目标 envelope；
- tile、program、storage、layout、primitive、boundary、pipeline 和 launch 都归 `intent_triton`；
- Triton emitter 不识别 softmax 名称，也不运行整 kernel matcher，只按 Kernel op 和已确定 realization 逐 op 发射；
- 当前真实可执行的后端能力是 rank-2 f32 rowwise kernel，stable softmax 是第一条活体路径。

## 2. 当前目录与职责

### 2.1 Python frontend

```text
python/intent/
├── api/                         # @intent.kernel / @intent.fn 定义对象
├── compiler/
│   ├── pipeline.py              # lower → realize → translate → materialize
│   └── toolchain.py             # C++ CLI stdin/stdout 边界
├── frontend/
│   ├── compilation/             # 一个 Definition 的编译入口与 helper specialization
│   ├── diagnostics/             # source span 与 frontend error
│   ├── source/                  # AST、closure、signature、constexpr ABI
│   ├── semantics/               # lowering 期间的 type/shape/op/effect 描述
│   ├── mlir/                    # canonical MLIR builder 与临时 lowering state
│   └── lowering/
│       ├── ast/                 # statement/expression/index lowering
│       └── intrinsics/          # control/tensor/structured/memory lowering
├── language/                    # 用户可见 dtype、annotation、intrinsic namespace
├── targets/                     # host target facts
└── runtime/                     # compiled artifact 与 Triton source materialization
```

这里不存在 `python/intent/ir` 或独立 `python/intent/mlir` 后处理链。MLIR 构造本身属于 frontend。

### 2.2 C++/MLIR

```text
include/Intent/
├── Dialect/
│   ├── Intent/IR/               # canonical Kernel dialect
│   └── Plan/IR/                 # target-neutral envelope
├── Target/Triton/
│   ├── Config/                  # target facts
│   ├── IR/                      # intent_triton dialect
│   ├── Realization/             # target realization API
│   └── Emission/                # target source translation API
└── Transforms/                  # Kernel MLIR trust-boundary verifier

lib/
├── Dialect/
│   ├── Intent/IR/
│   └── Plan/IR/
├── Target/Triton/
│   ├── IR/                      # target choice schema 与 verifier
│   ├── Realization/Rowwise.cpp  # rowwise capability analysis/selection
│   └── Emission/
│       ├── Translate.cpp        # resolved-realization translation boundary
│       └── Emitter.cpp          # per-op Triton emitter
└── Transforms/VerifyKernelIR.cpp
```

`IR`、`Realization`、`Emission` 分别是数据模型、物理选择和目标代码发射，不在同一个文件中混合职责。

## 3. Python frontend 如何直接形成 canonical MLIR

### 3.1 Source 与 signature

`SourceUnit` 从 decorated Definition 获得：

- 原始 Python AST；
- filename、line、column；
- globals、closure globals、nonlocals 和 builtins；
- 唯一同步 function definition。

Signature lowering 在构造函数参数前完成：

- `In`、`Out`、`InOut` view 被转换为 tensor shape、dtype、access 和 constraints；
- runtime scalar 保持运行时参数；
- constexpr 必须在 lowering 前得到 Python 值并检查其 Python type；
- helper 按实参 `ValueType` tuple specialization；
- kernel 结果只能通过 output view 表达。

### 3.2 只存在 lowering state，不存在第二套 IR

Frontend 保留的对象是编译期间临时状态：

| 对象 | 作用 | 是否是独立 Kernel IR |
|---|---|---|
| `ValueType` / shape descriptor | AST lowering 时推导 dtype、shape、domain 与 record schema | 否 |
| `MlirValue` | 记录即将写入 MLIR 的 SSA ID、type、source location 与 name hint | 否 |
| `RegionState` / `BlockState` | 暂存当前结构化 region 的文本、argument、terminator 和 effect summary | 否 |
| `FunctionState` | helper symbol、parameter 和最终 result schema | 否 |
| `EmittedOperation` | 仅把刚创建 operation 的 result handles 返回给 AST lowering | 否 |

没有 Python `Module → Function → Operation → Value` typed graph，也没有对该图执行的全局 verifier。

### 3.3 Operation 构造

所有 AST/intrinsic handler 最终调用同一个 `FunctionLowerer.emit()`。它立即进入 `MlirBuilder.emit()`：

1. 检查 operand、result type、effect target、result name 数量；
2. 检查 region 数量和 single-block terminator；
3. 分配稳定 operation/value ID；
4. 写入 `intent.node`、result node/type/name/shape、region argument 和 effect metadata；
5. 立即生成 generic Intent operation assembly；
6. 只返回本次 operation 的 SSA result handles。

Function 完成后，builder 组合 module assembly。`canonicalize_mlir()` 使用官方 LLVM 20.1.8 `mlir.ir.Module.parse()` 解析并重新打印，返回唯一 canonical Kernel MLIR 字符串。

MLIR Python bindings 位于项目外部：

```text
/home/kingdom/.venvs/intentdsl-mlir20
```

仓库中没有虚拟环境、LLVM 源码或 bindings 构建缓存。

## 4. Kernel MLIR 的信任边界

Kernel IR 中保存的是逻辑算法：

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

这里没有 program ID、BLOCK_SIZE、warp、stage、occupancy 或 launch grid。

Python 在每个构造点报告 source-located frontend error。进入 C++ 后，`VerifyKernelIR.cpp` 是任意 MLIR 文本的信任边界，检查：

- 唯一 kernel 与 function metadata；
- parameter/result/value/node ID schema；
- view ABI 与 constraints metadata；
- region argument、single block 和 terminator；
- effect 与 structured index metadata；
- operation-specific 必需属性的基本结构。

这不是第二套 Python 语义体系，而是跨进程 MLIR 输入必须经过的 C++ boundary verification。

## 5. Realization 与 Search Space 是两类对象

共享 Plan dialect 当前只有三个 operation：

| Operation | 含义 |
|---|---|
| `intent_plan.realization` | 已完成选择、允许 target emitter 消费的 realization |
| `intent_plan.search_space` | 尚未完成选择、只允许 realizer/search 使用的候选空间 |
| `intent_plan.yield` | envelope region terminator |

两种 envelope 都只保存：

- Kernel entry symbol；
- target 名称；
- target dialect 拥有的 body。

共享 Plan 不包含 Triton architecture、tile、program、warp、storage space 或 intrinsic 名称。Translator 如果看到 `search_space` 会直接失败；它只接受恰好一个 `realization`。

当前没有伪造搜索器、候选枚举、cost model 或 autotuner。`search_space` 是明确的 IR 类型边界，不会被当作已选择 Plan 使用。

## 6. Triton concrete realization

`intent_triton` 保存真正的目标机制：

| Operation | 关键字段 | 当前 stable-softmax realization |
|---|---|---|
| `target` | architecture、device、warp size | 当前 CUDA device facts |
| `axis` | Kernel node、source axis、role、tile | row=`one`，column=`next_power_of_two` |
| `program` | loop node、worker axis、traversal、mapping | program axis 0、persistent、grid-stride |
| `storage` | ABI value、address space | input/output global |
| `layout` | ABI value、kind、axis order | row-major `[0, 1]` |
| `reduction` | Kernel node、target lowering、axis | `tl.max` / `tl.sum`，axis 0 |
| `pointwise` | Kernel node、target lowering | alias、Python operators、`tl.exp` |
| `boundary` | load node、domain node、predicate、fill、store mask | `< n_cols`、`-inf`、predicate mask |
| `pipeline` | loop node、low/high stages、threshold | 2/4、200000、无 prefetch/async |
| `launch` | loop node、grid policy、warps | persistent occupancy、8 warps |

每种 `intent_triton` op 有本地 verifier；`verifyTritonRealization()` 再检查 target body 中的唯一 target/program/pipeline/launch，以及 axis、storage、layout、primitive 和 boundary binding 的重复情况。

## 7. Rowwise realizer 如何选择 stable softmax 方案

`realizeKernel()` 首先执行 Kernel boundary verifier，然后调用 `analyzeRowwiseKernel()`。该分析识别的是当前 Triton rowwise 能力，不读取函数名，也不要求一个固定 softmax operation 数量表。

当前能力条件是：

- module 中恰好一个 kernel entry；
- ABI 为一个 rank-2 f32 input view 和一个同 shape output view；
- innermost stride 为 1，layout 为 row-major，满足 noalias；
- 有一个 single-block、无 carried state 的 `intent.parallel` region；
- row domain 来自 input axis 0，column domain 来自 axis 1；
- body 中每个 op 都有当前 Triton realization handler；
- load/store 的 structured index relation 是同一个 row argument 与 column domain。

Boundary fill 不是无条件写死。对当前 `-inf` masked-load 机制，realizer 检查：

- loaded tensor 有 maximum reduction user；
- loaded tensor 的其他直接 user 必须是 `loaded - broadcast(maximum)`；
- 不允许其他直接 user 绕过这一 masked-lane 语义；
- output store 自身使用 predicate mask。

因此物理 padding lane 的 `-inf` 不进入逻辑输出；这个证明属于 Triton realization，而不是 Kernel IR 语言语义。

通过分析后，realizer 逐 Kernel op 写入 `intent_triton.reduction`、`pointwise` 和 `boundary` binding，再补充 axis/program/storage/layout/pipeline/launch。它不产生 Python Plan，也不修改 logical Kernel IR。

## 8. Generic per-op Triton emission

Translator 的入口顺序是：

1. 验证 Kernel module；
2. 执行 MLIR verifier；
3. 拒绝 `intent_plan.search_space`；
4. 要求一个 `intent_plan.realization`；
5. 验证 Triton realization；
6. 调用 generic emitter。

Emitter 先按 stable node/value ID 建立：

- Kernel operation index；
- ABI value、storage 与 layout binding；
- row/column axis binding；
- reduction、pointwise 与 boundary binding；
- program、pipeline 与 launch choice。

随后按 operation 遍历：

| Kernel op | Emission 行为 |
|---|---|
| top-level `constant` / `dim` / `domain` | 由 axis/program binding 解析为 runtime extent，不生成目标计算语句 |
| `parallel` | 从 program choice 生成 program ID、program count 和 grid-stride `tl.range` |
| body `constant` | 发射 Python scalar literal，包括 `-float('inf')` |
| `view_load` | 从 storage/layout/index/boundary 生成 offsets 与 masked `tl.load` |
| `reduce` | 按该 node 的 reduction binding 发射 `tl.max` 或 `tl.sum` |
| `broadcast` | SSA alias，不生成虚假的 copy |
| `unary` | 按该 node 的 pointwise binding 发射目标 intrinsic/operator |
| `binary` | 按该 node 的 pointwise binding发射 `+ - * /` |
| `view_store` | 从 index/layout/boundary 生成 masked `tl.store` |
| `yield` / function `return` | 结构 terminator，不生成目标表达式 |

每个 handler 都比较 Kernel op 的逻辑 operator 与 target binding。缺 binding、operator 不一致、operand 未发射或结构不支持时，错误直接挂在当前 MLIR operation 上。

Emitter 中没有：

- stable-softmax 函数名判断；
- 整 kernel operation multiset；
- operation 指针等同于 max/sum/subtract/divide 的判断；
- 默认切换到另一种算法；
- Python emitter fallback。

## 9. Runtime 与真实 GPU 执行

C++ emitter 生成一个独立可执行的 Triton source，其中包括：

- `@triton.jit` target kernel；
- target device/property 查询；
- input/output device、dtype、shape、stride、noalias guard；
- `BLOCK_SIZE = next_power_of_2(n_cols)`；
- 根据 shared-memory threshold 选择 stage；
- warmup/JIT 后读取 register/shared-memory usage；
- occupancy 驱动的 persistent program count；
- 底层 `launch(input, output)`；
- 用于公平 wrapper 对比的 `run(input)`。

`materialize_triton_artifact()` 编译并执行生成 source，要求 namespace 中存在 `launch` 和 `run`。第一次真实 launch 后，artifact 从 Triton compiled kernel 收集 TTIR、TTGIR、LLVM IR 和 PTX 文本。

Artifact 的 `mlir`、`source`、`backend_ir`、`ir`、callable entry 和 `run()` 是同一个 compiled artifact 的不同观察/调用接口，不是第二套编译路径。

## 10. 当前实际调用链与单路径核验

生产编译路径是：

```text
intent.compile
  → lower_to_mlir
  → TritonTarget.resolve
  → intent-realize / realizeKernel
  → intent-translate / emitTritonSource
  → materialize_triton_artifact
```

`intent-opt --verify-intent-kernel` 只在 repro 的 frontend coverage 阶段验证 11 个 Kernel MLIR module；它是 MLIR trust-boundary CLI，不是另一条 realization/emission 路径。

`source/` 下的 Triton、TileLang、cuTile 等文件属于高性能 baseline corpus。与上游源码相邻的 runtime 用于单独运行 baseline，不属于 Intent compiler 实现，因此不会因为当前 softmax repro 没调用某个 baseline runtime 就删除。

当前 compiler tree 中：

- 没有旧 Python typed IR graph/verifier；
- 没有独立 Python MLIR emitter；
- 没有 stable-softmax 共享 matcher/analysis library；
- 没有 stable-softmax 专用 realizer；
- 没有 stable-softmax 专用 emitter；
- 没有旧 concrete `intent_plan.plan` schema；
- 所有 include/lib `.cpp` 都由当前 CMake target 收录；
- 仓库没有 build、virtualenv、log、cache 或临时生成文件。

## 11. 唯一 repro 与本次实际结果

唯一验证命令：

```bash
bash examples/repro/run_frontend_softmax.sh
```

运行条件：

- CUDA device：`cuda:0`；
- shape：`(8192, 8192)`；
- dtype：f32；
- benchmark：`triton.testing.do_bench`；
- warmup：100；
- repetitions：500；
- generated 与上游 wrapper 都包含 output allocation。

Frontend：

```text
11 / 11 representative Kernel MLIR modules: PASS
50 / 50 logical operation kinds covered: PASS
```

真实 JIT backend IR：

```text
llir, ptx, source, ttgir, ttir
```

数值：

| 对比 | max absolute error |
|---|---:|
| generated vs `torch.softmax` | `1.862645149230957e-09` |
| upstream vs `torch.softmax` | `1.862645149230957e-09` |
| generated vs upstream | `1.862645149230957e-09` |

性能：

| wrapper latency | 上游 Triton | Intent generated | generated/upstream |
|---|---:|---:|---:|
| p50 | `0.3691 ms` | `0.3686 ms` | `0.9986x` |
| p95 | `0.3714 ms` | `0.3707 ms` | `0.9983x` |

这次运行中 generated wrapper 没有性能退化。亚百分之一差异只说明两者处于相同性能水平，不解释为稳定加速。

## 12. 当前硬边界

| 层次 | 当前状态 |
|---|---|
| Python frontend | 50 个 logical operation kind 均能形成并验证 Kernel MLIR |
| Canonical Kernel IR | 唯一产物是 Intent Kernel MLIR；无 Python typed IR |
| Shared Plan | 已区分 resolved realization 与 unresolved search space |
| Triton realization | 当前只接受单 input/output、rank-2 f32 rowwise capability |
| Triton emission | 对当前 rowwise 支持集逐 op 发射；其他 op 在自身位置失败 |
| 搜索 | 尚未实现 candidate enumeration、cost model 或 autotuning |
| 其他 kernel backend | GEMM、attention、MoE 当前只有 frontend representation |
| 其他 target | TileLang、cuTile、CPU、RVV 尚无 emission backend |

因此当前准确状态是：完整 frontend representation 已直接落到 canonical MLIR；realization 与 emission 已形成通用架构边界；stable softmax 已在新架构上从 DSL 走到真实 GPU，并达到上游 Triton baseline 的数值与性能水平。
