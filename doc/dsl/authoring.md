# 作者速查

本页是 [core.md](core.md) 与[数值规则](types-numerics-and-effects.md)的使用入口，不定义第二套语义。MCP 的 `search` 找概念/API/诊断/示例，`api` 查当前声明及返回规则，`read` 读完整章节和代码。公开声明不等于所有 target 已支持；没有性能测量不能宣称高效。

## 类型、literal 与 shape

在 kernel 中使用 `import intent.language as I`。`I.In/I.Out/I.InOut` 描述外部 views，`I.f32` 等描述 scalar dtype；Python literal 可按上下文实例化，但两个不同 dtype 的 runtime values 必须显式 `I.cast`。例如先把 bf16 输入 cast 到 f32，再和 f32 累加器计算，最后 cast 回输出 dtype。

`I.select` 的 bool 条件不提供数值分支的 expected dtype。两个分支都写成 literal 时，不要从生成条件的 tensor 推断结果 dtype；例如需要 f32 符号值时写 `I.cast(I.select(mask, -1.0, 1.0), I.f32)`。已经产生的 runtime value 不会因后续与 f32 相乘而重新实例化。

Tensor 和 view 有 `.shape`；scalar、tuple、record、domain 没有统一 `.shape`。`I.full(shape, fill, dtype)` 产生 tensor value，不分配跨 kernel workspace。`I.dot` 只接受两个 rank-1 tensor，返回 rank-0 tensor `[]`，不是 rank-1 `[1]` 或一个 Python number。Scalar 和 rank-0 tensor 是不同类型；pointwise scalar broadcast 由 frontend 显式表达。

## Domain、index 与 broadcast

`rows = I.domain(0, M)` 是 logical coordinate domain，不是整数 tensor，也不是归约的 axis 编号。`I.indices(rows)` 才产生 logical index tensor。`x[rows, columns]` 读取完整二维 logical region；`I.reduce.sum(value, axis=1)` 的 `1` 指 value 的第二个 axis。

Pointwise 按尾部对齐，允许 scalar/size-one broadcast。`[M]` 与 `[M,N]` 相加不会自动按行匹配：先 `I.reshape(row_value, (M, 1))`；`[N]` 可以直接广播到 `[M,N]`。两个未知 extent 不是因为都 dynamic 就兼容。

`I.transpose(value, permutation)` 显式重排；`I.reshape` 保持 row-major element order，不是任意 data permutation。Domain/subregion 与 integer coordinate tensor 不可互换；索引关系、有效范围和 fill 必须来自作者实际表达。

Domain 索引按资源索引顺序形成读取结果的 tensor axes，赋值仍按 positional axes 对齐，不按 domain 变量名自动换轴。例如两个等长 domains 下，`output[rows, columns] = input[columns, rows]` 不表示矩阵转置；应显式转置读取的 tensor value，或构造具有所需对应关系的坐标 tensor。不同 subregions 的动态长度也不会因本次输入碰巧等长而成为同一 extent；需要使用已成立的 shape relation，或在共同输出 domain 上表达坐标映射。

多个 tensor indices 按 broadcast 规则形成共同的索引 shape；domain index 则引入独立的 logical axis。例如二维逐元素按第 0 轴 gather，使用 `index=(indices[rows, columns], I.reshape(I.indices(columns), (1, N)))`。这里第二项是可广播的列坐标 tensor，直接传 `columns` domain 会额外引入一个轴。

## Reduce、tuple 与 helpers

`I.reduce.sum/max/any/all` 返回归约后的 values，没有 `keepdim`；非空 axis tuple 可同时归约多轴。若所有轴都被归约，结果是 scalar，而非零维 tensor。补 size-one 轴时，tensor value 使用 `I.reshape`；scalar 不能 reshape，可用 `I.full` 构造 tensor，或按赋值的广播规则直接写出。

`I.arg_reduce.max(value, axis=...)` 返回 `(values, indices)`，不能把整对结果当 indices。完整归约时两项均为 scalar；保留轴时两项均为保留这些轴的 tensor。分别核对两个 component 的 dtype；写入不同 dtype 的输出前必须显式 `I.cast`，包括 `I.i64` 与 `I.index` 之间，不能因它们都使用 64 bits 就视为同一类型。

Generic `I.reduce(value, axis=..., identity=..., combine=helper)` 的 identity、两组 combine 参数和返回值必须具有删除归约轴后的同一 schema。例如 `[M,N]` 沿 `1` 归约得到 `[M]`，identity 可写成 `I.full((M,), 0.0, dtype=I.f32)`。完整例子见 [reduction.py](examples/reduction.py)。

Python tuple 与 `I.record(field=value, ...)` 是结构化 products，不要求各 component 同 dtype/shape，但每个 component 必须与对应 identity/combine/result 一致。Tuple 静态解构，record 用 `.field`；都不直接成为 host-visible kernel return。

`@intent.fn` 是 typed kernel helper，不是任意 Python 调用；普通 Python `abs/math.*` 不会自动变成 DSL。先查当前 API，使用 `I.abs` 等已声明入口，不猜 `I.log1p`、`I.keepdim` 等名字。Runtime captures 显式传参，structured combine 必须 pure。

## Control、effects 与多个 kernels

普通 `for/while` 保持顺序与 loop carry；`I.parallel(domain)` 表达独立无序点，不允许 carry。Tensor predicate 使用 `I.select`，不控制 statement `if`。`Out` 进入 kernel 时未定义，读取前必须先定义；不能用 InOut 掩盖未定义读取。

一个 kernel 不自动拆成多个 launches。需要多个 kernels 时，host 分别编译，分配中间 tensor，显式依次调用。[split_k_pipeline.py](examples/split_k_pipeline.py) 包含完整 partial/combine kernels 和真实 host 编排；`parts` 是作者可见的算法分解，不是物理 tile 参数。

### Matmul 后的逐行归约与 normalization

选择 kernel 边界时，要权衡中间 tensor 的读写成本、值复用和各阶段可用的并行度。`I.matmul` 等生产阶段之后若要归约某个输出轴，可以先保存生产结果，再在后续 kernel 中归约，使两个阶段分别形成适合的并行划分；显式分块与在线 summary 则可用于避免完整中间 tensor。算法及 kernel 编排由作者表达，各 kernel 内的物理分块、布局和 target 配置由 compiler 形成。

例如 `Y = ReLU(X @ W + bias)` 后沿输出列做 LayerNorm 或 softmax：行数较少时，单个 kernel 中完整行的归约依赖可能限制 matmul 的列方向并行度。作者可先用一个 kernel 生成 `Y`，再用另一个 kernel 逐行归约，使 matmul 的行、列两个结果轴都能独立分块。比较完整算子耗时，计入中间 tensor 的读写和全部 launches，并保持各阶段的 dtype 与数值契约。

### Large/global reduction 与 partial/combine

`I.reduce.sum`、`I.arg_reduce.max` 等输出很少的大归约（large/global reduction），可以由作者显式分段生成 partials，再由后续 kernel 合并，以增加独立工作的数量。[逻辑 partition](../programming-model/logical-program.md) 使用 `I.domain`、`I.parallel` 和 source slice 表达；[split_k_pipeline.py](examples/split_k_pipeline.py) 展示 partial/combine kernels 及 host 的中间 tensor 分配与调用顺序。

分段与合并仍须保持操作的顺序、identity、NaN 和数值契约；arg-reduce 的 partials 还须保留原始逻辑索引，不能把分段内索引当作全局结果。每个 kernel 内的物理分块、布局与 target 配置仍由 compiler 负责。

### Host 编译与调用

Public 调用为 `intent.compile(kernel, compiler=..., target=..., constexprs=...)`，返回 artifact。显式调用 `artifact(...)` 按 kernel 声明顺序传入全部 runtime 参数，`Out` 保留在声明位置；`artifact.run(...)` 只省略 `Out`，其余 views 与 scalars 保持原顺序，由 runtime 分配并返回输出。例如声明顺序为 `A: In, B: Out, scale: f32` 时，调用为 `artifact(A, B, scale)` 或 `artifact.run(A, scale)`。`Constexpr` 在编译时绑定，不传入这两种 runtime 调用。

编译与 launch 分开。`intent.generate` 只生成 source/IR；target 在 host 选择，例如 `intent.targets.TritonTarget()`，不能在 kernel 查询设备或选择 warp/tile。

标量 constexpr 使用 Python 类型注解，例如 `STEP: I.Constexpr[int]`、`EPS: I.Constexpr[float]`、`ENABLED: I.Constexpr[bool]`。`I.f32` 等是 runtime scalar dtype 描述符。

`constexprs` 绑定 kernel 签名中声明的 `I.Constexpr[...]` 参数。View shape 中的 `"M"`、`"K"` 是 logical extent 名字；`M, K = input.shape` 读取这些 extents，不会声明同名 constexpr 参数。只有动态 shape 的 kernel 无需把本次输入尺寸传入 `constexprs`。

评测中的 `build(context)` 只是上述调用的薄适配：在 build 内分别 `context.compile("name", kernel)`，返回一个 host callable，在 callable 中分配中间 tensors 并调用 artifacts。它不改变 DSL，也不要求整个任务只能写一个 kernel。

## 诊断的含义

`unknown intrinsic` 应先核对当前公开名字；`axis must be an integer` 检查是否误传 domain；dtype mismatch 检查 runtime operands 的显式 cast；shape mismatch 检查尾部对齐和真实 index relation。不得通过换输出 shape、忽略 tuple component 或放大容差“修复”任务。

公开规则允许、但 lowering 报 `NotImplementedError` 的程序是实现缺口，不自动成为作者错误。`intent.binary ... schema` 或 `make_tuple ... wrong type` 等非法 compiler IR 需要保留完整原程序与诊断交给 compiler 开发侧，不能反向改写语言规则。MCP 查询只提供原文和声明，不执行程序或判断数值正确性。
