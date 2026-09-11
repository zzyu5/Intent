# 作者速查

本页是 [core.md](core.md) 与[数值规则](types-numerics-and-effects.md)的使用入口，不定义第二套语义。MCP 的 `search` 找概念/API/诊断/示例，`api` 查当前声明及返回规则，`read` 读完整章节和代码。公开声明不等于所有 target 已支持；没有性能测量不能宣称高效。

## 类型、literal 与 shape

在 kernel 中使用 `import intent.language as I`。`I.In/I.Out/I.InOut` 描述外部 views，`I.f32` 等描述 scalar dtype；Python literal 可按上下文实例化，但两个不同 dtype 的 runtime values 必须显式 `I.cast`。例如先把 bf16 输入 cast 到 f32，再和 f32 累加器计算，最后 cast 回输出 dtype。

Tensor 和 view 有 `.shape`；scalar、tuple、record、domain 没有统一 `.shape`。`I.full(shape, fill, dtype)` 产生 tensor value，不分配跨 kernel workspace。`I.dot` 只接受两个 rank-1 tensor，返回 rank-0 tensor `[]`，不是 rank-1 `[1]` 或一个 Python number。Scalar 和 rank-0 tensor 是不同类型；pointwise scalar broadcast 由 frontend 显式表达。

## Domain、index 与 broadcast

`rows = I.domain(0, M)` 是 logical coordinate domain，不是整数 tensor，也不是归约的 axis 编号。`I.indices(rows)` 才产生 logical index tensor。`x[rows, columns]` 读取完整二维 logical region；`I.reduce.sum(value, axis=1)` 的 `1` 指 value 的第二个 axis。

Pointwise 按尾部对齐，允许 scalar/size-one broadcast。`[M]` 与 `[M,N]` 相加不会自动按行匹配：先 `I.reshape(row_value, (M, 1))`；`[N]` 可以直接广播到 `[M,N]`。两个未知 extent 不是因为都 dynamic 就兼容。

`I.transpose(value, permutation)` 显式重排；`I.reshape` 保持 row-major element order，不是任意 data permutation。Domain/subregion 与 integer coordinate tensor 不可互换；索引关系、有效范围和 fill 必须来自作者实际表达。

## Reduce、tuple 与 helpers

`I.reduce.sum/max/any/all` 返回归约后的 values，没有 `keepdim`；需要保留一维时显式 reshape。`I.arg_reduce.max(value, axis=...)` 返回 `(values, indices)`，不能把整对结果当 indices。非空 axis tuple 可同时归约多轴。

Generic `I.reduce(value, axis=..., identity=..., combine=helper)` 的 identity、两组 combine 参数和返回值必须具有删除归约轴后的同一 schema。例如 `[M,N]` 沿 `1` 归约得到 `[M]`，identity 可写成 `I.full((M,), 0.0, dtype=I.f32)`。完整例子见 [reduction.py](examples/reduction.py)。

Python tuple 与 `I.record(field=value, ...)` 是结构化 products，不要求各 component 同 dtype/shape，但每个 component 必须与对应 identity/combine/result 一致。Tuple 静态解构，record 用 `.field`；都不直接成为 host-visible kernel return。

`@intent.fn` 是 typed kernel helper，不是任意 Python 调用；普通 Python `abs/math.*` 不会自动变成 DSL。先查当前 API，使用 `I.abs` 等已声明入口，不猜 `I.log1p`、`I.keepdim` 等名字。Runtime captures 显式传参，structured combine 必须 pure。

## Control、effects 与多个 kernels

普通 `for/while` 保持顺序与 loop carry；`I.parallel(domain)` 表达独立无序点，不允许 carry。Tensor predicate 使用 `I.select`，不控制 statement `if`。`Out` 进入 kernel 时未定义，读取前必须先定义；不能用 InOut 掩盖未定义读取。

一个 kernel 不自动拆成多个 launches。需要多个 kernels 时，host 分别编译，分配中间 tensor，显式依次调用。[split_k_pipeline.py](examples/split_k_pipeline.py) 包含完整 partial/combine kernels 和真实 host 编排；`parts` 是作者可见的算法分解，不是物理 tile 参数。

Public 调用为 `intent.compile(kernel, compiler=..., target=..., constexprs=...)`，返回 artifact；`artifact(*inputs, *outputs)` 显式传输出，`artifact.run(*inputs)` 分配并返回输出。编译与 launch 分开。`intent.generate` 只生成 source/IR；target 在 host 选择，例如 `intent.targets.TritonTarget()`，不能在 kernel 查询设备或选择 warp/tile。

评测中的 `build(context)` 只是上述调用的薄适配：在 build 内分别 `context.compile("name", kernel)`，返回一个 host callable，在 callable 中分配中间 tensors 并调用 artifacts。它不改变 DSL，也不要求整个任务只能写一个 kernel。

## 诊断的含义

`unknown intrinsic` 应先核对当前公开名字；`axis must be an integer` 检查是否误传 domain；dtype mismatch 检查 runtime operands 的显式 cast；shape mismatch 检查尾部对齐和真实 index relation。不得通过换输出 shape、忽略 tuple component 或放大容差“修复”任务。

公开规则允许、但 lowering 报 `NotImplementedError` 的程序是实现缺口，不自动成为作者错误。`intent.binary ... schema` 或 `make_tuple ... wrong type` 等非法 compiler IR 需要保留完整原程序与诊断交给 compiler 开发侧，不能反向改写语言规则。MCP 查询只提供原文和声明，不执行程序或判断数值正确性。
