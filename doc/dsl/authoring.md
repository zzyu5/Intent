# 作者速查

本页是 [core.md](core.md) 与[数值规则](types-numerics-and-effects.md)的使用入口，不定义第二套语义。MCP 的 `search` 找概念/API/诊断，`api` 查当前声明及返回规则，`read` 读语言规则与接口说明。公开声明不等于所有 target 已支持；没有性能测量不能宣称高效。

## 类型、literal 与 shape

程序使用 `import intent` 和 `import intent.language as I`；kernel/helper 分别用 `@intent.kernel`、`@intent.fn` 声明，装饰器不在 `I` 命名空间。`I.In/I.Out/I.InOut` 描述外部 views，必须同时给出 dtype 和 shape，例如 `I.In[I.f32, ("M", "N")]`；rank-0 view 的 shape 写 `()`。`I.f32` 等描述 scalar dtype；Python literal 可按上下文实例化，但两个不同 dtype 的 runtime values 必须显式 `I.cast`。例如先把 bf16 输入 cast 到 f32，再和 f32 累加器计算，最后 cast 回输出 dtype。

Literal 首次形成 runtime value 时若没有 expected dtype，Python `bool/int/float` 分别采用 `bool/i64/f64`。需要 f32 的循环状态可用 `I.cast(1.0, I.f32)` 初始化；后续使用不会反向改变它的 dtype。

`I.select` 的 bool 条件不提供数值分支的 expected dtype。两个分支都写成 literal 时，不要从生成条件的 tensor 推断结果 dtype；例如需要 f32 符号值时写 `I.cast(I.select(mask, -1.0, 1.0), I.f32)`。已经产生的 runtime value 不会因后续与 f32 相乘而重新实例化。

Tensor 和 view 有 `.shape`；scalar、tuple、record、domain 没有统一 `.shape`。`I.full(shape, fill, dtype)` 产生 tensor value，不分配跨 kernel workspace。`I.dot(lhs, rhs, acc_dtype=...)` 必须显式指定累加 dtype，只接受两个 rank-1 tensor，返回 rank-0 tensor `[]`，不是 rank-1 `[1]` 或一个 Python number。Scalar 和 rank-0 tensor 是不同类型；需要将 scalar 放入 rank-0 tensor 的分支或 carry schema 时，可用 `I.full((), value, dtype=...)` 显式构造。Pointwise scalar broadcast 由 frontend 显式表达。

## Domain、index 与 broadcast

`rows = I.domain(0, M)` 是 logical coordinate domain，不是整数 tensor，也不是归约的 axis 编号。Domain 本身不参与数值算术或比较；需要坐标值时先使用 `I.indices(rows)` 产生 logical index tensor。`x[rows, columns]` 读取完整二维 logical region；`I.reduce.sum(value, axis=1)` 的 `1` 指 value 的第二个 axis。

Pointwise 按尾部对齐，允许 scalar/size-one broadcast。`[M]` 与 `[M,N]` 相加不会自动按行匹配：先 `I.reshape(row_value, (M, 1))`；`[N]` 可以直接广播到 `[M,N]`。两个未知 extent 不是因为都 dynamic 就兼容。

`I.transpose(value, permutation)` 显式重排；permutation 的长度必须等于输入 rank，包含 `0..rank-1` 的每个轴且仅一次。省略 permutation 时反转全部轴顺序。`I.reshape` 保持 row-major element order，不是任意 data permutation。Domain/subregion 与 integer coordinate tensor 不可互换；索引关系、有效范围和 fill 必须来自作者实际表达。

Domain 索引按资源索引顺序形成读取结果的 tensor axes，赋值仍按 positional axes 对齐，不按 domain 变量名自动换轴。例如两个等长 domains 下，`output[rows, columns] = input[columns, rows]` 不表示矩阵转置；应显式转置读取的 tensor value，或构造具有所需对应关系的坐标 tensor。不同 subregions 的动态长度也不会因本次输入碰巧等长而成为同一 extent；需要使用已成立的 shape relation，或在共同输出 domain 上表达坐标映射。

多个 tensor indices 按 broadcast 规则形成共同的索引 shape，坐标逐位置配对，不自动形成 Cartesian product；所有 tensor indices 共同贡献一份 broadcast shape，每个 domain index 另外引入一个独立 logical axis。Tensor index 的 size-one 轴也属于这份 shape，不会因旁边有 domain 而自动消失。索引赋值的右值必须能 broadcast 到该索引表达式的结果 shape，目标 view 不会替右值隐式降维。例如二维逐元素按第 0 轴 gather，使用 `index=(indices[rows, columns], I.reshape(I.indices(columns), (1, N)))`。这里第二项是可广播的列坐标 tensor，直接传 `columns` domain 会额外引入一个轴。

## Reduce、tuple 与 helpers

普通 `I.reduce` 声明 combine 可结合、可交换，允许并行重结合与重排；不承诺输入顺序或逐 bit 重现。`I.scan` 保留各个 logical prefix 的成员顺序，只允许保序重结合；严格顺序累加使用普通 loop。这些是不同的 operation 合同。

`I.reduce.sum/max/any/all` 返回归约后的 values，没有 `keepdim`；非空 axis tuple 可同时归约多轴。若所有轴都被归约，结果是 scalar，而非零维 tensor。补 size-one 轴时，tensor value 使用 `I.reshape`；scalar 不能 reshape，可用 `I.full` 构造 tensor，或按赋值的广播规则直接写出。

`I.arg_reduce.max(value, axis=...)` 返回 `(values, indices)`，不能把整对结果当 indices。Indices 是输入 tensor 被归约轴内从 0 开始的位置，不自动返回底层 domain/subregion 的绝对坐标。完整归约时两项均为 scalar；保留轴时两项均为保留这些轴的 tensor。分别核对两个 component 的 dtype；写入不同 dtype 的输出前必须显式 `I.cast`，包括 `I.i64` 与 `I.index` 之间，不能因它们都使用 64 bits 就视为同一类型。

Generic `I.reduce(value, axis=..., identity=..., combine=helper)` 的 identity、两组 combine 参数和返回值必须具有删除归约轴后的同一 schema。例如 `[M,N]` 沿 `1` 归约得到 `[M]`，identity 可写成 `I.full((M,), 0.0, dtype=I.f32)`。这里检查的是进入 combine 的 dtype；builtin 的 `acc_dtype` 或默认 widening 会先转换 source，不要求原 external view 与 accumulator 同 dtype。

Python tuple 与 `I.record(field=value, ...)` 是结构化 products，不要求各 component 同 dtype/shape，但每个 component 必须与对应 identity/combine/result 一致。Tuple 静态解构，record 用 `.field`；都不直接成为 host-visible kernel return。

`@intent.fn` 是 typed kernel helper，不是任意 Python 调用；普通 Python `abs/math.*` 不会自动变成 DSL。先查当前 API，使用 `I.abs`、`I.sqrt` 等已声明入口，不猜 `I.log1p`、`I.keepdim` 等名字。Runtime captures 显式传参，structured combine 必须 pure。

## Control、effects 与多个 kernels

普通 `for/while` 保持顺序与 loop carry；carry 的初值与每轮更新必须保持 dtype、rank 和逻辑 shape，循环体内的 broadcast 不会改变初始 schema。`I.parallel(domain)` 表达独立无序点，不允许 carry。Tensor predicate 使用 `I.select`，不控制 statement `if`。`Out` 进入 kernel 时未定义，读取前必须先定义；不能用 InOut 掩盖未定义读取。

一个 kernel 不自动拆成多个 launches。多个 kernels 由 host 分别编译、显式调用；跨 kernel tensors 的分配与生命周期由 host 管理。Kernel 数量与算法编排由作者定义，各 kernel 内的物理分块、布局与 target 配置由 compiler 形成。

### Host 编译与调用

Public 调用为 `intent.compile(kernel, compiler=..., target=..., constexprs=...)`，返回 artifact。显式调用 `artifact(...)` 按 kernel 声明顺序传入全部 runtime 参数，`Out` 保留在声明位置；`artifact.run(...)` 只省略 `Out`，其余 views 与 scalars 保持原顺序，由 runtime 分配并返回输出。例如声明顺序为 `A: In, B: Out, scale: f32` 时，调用为 `artifact(A, B, scale)` 或 `artifact.run(A, scale)`。`Constexpr` 在编译时绑定，不传入这两种 runtime 调用。

编译与 launch 分开。`intent.generate` 只生成 source/IR；target 在 host 选择，例如 `intent.targets.TritonTarget()`，不能在 kernel 查询设备或选择 warp/tile。

标量 constexpr 使用 Python 类型注解，例如 `STEP: I.Constexpr[int]`、`EPS: I.Constexpr[float]`、`ENABLED: I.Constexpr[bool]`。`I.f32` 等是 runtime scalar dtype 描述符。

`constexprs` 绑定 kernel 签名中声明的 `I.Constexpr[...]` 参数。View shape 的静态 extent 使用非负整数，如 `1`；字符串必须是合法符号名，如 `"M"`、`"K"`，`"1"` 不是整数 extent。`M, K = input.shape` 读取这些 extents，不会声明同名 constexpr 参数。只有动态 shape 的 kernel 无需把本次输入尺寸传入 `constexprs`。

评测中的 `build(context)` 只是上述调用的薄适配：在 build 内分别 `context.compile("name", kernel)`，返回一个 host callable，在 callable 中分配中间 tensors 并调用 artifacts。它不改变 DSL，也不要求整个任务只能写一个 kernel。

## 诊断的含义

`unknown intrinsic` 应先核对当前公开名字；`axis must be an integer` 检查是否误传 domain；dtype mismatch 检查 runtime operands 的显式 cast；shape mismatch 检查尾部对齐和真实 index relation。不得通过换输出 shape、忽略 tuple component 或放大容差“修复”任务。

公开规则允许、但 lowering 报 `NotImplementedError` 的程序是实现缺口，不自动成为作者错误。`intent.binary ... schema` 或 `make_tuple ... wrong type` 等非法 compiler IR 需要保留完整原程序与诊断交给 compiler 开发侧，不能反向改写语言规则。MCP 查询只提供原文和声明，不执行程序或判断数值正确性。
