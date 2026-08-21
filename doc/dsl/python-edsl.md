# Python eDSL

## 包与装饰器

```python
import intent
import intent.language as I
```

```python
@intent.kernel   # runtime-visible kernel entry
@intent.fn       # kernel 内 helper，不产生额外 dispatch
```

`@intent.fn` 封装作者明确选择的算法片段，例如 Welford update、bitonic step、quantization routine 或 routing helper。它不是 opaque compiler service。

## Kernel signature

```python
class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm_kernel(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    alpha: I.f32,
    ACTIVATION: I.Constexpr[Activation],
):
    ...
```

View kinds：

- `I.In`：只读输入；
- `I.Out`：由本次 invocation 定义的输出；
- `I.InOut`：初始内容属于调用约定的读写 tensor。

Signature 可以声明 symbolic shape、dtype、stride/layout constraint、alias/noalias、alignment、runtime scalar 与 user specialization。Compiler-private scratch 不进入 source signature。

## 受限 Python 子集

Kernel body 可以包含：

- scalar 与 tensor SSA；
- tuple/record state；
- domain/region iterator；
- 支持的 `if`、`for`、`while`；
- `@intent.fn` helper；
- structured primitives；
- logical mutable buffer 与显式 effects。

普通 `for`/`while` 中的 Python `break` 与 `continue` 在 frontend AST lowering 时正规化为 loop-carried `live/active` 状态和结构化条件；Kernel IR 不定义 `break`/`continue` operation。普通 `for` 表达 source 顺序，`parallel` 表达实例独立，`state_stream` 表达 segment/carry recurrence；这些语义由 frontend 建成内部 control facts，不要求作者再写优化授权标签。Emitter 只处理结构化 `for`、`while`、`if`、parallel/state 与 SSA carry，不让三个目标各自解释 Python 控制转移。

Tuple 是 source 侧的结构化多值语法：解构、helper 多结果和 loop/stream carry 都按有序 SSA schema lowering。需要通过名称访问字段时使用 `I.record(...)`；tuple 不形成可变 Python object，也不形成 opaque runtime tuple。

解构或赋值目标 `_` 表示丢弃对应结果：它不进入 DSL 环境，也不会形成需要由 structured region 携带的 SSA state。

Kernel body 不允许：

- 任意 Python object mutation；
- host API 调用、文件或网络 I/O；
- 动态 import；
- 依赖 CPython object identity；
- 读取 target worker identity；
- 把 `I.auto` 当作普通 Python 值。

## 三类参数

### Runtime 参数

Runtime scalar 和 tensor shape 每次调用都可以变化，并可参与 runtime control flow。

```python
scale: I.f32
length: I.i32
```

### 用户 specialization

```python
CAUSAL: I.Constexpr[bool]
HEAD_DIM: I.Constexpr[int]
ACTIVATION: I.Constexpr[Activation]
PARTS: I.Constexpr[int]
```

`I.Constexpr` 由 wrapper 绑定，可以控制 source branch、固定 shape specialization、改变 wrapper-visible workspace、选择算法 specialization，并成为 JIT cache key。

### Segment-parametric `I.auto`

```python
I.state_stream(axis, extent=I.auto("KV_TILE"), init=state)
```

`I.auto` 不是普通 tile 值，也不是作者为任意 axis 放置 physical blocking 的 API。它只用于语义本身已经声明 segment-parametric 的 structured construct，例如 `state_stream`：作者写下 segment body 与可组合 state transition，compiler 绑定具体 segment extent。

它不能被读取、进入 source branch、输出 shape、workspace ABI、logical identity、RNG identity 或 effects，也不能用于普通 `partition`。

以下构造不存在：

```python
q_tile = I.auto("Q_TILE")
if q_tile == 64:
    ...

I.partition(axis, count=I.auto("PARTS"))
I.partition(axis, extent=I.auto("TILE"))
```

`partition(count=...)` 的 source 语义要求 part count 来自 runtime、shape、`I.Constexpr` 或 wrapper；完全不可见的 physical worker count 属于 Physical Plan，不能替代 source-visible count。

`state_stream(..., extent=I.auto(...))` 的语义要求作者的 transition 对所有合法连续 segmentation 保持同一 source 数值合同；只有该合同明确容许时，segment/reduction grouping 才能产生相应浮点低位差异。Plan 不能改变 logical workset、state transition、effect、ABI 或 final projection。

## Invocation

```python
kernel(x, y, FLAG=value)
```

用户不提供 Triton 式 `[grid]`。调用根据 target、tensor signature 与 `Constexpr` 获取 specialization 和 Physical Plan，然后提交一次 logical callable invocation。该 callable 可以按 Plan 在当前 stream 内执行多个 compiler-private stages；这些 stages 不进入用户 ABI，也不替代 wrapper 对多个 source kernels 的编排。
