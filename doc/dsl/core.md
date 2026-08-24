# Core Surface

## 1. Definitions

```python
@intent.kernel
def kernel(...):
    ...

@intent.fn
def helper(...):
    ...
```

`@intent.kernel` 定义一个 launchable kernel；`@intent.fn` 定义 kernel 内 helper。helper 不能成为隐藏 kernel 或 host dispatch。

## 2. Parameters

```python
x: I.In[I.f16, ("M", "N")]
y: I.Out[I.f16, ("M", "N")]
state: I.InOut[I.f32, ("M",)]
scale: I.f32
CAUSAL: I.Constexpr[bool]
```

- `In` 只读；
- `Out` 由本次调用定义；
- `InOut` 保留输入值并允许写回；
- dtype annotation 的 scalar 是 runtime scalar；
- `Constexpr` 是硬件无关算法 specialization。

## 3. Domains 与 tensor regions

```python
rows = I.domain(0, M)
columns = I.domain(0, N)
value = x[rows, columns]
y[rows, columns] = value * scale
```

`domain(begin, end, step=1)` 使用半开区间。使用多个 domains 索引产生具有对应逻辑轴的 tensor value。`I.indices(domain_or_region)` 返回 logical indices。

不提供作者可见的 physical tile domain。compiler blocking 不改变这些 logical domains。

## 4. Parallel iteration

当算法确实由独立 logical instances 组成时，可以写：

```python
for row in I.parallel(I.domain(0, M)):
    y[row] = f(x[row])
```

`parallel` 只声明无可观察迭代顺序和无跨迭代 state，不绑定任何 GPU execution unit。

整域的纯 tensor assignment 已经表达逐元素定义时，不强制再写 `parallel`。

## 5. Ordered control 与 carry

普通 Python `if`、`for`、`while` lower 为结构化 control。循环中更新的 SSA values 是 loop carry；不需要 `ordered` 标记。

```python
state = init
for index in I.domain(0, N):
    state = transition(state, x[index])
```

compiler 不能把该循环改成无序 reduce，除非作者改用具有 reassociation 合同的 structured operation。

## 6. State stream

```python
stream = I.state_stream(axis, init=initial_state, stop=logical_end)
with stream:
    for region, state in stream:
        stream.yield_(transition(region, state))
result = stream.result
```

`state_stream` 表达顺序保持的 region transition。它不接受作者提供的 physical extent；compiler 选择合法分段。需要固定 part identity 的算法应显式写 part domain。

## 7. Generic reduction

```python
@intent.fn
def combine(lhs, rhs):
    return lhs + rhs

result = I.reduce(
    value,
    axes=(axis,),
    identity=I.cast(0, I.f32),
    combine=combine,
)
```

combine 的输入与输出 schema 必须和 accumulator 一致；不得包含 load/store/atomic/RNG 等 effects。作者选择 reduce，即声明接受满足数值合同的合法 reassociation。

`reduce.sum`、`reduce.max`、`any`、`all` 和 `arg_reduce.max` 都是 generic reduce 的 sugar。

## 8. Scan

```python
prefix = I.scan(
    value,
    axis=axis,
    identity=identity,
    combine=combine,
    inclusive=True,
)
```

scan 保持 logical order 并产生每个位置的 prefix。它不能退化成只返回最终值的 reduce。

## 9. Contract

```python
acc = I.contract(
    a[m, k],
    b[k, n],
    reduce=((1, 0),),
    acc_dtype=I.f32,
)
```

axis relation、batch relation、multiply/combine 语义和 accumulator dtype 是算法合同。physical tile、MMA family、operand layout 和 staging 不属于该调用。

通用 semiring 如果目标矩阵 primitive不支持，可以由作者显式写 pointwise + reduce；compiler 不把不存在的任意 semiring 通道伪装成 contract 支持。

## 10. Ragged relation

```python
groups = I.ragged(
    outer=I.domain(0, G),
    members=I.domain(0, R),
    offsets=offsets,
    indices=optional_member_mapping,
)
```

`groups[g]` 是稳定的 member relation；`I.members(groups[g])` 返回 member logical indices。多个 relation 可以并存，不能按 extent 猜 relation identity。

## 11. Indexed access 与 effects

普通 view indexing、`gather`、`scatter_unique`、`scatter_reduce`、atomic 和 logical buffer 分别表达不同算法合同。越界索引不是默认 clamp；作者可用前置条件声明其合法性。

## 12. `partition` 的收敛

`partition` 不作为 Core。ABI 可见的 `P` 个连续 parts 可由现有 Core 直接表达：

```python
parts = I.domain(0, P)
width = I.ceil_div(N, P)
for part in I.parallel(parts):
    begin = I.minimum(part * width, N)
    end = I.minimum((part + 1) * width, N)
    region = I.domain(begin, end)
    partial[part] = I.reduce.sum(x[region], axis=0, identity=0.0)
```

这里 part identity、边界公式、空 part 结果和 partial ABI 都由作者明确表达。若作者不需要观察 part identity，则不应写 partition，compiler 自己选择 physical blocking。
