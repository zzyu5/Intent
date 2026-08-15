# Domain、Region 与控制

## Logical domain

```python
rows = I.domain(0, M)
cols = I.domain(0, N)
```

Domain 是逻辑索引集合，不是 physical thread、block 或 launch grid。rank-one domain 与 region 都采用半开区间 `[begin, end)`；`end <= begin` 时逻辑集合为空。当前正式语言只承诺 unit-step domain；显式 `step=1`、runtime bound、product 与 ragged descriptor 的 outer/member domain 可以保留各自的 IR flavor。Frontend 在构造 IR 前明确拒绝非 unit-step domain；需要 stride、偏移、整除或取模时，使用 unit-step logical identity 加显式 index relation。只有未来真实算法证明这种表达不足时，才重新打开该语义。

## Region 与位置式 tensor 语义

Region 是 domain 的逻辑子区域，可用于 tensor view slice、parallel work item、reduction/contraction 轴片段、state stream segment 与 gather/scatter indexing relation。

```python
m = I.full((q_region,), -I.inf, dtype=I.f32)
o = I.zeros((q_region, value_dim), dtype=I.f32)
q_idx = I.indices(q_region)
```

`I.indices(region)` 返回原始 logical indices，不返回 region ordinal、program id 或 lane id。

表面语法使用 NumPy/Triton 风格的位置语义：

```python
next_m[:, None]
alpha[:, None]
q_idx[:, None] >= k_idx[None, :]
I.transpose(k_block)
I.reshape(x, ...)
```

规则包括 positional shape、NumPy-style broadcasting、`None`/reshape/transpose，以及 region 作为 symbolic dimension。Physical layout 与 source tensor shape 分离；IR 仍保留 region 的 logical source 与 indexing relation。

Intent 不引入额外的 named-axis/reaxis 类型系统。

## Partition

### 按 extent

```python
for region in I.partition(axis, extent=B):
    ...
```

将 axis 切成连续 regions，每个 region 的最大逻辑长度为 `B`。`B` 可以来自 runtime、`I.Constexpr`，或内部的 `I.auto("TILE")`。

### 按 count

```python
for part, region in I.partition(axis, count=P):
    ...
```

将 axis 分成 source-visible 的 `P` 个连续 regions。它适用于 split-K、partial buffer、host-visible shard 或多-kernel 共同观察的 part identity。

`P` 来自 runtime、shape、`I.Constexpr` 或 wrapper，不接受 `I.auto`。

这是保留的 source 语义，但当前 realizer 尚未实现。Frontend 对 `partition(count=...)` 给出源码定位的明确诊断，不生成一个会在深层失败的 Kernel IR，也不会把它静默改成 `extent` 模式。

显式 `partition` 是算法决定：source body 从“一个 logical element”变成“一个 logical region”，因而可以合法写 region reduction、contraction、scan、区域 mask 或片上复用。Compiler 不能因为某个标量写法性能差，就替作者补一个 partition 并把它改成块算法。

## Parallel

```python
for row in I.parallel(rows):
    ...
```

表示 logical iterations 相互独立。Compiler 可以顺序执行、分配给不同 workers、persistent traversal、SIMD 打包或把同构 point-level 运算 tensorize，但不能给 source body 增加新的可观察 region、state 或 effect。

其中 scalar lane packing 只允许在 body 没有任何依赖“看见一块区域”的语义时使用：每条 physical lane 仍执行一个 source scalar instance。出现 reduction、contract、scan、tensor-valued中间量、logical buffer、atomic/scatter 或 region-level mask 时，必须保持作者写下的 element/region 边界，不能自动打包成另一种算法。

Region-level algorithm 必须在 source 中显式写出 region：

```python
for qr in I.parallel(
    I.partition(q_axis, extent=I.auto("Q_TILE"))
):
    ...
```

## Ordered

```python
for i in I.ordered(axis):
    state = update(state, x[i])
```

`ordered` 固定逐元素 source 顺序。Compiler 可以保持顺序地 strip-mine、unroll、vectorize 或 pipeline，但不能改成 parallel partial reduction。

## State stream

```python
stream = I.state_stream(
    axis,
    extent=I.auto("K_TILE"),
    init=state0,
)

with stream:
    for segment, state in stream:
        next_state = step(segment, state)
        stream.yield_(next_state)

result = stream.result
```

Source 固定 streamed axis、carry schema、step body、segment order、state update 与 final projection。当前 `extent` 只能是 compile-time integer 或 `I.auto(...)`；compiler 只在这份合同内选择内部 segment extent 和 physical realization。

`state_stream` 不暗示 parallel partial-state merge。Runtime 数据可以通过 logical stop 收紧实际读取终点，但不能作为 runtime segment extent；若 segment boundary 本身是 runtime-visible 算法决定，当前 frontend 会明确拒绝该写法。

## 逻辑读取终点

```python
stream = I.state_stream(
    keys,
    extent=I.auto("K_TILE"),
    init=state0,
    stop=I.end(query_region),
)
```

`I.end(x)` 只接受 rank-one domain 或 region，并返回 `x` 在自身逻辑坐标系中的 exclusive endpoint；它不是 physical tile end。空 region 的 endpoint 等于它的 begin。把该值用作 `state_stream.stop` 时，实际迭代集合是 streamed axis 与 `(-∞, stop)` 的交集，仍按原 axis 顺序遍历：`stop <= axis.begin` 时不执行 step、结果等于 initial state；`stop >= axis.end` 时不扩展原 axis；最后一个 segment 可以是 partial segment，其无效 lane 不可影响可观察结果。

`stop` 必须来自 `I.end(domain_or_region)`，且其坐标必须能与 streamed axis 建立同一逻辑索引关系。无法证明或投影这种关系的 realizer 必须在发射前拒绝，不能把 endpoint 当成 shape、segment 数或 physical block ordinal 猜回去。它表达的是作者写下的逻辑读取上界，不是“carry 收敛后提前退出”的数据依赖终止条件。

## 调用前置条件

```python
I.assume_in_bounds(index, view, axis=1)
value = view[row, index]
```

`I.assume_in_bounds(index, view, axis=a)` 是 unsafe 的调用前置条件：对 tensor index，它声明每个元素都满足 `0 <= index < view.shape[a]`；对 scalar index，它声明该标量满足同一关系。声明从当前位置支配的后续访问及其嵌套 region 生效，只匹配同一 SSA index、同一 view/logical buffer 与归一化后的同一 axis；它不回溯影响之前的访问，也不靠“形状相同”匹配别的值。

该构造不产生值，不执行 clamp 或 runtime check，也不是性能 hint。违反声明属于调用方错误，程序语义未定义；空 axis 上任何实际索引都无法满足该前置条件。Compiler 可以用它证明访问合法或消除 mask，target 也可以发射自己的 assumption，但不能在缺少声明时凭数据分布猜测索引安全。
