# Domain、Region 与控制

## Logical domain

```python
rows = I.domain(0, M)
cols = I.domain(0, N)
```

Domain 是逻辑索引集合，不是 physical thread、block 或 launch grid。它可以是 dense、strided、product、runtime dynamic，或 ragged descriptor 的 outer/member domain。

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

Source 固定 streamed axis、carry schema、step body、segment order、state update 与 final projection。Compiler 只选择内部 segment extent 和 physical realization。

`state_stream` 不暗示 parallel partial-state merge。若 segment boundary 本身影响算法，就使用 runtime 或 `I.Constexpr` extent。
