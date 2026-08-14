# Tensor-flow 与 Core primitives

Intent 的表达力来自通用 tensor-flow、控制流、logical buffer 与 effects，不依赖常见算子名字。

## Pure tensor SSA

```python
z = I.exp(x - m)
```

它只定义 tensor value 与数据依赖，不要求独立 buffer、固定 address space、固定执行次数或物理阶段。

Realizer 可以 inline、CSE、fusion、rematerialization、spill/reload、vectorization 与合法 instruction selection。因此 source 不提供 `stage`、`materialize`、`resident`、`recompute` 或 `no_remat`。

## Pointwise math 与 logical mask

```python
p = I.exp(x)
p2 = I.exp2(x)
y = I.log(x)
z = I.rsqrt(x)
hi = I.maximum(x, y)
lo = I.minimum(x, y)

masked = I.mask(values, valid=predicate, fill=0)
```

用户明确选择数学表达。`I.LOG2E` 是 `log2(e)` 的语言常量，用于把自然指数表达式显式改写到 `exp2` 路径。`I.mask` 固定 logical predicate 和 fill relation，不固定 GPU predicate 或物理 tail 机制。

NumPy-style positional broadcasting 是 source 语义；frontend 将每个实际扩张的 operand 正规化为显式 Kernel IR `broadcast` node。Backend 因而消费已经确定的 logical result shape，不重新推断 source broadcasting。

## Reduction

```python
m = I.reduce.max(x, axis=0, identity=-I.inf)
s = I.reduce.sum(x, axis=0, identity=0.0, acc_dtype=I.f32)
```

用户可以定义复合 combiner：

```python
@intent.fn
def welford_combine(a, b):
    n = a.n + b.n
    delta = b.mean - a.mean
    safe_n = I.maximum(n, I.cast(1, I.i32))
    mean = a.mean + delta * I.cast(b.n, I.f32) / I.cast(safe_n, I.f32)
    m2 = (
        a.m2
        + b.m2
        + delta * delta * I.cast(a.n * b.n, I.f32) / I.cast(safe_n, I.f32)
    )
    return I.record(n=n, mean=mean, m2=m2)


stats = I.reduce(
    partial_stats,
    axis=0,
    identity=I.record(
        n=I.cast(0, I.i32),
        mean=I.cast(0.0, I.f32),
        m2=I.cast(0.0, I.f32),
    ),
    combine=welford_combine,
)
```

上例是当前正式 generic combine 形态。Frontend 把 helper lower 成 canonical Kernel IR 中的 typed combiner body：参数是两组 accumulator components，返回 schema 与 accumulator 完全一致，identity 逐 component 显式给出。Combiner 必须 pure，不能包含 load/store/atomic/RNG；runtime capture 必须通过 `combine_operands=(...)` 成为 reduce/scan 的显式 scalar operand，只有 `Constexpr` 可以直接捕获。

选择 `reduce` 表示作者接受合法 reassociation，compiler 不反向证明 closure 的数学结合律或 identity law，可以选择物理 reduction tree 与 hierarchy。作者给出的 identity 必须对 closure 真正中性，closure 也必须能处理 identity 与 identity 的组合；上例用 `safe_n` 保证物理尾块中的空 partial 不产生除零，同时零 identity自然保持零 mean/M2。需要严格逐元素顺序时使用 `ordered` 或 `state_stream`，不增加 `mergeable` 或 `@associative` 合同。

一个 logical reduction 可以在同一 callable 内使用 serial strip-mine、SIMD horizontal reduction、warp/block tree、private partial、compiler-private scratch 或 target 允许的 atomic accumulation。Triton/cuTile 将 typed helper 机械投影到原生 generic reduce；TileLang 0.1.13 的当前 PrimFunc surface 没有等价入口，因此 generic closure 明确 unsupported，固定内建 reduction 不受影响。

`I.arg_reduce.max` 是 convenience sugar：frontend 将 `(value, index)` 与 lowest-index tie closure lower 成同一个 tuple-valued generic reduction。目标可以使用经过语义对齐的原生 `max_with_index`，但 Kernel IR helper仍是权威语义。

## Scan

```python
prefix = I.scan(
    x,
    axis=0,
    identity=0,
    combine=I.add,
    inclusive=True,
)
```

Source 固定 logical prefix relation，realizer 决定物理 scan hierarchy。

`I.scan` 与 `I.reduce` 共用 typed combiner、component identity、purity 和显式 capture 合同。Triton/cuTile 委托给原生 associative scan；长轴可以由 Plan 选择 block-local scan 加 block 间 scalar carry。TileLang 当前只支持固定 combiner的机械投影，generic scan明确 unsupported。

## Contraction

```python
acc = I.contract(
    a_block,
    b_block,
    reduce=((1, 0),),
    acc_dtype=I.f32,
)
```

`reduce` 使用 positional reduction-axis pairs。Source 固定 operand shapes、配对归约轴、operand dtype、multiply-add 数值角色、accumulator dtype 与 epilogue tensor-flow。

Realizer 决定依赖算法结构的 reduction subtile、复用边界和 primitive 数值角色。当前 GPU target 将其投影到 MMA/`tl.dot`/`T.gemm`/cuTile matmul；未来 CPU/RVV target 可投影到自己的 FMA microkernel。具体 layout、寄存器分配、指令选择与给定参数后的低层 pipeline 交给目标 compiler。

Generic reduce/scan closure 不会使 `contract` 自动变成任意 semiring。当前 `contract` 只覆盖目标矩阵原语支持的 multiply/add 与 dtype/accumulator 组合；其他 semiring 必须由作者显式写成 pointwise + reduce，或由未来 target capability正式扩展。

显式缩窄必须写在 source 中：

```python
p_low = I.cast(p, I.f16)
acc = I.contract(p_low, v, reduce=((1, 0),), acc_dtype=I.f32)
```

## Gather、scatter 与 effects

```python
x = I.gather(src, index=indices, valid=valid, fill=0)
I.scatter_unique(dst, index=indices, value=values)
I.scatter_reduce(dst, index=indices, value=values, combine=I.add)
```

Source 固定 logical index relation、invalid/fill、duplicate conflict semantics、combine 与明确要求的 memory order。Realizer 决定 coalescing、vector gather/scatter、privatization、atomics 与 physical scheduling。

索引关系可以使某个输出块的输入覆盖范围更大并与相邻块重叠。Physical Plan 可以保存该 access footprint，但不会默认先物化一份去重 halo：现有目标上这种物化比直接使用目标原生块读取更慢。类似地，同一个边界谓词不会默认展开成每元素搬运；能用整块守卫、收紧范围、checked transfer 或 mask 时优先保留整块结构。

Effectful 操作显式写出：

```python
I.atomic_add(...)
I.atomic_cas(...)
I.store(...)
I.mutable_load(...)
```

Effects 不能被非法复制、删除或跨依赖重排。
三个目标语言没有共同的显式 fence 语义，当前 public API 不导出 `I.fence(...)`；未来若加入，必须先定义 scope、ordering 与 participant 合同，不能降成 no-op。

## Logical buffer

```python
work = I.buffer(shape=(region, D), dtype=I.f32, init=0.0)
```

`I.buffer` 是 kernel 内 logical mutable object：不进入 ABI，不指定 register/shared/global，不提供 grid-wide barrier。Placement、layout 和必要的局部同步由 realizer 决定。跨-kernel workspace 必须由 wrapper 分配并作为参数传入。

## Ragged descriptor

```python
groups = I.ragged(
    outer=I.domain(0, E),
    offsets=expert_offsets,
    indices=route_ids,
)
```

`I.ragged` 只描述调用方已提供的 membership，不执行 grouping、不生成 offsets，也不在 histogram、sort 与 atomic bucket 之间选择算法。

Member domain 可以继续内部 partition：

```python
for rr in I.parallel(
    I.partition(groups[expert], extent=I.auto("ROUTE_TILE"))
):
    ...
```

## RNG

```python
r = I.random(seed, logical_index)
```

Counter identity 来自 source logical index，不来自 `program_id` 或 auto-region ordinal。

`I.random` 是纯函数，不持有隐式 RNG 状态。当前 canonical 合同为
`counter_xorshift32(seed, logical_index)`：计数器与 seed 先转成 `u32`，与
`1831565813` 异或后依次执行 `x ^= x << 13`、`x ^= x >> 17`、
`x ^= x << 5`，最终用高 24 bit 生成 `[0, 1)` 的 `f32`。相同 seed 与逻辑
坐标在前向、反向以及不同表面语言中必须得到同一个值。
