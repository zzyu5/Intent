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
    mean = a.mean + delta * b.n / n
    m2 = a.m2 + b.m2 + delta * delta * a.n * b.n / n
    return I.record(n=n, mean=mean, m2=m2)


stats = I.reduce(
    partial_stats,
    axis=0,
    identity=I.record(n=0, mean=0.0, m2=0.0),
    combine=welford_combine,
)
```

上例保留的是 generic combine 将来应有的 source 形态，不是当前可运行示例。当前 Kernel IR/realization 只闭合内建 reduction/scan combiner；任意 helper closure 尚未进入 typed Kernel IR region，编译器会在进入 target emission 前明确拒绝。Welford 数值算法目前可用 `state_stream`、内建块内归约与普通标量 carry 合并表达，但这不等于 generic combine 通道已经实现。

选择 `reduce` 表示 compiler 可以选择物理 reduction tree 与 hierarchy。需要严格逐元素顺序时使用 `ordered`，不增加 `mergeable` 或 `@associative` 合同。

一个 logical reduction 可以在同一 target entry 内使用 serial strip-mine、SIMD horizontal reduction、warp/block tree、private partial、compiler-private scratch 或 target 允许的 atomic accumulation。

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

## Contraction

```python
acc = I.contract(
    a_block,
    b_block,
    reduce=((1, 0),),
    acc_dtype=I.f32,
)
```

`reduce` 使用 positional reduction-axis pairs。Source 固定 operand shapes、配对归约轴、operand dtype、multiply/combine、accumulator dtype 与 epilogue tensor-flow。

Realizer 决定依赖算法结构的 reduction subtile、复用边界和 primitive 数值角色，并把它投影到 MMA/`tl.dot`/`T.gemm`/cuTile matmul/CPU-RVV FMA microkernel。具体 layout、寄存器分配、指令选择与给定参数后的低层 pipeline 交给目标 compiler。

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
