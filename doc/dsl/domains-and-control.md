# Domain、Region 与控制

## Logical domain

```python
rows = I.domain(0, M)
cols = I.domain(0, N)
```

Domain 是逻辑索引集合，不是 physical thread、block 或 launch grid。rank-one domain 与 region 都采用半开区间 `[begin, end)`；`end <= begin` 时逻辑集合为空。Core domain 采用 unit-step；显式 `step=1`、runtime bound、product 与 ragged descriptor 的 outer/member domain 可以保留各自的 IR flavor。Stride、偏移、整除或取模通过 unit-step logical identity 加显式 index relation 表达，非 unit-step domain 不属于该 Core 合同。

## Region 与位置式 tensor 语义

Domain 可以直接作为 tensor indexing 的完整逻辑维度；作者不需要先把它切成 region 才能写 tensor expression、reduction 或 contraction。Source region 是 domain 的作者可观察子区域，用于真正依赖 segment identity/boundary 的 tensor view、partial result、state transition、window/page 或 gather/scatter relation；compiler 从完整 domain 引入的 physical region 不形成 source Region，也不能被 source body 观察。

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

## Source-visible partition

`partition` 只表达作者可观察的逻辑分段，不承担普通 GPU blocking。判断标准不是 body 是否在语法上拿到了一个 region，而是换一种合法机器实现后，part identity 和 boundary 是否仍必须保持。

下面这些情况属于 source partition：

- part index 写入输出或索引 wrapper-visible partial buffer；
- region boundary 进入 mask、RNG identity、effect 或显式 index relation；
- body 在一个 part 内共同归约、扫描或维护跨成员 state，并且该 part result 可观察；
- page、window、chunk 或稀疏 block 是输入数据格式/算法本身的一部分；
- 多个 source kernels 或 wrapper 共同观察同一 partition count/identity。

仅仅为了把完整 GEMM 的 M/N 轴、MoE route 轴或独立 query 轴凑成块张量，不构成 source partition。Compiler 可以从完整 logical domain、`parallel` independence 和 structured op 自行形成 physical region。

### 按 extent

```python
for region in I.parallel(I.partition(axis, extent=B)):
    ...
```

将 axis 切成连续 regions，每个 region 的最大逻辑长度为作者可观察的 `B`。`B` 来自 runtime、shape、`I.Constexpr` 或 wrapper，并参与 source specialization/ABI/algorithm relation。纯 compiler-owned tile 不使用这个构造。

### 按 count

```python
for part, region in I.parallel(I.partition(axis, count=P)):
    ...
```

设 axis 长度为 `N`，`block = ceil(N / P)`。`partition(axis, count=P)` 产生 source-visible 的 part identity `i ∈ [0, P)`；第 `i` 个 region 是半开区间

```text
[min(i * block, N), min((i + 1) * block, N))
```

因此它与 `partition(axis, extent=ceil(N/P))` 使用同一套连续切分和普通 tail 语义。尾部 part 可以较短或为空；特别地，`P > N` 时多出来的 part 为空。空 part 保留 identity 和 wrapper-visible ABI slot，但不执行 body，也不产生写出或 effect。需要读取全部 `P` 个 partial slots 的后续 kernel，wrapper 必须先按该 reduction 的 identity 初始化 buffer。

它适用于 split-K、partial buffer、host-visible shard 或多-kernel 共同观察的 part identity。

`P` 来自 runtime、shape、`I.Constexpr` 或 wrapper，并满足 `P >= 1`；runtime 非正值在 launch specialization 时拒绝。Part identity 是 source 值，不能由 physical worker count 替代。Target 可以不启动空 part 对应的 physical worker，但不能压缩或重编号非空 part identity。

`I.partition(..., extent=I.auto(...))` 不属于作者编程模型。它让作者预先决定“这里必须存在某层 physical blocking”，却没有给出 source-visible boundary。需要任意 segment composition 的 recurrence 使用 `state_stream`；普通 tensor computation 直接使用完整 domain，由 compiler 形成 physical regions。

## Parallel

```python
for row in I.parallel(rows):
    ...
```

表示 logical iterations 相互独立：一个实例不读取另一个实例的 private state，也不存在由 source iteration order 定义的依赖。它不指定 program id、grid、worker、lane 或 tile。

Compiler 可以顺序执行、分配给不同 workers、persistent traversal、SIMD 打包，或把多个独立实例批处理成一个 tensor/matrix primitive。即使单个实例包含 reduction、contract、scan、logical buffer 或 effect，也不因为物理 batching 就自动成为另一种算法；前提是实例之间没有 source-defined happens-before，并且目标实现逐实例保持 value/state/effect identity、冲突语义和结果。Compiler 不能让 source body 观察到新建 physical region，也不能引入跨实例 reduction/state/effect；目标无法保持完整 effect 合同时必须拒绝该 realization。

## 普通顺序循环

```python
for i in axis:
    state = update(state, x[i])
```

普通 `for` 按 logical iteration order 执行，并自然携带循环中的 SSA state 与 effects。Compiler 可以在保持逐实例 happens-before 的前提下 strip-mine、unroll、vectorize 或 pipeline，但不能改成 unordered partial merge。

顺序不是作者额外授予或拒绝并行化的标签。Frontend 必须从普通循环建立内部 sequential/ordered fact；public source 不要求再包一层 `I.ordered(...)`。需要没有 source order 的独立实例时，作者明确使用 `I.parallel(...)`。

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

Source 固定 streamed axis、carry schema、step body、segment order、state update 与 final projection。`state_stream` 的 body 看见一个 segment，并可以在 segment 内执行 reduction/contraction 后把 state 合并到下一段；因此“存在分段”是算法结构，不是普通 blocking hint。

固定正整数/`I.Constexpr` extent 表达作者可观察的固定 segmentation。`I.auto(...)` 表达 segment-parametric recurrence：作者声明任意合法连续 segmentation 依序组合都满足同一 source 数值合同，具体 segment extent 由 compiler 绑定；若算法要求固定逐元素舍入顺序或 boundary-dependent 行为，就不能使用 `auto`。这是 `state_stream` 自身的语义维度，不是为普通 loop 发放优化权限；它仍保持 segment order，也不等同于 unordered reduction。严格逐元素 recurrence 使用普通 `for`。

`state_stream` 不暗示 parallel partial-state merge。Runtime 数据可以通过 logical stop 收紧实际读取终点；runtime-visible fixed segment boundary 必须由 source algorithm 显式表达，不能伪装成 compiler-owned extent。

当 streamed axis 是一个 source partition region 时，stream 的逻辑遍历集合就是该 region 与 `stop` 的交集，carry 只跨该 part 内的连续 segments 传播；part identity、空 part 和 count ABI 仍按上节规则保留。不同 parts 的 carry 不会被 compiler 自动合并，跨 part 的 partial 合并必须由作者在后续 kernel 或 wrapper 中明确写出。这是 `partition(count)` 与 `state_stream` 的组合语义，不是 compiler-private split-K。

## 逻辑读取终点

```python
stream = I.state_stream(
    keys,
    extent=I.auto("K_TILE"),
    init=state0,
    stop=query_index + 1,
)
```

`state_stream.stop` 是与 streamed axis 同一逻辑坐标系中的 exclusive integer endpoint，可以来自普通 index expression；`I.end(x)` 是从 rank-one domain 或 region 取得其 exclusive endpoint 的 convenience，它不是 physical tile end。空 region 的 endpoint 等于它的 begin。实际迭代集合是 streamed axis 与 `(-∞, stop)` 的交集，仍按原 axis 顺序遍历：`stop <= axis.begin` 时不执行 step、结果等于 initial state；`stop >= axis.end` 时不扩展原 axis；最后一个 segment 可以是 partial segment，其无效 lane 不可影响可观察结果。

Frontend/KIR 必须保存 stop expression 的 SSA provenance；realizer 必须证明它能投影到 streamed axis 的坐标关系。无法证明或投影时必须在发射前拒绝，不能把 endpoint 当成 shape、segment 数或 physical block ordinal 猜回去。它表达的是作者写下的逻辑读取上界，不是“carry 收敛后提前退出”的数据依赖终止条件。

## 调用前置条件

```python
I.assume_in_bounds(index, view, axis=1)
value = view[row, index]
```

`I.assume_in_bounds(index, view, axis=a)` 是 unsafe 的调用前置条件：对 tensor index，它声明每个元素都满足 `0 <= index < view.shape[a]`；对 scalar index，它声明该标量满足同一关系。声明从当前位置支配的后续访问及其嵌套 region 生效，只匹配同一 SSA index、同一 view/logical buffer 与归一化后的同一 axis；它不回溯影响之前的访问，也不靠“形状相同”匹配别的值。

该构造不产生值，不执行 clamp 或 runtime check，也不是性能 hint。违反声明属于调用方错误，程序语义未定义；空 axis 上任何实际索引都无法满足该前置条件。Compiler 可以用它证明访问合法或消除 mask，target 也可以发射自己的 assumption，但不能在缺少声明时凭数据分布猜测索引安全。
