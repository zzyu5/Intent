# Logical Program

## 1. Domain、region 与 identity

domain 表示一个有序 logical index set。最基本的 domain 是半开区间：

```text
[begin, end), step > 0
```

空 domain 合法。domain 的 begin/end 可以来自 runtime shape；logical index 类型具有跨 target 一致的整数语义。

region 是 domain 的逻辑子区间或子集。region 保留其来源 domain 与 logical identity，因此：

- `indices(region)` 返回原 domain 中的 logical indices；
- region 长度可以为零或运行时值；
- region 不携带 program/block/thread ownership；
- physical padding 不改变 region 的逻辑成员。

## 2. 整域张量运算

使用 domain/region 索引 view 会得到逻辑张量值。张量值的 shape 来自所用 domains/regions；广播、reshape、transpose 和 join 都是算法值变换。

对一个 region 的唯一写入定义该 region 上的结果，不要求作者先写 physical loop。编译器可以分块或向量化，只要每个 logical element 的定义与 effect 合同保持不变。

## 3. Parallel independence

`parallel(domain)` 只声明不同 logical iterations 独立：

- 迭代之间没有 loop-carried value；
- effect 不能互相冲突，除非作者使用具有明确冲突语义的 scatter/atomic；
- 执行顺序不可观察；
- 它不指定 grid dimension、program count、thread、lane 或 tile。

当整域张量表达式已经具有唯一写入和纯依赖时，编译器也可以从数据流证明独立性；作者不必为每一个 pointwise tensor expression 包一层 `parallel`。

## 4. Ordinary control flow

普通 `if`、`for` 和 `while` 保留程序式语义：

- `if` 通过 SSA merge 合并分支值；
- `for` 按 logical iteration order 执行，并可以携带值；
- `while` 按作者条件推进状态；
- `break`、`continue` 和 early-exit 若出现在表面语法中，必须归一化为等价的结构化控制，而不是静默丢失。

普通有序程序不需要额外的 `ordered` 标记。是否能够并行、向量化或重结合，应由具体构造的语义决定。

## 5. State stream

`state_stream(domain, init, stop)` 表达按 logical order 推进的分段状态递推。它与普通 scalar loop 的区别是：body 看见一个 logical region，而不是单个 index。

语义合同是：

- 遍历范围是 `[domain.begin, stop)`，`stop` 为开区间终点；
- `stop` 不得超出 source domain；
- compiler 可以选择任意保持顺序、无重叠且完整覆盖读取范围的分段；
- body 对每个 region 和当前 state 计算下一 state；
- 空读取范围直接产生 `init`；
- 作者选择该构造，表示接受在声明的数值合同内改变合法分段；若算法依赖固定分段身份或长度，应显式构造 part domain 与 subdomains，而不能使用 `state_stream`。

physical segment/tile extent 不属于该构造的作者表面。

## 6. Structured operations

`reduce`、`scan` 和 `contract` 是算法结构，不是 target primitive 请求：

- reduce 声明 axes、typed accumulator schema、identity 与 associative combine；
- scan 额外声明 inclusive/exclusive 与 logical order；
- contract 声明 operand axes relation、batch axes、combine/multiply 语义和 accumulator dtype。

作者使用这些构造，是在声明编译器可采用该构造允许的合法 reassociation；作者不指定 reduction tree、MMA、warp participants 或 shared staging。

## 7. Ragged 与间接关系

ragged relation 显式关联：

- outer logical domain；
- member logical domain；
- offsets；
- 可选 member index mapping。

它描述算法读取集合，不描述物理 gather tile。多个 ragged relations 可以同时存在；relation identity 必须稳定，不能靠 shape 或 extent 猜测。

数据产生的索引默认由调用方满足前置条件。作者可以声明 in-bounds、sorted、unique 等合同；编译器不自动 clamp 或改变索引。

## 8. Effects 与资源

纯 SSA 值不携带 effects。external views 与 logical mutable buffers 上的操作必须显式区分：

- read/write；
- unique scatter；
- reduction scatter；
- atomic operation 与其 scope/order；
- deterministic counter-based RNG。

effects 限制重排、复制和删除。编译器选择 physical access form，但不能把 unique write 改成另一种冲突语义，也不能用串行 fallback 冒充 target 原生支持。
