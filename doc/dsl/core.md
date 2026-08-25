# 语言构造

## 1. Definitions

```python
@intent.kernel
def kernel(...):
    ...

@intent.fn
def helper(...):
    ...
```

`@intent.kernel` 定义一个 logical kernel computation；外部编译调用选择 target。`@intent.fn` 定义 kernel 内 typed helper，不形成隐藏 kernel或host dispatch。

helper可以返回 scalar、tensor、tuple或record，并可包含调用点本来允许的effects。runtime captures必须成为显式参数，只有不可变 `Constexpr` 可以被词法捕获；递归、target query与kernel launch非法。

Helper边界不会把logical coordinate降级为无来源的integer tensor。传入或返回的tensor/tuple/record components若源自`I.indices`、subregion或indexed relation，canonical KIR必须保留它们的source identity、axis mapping与typed coordinate expression。Helper是否inline不得改变这些facts。

## 2. Parameters

```python
x: I.In[I.f16, ("M", "N")]
y: I.Out[I.f16, ("M", "N")]
state: I.InOut[I.f32, ("M",)]
scale: I.f32
CAUSAL: I.Constexpr[bool]
```

- `In` 只读；
- `Out` 进入kernel时不可读，作者必须在读取前定义其内容；
- `InOut` 保留输入内容并允许写回；
- dtype annotation的scalar是runtime scalar；
- `Constexpr` 只选择硬件无关算法分支。

`I.Enum`是用户定义的有限、非空、封闭、具名constexpr类型。成员名与编译期枚举值在该类型内唯一；值只能作为`Constexpr[EnumType]`参数、constexpr默认值、分支条件或helper constexpr capture。它不是runtime scalar或tensor dtype，不进入external view/logical-buffer ABI，不同enum类型之间不隐式比较或转换，也不继承Python `IntEnum`的整数算术语义。

## 3. Domains 与 source-derived subregions

```python
rows = I.domain(0, M)
columns = I.domain(0, N)
value = x[rows, columns]
y[rows, columns] = value * scale

window = columns[begin:end]
```

`domain(begin,end,step=1)` 使用半开整数序列。`axis[begin:end]` 产生source-derived连续subregion，保留source axis、bounds、empty/tail与provenance，不隐式clamp。

遍历subregion得到source logical coordinate；`I.indices(subregion)`也返回source coordinates。`I.indices`产生的是logical `index` tensor，其数值是source coordinate，同时canonical relation保存source identity、source axis与member mapping；不能在helper、slice或shape transform后只留下普通`i32/i64`数值。非连续、重复或重排成员使用indexed relation。

## 4. Tensor values 与 shape transforms

pointwise surface允许scalar与size-one broadcast；frontend必须将其归一成显式broadcast relation。dynamic extents保留identity/equality conditions，不以“都是dynamic”判为兼容。

`reshape` 保持logical row-major element order与元素总数；`transpose/permute` 显式给出axis permutation。

Ranked tensor与external view暴露`.shape`，返回保留dynamic-extent identity的logical extent tuple。Scalar、tuple、record与domain/subregion本身没有统一`.shape`。`.shape`可用于shape arithmetic、domain、shape transform和tensor construction，不表示physical fragment或block shape。

```python
mask = I.full(scores.shape, fill=True, dtype=I.bool)
zeros = I.full((M, N), fill=0.0, dtype=I.f32)
```

`I.full(shape, fill, dtype=...)`产生无effect、无alias的ranked tensor SSA value。`shape`的每一维是非负logical extent，可为静态值或保留identity的runtime shape value；`fill`是按显式`dtype`实例化并广播到所有logical elements的scalar。它不分配logical buffer、不初始化external view，也不携带storage/layout/padding。`I.zeros(shape,dtype)`是`I.full(shape, fill=0, dtype=dtype)`的surface shorthand。

`join` 只表示两个同dtype、同shape values沿新的trailing logical axis堆叠：

```python
pair = I.join(lhs, rhs)

# pair[..., 0] == lhs[...]
# pair[..., 1] == rhs[...]
# pair.shape == lhs.shape + (2,)
```

它不是record、concatenate或一般interleave。

Python tuple literal与tuple-valued helper result构造固定长度、按位置编号的typed product value。`I.record(field=value, ...)`构造非空、字段名唯一、字段顺序固定的typed named product value。两者都是immutable SSA aggregates：

- component可以是scalar、ranked tensor、tuple或nested record，不要求相同dtype、rank或shape；
- tuple使用静态位置选择或解构，record使用静态`.field`选择；
- type identity分别包含tuple component顺序，或record字段名、字段顺序与逐字段类型；
- 它们可作helper values、loop carry、logical-buffer element schema与structured-operation accumulator/identity；
- 它们不直接成为kernel public runtime parameter、external-view element或host-visible return。跨kernel状态要拆成显式views/scalars。

Tuple/record不拥有统一`.shape`或`.dtype`；要取tensor shape，先选中具体component。它们也不是`join`：`join`产生具有单一element dtype和新logical axis的ranked tensor。

## 5. Unordered parallel iteration

```python
for row in I.parallel(I.domain(0, M)):
    y[row] = f(x[row])
```

`parallel` 是unordered forall control：每个logical point执行一次，points之间没有顺序，不允许loop-carried values。非atomic/reduction的冲突effects非法。它不指定任何physical execution level。

普通整域tensor assignment已经具有唯一写入时，不要求额外包一层`parallel`。

## 6. Ordered control 与 carry

普通Python `if`、`for`、`while` lower为structured control。普通domain iteration按logical order执行，循环中更新的SSA values是loop carry。

```python
state = initial
for index in I.domain(0, N):
    state = transition(state, x[index])
```

不提供public `ordered`。`break/continue`使用普通结构化语义；tensor predicate使用`select`，不作为structured `if`条件。

不提供允许arbitrary body随compiler-selected extent重新分段的`state_stream`。逐element的可重结合summary写成reduce/scan；需要在任意连续source slice内显式执行contract等tensor operations时使用region fold/scan；严格顺序或动态停止写成ordinary loop。

## 7. Generic reduce

```python
@intent.fn
def combine(lhs, rhs):
    return lhs + rhs

result = I.reduce(
    value,
    axis=axis,
    identity=I.cast(0, I.f32),
    combine=combine,
)
```

source与accumulator都可以是scalar、tensor、tuple或typed record。source component包含被归约axes；删除这些axes后的component shape就是accumulator、identity、combine参数与result shape。combine：

- 接受两组与accumulator相同schema的参数；
- 返回同一schema；
- 是typed pure helper；
- runtime captures必须成为显式operands；
- 不得包含read/write/atomic/RNG或buffer mutation。

作者选择reduce，即选择其允许的logical-order-preserving reassociation；不保证ordinary left fold。identity逐component显式给出，空reduction返回identity。

`reduce.sum/max`、`any/all`与`arg_reduce.max`是surface shorthand，统一归一到generic reduce。`arg_reduce.max`固定lowest logical index tie-break。

## 8. Scan

```python
prefix = I.scan(
    value,
    axis=axis,
    identity=identity,
    combine=combine,
    inclusive=True,
    reverse=False,
)
```

scan使用与reduce相同的typed pure combine，定义每个logical prefix。`inclusive/exclusive`与forward/reverse是operation semantics。严格顺序、不可重结合的recurrence使用ordinary loop。

## 9. Region fold 与 region scan

### 9.1 Region fold

```python
summary = I.region_fold(
    source=(keys, values, key_coordinates),
    axis=0,
    summarize=summarize_chunk,
    combine=merge_summaries,
    identity=empty_summary,
    operands=(queries, query_coordinates, scale),
)
```

`region_fold`表达一个有序source axis上的list homomorphism。对非空source，compiler可以把该axis分成任意数量的连续非空、保持顺序且完整覆盖source的slices；slice extent不是DSL value或KIR parameter。所有source components在`axis`位置具有同一logical extent，并按同一boundaries切片后传给`summarize`。`operands`是不随slice切分的显式captures；runtime values成为KIR operands，constexpr values保持specialization facts。

`summarize`：

- 接收各source components的当前slice，随后接收`operands`；
- 返回固定typed summary schema；
- 可以调用pure pointwise、reduce、scan、contract、scaled/sparse contract与其它pure helpers；
- 不得写external view或logical buffer，不得scatter、atomic、RNG或依赖调用次数；
- 不得读取segment ordinal、segment count、chosen extent或chunk-relative coordinates。

需要logical coordinates时，把`I.indices(source_axis)`作为一个source component传入。该component被切片后仍保存absolute source coordinates。

`combine`接受两个summary并返回同一schema。`identity`也具有该schema。作者选择`region_fold`即要求：

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

其中`A`、`B`是相邻source slices。Combine保持logical order但允许任意parenthesization。Empty source返回identity；physical tail或padding也可以安全使用identity，因此不能把在combine中产生NaN的sentinel冒充identity。需要区分“没有成员”时，summary显式携带bool validity或等价typed状态。

Ordinary reduce是element-summary的受限形式。二者共享homomorphic summary interface与physical reduction framework；若region summarizer只是用同一个combine折叠slice elements，frontend canonicalize为ordinary reduce。

### 9.2 Region scan

```python
outputs, final_state = I.region_scan(
    source=source,
    axis=0,
    summarize=summarize_transition,
    combine=compose_transitions,
    identity=identity_transition,
    initial_state=initial_state,
    apply=apply_transition,
    emit=emit_slice,
    operands=captures,
)
```

`region_scan`在同一homomorphism上增加incoming-state与slice output。`summarize`/`combine`/`apply`/`emit`都是typed pure helpers：

- `summarize(slice, captures) -> Transition`；
- `combine(lhs, rhs) -> Transition`按source顺序组合transitions；
- `apply(prefix_transition, initial_state) -> incoming_state`；
- `emit(slice, incoming_state, captures) -> output_slice`。

Transition identity与composition必须对state构成合法action：

```text
apply(identity, state) == state
apply(combine(a, b), state) == apply(b, apply(a, state))
```

对相邻slices `A`、`B`，还必须满足：

```text
emit(A ++ B, state)
  == concat(
       emit(A, state),
       emit(B, apply(summarize(A), state)))
```

`emit`的result axis必须与输入slice保存同一source member relation；operation返回按source order拼回的完整logical output以及final state `apply(summarize(full_source), initial_state)`。Compiler-selected slice boundaries、内部prefix states与segment count不可观察。需要host-visible chunk states或固定chunk ABI时，作者使用显式logical chunk domain、subregions与ordinary scan。

Ordinary scan是element-summary/element-output的受限形式；退化的region写法canonicalize回ordinary scan。`region_scan`不是任意effectful state loop，不能代替严格ordered recurrence。

完整FlashAttention写法见[`examples/flash_attention.py`](examples/flash_attention.py)。其中QK与PV都是作者显式写下的contract；region fold只抽象K source的合法连续segmentation，不依赖compiler从tuple reduction识别attention模式。

完整region-scan写法见[`examples/causal_linear_attention.py`](examples/causal_linear_attention.py)；显式算法chunk的反例见[`examples/mamba_state_passing.py`](examples/mamba_state_passing.py)。前者的segment boundaries不可观察且满足summary/emit等价律；后者的chunk axis与per-chunk state进入ABI，因而使用ordinary ordered carry。

## 10. Contract

```python
acc = I.contract(
    a[m, k],
    b[k, n],
    reduce=((1, 0),),
    batch=(),
    acc_dtype=I.f32,
)
```

contract定义二元paired-axis multiply-add contraction：

- `reduce` 是非空、唯一、extent-compatible的lhs/rhs axis pairs；
- `batch` 是唯一、extent-compatible且不与reduction重叠的axis pairs；每个pair在result中只出现一次，由lhs axis代表；
- lhs中不在`reduce`里的axes按lhs顺序成为result axes，其中包括batch-pair的lhs representatives；
- rhs中不在`reduce`且不是batch-pair rhs member的axes随后按rhs顺序成为result axes；
- multiply固定为数值乘法，combine固定为加法；
- accumulator dtype显式给出。

`reduce`列表非空；其中某个logical reduction extent可以为零，此时该输出element是accumulator dtype中的加法零。该规则同样适用于scaled与sparse contract。

public surface不提供假的`multiply=`/`combine=`参数。其它semiring写成pointwise + generic reduce；没有reduction axis的outer product写成explicit broadcast multiply。

axis permutation、将多个free/reduction axes双射flatten为M/K/N、MMA选择与staging都是compiler工作，不是作者surface。

## 11. Scaled contract

```python
acc = I.scaled_contract(
    lhs,
    lhs_scale,
    rhs,
    rhs_scale,
    lhs_format=I.e2m1,
    rhs_format=I.e4m3,
    lhs_group_size=32,
    rhs_group_size=32,
    reduce=((1, 0),),
    batch=(),
    acc_dtype=I.f32,
)
```

scaled contract是first-class local tensor operation。formats、packed logical element order、scale tensors到logical groups的relation、contraction axes、accumulator与rounding属于算法语义。K packing、native scaled MMA、layout与storage不属于调用。`e2m1/e4m3/e8m0`的bit encoding、group coordinate与特殊值见[`types-numerics-and-effects.md`](types-numerics-and-effects.md)。

它继承ordinary contract的全部axis规则：非空且唯一的reduction pairs、互不重叠的batch pairs、free/result axis order与multiply-add accumulator semantics。scale relation只增加operand value interpretation，不改变contraction定义。

普通packed INT4/INT2不是scaled contract。作者使用carrier tensor、bit/index arithmetic、sign extension、zero-point与scale表达其logical values，再调用ordinary contract。

## 12. Sparse contract

```python
acc = I.sparse_contract(
    compressed,
    metadata,
    dense_rhs,
    format=I.sparse.two_of_four(...),
    reduce=((1, 0),),
    batch=(),
    acc_dtype=I.f32,
)
```

`format` 是语言定义的closed typed schema，不是provider字符串或opaque descriptor。每个schema定义group size、nonzero count、compressed ordering、metadata到logical reduction positions的映射、合法metadata与logical dtype rules。

本规范定义的schemas包括`one_of_two`与`two_of_four`；它们的logical position metadata与compressed ordering见[`types-numerics-and-effects.md`](types-numerics-and-effects.md)。增加新格式需要定义新的concrete schema。external packed metadata由作者显式解释，instruction-native metadata layout、physical repacking、memory placement与sparse MMA属于target lowering。`sparse_contract_2to4`可作为surface shorthand，但不形成第二条canonical path。

sparse contract同样继承ordinary contract的非空reduction、batch/free-axis、result order与accumulator规则；format只替换一个operand沿声明compression axis的logical value relation。

## 13. Histogram

```python
counts = I.histogram(
    values,
    bins=BINS,
    valid=valid,
    count_dtype=I.u32,
)
```

histogram是pure value-producing structured operation。每个active value必须满足`0 <= value < bins`并向对应bin贡献一次；需要忽略越界时，作者把范围条件并入`valid`。空输入返回全零，计数overflow遵循count dtype的整数语义。

`values`必须是integer tensor，`bins`是正logical index，result是shape `(bins,)` 的count tensor。`valid`是bool scalar或可broadcast到`values.shape`的bool tensor；`valid=False`的lane不读取其value是否落在range内，也不贡献计数。

它不等价于作者预写external buffer初始化与atomic updates；后者已选择一种effectful lowering。

## 14. Ragged relation 的普通组成

```python
member_source = I.domain(0, R)
members = member_source[offsets[group]:offsets[group + 1]]
logical_members = mapping[members] if HAS_MAPPING else I.indices(members)
```

offsets-derived subregion、可选indexed mapping及其SSA provenance就是完整ragged语义。`I.ragged(...)`/`I.members(...)`可以是机械surface helper，但不形成专用canonical RaggedOp/MemberOp。

## 15. Indexed access、buffers 与 copy

普通indexing与显式gather共享typed index relation和active validity。relation保存source identity、source rank、result logical axes，以及每个source axis的coordinate expression；coordinate可以来自constant、domain/subregion coordinate或data-derived integer value。relation composition必须保存这些SSA dependencies与provenance，不能只留下result shape。invalid read不访问source并返回同dtype、可broadcast到result shape的显式fill；invalid write不产生effect。

canonical operations继续区分：

- pure tensor gather；
- external view load/store；
- logical-buffer load/store；
- arbitrary-index unique store；
- `scatter_reduce`。

unique store要求destination relation可证明injective或由precondition保证。`scatter_reduce`保存typed combine与collision semantics。

logical buffer是kernel-local mutable state；作者定义shape、dtype、initialization与read/write order，不指定physical residency。

不提供canonical `copy`。一次indexed read产生immutable SSA value，再由indexed write消费，已经完整定义snapshot与effects。provider可从该def-use/access relation形成bulk、vector、async或DMA copy。

## 16. Atomic operations

public surface提供具体operations：

```python
value = I.atomic.load(address, order="acquire")
I.atomic.store(address, value, order="release")
old = I.atomic.add(address, delta, order="relaxed")
result = I.atomic.compare_exchange(address, expected, desired, order="acq_rel")
# result.old_value, result.success
```

canonical operations是：

- `atomic_load`；
- `atomic_store`；
- `atomic_rmw(kind=exchange/add/max/min/and/or/xor)`；
- `atomic_compare_exchange`，返回`old_value`与`success`。

load只允许`relaxed/acquire`，store只允许`relaxed/release`，RMW/CAS允许`relaxed/acquire/release/acq_rel`。CAS返回typed record `{old_value, success}`；failure order由success order机械导出：`relaxed -> relaxed`、`acquire -> acquire`、`release -> relaxed`、`acq_rel -> acquire`。

atomic不带作者可见`scope=`。logical allocation identity、alias/index relation与本次kernel invocation确定参与同一atomic object的executions；provider选择physical scope。

## 17. RNG

```python
bits = I.random.bits(seed, logical_counter)
uniform = I.random.uniform(seed, logical_counter, dtype=I.f32)
```

canonical bits operation固定Philox4x32-10。scalar logical counter按`block_counter = counter // 4`与`word = counter % 4`映射到一个Philox output word；rounds与常量见[`types-numerics-and-effects.md`](types-numerics-and-effects.md)。它不读取program/thread/lane identity，不依赖调用顺序或mutable provider state。

`uniform`是从canonical bits到浮点值的固定转换。normal等复合distribution由作者helper构造。

## 18. Logical parts 不使用 `partition`

```python
parts = I.domain(0, P)
source = I.domain(0, N)
width = (N + P - 1) // P

for part in I.parallel(parts):
    begin = I.minimum(part * width, N)
    end = I.minimum((part + 1) * width, N)
    region = source[begin:end]
    partial[part] = I.reduce.sum(x[region], axis=0, identity=0.0)
```

part identity、boundary formula、empty/tail与partial tensor interface由普通constructs明确表达。`partition(auto/count/extent)`均不是public或canonical operation。

## 19. Surface 归属总表

| family | canonical semantics | surface shorthand / accessor | 不属于语言 |
|---|---|---|---|
| definitions/interface | kernel、helper、`In/Out/InOut`、runtime/constexpr | Python decorators与type spelling | target selection、hidden launch、provider dispatch |
| domain/control | domain、source subregion、`if/for/while`、unordered parallel、loop carry | slices、`indices`、`break/continue` | `auto`、partition、state_stream、ordered、program/lane id |
| tensor values | arithmetic、compare/select、broadcast、reshape、transpose、join、tuple、record、full、cast/bitcast | zeros、activation helpers、value mask | physical tile/layout/padding |
| structured ops | generic reduce、scan、region fold/scan、contract、scaled contract、sparse contract、histogram | built-in reduces、arg-reduce、format-specific sparse spelling | whole-operator softmax/attention/MoE |
| relations | source subregion、index relation、sparse format schema | ragged/members/index helpers | target metadata layout、MMA hint |
| memory/effects | external/buffer read-write、unique/reduction scatter、atomic ops、Philox bits | ordinary indexing/assignment、atomic convenience names | physical scope、storage、copy instruction、barrier/pipeline |

surface shorthand必须归一到唯一canonical path；target physical information不得借shorthand回流作者程序。
