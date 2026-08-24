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

## 3. Domains 与 source-derived subregions

```python
rows = I.domain(0, M)
columns = I.domain(0, N)
value = x[rows, columns]
y[rows, columns] = value * scale

window = columns[begin:end]
```

`domain(begin,end,step=1)` 使用半开整数序列。`axis[begin:end]` 产生source-derived连续subregion，保留source axis、bounds、empty/tail与provenance，不隐式clamp。

遍历subregion得到source logical coordinate；`I.indices(subregion)`也返回source coordinates。非连续、重复或重排成员使用indexed relation。

## 4. Tensor values 与 shape transforms

pointwise surface允许scalar与size-one broadcast；frontend必须将其归一成显式broadcast relation。dynamic extents保留identity/equality conditions，不以“都是dynamic”判为兼容。

`reshape` 保持logical row-major element order与元素总数；`transpose/permute` 显式给出axis permutation。

`join` 只表示两个同dtype、同shape values沿新的trailing logical axis堆叠：

```python
pair = I.join(lhs, rhs)

# pair[..., 0] == lhs[...]
# pair[..., 1] == rhs[...]
# pair.shape == lhs.shape + (2,)
```

它不是record、concatenate或一般interleave。

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

不提供`state_stream`：可重结合summary写成reduce，需要prefix写成scan，严格顺序或动态停止写成ordinary loop。

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

## 9. Contract

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
- `batch` 是唯一、extent-compatible且不与reduction重叠的axis pairs；
- 其余lhs/rhs axes分别成为result free axes；
- result axis order是lhs未归约axes，随后是rhs未归约且非rhs-batch axes；
- multiply固定为数值乘法，combine固定为加法；
- accumulator dtype显式给出。

`reduce`列表非空；其中某个logical reduction extent可以为零，此时该输出element是accumulator dtype中的加法零。该规则同样适用于scaled与sparse contract。

public surface不提供假的`multiply=`/`combine=`参数。其它semiring写成pointwise + generic reduce；没有reduction axis的outer product写成explicit broadcast multiply。

axis permutation、将多个free/reduction axes双射flatten为M/K/N、MMA选择与staging都是compiler工作，不是作者surface。

## 10. Scaled contract

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

## 11. Sparse contract

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

## 12. Histogram

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

## 13. Ragged relation 的普通组成

```python
member_source = I.domain(0, R)
members = member_source[offsets[group]:offsets[group + 1]]
logical_members = mapping[members] if HAS_MAPPING else I.indices(members)
```

offsets-derived subregion、可选indexed mapping及其SSA provenance就是完整ragged语义。`I.ragged(...)`/`I.members(...)`可以是机械surface helper，但不形成专用canonical RaggedOp/MemberOp。

## 14. Indexed access、buffers 与 copy

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

## 15. Atomic operations

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

## 16. RNG

```python
bits = I.random.bits(seed, logical_counter)
uniform = I.random.uniform(seed, logical_counter, dtype=I.f32)
```

canonical bits operation固定Philox4x32-10。scalar logical counter按`block_counter = counter // 4`与`word = counter % 4`映射到一个Philox output word；rounds与常量见[`types-numerics-and-effects.md`](types-numerics-and-effects.md)。它不读取program/thread/lane identity，不依赖调用顺序或mutable provider state。

`uniform`是从canonical bits到浮点值的固定转换。normal等复合distribution由作者helper构造。

## 17. Logical parts 不使用 `partition`

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

## 18. Surface 归属总表

| family | canonical semantics | surface shorthand / accessor | 不属于语言 |
|---|---|---|---|
| definitions/interface | kernel、helper、`In/Out/InOut`、runtime/constexpr | Python decorators与type spelling | target selection、hidden launch、provider dispatch |
| domain/control | domain、source subregion、`if/for/while`、unordered parallel、loop carry | slices、`indices`、`break/continue` | `auto`、partition、state_stream、ordered、program/lane id |
| tensor values | arithmetic、compare/select、broadcast、reshape、transpose、join、record、full、cast/bitcast | zeros、activation helpers、value mask | physical tile/layout/padding |
| structured ops | generic reduce、scan、contract、scaled contract、sparse contract、histogram | built-in reduces、arg-reduce、format-specific sparse spelling | whole-operator softmax/attention/MoE |
| relations | source subregion、index relation、sparse format schema | ragged/members/index helpers | target metadata layout、MMA hint |
| memory/effects | external/buffer read-write、unique/reduction scatter、atomic ops、Philox bits | ordinary indexing/assignment、atomic convenience names | physical scope、storage、copy instruction、barrier/pipeline |

surface shorthand必须归一到唯一canonical path；target physical information不得借shorthand回流作者程序。
