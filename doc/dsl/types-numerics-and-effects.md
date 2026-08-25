# Types、Numerics 与 Effects

## 1. Canonical value types

普通logical values使用：

- `bool`；
- signed 64-bit logical `index`；
- `i8/i16/i32/i64`；
- `u8/u16/u32/u64`；
- `f16/bf16/f32/f64`；
- 明确定义bit encoding的`f8e4m3fn/f8e5m2`；
- typed tuples/records；
- ranked tensor values；
- external views与kernel-local logical buffers。

physical lane shape、padding、layout、address width和target encoding不进入canonical value type。

Tuple和record是immutable structural product types，不是ranked tensor或memory layout。Tuple type由固定component序列定义；record type由有序、唯一的field names及逐字段value types定义。它们可包含不同shape/dtype的tensor components，也可嵌套；不提供统一`.shape`或`.dtype`。它们可作为SSA、helper result、loop carry、logical-buffer element schema和structured accumulator，但不直接进入public kernel ABI或external-view element type。

`i4/u4/fp4`不作为普通可寻址tensor element type。普通packed data使用`u8/u16/u32` carrier，加显式index、shift、mask、sign extension与scale arithmetic。FP4等microscaling formats由scaled-contract schema定义。

## 2. Logical index 与 shape values

logical `index`使用signed 64-bit arithmetic。physical address width由compiler根据shape、stride与bounds证明后选择，不能改变logical index结果。

dynamic extent是有identity的runtime shape value。不同dynamic extents只有在来自同一value、operation产生明确equality relation，或调用前置条件声明相等时才兼容。

tensor rank与logical extents来自domains/subregions和shape transforms。broadcast、reshape与join的规则见[`logical-program.md`](../programming-model/logical-program.md)。

Ranked tensor与external view的`.shape`是logical extent tuple；dynamic members保留shape identity。`I.full(shape, fill, dtype)`要求每个extent非负，把scalar `fill`按本文的literal/cast规则实例化为`dtype`并广播到所有elements；zero extent产生empty tensor。`value.shape`可直接作为`I.full`的shape，但不转化成physical fragment shape。

`I.Enum`是constexpr-only closed named type。成员及其唯一值只参与specialization identity、同类型比较与硬件无关constexpr control；不同enum不隐式比较，enumeration不进入runtime scalar/tensor/buffer/view ABI，不允许算术、cast或bitcast。Python surface可使用`IntEnum`-like declaration convenience，但其underlying Python integer representation不是canonical ABI事实。

## 3. Literals 与 promotion

Python literal是untyped source literal，可以按直接使用位置的expected dtype实例化，前提是其值可表示。

两个runtime numeric operands不会采用provider自己的promotion rules：

- dtype相同可直接运算；
- dtype不同必须显式`I.cast`；
- comparison返回`bool`；
- structured operation的accumulator/result dtype由operation显式给出或使用本文定义的固定builtin规则。

`reduce.sum`在未显式给出accumulator dtype时使用：

- `i8/i16 -> i32`，`u8/u16 -> u32`；
- `i32/i64/u32/u64`保持输入dtype；
- `f8e4m3fn/f8e5m2/f16/bf16 -> f32`；
- `f32/f64`保持输入dtype。

其它builtin reduce保持其输入dtype，除非surface明确要求另一result schema。

## 4. Integer arithmetic

fixed-width integers使用二进制补码与modulo arithmetic：

- add/sub/mul按`2^width` wrap；
- signed `//`与`%`使用Python floor-division relation：`a == q*b + r`，`q=floor(a/b)`，非零`b`时`r`与`b`同号或为零；
- unsigned `//`与`%`使用Euclidean quotient/remainder：`a == q*b + r`且`0 <= r < b`；
- integer operands不重载`/`；需要floating division时必须先显式cast；
- signed right shift是arithmetic shift，unsigned right shift是logical shift；
- left shift保留低`width` bits；
- bitwise operations作用于固定宽度bit pattern；
- shift amount必须满足`0 <= amount < width`；
- division by zero与`signed_min // -1`是非法输入。

`%`只用于integer remainder。floating remainder必须使用具有独立、明确定义语义的named pointwise operation，不能继承Python、C或provider对`%`的偶然解释。

这些规则由每个target机械兑现，不继承C/CUDA/Triton的偶然差异。

## 5. Floating point、cast 与 bitcast

普通floating operations遵循对应格式的IEEE-754值与round-to-nearest-even。除非作者使用structured operation或显式approximate math，compiler保持source expression的数据依赖与求值关系，不启用会改变结果集合的隐式fast-math。

`I.cast`定义：

- integer→integer先把source解释为数学整数，再对`2^destination_width`取模，最后按destination signedness解释该bit pattern；同signedness widening因此分别等价于sign/zero extension，narrowing等价于保留低bits；
- integer→float使用round-to-nearest-even；
- float→integer向零截断，NaN、infinity或越界是非法输入；
- float narrowing到`bf16/f16/f32`使用round-to-nearest-even，有限溢出产生对应infinity；到8-bit float使用下述format-specific规则；
- saturating conversion使用单独、明确的saturating operation，不是普通cast的隐藏模式。

`I.bitcast`要求source/result总bitwidth相同，只保存bit pattern，不进行数值转换或storage repacking。

`I.maximum/I.minimum`传播NaN；忽略单侧NaN的`maximum_num/minimum_num`是不同pointwise operations，不能依赖target默认行为。普通float cast与IEEE arithmetic保留signed zero；NaN result只保证仍是NaN，不把payload、signaling状态或NaN sign定义为可观察语义。

两种8-bit float的logical encoding固定为：

- `f8e4m3fn`：1 sign bit、4 exponent bits、3 fraction bits、bias 7、支持subnormal、无infinity；`S.1111.111`是NaN，其余编码按finite E4M3FN解释，最大finite magnitude为448。普通cast使用round-to-nearest-even，有限输入超出可表示范围是非法输入；需要clamp时使用显式saturating conversion；
- `f8e5m2`：1 sign bit、5 exponent bits、2 fraction bits、bias 15、支持subnormal；exponent全1且fraction为0表示infinity，fraction非零表示NaN。普通cast使用round-to-nearest-even，finite overflow产生对应infinity。

## 6. Reduce 与 scan 数值语义

generic reduce/scan的combine按logical element order允许任意parenthesization，但不允许任意permutation。作者选择这些operations，即接受这种reassociation可能导致的finite-precision差异；要求严格left fold时使用ordinary loop。source component可以含被归约axes；删除这些axes后的component shape是accumulator、identity、combine参数与result shape。

identity逐component显式给出：

- empty reduce返回identity；
- empty scan返回empty tensor；
- inclusive/exclusive与forward/reverse由scan参数决定。

NaN与tie behavior来自明确combine。`reduce.max`使用propagating maximum；`arg_reduce.max`在values相等时选择lowest logical index，并传播NaN。其它策略必须通过不同typed combine明确写出。

Region fold/scan沿compiler-selected连续source slices允许同样的logical-order-preserving reassociation。它们另外要求region summarizer与summary combine满足：

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

Compiler不证明这些代数定律；作者选择operation即声明它们成立。Verifier必须检查source-axis一致、summary/identity/combine schema、captures、purity、output relation及禁止的effects。

`identity`必须在本节定义的float/NaN语义下真正中立。把`maximum=-inf`、其它components为零的record无条件送进包含`exp(maximum - merged_maximum)`的combine，会在`-inf - -inf`处产生NaN，因此不是合法identity。此类summary必须携带显式validity，并在执行指数运算前把invalid分支规范化为有限差值；或者使用另一种能够证明双侧中立的typed表示。Empty source的region fold返回identity；empty region scan返回empty output与initial state。

Region summarizer不能观察compiler-selected segment identity或extent。它若需要位置，必须消费由source axis产生的absolute logical coordinates；这些coordinates切片后不重新编号。Region scan除summary homomorphism外，还要求`apply(identity,state) == state`，以及整段emit等于按相邻slices分段emit，后一段消费前一段summary作用后的incoming state。Floating-point region fold/scan接受由合法segmentation与parenthesization造成的舍入差异，但不允许改变source order、NaN policy、accumulator dtype或approximation contract。

## 7. Scaled、sparse 与 histogram numerics

scaled-contract schema定义logical element format、carrier packing、scale encoding、group relation、rounding、special values与accumulator dtype。普通target不能以native primitive限制反向缩窄该语义。

本规范的canonical microscaling formats定义如下：

- `e2m1` logical element占4 bits，sign为bit 3；正值编码`0..7`依次表示`0, 0.5, 1, 1.5, 2, 3, 4, 6`，sign bit取反数值符号；无NaN或infinity。两个elements按logical reduction coordinate递增顺序打包到一个`u8`，较小coordinate使用低nibble；
- `e4m3`采用上文`f8e4m3fn`的8-bit encoding；
- `e8m0` scale是8-bit无sign exponent，code `0..254`表示`2^(code-127)`，code `255`表示NaN scale。

group size是正logical index。沿声明的group axis，logical coordinate `k`读取scale element `k // group_size`；decoded logical element先乘对应scale，再进入paired-axis multiply-add。scale tensor的其它axes按operation声明的shape relation与operand axes对齐，不允许provider自行选择另一group relation。

sparse-contract schema定义compressed ordering、metadata到logical reduction positions的解释、invalid metadata与accumulator。canonical metadata是logical positions，不直接使用某个instruction要求的packed metadata layout：

- `one_of_two`要求logical compression axis extent可被2整除。每组2个logical positions保存1个compressed value；metadata是每组一个logical `index`，合法值为`0`或`1`，指明该value位于组内哪个position；
- `two_of_four`要求logical compression axis extent可被4整除。每组4个logical positions保存2个compressed values，按position递增顺序存放；metadata是每组一个typed record `{first: index, second: index}`，要求`0 <= first < second < 4`。

其它logical positions为加法零。若external input使用packed metadata carrier，作者以普通bit/index arithmetic把它解释成上述logical positions；这不会物化dense operand，也不会丢失structured sparsity。非法metadata或不满足group整除关系是调用方错误；provider为native sparse instruction重新编码metadata属于physical representation，不能改变上述logical mapping。

histogram要求`values`是integer tensor，`bins`是正logical index，`valid`是bool scalar或可broadcast到`values.shape`的bool tensor。每个active value必须位于`[0,bins)`并贡献一次；inactive lane既不贡献，也不要求其value在range内。result shape为`(bins,)`。count dtype显式给出，overflow使用该integer dtype的wrap语义，empty input返回全零。

## 8. External views

external view定义：

- element dtype；
- logical shape；
- base allocation identity与element offset；
- runtime strides，单位是elements而不是bytes；
- `In/Out/InOut`访问方向；
- bounds与必要alias relation。

对logical coordinate tuple `c`，element address是`base_allocation[offset + sum(c_i * stride_i)]`。negative stride合法。zero stride对只读broadcast view合法；可写view只有在对应write/collision semantics允许地址重复时才合法。每个active logical coordinate都必须映射到有效allocation；inactive access不形成地址，语言也不隐式clamp。

不同view参数默认可以alias。`noalias`若出现，是调用方必须满足的语义前置条件，不是performance hint。alignment、contiguity、vector width、descriptor/TMA eligibility由runtime/compiler/provider处理，除非某个外部数据格式的正确解释本身要求特定alignment。

## 9. Logical buffer

logical buffer是kernel-local mutable state：

- shape与dtype由作者定义；
- 新buffer不alias external views或其它新buffer；
- 可以用完整initial value创建，也可以未初始化创建；
- verifier必须证明每次read之前对应element已经被写入；
- lifetime是lexical，buffer不能逃逸到kernel外；
- storage、materialization与placement由compiler决定。

跨kernel存活的tensor由host显式拥有，不是logical buffer。

## 10. Indexed access 与 collision semantics

所有access使用typed index relation与active validity。relation保存source identity/rank、result logical axes、每个source axis的typed coordinate expression，以及这些expressions对domain/subregion/data-derived indices的SSA provenance。relation composition必须组合这些expressions，不能只保存result shape。invalid read不产生memory access并返回同dtype、可broadcast到result shape的显式fill；invalid write不产生effect。

canonical effects区分：

- external read/write；
- logical-buffer read/write；
- arbitrary-index unique store；
- `scatter_reduce`；
- atomic load/store/RMW/CAS。

unique store要求active destination mapping injective，或由调用前置条件保证。`scatter_reduce`使用typed pure combine定义重复地址的合并，不承诺每次更新的linearization或old value。ordinary ordered control中的effects保持program order；unordered parallel iterations之间若可能访问同一地址，必须由unique、scatter-reduction或atomic semantics消解，否则程序非法。

plain logical copy使用read→immutable SSA value→write表达，不建立独立canonical op。non-atomic read/write/gather/scatter不带memory order或physical scope。

## 11. Atomic memory semantics

canonical atomic operations是：

- `atomic_load`；
- `atomic_store`；
- `atomic_rmw(kind=exchange/add/max/min/and/or/xor)`；
- `atomic_compare_exchange`。

每个logical atomic address具有单一modification order。RMW不可分割并返回old value；CAS返回typed record `{old_value, success}`。

memory order定义：

- `relaxed`：只保证该atomic object的atomicity与modification order；
- `acquire`：读取matching release/release sequence后建立happens-before；
- `release`：发布该operation之前的ordinary/atomic effects；
- `acq_rel`：同时具有acquire与release。

load只允许`relaxed/acquire`；store只允许`relaxed/release`；RMW与CAS允许全部四种。CAS failure order由success order机械派生：`relaxed -> relaxed`、`acquire -> acquire`、`release -> relaxed`、`acq_rel -> acquire`。

atomic operation没有public/canonical `scope`。同一kernel invocation中，通过同一logical allocation/address relation访问该atomic object的executions是逻辑参与者；compiler在target physical program中选择正确scope。

普通conflicting non-atomic accesses是非法data race。语言不保证parallel iterations同时驻留，也不保证依赖spin-wait的程序取得forward progress。

## 12. Deterministic RNG

canonical RNG是pure stateless Philox4x32-10 random-bits operation。`seed`是`u64`，`logical_counter`是非负`u64`或可无损转换为`u64`的logical index；`I.random.bits`返回一个`u32`：

```text
block = logical_counter // 4
word  = logical_counter % 4
(c0, c1, c2, c3) = (low32(block), high32(block), 0, 0)
(k0, k1) = (low32(seed), high32(seed))

repeat 10 rounds:
    (hi0, lo0) = mul_wide_u32(0xD2511F53, c0)
    (hi1, lo1) = mul_wide_u32(0xCD9E8D57, c2)
    (c0, c1, c2, c3) = (hi1 ^ c1 ^ k0, lo1,
                         hi0 ^ c3 ^ k1, lo0)
    k0 = k0 + 0x9E3779B9  (mod 2^32)
    k1 = k1 + 0xBB67AE85  (mod 2^32)

result = (c0, c1, c2, c3)[word]
```

它不读取target、program、thread、lane或调用序号。`uniform(..., dtype=f32)`固定为`f32(bits >> 8) * 2^-24`，结果位于`[0,1)`；normal等复合distribution使用普通DSL helper构造。provider只有在证明native RNG产生同一bit stream时才能替换该integer computation。

## 13. Preconditions

作者可以声明不能由type/shape单独证明的调用前置条件，例如：

- index in bounds；
- indices sorted/unique；
- dynamic extents equal；
- views non-alias；
- external encoded data满足format要求。

precondition不生成runtime clamp、修复或fallback。违反前置条件是调用方错误。
