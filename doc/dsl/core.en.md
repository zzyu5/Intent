# Language constructs

## 1. Definitions

```python
@intent.kernel
def kernel(...):
    ...

@intent.fn
def helper(...):
    ...
```

Kernel defines one logical computation; an external call selects its target. Helpers are typed kernel functions, never hidden kernels or host dispatch. They may return scalars, tensors, tuples or records and have effects allowed at their call site. Runtime captures are explicit parameters; only immutable constexprs may be captured. Recursion, target queries and launches are illegal.

Helper arguments follow the public Python signature, including named and keyword-only parameters. Expressions evaluate in source order, not reordered parameter order; each lowers once before binding typed values. Named computations normalize identically in ordinary and structured pure helpers; pure sites check actual body effects.

Coordinates retain source identity, axis maps and typed expressions componentwise through helpers, regardless of inlining.

## 2. Parameters

```python
x: I.In[I.f16, ("M", "N")]
y: I.Out[I.f16, ("M", "N")]
state: I.InOut[I.f32, ("M",)]
scale: I.f32
CAUSAL: I.Constexpr[bool]
```

In is read-only; Out is unreadable until defined; InOut retains input and permits writes. Scalar dtype annotations are runtime values. Constexpr selects only hardware-independent algorithm branches.

`I.Enum` is a finite, nonempty, closed named constexpr type with unique names/values. It is allowed in Constexpr parameters/defaults, constexpr conditions and captures, not runtime scalar/tensor/buffer/view ABI. Different enum types do not implicitly compare/convert, and Python IntEnum arithmetic is not its semantics.

## 3. Domains and source-derived subregions

```python
rows = I.domain(0, M)
columns = I.domain(0, N)
value = x[rows, columns]
y[rows, columns] = value * scale
window = columns[begin:end]
```

`domain(begin,end,step=1)` is half-open with positive step. Reverse traversal uses forward ordinals and explicit reverse coordinates:

```python
for ordinal in I.domain(0, n):
    i = n - 1 - ordinal
```

Subregions preserve source axes, bounds, empty/tail and provenance without clamping. Iteration/indices yield source coordinates. Index tensors keep typed relations through helpers/slices/transforms. Noncontiguous, repeated or reordered members use indexed relations.

## 4. Tensor values and shape transforms

Scalar and size-one pointwise broadcast normalizes to explicit relations. Dynamic extents retain identity/equality conditions, not automatic compatibility.

Declared unary math includes `exp/log/log1p/lgamma/sin/asin/cos/floor/erf/erfc/i0/rsqrt/sqrt/sigmoid/abs`. Lgamma computes `log|Gamma(x)|`; asin returns the principal arcsine in radians; log1p/erfc/i0 are independent library operations, not handwritten approximations or ordinary arithmetic expansions. All preserve input floating dtype/shape and use the [numerical contracts](types-numerics-and-effects.md).

`I.fdiv(lhs,rhs,*,approximate=False,flush_to_zero=False)` defaults to `/`. Exp2 accepts those two options; tanh accepts approximate. Options are hardware-independent constexpr bool. Non-default modes require f32; FTZ requires approximate=True. These are numerical attributes on ordinary canonical operations, not target queries, global fast math or another algorithm.

Reshape preserves row-major order and element count. Transpose/permute require a complete unique permutation; omitted transpose permutation reverses all axes. Tensors/views expose logical `.shape` with dynamic identity; scalars/aggregates/domains do not. Shapes can build domains/transforms/tensors, not physical fragments.

```python
mask = I.full(scores.shape, fill=True, dtype=I.bool)
zeros = I.full((M, N), fill=0.0, dtype=I.f32)
```

Full produces pure alias-free tensor SSA, with nonnegative static/runtime logical extents and scalar fill instantiated to explicit dtype. Empty shape makes rank-zero tensors. It neither allocates buffers nor initializes external views or specifies layout/padding. Zeros is shorthand.

Join requires equal dtypes/shapes and adds one trailing axis of two elements, lhs at0/rhs at1. It is not record/concatenate/general interleave.

Tuples are fixed ordered products; records are nonempty products with unique names and fixed field order. Components may be scalars, differently shaped/dtyped tensors or nested products. Static tuple positions/destructuring and record fields select components. Type identity includes component order and, for records, names/order/types. Products serve SSA/helpers/carries/identities but not public runtime parameters, view elements or host kernel returns. Cross-kernel state uses explicit views/scalars. Products have no common shape/dtype and are not join.

## 5. Unordered parallel iteration

```python
for row in I.parallel(I.domain(0, M)):
    y[row] = f(x[row])
```

Parallel is unordered forall, exactly once per logical point, with no carries or conflicting non-atomic/non-reduction effects. It specifies no execution level. Whole-tensor assignment with unique writes needs no extra parallel wrapper.

## 6. Ordered control and carries

Ordinary Python if/for/while lower to structured control. Domain iteration is ordered; SSA updates are carries. Initialization, region arguments and updates preserve dtype/rank/provably equal logical shapes componentwise. Body broadcasting cannot enlarge the initial schema.

```python
state = initial
for index in I.domain(0, N):
    state = transition(state, x[index])
```

There is no public ordered marker. Break/continue have ordinary structured meaning; tensor predicates use select. No arbitrary resegmentable state_stream exists: element summaries use reduce/scan, region tensor summarization uses region fold/scan, strict sequence or dynamic stopping uses loops.

## 7. Generic reduce

```python
@intent.fn
def combine(lhs, rhs):
    return lhs + rhs

result = I.reduce(value, axis=axis,
                  identity=I.cast(0, I.f32), combine=combine)
```

Sources/accumulators may be scalar/tensor/tuple/record. Removing source reduction axes gives the identity, accumulator, combine-argument and result shapes. Combine takes two groups with that schema and returns it; it is typed/pure, has explicit runtime captures, and no reads/writes/atomics/RNG/buffer mutation.

Choosing reduce declares associativity, commutativity and neutral identity. Elements may be reassociated/permuted, not a strict left fold or source-order guarantee. Floating-point tree differences follow the numerical chapter. The compiler checks schema/purity/effects rather than reproving arbitrary algebra. Empty reductions return componentwise identity.

Generic reduce has no hidden accumulator dtype independent of source components. Builtin acc_dtype/widening performs defined source conversions before constructing reduce; external input storage need not match accumulation.

```python
I.reduce.sum(value, *, axis, acc_dtype=None)
I.reduce.max(value, *, axis, acc_dtype=None)
I.reduce.any(value, *, axis)
I.reduce.all(value, *, axis)
```

Builtins accept ranked tensors and compile-time integer/nonempty-tuple axes, normalizing negatives. Sum identity is additive zero; max identity is result minimum or negative infinity; bool any/all use false/true. Builtins fix combine/identity, not caller-supplied generic parameters. Widening and NaN rules are in the numerical chapter. Empty reductions return identity.

Arg-reduce max normalizes to generic reduce with lowest-logical-index ties. It returns `(values,indices)` with reduced axes removed, scalars when none remain. Index dtype is index; positions start at0 in the input tensor's reduced axis, not absolute source/view coordinates.

## 8. Scan

```python
prefix = I.scan(value, axis=axis, identity=identity, combine=combine,
                inclusive=True, reverse=False)
```

Scan uses typed pure schemas and associativity without commutativity. It defines logical prefixes with order-preserving reassociation, not permutations. Inclusive/exclusive and direction are semantics. Strict recurrences use ordinary loops.

```python
I.cumsum(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)
I.cummax(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)
```

Inputs are ranked tensors with one explicit possibly negative axis. Results keep shape and values only. Cumsum uses sum's widening and zero; cummax uses propagating maximum/minimum identity and retains dtype by default. Explicit acc_dtype uses structured conversions. Both normalize to scan; empty input produces an empty tensor. Defaults are forward inclusive, not strict left fold.

## 9. Region fold and region scan

### 9.1 Region fold

```python
summary = I.region_fold(
    source=(keys, values, key_coordinates), axis=0,
    summarize=summarize_chunk, combine=merge_summaries,
    identity=empty_summary,
    operands=(queries, query_coordinates, scale))
```

Region fold defines a list homomorphism over one ordered source axis. A nonempty source may be split into any number of ordered contiguous nonempty slices covering it exactly. Extents are not DSL values/KIR parameters. Source components have one equal extent on axis and use identical boundaries. Explicit operands are unsliced captures; runtime captures are KIR operands, constexpr captures are specialization facts.

Summarize receives sliced components then captures and returns a fixed typed summary. Pure pointwise/reduce/scan/contractions/helpers are allowed. External/buffer writes, scatter, atomics, RNG, invocation-count dependence and segment ordinal/count/extent/local-coordinate observation are forbidden. For coordinates, pass sliced source indices retaining absolute positions.

Combine accepts two summaries, returns the same schema and shares that schema with identity. Authors declare:

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

Adjacent slices preserve order; parenthesization may vary, permutation may not. Empty source and physical padding safely use identity. NaN-producing sentinels are not identities; use explicit validity if needed. The operation lowers to segmentation, local computation and merging; nested structured operations retain their own lowering semantics.

### 9.2 Region scan

```python
outputs, final_state = I.region_scan(
    source=source, axis=0, summarize=summarize_transition,
    combine=compose_transitions, identity=identity_transition,
    initial_state=initial_state, apply=apply_transition,
    emit=emit_slice, operands=captures)
```

Typed pure helpers define summarize(slice,captures)→Transition, ordered combine, apply(prefix_transition,initial_state)→incoming_state, and emit(slice,incoming_state,captures)→output_slice. They satisfy:

```text
apply(identity, state) == state
apply(combine(a, b), state) == apply(b, apply(a, state))
emit(A ++ B, state)
  == concat(emit(A, state),
            emit(B, apply(summarize(A), state)))
```

Emit's result axis keeps the slice member relation. Outputs reassemble in source order; final state is apply(summarize(full_source),initial_state). Internal boundaries/prefixes/counts cannot be observed. Host-visible chunk states or fixed chunk ABI require explicit chunk domains/subregions/ordinary scan.

Ordinary scan is the element-summary/output case; degenerate region forms canonicalize to it. Region scan is not an effectful arbitrary state loop. [FlashAttention](examples/flash_attention.py) explicitly writes QK/PV contractions rather than relying on pattern recognition. [Causal linear attention](examples/causal_linear_attention.py) has unobservable legal segmentation; [Mamba](examples/mamba_state_passing.py) exposes chunk ABI and therefore uses ordered carry.

## 10. Named multiplication and contract

### 10.1 Author entry points

```python
I.dot(lhs, rhs, *, acc_dtype)
I.matvec(matrix, vector, *, acc_dtype, transpose=False)
I.vecmat(vector, matrix, *, acc_dtype, transpose=False)
I.matmul(lhs, rhs, *, acc_dtype, transpose_lhs=False, transpose_rhs=False)
I.outer(lhs, rhs)
```

| Operation | Core input shapes | Result |
|---|---|---|
| dot | `[K]`, `[K]`, both rank1 | rank-zero tensor `[]` |
| matvec | `[...,M,K]`, `[...,K]` | `[...,M]` |
| vecmat | `[...,K]`, `[...,K,N]` | `[...,N]` |
| matmul | `[...,M,K]`, `[...,K,N]` | `[...,M,N]` |
| outer | `[M]`, `[N]`, both rank1 | `[M,N]` |

Matrix final two/vector final one axes are core; leading axes are batch. Matrix batch axes broadcast from the right; K must match without reduction-axis broadcast. Transpose swaps the matrix's last two axes only. Matmul inputs are at least rank2, without implicit vector promotion/result squeezing. Results list broadcast batch then M/N.

Dimensions describe post-transpose computation. Lhs `[64,32]`, rhs `[64,16]`, transpose_lhs=True gives `[32,16]`; transposed matvec on `[64,32]` needs vector64 and yields32. Matching remains required.

Multiplication operands share one numeric dtype; explicit casts handle mixed operands. Accumulator/result dtype is independent and explicit. No implicit conjugation, TF32, alpha/beta or mutable C initialization exists. Empty K yields accumulation zero. Frontend transpose/broadcast/paired axes normalize to contract retaining dynamic identity and runtime conditions, independent of provider matrix ranks.

Outer broadcasts ordinary multiplication, preserving dtype/pointwise numerics, without an empty-reduction contract or accumulator.

### 10.2 Generic contract

```python
acc = I.contract(a[m,k], b[k,n], reduce=((1,0),), batch=(), acc_dtype=I.f32)
```

Reduce pairs are nonempty, unique and extent-compatible. Batch pairs are unique/compatible/disjoint from reduction; each occurs once in the result via its lhs axis. Result axes first include lhs non-reduced axes in order (including batch representatives), then rhs non-reduced non-batch-member axes. Multiply is numeric multiplication, combine is addition, accumulation dtype explicit. Zero logical reduction extent yields additive zero, including scaled/sparse contracts.

There are no pretend multiply/combine surface arguments. Other semirings use pointwise + generic reduce; outer products use broadcast multiply. Axis permutations, flattening into M/K/N, MMA and staging belong to compilation.

## 11. Scaled contract

```python
I.scaled_matmul(lhs, lhs_scale, rhs, rhs_scale, *,
                lhs_format, rhs_format, group_size, acc_dtype)
```

This shorthand uses the same closed positional schema, fixed reduce, empty batch and equal group sizes as:

```python
acc = I.scaled_contract(
    lhs_mgc, lhs_scale_mg, rhs_gcn, rhs_scale_ng,
    lhs_format=I.e2m1, rhs_format=I.e4m3,
    lhs_group_size=32, rhs_group_size=32,
    reduce=((1,0),(2,1)), batch=(), acc_dtype=I.f32)
```

Lhs carrier/scale are `[M,G,C_lhs]`/`[M,G]`, rhs `[G,C_rhs,N]`/`[N,G]`, result `[M,N]`. G is one scale-group axis, groups agree, and carrier-inner extents follow elements-per-carrier and group size. This explicit schema prevents shape-based guessed scale relations.

G, formats, packing, scale relations, accumulation and rounding are algorithm meaning. Native scaled MMA/layout/storage/provider K flattening are lowering. E2M1/E4M3/E8M0 encodings/special values are in the numerical chapter. Other ranks/orders/batches explicitly normalize through ordinary transforms/control, not another schema. Packed INT4/INT2 uses explicit carrier decode/sign extension/zero point/scale and ordinary contract.

### 11.1 Closed quantized operations

Quantize q8_k and quantized_dot q4_k×q8_k are independent pure structured operations, not scaled contract. See [quantized operations](quantized-operations.md) for complete shapes, record maps and numerics. Formats do not select implementations; author control may share quantized preparation.

## 12. Sparse contract

```python
I.sparse_matmul(compressed, metadata, rhs, *, format, acc_dtype)
I.sparse_contract(compressed, metadata, dense_rhs,
                  format=I.sparse.two_of_four(...),
                  reduce=((1,0),), batch=(), acc_dtype=I.f32)
```

Named sparse matmul takes rank2 data with lhs K compression and supplies fixed reduce/empty batch; format/metadata remain explicit. Formats are closed typed schemas, not provider strings/opaque descriptors. Each defines group/nonzero count/compressed ordering/logical position mapping/validity/dtype.

One_of_two and two_of_four logical metadata/order are defined in the numerical chapter. New formats require concrete semantics. Authors interpret external packed metadata; native metadata packing/storage/sparse MMA belongs to lowering. Format-specific shorthand shares the canonical path. Sparse contract retains ordinary contraction reduction/batch/free/result/accumulator rules, replacing one operand's value relation along its compression axis.

## 13. Histogram

```python
counts = I.histogram(values, bins=BINS, valid=valid, count_dtype=I.u32)
```

Pure histogram accepts integer tensors, positive logical bins and Boolean scalar/tensor validity broadcastable without enlargement. Active values must be in `[0,bins)` and contribute once. Include a bounds predicate to ignore invalid values. Inactive members do not require in-range values or contribute. Output shape is `(bins,)`; empty input returns zeros; count overflow follows integer semantics. It is not an author-selected external initialization/atomic update algorithm.

## 14. Ordinary ragged composition

```python
member_source = I.domain(0, R)
members = member_source[offsets[group]:offsets[group+1]]
logical_members = mapping[members] if HAS_MAPPING else I.indices(members)
```

Offsets/subregions/optional mapping with SSA provenance fully express ragged semantics. Ragged/members helpers may expand mechanically but create no canonical dedicated operations.

## 15. Indexed access, buffers and copy

Indexing/gather share typed source identity/rank/result axes/coordinate expressions/dependencies/provenance and active validity. Invalid reads avoid source and use same-dtype broadcastable fills; invalid writes have no effect. Canonical forms distinguish pure tensor gather, external loads/stores, buffer loads/stores, arbitrary unique stores and scatter_reduce. Unique destinations must be injective; reduction scatter defines collisions by typed combine.

Logical buffers are mutable kernel-local state with author shape/dtype/initialization/order and compiler placement. Read→immutable SSA→write fully specifies snapshots/effects without canonical copy. Providers may form bulk/vector/async/DMA transfers from these relations.

## 16. Atomics

```python
value = I.atomic.load(address, order="acquire")
I.atomic.store(address, value, order="release")
old = I.atomic.add(address, delta, order="relaxed")
result = I.atomic.compare_exchange(address, expected, desired, order="acq_rel")
# result.old_value, result.success
```

Canonical forms are load/store/RMW(exchange/add/max/min/and/or/xor)/compare_exchange. Load permits relaxed/acquire; store relaxed/release; RMW/CAS all four. CAS returns old_value/success and mechanically derives failure order: relaxed→relaxed, acquire→acquire, release→relaxed, acq_rel→acquire.

No author scope exists. Allocation identity, alias/index relation and invocation determine logical participants; providers choose physical scope.

## 17. RNG

```python
bits = I.random.bits(seed, logical_counter)
uniform = I.random.uniform(seed, logical_counter, dtype=I.f32)
```

Canonical Philox4x32-10 maps scalar counters via block=counter//4, word=counter%4 to one output word. The numerical chapter fixes rounds/constants/conversion. It depends on no execution IDs/call ordering/mutable provider state. Composite distributions such as normal are author helpers.

## 18. Logical parts without partition

Observable part counts/identities/bounds/intermediate interfaces use domains/subregions/arithmetic/control. Empty/tail preserve member contracts. Logical parts specify no tiles/blocks/extra launches; unobservable intra-kernel blocking belongs to compilation. Partition(auto/count/extent) is neither public nor canonical.

## 19. Surface ownership

| Family | Canonical meaning | Shorthand | Excluded |
|---|---|---|---|
| Definitions/interface | kernel/helper/views/runtime/constexpr | decorators/types | target/hidden launch/provider dispatch |
| Domains/control | domains/subregions/control/parallel/carry | slices/indices/break/continue | auto/partition/state_stream/ordered/IDs |
| Tensor values | arithmetic/select/transforms/products/full/casts | zeros/activations/mask | physical tile/layout/padding |
| Structured compute | reduce/scan/region/contractions/quantize/quantized dot/histogram | named computations/prefixes/formats | whole-operator softmax/attention/MoE |
| Relations | subregions/index relations/sparse schema | ragged/member/index helpers | native metadata/MMA hints |
| Memory/effects | reads/writes/scatters/atomics/Philox | indexing/atomic conveniences | scope/storage/copy/barriers/pipelines |

Shorthand normalizes to one canonical path; physical facts never flow back into source through shorthand.
