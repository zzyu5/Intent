# Types, numerics and effects

## 1. Canonical value types

Logical values use bool, signed64 logical index, i8/i16/i32/i64, u8/u16/u32/u64, f16/bf16/f32/f64, explicitly encoded f8e4m3fn/f8e5m2, tuples/records, ranked tensors, external views and kernel-local buffers. Physical lanes, padding, layout, address width and target encodings are absent from canonical types.

Tuples/records are immutable products, not tensors or memory layouts. Identity includes ordered component types and record field names/order. Components may be heterogeneous/nested, without common shape/dtype. Products are SSA/helper/carry/accumulator values, not buffer elements, public ABI parameters or view elements. Buffers store one scalar element dtype in a ranked tensor.

I4/u4/fp4 are not ordinary addressable elements. Packed data uses u8/u16/u32 carriers with explicit indexing/shifts/masks/sign extension/scales. Scaled schemas define microscaling; [Q4_K/Q8_K](quantized-operations.md) uses ordinary u8 carriers with independent record/operation semantics, not new scalar dtypes.

## 2. Logical indices and shapes

Index uses signed64 arithmetic. Address widths may narrow with proven bounds without changing index results. Dynamic extents have identity and are compatible only through the same value or explicit equality. Domains/subregions/transforms determine rank/extents; [logical-program rules](../programming-model/logical-program.md) define broadcast/reshape/join.

Tensor/view `.shape` is a logical extent tuple. Full requires nonnegative extents and typed scalar fill broadcast across members; zero extents are empty. Logical shapes do not become physical fragments.

I.Enum is constexpr-only: closed names/unique values participate in specialization, same-type comparison and hardware-independent constexpr control, not runtime ABI/arithmetic/cast/bitcast. Python integer conveniences do not define canonical representation.

## 3. Literals and promotion

Untyped Python literals instantiate to a directly expected dtype if representable. First runtime creation without context uses bool/i64/f64 for bool/int/float; later uses cannot change dtype.

Frontend fixes pointwise result types with explicit typed operations. Same-dtype runtime operands operate directly; mixed runtime dtypes require explicit cast; comparisons yield bool. This is Intent's language choice, not an inherent typed-IR or Triton restriction. Cross-provider consistency comes from one contract and lowering.

Default sum/cumsum accumulator widening:

- i8/i16→i32; u8/u16→u32.
- i32/i64/u32/u64 retain dtype.
- f8e4m3fn/f8e5m2/f16/bf16→f32.
- f32/f64 retain dtype.

Other builtin reductions/cummax retain dtype unless their schema says otherwise. Dot/matvec/vecmat/matmul require explicit contract accumulator/result dtype; outer uses ordinary same-dtype multiply.

Storage dtype, operand dtype, accumulator/result dtype and internal instruction dtype are distinct. Builtin conversions precede combines, without requiring authors to cast an entire external tensor. Generic components match corresponding identities/results. Different backend internal representations may implement the same contract.

## 4. Integer arithmetic

Fixed-width integers use two's complement and modulo semantics:

- Add/subtract/multiply wrap modulo `2^width`.
- Signed floor division/remainder obey Python `a=q*b+r`, q=floor(a/b), with remainder zero or divisor's sign.
- Unsigned division/remainder are Euclidean, `0<=r<b`.
- `/` is not overloaded for integers; explicitly cast to float.
- Signed right shift is arithmetic, unsigned right shift logical; left shift keeps low width bits.
- Bitwise operations use fixed-width patterns; shifts require `0<=amount<width`.
- Division by zero and signed_min//-1 are invalid inputs.

Modulo is integer-only. Floating remainder requires a separate defined named operation, not incidental Python/C/provider behavior. Every target implements the same rules.

## 5. Floating point, cast and bitcast

Ordinary floating operations use the format's IEEE-754 values and nearest-even rounding. Except allowed FMA/local fusion, structured operations or explicit approximation below, dependencies/evaluation relationships are preserved without implicit fast math that changes result sets.

Lgamma computes log|Gamma(x)|, returns +0 at1/2, +inf at nonpositive integer poles (including signed zeros) and either infinity, and propagates NaN, retaining dtype/range/conversions.

Asin returns the principal value in `[-pi/2,pi/2]`, preserves signed zero and yields NaN for |x|>1, infinities or NaN. It uses library precision, not guaranteed correct rounding or a prescribed approximation's internal rounds. Sub-f32 inputs evaluate at f32 then cast back; f32/f64 use their precision. Special values align with [libdevice asin](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinf.html).

Log1p/erfc/i0 are pure elementwise library operations preserving floating dtype/shape:

- Log1p computes log(1+x) without a separately rounded 1+x, preserves signed zero, returns -inf at -1, NaN below -1 and +inf at +inf.
- Erfc is not rounded 1-erf: signed zeros→1, +inf→+0, -inf→2.
- I0 is even: signed zeros→1, infinities→+inf; large finite values obey format overflow.

All propagate NaN. Maximum ULP errors against the corresponding nearest-even result for f32/f64 are log1p1/1, erfc4/5, i0 6/6. Lower precision evaluates in f32 then casts. These library bounds follow [CUDA math accuracy](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix) without changing adjacent arithmetic/approximation/FTZ.

### 5.1 Explicit approximate math

Approximate is operation semantics, not a hint. Fdiv/exp2/tanh default false. Fdiv/exp2 permit FTZ only with approximate=True. Options are constexpr bool, non-default operands/results f32 without implicit conversion. Neighboring operations and collectives retain their contracts.

The cross-target approximate contracts do not promise correct rounding or matching bits:

- Exp2 error≤2 ULP; -inf→+0, +inf→+inf, NaN→NaN, either zero→1.
- Tanh finite nonzero relative error≤`2^-11`; signed zero preserved, infinities→signed1, NaN propagated, subnormal input retains near-zero value.
- Fdiv uses approximate reciprocal/multiply semantics. For `2^-126<=abs(rhs)<=2^126`, error≤2 ULP. For `2^126<abs(rhs)<2^128`, finite lhs produces quotient-signed zero and infinite lhs produces NaN. Other special values follow division classification.

FTZ converts subnormal inputs/outputs at the selected operation's boundaries to signed zero without modifying memory or neighbors. Unsupported modes reject. Typed attributes survive CSE/cloning/recomputation/lowering; operator kind alone does not define equal pure values. Bounds use the closed ranges of [PTX floating instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions); other hardware must satisfy them too.

### 5.2 Cast and bitcast

Cast defines integer→integer by mathematical value modulo destination width then destination signedness (sign/zero widening, low-bit narrowing); integer→float nearest-even; float→integer truncation toward zero with NaN/infinity/out-of-range invalid; float narrowing to bf16/f16/f32 nearest-even with finite overflow→infinity. Float8 rules follow below. Saturation is a separate explicit operation.

Bitcast requires equal total bit widths, preserving bits without numerical conversion or repacking.

Maximum/minimum propagate NaN; maximum_num/minimum_num ignore one-sided NaN and are different operations. IEEE arithmetic/casts preserve signed zero. NaN results promise NaN, not payload/signaling/sign identity.

Float8 encodings:

- E4M3FN has sign1/exponent4/fraction3/bias7, subnormals and no infinity. `S.1111.111` is NaN; remaining encodings are finite, maximum448. Nearest-even casts require finite input within representable range; explicit saturation handles clamp.
- E5M2 has sign1/exponent5/fraction2/bias15 and subnormals. All-one exponent/fraction0 is infinity; nonzero fraction NaN. Nearest-even finite overflow produces infinity.

### 5.3 Local contraction-add fusion

Zero-initialized ordinary contract with a sole same-dtype add consumer may use the other operand C as its accumulator initial value when result coordinates correspond elementwise. Contract(A,B)+C or C+contract(A,B) then uses fused accumulation rounding/special values, not separately rounded whole-contract plus add. Empty reduction yields C.

Preserve input precision, paired axes, accumulator dtype and effects, without TF32/FTZ/other approximations. This is target/name-independent and needs no hint/tuning parameter. It does not cross casts, multiple consumers or reassociate an already nonzero initial contraction. Pure shape projections may accompany fusion only with unchanged elementwise mapping.

### 5.4 Local multiply-reduction fusion

A floating multiply used only by an additive-zero reduce with ordinary same-dtype add, no intervening cast and product=accumulator/result dtype may become ordinary contract. Pure projections preserve correspondence; broadcast/reduction relations determine paired/batch/free axes without changing members/results/effects. Widening casts, scans and ordered loops are excluded.

Fusion retains input/accumulator precision with no TF32/FTZ/other approximation. Contract FMA semantics may change rounding or extreme-overflow/cancellation Inf/NaN compared to separately rounded products. Empty reduction still returns typed zero. No new hints or unrelated numerical permission is implied.

### 5.5 Ordinary multiply-add FMA fusion

Same-dtype a*b+c, c+a*b, a*b-c and c-a*b may use native FMA with one nearest-even rounding. Last bits, signed zero and extreme Inf/NaN may differ; unfused uses retain independent multiply results. Multiplication need not have a sole consumer, but fusion at one use cannot change another use's required product.

Preserve dtypes/input precision/casts/loop order/effects. FMA grants no general reassociation, precision reduction, TF32, FTZ or undeclared approximation. Provider selects fusion without hints/config/new algorithms. Legal implementations need not match bits operationwise. Explicit independent rounding boundaries, notably quantized-record R32 sequences, take precedence and cannot be crossed.

### 5.6 Numerical permissions on compilation calls

CompileOptions.numerics defaults to source, retaining ordinary float/FMA/local fusion/structured contracts without extra reassociation or bitwise identity requirements.

Explicit relaxed_normalization applies only to normalized summaries on one member domain built from validity, maximum, exponential mass and weighted contraction. It permits contiguous local-max segmentation and exponential rescaling/merging of partial mass/moments, including changing low-precision weight casts from global to segment maxima and resulting accumulation/rescaling differences.

Callers must ensure finite valid scores and every value actually entering weighted contraction, including zero-weight products. Unperformed masked accesses create no new requirement. The compiler inserts no input scan/finite check. Inputs violating these requirements have no guarantee under this mode; source adds no such precondition.

Allowed differences include rounding, underflow and intermediate overflow from that particular rewrite; low-precision weights need not match global-normalization values. Empty/all-inactive domains retain identity. Members/result coordinates/effects/ABI/declared dtypes stay fixed. The mode permits no unrelated reassociation/TF32/FTZ/approximation and crosses no quantization rounding boundary.

Online_reduction=False disables only the optional rewrite, not source meaning. Enabling it guarantees no adoption. Explicit author region summarizers/combiners follow their own contracts without this extra mode.

## 6. Reduce and scan numerics

Reduce declares associative/commutative/neutral-identity parallel aggregation; parenthesization and permutation may change. Floating finite-precision tree/reorder differences are accepted without bitwise algebraic proofs, while members/dtypes/NaN/tie/effects remain. No undeclared precision/FTZ permission follows.

Scan defines every logical prefix with ordered reassociation and no member permutation. Strict left folds, nonassociative recurrences and ordered effects use loops. Authors undertake algebraic preconditions; verifiers check schemas/axes/purity/captures/effects. Deleting source reduction axes yields accumulation schema.

Generic identities are explicit componentwise; builtins supply canonical identities. Empty reduce returns identity; empty scan returns empty tensor. Scan arguments define inclusive/exclusive/direction.

Max/cummax propagate NaN and use minima: floats with infinity→-inf, E4M3FN→-448, signed integer/index→minimum, unsigned→0. Sum/cumsum use additive zero; any/all false/true. Arg-max chooses lowest logical index on equal values and propagates NaN. Other policies require another typed combine.

Region fold/scan allow source-order-preserving reassociation over compiler slices, never reduce's permutation permission. Author equations are:

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

The compiler verifies source axes/schemas/captures/purity/output relations/effects, not algebra. Identity must be neutral under actual float/NaN semantics. A max=-inf, mass/moment=0 record fed unconditionally into exp(max-merged_max) yields NaN at -inf-(-inf) and is not a valid identity. Use explicit validity with finite normalized differences before exponentiation, or another neutral typed representation. Empty fold yields identity; empty region scan yields empty output/initial state.

Summarizers observe no segment identity/extent; coordinates are absolute source indices, never renumbered. Region scan additionally requires identity action, ordered composition and emit equivalence across adjacent slices with propagated incoming state. Legal segmentation/parenthesization may alter rounding but not order/NaN/dtypes/approximation.

## 7. Scaled, sparse and histogram numerics

Scaled schemas define logical formats, carriers, scales, groups, rounding, special values and accumulation without native restrictions redefining them.

- E2M1 occupies4 bits: sign at bit3, positive codes0..7→0,0.5,1,1.5,2,3,4,6; sign negates. No NaN/infinity. Two increasing reduction-coordinate elements share u8 with earlier element in low nibble.
- E4M3 uses E4M3FN encoding above.
- E8M0 scales are unsigned8 exponents: codes0..254→`2^(code-127)`,255→NaN.

Positive group_size and closed carriers/scales are lhs `[M,G,C_lhs]`/`[M,G]`, rhs `[G,C_rhs,N]`/`[N,G]`, with equal G/groups. Reduction pairs G and inner carrier axes. Flattened k reads scale group k//group_size; C=group_size/elements_per_carrier. Decode then scale before contraction. Other ranks/orders explicitly normalize; lowering cannot guess another group relation.

Sparse metadata denotes logical positions, not native packed instruction layout:

- One_of_two requires compression extent divisible by2; one value per pair, index metadata0/1.
- Two_of_four requires divisibility by4; two values ordered by position, record `{first:index,second:index}` with `0<=first<second<4`.

Other positions are additive zero. Authors decode external packed metadata with bit/index operations without dense materialization or loss of structure. Invalid metadata/divisibility is caller error. Native repacking is a physical representation preserving mapping.

Histogram accepts integer values, positive bins, broadcastable Boolean validity. Active values in `[0,bins)` contribute once; inactive values need no range validity. Result `(bins,)` uses explicit count dtype with wrap overflow; empty input returns zeros.

## 8. External views

Views declare dtype, logical shape, allocation identity/element offset, runtime element strides, access direction, bounds and needed aliases. Element addresses are `base[offset+sum(c_i*stride_i)]`. Negative strides are legal; zero stride is legal for read-only broadcasting and writable only when collision semantics permit. Active coordinates map in bounds; inactive accesses form no address and nothing clamps.

View parameters may alias by default. Noalias is a caller precondition, not a hint. Alignment/contiguity/vector/TMA/descriptor eligibility belong to runtime/compiler/provider unless interpreting an external format itself requires alignment.

## 9. Logical buffers

`I.buffer(shape,dtype,init=None)` requires shape/dtype. Missing init/None is uninitialized; tensor init explicitly supplies contents. Buffers are kernel-local mutable state, newly disjoint from external/other new buffers, with lexical lifetime and no escape. Authors ensure every active read follows a write of that element in the same allocation instance. Compiler chooses storage/materialization/placement. Cross-kernel tensors are host-owned.

Insufficient static initialization proof neither rejects ordinary lowering nor proves initialization. Optimizations relying on it must prove their own requirements. Reading unwritten elements violates the contract even if compilation succeeds.

## 10. Indexed access and collisions

All accesses share typed relations with source identity/rank/result axes/coordinate expressions/SSA dependencies. Composition preserves expressions, not just shapes. Invalid reads avoid access and return same-dtype fills broadcastable to result; invalid writes do nothing.

Effects distinguish external/buffer reads/writes, unique arbitrary stores, scatter-reduction and atomics. Unique active destinations must be injective. Scatter-reduction uses typed pure combines without per-update linearization/old-value guarantees. Ordered control preserves effects; unordered conflicts need uniqueness/reduction/atomic semantics or are illegal.

Logical copy is read→immutable SSA→write, not a canonical operation. Non-atomic access has no memory-order/scope fields.

## 11. Atomic memory semantics

Canonical atomics are load/store/RMW(exchange/add/max/min/and/or/xor)/CAS. Each logical address has one modification order. RMW is indivisible and returns old value; CAS returns `{old_value,success}`.

Relaxed guarantees atomicity/modification order. Acquire reads matching release/release sequence and establishes happens-before. Release publishes preceding ordinary/atomic effects; acq_rel does both.

Load permits relaxed/acquire, store relaxed/release, RMW/CAS all four. CAS failure derives relaxed→relaxed, acquire→acquire, release→relaxed, acq_rel→acquire. No public/canonical scope exists; same-invocation allocation/address relations define logical participants and mapping determines physical scope. Conflicting non-atomic access is a race. Parallel residency and spin-wait progress are not guaranteed.

## 12. Deterministic RNG

Pure stateless Philox4x32-10 takes u64 seed and nonnegative u64/losslessly convertible index counter, returning u32:

```text
block = logical_counter // 4
word  = logical_counter % 4
(c0,c1,c2,c3) = (low32(block),high32(block),0,0)
(k0,k1) = (low32(seed),high32(seed))
repeat 10 rounds:
    (hi0,lo0) = mul_wide_u32(0xD2511F53,c0)
    (hi1,lo1) = mul_wide_u32(0xCD9E8D57,c2)
    (c0,c1,c2,c3) = (hi1 ^ c1 ^ k0,lo1,hi0 ^ c3 ^ k1,lo0)
    k0 = k0 + 0x9E3779B9 (mod 2^32)
    k1 = k1 + 0xBB67AE85 (mod 2^32)
result = (c0,c1,c2,c3)[word]
```

No target/program/thread/lane/call counter is read. Uniform f32 is `f32(bits>>8)*2^-24` in `[0,1)`; authors compose other distributions. Providers substitute native RNG only with the same bit stream.

## 13. Closed typed input constraints

Only two constraint classes exist: `I.assume_in_bounds(index,resource,axis=...)` for typed index bounds and external-view alias/noalias for allocations. Assume_in_bounds is a no-return statement; retain the original index instead of assigning its result.

There is no arbitrary Boolean assume or optimization hint for alignment, contiguity, sortedness, uniqueness, equality or format validity. Future constraints need a real expressiveness need, closed typed schema, explicit violation semantics and unique KIR representation, not open assertions authorizing optimization.
