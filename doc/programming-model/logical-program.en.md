# Logical program

## 1. Domains, subregions and identity

A domain is a finite ordered coordinate set, normally `[begin,end)` with `step > 0`. Empty domains are legal; bounds/step may be runtime scalars. Logical `index` has one cross-target numerical definition.

```python
axis = I.domain(0, N)
prefix = axis[0:valid_length]
window = axis[begin:end]
```

A subregion denotes `[begin,end)` with `source.begin <= begin <= end <= source.end`. Equal endpoints give an empty region; there is no implicit clamp. It retains source identity, bounds and provenance. Iteration and `I.indices` return absolute source coordinates, not local ordinals. Use `enumerate` or an explicit ordinal domain for ordinals. Physical padding changes no members.

Noncontiguous, repeated or reordered members use indexed relations. Only author bounds, input relations or algorithm metadata create KIR subregions; compiler blocking exists in physical programs.

### 1.1 Coordinate provenance

`I.indices(axis_or_subregion)` yields logical `index` tensors whose canonical relations retain source identity/axis/rank/dtype, typed maps from result axes to source coordinates, SSA dependencies on domains/subregions/indices/predicates, active members and known bounds.

Provenance is part of canonical values/relations, not axis-name strings or debug origins. Helpers preserve it componentwise with or without inlining. Slicing composes bounds without renumbering coordinates; tuples/records retain each component's facts.

Broadcast stores axis maps; permutations compose maps; reshape composes old/new axes via logical row-major linear coordinates. Coordinate integer arithmetic, comparisons and selects retain representable typed expressions/dependencies. When an operation cannot be represented exactly, its numerical result remains correct but coordinate/range analysis returns unknown rather than guessing from names, shapes or nearby structure.

## 2. Shapes and tensor values

Each dynamic tensor extent is an identity-bearing runtime value. Unknown extents are not equal merely because both are dynamic. Equality requires the same value or an operation's explicit relation.

Scalar/size-one broadcasting aligns axes from the right. Aligned extents must agree or one must be `1`. `0` with `1` produces `0`; `0` with another positive extent is incompatible. Runtime equality/size-one conditions remain explicit.

Reshape preserves row-major element order/count, allows at most one uniquely inferred extent, and changes no dtype, bits or storage. Transpose/permute specify logical permutations. Both compose coordinate relations, not just result shapes.

`join(lhs,rhs)` requires equal dtypes/shapes and adds one trailing axis: `result[...,0]=lhs[...]`, `result[...,1]=rhs[...]`, `shape(result)=shape(lhs)+[2]`. It is neither a record nor general concatenate/interleave. Subsequent reshape interleaving follows this map and row-major order.

## 3. Unordered parallel iteration

```python
for point in I.parallel(domain):
    ...
```

Every logical point executes once with no observable order between points. Carries across iterations and `break` are forbidden; `continue` skips only the current point. Conflicting non-atomic/non-reduction effects are illegal. No program count, lane, tile or actual simultaneous execution is specified.

The compiler may serialize, flatten, thread or vectorize the iteration. It may also infer equivalent parallel work from tensor flows with unique writes without an explicit `parallel` wrapper.

## 4. Ordered control and carries

Ordinary `if/for/while` preserve order. Scalar Boolean conditions control SSA merges; domains iterate logically; updates create loop carries; break, continue and early return normalize as structured control. No `ordered` marker is required. The compiler changes organization only with a dependence/effect proof of equivalent results.

## 5. Recurrences and homomorphic region operations

There is no `state_stream` allowing arbitrary bodies to be resegmented by compiler-selected extents: invocation counts, tails, local shapes and effects would generally make the semantics ambiguous.

Five distinct structures are available:

1. Associative, commutative element summaries needing a final result: reduce.
2. Associative element summaries needing ordered prefixes: scan.
3. Author-defined summaries of arbitrary contiguous slices: region fold.
4. Slice summaries, composition, incoming-state application and per-position output: region scan.
5. Strict sequence dependence, dynamic termination, nonassociative state or ordered effects: ordinary loops/carries.

For every ordered complete contiguous segmentation `R0 ... Rn`, region fold uses `summarize(Ri)->Summary`, `combine(Summary,Summary)->Summary` and an identity satisfying:

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

Adjacent `A,B` preserve source order. Combine permits any order-preserving parenthesization, not permutations. Choosing the operation declares these equations as algorithm semantics. The compiler checks types, schemas, purity, effects and source-axis relations, not arbitrary mathematical associativity.

Summarize receives tensor components sliced along the same axis and explicit captures. Pure tensor operations, including reduce/scan/contract, are allowed. External/buffer writes, scatter, atomics, RNG and other observable effects are forbidden. It cannot observe segment ordinal/count/extent or chunk-relative coordinates. Pass `I.indices(source_axis)` as a sliced component to consume absolute coordinates.

Region scan additionally defines `apply(prefix_summary,initial_state)->incoming_state` and `emit(source_slice,incoming_state,captures)->output_slice` with:

```text
apply(identity, state) == state
apply(combine(a, b), state) == apply(b, apply(a, state))
emit(A ++ B, state)
  == concat(emit(A, state),
            emit(B, apply(summarize(A), state)))
```

All helpers are typed and pure. Emit preserves the slice's member relation. Outputs reassemble on the original source axis; final state is `apply(summarize(full_source),initial_state)`. Compiler segment boundaries/count and internal prefixes cannot become shapes or ABI values. Observable chunk states/counts require explicit logical chunk domains, subregions and ordinary scans.

Reduce/scan operate on element summaries; region operations require actual author-written region summarizers/emitters. The compiler realizes their segmentation/traversal/merging/output while preserving each order/effect contract; nested operations lower by their own contracts. Observable page/window/group/chunk boundaries are source-derived subregions. Pure physical blocking extents remain absent from source.

## 6. Structured tensor operations

Reduce, scan, contract, scaled/sparse contract and histogram are logical tensor operations, not requests for target primitives. They preserve respectively: accumulator axes/identity/pure associative-commutative combine; ordered prefixes and associative combine; paired batch/reduction and free axes with accumulation; closed `[M,G,C]/[M,G] × [G,C,N]/[N,G]` microscale schema; compressed values/format/metadata interpretation; binning into counts.

Each permits its stated reassociation, not a selected tree, MMA, layout, storage or pipeline. Targets map natively, expand legally or reject without redefining logical operations.

## 7. Logical parts do not use partition

Observable part counts, identities, boundary formulas and intermediate interfaces use ordinary domains, subregions, arithmetic and control. Empty/tail/intermediate shapes belong to the algorithm, not tiles, blocks or launches. `partition(auto/count/extent)` is absent; unobservable blocking is physical.

## 8. Ragged and indexed relations

Ragged data consists of an outer domain, member source domain, offsets and optional index mapping. For `g`, contiguous members are `member_source[offsets[g]:offsets[g+1]]`; noncontiguous mapping forms an indexed relation. SSA identity/provenance distinguishes simultaneous relations.

No independent canonical RaggedOp/MembersOp exists. Surface helpers must expand mechanically into this sole representation.

## 9. Indexed memory and effects

Indexed access shares typed relations and active validity: source identity/rank, result axes, typed expressions for source coordinates and their SSA provenance. Composition preserves relations, not just shapes.

Immutable tensor selection is pure gather; external-view loads are external reads; buffer loads are mutable-resource reads. Invalid reads do not access source and yield explicit fills; invalid writes produce no effects.

Ordinary assignment and unique scatter require provably injective arbitrary-index stores. Collision reductions use separate typed `scatter_reduce`. Atomics retain indivisibility, modification order, old values and memory order.

There is no canonical copy. Immutable SSA reads followed by indexed writes define snapshots, mapping, validity, casts, aliases and effect ordering. Physical programs form bulk/async/DMA/TMA transfer and synchronization.

## 10. Atomics and RNG

Atomics have no author-visible physical scope. Executions accessing the same atomic object through the same logical allocation/address relation in one invocation share modification order. `relaxed/acquire/release/acq_rel` are semantic happens-before choices; mapping and targets determine physical scope.

Conflicting non-atomic accesses are illegal races. Parallel iterations are not guaranteed simultaneous residency or spin-wait progress.

RNG is pure stateless Philox4x32-10 with fixed seed, logical counter, uint32 wrap, round constants, output words and uniform conversion. It observes no program/thread/lane identity, call ordering or mutable provider state.
