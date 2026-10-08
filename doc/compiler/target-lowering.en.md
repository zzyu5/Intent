# Target extensions and lowering

## 1. Orthogonal choices

Compilation selects a source provider (Triton/cuTile/future direct backend) and hardware (NVIDIA/AMD SM/gfx/other GPU architecture) separately. Triton source may compile externally to NVIDIA/AMD; Intent does not prebuild a vendor IR per provider or dialect per architecture version.

## 2. Target capabilities

External typed capability facts describe grid/program limits, warp/wave sizes/resources, supported scalar/fragment dtypes, structured/atomic operations, source forms and local extension legality. Passes query features, not device names. One IR with feature predicates/pipelines/patterns covers SM90/100/120 and gfx differences.

## 3. Common IR and local extensions

Common GPU IR stays the sole complete executable authority. Extensions augment its program without duplicating mapping/value/access/structured graphs. An extension must be genuinely inexpressible without loss and inappropriate to common IR, more than spelling/order differences, consumed by later passes, independently verified, and in current IR before serialization. Otherwise use thin legalization/direct serialization.

Before local forms, compare source/provider operation contracts. Equal contracts with native primitives map directly. Internal helper organization does not justify reimplementing provider collective trees/communications/layout strategies. Stronger order/numerical requirements need author-semantic reasons, not accidental historical limitations.

Approximation/FTZ are shared unary/binary numerics, not new API-name dialects. Legality checks dtype/capability and serialization emits matching primitives/wrappers without kernel-wide fast math. Cloning/bufferization/scalarization preserve both attributes; ordinary neighbors remain unchanged.

## 4. Triton

Deterministic mappings include coordinates→program_id/grid, physical fragments→arange/tensor values, accesses→pointers/masks/fills/load-store, proven effective ranges→actual loops/unmasked or guarded bodies, gather/scatter/atomic primitives, reduce→tl.reduce with defined types/identity/NaN/ties, scan→associative_scan with ordered prefix/direction/inclusion, contract/scaled→dot/dot_scaled or legal expansions, region operations→already-formed segment/summarizer/combine/apply/emit structure, and structured control.

Reduce's associative/commutative declaration requires neither provider reproof nor extra ordered tree or scan-terminal lowering. Provider promotion/NaN/dot precision defaults may differ: close gaps through determined conversions/callbacks/precision options instead of blindly adopting defaults or copying machine instruction types.

Ordinary float multiply-add uses `enable_fp_fusion=True` under [FMA semantics](../dsl/types-numerics-and-effects.md#55-ordinary-multiply-add-fma-fusion); LLVM/PTX forms legal FMAs. Do not add a duplicate matcher/config. This is separate from arbitrary reassociation, dot precision and FTZ; libdevice FTZ reflection remains off. Explicit casts/independent rounding survive.

Real Triton local forms may represent descriptors/accesses, compile-time branches and Config bindings. Pure pointer/descriptor spelling serializes directly; changed operands/static blocks/multiple consumers justify local operations.

Descriptors explicitly preserve base/shape/strides/block shape/padding/alignment/accesses and declared runtime allocation size/alignment/allocator ABI/launch lifetime. Launchers bind declared allocator requirements; missing support rejects before serialization. Serializers do not invent descriptor workspace/allocation paths.

Intent does not copy TTGIR encodings/layout conversion, coalescing/thread locality/MMA/shared/TMEM, TMA/software pipelines/warp specialization, fences/barriers/register/LLVM/PTX lowering or autotune winners.

## 5. cuTile

cuTile shares block identity/static fragments/pointwise/access/collectives/MMA/control with Triton. Thin projection maps coordinates→bid/grid; shapes/relations→tile-space indices; accesses→load/store/gather/scatter; structured compute→sum/max/cumsum/mma/mma_scaled; regions→shared segment/summary/state/output flows.

Local legality handles ≤3 block dimensions, tile-index multiplication, bounds/padding, advanced indexing, scaled-MMA layout and tuning constraints. Only nonmechanical structure requires extensions. Missing cooperative copy/barrier/sync forms must legally map to supported equivalents or reject, not pretend support with a serial slow path.

## 6. Vendor and architecture extensions

```text
common GPU program + optional provider/vendor extension ops
    → target-specific legalization → source or lower IR
```

Extensions may require features; one pass may choose feature-specific patterns. Architecture differences stay out of KIR and never use kernel-name branches.

## 7. Provider verifier

Check grid ranks/coordinates/static fragments, unique lowering of common operations, complete local operands/results/regions, rejection of unsupported dtype/primitive/access/sync, bound constexpr/config parameters, and no rereading KIR to reconstruct ownership/axes/ranges/provenance/access/validity.

## 8. Terminal serialization

Serializers emit imports/signatures/decorators; print current values/control/operations in order; mechanically translate types/attributes/spelling; and publish declared configuration sets/launch wrappers.

They do not infer widths from result shapes, rebuild masks/pointers from KIR, choose ownership/primitives by roles/names, create buffers/workspace/persistent loops/pipelines, add parameters, or catch errors for fallback.

## 9. Unsupported categories

The earliest informed layer reports unsupported: shared program execution legality; missing provider operation/form; hardware dtype/resource/primitive capability; or lower compiler cost limits despite valid source. These are distinct diagnoses, never hidden by algorithm substitution, narrowed scope or order-of-magnitude-slower replacement.
