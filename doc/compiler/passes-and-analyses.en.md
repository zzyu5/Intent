# Analyses, passes and verification

## 1. Basic discipline

A pass is justified by changing the current program while preserving verifiable invariants, not by filenames, pass counts or side-table fields. Transformations declare semantic/current-physical inputs, legality/preconditions, rewritten types/operations/regions/def-use, preserved logical values/control/effects/ABI, invalidated/recomputed analyses and post-pass verification.

Shared policy is not driven by kernel names, operation counts, whole-region templates or provider strings.

Module `intent.compile_options` preserves effective permissions and controls: numerics=source or relaxed_normalization, onlineReduction for optional normalized-summary rewrites, optimizationRemarks for decisions. Defaults are source/true/false. Construction, branch cloning and provider containers carry the attribute. Continuing physical IR consumes its existing permissions without silently replacing/defaulting them; metadata agrees and caches use actual options.

Online transformation checks current summary/axes, numerical permission, identity/dtype and replay/effects before creating structure. Controls never bypass validation/required lowering. Additional-mode implementation eligibility currently requires f32 score/maximum/mass-moment accumulation and matching f16/bf16/f32 weights/values; unmatched cases keep ordinary blocking. This limits an optimization, not legal DSL dtypes. [Numerical permission §5.6](../dsl/types-numerics-and-effects.md#56-numerical-permissions-on-compilation-calls) defines finite-input obligations and specific rounding differences.

Optional remarks locate adopted/disabled/not-permitted/ineligible decisions on current operations. They are not execution plans. An execution family without the optimization may leave permission unused rather than change its computation.

## 2. Analysis layers

### 2.1 Canonical analyses

Immutable KIR analyses cover shape/domain/subregion identity; coordinate provenance/axis maps/active index sets/relations/alias; effects/collisions; control/dependence/carries/state; structured schemas; buffer lifetimes; and def-use/producer-consumer/reuse. Stable KIR IDs key these analyses. Do not rerun a pretend canonical KernelModel on physical programs.

### 2.2 Construction facts

Conversion produces logical worksets/groups, initial mapping/origin and complete scalar/fragment/value/access/control graphs. After construction these are in GPU IR, not another authority consumed by later passes.

### 2.3 Physical analyses

Current GPU analyses cover coverage/mapping; dominance/def-use/loop dependence; fragment axes/compositional coordinates/predicate ranges/footprints/validity; reuse/rematerialization/lifetimes/sharing; structured accumulator flows; resource estimates; and provider-form eligibility. Origins may reference immutable semantics but KIR adjacency/shapes do not supply absent physical structure.

Relevant mutations invalidate analyses unless preservation is proven. A transformation group performs the complete rewrite and related value/access/aggregate/accumulator maintenance; intermediate repair helpers are not independently exposed passes. Verify the same current program after each group. Diagnose rewrite/postcondition failures with group identity; verifiers do not repair.

## 3. Canonical GPU pipeline

### 3.1 Construct an executable program

Construction from KIR/analyses/capabilities forms the complete program in [KIR to GPU](kir-to-gpu.md), closing types/regions/values/access/origin without emitter-interpreted holes.

Construction closes the initial program/indexed-access composition only. Structured source normalization, ownership/blocking, online summaries and region/reduction/contraction realization are subsequent explicit groups. Normalize source relations before ownership, form physical ranges before realization, recompose new accesses before provider legalization, and let refinement/configuration materialization consume completed programs.

### 3.2 Refine program mapping

Worksets/access/reuse/effects/resources drive program rank/linearization, group placement, ownership extents, grouping/swizzling and grid-stride/persistent structures. If lower providers reliably form persistence from ordinary structure, preserve their legal input rather than duplicate loops. If Intent owns it, create actual physical loops.

### 3.3 Form blocking and fragments

Rewrite scalar/small ownership into parameterized fragments and nested loops: enlarge owned instances, separate free/ordered/reduction/scan axes, create conversions, update all users/accesses/validity and preserve runtime subregion source coordinates. Never modify logical subregions or observable page/window/chunk boundaries.

### 3.4 Co-realize values, accesses and structured operations

Fragment shapes, accesses/materialization and accumulator/reuse structures interact and cannot be three uninformed side-table decisions. Multiple passes/canonicalization are allowed while every step keeps a complete program:

- Replay/rematerialization changes actual def-use.
- Bufferization creates allocation/lifetime/reads/writes.
- Access realization creates coordinates/validity/fill/effect operands.
- Structured realization creates fragments/loops/carries/accumulation. Region summarizer contractions remain explicit, not rediscovered by attention algebra recognition.
- Zero-initial ordinary contract may merge with a sole same-dtype add consumer under local fusion semantics. Current def-use/result coordinates/dominance prove legality and the other operand becomes the accumulator. Do not cross casts or add fusion in serializers.
- Sole-use floating multiplication with same-dtype sum may normalize to contract under local fusion rules. Broadcast/reduction/result relations determine paired/free/batch axes and reuse existing contract lowering, without crossing casts or ordered control.
- Neutralization actually simplifies validity/fills, not only padding records.

### 3.5 Predicate range narrowing and summary emptiness

Canonical provenance composes indices/domains/subregions/relations through helpers/slicing/transforms/arithmetic/comparisons. After blocking, physical predicate analysis uses typed coordinate expressions, current ranges/ownership and identity/effect facts. Exact monotonicity/bounds/subset proofs yield all-true/mixed/all-false contiguous intervals; otherwise preserve traversal/predicates.

All-true may remove validity, mixed retains it, all-false may skip only when every free lane yields identity with no effect. Actual loop bounds/coordinates/active sets/validity change while logical source relations remain.

Typed propagation substitutes predicates into select/mask and proves identities through builtin reduction identities, zero contractions, constant folding and component equality, without names/attention shapes. Any unproven component prevents deletion. For current q in `[q_begin,q_end)` and q>=k, k before q_begin is all-true, inside the query interval mixed, after q_end all-false: a coordinate-set rule, not causal matching.

After narrowing, summary emptiness may prove each free lane first receives a nonempty summary or that the concrete graph never combines two empty summaries. Subject to NaN/identity/order/empty semantics, eliminate optional validity carries/selects. Canonical total identity stays unchanged; failed proofs or possible empty-empty paths retain validity.

### 3.6 Capability legalization

Shared verifiers check neutral legality. Provider/hardware checks cover grid/static fragments, native structured inputs/dtypes, descriptor/tile/storage/copy/sync forms, parameters and resource constraints. Reject where enough information is known; do not generate order-of-magnitude-slower pseudo-support or await JIT timeouts.

### 3.7 Deterministic serialization

Serializers traverse legalized current IR only. They do not query canonical KernelModels, infer fragments from result shapes, parse roles, create workspace/loops/masks/grid or append search parameters.

## 4. Semantics-preserving rewrites

Physical graphs may differ from KIR under its semantics and explicit compilation permissions. Legal examples include pure fusion/rematerialization; reduce reassociation/permutation vs ordered scan/region reassociation and contract's own rules; local contraction-add and multiply-reduction fusion; paired-axis flattening/permutation; blocking/accumulators; permitted normalized online summaries; proven identity-only traversal/validity removal; exact read-write bulk transfers; grouping/swizzling/grid strides.

Illegal changes include ordered recurrence→reduce, altered combines/dtypes/NaN/ties/atomic orders, changed logical members, kernel-count changes/hidden cross-kernel workspace, or whole-operator template substitution.

## 5. Verifiers

### 5.1 KIR verifier

Checks programming-model/DSL legality, not GPU profitability/provider capabilities.

### 5.2 Construction verifier

Checks KIR coverage/origin/first complete program under [construction rules](kir-to-gpu.md).

### 5.3 Shared GPU verifier

After each pass, check program space/groups/ownership; scalar/fragment shapes/parameter expressions; control/dominance/carries/terminators; accesses/validity/fills/effects/aliases/conflicts; compositional provenance/active-set/range proofs; buffer initialization or first-write/lifetimes/visibility; structured operands/results/accumulators; declared parameter domains; and absence of KIR operations, unresolved records or unknown execution fields.

### 5.4 Provider verifier

Checks target-surface legality/local extension completeness without redeciding shared structure.

### 5.5 Serializer verifier

Every operation has unique spelling or explicit unsupported before serialization; no fallback.

## 6. Legal side information

Diagnostic origins/provenance, caches, diagnostics, target facts, parameter declarations/domains and tuning artifacts may be separate. Executable coordinates/member sets/validity/narrowed ranges/identity elimination must still be materialized in current IR.

Ownership/grid, loop nesting, physical shapes/required-residency uses, pointers/indices/validity, accumulators/replay/materialization, buffer allocation/lifetimes, persistent/pipeline control and primitive operands/results cannot exist only in side records.

## 7. Evidence of pass quality

A compiler capability shows actual before/after IR, legality/preservation, changed decisions on another kernel/shape, independence from triggering names, and numerical/performance reproduction on affected programs. Attribute strings or candidate names without physical changes are not new compilation capabilities.
