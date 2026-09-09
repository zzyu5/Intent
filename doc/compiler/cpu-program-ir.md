# CPU executable programs

## Execution family

CPU construction consumes immutable canonical KIR, before any GPU construction.
It shares canonical semantics and analyses, not GPU program topology or fragment
types. A CPU program is independently executable and verifiable; later passes
and providers cannot read KIR to recover missing execution structure.

The initial carrier uses standard MLIR `func`, `arith`, `math`, `memref`, `scf`
and structured `linalg` operations. External views become memory references;
pure tensor intermediates have explicit storage, computation and lifetime.
Structured contractions retain their indexing maps, iteration kinds and numeric
combine. CPU-specific attributes declare ABI requirements, target capabilities
and concrete parameter bindings, never an alternate executable graph.

## Access and ABI

Function arguments declare external element type, rank, static extents, dynamic
dimension identities, access direction and alias constraints. Native lowering
expands a memory-reference argument into its pointer and descriptor fields.
Offsets, strides and rank-reduced selections remain explicit memory-reference
operations until their mechanical provider lowering.

Contiguity or disjointness required by an implementation is an explicit physical
entry precondition, checked against actual runtime views. It is not an added
author promise. A provider which cannot implement an otherwise legal view or
alias combination diagnoses it before invocation; it cannot silently copy a
view, assume `noalias`, or overwrite an input snapshot while consuming it.

Internal allocations have a lexical lifetime. Forwarding an intermediate into
an output requires matching coordinates, dominance, bounds and effect ordering.
Rematerializing a producer requires purity or proof that the source memory is
unchanged; a name or diagnostic origin is not such proof.

## Transformations

CPU transformations operate on the current memory/access/value graph:

- Intermediate fusion replaces actual loads with a proven producer computation
  and removes dead stores, traversals and allocations.
- Vectorization changes loop steps, arithmetic types and memory accesses. Full
  vector iterations and a scalar or masked remainder explicitly cover the same
  elements. Alignment and contiguity are proved rather than guessed.
- Reduction realization preserves identity, accumulator dtype and the permitted
  source-order-preserving parenthesization. Lane striping followed by a sum is
  not automatically a legal ordering. Adjacent-lane trees and contiguous chunks
  provide an order-preserving vector implementation.
- Contraction blocking consumes typed indexing maps and multiply-add semantics
  to form cache tiles, register/SIMD accumulators, reuse and necessary packing.
  Packing loops, storage and ownership are present in the rewritten program.
- Task partitioning forms a finite set of independent work items and the loops
  each task owns. `scf.parallel` expresses unordered execution and a completion
  boundary. Nested logical iteration may be flattened, but dependencies and
  unique writes remain proved. Thread-pool scheduling is a runtime mechanism,
  not an algorithm decision.

Each group verifies its completed program and invalidates affected analyses.
Unsupported structure remains a diagnostic, never an emitter fallback. The
provider receives no unresolved tensor computation or unmaterialized decision.

## Parameters and specialization

Finite CPU candidate data belongs with CPU transformations. It binds vector
width, task granularity and implemented cache/microkernel tile roles. It does
not select an algorithm or a kernel-name template. Binding a candidate changes
actual vector types, loops, accesses or supported lower-compiler options.

Shared CPU construction and semantics-preserving normalization precede candidate
instantiation. Each candidate is a complete function specialization in the
current module, with the same external ABI. The runtime compiles and measures
the declared candidates and keeps the winner outside IR. Compiled-artifact reuse
and winner reuse are distinct; their identities include source specialization,
view facts affecting legality, selected hardware and relevant compiler options.

`intent.compile(..., tuning_config=path)` uses a finite JSON override with the
`cpu` namespace. `vector` rows contain `[vector_width, task_grain]`;
`contraction` rows contain `[vector_width, task_grain, tile_m, tile_n, tile_k,
micro_m]`. A supplied family replaces its complete default rows. Widths must be
powers of two supported by the selected CPU; contraction tiles must satisfy
their vector/microtile divisibility and task-local storage constraints. These
rows are parameter data, not executable policy or kernel-name overrides.

## Mojo provider

Mojo legalization checks supported operations, types, accesses and ABI forms.
Serialization mechanically spells the existing loops, vector operations,
allocations and task regions. It cannot choose blocking, introduce packing,
recover missing masks or decide an arithmetic algorithm. The external Mojo/LLVM
compiler owns machine lowering, instruction selection and register allocation.

The native entry uses a C ABI and synchronously completes all tasks from one
kernel invocation. A runtime library may execute an already declared parallel
region; it cannot provide a whole-operator implementation in place of the CPU
program. Runtime initialization and library loading precede ordinary invocation.

Mojo global floating-point contraction is disabled. Only explicit physical
multiply-add operations originating in permitted Intent semantics may fuse;
plain arithmetic does not acquire global fast-math or FTZ. Hardware CPU/features
are explicit compile inputs rather than an implicit promise of portability.

Provider legalization materializes floating-environment enter/restore calls at
the native entry and each parallel task boundary. On x86, the provider runtime
preserves the caller/worker MXCSR while disabling FTZ/DAZ and selecting RNE for
the task. This also covers runtime libraries that enable denormal flushing
around a callback; ordinary Intent arithmetic does not inherit that setting.

Native benchmarking repeatedly calls the compiled entry in native code using a
monotonic clock. Per-invocation packing, dispatch and synchronization remain timed;
compilation, tuning, loading and external output allocation are separate. CPU
timing is neither CUDA Graph timing nor Python dispatch timing.
