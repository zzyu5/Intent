# Kernel and host boundaries

## 1. Kernel definition

An `@intent.kernel` defines one independent logical computation. A compilation call selects the target outside source. For a specialization, interface and target, compilation produces one target artifact; host/runtime determines when and how to launch it.

Kernels receive `In`, `Out`, `InOut` views and runtime scalars, do not return device tensors through Python return, may contain hardware-independent control/state/structured operations/effects, query no target/resources/capabilities/configurations, and allocate or launch no hidden second kernel.

The compiler must not split a kernel into multiple host-visible invocations or create invisible cross-kernel workspace and synchronization.

## 2. Multi-kernel algorithms

An author may implement one external callable contract with one or multiple kernels while preserving its required numerics, effects and interface. Authors may design internal interfaces, logical groupings and intermediate shapes without those appearing in the external task signature. Views and scalars actually passed inside the wrapper are observable interfaces too.

For multiple kernels, ordinary Python host code explicitly selects specialization/target, allocates outputs and intermediates, establishes call ordering/concurrency, and manages cross-kernel state and lifetimes. Intent compiles kernels individually. The Python wrapper owns orchestration. Compilation, artifact creation and launch are distinct stages, not one KIR operation.

## 3. Helpers

`@intent.fn` declares a hardware-independent typed helper inside a kernel. It can return scalars, tensors, tuples or records and have the same read/write effects allowed at its call site. Runtime captures become explicit parameters; only immutable constexprs may be captured. Recursion, host dispatch, launches, target queries and provider selection are forbidden.

Inlining or generating a target-local device function is an implementation choice. It does not change call semantics or index relations. Coordinates retain source identity, axis maps and SSA provenance through helper parameters/results regardless of inlining.

## 4. Runtime and constexpr

Runtime parameters vary per invocation and participate in computations and runtime control. `Constexpr[T]` is bound at specialization and chooses hardware-independent algorithm variants: causal mode, an activation, algorithm group counts, sparse formats or numerical modes.

Constexprs must not query/select providers, device models, ISA/capabilities, TMA/MMA/layout, warps/stages/tiles/storage or autotune configurations. The selected hardware-independent branch enters canonical KIR; target-specific selection occurs afterward.

## 5. Control within a kernel

Kernels permit runtime scalar-Boolean `if`, algorithmic constexpr `if`, ordered `for/while` with carries/`break`/`continue`, and unordered `parallel`. Tensor predicates use value-level `select`, not structured `if`. There is no independent stop operation: conditions, breaks and domain/subregion endpoints express termination. These constructs remain in one kernel, not multi-kernel dispatch.

## 6. Interfaces and local resources

External views expose element dtype, logical shape, runtime element strides, access direction, bounds and needed alias relations. Authors do not calculate target pointer arithmetic or make alignment, contiguity, TMA eligibility or vector widths algorithm parameters.

A kernel-local logical buffer is algorithm state. Authors define shape, dtype, initialization and read/write relations; the compiler chooses SSA, registers, stack, shared/local memory or other storage. Cross-kernel intermediates are allocated explicitly by the host.

Concurrent access to an external allocation across kernels or host/device is governed by wrapper/runtime invocation and synchronization semantics. A kernel's physical atomic scope does not implicitly expand its participants.
