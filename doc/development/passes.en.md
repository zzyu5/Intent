# Extending passes

Passes are the entry point for optimization knowledge. A pass reads current typed semantics and execution relationships, rewrites real IR and hands a complete program to later consumers. Before adding an optimization, read [passes and analyses](../compiler/passes-and-analyses.md), the relevant [GPU](../compiler/gpu-program-ir.md)/[CPU](../compiler/cpu-program-ir.md) contract and [target lowering](../compiler/target-lowering.md).

## Locate the responsible layer

| Change | Responsible location |
|---|---|
| Domain, dtype, effects and algorithm semantics | DSL / canonical KIR; discuss design changes separately |
| Def-use, coordinates, aliasing, dependence and working-set queries | The dialect's `Analysis/` |
| Ownership, traversal, blocking, reuse and materialization | Responsibility groups in GPU/CPU/DSA `Transforms/` |
| Capabilities, implementations, local forms and legality | `lib/Target/<Provider>/` |
| Source API spelling | Target serialization |
| ABI binding, toolchain invocation and tuning calls | `python/intent/runtime/<provider>/` |

GPU and CPU may reuse analysis ideas without sharing a physical topology. A pass need not run on every backend. Shared facts, independent decisions and appropriate target consumers are useful reuse.

## A complete optimization component

1. Define legality from current typed operands, regions, def-use, access relations, effects, lifetimes and capabilities, never kernel names, operation counts or string labels.
2. Let analyses provide recomputable facts. Unknown is not permission to replay, alias or assume initialization.
3. Rewrite operations, types, regions, coordinates, carries or resource lifetimes. Do not introduce a schema/plan that jointly interprets execution alongside IR.
4. Preserve logical members, dtypes, numerical permissions, ordered control, effects, ABI and the author's kernel count.
5. Declare invalidated analyses and check postconditions after a complete transformation group. Verifiers do not repair programs; serializers do not add loops, workspace or parameters.

Use existing MLIR pass declaration, registration and pipeline mechanisms. Inspect `include/Intent/Dialect/<Family>/Transforms/Passes.td`, `lib/Dialect/<Family>/Transforms/` and its `Passes.cpp`; target-local transformations live in `lib/Target/<Provider>/Transforms/`. `intent-opt --help` lists actual callable passes and options.

Changing complete empirical configurations does not require a new pass. When a provider owns collective trees, thread communication, distributed layouts or machine pipelines, supply legal input instead of rebuilding that mechanism in Intent.

## Observe existing programs

Choose affected complete programs, inspect before/after IR and generated source, verify actual calls and observe warm execution for the same algorithm and input. Do not copy algorithms or expand a test matrix. Benefits on another related program show the reuse scope. Report generation, native compilation, execution, numerical checking and performance improvement separately.

The repository [contribution guide](https://github.com/zzyu5/Intent/blob/main/CONTRIBUTING.md) explains development entry points. The [detailed compiler reference](compiler-reference.md) preserves deeper implementation material in Chinese.
