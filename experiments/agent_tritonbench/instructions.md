You are the single programming agent in a GPU kernel experiment. Implement the
given task correctly and efficiently for the supplied, fixed invocation profile.
This is forward execution, not autograd. Do not change the task, dtype, output
structure, numerical tolerance, or observable out/alias behavior.
Choose the algorithm for the complete invocation before writing the kernels.
For each stage, settle its logical inputs and outputs, independent work and
required sequential dependencies. Assess the sequential critical path as well as
total computation, intermediate memory traffic and the cost of all kernel calls.
A dependency between stages requires their execution order to be preserved; it
does not require putting those stages in the same kernel. Internal kernel
interfaces and logical grouping are your choices unless TASK.md constrains them.

Preserve computation stages and intermediate dtypes explicitly required by TASK.md.
Algebraic equivalence alone does not preserve a stated floating-point contract.
A mathematical formula alone does not require separately rounded intermediates.
Unless TASK.md explicitly fixes intermediate dtypes, evaluation order or observable
stages, choose an organization satisfying its logical result and accuracy contract.
The language's numerical rules govern the program you write and its compilation,
not the reference library's internal implementation.
Choose among documented exact and explicit approximate operations according to
TASK.md's accuracy and input-domain requirements. Approximate modes must respect
their documented error bounds, range and special-value rules.

Read TASK.md and the provided language materials.
<!-- intent -->
For Intent, first call
intent_manual.read(id="doc/dsl/authoring.md") for the language and host interface
rules, then read(id="doc/programming-model/kernel-and-host.md") for kernel,
specialization and multi-kernel composition semantics. Use the public contracts
to express the independent logical work and dependencies of your algorithm.
Use api for declarations and signatures, then read(id=<returned rule ID>)
for return shapes, dtypes and semantics; api does not infer your program's result
schema. Consult the manual before writing code; it does not execute programs or
provide task answers.
For syntax or semantics questions, use search(kind="concept") rather than searching
implementation diagnostics. Copy returned IDs; do not guess section titles or
resource URIs. Read accepts document/rule IDs and exact public API names; section
accepts a published heading title or line such as L134. The MCP tools are search,
api and read. The evaluator's context.compile interface is defined below, not an
Intent intrinsic to discover in the language API.
<!-- /intent -->
Write candidate.py containing
build(context), which returns a callable with the task's original wrapper
signature. The evaluator calls build once outside timing, then calls that wrapper
with the documented inputs. Multiple kernels and explicit host composition are
allowed. Torch may allocate empty tensors and perform metadata-only views; GPU
arithmetic, reductions, copies, conversions and library kernels must use the
assigned language. Do not replace computation with PyTorch, CUDA extensions or
another language. Do not infer outputs from the particular input distribution.
When the task contract requires a constant CPU zero tensor, its host construction
may use torch.zeros with device="cpu"; this does not permit Torch GPU operations.
The editing directory is not a Git repository. Do not run Git commands or try to
install/import the execution environment here; submission invokes that environment.

Before submitting, review the complete source against TASK.md and the queried
rules. Check the returned callable and its defaults, the result tree, runtime
argument order, and the dtype and shape of loop state and helper results.
Check every branch against the task's formula, coefficients and explicitly
required intermediate dtypes or stages.
Check that build(context) itself returns the host callable on the supplied
profile, rather than only defining or returning from that inner callable.
Derive shape-dependent counts from the actual input or intermediate dimensions.
Compare the actual source with the algorithm you chose: its kernel calls,
independent logical work, intermediate values and sequential dependencies must
still be present after translation. Check each ordinary loop against the
dependency it expresses, including loops introduced while writing the code.
Do this review on candidate.py; no separate planning or review file is needed.

<!-- intent -->
In that source review, match every constexpr binding to an explicitly declared
I.Constexpr parameter; a symbol in a view shape does not declare one. Write view
shape symbols as strings, and bind local extent variables from input.shape before
using them. Same symbols denote equal extents; different symbols do not become
equal merely because the supplied profile has the same sizes. Derive each
initializer, assignment and loop carry's source and destination schemas from the
actual tensors and documented API results, and check their compatibility.
For each indexed expression, derive its result axes before assigning it: domain
indices introduce independent axes, while tensor indices share a broadcasted
index shape. Mixing the two does not implicitly pair their elements; use explicit
size-one axes where broadcasting requires them.
Tensor values are immutable; indexed writes require a writable view or I.buffer.
Arg-reduce indices are positions in the value being reduced, including when that
value was read from a subregion. For possibly invalid reads, express their validity
and fill with I.gather; selecting a value after an invalid read cannot undo the read.
Keep kernel definitions, compiled artifacts and host wrappers under distinct names.
At each launch, match the actual tensor rank and argument order to that kernel's
declared interface; a host view change must be passed to the call that needs it.

Intent expresses Triton-style kernel algorithms over logical domains. Retain the
algorithm's independent work, logical groups, partial results, kernel stages and
host orchestration; leave physical tiles, threads, layouts and pipelines to the
compiler. Follow the organization guidance at the start of the authoring manual.
A scalar external result does not imply one full-domain reduction. Do not erase
the chosen work decomposition merely because Intent can express the mathematical
result in one expression. Intent's type, numerical and effect rules still apply.

The source constructs have distinct meanings:
- Domains, subregions and index relations describe logical members and values.
  I.parallel declares independent, unordered iterations; it does not specify a
  GPU program or thread-block count.
- Ordinary for/while loops declare ordered execution and loop-carried state.
  When the algorithm permits their documented numerical contracts, express sums,
  inner products and prefixes with reduction, contraction and scan operations.
  Introducing a scalar accumulator loop imposes order; it does not inherit the
  reassociation allowed by those structured operations.
  An outer I.parallel does not make an inner ordinary loop parallel. Preserve
  vector expressions as logical tensor operations; their values can be consumed
  by independent logical points when the effects are disjoint.
- I.reduce and I.arg_reduce operate on logical tensor axes in the current kernel.
  Reducing every axis yields scalar result(s); it does not call a library
  reduction or create an implicit cross-kernel reduction tree. I.parallel around
  a producer or consumer does not partition a full-domain collective for you.
- Each @intent.kernel produces one GPU launch. Separate expressions or named
  intermediates inside it do not create additional launches or independent work.
  Collectives have no hidden communication between independent GPU programs.
  A prefix over an entire axis remains global when its consumer uses I.parallel.

Free axes of tensor expressions and structured operations already express
independent result coordinates. Do not introduce fixed logical subregions solely
to reproduce physical BLOCK_M/BLOCK_N values or every use of program_id. Use the
complete logical tensor expression when those free axes describe the independent
work. Preserve partitions that define the algorithm's partial results, member
selection or stage interfaces; reduction axes do not become free axes when
physical tile sizes are omitted. Such algorithmic group counts, domain boundaries
and intermediate shapes may be fixed or constexpr and need not appear in TASK.md
or the external signature. The compiler chooses physical tiles, program mapping,
thread layouts and provider configuration; it does not reconstruct missing stages.

When the chosen Triton algorithm uses multiple kernels, retain those kernels and
their host calls in Intent. Allocate cross-kernel tensors in the host callable and pass
them as views: Out for outputs, In for read-only inputs, and InOut for reading and
updating existing contents. Kernel-local I.buffer state cannot cross kernels.
Host Python control flow may repeatedly call an already compiled artifact with
changing runtime scalars; each call is a launch and does not require recompilation.

<!-- host-interface -->
For Intent generation, define ordinary @intent.kernel / @intent.fn programs using
intent.language. Inside build, use context.compile("unique_literal_name", kernel,
constexprs={...}) for each kernel. This invokes the unmodified public
intent.generate path and materializes its generated Triton. The returned artifact
supports explicit-output launch with runtime arguments in kernel declaration
order: artifact(*declared_runtime_arguments). Keep Out arguments in their declared
positions, even when scalar inputs follow them. artifact.run(...) allocates and
returns Out tensors; omit only Out arguments and preserve the order of all other
runtime arguments. Constexpr arguments are bound at compilation and omitted here.
All compile calls must execute during build, not inside the timed wrapper. Do not
call intent.compile/generate or invoke a different compiler yourself. Choose the
algorithm's logical partitions using the rules above.
<!-- /host-interface -->
Python math and Torch calls are host-only; inside Intent kernels and helpers,
use documented DSL intrinsics and syntax shorthands. Reduction removes its axes,
and broadcasting aligns trailing axes; add explicit size-one axes when needed.
A name read after a runtime if must already be defined or be assigned in every
normally continuing branch; implications between conditions do not define it.
<!-- /intent -->

<!-- triton -->
For direct Triton generation, define @triton.jit kernels and an ordinary wrapper;
context need not be used.
Read materials/__init__.py for names exported by triton.language; core.py,
standard.py, math.py and extra/cuda/libdevice.py provide their public definitions.
Use the namespace that actually exports an operation.
<!-- /triton -->

The submission is the complete candidate.py. Use only torch, triton, intent, math, functools,
typing, collections, dataclasses and __future__ imports. File/network access,
dynamic code loading and evaluator introspection are not part of the task.

Submitting triggers the production paired benchmark: one numerical comparison
and complete-operator timing using the CUDA Graph or CUDA event path declared in
TASK.md, including all of your kernels and internal data handling. Compilation,
JIT, tuning and allocations outside timed execution do not count as operator
milliseconds. Do not run independent tests, input sweeps
or benchmarks. You submit once; there is no benchmark-feedback repair round.
No reference implementation or reference output values are available to you. Do not seek other task answers,
personal memory, external websites or subagents.

Triton's native autotuner measures the configurations remaining after existing
legality pruning. Both arms use the same median-only CUDA Graph
measurement policy. Tuning is evaluator-side execution preparation, not another
agent submission.

Write the complete candidate.py to disk before finishing. Planning prose or code
in the final response does not create the submission file. Once it is written,
finish with the specified JSON response:
{"action": "submit", "reason": "brief implementation or change description"}.
Do not claim correctness or speed that has not been measured by the evaluator.
