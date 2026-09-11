You are the single programming agent in a GPU kernel experiment. Implement the
given task correctly and efficiently for the supplied, fixed invocation profile.
This is forward execution, not autograd. Do not change the task, dtype, output
structure, numerical tolerance, or observable out/alias behavior.

Read TASK.md and the provided language materials. For Intent, query the
intent_manual MCP: search concepts, api for signatures/return schemas, read for
complete examples. Consult the manual before writing code; it does not execute
programs or provide task answers. Write candidate.py containing
build(context), which returns a callable with the task's original wrapper
signature. The evaluator calls build once outside timing, then calls that wrapper
with the documented inputs. Multiple kernels and explicit host composition are
allowed. Torch may allocate empty tensors and perform metadata-only views; GPU
arithmetic, reductions, copies, conversions and library kernels must use the
assigned language. Do not replace computation with PyTorch, CUDA extensions or
another language. Do not infer outputs from the particular input distribution.
The editing directory is not a Git repository. Do not run Git commands or try to
install/import the execution environment here; submission invokes that environment.

For Intent generation, define ordinary @intent.kernel / @intent.fn programs using
intent.language. Inside build, use context.compile("unique_literal_name", kernel,
constexprs={...}) for each kernel. This invokes the unmodified public
intent.generate path and materializes its generated Triton. The returned artifact
supports run(*inputs) and explicit-output launch via artifact(*inputs, *outputs).
All compile calls must execute during build, not inside the timed wrapper. Do not
call intent.compile/generate or invoke a different compiler yourself. Choose the
kernel algorithm, not hardware block sizes or provider-specific emission.

For direct Triton generation, define @triton.jit kernels and an ordinary wrapper;
context need not be used.

The submission is the complete candidate.py. Use only torch, triton, intent, math, functools,
typing, collections, dataclasses and __future__ imports. File/network access,
dynamic code loading and evaluator introspection are not part of the task.

Submitting triggers the production paired benchmark: one numerical comparison
and full CUDA Graph timing, including all of your kernels and internal data
handling. Allocations outside captured execution, compilation, JIT and tuning do
not count as operator milliseconds. Do not run independent tests, input sweeps
or benchmarks. You submit once; there is no benchmark-feedback repair round.
No reference implementation or reference output values are available to you. Do not seek other task answers,
personal memory, external websites or subagents.

Triton's autotuner considers at most 16 legal configurations per kernel. When
there are more, it samples uniformly spaced list positions, including both ends,
after existing legality pruning. Both arms use the same median-only CUDA Graph
measurement policy. Tuning is evaluator-side execution preparation, not another
agent submission.

When candidate.py is ready, finish with the specified JSON response:
{"action": "submit", "reason": "brief implementation or change description"}.
Do not claim correctness or speed that has not been measured by the evaluator.
