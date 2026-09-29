Implement the supplied GPU task correctly and efficiently. Write one complete
candidate.py defining build(context), which returns the task's original callable.
The evaluator calls build once before timing, then invokes the returned callable.
The task is included in the request; TASK.md contains an identical copy.

The disclosed profile fixes shapes, dtypes and Python options. Specialize the
implementation to those values and omit unused option branches. Tensor contents
remain runtime inputs. Preserve the task's formula, callable defaults, result
structure, numerical tolerance and observable out/alias behavior. Preserve any
explicitly required intermediate dtypes or stages; otherwise choose an algorithm
satisfying the result and accuracy contract. This is forward execution, not autograd.

Choose the algorithm and its kernel boundaries. Multiple kernels and explicit host
composition are allowed. Count all stages, intermediate traffic and sequential
dependencies when considering performance. Torch may allocate empty tensors and
perform metadata-only views. GPU arithmetic, reductions, copies, conversions and
library kernels must use the assigned language. Only a task requiring a constant
CPU zero tensor may use torch.zeros(device="cpu"). Do not infer output values from
the particular input distribution.

<!-- intent -->
Use these imports for Intent:
```python
import torch
import intent
import intent.language as I
```

Read materials/api.txt for available names and signatures, then use
intent_manual.read(id="doc/dsl/authoring.md") for language rules. Query
intent_manual.api(name=...) for the operations you need; it includes linked type,
shape and semantic rules. Use search(kind="concept") or read for further detail.
The available manual tools are search, api and read. Use the returned document or
rule IDs; do not guess resource URIs. The manual contains public language rules,
not implementations or task answers. It cannot execute a candidate.

Intent expresses Triton-style algorithms over logical domains. The author chooses
algorithmic grouping, partial results and kernel stages; the compiler chooses
physical tiles, threads, layouts and pipelines within those kernels. Tensor free
axes already express independent results. Preserve algorithmic stage interfaces,
but do not reproduce physical BLOCK_M/BLOCK_N values as logical partitions.
An ordinary loop is ordered; I.parallel declares independent iterations, and
reduce/contract/scan express their documented collective semantics. A kernel is
one launch; a full-domain collective does not implicitly become several kernels.

The following syntax distinctions apply throughout the language:
- Shape symbols such as "N" in annotations are names, not runtime variables.
  Obtain extents from x.shape. Reusing a symbol declares equal extents; different
  symbols do not become equal just because one profile has the same sizes.
- External input view parameters used as value expressions are read over their
  full logical shape; explicit indexing selects a region. I.buffer is mutable
  storage and must be read explicitly before numeric use. Tensor values are
  immutable; indexed writes require Out, InOut or I.buffer storage.
- I.domain describes members; I.indices(domain) produces coordinates. A for-loop
  variable over a domain is already scalar I.index. I.parallel is used by for,
  cannot carry shared SSA state, and is not a tensor to pass to I.indices.
- Broadcasting aligns trailing axes; add explicit size-one axes where needed.
  Domain indices introduce independent axes; tensor indices share a broadcasted
  index shape. Derive result axes before assigning or combining values.
- Reduction removes the reduced axes. Arg-reduce indices are local positions in
  the value being reduced. Read the API's return schema before using the result.
- Loop-carried values retain dtype, rank and logical shape. Initialize the actual
  state schema. A value used after a runtime if must be defined on every branch.
- Mixed typed operands require the documented casts; I.index and I.i64 are
  distinct. Python math/Torch calls are host-only. Use documented intrinsics in
  kernels and helpers; do not guess operation names from other libraries.
- Use I.gather's validity and fill for possibly invalid reads. Selecting a value
  after an invalid memory read cannot make that read valid.

These are syntax fragments for an existing rank-2 view x, not a kernel algorithm:
```python
rows = I.domain(0, x.shape[0])
row_indices = I.indices(rows)
values = x[:, :]
for row in I.parallel(rows):
    ...  # row is already a scalar coordinate
```

<!-- host-interface -->
Define device functions with @intent.kernel and helpers with @intent.fn. Kernel
outputs are explicit I.Out/I.InOut parameters written by the kernel. Returning an
I.buffer or adding a Python return annotation does not create a host output.

Use the unannotated signature def build(context):. The evaluator supplies context;
it is not a type exported by intent. Inside build, call
context.compile("unique_literal_name", kernel_definition, constexprs={...}).
Bind every declared I.Constexpr parameter without a default in constexprs, using
its exact parameter name. The annotation is I.Constexpr[value_type], for example
I.Constexpr[bool] or I.Constexpr[str], not bare I.Constexpr. View shape symbols do
not declare constexprs. Runtime
launch arguments cannot supply constexprs. All compile calls execute inside build,
outside the returned callable; do not call intent.compile/generate yourself.

The returned artifact supports two launch forms:
- artifact(*runtime_arguments): supply every runtime parameter in declaration
  order, including Out parameters at their declared positions.
- artifact.run(*runtime_arguments_without_Out): omit only Out parameters; they
  are allocated and returned. Preserve the order of all other runtime parameters.
Both forms omit constexpr parameters. Match actual tensor ranks and dtypes to the
kernel interface. Read extents from view shapes or pass runtime scalars; build
receives no runtime tensors. Host control flow may call compiled artifacts more
than once. Cross-kernel state uses host-allocated tensors passed as views, not
kernel-local I.buffer values.

This shows declaration and launch syntax only. The device computation is omitted;
choose all schemas, allocations and the callable signature for your task:
```python
@intent.kernel
def kernel_definition(x: I.In[I.f32, ("N",)], y: I.Out[I.f32, ("N",)]):
    ...  # implement the device computation and write y

def build(context):
    artifact = context.compile("kernel_name", kernel_definition)
    def wrapper(x):
        y = torch.empty_like(x)
        artifact(x, y)
        return y
    return wrapper
```
<!-- /host-interface -->
Do not import or call Triton in the Intent submission.
<!-- /intent -->

<!-- triton -->
Define @triton.jit kernels and an ordinary wrapper; context need not be used.
Read materials/__init__.py for triton.language exports. core.py, standard.py,
math.py and extra/cuda/libdevice.py contain their public definitions. Use the
namespace that actually exports an operation.
<!-- /triton -->

Review candidate.py before submission: exact kernel names in compile calls,
constexpr bindings, launch argument order, callable defaults, formulas, shapes,
dtypes and returned structure. No separate plan or review file is needed.

Use only torch, triton, intent, math, functools, typing, collections, dataclasses
and __future__ imports. The editing directory is not a Git repository. Do not
install or import the execution environment here. File/network access, dynamic
code loading and evaluator introspection are not candidate functionality.
Do not seek reference implementations, output values, other task answers,
personal memory, external websites or subagents.

Submit once. The evaluator then performs the production numerical comparison and
complete-operator timing, including all kernels and internal data handling, using
the timing policy in TASK.md. There is no benchmark-feedback repair round. Do not
run independent tests, sweeps or benchmarks. Compilation, JIT, tuning and allocation
outside execution are excluded from operator milliseconds. Evaluator-side tuning
uses native Triton autotuning after legality pruning and the same median-only
CUDA Graph policy for both arms.

Write candidate.py to disk before finishing; prose or code in the final response
does not create that file. Then finish with:
{"action": "submit", "reason": "brief implementation description"}
Do not claim unmeasured correctness or speed.
