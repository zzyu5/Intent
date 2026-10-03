# IntentDSL

**Write an operator algorithm once. Compile it for different execution models.**

IntentDSL is a Python kernel language and compiler for GPU, CPU, and accelerator programs. You write tensor computations, logical domains, state, and explicit multi-kernel orchestration. The compiler builds the physical execution structure and lowers it through target compilers such as Triton, cuTile, Mojo, Weft, and BANG C.

![IntentDSL algorithm, compiler, and target workflow](assets/overview.svg)

- **Programmable algorithms:** reductions, contractions, indexed access, control flow, and structured region computations compose inside kernels.
- **Reusable compilation:** shared semantic analyses and execution-family passes preserve the algorithm while forming target programs.
- **Python integration:** compile a kernel into a callable, pass PyTorch tensors, and inspect the generated source and IR.

## Install

The first public setup path is **Linux + NVIDIA + Triton**, using Python 3.10–3.12. Source builds require the LLVM/MLIR 20 C++ SDK. Python MLIR bindings are not required: the packaged C++ compiler parses, normalizes, and verifies KIR.

From this checkout, with the SDK installed:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

The installer installs CUDA PyTorch, Triton, Intent's compiler and profiles, and MCP dependencies. It does not require a project `PYTHONPATH`. The same installer provides a separate `--backend cutile` environment and explicit external-toolchain setup for Mojo, Weft, and BANG C. See the [installation guide](environment/README.md). Override `--mlir-dir` and `--llvm-dir` when the build SDK is outside `/usr/lib/llvm-20`.

If you already have a locally built Intent wheel, set `INTENT_WHEEL` to its actual filename and install it with the same backend setup:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton --wheel "$INTENT_WHEEL"
```

The wheel carries the compiler's non-system shared libraries and their third-party notices. Its runtime needs neither an LLVM/MLIR SDK nor Python MLIR bindings. It still requires a compatible Linux system, glibc/C++ runtime, and the selected backend's dependencies and driver. These are local platform wheels, not a published PyPI release or a claim of manylinux compatibility. The guide describes building a wheel and supplying notices for custom SDKs.

## Write and call a kernel

Save this as a Python file and run it in the installed environment. This is the same FP16 softmax algorithm used by the [existing production example](examples/kernels/normalization/softmax.py).

```python
import torch
import intent
import intent.language as I


@intent.fn
def maximum_number(lhs, rhs):
    return I.maximum_num(lhs, rhs)


@intent.fn
def softmax_maximum(values):
    return I.reduce(values, axis=0, identity=-I.inf, combine=maximum_number)


@intent.kernel
def softmax_kernel(
    x: I.In[I.f16, ("M", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        maximum = softmax_maximum(values)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0)
        y[row, columns] = I.cast(numerator / denominator, I.f16)


softmax = intent.compile(softmax_kernel, target=intent.TritonTarget())
x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
y = softmax.run(x)
print(y.shape, y.dtype, y.device)
```

`intent.compile` creates a callable artifact. On first use, the provider compiles or loads its native specialization and selects a configuration. Calls with the same specialization reuse it; new shapes or other specialization inputs can trigger compilation or tuning again. Keep the artifact and call it from your ordinary Python wrapper. `artifact.run(...)` allocates declared `Out` tensors; `artifact(...)` accepts all runtime arguments, including outputs, in declaration order.

The target is selected on the host. Changing it does not require a device branch in the kernel, but the selected backend must support the program's operations, types, and effects. The GPU providers are Triton and cuTile. In the cuTile environment, run `python examples/softmax.py --target cutile` to use the same definition. [Public host examples](examples/README.md) also show explicit forward/backward composition; [GPU](experiments/gpu/README.md), [CPU](experiments/cpu/README.md), and [MLU](experiments/mlu/README.md) retain the existing measured coverage.

For GPU and Mojo CPU artifacts, `artifact.as_torch_op("your_project::name")` returns an opaque PyTorch custom operator with a FakeTensor implementation derived from its declared interface. Declared `Out` tensors are fresh return values; `InOut` arguments are registered as mutations and updated in place. With no `Out`, the call returns `None`. Each `InOut` must have independent Torch storage from every other input; returned aliases are unsupported. The fake path checks the same interface without running a provider or accessing tensor data.

Authors can register their own backward for functional operators with `register_autograd`; Intent does not infer it. PyTorch does not accept that registration for mutable custom operators.

Run `python examples/softmax.py --target triton --torch-compile` for a complete `torch.compile(fullgraph=True)` call. The example first calls the same operator normally to complete provider compilation/tuning before graph capture. The operator remains opaque to PyTorch; its implementation is not fused into surrounding PyTorch operations.

## Use with an agent

The installed `intent-manual` command starts a read-only stdio MCP server. For clients that accept this JSON configuration, use the absolute path to the command in your environment:

```json
{
  "mcpServers": {
    "intent_manual": {
      "command": "/absolute/path/to/.venv/bin/intent-manual",
      "args": []
    }
  }
}
```

Start with `read(id="doc/dsl/authoring.md")`, then use `api(name="intent.compile")`, `api(name="I.matmul")`, or `search(query="region_fold")`. The manual ships with the package and needs neither a checkout nor an experiment-generated corpus. It provides declarations, language contracts, and small syntax fragments; complete algorithms live in the public examples. It does not execute code or certify numerical correctness.

For an agent that should compile a user-supplied program, explicitly add a second server using `/absolute/path/to/.venv/bin/intent-compiler-mcp`. Its `compile` tool requires an existing `program_path` and `kernel`; `target` is required for the shared and provider stages, while `stage="kir"` needs no target or device. It uses the same public compiler pipeline as the CLI below; importing the file executes ordinary module-level Python. Kernel execution and numerical checks remain separate.

## Inspect and learn

Capture and verify canonical KIR without selecting a device, or continue to a provider:

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --stage kir --json

intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 \
  --target triton --json
```

The result reports generated artifacts or the actual failure stage and diagnostic. `--materialize` additionally creates the callable; provider JIT or tuning may still be deferred until invocation. The tool does not launch the selected kernel, while module-level Python code executes normally. Use `--constexpr NAME=JSON` and `--target-option NAME=JSON` for existing public parameters. `intent doctor --target cutile --json` checks only the selected environment and target facts, without validating a kernel's results.

- `artifact.source` contains the generated provider program; `artifact.mlir` contains the target IR. `artifact.cache_directory` locates compiler inputs, outputs, and diagnostics. `intent.generate(...)` emits source and IR without creating a callable.
- `intent.CompilationStageError.stage` identifies the failed compilation stage. Its `cache_directory`, when available, points to the diagnostic artifacts.
- The compiler is discovered from an explicit `compiler=`, then `INTENT_COMPILER`, the installed package, or `PATH`. A supplied invalid path is an error.
- [Author guide](doc/dsl/authoring.md) · [Language and compiler specification](doc/index.md) · [Algorithm examples](examples/README.md) · [Existing experiments and results](experiments/README.md) · [Developer guide](CONTRIBUTING.md)
