from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import intent
from intent.runtime.artifact import CompiledArtifact
from intent.runtime.source import materialize_python_source
from intent.runtime.triton import materialize_triton_artifact, TuningHooks
from intent.targets import TritonTarget
import triton
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources, PTXASError
from triton.runtime.jit import JITFunction
from torch.utils._python_dispatch import _disable_current_modes

from repro.common.support import benchmark
from repro.v2.loading import load_module


class ProgramContext:
    def __init__(self, compiler: Path, directory: Path, *, language: str, keep_ir: bool = False):
        self.compiler = compiler
        self.directory = directory
        self.language = language
        self.keep_ir = keep_ir
        self.generated: dict[str, CompiledArtifact] = {}

    def compile(self, name: str, definition, *, constexprs=None):
        if self.language != "intent":
            raise ValueError("only Intent generation may invoke the Intent compiler")
        if not name.isidentifier() or name in self.generated:
            raise ValueError("each compile() needs a distinct literal identifier")
        try:
            program = intent.generate(definition, target=TritonTarget(), compiler=self.compiler,
                                      constexprs=constexprs)
        except intent.CompilationStageError as error:
            if error.stage in {"provider_lowering", "provider_program_verification", "serialization"}:
                shared = intent.compile_shared_gpu(
                    definition, target=TritonTarget(), compiler=self.compiler, constexprs=constexprs)
                (self.directory / f"{name}.shared.mlir").write_text(shared)
            raise
        (self.directory / f"{name}.py").write_text(program.source)
        if self.keep_ir:
            (self.directory / f"{name}.mlir").write_text(program.ir)
        artifact = materialize_triton_artifact(program.source, program.ir, definition.__name__, 0)
        self.generated[name] = artifact
        return artifact

    def save_backend_ir(self, executed):
        if not self.keep_ir:
            return
        for name, artifact in self.generated.items():
            for entry in artifact._namespace.values():
                if isinstance(entry, Autotuner) and entry in executed:
                    artifact.backend_ir = artifact._backend_ir_collector(executed[entry])
            for kind, text in artifact.backend_ir.items():
                (self.directory / f"{name}.backend.{kind}").write_text(text)

    def load_source(self, filename: str):
        if self.language != "triton":
            raise ValueError("Intent generation must compile Intent definitions, not load Triton sources")
        path = self.directory / filename
        if path.parent != self.directory or path.suffix != ".py":
            raise ValueError("load_source() accepts an adjacent generated Python file")
        validate_program(path, language="triton", generated_source=True)
        # Agent-edited Triton may launch multiple kernels without returning
        # compiler debug metadata. Only its operator outputs are evaluated.
        return materialize_python_source(
            target_name="triton", source=path.read_text(), module_text="",
            entry_name=path.stem, device=0, backend_ir_collector=None)


class TuningBudget:
    """Apply the shared measurement and preparation policy to native autotuning."""

    def __init__(self, policy):
        self.policy = policy
        self.tuners = []
        self.executed = {}
        self.precompile_failures = []

    @contextmanager
    def compilation_only(self):
        original_jit = JITFunction.run
        original_autotuner = Autotuner.run
        autotuning = 0
        future_names = {}
        compilation_errors = []
        policy = self.policy

        class CompileExecutor(ThreadPoolExecutor):
            def submit(self, function, /, *args, **kwargs):
                def compile_with_policy():
                    with policy():
                        return function(*args, **kwargs)
                return super().submit(compile_with_policy)

        def compile_kernel(kernel, *args, **kwargs):
            kwargs["warmup"] = True
            try:
                compiled = original_jit(kernel, *args, **kwargs)
                if isinstance(compiled, triton.FutureKernel):
                    if not autotuning:
                        return compiled.result()
                    future_names[compiled] = kernel.__name__
                    return compiled
                # Triton initializes device handles on the first real launch,
                # inside the evaluator's GPU lock.
                return compiled
            except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as error:
                if not autotuning:
                    raise
                # Match Triton's candidate-failure policy. Normal autotuning
                # still evaluates the same legal configurations and rejects these forms.
                self.precompile_failures.append({"kernel": kernel.__name__, "error": str(error)})
                compilation_errors.append(error)
                return None

        def compile_tuner(tuner, *args, **kwargs):
            nonlocal autotuning
            kwargs.pop("warmup", None)
            autotuning += 1
            try:
                error_count = len(compilation_errors)
                tuner.nargs = dict(zip(tuner.arg_names, args))
                compiled = []
                for configuration in tuner.prune_configs(kwargs):
                    meta = configuration.all_kwargs()
                    # Descriptor block shapes are part of the compiled argument
                    # type. Triton's warmup skips this normal run-time hook.
                    if configuration.pre_hook is not None:
                        configuration.pre_hook({**tuner.nargs, **kwargs, **meta})
                    compiled.append(tuner.fn.warmup(*args, **kwargs, **meta))
                resolved = []
                for kernel in compiled:
                    if isinstance(kernel, triton.FutureKernel):
                        try:
                            resolved.append(kernel.result())
                        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as error:
                            self.precompile_failures.append({"kernel": future_names[kernel], "error": str(error)})
                            compilation_errors.append(error)
                    elif kernel is not None:
                        resolved.append(kernel)
                # Return a resolved kernel, preserving the explicit-output
                # artifact interface. All candidate failures were inspected.
                if resolved:
                    return resolved[0]
                cause = compilation_errors[-1] if len(compilation_errors) > error_count else None
                raise RuntimeError("no legal autotune configuration is executable on the current device") from cause
            finally:
                autotuning -= 1

        JITFunction.run = compile_kernel
        Autotuner.run = compile_tuner
        try:
            # Expected per-configuration failures are handled above; unknown
            # failures still propagate from result() before this context exits.
            with CompileExecutor(max_workers=2) as executor:
                with triton.AsyncCompileMode(executor, ignore_errors=True):
                    yield
        finally:
            JITFunction.run = original_jit
            Autotuner.run = original_autotuner

    def _decorate(self, *args, **kwargs):
        kwargs["do_bench"] = self._measure
        kwargs["cache_results"] = False
        decorate = self._original(*args, **kwargs)

        def register(function):
            tuner = decorate(function)

            for name in ("pre_hook", "post_hook"):
                hook = getattr(tuner, name)
                provided = kwargs.get(name)
                trusted = provided is None or isinstance(getattr(provided, "__self__", None), TuningHooks)
                if trusted:
                    def runtime_hook(*arguments, _hook=hook, **keywords):
                        with _disable_current_modes():
                            return _hook(*arguments, **keywords)
                    setattr(tuner, name, runtime_hook)

            self.tuners.append(tuner)
            return tuner
        return register

    def _measure(self, function, *, quantiles):
        # Triton ranks lexicographically; use only the measured median. Its
        # failure sentinel starts with infinity and remains comparable.
        with _disable_current_modes():
            def candidate_call():
                with self.policy():
                    return function()
            return [benchmark(candidate_call, warmup=5, repetitions=30, cuda_graph=True, time_budget_ms=30)[0]]

    def __enter__(self):
        self._original = triton.autotune
        self._original_run = Autotuner.run

        def run(tuner, *args, **kwargs):
            compiled = self._original_run(tuner, *args, **kwargs)
            self.executed[tuner] = compiled
            return compiled

        triton.autotune = self._decorate
        Autotuner.run = run
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        triton.autotune = self._original
        Autotuner.run = self._original_run

    def records(self) -> list[dict]:
        return [{"kernel": tuner.base_fn.__name__, "declared": len(tuner.configs),
                 "measured": len(getattr(tuner, "configs_timings", {})),
                 "seconds": getattr(tuner, "bench_time", 0.0),
                 "winner": str(tuner.best_config) if hasattr(tuner, "best_config") else None}
                for tuner in self.tuners]


def validate_program(path: Path, *, language: str, generated_source: bool = False) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed = {"torch", "triton", "intent", "math", "functools", "typing", "collections", "dataclasses", "__future__"}
    if language == "intent":
        allowed.remove("triton")
    intent_names = set()
    torch_names = set()
    torch_result_types = {"max", "min"}
    torch_api = {"Tensor", "Size", "dtype", "device", "empty", "empty_like", "empty_strided", "zeros", "finfo", "iinfo", "is_tensor", "numel", "broadcast_to",
                 "bool", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "long", "short",
                 "float", "float16", "float32", "float64", "bfloat16", "half", "double", "complex64", "complex128",
                 "strided", "contiguous_format", "preserve_format", "channels_last"}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in tree.body:
        if isinstance(node, ast.Import):
            intent_names.update(item.asname or item.name for item in node.names if item.name == "intent")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            torch_names.update(item.asname or item.name for item in node.names if item.name == "torch")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [item.name.split(".")[0] for item in node.names]
            if any(item.name.startswith("torch.") for item in node.names):
                raise ValueError("Torch imports are limited to allocation, tensor metadata and dtype APIs")
        elif isinstance(node, ast.ImportFrom):
            names = [node.module.split(".")[0]] if node.module else []
            if node.level:
                raise ValueError("candidate imports must not reach other programs or the evaluator")
            runtime_hooks = generated_source and node.module == "intent.runtime.triton" and all(item.name == "TuningHooks" for item in node.names)
            if node.module is not None and node.module.startswith("intent.") and node.module != "intent.language" and not runtime_hooks:
                raise ValueError("candidate author code may import only the public Intent language API")
            if node.module is not None and node.module.split(".")[0] == "torch":
                result_types = node.module == "torch.return_types" and all(
                    item.name in torch_result_types for item in node.names)
                if not result_types and (node.module != "torch" or any(item.name not in torch_api for item in node.names)):
                    raise ValueError("Torch imports are limited to allocation, tensor metadata and dtype APIs")
            if node.module == "intent" and any(item.name in {"compile", "generate", "compile_shared_gpu"} for item in node.names):
                raise ValueError("compiler calls belong to context.compile(), not the candidate host")
        else:
            names = []
        if set(names) - allowed:
            raise ValueError(f"candidate import outside the language/host API allowlist: {names}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in torch_names:
            parent = parents.get(node)
            container = parents.get(parent)
            result_type = (isinstance(parent, ast.Attribute) and parent.attr == "return_types"
                           and isinstance(container, ast.Attribute) and container.value is parent
                           and container.attr in torch_result_types)
            if not result_type and (not isinstance(parent, ast.Attribute) or parent.value is not node or parent.attr not in torch_api):
                raise ValueError("Torch is available only through allocation, tensor metadata and dtype APIs")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in {
            "open", "from_file", "fromfile", "tofile", "read_text", "read_bytes", "write_text", "write_bytes",
            "unlink", "rename", "mkdir", "rmdir", "iterdir", "glob"
        }):
            raise ValueError(f"candidate may not access files or runtime internals through {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "open", "exec", "eval", "compile", "__import__", "globals", "locals", "breakpoint", "getattr", "setattr", "delattr", "vars"
        }:
            raise ValueError(f"candidate may not use {node.func.id}()")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in intent_names and node.func.attr in {"compile", "generate", "compile_shared_gpu"}:
                raise ValueError("compiler calls belong to context.compile(), not the candidate host")


def load_program(path: Path, *, language: str):
    validate_program(path, language=language)
    return load_module(path, "intent_agent_candidate")
