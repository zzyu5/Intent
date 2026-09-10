from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path

import intent
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
        self.generated: dict[str, object] = {}

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
        self.generated[name] = program
        (self.directory / f"{name}.py").write_text(program.source)
        if self.keep_ir:
            (self.directory / f"{name}.mlir").write_text(program.ir)
        return materialize_triton_artifact(program.source, program.ir, definition.__name__, 0)

    def load_source(self, filename: str):
        if self.language != "triton":
            raise ValueError("Intent generation must compile Intent definitions, not load Triton sources")
        path = self.directory / filename
        if path.parent != self.directory or path.suffix != ".py":
            raise ValueError("load_source() accepts an adjacent generated Python file")
        validate_program(path, language="triton", generated_source=True)
        return materialize_triton_artifact(path.read_text(), "", path.stem, 0)


class TuningBudget:
    """Apply the same finite search policy at Triton's public autotune boundary."""

    def __init__(self, limit: int, policy):
        self.limit = limit
        self.policy = policy
        self.tuners = []
        self.precompile_failures = []

    @contextmanager
    def compilation_only(self):
        original_jit = JITFunction.run
        original_autotuner = Autotuner.run
        autotuning = 0

        def compile_kernel(kernel, *args, **kwargs):
            kwargs["warmup"] = True
            try:
                return original_jit(kernel, *args, **kwargs)
            except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as error:
                if not autotuning:
                    raise
                # Match Triton's candidate-failure policy. Normal autotuning
                # still evaluates the same bounded set and rejects these forms.
                self.precompile_failures.append({"kernel": kernel.__name__, "error": str(error)})
                return None

        def compile_tuner(tuner, *args, **kwargs):
            nonlocal autotuning
            kwargs.pop("warmup", None)
            autotuning += 1
            try:
                compiled = tuner.warmup(*args, **kwargs)
                # Autotuner.run returns one compiled kernel. Preserve that
                # interface for explicit-output artifact calls during warmup.
                for kernel in compiled:
                    if kernel is not None:
                        return kernel
                raise RuntimeError("no bounded autotune configuration compiled successfully")
            finally:
                autotuning -= 1

        JITFunction.run = compile_kernel
        Autotuner.run = compile_tuner
        try:
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
            original_prune = tuner.prune_configs

            for name in ("pre_hook", "post_hook"):
                hook = getattr(tuner, name)
                provided = kwargs.get(name)
                trusted = provided is None or isinstance(getattr(provided, "__self__", None), TuningHooks)
                if trusted:
                    def runtime_hook(*arguments, _hook=hook, **keywords):
                        with _disable_current_modes():
                            return _hook(*arguments, **keywords)
                    setattr(tuner, name, runtime_hook)

            def prune(arguments):
                legal = original_prune(arguments)
                if len(legal) <= self.limit:
                    return legal
                return [legal[index * (len(legal) - 1) // (self.limit - 1)]
                        for index in range(self.limit)]

            tuner.prune_configs = prune
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
            return [benchmark(candidate_call, warmup=5, repetitions=30, cuda_graph=True)[0]]

    def __enter__(self):
        self._original = triton.autotune
        triton.autotune = self._decorate
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        triton.autotune = self._original

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
    torch_api = {"Tensor", "dtype", "device", "empty", "empty_like", "empty_strided", "finfo", "iinfo", "is_tensor", "numel",
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
                if node.module != "torch" or any(item.name not in torch_api for item in node.names):
                    raise ValueError("Torch imports are limited to allocation, tensor metadata and dtype APIs")
            if node.module == "intent" and any(item.name in {"compile", "generate", "compile_shared_gpu"} for item in node.names):
                raise ValueError("compiler calls belong to context.compile(), not the candidate host")
        else:
            names = []
        if set(names) - allowed:
            raise ValueError(f"candidate import outside the language/host API allowlist: {names}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in torch_names:
            parent = parents.get(node)
            if not isinstance(parent, ast.Attribute) or parent.value is not node or parent.attr not in torch_api:
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


def export_seed(candidate: Path, context: ProgramContext, destination: Path) -> None:
    if not context.generated:
        raise ValueError("Intent submission did not compile any Intent kernels")
    tree = ast.parse(candidate.read_text())

    class ReplaceCompilation(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "compile":
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    raise ValueError("compile() names must be literal identifiers for source export")
                name = node.args[0].value
                if name not in context.generated:
                    raise ValueError(f"compile call {name!r} was not executed during build()")
                return ast.copy_location(ast.Call(
                    ast.Attribute(node.func.value, "load_source", ast.Load()),
                    [ast.Constant(f"{name}.py")], []), node)
            return self.generic_visit(node)

    tree = ReplaceCompilation().visit(tree)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
    pending, required = ["build"], set()
    while pending:
        name = pending.pop()
        if name in required or name not in definitions:
            continue
        required.add(name)
        pending.extend(node.id for node in ast.walk(definitions[name])
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))
    kept = {id(definitions[name]) for name in required}
    tree.body = [node for node in tree.body if id(node) in kept or isinstance(node, (ast.Import, ast.ImportFrom))]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidate.py").write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    for name, program in context.generated.items():
        (destination / f"{name}.py").write_text(program.source)
