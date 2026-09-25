from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import linecache
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import intent
from intent.runtime.artifact import CompiledArtifact
from intent.runtime.source import materialize_python_source
from intent.runtime.cutile import materialize_cutile_artifact
from intent.runtime.triton import materialize_triton_artifact, TuningHooks
from intent.targets import CuTileTarget, TritonTarget
import triton
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources, PTXASError
from triton.runtime.jit import JITFunction
from torch.utils._python_dispatch import _disable_current_modes

from experiments._common.support import benchmark
from experiments._common.loading import load_module
from experiments._common.measurement import CUTILE_TUNING_LAUNCH_TIMEOUT_SECONDS


def _initialize_cutile_compiler(sources):
    global _cutile_compile_namespaces
    _cutile_compile_namespaces = {}
    for module_name, filename, source in sources:
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        namespace = {"__name__": module_name}
        exec(compile(source, filename, "exec", dont_inherit=True), namespace)
        _cutile_compile_namespaces[module_name] = namespace


def _compile_cutile_configuration(module_name, function_name, parameters, static_arrays,
                                  symbol, options, architecture, timeout):
    import cuda.tile as ct
    from cuda.tile._compile import compile_tile
    from cuda.tile.compilation import CallingConvention, KernelSignature

    kernel = _cutile_compile_namespaces[module_name][function_name]
    convention = (CallingConvention.cutile_python_v2() if static_arrays
                  else CallingConvention.cutile_python_v1())
    signature = KernelSignature(parameters, convention, symbol)
    try:
        with ct.compiler_timeout(timeout):
            compiled = compile_tile(kernel._annotated_function, (signature,),
                                    architecture, options)
    except ct.TileError as error:
        return None, f"{type(error).__name__}: {error}"
    return (compiled.cubin, compiled.kernel_signatures[0].symbol, None, []), None


class ProgramContext:
    def __init__(self, compiler: Path, directory: Path, *, language: str,
                 target: str = "triton", tuning_config: Path | None = None):
        self.compiler = compiler
        self.directory = directory
        self.language = language
        self.target = {"triton": TritonTarget, "cutile": CuTileTarget}[target]()
        self.target_name = target
        self.tuning_config = tuning_config
        self.generated: dict[str, CompiledArtifact] = {}
        self.tuning: list[dict] = []
        self.precompile_failures: list[dict] = []
        self.native_compile_reuses = 0
        self._native_compilations = []

    def compile(self, name: str, definition, *, constexprs=None):
        if self.language != "intent":
            raise ValueError("only Intent generation may invoke the Intent compiler")
        if not name.isidentifier() or name in self.generated:
            raise ValueError("each compile() needs a distinct literal identifier")
        program = intent.generate(definition, target=self.target, compiler=self.compiler,
                                  constexprs=constexprs, tuning_config=self.tuning_config)
        materialize = {"triton": materialize_triton_artifact, "cutile": materialize_cutile_artifact}[self.target_name]
        try:
            artifact = materialize(program.source, program.ir, definition.__name__, 0)
        except Exception as error:
            raise intent.CompilationStageError("generated_source_materialization", str(error),
                                               cache_directory=program.cache_directory) from error
        artifact.cache_directory = program.cache_directory
        if self.target_name == "cutile":
            search = artifact._namespace["exhaustive_search"]

            def record_search(configs, *args, **kwargs):
                kwargs.setdefault("single_run_timeout_sec",
                                  CUTILE_TUNING_LAUNCH_TIMEOUT_SECONDS)
                result = search(configs, *args, **kwargs)
                self.tuning.append({"kernel": name, "declared": len(configs),
                                    "measured": len(result.successes),
                                    "failures": [(str(cfg), kind.__name__, message)
                                                 for cfg, kind, message in result.failures],
                                    "winner": str(result.best.config)})
                return result

            artifact._namespace["exhaustive_search"] = record_search

            def trusted_runtime(function):
                def invoke(*args, **kwargs):
                    # Compiler-owned trial buffers may copy state. Candidate
                    # host code remains under CandidateTorchPolicy.
                    with _disable_current_modes():
                        return function(*args, **kwargs)
                return invoke

            artifact._launcher = trusted_runtime(artifact._launcher)
            artifact._runner = trusted_runtime(artifact._runner)
        self.generated[name] = artifact
        return artifact

    @contextmanager
    def native_compilation_cache(self):
        if self.target_name != "cutile":
            yield
            return
        import cuda.tile as ct
        from cuda.tile._compile import format_sm_arch, get_sm_arch

        original_compile = ct.kernel._compile
        compiled = self._native_compilations
        compiled.clear()

        def compile_kernel(kernel, signature, context, compute_capability=None):
            # Native signatures contain unhashable constraints. Value equality
            # also lets the JIT's unnamed signature reuse the prepared symbol.
            architecture = (get_sm_arch() if compute_capability is None
                            else format_sm_arch(*compute_capability))
            key = (kernel._pyfunc, kernel._compiler_options,
                   signature.with_symbol(None), architecture, context)
            for previous, result, error in compiled:
                if key == previous:
                    self.native_compile_reuses += 1
                    if error is not None:
                        raise error
                    return result
            try:
                result = original_compile(kernel, signature, context, compute_capability)
            except ct.TileError as error:
                compiled.append((key, None, error))
                raise
            compiled.append((key, result, None))
            return result

        ct.kernel._compile = compile_kernel
        try:
            yield
        finally:
            ct.kernel._compile = original_compile

    @contextmanager
    def compilation_only(self):
        import cuda.tile as ct
        from cuda.tile._compile import get_sm_arch
        from cuda.tile.compilation import CallingConvention, KernelSignature
        from cuda.tile._cext import default_tile_context

        def signature_for(kernel, arguments):
            # Generated kernels have the view/scalar ABI. Match cuTile's JIT
            # convention for static array dimensions, including rank-zero views.
            static_arrays = any(annotation.array is not None and annotation.array.static_shape_dims
                                for annotation in kernel._annotated_function.parameter_annotations)
            convention = (CallingConvention.cutile_python_v2() if static_arrays
                          else CallingConvention.cutile_python_v1())
            signature = KernelSignature.from_kernel_args(kernel, arguments, convention)
            return signature, static_arrays

        def compile_kernel(stream, grid, kernel, arguments):
            signature, _ = signature_for(kernel, arguments)
            kernel._compile(signature, default_tile_context)

        def compile_search(configs, stream, grid_fn, kernel, args_fn, hints_fn=None, **kwargs):
            first = None
            last_error = None
            pending = []
            architecture = get_sm_arch()
            for config in configs:
                candidate = kernel.replace_hints(**(hints_fn(config) if hints_fn else {}))
                signature, static_arrays = signature_for(candidate, args_fn(config))
                pending.append((config, candidate, signature, executor.submit(
                    _compile_cutile_configuration, candidate._pyfunc.__module__,
                    candidate._pyfunc.__name__, signature.parameters, static_arrays,
                    signature.symbol, candidate._compiler_options, architecture,
                    default_tile_context.config.compiler_timeout_sec)))
            for config, candidate, signature, future in pending:
                result, diagnostic = future.result()
                error = ct.TileError(diagnostic) if diagnostic is not None else None
                key = (candidate._pyfunc, candidate._compiler_options,
                       signature.with_symbol(None), architecture, default_tile_context)
                self._native_compilations.append((key, result, error))
                if error is not None:
                    last_error = error
                    self.precompile_failures.append({"kernel": kernel._pyfunc.__name__,
                                                     "config": str(config), "error": str(error)})
                    continue
                if first is None:
                    first = config
            if first is None:
                raise RuntimeError("no cuTile configuration compiled for the current device") from last_error
            # The launch dispatcher receives these native compile results from
            # the same signature-keyed hook; no kernel executes during prepare.
            return SimpleNamespace(best=SimpleNamespace(config=first))

        class CompilationState:
            def __init__(self, views, writable):
                self.views = views

            def arguments(self, arguments):
                return arguments

        original_launch = ct.launch
        saved = []
        sources = []
        for artifact in self.generated.values():
            namespace = artifact._namespace
            kernel = namespace["_intent_kernel"]
            sources.append((kernel._pyfunc.__module__, kernel._pyfunc.__code__.co_filename,
                            artifact.source))
            saved.append((namespace, namespace["TuningState"], namespace["exhaustive_search"]))
            namespace["TuningState"] = CompilationState
            namespace["exhaustive_search"] = compile_search
        ct.launch = compile_kernel
        try:
            # Each process owns the SDK's compiler lock. Only native compilation
            # runs here; device execution and numerical comparison stay in the parent.
            with ProcessPoolExecutor(max_workers=4,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_cutile_compiler,
                                     initargs=(sources,)) as executor:
                yield
        finally:
            ct.launch = original_launch
            for namespace, state, search in saved:
                namespace["TuningState"] = state
                namespace["exhaustive_search"] = search
                namespace["_TUNE_CACHE"].clear()

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
