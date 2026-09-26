from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import linecache
import multiprocessing
from types import SimpleNamespace


def _initialize_compiler(sources):
    global _compile_namespaces
    _compile_namespaces = {}
    for module_name, filename, source in sources:
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        namespace = {"__name__": module_name}
        exec(compile(source, filename, "exec", dont_inherit=True), namespace)
        _compile_namespaces[module_name] = namespace


def _compile_configuration(module_name, function_name, parameters, static_arrays,
                           symbol, options, architecture, timeout):
    import cuda.tile as ct
    from cuda.tile._compile import compile_tile
    from cuda.tile.compilation import CallingConvention, KernelSignature

    kernel = _compile_namespaces[module_name][function_name]
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


class CuTileCompilation:
    """Prepare generated kernels without running their tuning candidates."""

    def __init__(self):
        self.compiled = []
        self.failures = []
        self.reuses = 0

    @contextmanager
    def cache(self):
        import cuda.tile as ct
        from cuda.tile._compile import format_sm_arch, get_sm_arch

        original_compile = ct.kernel._compile
        self.compiled.clear()

        def compile_kernel(kernel, signature, context, compute_capability=None):
            # Native constraints are unhashable. Ignore only the prepared symbol;
            # the launch dispatcher's otherwise identical signature is unnamed.
            architecture = (get_sm_arch() if compute_capability is None
                            else format_sm_arch(*compute_capability))
            key = (kernel._pyfunc, kernel._compiler_options,
                   signature.with_symbol(None), architecture, context)
            for previous, result, error in self.compiled:
                if key == previous:
                    self.reuses += 1
                    if error is not None:
                        raise error
                    return result
            try:
                result = original_compile(kernel, signature, context, compute_capability)
            except ct.TileError as error:
                self.compiled.append((key, None, error))
                raise
            self.compiled.append((key, result, None))
            return result

        ct.kernel._compile = compile_kernel
        try:
            yield
        finally:
            ct.kernel._compile = original_compile

    @contextmanager
    def compilation_only(self, artifacts):
        import cuda.tile as ct
        from cuda.tile._compile import get_sm_arch
        from cuda.tile.compilation import CallingConvention, KernelSignature
        from cuda.tile._cext import default_tile_context

        def signature_for(kernel, arguments):
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
                    _compile_configuration, candidate._pyfunc.__module__,
                    bindings[candidate._pyfunc], signature.parameters, static_arrays,
                    signature.symbol, candidate._compiler_options, architecture,
                    default_tile_context.config.compiler_timeout_sec)))
            for config, candidate, signature, future in pending:
                result, diagnostic = future.result()
                error = ct.TileError(diagnostic) if diagnostic is not None else None
                key = (candidate._pyfunc, candidate._compiler_options,
                       signature.with_symbol(None), architecture, default_tile_context)
                self.compiled.append((key, result, error))
                if error is not None:
                    last_error = error
                    self.failures.append({"kernel": kernel._pyfunc.__name__,
                                          "config": str(config), "error": str(error)})
                    continue
                if first is None:
                    first = config
            if first is None:
                raise RuntimeError("no cuTile configuration compiled for the current device") from last_error
            return SimpleNamespace(best=SimpleNamespace(config=first))

        class CompilationState:
            def __init__(self, views, writable):
                self.views = views

            def arguments(self, arguments):
                return arguments

        original_launch = ct.launch
        saved = []
        sources = []
        bindings = {}
        for artifact in artifacts:
            namespace = artifact._namespace
            kernel = namespace["_intent_kernel"]
            for name, value in namespace.items():
                if isinstance(value, ct.kernel):
                    bindings[value._pyfunc] = name
            sources.append((kernel._pyfunc.__module__, kernel._pyfunc.__code__.co_filename,
                            artifact.source))
            saved.append((namespace, namespace["TuningState"], namespace["exhaustive_search"]))
            namespace["TuningState"] = CompilationState
            namespace["exhaustive_search"] = compile_search
        ct.launch = compile_kernel
        try:
            # Each process owns the SDK compiler lock. Device execution remains
            # in the benchmark process after this preparation context exits.
            with ProcessPoolExecutor(max_workers=4,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_compiler,
                                     initargs=(sources,)) as executor:
                yield
        finally:
            ct.launch = original_launch
            for namespace, state, search in saved:
                namespace["TuningState"] = state
                namespace["exhaustive_search"] = search
                namespace["_TUNE_CACHE"].clear()
