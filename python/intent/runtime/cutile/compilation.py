"""Program-owned cuTile compilation, shared by preparation and SDK dispatch."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import linecache
import multiprocessing
from threading import RLock


def _initialize_compiler(module_name, filename, source):
    global _compile_namespace
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    _compile_namespace = {"__name__": module_name}
    exec(compile(source, filename, "exec", dont_inherit=True), _compile_namespace)


def _compile_configuration(function_name, parameters, convention, symbol, options,
                           architecture, context_config):
    import cuda.tile as ct
    from cuda.tile._cext import TileContext
    from cuda.tile._compile import compile_tile
    from cuda.tile.compilation import CallingConvention, KernelSignature

    kernel = _compile_namespace[function_name]
    signature = KernelSignature(parameters, CallingConvention.from_code(convention), symbol)
    try:
        compiled = compile_tile(kernel._annotated_function, (signature,), architecture,
                                options, TileContext(config=context_config))
    except ct.TileError as error:
        return None, (type(error).__name__, str(error))
    return (compiled.cubin, compiled.kernel_signatures[0].symbol, None, []), None


@dataclass(frozen=True)
class _Compilation:
    key: tuple
    result: object
    error: tuple[str, str] | None


class CuTileCompilation:
    """Own the exact native results passed to this program's SDK dispatchers.

    The SDK has no dispatcher preloading API. Its kernel subclass callback is
    used for both ordinary JIT and explicit compilation; global SDK functions,
    autotuning and launch remain untouched. No tensors enter this result cache.
    """

    def __init__(self, source: str, kernels: dict[str, object]):
        import cuda.tile as ct

        self._source = source
        self._kernels = kernels
        self._compiled: list[_Compilation] = []
        self._variants = []
        self._lock = RLock()
        owner = self

        class ProgramKernel(ct.kernel):
            def _compile(self, signature, context, compute_capability=None):
                compute_capability = owner._capability(compute_capability)
                key = owner._key(self, signature, context, compute_capability)
                with owner._lock:
                    cached = owner._lookup(key)
                    if cached is not None:
                        if cached.error is not None:
                            kind, message = cached.error
                            raise ct.TileError(f"{kind}: {message}")
                        return cached.result
                    try:
                        result = super()._compile(signature, context, compute_capability)
                    except ct.TileError as error:
                        owner._compiled.append(_Compilation(key, None, (type(error).__name__, str(error))))
                        raise
                    owner._compiled.append(_Compilation(key, result, None))
                    return result

            def replace_hints(self, **hints):
                return owner._variant(self._intent_entry, replace(self._compiler_options, **hints))

        self._kernel_type = ProgramKernel

    def _variant(self, name, options):
        with self._lock:
            for previous_name, previous_options, kernel in self._variants:
                if previous_name == name and previous_options == options:
                    return kernel
            kernel = self._kernel_type(self._kernels[name]._pyfunc, **asdict(options))
            kernel._intent_entry = name
            self._variants.append((name, options, kernel))
            return kernel

    def kernel(self, name):
        return self._variant(name, self._kernels[name]._compiler_options)

    @staticmethod
    def _capability(compute_capability):
        if compute_capability is not None:
            return compute_capability
        import torch
        from cuda.tile._cext import get_compute_capability

        return get_compute_capability(torch.cuda.current_device())

    @staticmethod
    def _key(kernel, signature, context, compute_capability=None):
        from cuda.tile._compile import format_sm_arch

        architecture = format_sm_arch(*CuTileCompilation._capability(compute_capability))
        # Dispatcher-selected symbols do not change the native signature.
        return (kernel._intent_entry, kernel._compiler_options, signature.with_symbol(None),
                architecture, context, replace(context.config))

    def _lookup(self, key):
        return next((entry for entry in self._compiled if entry.key == key), None)

    def compile(self, requests):
        """Compile (kernel, bound arguments) requests in their declared order.

        Return each request's SDK failure, or None on success. A failed candidate
        remains a failed candidate when the unchanged SDK tuner encounters it.
        """
        from cuda.tile._cext import default_tile_context
        from cuda.tile.compilation import CallingConvention, KernelSignature

        context = default_tile_context
        keys, pending = [], []
        for kernel, arguments in requests:
            static_arrays = any(annotation.array is not None and annotation.array.static_shape_dims
                                for annotation in kernel._annotated_function.parameter_annotations)
            convention = (CallingConvention.cutile_python_v2() if static_arrays
                          else CallingConvention.cutile_python_v1())
            signature = KernelSignature.from_kernel_args(kernel, arguments, convention)
            key = self._key(kernel, signature, context)
            keys.append(key)
            with self._lock:
                if self._lookup(key) is not None or any(key == item[0] for item in pending):
                    continue
            pending.append((key, kernel._intent_entry, signature, kernel._compiler_options))

        if pending:
            exemplar = next(iter(self._kernels.values()))._pyfunc
            with ProcessPoolExecutor(max_workers=4,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_compiler,
                                     initargs=(exemplar.__module__, exemplar.__code__.co_filename,
                                               self._source)) as executor:
                submitted = [(key, executor.submit(_compile_configuration, name, signature.parameters,
                                                   signature.calling_convention.code, signature.symbol,
                                                   options, key[3], key[5]))
                             for key, name, signature, options in pending]
                for key, future in submitted:
                    result, error = future.result()
                    with self._lock:
                        if self._lookup(key) is None:
                            self._compiled.append(_Compilation(key, result, error))
        with self._lock:
            return tuple(self._lookup(key).error for key in keys)
