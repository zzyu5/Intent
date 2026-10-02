from contextlib import contextmanager
from types import SimpleNamespace

from ..artifact import CompiledArtifact
from ..gpu.expressions import evaluate_shape
from ..gpu.program import LaunchResult, materialize_gpu_program
from ..tuning import TuningState
from ..diagnostics import CandidateObservation, bindings, observation, unavailable_resources
from intent.compiler.toolchain import CompilationStageError


def bind_array_view(view, group_ends: tuple[int, ...]):
    shape, strides = [], []
    begin = 0
    for end in group_ends:
        extent = 1
        for axis in range(begin, end):
            if view.shape[axis] <= 0:
                return view, False
            if axis > begin and (view.stride(axis) <= 0 or
                                 view.stride(axis - 1) != view.shape[axis] * view.stride(axis)):
                return view, False
            extent *= view.shape[axis]
        shape.append(extent)
        strides.append(view.stride(end - 1))
        begin = end
    return view.as_strided(shape, strides), True


def array_index_kernels(function, view_names: tuple[str, ...]):
    from types import FunctionType
    from typing import Annotated, get_args
    import cuda.tile as ct

    narrow = FunctionType(function.__code__, function.__globals__, function.__name__,
                          function.__defaults__, function.__closure__)
    narrow.__qualname__ = function.__qualname__
    narrow.__annotations__ = dict(function.__annotations__)
    for name in view_names:
        array = get_args(function.__annotations__[name])[1]
        narrow.__annotations__[name] = Annotated[
            ct.Array, ct.ArrayAnnotation(static_shape_dims=array.static_shape_dims,
                                        static_stride_dims=array.static_stride_dims)
        ]
    return ct.kernel(narrow), ct.kernel(function)


def can_use_i32_array_indices(views: tuple, tile_bounds: tuple) -> bool:
    limit = (1 << 31) - 1
    for view, tiles in zip(views, tile_bounds, strict=True):
        span = 0
        padded_elements = 1
        for extent, stride, tile in zip(view.shape, view.stride(), tiles, strict=True):
            if extent <= 0 or tile <= 0 or stride < 0 or stride > limit:
                return False
            # Native access origins are in the view; include the physical tail.
            padded_extent = extent + tile - 1
            if padded_extent > limit:
                return False
            span += (padded_extent - 1) * stride
            padded_elements *= padded_extent
        if (span + 1) * view.element_size() > limit or padded_elements > limit:
            return False
    return True


def _search(*arguments, **keywords):
    from cuda.tile.tune import exhaustive_search

    return exhaustive_search(*arguments, **keywords)


class CuTileProgram:
    def __init__(self, interface, namespace: dict, facts, target: dict) -> None:
        self.interface = interface
        self.configurations = interface.configuration_space
        self.facts = facts
        self.target = target
        self.kernel = namespace[facts.kernel]
        self.narrow_kernel = None if facts.narrow_kernel is None else namespace[facts.narrow_kernel]
        self.native_kernels = {name: namespace[name] for name in (facts.kernel, facts.narrow_kernel)
                               if name is not None}
        self.tile_bounds = facts.index_tile_bounds
        self.search = _search
        self.trial_state = TuningState
        self.observe_tuning = None
        self.tuning_options: dict = {}
        self._winners: dict = {}
        self._compiling = False

    def _array_values(self, values: dict) -> None:
        for entry in self.facts.array_views:
            values[entry.name], values[entry.eligible] = bind_array_view(
                values[entry.base], entry.group_ends)

    def _hints(self, config) -> dict:
        hints = {hint: getattr(config, parameter) for hint, parameter in self.facts.compiler_hints}
        if hints.get("num_worker_warps") == self.facts.inferred_worker_warps:
            hints["num_worker_warps"] = None
        return hints

    def _arguments(self, values: dict, config) -> tuple:
        bindings = {**values, **vars(config)}
        return tuple(self.interface.native_value(name, bindings) for name in self.facts.kernel_arguments)

    def _grid(self, values: dict, config) -> tuple:
        return (*evaluate_shape(self.interface.grid, {**values, **vars(config)}), 1, 1)

    def _key(self, invocation, values: dict) -> tuple:
        return (tuple((tuple(view.shape), tuple(view.stride()), view.dtype, view.device)
                      for view in invocation.views),
                tuple(self.interface.native_value(name, values) for name in self.facts.tuning_key_scalars),
                tuple(values[name] for name in self.configurations.coverage_names),
                tuple(values[entry.eligible] for entry in self.facts.array_views),
                tuple(values[entry.name] for entry in self.interface.overlaps))

    def launch(self, invocation) -> LaunchResult:
        import torch
        import cuda.tile as ct

        values = dict(invocation.values)
        self._array_values(values)
        key = self._key(invocation, values)
        cached = self._winners.get(key)
        reused = cached is not None
        candidates = ()
        if cached is None:
            configurations = tuple(SimpleNamespace(**config) for config in self.configurations.candidates(values))
            kernel = self.kernel
            if self.tile_bounds is not None:
                bounds = tuple(evaluate_shape(bound, values) for bound in self.tile_bounds)
                if can_use_i32_array_indices(invocation.views, bounds):
                    kernel = self.narrow_kernel
            state = self.trial_state(invocation.public_views,
                                     tuple(view.writable for view in self.interface.public_views))
            trial_values = dict(values)
            trial_values.update((view.id, trial) for view, trial in
                                zip(self.interface.public_views, state.views, strict=True))
            self._array_values(trial_values)
            hints = (self._hints,) if self.facts.compiler_hints else ()
            try:
                result = self.search(configurations, torch.cuda.current_stream(),
                                      lambda config: self._grid(values, config), kernel,
                                      lambda config: state.arguments(self._arguments(trial_values, config)),
                                      *hints, quiet=True, **self.tuning_options)
            except Exception as error:
                details = observation("cutile", self.target, invocation, None, _native_resources(),
                    stage="failed", history_unavailable=
                    "cuTile did not return a TuningResult; its exception is preserved, but no complete candidate history is available")
                stage = "provider_native_compilation" if self._compiling else "provider_tuning"
                raise CompilationStageError(stage, str(error), observation=details) from error
            if self.observe_tuning is not None:
                self.observe_tuning(configurations, result)
            config = result.best.config
            selected = kernel.replace_hints(**self._hints(config)) if hints else kernel
            candidates = () if self._compiling else tuple(
                [CandidateObservation(bindings(vars(item.config)), "trial_completed", "provider_tuning")
                 for item in result.successes] +
                [CandidateObservation(bindings(vars(config)), "failed", "provider_tuning", error.__name__, message)
                 for config, error, message in result.failures])
            cached = config, selected
            if not self._compiling:
                self._winners[key] = cached
        config, selected = cached
        grid = self._grid(values, config)
        arguments = self._arguments(values, config)

        def invoke():
            try:
                return ct.launch(torch.cuda.current_stream(), grid, selected, arguments)
            except Exception as error:
                details = observation("cutile", self.target, invocation, vars(config), _native_resources(),
                                      candidates, stage="failed")
                raise CompilationStageError("provider_invocation", str(error), observation=details) from error

        invoke()
        details = None if self._compiling else observation(
            "cutile", self.target, invocation, vars(config), _native_resources(), candidates,
            tuning_cache_hit=reused, history_unavailable=
            "The existing tuning winner was reused; no candidates were retried in this invocation" if reused else None)
        return LaunchResult(None if self._compiling else invoke, selected, details)

    def tuning_configurations(self, invocation):
        return self.configurations.enumerate(invocation.values,
                                             self.configurations.candidates(invocation.values))

    @contextmanager
    def compilation_only(self, search, trial_state):
        """Use a native compilation driver without recording its placeholders as tuning winners."""
        previous = self.search, self.trial_state, self.observe_tuning, self._compiling
        self.search, self.trial_state, self.observe_tuning = search, trial_state, None
        self._compiling = True
        try:
            yield
        finally:
            self.search, self.trial_state, self.observe_tuning, self._compiling = previous
            self._winners.clear()


def _native_resources():
    return unavailable_resources("cuda.tile.CompilationResult",
        "The selected cuTile dispatcher exposes no native register/shared-memory resource fields")


def materialize_cutile_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    contract,
) -> CompiledArtifact:
    return materialize_gpu_program(
        provider_name="cutile", provider_type=CuTileProgram, contract=contract,
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=None,
    )
