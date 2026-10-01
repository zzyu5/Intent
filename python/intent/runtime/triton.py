from .artifact import CompiledArtifact
from .gpu.expressions import evaluate_shape, read_expressions
from .gpu.program import LaunchResult, materialize_gpu_program
from .tuning import TuningState


class TuningHooks:
    def __init__(self, names: tuple[str, ...], writable: tuple[bool, ...],
                 readable: tuple[bool, ...]):
        self.names = names
        self.writable = writable
        self.readable = readable
        self.state = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.finish()

    def finish(self) -> None:
        if self.state is None:
            return
        from torch.utils._python_dispatch import _disable_current_modes

        # Cleanup must also run when measurement fails outside Triton's hooks.
        try:
            with _disable_current_modes():
                self.state.restore()
        finally:
            self.state = None

    def before(self, arguments: dict, reset_only: bool = False) -> None:
        if reset_only:
            self.finish()
            return
        if self.state is None:
            self.state = TuningState(tuple(arguments[name] for name in self.names),
                                     self.writable)
        self.state.restore(self.readable)

    def after(self, arguments: dict, exception: Exception | None) -> None:
        if exception is not None:
            self.finish()
        else:
            self.state.restore(self.readable)


def _descriptor_allocator(size, alignment, stream):
    import torch

    buffer = torch.empty(size, dtype=torch.int8, device="cuda")
    if buffer.data_ptr() % alignment:
        raise RuntimeError("Triton descriptor allocator returned a misaligned buffer")
    return buffer


class TritonProgram:
    def __init__(self, interface, namespace: dict, facts: dict) -> None:
        import triton

        self.interface = interface
        self.facts = facts
        self.descriptors = tuple({**entry,
                                  "shape": read_expressions(entry["shape"]),
                                  "strides": read_expressions(entry["strides"]),
                                  "block_shape": read_expressions(entry["block_shape"])}
                                 for entry in facts["descriptors"])
        self.hooks = TuningHooks(tuple(view.kernel_name for view in interface.public_views),
                                 tuple(view.writable for view in interface.public_views),
                                 tuple(view.access != 1 for view in interface.public_views))
        configs = [triton.Config(entry["parameters"], num_warps=entry["num_warps"],
                                 num_stages=entry["num_stages"], num_ctas=entry["num_ctas"])
                   for entry in facts["configs"]]
        kernel = namespace[facts["kernel"]]
        if self.descriptors:
            kernel = triton.heuristics({entry["name"]: self._descriptor_hook(entry)
                                        for entry in self.descriptors})(kernel)
        options = {}
        if facts["descriptor_choice"] is not None or interface.resource_bounds:
            options["prune_configs_by"] = {"early_config_prune": self._prune}
        kernel = triton.autotune(configs=configs, key=facts["autotune_key"],
                                pre_hook=self.hooks.before, post_hook=self.hooks.after,
                                **options)(kernel)
        if interface.coverage:
            kernel = triton.heuristics({name: self._coverage_hook(name, expression, candidates)
                                        for name, expression, candidates in interface.coverage})(kernel)
        self.kernel = kernel

    def _context(self, arguments: dict) -> dict:
        values = dict(arguments)
        packed = self.facts["metadata_argument"]
        if packed is not None:
            for entry, value in zip(self.interface.metadata, values[packed], strict=True):
                values[entry["name"]] = value
                values[entry["kernel_name"]] = value
        return values

    def _coverage_hook(self, name, expression, candidates):
        def bind(arguments):
            required = expression(self._context(arguments))
            for candidate in candidates:
                if candidate >= required:
                    return candidate
            raise ValueError(f"no legal full-coverage extent for {name}")
        return bind

    def _descriptor_hook(self, entry):
        def bind(arguments):
            from triton.tools.tensor_descriptor import TensorDescriptor

            choice = self.facts["descriptor_choice"]
            if not arguments[choice["config"]]:
                return arguments[entry["base"]]
            values = self._context(arguments)
            return TensorDescriptor(values[entry["base"]],
                                    shape=list(evaluate_shape(entry["shape"], values)),
                                    strides=list(evaluate_shape(entry["strides"], values)),
                                    block_shape=list(evaluate_shape(entry["block_shape"], values)),
                                    padding=entry["padding"])
        return bind

    @staticmethod
    def _eligible(entry, values) -> bool:
        tensor = values[entry["base"]]
        shape = evaluate_shape(entry["shape"], values)
        strides = evaluate_shape(entry["strides"], values)
        alignment = entry["alignment"]
        return (tensor.ndim == entry["rank"] and tensor.data_ptr() % alignment == 0
                and (not entry["require_positive_shape"] or all(extent > 0 for extent in shape))
                and all(extent <= entry["maximum_shape_extent"] for extent in shape)
                and (not entry["require_positive_strides"] or all(stride > 0 for stride in strides))
                and strides[-1] == 1
                and all(tensor.stride(axis) == 1 for axis in entry["unit_stride_axes"])
                and all(tensor.stride(axis) * tensor.element_size() % alignment == 0
                        for axis in entry["aligned_stride_axes"]))

    @staticmethod
    def _block_legal(entry, values, stages) -> bool:
        from math import prod

        shape = evaluate_shape(entry["block_shape"], values)
        if any(extent <= 0 or (entry["require_power_of_two_block_shape"] and extent & (extent - 1))
               for extent in shape):
            return False
        elements = prod(shape)
        size = values[entry["base"]].element_size()
        return (elements <= entry["maximum_block_elements"]
                and shape[-1] * size >= entry["minimum_contiguous_bytes"]
                and (stages <= 1 or elements * size % entry["pipeline_block_alignment"] == 0))

    def _prune(self, configs, named_args, **kwargs):
        retained = []
        for config in configs:
            values = self._context({**named_args, **kwargs, **config.kwargs})
            values.update((name, getattr(config, field)) for name, field in self.facts["config_options"].items())
            choice = self.facts["descriptor_choice"]
            if choice is not None and values[choice["config"]]:
                if not values[choice["eligibility"]] or not all(
                    self._block_legal(entry, values, config.num_stages) for entry in self.descriptors
                ):
                    continue
            if all(lhs(values) <= rhs(values) for lhs, rhs in self.interface.resource_bounds):
                retained.append(config)
        return retained

    def launch(self, invocation) -> LaunchResult:
        import triton

        values = dict(invocation.values)
        packed = self.facts["metadata_argument"]
        if packed is not None:
            values[packed] = tuple(values[entry["name"]] for entry in self.interface.metadata)
        packed = self.facts["overlap_argument"]
        if packed is not None:
            values[packed] = tuple(values[entry["name"]] for entry in self.interface.overlaps)
        choice = self.facts["descriptor_choice"]
        if choice is not None:
            values[choice["eligibility"]] = all(self._eligible(entry, values) for entry in self.descriptors)
        arguments = tuple(values[name] for name in self.facts["kernel_arguments"])

        def grid(config):
            return evaluate_shape(self.interface.grid, {**values, **config})

        def invoke():
            if self.facts["allocator"] is not None:
                triton.set_allocator(_descriptor_allocator)
            with self.hooks:
                return self.kernel[grid](*arguments, enable_fp_fusion=True, enable_reflect_ftz=False)

        compiled = invoke()
        return LaunchResult(invoke, compiled)

    def tuning_configurations(self, invocation):
        from .artifact import TuningConfiguration

        parameters = self.interface.tuning_parameters
        return tuple(TuningConfiguration(parameters, tuple(({**invocation.values, **config})[p.name]
                                                           for p in parameters))
                     for config in self.interface.configurations)

def _collect_triton_ir(compiled_kernel: object) -> dict[str, str]:
    asm = getattr(compiled_kernel, "asm", None)
    if not isinstance(asm, dict):
        raise RuntimeError("Triton launch did not return a compiled kernel artifact")
    result = {name: value for name, value in asm.items() if isinstance(value, str)}
    if not result:
        raise RuntimeError("Triton compiled artifact exposes no textual backend IR")
    return result


def materialize_triton_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    metadata: dict,
) -> CompiledArtifact:
    return materialize_gpu_program(
        provider_name="triton", provider_type=TritonProgram, metadata=metadata,
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=_collect_triton_ir,
    )
