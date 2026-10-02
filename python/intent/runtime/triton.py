from .artifact import CompiledArtifact
from .gpu.expressions import evaluate_shape, read_expressions
from .gpu.program import LaunchResult, materialize_gpu_program
from .tuning import TuningState
from .diagnostics import CandidateRecorder, observation, resource, unavailable_resources
from intent.compiler.toolchain import CompilationStageError


class TuningHooks:
    def __init__(self, names: tuple[str, ...], writable: tuple[bool, ...],
                 readable: tuple[bool, ...]):
        self.names = names
        self.writable = writable
        self.readable = readable
        self.state = None
        self.observe_trial = None
        self._observed_arguments = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.finish()
        finally:
            self._observed_arguments = None

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
        try:
            # Triton reuses full_nargs through the warmup/measurement calls of a
            # candidate. Capture the first completed trial and any failure, not
            # one diagnostic record allocation on every timed callback.
            if self.observe_trial is not None and (
                exception is not None or arguments is not self._observed_arguments
            ):
                self._observed_arguments = arguments
                self.observe_trial(arguments, exception)
        finally:
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
    def __init__(self, interface, namespace: dict, facts: dict, target: dict) -> None:
        import triton

        self.interface = interface
        self.configurations = interface.configuration_space
        self.facts = facts
        self.target = target
        self.descriptors = tuple({**entry,
                                  "shape": read_expressions(entry["shape"]),
                                  "strides": read_expressions(entry["strides"]),
                                  "block_shape": read_expressions(entry["block_shape"])}
                                 for entry in facts["descriptors"])
        self.hooks = TuningHooks(tuple(view.kernel_name for view in interface.public_views),
                                 tuple(view.writable for view in interface.public_views),
                                 tuple(view.access != 1 for view in interface.public_views))
        kernel_parameters = tuple(name for name in facts["kernel_parameters"]
                                  if name not in self.configurations.coverage_names)
        native_options = facts["native_options"]
        configs = [triton.Config(
            {name: row[name] for name in kernel_parameters},
            num_warps=row[native_options["num_warps"]],
            num_stages=row[native_options["num_stages"]],
            num_ctas=row[native_options["num_ctas"]],
        ) for row in self.configurations.rows]
        self._config_rows = {id(config): row for config, row in
                             zip(configs, self.configurations.rows, strict=True)}
        kernel = namespace[facts["kernel"]]
        if self.descriptors:
            kernel = triton.heuristics({entry["name"]: self._descriptor_hook(entry)
                                        for entry in self.descriptors})(kernel)
        options = {}
        if facts["descriptor_choice"] is not None or self.configurations.requirements:
            options["prune_configs_by"] = {"early_config_prune": self._prune}
        kernel = triton.autotune(configs=configs, key=facts["autotune_key"],
                                pre_hook=self.hooks.before, post_hook=self.hooks.after,
                                **options)(kernel)
        self.kernel = kernel

    def _trial_configuration(self, arguments):
        row = {name: arguments[name] for name in self.facts["kernel_parameters"]
               if name not in self.configurations.coverage_names}
        row.update((parameter, arguments[option]) for option, parameter in self.facts["native_options"].items())
        return row

    def _context(self, arguments: dict) -> dict:
        values = self.interface.callback_values(arguments)
        packed = self.facts["metadata_argument"]
        if packed is not None:
            for entry, value in zip(self.interface.metadata, values[packed], strict=True):
                values[entry.id] = value
        return values

    def _descriptor_hook(self, entry):
        def bind(arguments):
            from triton.tools.tensor_descriptor import TensorDescriptor

            choice = self.facts["descriptor_choice"]
            values = self._context(arguments)
            if not arguments[choice["config"]]:
                return values[entry["base"]]
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
        invocation_values = self._context({**named_args, **kwargs})
        retained = []
        for config in configs:
            values = {**invocation_values, **self._config_rows[id(config)]}
            choice = self.facts["descriptor_choice"]
            if choice is not None and values[choice["config"]]:
                if not values[choice["eligibility"]] or not all(
                    self._block_legal(entry, values, config.num_stages) for entry in self.descriptors
                ):
                    continue
            if self.configurations.within_resources(values):
                retained.append(config)
        if not retained:
            self.configurations.candidates(invocation_values)
            raise CompilationStageError(
                "provider_eligibility",
                "No Triton configuration satisfies the declared descriptor alignment and block-shape requirements",
            )
        return retained

    def launch(self, invocation) -> LaunchResult:
        import triton

        values = dict(invocation.values)
        packed = self.facts["metadata_argument"]
        if packed is not None:
            values[packed] = tuple(values[entry.id] for entry in self.interface.metadata)
        packed = self.facts["overlap_argument"]
        if packed is not None:
            values[packed] = tuple(values[entry["name"]] for entry in self.interface.overlaps)
        choice = self.facts["descriptor_choice"]
        if choice is not None:
            values[choice["eligibility"]] = all(self._eligible(entry, values) for entry in self.descriptors)
        arguments = tuple(self.interface.native_value(name, values) for name in self.facts["kernel_arguments"])
        coverage = {name: values[name] for name in self.configurations.coverage_names}
        recorder = CandidateRecorder()

        def trial(arguments, error):
            recorder.record(self._trial_configuration(arguments),
                            "failed" if error is not None else "trial_completed", "provider_tuning", error)

        def grid(config):
            return evaluate_shape(self.interface.grid, {**values, **config})

        def invoke():
            previous = self.hooks.observe_trial
            self.hooks.observe_trial = trial
            try:
                if self.facts["allocator"] is not None:
                    triton.set_allocator(_descriptor_allocator)
                with self.hooks:
                    return self.kernel[grid](*arguments, **coverage,
                                             enable_fp_fusion=True, enable_reflect_ftz=False)
            except CompilationStageError:
                raise
            except Exception as error:
                details = observation("triton", self.target, invocation, None,
                    unavailable_resources("triton", "The invocation did not return a loaded native kernel"),
                    recorder.snapshot(), stage="failed")
                raise CompilationStageError("provider_invocation", str(error), observation=details) from error
            finally:
                self.hooks.observe_trial = previous

        compiled = invoke()
        # Triton's disk winner cache reconstructs Config objects. The selected
        # binding, not Python object identity, must match the declared IR rows.
        config = self._trial_configuration(self.kernel.best_config.all_kwargs())
        if config not in self.configurations.rows:
            raise CompilationStageError("provider_configuration",
                                        "Triton selected a configuration absent from the generated program")
        history = recorder.snapshot()
        recorder.record(config, "selected", "provider_invocation")
        details = observation("triton", self.target, invocation, config, _native_resources(compiled),
                              recorder.snapshot(), history_unavailable=None if history else
                              "No tuning trial callbacks occurred in this invocation; only the selected kernel was observed")
        return LaunchResult(invoke, compiled, details)

    def tuning_configurations(self, invocation):
        return self.configurations.enumerate(invocation.values, self.configurations.rows)

def _collect_triton_ir(compiled_kernel: object) -> dict[str, str]:
    asm = getattr(compiled_kernel, "asm", None)
    if not isinstance(asm, dict):
        raise RuntimeError("Triton launch did not return a compiled kernel artifact")
    result = {name: value for name, value in asm.items() if isinstance(value, str)}
    if not result:
        raise RuntimeError("Triton compiled artifact exposes no textual backend IR")
    return result


def _native_resources(kernel):
    metadata = kernel.metadata
    fields = (
        ("registers_per_thread", getattr(kernel, "n_regs", None), "registers", "CompiledKernel.n_regs", "native_loading"),
        # NVIDIA load_binary reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4.
        ("local_memory_words_per_thread", getattr(kernel, "n_spills", None), "32-bit words",
         "CompiledKernel.n_spills (CUDA LOCAL_SIZE_BYTES / 4)", "native_loading"),
        ("shared_memory_bytes", getattr(metadata, "shared", None), "bytes", "CompiledKernel.metadata.shared", "native_compilation"),
        ("max_threads_per_block", getattr(kernel, "n_max_threads", None), "threads", "CompiledKernel.n_max_threads", "native_loading"),
    )
    return tuple(resource(name, value, unit, "triton." + source, stage,
                          unavailable="The selected Triton compiled kernel does not expose this field")
                 for name, value, unit, source, stage in fields)


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
