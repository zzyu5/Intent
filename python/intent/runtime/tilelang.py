from .artifact import CompiledArtifact
from .gpu.program import LaunchResult, materialize_gpu_program
from .tuning import TuningState
from .diagnostics import (CandidateRecorder, describe_tuning_failure, observation,
                          unavailable_resources)
from intent.compiler.toolchain import CompilationStageError
from threading import Lock


def tune_kernel(jit, configs: list[dict], parameters: dict,
                arguments: tuple, writable: tuple[bool, ...], recorder: CandidateRecorder):
    import torch
    from tilelang.autotuner import AutoTuner

    inputs_ready = torch.cuda.Event()
    inputs_ready.record()
    compiled_rows = {}
    compiled_lock = Lock()

    class InvocationTuner(AutoTuner):
        def _benchmark_target(self, jit_kernel, warmup, rep, early_stop_factor,
                              benchmark_state, benchmark_device=None):
            inputs_ready.wait()
            state = TuningState(arguments[:len(writable)], writable)
            trial_arguments = state.views + arguments[len(writable):]

            def invoke():
                state.reset()
                jit_kernel(*trial_arguments)

            profiler = jit_kernel.get_profiler()
            with compiled_lock:
                _, configuration = compiled_rows.get(id(jit_kernel), (None, None))
            try:
                latency = profiler.do_bench(
                    func=invoke, input_tensors=[], n_warmup=warmup, n_repeat=rep,
                    backend=self.profile_args.backend, device=benchmark_device)
            except Exception as error:
                recorder.record(configuration, "failed", "provider_benchmark", error)
                raise
            recorder.record(configuration, "trial_completed", "provider_benchmark")
            return latency, None

    tuner = InvocationTuner(jit.func, configs=configs).set_compile_args(
        out_idx=jit.out_idx, execution_backend=jit.execution_backend,
        target=jit.target, target_host=jit.target_host, verbose=jit.verbose,
        pass_configs=jit.pass_configs)
    tuner.set_profile_args(supply_prog=lambda _: arguments)
    tuner.set_kernel_parameters(((), tuple(sorted(parameters.items()))),
                                jit.signature.parameters)
    def compile_candidate(*, __pass_configs__=None, **config):
        try:
            compiled = jit.compile(**parameters, **config)
        except Exception as error:
            recorder.record(config, "failed", "provider_native_compilation", error)
            raise
        recorder.record(config, "compiled", "provider_native_compilation")
        with compiled_lock:
            previous = compiled_rows.get(id(compiled))
            # An SDK may share a compiled object across equivalent candidates.
            # Its benchmark callback then cannot identify one of those rows.
            row = config if previous is None or previous[1] == config else None
            compiled_rows[id(compiled)] = compiled, row
        return compiled

    def elaborate_candidate(*, __pass_configs__=None, **config):
        try:
            return jit.get_tir(**parameters, **config)
        except Exception as error:
            recorder.record(config, "failed", "provider_elaboration", error)
            raise

    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = elaborate_candidate
    return tuner.run(warmup=3, rep=10, benchmark_multi_gpu=False).kernel


class TileLangProgram:
    def __init__(self, interface, namespace: dict, facts: dict, target: dict) -> None:
        self.interface = interface
        self.configurations = interface.configuration_space
        self.facts = facts
        self.target = target
        self.kernel = namespace[facts["kernel"]]
        self._winners = {}

    def launch(self, invocation) -> LaunchResult:
        values = invocation.values
        key = (tuple((tuple(view.shape), tuple(view.stride()), view.dtype, view.device)
                     for view in invocation.views),
               tuple(values[scalar.id] for scalar in self.interface.scalars),
               tuple(values[entry["name"]] for entry in self.interface.overlaps))
        arguments = tuple(self.interface.native_value(name, values) for name in self.facts["kernel_arguments"])
        reused = key in self._winners
        recorder = CandidateRecorder()
        if not reused:
            bindings = {name: self.interface.native_value(name, values) for name in self.facts["builder_arguments"]
                        if name not in self.configurations.bound_names}
            configs = list(self.configurations.candidates(values))
            try:
                self._winners[key] = tune_kernel(self.kernel, configs, bindings,
                    arguments, tuple(view.writable for view in self.interface.views), recorder)
            except Exception as error:
                details = observation("tilelang", self.target, invocation, None, _native_resources(),
                                      recorder.snapshot(), stage="failed", history_unavailable=
                                      "Only the SDK compilation/benchmark callbacks reached during this invocation are recorded")
                raise CompilationStageError(
                    "provider_tuning", describe_tuning_failure(error, details.candidates),
                    observation=details) from error
        compiled = self._winners[key]
        configuration = compiled.config

        def invoke():
            try:
                return compiled(*arguments)
            except Exception as error:
                details = observation("tilelang", self.target, invocation, configuration, _native_resources(),
                                      recorder.snapshot(), stage="failed")
                raise CompilationStageError("provider_invocation", str(error), observation=details) from error

        invoke()
        details = observation("tilelang", self.target, invocation, configuration, _native_resources(),
                              recorder.snapshot(), tuning_cache_hit=True if reused else None,
                              history_unavailable="Only this invocation's SDK callbacks are recorded; SDK cache hits do not replay candidate history")
        return LaunchResult(invoke, compiled, details)

    def tuning_configurations(self, invocation):
        return self.configurations.enumerate(invocation.values,
                                             self.configurations.candidates(invocation.values))


def _native_resources():
    return unavailable_resources("tilelang.JITKernel",
        "The supported TileLang CUDA runtime does not expose native register/shared-memory resource counts")


def _collect_tilelang_ir(kernel):
    return {"tilelang_native_source": kernel.get_kernel_source()}


def materialize_tilelang_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    metadata: dict,
) -> CompiledArtifact:
    return materialize_gpu_program(
        provider_name="tilelang", provider_type=TileLangProgram, metadata=metadata,
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=_collect_tilelang_ir,
    )
