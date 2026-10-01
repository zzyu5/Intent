from .artifact import CompiledArtifact
from .gpu.program import LaunchResult, materialize_gpu_program
from .tuning import TuningState


def tune_kernel(jit, configs: list[dict], parameters: dict,
                arguments: tuple, writable: tuple[bool, ...]):
    import torch
    from tilelang.autotuner import AutoTuner

    inputs_ready = torch.cuda.Event()
    inputs_ready.record()

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
            latency = profiler.do_bench(
                func=invoke, input_tensors=[], n_warmup=warmup, n_repeat=rep,
                backend=self.profile_args.backend, device=benchmark_device)
            return latency, None

    tuner = InvocationTuner(jit.func, configs=configs).set_compile_args(
        out_idx=jit.out_idx, execution_backend=jit.execution_backend,
        target=jit.target, target_host=jit.target_host, verbose=jit.verbose,
        pass_configs=jit.pass_configs)
    tuner.set_profile_args(supply_prog=lambda _: arguments)
    tuner.set_kernel_parameters(((), tuple(sorted(parameters.items()))),
                                jit.signature.parameters)
    def compile_candidate(*, __pass_configs__=None, **config):
        return jit.compile(**parameters, **config)

    def elaborate_candidate(*, __pass_configs__=None, **config):
        return jit.get_tir(**parameters, **config)

    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = elaborate_candidate
    return tuner.run(warmup=3, rep=10, benchmark_multi_gpu=False).kernel


class TileLangProgram:
    def __init__(self, interface, namespace: dict, facts: dict) -> None:
        self.interface = interface
        self.facts = facts
        self.kernel = namespace[facts["kernel"]]
        self._winners = {}

    def launch(self, invocation) -> LaunchResult:
        values = invocation.values
        key = (tuple((tuple(view.shape), tuple(view.stride()), view.dtype, view.device)
                     for view in invocation.views),
               tuple(values[scalar.kernel_name] for scalar in self.interface.scalars),
               tuple(values[entry["name"]] for entry in self.interface.overlaps))
        arguments = tuple(values[name] for name in self.facts["kernel_arguments"])
        if key not in self._winners:
            parameter_names = {parameter.name for parameter in self.interface.tuning_parameters
                               if parameter.name not in {name for name, _, _ in self.interface.coverage}}
            bindings = {name: values[name] for name in self.facts["builder_arguments"]
                        if name not in parameter_names}
            self._winners[key] = tune_kernel(self.kernel, list(self.interface.candidates(values)), bindings,
                                             arguments, tuple(view.writable for view in self.interface.views))
        compiled = self._winners[key]

        def invoke():
            return compiled(*arguments)

        invoke()
        return LaunchResult(invoke, compiled)

    def tuning_configurations(self, invocation):
        return self.interface.tuning_configurations(invocation)


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
        backend_ir_collector=None,
    )
