from .artifact import CompiledArtifact
from .source import materialize_python_source
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


def materialize_tilelang_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
) -> CompiledArtifact:
    return materialize_python_source(
        target_name="tilelang",
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=None,
    )
