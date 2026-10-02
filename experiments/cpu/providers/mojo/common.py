from __future__ import annotations

import os
from pathlib import Path
import time

import intent
import torch

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch


def configure_cpu_budget(workers: int = 8) -> None:
    allowed = os.sched_getaffinity(0)
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        candidates = []
        for component in (node / "cpulist").read_text().strip().split(","):
            span = component.split("-")
            candidates.extend(range(int(span[0]), int(span[-1]) + 1))
        selected = []
        physical = set()
        for cpu in candidates:
            if cpu not in allowed:
                continue
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = ((topology / "physical_package_id").read_text(), (topology / "core_id").read_text())
            if key not in physical:
                selected.append(cpu)
                physical.add(key)
            if len(selected) == workers:
                os.sched_setaffinity(0, selected)
                torch.set_num_threads(workers)
                return
    raise RuntimeError(f"CPU benchmark needs {workers} available physical cores on one NUMA node")


def _compile(context, definition, *, constexprs=None):
    report_stage("generated_compilation")
    started = time.monotonic()
    artifact = intent.compile(definition, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options, constexprs=constexprs)
    elapsed = time.monotonic() - started
    libraries = artifact.runtime.compilation.libraries
    reasons = sorted({library.cache_reason for library in libraries if library.cache_reason})
    print(f"mojo: generated_materialization_s={elapsed:.6f}; "
          f"native_cache_hits={sum(library.cache_hit for library in libraries)}/{len(libraries)}; "
          f"cache_unavailable={reasons}", flush=True)
    return artifact


def prepare_comparison(context, definition, arguments, runtime_path, tolerance, note=""):
    artifact = _compile(context, definition)
    generated = artifact.prepare(*arguments)
    report_stage("source_compilation")
    runtime = load_module(context.project_root / runtime_path, "intent_mojo_" + definition.__name__)
    source = runtime.prepare(context.target.resolve(), *arguments)
    report_stage("adapter_preparation")
    def benchmark_generated():
        elapsed = generated.benchmark()
        print(f"mojo: selected {generated.program.candidates[generated.winner]}; measured_ms={elapsed}; "
              f"candidate_ms={generated.program.timings[generated.key]}", flush=True)
        return elapsed
    return PreparedComparison(
        PreparedLaunch(generated.launch, generated.result, native_benchmark=benchmark_generated),
        PreparedLaunch(source.launch, source.result, native_benchmark=source.benchmark),
        tolerance, cuda_graph=False, device_type="cpu",
        note="同算法、f32、单 NUMA 8 核；native 执行计时含 packing/任务同步，不含输出分配。" + note,
    )


def prepare_host_comparison(context, definition, arguments, reference, tolerance, *, constexprs=None, note=""):
    artifact = _compile(context, definition, constexprs=constexprs)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function, program=None):
        state = {}

        def launch():
            state["output"] = function(*arguments)

        def outputs():
            if program is not None:
                for key, winner in program.winners.items():
                    print(f"mojo: selected {program.candidates[winner]}; candidate_ms={program.timings[key]}", flush=True)
            return state["output"]

        return PreparedLaunch(launch, outputs)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run, artifact.runtime), side(getattr(runtime, reference)), tolerance,
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 example 同算法、输入规模和外部 dtype；PyTorch eager CPU reference，单 NUMA 8 核；双方计完整 host 调用，含 ABI 处理、输出分配和任务同步。" + note,
    )


def prepare_host_run_only(context, definition, arguments, *, constexprs=None, note):
    artifact = _compile(context, definition, constexprs=constexprs)
    state = {}
    interface = artifact.interface
    mutable = {parameter.position: argument
               for parameter, argument in zip(interface.inputs, arguments, strict=True)
               if parameter in interface.mutable_inputs}
    observed = tuple(parameter for parameter in interface.views if parameter.writable)

    def launch():
        state["output"] = artifact.run(*arguments)

    def outputs():
        returned = state["output"]
        values = () if not interface.outputs else (returned,) if len(interface.outputs) == 1 else returned
        available = dict(mutable)
        available.update((parameter.position, value)
                         for parameter, value in zip(interface.outputs, values, strict=True))
        result = tuple(available[parameter.position] for parameter in observed)
        return result[0] if len(result) == 1 else result

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, outputs), None, None,
        cuda_graph=False, status="run_only", device_type="cpu", cpu_host_timing=True, note=note)
