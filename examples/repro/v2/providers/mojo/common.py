from __future__ import annotations

import os
from pathlib import Path

import intent
import torch

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch


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


def prepare_comparison(context, definition, arguments, runtime_path, tolerance, note=""):
    report_stage("generated_compilation")
    artifact = intent.compile(definition, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    generated = artifact._namespace["native_program"].prepare(arguments)
    report_stage("source_compilation")
    runtime = load_module(context.project_root / runtime_path, "intent_mojo_" + definition.__name__)
    source = runtime.prepare(context.target.resolve(), *arguments)
    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(lambda: artifact(*generated.arguments), generated.result, native_benchmark=generated.benchmark),
        PreparedLaunch(source.launch, source.result, native_benchmark=source.benchmark),
        tolerance, cuda_graph=False, device_type="cpu",
        note="同算法、f32、单 NUMA 8 核；native 执行计时含 packing/任务同步，不含输出分配。" + note,
    )
