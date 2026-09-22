from __future__ import annotations

import intent
import torch

from kernels.compaction.unique_consecutive import (
    ROWS,
    VALUES,
    unique_consecutive_rows,
)

from ...loading import load_module
from ...measurement import report_stage
from ...model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget


def unique_consecutive(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    increments = torch.randint(
        0, 5, (ROWS, VALUES), dtype=torch.int32
    )
    values = torch.cumsum(increments == 0, dim=1, dtype=torch.int32)
    generated_values = values.clone()
    source_values = values.clone()
    generated_lengths = torch.zeros_like(generated_values)
    source_lengths = torch.zeros_like(source_values)

    report_stage("generated_compilation")
    artifact = intent.compile(
        unique_consecutive_rows,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_unique_consecutive",
    )
    generated_state = {}
    source_state = {}

    def generated_launch():
        generated_state["output"] = artifact.run(
            generated_values, generated_lengths
        )

    def generated_outputs():
        unique_values, returned_lengths, inverse, counts = generated_state["output"]
        return unique_values, inverse, counts, returned_lengths

    def source_launch():
        source_state["output"] = runtime.unique_consecutive(
            source_values, source_lengths
        )

    def source_outputs():
        return (*source_state["output"], source_lengths)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            generated_outputs,
            prepare=lambda: generated_lengths.zero_(),
        ),
        PreparedLaunch(
            source_launch,
            source_outputs,
            prepare=lambda: source_lengths.zero_(),
        ),
        tuple(Tolerance(atol=0.0) for _ in range(4)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 unique-consecutive row-wise run-length encoding；输入/输出均 i32，生成端与 CPU reference 使用独立 InOut run_lengths。",
    )


CASES = {"unique_consecutive": unique_consecutive}
