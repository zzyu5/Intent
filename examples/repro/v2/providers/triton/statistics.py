from __future__ import annotations

import torch

from kernels.statistics.histogram import histogram_256

from ...loading import load_module
from ...measurement import compile_single
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def histogram(context: Context) -> PreparedComparison:
    samples = torch.randint(
        0,
        256,
        (8 * 1024 * 1024,),
        device="cuda",
        dtype=torch.int32,
    ).to(torch.float32)
    _, generated_base = compile_single(
        context,
        histogram_256,
        (samples,),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=generated_base.outputs,
        prepare=generated_base.outputs().zero_,
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/statistics/histogram/histc_runtime.py",
        "intent_v2_triton_flaggems_histogram",
    )
    source_histogram = torch.zeros_like(generated_base.outputs())

    def source_launch():
        runtime.MODULE.histc_kernel_simple[
            (runtime.MODULE.triton.cdiv(samples.numel(), 1024),)
        ](
            samples,
            source_histogram,
            samples.numel(),
            256,
            0.0,
            256.0,
            BLOCK_SIZE=1024,
        )

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: source_histogram,
        prepare=source_histogram.zero_,
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=False,
    )


CASES = {"flaggems_histogram": histogram}
