from __future__ import annotations

import torch

from kernels.optimization.adamw import adamw_update

from ...loading import load_module
from ...measurement import compile_single
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def fused_adamw(context: Context) -> PreparedComparison:
    elements = 8 * 1024 * 1024
    gradient = torch.randn((elements,), device="cuda", dtype=torch.float32)
    initial_parameter = torch.randn_like(gradient)
    initial_first = torch.randn_like(gradient)
    initial_second = torch.rand_like(gradient) + 0.5
    generated_parameter = initial_parameter.clone()
    generated_first = initial_first.clone()
    generated_second = initial_second.clone()
    scalars = (1e-3, 0.9, 0.999, 1.0 - 0.9**10, 1.0 - 0.999**10, 1e-8, 0.01)
    _, generated_base = compile_single(
        context,
        adamw_update,
        (
            gradient,
            generated_parameter,
            generated_first,
            generated_second,
            *scalars,
        ),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (generated_parameter, generated_first, generated_second),
        prepare=lambda: (
            generated_parameter.copy_(initial_parameter),
            generated_first.copy_(initial_first),
            generated_second.copy_(initial_second),
        ),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/optimization/adamw/fused_adam_runtime.py",
        "intent_v2_triton_flaggems_adamw",
    )
    source_parameter = initial_parameter.clone()
    source_first = initial_first.clone()
    source_second = initial_second.clone()
    source_arguments = (
        gradient,
        source_parameter,
        source_first,
        source_second,
        *scalars,
    )
    source = PreparedLaunch(
        launch=lambda: runtime.upstream(source_arguments),
        outputs=lambda: (source_parameter, source_first, source_second),
        prepare=lambda: (
            source_parameter.copy_(initial_parameter),
            source_first.copy_(initial_first),
            source_second.copy_(initial_second),
        ),
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-5, rtol=1e-5),
        cuda_graph=False,
    )


CASES = {"flaggems_fused_adamw": fused_adamw}
