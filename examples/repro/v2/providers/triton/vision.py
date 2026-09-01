from __future__ import annotations

import torch

from kernels.vision.max_pool_with_indices import max_pool2d_with_indices

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance


def max_pool_with_indices(context: Context) -> PreparedComparison:
    x = torch.randn((8, 32, 128, 128), device="cuda", dtype=torch.float16)
    _, generated = compile_single(
        context,
        max_pool2d_with_indices,
        (x,),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/vision/max_pool2d/max_pool2d_with_indices_runtime.py",
        "intent_v2_triton_flaggems_max_pool_with_indices",
    )
    source = functional_launch(lambda: runtime.upstream((x,)))
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=0.0), Tolerance(atol=0.0)),
        cuda_graph=True,
    )


CASES = {"flaggems_max_pool2d_with_indices": max_pool_with_indices}
