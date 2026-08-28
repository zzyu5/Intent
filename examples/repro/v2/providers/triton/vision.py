from __future__ import annotations

import torch

from kernels.vision.max_pool_with_indices import max_pool2d_with_indices

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance


def max_pool_with_indices(context: Context) -> PreparedComparison:
    x = torch.randn((8, 32, 128, 128), device="cuda", dtype=torch.float16)
    source_configs = {
        (16, 16, 4, 4),
        (32, 16, 4, 3),
        (16, 32, 4, 3),
        (32, 32, 8, 2),
        (8, 8, 2, 5),
        (16, 8, 2, 5),
        (8, 16, 2, 5),
        (64, 16, 8, 2),
        (16, 64, 8, 2),
        (32, 64, 8, 3),
        (64, 32, 8, 3),
        (64, 64, 8, 2),
    }

    def source_config(config) -> bool:
        row = triton_parameter_value(
            config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
        )
        column = triton_parameter_value(
            config, TRITON_PARAMETER_OWNERSHIP_N, dimension=2
        )
        return (
            row is not None
            and column is not None
            and (row, column, config.num_warps, config.num_stages)
            in source_configs
            and config.num_ctas == 1
        )

    _, generated = compile_single(
        context,
        max_pool2d_with_indices,
        (x,),
        triton_config_filter=source_config,
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
