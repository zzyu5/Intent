from __future__ import annotations

import torch

from kernels.activation.pointwise import addcmul_broadcast_bf16

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance


def addcmul(context: Context) -> PreparedComparison:
    shape = (64, 128, 4096)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        addcmul_broadcast_bf16,
        (x, scale, bias),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
            )
            == 1
            and triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=3
            )
            == 512
            and config.num_warps == 4
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/pointwise/addcmul/addcmul_runtime.py",
        "intent_v2_triton_flaggems_addcmul",
    )
    source = functional_launch(lambda: runtime.upstream((x, scale, bias)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
    )


CASES = {"flaggems_addcmul": addcmul}
