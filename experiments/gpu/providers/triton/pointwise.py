from __future__ import annotations

import torch

from kernels.activation.pointwise import addcmul_broadcast_bf16

from experiments._common.loading import load_module
from experiments._common.measurement import compile_single
from experiments._common.measurement import functional_launch
from experiments._common.model import Context
from experiments._common.model import PreparedComparison
from experiments._common.model import Tolerance


def addcmul(context: Context) -> PreparedComparison:
    shape = (64, 128, 4096)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        addcmul_broadcast_bf16,
        (x, scale, bias),
    )
    runtime = load_module(
        context.project_root
        / "experiments/gpu/baselines/triton/flag-gems/pointwise/addcmul/addcmul_runtime.py",
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
