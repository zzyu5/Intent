from __future__ import annotations

import torch

from kernels.streaming.ordered_prefix import row_cumsum_f32

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance


def cumsum(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float32)
    _, generated = compile_single(context, row_cumsum_f32, (x,))
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/scan/cumsum/cumsum_runtime.py",
        "intent_v2_triton_flaggems_cumsum",
    )
    source = functional_launch(lambda: runtime.upstream((x,)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-3, rtol=1e-5),
        cuda_graph=True,
    )


CASES = {"flaggems_cumsum": cumsum}
