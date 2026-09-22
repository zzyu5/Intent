from __future__ import annotations

import torch

from kernels.convolution.direct import conv2d_nhwc

from experiments._common.measurement import compile_single
from experiments._common.measurement import functional_launch
from experiments._common.model import Context
from experiments._common.model import PreparedComparison
from experiments._common.model import Tolerance
from .common import source_from_runtime


def conv2d(context: Context) -> PreparedComparison:
    data = torch.randn(
        (32, 128, 128, 256), device="cuda", dtype=torch.float16
    )
    weight = torch.randn((3, 3, 256, 512), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, conv2d_nhwc, (data, weight))
    _, source_module = source_from_runtime(
        context,
        "experiments/gpu/baselines/tilelang/tilelang/convolution/basic/example_convolution_runtime.py",
        "intent_v2_tilelang_conv2d",
    )
    source = functional_launch(
        lambda: source_module.convolution(
            data,
            weight,
            1,
            1,
            1,
            64,
            128,
            32,
            3,
            256,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=2e-2),
        cuda_graph=True,
    )


CASES = {"conv2d": conv2d}
