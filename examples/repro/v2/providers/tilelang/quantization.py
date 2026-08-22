from __future__ import annotations

import torch

from kernels.quantization.fp8 import f32_groupwise_fp8_quantize

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module
from .. import implementation_gap


def per_token_fp8(context: Context) -> PreparedComparison:
    rows = features = 8192
    group_size = 128
    x = torch.randn((rows, features), device="cuda", dtype=torch.float32)
    generated_scales = torch.empty(
        (rows, features // group_size), device="cuda", dtype=torch.float32
    )
    _, generated_base = compile_single(
        context,
        f32_groupwise_fp8_quantize,
        (x, generated_scales),
    )
    generated = PreparedLaunch(
        generated_base.launch,
        lambda: (generated_base.outputs(), generated_scales),
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/quantization/per_token_fp8/example_per_token_cast_to_fp8.py",
        "intent_v2_tilelang_per_token_fp8",
    )
    source = functional_launch(
        lambda: tuple(source_module.per_token_cast_to_fp8(x, 8))
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=16.0, rtol=0.125),
            Tolerance(atol=1e-6, rtol=1e-4),
        ),
        cuda_graph=False,
    )


CASES = {
    "per_token_fp8": per_token_fp8,
    "block_fp4_quant": implementation_gap(
        "the source output is a packed float4_e2m1fn_x2 tensor with explicit "
        "nibble ordering; that storage contract is not a current Intent dtype"
    ),
}
