from __future__ import annotations

import torch

from kernels.quantization.fp8 import f32_groupwise_fp8_quantize

from experiments._common.measurement import compile_single
from experiments._common.measurement import functional_launch
from experiments._common.model import Context
from experiments._common.model import PreparedComparison
from experiments._common.model import PreparedLaunch
from experiments._common.model import Tolerance
from .common import runtime_module
from experiments._common.providers import implementation_gap


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
        "experiments/gpu/baselines/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "experiments/gpu/baselines/tilelang/tilelang/quantization/per_token_fp8/example_per_token_cast_to_fp8.py",
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
        "the source contract converts each bf16 group of 32 with a power-of-two "
        "absmax scale, rounds to E2M1, and stores even logical elements in the "
        "low nibble and odd elements in the high nibble; Intent has no typed "
        "E2M1 conversion and packed-nibble output ABI"
    ),
}
