from __future__ import annotations

import os

import torch

from kernels.quantization.fp8 import bf16_groupwise_fp8_quantize

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import TRITON_PARAMETER_REDUCTION
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def fp8_groupwise_quantize(context: Context) -> PreparedComparison:
    os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
    scales = torch.empty((8192, 32), device="cuda", dtype=torch.float32)
    _, generated_base = compile_single(
        context,
        bf16_groupwise_fp8_quantize,
        (x, scales),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=3
            )
            == 128
            and triton_parameter_value(
                config, TRITON_PARAMETER_REDUCTION, dimension=3
            )
            == 128
            and config.num_warps == 4
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (generated_base.outputs(), scales),
    )
    runtime_helper = load_module(
        context.project_root
        / "source/triton/meta-applied-ai/support/runtime.py",
        "intent_v2_triton_meta_runtime_quantize",
    )
    source_module = runtime_helper.load_source(
        context.project_root
        / "source/triton/meta-applied-ai/quantization/fp8_groupwise/float8_groupwise_quant.py",
        "intent_v2_triton_fp8_groupwise_quantize",
    )
    source = functional_launch(lambda: source_module.float8_groupwise_quantize(x, 128))
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=16.0, rtol=0.125),
            Tolerance(atol=1e-6, rtol=1e-4),
        ),
        cuda_graph=True,
    )


CASES = {
    "fp8_groupwise_quantize": fp8_groupwise_quantize,
}
