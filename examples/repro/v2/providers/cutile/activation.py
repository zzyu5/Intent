from __future__ import annotations

import torch

from kernels.activation.pointwise import geglu_tanh
from kernels.activation.pointwise import gelu_tanh
from kernels.activation.pointwise import relu_forward
from kernels.activation.swiglu import silu_and_mul_packed
from kernels.activation.swiglu import swiglu_forward

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import tilegym_source


def silu_and_mul(context: Context) -> PreparedComparison:
    packed = torch.randn((4096, 28672), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(context, silu_and_mul_packed, (packed,))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/activation/silu_and_mul/silu_and_mul.py",
        "silu_and_mul",
    )
    source = functional_launch(lambda: source_module.silu_and_mul(packed))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        status="source_ftz_approx_division_contract_gap",
    )


def swiglu(context: Context) -> PreparedComparison:
    shape = (8192, 14336)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    _, generated = compile_single(context, swiglu_forward, (gate, up))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/activation/swiglu/swiglu.py",
        "swiglu",
        needs_utils=True,
    )
    source = functional_launch(lambda: source_module.swiglu(gate, up))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        status="source_ftz_approx_division_contract_gap",
    )


def gelu(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, gelu_tanh, (x,))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/activation/fused/gelu.py",
        "gelu",
    )
    source = functional_launch(lambda: source_module.gelu(x, approximate="tanh"))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=True,
        # Source helper constants and intermediate arithmetic remain f16.
        status="source_f16_intermediates_contract_gap",
    )


def geglu(context: Context) -> PreparedComparison:
    x = torch.randn((4096, 28672), device="cuda", dtype=torch.float16)
    output = torch.empty((4096, 14336), device="cuda", dtype=torch.float16)
    _, generated_base = compile_single(context, geglu_tanh, (x, output))
    generated = PreparedLaunch(generated_base.launch, lambda: output)
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/activation/fused/geglu.py",
        "geglu",
        needs_gelu=True,
    )
    source = functional_launch(
        lambda: source_module.geglu(x, dim=-1, approximate="tanh")
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        status="source_f16_intermediates_contract_gap",
    )


def relu(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, relu_forward, (x,))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/activation/relu/relu.py",
        "relu",
    )
    source = functional_launch(lambda: source_module.relu(x))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=True,
        # cuda-tile maximum does not propagate NaNs; Intent maximum does.
        status="source_nonpropagating_maximum_contract_gap",
    )


CASES = {
    "silu_and_mul": silu_and_mul,
    "swiglu": swiglu,
    "gelu": gelu,
    "geglu": geglu,
    "relu": relu,
}
