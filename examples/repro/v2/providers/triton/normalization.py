from __future__ import annotations

import torch

from kernels.activation.swiglu import swiglu_forward
from kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from kernels.normalization.layer_norm import layer_norm_f16
from kernels.normalization.rms_norm import rms_norm_bf16
from kernels.normalization.softmax import stable_softmax_f16

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance


def _runtime(context: Context, path: str, name: str):
    return load_module(context.project_root / path, name)


def fused_softmax(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, stable_softmax_f16, (x,))
    runtime = _runtime(
        context,
        "source/triton/triton/normalization/softmax/02-fused-softmax_runtime.py",
        "intent_v2_triton_fused_softmax",
    )
    source_function = runtime.load_softmax()
    source = functional_launch(lambda: source_function(x))
    return PreparedComparison(generated, source, Tolerance(atol=1e-2, rtol=1e-2), cuda_graph=True)


def layer_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.float16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    arguments = (x, weight, bias, 1.0 / hidden, 1e-5)
    _, generated = compile_single(context, layer_norm_f16, arguments)
    runtime = _runtime(
        context,
        "source/triton/triton/normalization/layer_norm/05-layer-norm_runtime.py",
        "intent_v2_triton_layer_norm",
    )
    source_function = runtime.load_layer_norm()
    source = functional_launch(
        lambda: source_function(x, (hidden,), weight, bias, 1e-5)
    )
    return PreparedComparison(generated, source, Tolerance(atol=1e-2), cuda_graph=True)


def swiglu(context: Context) -> PreparedComparison:
    shape = (8192, 14336)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(context, swiglu_forward, (gate, up))
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/activation/swiglu/swiglu_runtime.py",
        "intent_v2_triton_swiglu",
    )
    source = functional_launch(lambda: runtime.upstream((gate, up)))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


def fused_add_rms(context: Context) -> PreparedComparison:
    hidden = 4096
    shape = (8192, hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, residual, weight, 1.0 / hidden, 1e-6, 0.0)
    _, generated = compile_single(context, fused_add_rms_norm, arguments)
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm_runtime.py",
        "intent_v2_triton_fused_add_rms_norm",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


def rms_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, weight, 1.0 / hidden, 1e-6)
    _, generated = compile_single(context, rms_norm_bf16, arguments)
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/normalization/rms_norm/rms_norm_runtime.py",
        "intent_v2_triton_rms_norm",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


CASES = {
    "fused_softmax": fused_softmax,
    "layer_norm": layer_norm,
    "swiglu": swiglu,
    "fused_add_rms_norm": fused_add_rms,
    "rms_norm": rms_norm,
}

