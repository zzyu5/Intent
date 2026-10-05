from dataclasses import replace

import torch

from kernels.activation.pointwise import relu_forward
from kernels.contraction.gemm import Activation, gemm
from kernels.normalization.softmax import stable_softmax_f16
from experiments._common.loading import load_module
from experiments._common.model import Tolerance
from .attention import CASES as ATTENTION_CASES
from .common import RemoteSequence, tilegym_source
from .indexed import CASES as INDEXED_CASES
from .mamba import CASES as MAMBA_CASES
from .streaming import CASES as STREAMING_CASES


def relu(context):
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
    sequence = RemoteSequence(context)
    output = sequence.add(relu_forward, {"x": x})["output"]
    source = tilegym_source(context, "experiments/gpu/baselines/cutile/tilegym/activation/relu/relu.py", "relu")
    return sequence.comparison(output, lambda: source.relu(x), Tolerance(atol=0.0), source=("relu", (x,)))


def fused_softmax(context):
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    sequence = RemoteSequence(context)
    output = sequence.add(stable_softmax_f16, {"x": x}, target=replace(context.target, tile=8192))["y"]
    runtime = load_module(context.project_root / "experiments/gpu/baselines/triton/triton/normalization/softmax/02-fused-softmax_runtime.py",
        "intent_bangc_source_softmax")
    source = runtime.load_softmax()
    return sequence.comparison(output, lambda: source(x), Tolerance(atol=1e-2, rtol=1e-2), source=("softmax", (x,)))


def dense_gemm(context):
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn((4096, 14336), device="cuda", dtype=torch.float16)
    sequence = RemoteSequence(context)
    output = sequence.add(gemm, {"a": a, "b": b}, constexprs={"ACTIVATION": Activation.NONE})["c"]
    runtime = load_module(context.project_root / "experiments/gpu/baselines/triton/triton/gemm/dense/03-matrix-multiplication_runtime.py",
        "intent_bangc_source_gemm")
    source = runtime.load_matmul()
    return sequence.comparison(output, lambda: source(a, b), Tolerance(atol=1e-2, rtol=1e-2), source=("matmul", (a, b)))


CASES = {"relu": relu, "fused_softmax": fused_softmax, "dense_gemm": dense_gemm,
         **STREAMING_CASES, **INDEXED_CASES, **MAMBA_CASES, **ATTENTION_CASES}
