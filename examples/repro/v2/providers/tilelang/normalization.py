from __future__ import annotations

import torch

from kernels.normalization.rms_norm import rms_norm_f32
from kernels.streaming.online_softmax import streamed_online_softmax_f16

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance
from .common import runtime_module
from .common import source_from_runtime


def online_softmax(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, streamed_online_softmax_f16, (x,))
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/normalization/online_softmax/online_softmax_runtime.py",
        "intent_v2_tilelang_online_softmax_runtime",
    )
    source_kernel = runtime.load_softmax(*x.shape)
    source = functional_launch(lambda: source_kernel(x))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=True,
    )


def rms_norm(context: Context) -> PreparedComparison:
    rows, hidden = 8192, 4096
    x = torch.randn((rows, hidden), device="cuda", dtype=torch.float32)
    _, generated = compile_single(
        context,
        rms_norm_f32,
        (x, 1.0 / hidden, 1e-5),
    )
    _, source_module = source_from_runtime(
        context,
        "source/tilelang/tilelang/normalization/rms_norm/rms_norm_runtime.py",
        "intent_v2_tilelang_rms_norm",
    )
    source_kernel = source_module.rms_norm.compile(M=rows, N=hidden, blk_m=1)
    source = functional_launch(lambda: source_kernel(x))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=True,
    )


CASES = {
    "online_softmax": online_softmax,
    "rms_norm": rms_norm,
}
