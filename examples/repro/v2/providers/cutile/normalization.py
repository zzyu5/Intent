from __future__ import annotations

import torch

from kernels.normalization.layer_norm import layer_norm_bf16
from kernels.normalization.rms_norm import rms_norm_bf16
from kernels.normalization.softmax import chunked_softmax_bf16

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance
from .common import official_source
from .common import tilegym_source


def layer_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        layer_norm_bf16,
        (x, weight, bias, 1.0 / hidden, 1e-5),
    )
    source_module = official_source(
        context,
        "source/cutile/cutile-python/normalization/layer_norm/LayerNorm_runtime.py",
        "intent_v2_cutile_layer_norm",
    )
    source = functional_launch(
        lambda: source_module.cutile_layer_norm(x, weight, bias, 1e-5)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        # Source uses reciprocal(sqrt) and also writes Mean/Rstd for backward.
        note="同算法；source 额外写 Mean/Rstd",
    )


def chunked_softmax(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 32768), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(context, chunked_softmax_bf16, (x,))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/normalization/softmax/softmax.py",
        "chunked_softmax",
    )
    source = functional_launch(
        lambda: source_module.softmax(x, use_tma=False, use_chunked=True)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        # Source is a max/denominator/output three-pass algorithm, not online merge.
        note="source 使用三遍 softmax",
    )


def rms_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        rms_norm_bf16,
        (x, weight, 1.0 / hidden, 1e-5),
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/normalization/rms_norm/rms_norm.py",
        "rms_norm",
        needs_utils=True,
    )
    source = functional_launch(
        lambda: source_module.rms_norm(
            x,
            (hidden,),
            weight,
            1e-5,
            mode="multi_wave_reload",
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
        # The selected source forward stores Rstd, absent from the Intent ABI.
        note="同算法；source 额外写 Rstd",
    )


CASES = {
    "layer_norm": layer_norm,
    "chunked_softmax": chunked_softmax,
    "rms_norm": rms_norm,
}
