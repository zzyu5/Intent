from __future__ import annotations

import torch

from kernels.routing.mhc import mhc_apply_residual
from kernels.routing.mhc import mhc_gemm_rms_scale
from kernels.routing.mhc import mhc_sinkhorn

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import tilegym_source


def _source(context: Context):
    return tilegym_source(
        context,
        "source/cutile/tilegym/mhc/fused/mhc.py",
        "mhc",
    )


def gemm_rms_scale(context: Context) -> PreparedComparison:
    tokens, hidden, streams = 2048, 4096, 4
    x = torch.randn(
        (tokens, streams * hidden), device="cuda", dtype=torch.bfloat16
    )
    width = streams * (streams + 2)
    weight = torch.randn(
        (streams * hidden, width), device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        mhc_gemm_rms_scale,
        (x, weight, bias),
        constexprs={
            "STREAMS": streams,
            "ALPHA_PRE": 1.0,
            "ALPHA_POST": 1.0,
            "ALPHA_RESIDUAL": 1.0,
        },
    )
    source_module = _source(context)
    config = {
        "TILE_SIZE_M": 64,
        "TILE_SIZE_N": 32,
        "TILE_SIZE_K": 64,
        "SPLIT_K": 4,
        "GROUP_SIZE_M": 8,
    }
    source = functional_launch(
        lambda: source_module.mhc_gemm_rms_scale(
            x,
            weight,
            streams,
            1.0,
            1.0,
            1.0,
            bias,
            cfg=config,
        )
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=5e-2, rtol=2e-2),
            Tolerance(atol=5e-3, rtol=1e-3),
        ),
        cuda_graph=False,
    )


def apply_residual(context: Context) -> PreparedComparison:
    tokens, hidden, streams = 2048, 4096, 4
    residual = torch.randn(
        (tokens, streams, hidden), device="cuda", dtype=torch.bfloat16
    )
    layer_output = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16
    )
    post_bf16 = torch.randn(
        (tokens, streams), device="cuda", dtype=torch.bfloat16
    )
    residual_bf16 = torch.randn(
        (tokens, streams, streams), device="cuda", dtype=torch.bfloat16
    )
    post_mix = post_bf16.float()
    residual_mix = residual_bf16.float()
    _, generated = compile_single(
        context,
        mhc_apply_residual,
        (residual, layer_output, post_mix, residual_mix),
    )
    packed = torch.zeros(
        (tokens, streams * (streams + 2)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    packed[:, streams : 2 * streams] = post_bf16
    packed[:, 2 * streams :] = residual_bf16.view(tokens, -1)
    source_module = _source(context)
    source = functional_launch(
        lambda: source_module.mhc_apply_residual(
            residual.view(tokens, streams * hidden),
            layer_output,
            packed,
            streams,
        ).view(tokens, streams, hidden)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=True,
    )


def sinkhorn(context: Context) -> PreparedComparison:
    tokens, streams = 8192, 4
    logits = torch.randn(
        (tokens, streams, streams), device="cuda", dtype=torch.float32
    )
    _, generated = compile_single(context, mhc_sinkhorn, (logits,))
    source_module = _source(context)
    packed = torch.zeros(
        (tokens, streams * (streams + 2)), device="cuda", dtype=torch.float32
    )

    def prepare():
        packed[:, 2 * streams :].copy_(logits.view(tokens, -1))

    def launch():
        source_module.mhc_sinkhorn(packed, streams)

    prepare()
    launch()
    source = PreparedLaunch(
        launch=launch,
        outputs=lambda: packed[:, 2 * streams :].view(tokens, streams, streams),
        prepare=prepare,
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-4, rtol=1e-4),
        cuda_graph=False,
    )


CASES = {
    "mhc_gemm_rms_scale": gemm_rms_scale,
    "mhc_apply_residual": apply_residual,
    "mhc_sinkhorn": sinkhorn,
}
