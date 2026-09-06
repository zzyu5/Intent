from __future__ import annotations

import cuda.tile as ct
import torch

from kernels.routing.mhc import mhc_apply_residual
from kernels.routing.mhc import mhc_gemm_rms_finalize
from kernels.routing.mhc import mhc_gemm_rms_partial
from kernels.routing.mhc import mhc_sinkhorn

from ...measurement import compile_single
from ...measurement import initial_launch
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
    split_count = 16
    _, partial = compile_single(
        context,
        mhc_gemm_rms_partial,
        (x, weight),
        constexprs={"P": split_count},
    )
    partial_linear, partial_square_sum = partial.outputs()
    _, finalize = compile_single(
        context,
        mhc_gemm_rms_finalize,
        (partial_linear, partial_square_sum, bias, x.shape[1]),
        constexprs={
            "P": split_count,
            "STREAMS": streams,
            "ALPHA_PRE": 1.0,
            "ALPHA_POST": 1.0,
            "ALPHA_RESIDUAL": 1.0,
        },
    )

    def generated_launch():
        partial.launch()
        finalize.launch()

    generated = PreparedLaunch(generated_launch, finalize.outputs)
    source_module = _source(context)
    config = {
        "TILE_SIZE_M": 64,
        "TILE_SIZE_N": 32,
        "TILE_SIZE_K": 64,
        "SPLIT_K": split_count,
        "GROUP_SIZE_M": 8,
    }
    tile_m = config["TILE_SIZE_M"]
    tile_n = config["TILE_SIZE_N"]
    tile_k = config["TILE_SIZE_K"]
    group_m = config["GROUP_SIZE_M"]
    row_tiles = (tokens + tile_m - 1) // tile_m
    column_tiles = (width + tile_n - 1) // tile_n
    source_partial_linear = torch.empty(
        (tokens * split_count, width), device="cuda", dtype=torch.float32
    )
    source_partial_square_sum = torch.empty(
        (tokens * split_count, column_tiles),
        device="cuda",
        dtype=torch.float32,
    )
    source_mixed = torch.empty(
        (tokens, width), device="cuda", dtype=torch.bfloat16
    )
    source_rms = torch.empty(
        (tokens, 1), device="cuda", dtype=torch.float32
    )

    def source_launch():
        ct.launch(
            torch.cuda.current_stream(),
            (row_tiles * column_tiles, split_count, 1),
            source_module._mhc_split_gemm_rms_kernel,
            (
                x,
                weight,
                source_partial_linear,
                source_partial_square_sum,
                tokens,
                width,
                x.shape[1],
                tile_m,
                tile_n,
                tile_k,
                split_count,
                group_m,
            ),
        )
        ct.launch(
            torch.cuda.current_stream(),
            (row_tiles, column_tiles, 1),
            source_module._mhc_finalize_scale_bias_sigmoid_kernel,
            (
                source_partial_linear,
                source_partial_square_sum,
                source_mixed,
                source_rms,
                streams,
                1.0,
                1.0,
                1.0,
                bias,
                tokens,
                width,
                x.shape[1],
                tile_m,
                tile_n,
                split_count,
            ),
        )

    initial_launch(source_launch, side="source")
    source = PreparedLaunch(
        source_launch,
        lambda: (source_mixed, source_rms),
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
    generated_values = logits.clone()
    _, generated_call = compile_single(context, mhc_sinkhorn, (generated_values,))

    def prepare_generated():
        generated_values.copy_(logits)

    generated = PreparedLaunch(
        launch=generated_call.launch,
        outputs=lambda: generated_values,
        prepare=prepare_generated,
    )
    source_module = _source(context)
    packed = torch.zeros(
        (tokens, streams * (streams + 2)), device="cuda", dtype=torch.float32
    )

    def prepare():
        packed[:, 2 * streams :].copy_(logits.view(tokens, -1))

    def launch():
        source_module.mhc_sinkhorn(packed, streams)

    prepare()
    initial_launch(launch, side="source")
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
