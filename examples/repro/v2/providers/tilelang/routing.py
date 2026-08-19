from __future__ import annotations

import torch

from kernels.routing.mhc import mhc_apply_residual
from kernels.routing.mhc import mhc_pre_fuse
from kernels.routing.mhc import mhc_pre_gemm_sqrsum
from kernels.routing.mqa_logits import fp8_mqa_logits

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module


def mhc_pre(context: Context) -> PreparedComparison:
    tokens, hidden, streams = 2048, 4096, 4
    components = streams * (streams + 2)
    residual = (
        torch.randn(
            (tokens, streams, hidden), device="cuda", dtype=torch.float32
        )
        .mul(
            1
            + torch.arange(streams, device="cuda").mul(0.01).view(1, -1, 1)
        )
        .bfloat16()
    )
    weight = (
        torch.randn(
            (components, streams, hidden), device="cuda", dtype=torch.float32
        )
        * 1e-4
        * (
            1
            + torch.arange(streams, device="cuda").mul(0.01).view(1, -1, 1)
        )
    ).flatten(1, 2)
    scale = torch.randn((3,), device="cuda", dtype=torch.float32) * 0.1
    base = torch.randn(
        (components,), device="cuda", dtype=torch.float32
    ) * 0.1
    residual_flat = residual.view(tokens, streams * hidden)
    _, first = compile_single(
        context,
        mhc_pre_gemm_sqrsum,
        (residual_flat, weight),
    )
    mixes, square_sum = first.outputs()
    constexprs = {
        "RMS_EPS": 1e-6,
        "PRE_EPS": 1e-6,
        "SINKHORN_EPS": 1e-6,
        "POST_MULTIPLIER": 1.0,
        "SINKHORN_REPEATS": 10,
    }
    _, second = compile_single(
        context,
        mhc_pre_fuse,
        (mixes, square_sum, scale, base, residual),
        constexprs=constexprs,
    )

    def generated_launch():
        first.launch()
        second.launch()

    generated = PreparedLaunch(generated_launch, second.outputs)
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root / "source/tilelang/tilelang/mhc/pre/example_mhc_pre.py",
        "intent_v2_tilelang_mhc_pre",
    )
    source_mixes = torch.empty(
        (1, tokens, components), device="cuda", dtype=torch.float32
    )
    source_square_sum = torch.empty(
        (1, tokens), device="cuda", dtype=torch.float32
    )
    source_post = torch.empty(
        (tokens, streams), device="cuda", dtype=torch.float32
    )
    source_residual_mix = torch.empty(
        (tokens, streams * streams), device="cuda", dtype=torch.float32
    )
    source_layer_input = torch.empty(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16
    )

    def source_launch():
        source_module.mhc_pre_gemm_sqrsum_tilelang(
            residual_flat,
            weight,
            source_mixes.squeeze(0),
            source_square_sum.squeeze(0),
            components,
            streams * hidden,
        )
        source_module.mhc_pre_big_fuse_tilelang(
            source_mixes,
            source_square_sum,
            scale,
            base,
            residual,
            source_post,
            source_residual_mix,
            source_layer_input,
            hidden,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            10,
            1,
            streams,
        )

    source_launch()
    source = PreparedLaunch(
        source_launch,
        lambda: (
            source_post,
            source_residual_mix.view(tokens, streams, streams),
            source_layer_input,
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=1e-2, rtol=1e-3),
            Tolerance(atol=1e-2, rtol=1e-3),
            Tolerance(atol=5e-2, rtol=2e-2),
        ),
        cuda_graph=False,
    )


def mhc_post(context: Context) -> PreparedComparison:
    tokens, hidden, streams = 4096, 2560, 4
    layer_output = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(
        (tokens, streams, hidden), device="cuda", dtype=torch.bfloat16
    )
    post_mix = torch.randn(
        (tokens, streams), device="cuda", dtype=torch.float32
    )
    residual_mix = torch.randn(
        (tokens, streams, streams), device="cuda", dtype=torch.float32
    )
    _, generated = compile_single(
        context,
        mhc_apply_residual,
        (residual, layer_output, post_mix, residual_mix),
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root / "source/tilelang/tilelang/mhc/post/example_mhc_post.py",
        "intent_v2_tilelang_mhc_post",
    )
    source_output = torch.empty_like(residual)

    def source_launch():
        source_module.mhc_post_tilelang(
            residual_mix,
            residual,
            post_mix,
            layer_output,
            source_output,
            streams,
            hidden,
        )

    source_launch()
    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def fp8_lighting_indexer(context: Context) -> PreparedComparison:
    queries, keys, heads, dimension = 4096, 8192, 32, 64
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_file = (
        context.project_root
        / "source/tilelang/tilelang/attention/fp8_lighting_indexer/fp8_lighting_indexer.py"
    )
    source_module = support.load_source(
        source_file,
        "intent_v2_tilelang_fp8_lighting_indexer",
        aliases=(
            (
                "utils",
                context.project_root
                / "source/tilelang/tilelang/attention/support/deepseek_v32_utils.py",
            ),
        ),
    )
    q = torch.randn(
        (queries, heads, dimension), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    kv_source = torch.randn(
        (keys, dimension), device="cuda", dtype=torch.bfloat16
    )
    kv, kv_scale = source_module.per_custom_dims_cast_to_fp8(
        kv_source, (0,), False
    )
    weights = torch.randn(
        (queries, heads), device="cuda", dtype=torch.float32
    )
    key_start = torch.zeros((queries,), device="cuda", dtype=torch.int32)
    key_end = torch.full(
        (queries,), keys, device="cuda", dtype=torch.int32
    )
    _, generated = compile_single(
        context,
        fp8_mqa_logits,
        (q, kv, kv_scale, weights, key_start, key_end),
    )
    source = functional_launch(
        lambda: source_module.mqa_attn_return_logits_interface(
            q,
            kv,
            kv_scale,
            weights,
            key_start,
            key_end,
            clean_logits=False,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=False,
    )


CASES = {
    "mhc_pre": mhc_pre,
    "mhc_post": mhc_post,
    "fp8_lighting_indexer": fp8_lighting_indexer,
}
