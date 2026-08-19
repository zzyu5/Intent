from __future__ import annotations

import torch

from kernels.position.rope import rotary_qk_bf16_inplace

from ...measurement import compile_single
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import tilegym_source


def rope_qk(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, key_heads, dimension = 2, 4096, 32, 8, 128
    query = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cosine = torch.randn(
        (1, sequence, dimension), device="cuda", dtype=torch.bfloat16
    )
    sine = torch.randn_like(cosine)
    initial_query = query.clone()
    initial_key = key.clone()
    source_query = initial_query.clone()
    source_key = initial_key.clone()
    _, generated_base = compile_single(
        context,
        rotary_qk_bf16_inplace,
        (query, key, cosine, sine),
    )
    def generated_prepare():
        query.copy_(initial_query)
        key.copy_(initial_key)

    generated_prepare()
    generated = PreparedLaunch(
        generated_base.launch,
        lambda: (query, key),
        generated_prepare,
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/position/rope/rope.py",
        "rope_qk",
    )
    def source_prepare():
        source_query.copy_(initial_query)
        source_key.copy_(initial_key)

    def source_launch():
        source_module.apply_rope_base(source_query, source_key, cosine, sine)

    source_prepare()
    source_launch()
    source_prepare()
    source = PreparedLaunch(
        source_launch,
        lambda: (source_query, source_key),
        source_prepare,
    )
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-2, rtol=1e-2), Tolerance(atol=2e-2, rtol=1e-2)),
        cuda_graph=False,
    )


CASES = {"rope_qk": rope_qk}
