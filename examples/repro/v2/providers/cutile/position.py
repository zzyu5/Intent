from __future__ import annotations

import torch

from kernels.position.rope import rotary_qk_bf16_inplace

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
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
    source_query = query.clone()
    source_key = key.clone()
    _, generated = compile_single(
        context,
        rotary_qk_bf16_inplace,
        (query, key, cosine, sine),
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/position/rope/rope.py",
        "rope_qk",
    )
    source = functional_launch(
        lambda: source_module.apply_rope_base(
            source_query, source_key, cosine, sine
        )[:2]
    )
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-2, rtol=1e-2), Tolerance(atol=2e-2, rtol=1e-2)),
        cuda_graph=False,
    )


CASES = {"rope_qk": rope_qk}
