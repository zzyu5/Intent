from __future__ import annotations

import torch

from kernels.streaming.mla import paged_mla_decode

from ...model import Context, PreparedComparison, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def paged_mla(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, query_heads, key_heads = 32, 128, 1
    sequence, value_dimension, rope_dimension = 8192, 128, 64
    page_size = 64
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn(
        (batch, query_heads, value_dimension), dtype=torch.float16
    )
    q_rope = torch.randn(
        (batch, query_heads, rope_dimension), dtype=torch.float16
    )
    latent_cache = torch.randn(
        (pages, page_size, key_heads, value_dimension), dtype=torch.float16
    )
    rope_cache = torch.randn(
        (pages, page_size, key_heads, rope_dimension), dtype=torch.float16
    )
    page_indices = torch.arange(pages, dtype=torch.int32)
    page_offsets = torch.arange(
        0,
        pages + 1,
        pages_per_sequence,
        dtype=torch.int32,
    )
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    scale = (value_dimension + rope_dimension) ** -0.5
    return prepare_host_comparison(
        context,
        paged_mla_decode,
        (
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            page_offsets,
            page_indices,
            lengths,
            scale,
        ),
        "paged_mla_decode",
        Tolerance(atol=5e-2, rtol=5e-2),
        constexprs={
            "PAGE_SIZE": page_size,
            "HEAD_GROUP": query_heads // key_heads,
        },
    )


CASES = {"paged_mla_decode": paged_mla}
