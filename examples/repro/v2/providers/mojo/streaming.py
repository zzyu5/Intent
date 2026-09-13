from __future__ import annotations

import math

import torch

from kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from kernels.streaming.online_softmax import streamed_online_softmax_f16
from kernels.streaming.selective_scan import (
    BATCH as SELECTIVE_SCAN_BATCH,
    LENGTH as SELECTIVE_SCAN_LENGTH,
    mamba_chunk_scan_fwd,
    selective_state_scan,
)

from ...model import Context, PreparedComparison, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def selective_scan(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    x = torch.randn(
        (SELECTIVE_SCAN_BATCH, SELECTIVE_SCAN_LENGTH), dtype=torch.float32
    ) * 0.05
    decay = 0.9 + 0.09 * torch.rand_like(x)
    drive = torch.randn_like(x) * 0.05
    return prepare_host_comparison(
        context,
        selective_state_scan,
        (x, decay, drive),
        "selective_state_scan",
        Tolerance(atol=2.0e-5),
    )


def online_softmax(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    x = torch.randn((8192, 8192), dtype=torch.float16)
    return prepare_host_comparison(
        context,
        streamed_online_softmax_f16,
        (x,),
        "streamed_online_softmax_f16",
        Tolerance(atol=1.0e-2, rtol=1.0e-2),
    )


def recurrent_gated_delta(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, sequence, heads, key_dimension, value_dimension = 2, 2048, 8, 128, 128
    query = torch.randn(
        (batch, sequence, heads, key_dimension), dtype=torch.bfloat16
    ) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(
        (batch, sequence, heads, value_dimension), dtype=torch.bfloat16
    ) * 0.1
    gate = -torch.rand(
        (batch, sequence, heads), dtype=torch.bfloat16
    ) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    scale = 1.0 / math.sqrt(key_dimension)
    return prepare_host_comparison(
        context,
        recurrent_gated_delta_fwd,
        (query, key, value, gate, beta, scale),
        "recurrent_gated_delta_fwd",
        (
            Tolerance(atol=1.0e-1, rtol=5.0e-2),
            Tolerance(atol=1.0e-1, rtol=5.0e-2),
        ),
        constexprs={"HEAD_GROUP": 1},
    )


def mamba_chunk_scan(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, sequence, heads, groups, dimension, state, chunk = (
        1, 2048, 32, 8, 64, 128, 256
    )
    chunks = sequence // chunk
    cb = torch.randn(
        (
            batch,
            chunks,
            groups,
            chunk,
            chunk,
        ),
        dtype=torch.float16,
    ) * 0.05
    x = torch.randn(
        (batch, sequence, heads, dimension),
        dtype=torch.float16,
    ) * 0.1
    dt = torch.rand(
        (batch, heads, chunks, chunk),
        dtype=torch.float16,
    ) * 0.1
    dA = -torch.rand_like(dt) * 0.1
    state_matrix = torch.randn(
        (batch, sequence, groups, state),
        dtype=torch.float16,
    ) * 0.05
    previous = torch.randn(
        (
            batch,
            chunks,
            heads,
            dimension,
            state,
        ),
        dtype=torch.float16,
    ) * 0.05
    residual_scale = torch.randn(
        (heads,), dtype=torch.float16
    ) * 0.1
    return prepare_host_comparison(
        context,
        mamba_chunk_scan_fwd,
        (cb, x, dt, dA, state_matrix, previous, residual_scale),
        "mamba_chunk_scan_fwd",
        Tolerance(atol=1.0e-1, rtol=5.0e-2),
        constexprs={"HEADS_PER_GROUP": heads // groups},
    )


CASES = {
    "selective_state_scan": selective_scan,
    "streamed_online_softmax_f16": online_softmax,
    "recurrent_gated_delta_fwd": recurrent_gated_delta,
    "mamba_chunk_scan": mamba_chunk_scan,
}
