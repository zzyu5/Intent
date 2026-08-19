from __future__ import annotations

import math

import torch

from kernels.streaming.linear_attention import chunk_retention_fwd
from kernels.streaming.linear_attention import fused_chunk_linear_attention_fwd
from kernels.streaming.mamba import mamba_chunk_state_fwd
from kernels.streaming.selective_scan import mamba_chunk_scan_fwd

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module


def mamba_chunk_scan(context: Context) -> PreparedComparison:
    batch, sequence, heads, groups, dimension, state, chunk = 1, 2048, 32, 8, 64, 128, 256
    chunks = sequence // chunk
    cb = torch.randn(
        (batch, chunks, groups, chunk, chunk),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    x = torch.randn(
        (batch, sequence, heads, dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    dt = torch.rand(
        (batch, heads, chunks, chunk), device="cuda", dtype=torch.float16
    ) * 0.1
    cumulative_decay = -torch.rand_like(dt) * 0.1
    state_matrix = torch.randn(
        (batch, sequence, groups, state), device="cuda", dtype=torch.float16
    ) * 0.05
    previous_states = torch.randn(
        (batch, chunks, heads, dimension, state),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    residual_scale = torch.randn(
        (heads,), device="cuda", dtype=torch.float16
    ) * 0.1
    arguments = (
        cb,
        x,
        dt,
        cumulative_decay,
        state_matrix,
        previous_states,
        residual_scale,
    )
    _, generated = compile_single(
        context,
        mamba_chunk_scan_fwd,
        arguments,
        constexprs={"HEADS_PER_GROUP": heads // groups},
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/scan/mamba_chunk_scan/example_mamba_chunk_scan_runtime.py",
        "intent_v2_tilelang_mamba_chunk_scan_runtime",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


def mamba_chunk_state(context: Context) -> PreparedComparison:
    batch, sequence, heads, groups, dimension, state, chunk = 1, 2048, 32, 8, 64, 128, 256
    chunks = sequence // chunk
    state_basis = torch.randn(
        (batch, sequence, groups, state), device="cuda", dtype=torch.float16
    )
    x = torch.randn(
        (batch, sequence, heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    dt = torch.randn(
        (batch, heads, chunks, chunk), device="cuda", dtype=torch.float16
    )
    cumulative_decay = torch.cumsum(
        -torch.rand_like(dt) * 0.1,
        dim=-1,
    )
    _, generated = compile_single(
        context,
        mamba_chunk_state_fwd,
        (state_basis, x, dt, cumulative_decay),
        constexprs={"HEAD_GROUP": heads // groups},
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/scan/mamba_chunk_state/example_mamba_chunk_state.py",
        "intent_v2_tilelang_mamba_chunk_state",
    )
    compiled = source_module.chunk_state_fwd.compile(
        batch=batch,
        seqlen=sequence,
        chunk_size=chunk,
        ngroups=groups,
        nheads=heads,
        headdim=dimension,
        dstate=state,
        block_M=64,
        block_N=128,
        block_K=64,
        num_stages=4,
        threads=128,
    )
    executable = compiled.adapter._get_executable()
    source_output = torch.empty(
        (batch, chunks, heads, dimension, state),
        device="cuda",
        dtype=torch.float16,
    )

    def source_launch():
        executable(state_basis, x, dt, cumulative_decay, source_output)

    source_launch()
    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=True,
    )


def linear_attention(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 1, 2048, 16, 128
    q = torch.randn(
        (batch, sequence, heads, dimension), device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context, fused_chunk_linear_attention_fwd, (q, k, v, scale)
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/linear_attention/fused_chunk_forward/example_linear_attn_fwd.py",
        "intent_v2_tilelang_linear_attention",
        needs_fla_linear=True,
    )
    source_kernel = source_module.tl_fused_chunk_fwd_kernel(
        batch, sequence, heads, dimension, dimension
    )
    source_output = torch.zeros(
        (batch, sequence, heads, dimension), device="cuda", dtype=torch.float32
    )
    state: dict[str, torch.Tensor] = {}

    def prepare():
        source_output.zero_()

    def source_launch():
        state["final"] = source_kernel(q, k, v, source_output)

    prepare()
    source_launch()
    source = PreparedLaunch(
        source_launch,
        lambda: (source_output, state["final"]),
        prepare=prepare,
    )
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=1e-1, rtol=5e-2), Tolerance(atol=1e-1, rtol=5e-2)),
        cuda_graph=False,
    )


def retention(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 1, 2048, 16, 128
    q = torch.randn(
        (batch, sequence, heads, dimension), device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(context, chunk_retention_fwd, (q, k, v, scale))
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/linear_attention/retention/example_retention_fwd.py",
        "intent_v2_tilelang_retention",
    )
    source_kernel = source_module.chunk_retention_fwd_kernel(
        batch, sequence, heads, dimension, dimension
    )
    source = functional_launch(
        lambda: source_module.postprocess(source_kernel(q, k, v))
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


CASES = {
    "mamba_chunk_scan": mamba_chunk_scan,
    "mamba_chunk_state": mamba_chunk_state,
    "linear_attention_forward": linear_attention,
    "retention_forward": retention,
}
