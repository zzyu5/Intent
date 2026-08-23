from __future__ import annotations

import importlib

import torch

from kernels.streaming.mamba import mamba3_siso_step
from kernels.streaming.mamba import mamba3_siso_forward as mamba3_siso_forward_kernel
from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from kernels.streaming.mamba import mamba_state_passing_fwd
from kernels.streaming.selective_scan import mamba_chunk_scan_bf16_fwd

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .. import implementation_gap


def _activate(context: Context, runtime_path: str, name: str):
    return load_module(context.project_root / runtime_path, name)


def mamba3_step(context: Context) -> PreparedComparison:
    batch, qk_heads, heads = 32, 4, 16
    qk_dimension, value_dimension, angle_dimension = 32, 64, 16
    dtype = torch.bfloat16
    query = torch.randn(
        (batch, qk_heads, qk_dimension), device="cuda", dtype=dtype
    ) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(
        (batch, heads, value_dimension), device="cuda", dtype=dtype
    ) * 0.1
    adt = -torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    dt = torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    trap = torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    query_bias = torch.randn(
        (heads, qk_dimension), device="cuda", dtype=dtype
    ) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn(
        (batch, heads, angle_dimension), device="cuda", dtype=torch.float32
    ) * 0.01
    residual = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.01
    gate = torch.randn(
        (batch, heads, value_dimension), device="cuda", dtype=dtype
    ) * 0.1
    states = (
        torch.zeros(
            (batch, heads, angle_dimension), device="cuda", dtype=torch.float32
        ),
        torch.zeros(
            (batch, heads, value_dimension, qk_dimension),
            device="cuda",
            dtype=torch.float32,
        ),
        torch.zeros(
            (batch, heads, qk_dimension), device="cuda", dtype=torch.float32
        ),
        torch.zeros(
            (batch, heads, value_dimension), device="cuda", dtype=torch.float32
        ),
    )
    arguments = (
        query,
        key,
        value,
        adt,
        dt,
        trap,
        query_bias,
        key_bias,
        angles,
        residual,
        gate,
        *states,
    )
    _, generated = compile_single(
        context,
        mamba3_siso_step,
        arguments,
        constexprs={"HEAD_GROUP": heads // qk_heads},
    )
    _activate(
        context,
        "source/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_step_runtime.py",
        "intent_v2_triton_mamba3_runtime",
    )
    source_function = importlib.import_module(
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_step"
    ).mamba3_siso_step
    source_base = functional_launch(
        lambda: source_function(
            query,
            key,
            value,
            adt,
            dt,
            trap,
            query_bias,
            key_bias,
            angles,
            D=residual,
            Z=gate,
            Input_States=states,
        )
    )
    source = PreparedLaunch(
        launch=source_base.launch,
        outputs=lambda: (
            source_base.outputs()[0],
            source_base.outputs()[1][0],
            source_base.outputs()[1][1],
            source_base.outputs()[1][2],
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=5e-3, rtol=5e-2),
            Tolerance(atol=1e-5, rtol=1e-5),
            Tolerance(atol=1e-4, rtol=1e-4),
            Tolerance(atol=5e-3, rtol=5e-2),
        ),
        cuda_graph=False,
    )


def mamba3_siso_forward(context: Context) -> PreparedComparison:
    batch, sequence, qk_heads, heads = 1, 2048, 4, 16
    qk_dimension, value_dimension, angle_dimension = 32, 64, 16
    dtype = torch.bfloat16
    query = torch.randn(
        (batch, sequence, qk_heads, qk_dimension), device="cuda", dtype=dtype
    ) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(
        (batch, sequence, heads, value_dimension), device="cuda", dtype=dtype
    ) * 0.1
    adt = -torch.rand(
        (batch, heads, sequence), device="cuda", dtype=torch.float32
    ) * 0.1
    dt = torch.rand_like(adt) * 0.1
    trap = torch.rand_like(adt) * 0.1
    query_bias = torch.randn(
        (heads, qk_dimension), device="cuda", dtype=dtype
    ) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn(
        (batch, sequence, heads, angle_dimension),
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    residual = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.01
    gate = torch.randn(
        (batch, sequence, heads, value_dimension), device="cuda", dtype=dtype
    ) * 0.1
    arguments = (
        query,
        key,
        value,
        adt,
        dt,
        trap,
        query_bias,
        key_bias,
        angles,
        residual,
        gate,
        torch.empty(
            (batch, sequence, heads, qk_dimension),
            device="cuda",
            dtype=dtype,
        ),
        torch.empty(
            (batch, sequence, heads, qk_dimension),
            device="cuda",
            dtype=dtype,
        ),
        torch.empty(
            (batch, heads, sequence), device="cuda", dtype=torch.float32
        ),
        torch.empty(
            (batch, heads, sequence), device="cuda", dtype=torch.float32
        ),
        torch.empty(
            (batch, heads, sequence), device="cuda", dtype=torch.float32
        ),
    )
    _, generated_base = compile_single(
        context,
        mamba3_siso_forward_kernel,
        arguments,
        constexprs={"HEAD_GROUP": heads // qk_heads},
    )
    generated = PreparedLaunch(
        generated_base.launch,
        generated_base.outputs,
    )
    _activate(
        context,
        "source/triton/state-spaces-mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd_runtime.py",
        "intent_v2_triton_mamba3_siso_forward_runtime",
    )
    source_function = importlib.import_module(
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd"
    ).mamba3_siso_fwd
    source_base = functional_launch(
        lambda: source_function(
            query,
            key,
            value,
            adt,
            dt,
            trap,
            query_bias,
            key_bias,
            angles,
            D=residual,
            Z=gate,
            chunk_size=64,
        )
    )
    source = PreparedLaunch(source_base.launch, lambda: source_base.outputs()[0])
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


def chunk_state(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 1, 2048, 32, 64
    groups, state_dimension, chunk = 8, 128, 256
    chunks = sequence // chunk
    x = torch.randn(
        (batch, sequence, heads, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_basis = torch.randn(
        (batch, sequence, groups, state_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    dt = torch.rand(
        (batch, heads, chunks, chunk), device="cuda", dtype=torch.float32
    ) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    _, generated = compile_single(
        context,
        mamba_chunk_state_bf16_fwd,
        (state_basis, x, dt, decay),
        constexprs={"HEAD_GROUP": heads // groups},
    )
    _activate(
        context,
        "source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_state_runtime.py",
        "intent_v2_triton_chunk_state_runtime",
    )
    source_function = importlib.import_module(
        "mamba_ssm.ops.triton.ssd_chunk_state"
    )._chunk_state_fwd
    source = functional_launch(
        lambda: source_function(
            state_basis,
            x,
            dt,
            decay,
            states_in_fp32=True,
        )
    )
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=2e-2), cuda_graph=True)


def state_passing(context: Context) -> PreparedComparison:
    batch, chunks, heads, state = 1, 8, 32, 8192
    chunk_states = torch.randn(
        (batch, chunks, heads, state), device="cuda", dtype=torch.float32
    ) * 0.01
    decay = -torch.rand(
        (batch, heads, chunks), device="cuda", dtype=torch.float32
    ) * 0.1
    initial = torch.zeros(
        (batch, heads, state), device="cuda", dtype=torch.float32
    )
    _, generated = compile_single(
        context,
        mamba_state_passing_fwd,
        (chunk_states, decay, initial),
    )
    _activate(
        context,
        "source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_state_passing_runtime.py",
        "intent_v2_triton_state_passing_runtime",
    )
    source_function = importlib.import_module(
        "mamba_ssm.ops.triton.ssd_state_passing"
    )._state_passing_fwd
    source = functional_launch(
        lambda: source_function(chunk_states, decay, initial)
    )
    return PreparedComparison(generated, source, Tolerance(atol=1e-5, rtol=1e-5), cuda_graph=True)


def chunk_scan(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 1, 2048, 32, 64
    groups, state_dimension, chunk = 8, 128, 256
    chunks = sequence // chunk
    x = torch.randn(
        (batch, sequence, heads, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_matrix = torch.randn(
        (batch, sequence, groups, state_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cb = torch.randn(
        (batch, chunks, groups, chunk, chunk),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.01
    dt = torch.rand(
        (batch, heads, chunks, chunk), device="cuda", dtype=torch.float32
    ) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn(
        (batch, chunks, heads, dimension, state_dimension),
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    residual = torch.randn(
        (heads,), device="cuda", dtype=torch.float32
    ) * 0.01
    arguments = (cb, x, dt, decay, state_matrix, previous, residual)
    _, generated = compile_single(
        context,
        mamba_chunk_scan_bf16_fwd,
        arguments,
        constexprs={"HEADS_PER_GROUP": heads // groups},
    )
    _activate(
        context,
        "source/triton/state-spaces-mamba/mamba_ssm/ops/triton/ssd_chunk_scan_runtime.py",
        "intent_v2_triton_chunk_scan_runtime",
    )
    source_function = importlib.import_module(
        "mamba_ssm.ops.triton.ssd_chunk_scan"
    )._chunk_scan_fwd
    source_base = functional_launch(
        lambda: source_function(
            cb,
            x,
            dt,
            decay,
            state_matrix,
            previous,
            D=residual,
        )
    )
    source = PreparedLaunch(
        launch=source_base.launch,
        outputs=lambda: source_base.outputs()[0],
    )
    return PreparedComparison(generated, source, Tolerance(atol=1e-1, rtol=5e-2), cuda_graph=True)


CASES = {
    "mamba3_siso_step": mamba3_step,
    "mamba3_siso_forward": mamba3_siso_forward,
    "mamba_chunk_state": chunk_state,
    "mamba_state_passing": state_passing,
    "mamba_chunk_scan": chunk_scan,
}
