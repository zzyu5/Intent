from __future__ import annotations

import math

import intent
import torch

from kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from kernels.streaming.gated_delta import chunk_gated_delta_prepare, chunk_gated_delta_recurrence
from kernels.streaming.linear_attention import fused_chunk_linear_attention_fwd
from kernels.streaming.linear_attention import fused_chunk_linear_attention_bwd, chunk_retention_fwd
from kernels.streaming.mamba import mamba_chunk_state_fwd
from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd, mamba_state_passing_fwd
from kernels.streaming.mamba import mamba3_siso_step, mamba3_siso_forward
from kernels.streaming.online_softmax import streamed_online_softmax_f16
from kernels.streaming.online_softmax import streamed_online_softmax
from kernels.streaming.selective_scan import (
    BATCH as SELECTIVE_SCAN_BATCH,
    LENGTH as SELECTIVE_SCAN_LENGTH,
    mamba_chunk_scan_fwd,
    mamba_chunk_scan_bf16_fwd,
    selective_state_scan,
)

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison, prepare_host_run_only


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


def online_softmax_f32(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_host_comparison(context, streamed_online_softmax, (x,), "softmax", Tolerance(atol=1e-5))


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


def mamba_chunk_state(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, sequence, heads, groups, dimension, state, chunk = (
        1, 2048, 32, 8, 64, 128, 256
    )
    chunks = sequence // chunk
    state_basis = torch.randn(
        (batch, sequence, groups, state), dtype=torch.float16
    )
    x = torch.randn(
        (batch, sequence, heads, dimension), dtype=torch.float16
    )
    dt = torch.randn(
        (batch, heads, chunks, chunk), dtype=torch.float16
    )
    cumulative_decay = torch.cumsum(
        -torch.rand_like(dt) * 0.1,
        dim=-1,
    )
    return prepare_host_comparison(
        context,
        mamba_chunk_state_fwd,
        (state_basis, x, dt, cumulative_decay),
        "mamba_chunk_state_fwd",
        Tolerance(atol=1.0e-1, rtol=5.0e-2),
        constexprs={"HEAD_GROUP": heads // groups},
    )


def linear_attention(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, sequence, heads, dimension = 1, 2048, 16, 128
    q = torch.randn((batch, sequence, heads, dimension), dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    return prepare_host_comparison(
        context,
        fused_chunk_linear_attention_fwd,
        (q, k, v, scale),
        "linear_attention_forward",
        (
            Tolerance(atol=1.0e-1, rtol=5.0e-2),
            Tolerance(atol=1.0e-1, rtol=5.0e-2),
        ),
    )


def mamba_chunk_state_bf16(context):
    x = torch.randn((1, 2048, 32, 64), dtype=torch.bfloat16)
    basis = torch.randn((1, 2048, 8, 128), dtype=torch.bfloat16)
    dt = torch.rand((1, 32, 8, 256), dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    return prepare_host_comparison(context, mamba_chunk_state_bf16_fwd, (basis, x, dt, decay),
        "mamba_chunk_state_bf16_fwd", Tolerance(atol=2e-2, rtol=2e-2), constexprs={"HEAD_GROUP": 4},
        note="原BF16 chunk-state生产输入，dt/decay及输出f32；双方保留scaled basis的bf16转换，沿用原容差。")


def mamba_state_passing(context):
    states = torch.randn((1, 8, 32, 8192), dtype=torch.float32) * 0.01
    decay = -torch.rand((1, 32, 8), dtype=torch.float32) * 0.1
    initial = torch.zeros((1, 32, 8192), dtype=torch.float32)
    return prepare_host_comparison(context, mamba_state_passing_fwd, (states, decay, initial),
        "mamba_state_passing_fwd", (Tolerance(atol=1e-5, rtol=1e-5), Tolerance(atol=1e-5, rtol=1e-5)))


def mamba_chunk_scan_bf16(context):
    x = torch.randn((1, 2048, 32, 64), dtype=torch.bfloat16)
    state_matrix = torch.randn((1, 2048, 8, 128), dtype=torch.bfloat16)
    cb = torch.randn((1, 8, 8, 256, 256), dtype=torch.bfloat16) * 0.01
    dt = torch.rand((1, 32, 8, 256), dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((1, 8, 32, 64, 128), dtype=torch.float32) * 0.01
    residual = torch.randn((32,), dtype=torch.float32) * 0.01
    return prepare_host_comparison(context, mamba_chunk_scan_bf16_fwd,
        (cb, x, dt, decay, state_matrix, previous, residual), "mamba_chunk_scan_fwd",
        Tolerance(atol=1e-1, rtol=5e-2), constexprs={"HEADS_PER_GROUP": 4},
        note="原BF16 chunk-scan生产输入与f32 dt/decay/previous；生成端保留previous与scan coefficient的bf16转换，CPU数学reference以f32计算，最终均bf16，沿用原容差。")


def mamba3_step(context):
    query = torch.randn((32, 4, 32), dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((32, 16, 64), dtype=torch.bfloat16) * 0.1
    adt = -torch.rand((32, 16), dtype=torch.float32) * 0.1
    dt, trap = torch.rand_like(adt) * 0.1, torch.rand_like(adt) * 0.1
    query_bias = torch.randn((16, 32), dtype=torch.bfloat16) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((32, 16, 16), dtype=torch.float32) * 0.01
    residual = torch.randn((16,), dtype=torch.float32) * 0.01
    gate = torch.randn_like(value) * 0.1
    states = tuple(torch.zeros(shape, dtype=torch.float32) for shape in
                   ((32, 16, 16), (32, 16, 64, 32), (32, 16, 32), (32, 16, 64)))
    return prepare_host_run_only(context, mamba3_siso_step,
        (query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, *states),
        constexprs={"HEAD_GROUP": 4},
        note="原Mamba3 step B32/H16/QKH4/DQK32/DV64 BF16输入、四个零初始f32状态；原source标记semantics gap，无可比CPU reference，仅运行完整kernel和其四项输出，不作数值或相对性能结论。")


def mamba3_forward(context):
    query = torch.randn((1, 2048, 4, 32), dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((1, 2048, 16, 64), dtype=torch.bfloat16) * 0.1
    adt = -torch.rand((1, 16, 2048), dtype=torch.float32) * 0.1
    dt, trap = torch.rand_like(adt) * 0.1, torch.rand_like(adt) * 0.1
    query_bias = torch.randn((16, 32), dtype=torch.bfloat16) * 0.01
    key_bias = torch.randn_like(query_bias) * 0.01
    angles = torch.randn((1, 2048, 16, 16), dtype=torch.float32) * 0.01
    residual = torch.randn((16,), dtype=torch.float32) * 0.01
    gate = torch.randn_like(value) * 0.1
    stores = (torch.empty((1, 2048, 16, 32), dtype=torch.bfloat16),
              torch.empty((1, 2048, 16, 32), dtype=torch.bfloat16),
              *(torch.empty((1, 16, 2048), dtype=torch.float32) for _ in range(3)))
    return prepare_host_run_only(context, mamba3_siso_forward,
        (query, key, value, adt, dt, trap, query_bias, key_bias, angles, residual, gate, *stores),
        constexprs={"HEAD_GROUP": 4},
        note="原Mamba3 forward B1/S2048/H16/QKH4/DQK32/DV64 BF16输入；五个InOut stores在每次kernel内先完整写入再读取，host预分配；原source标记semantics gap，无可比CPU reference，仅运行，不作数值或相对性能结论。")


def chunk_gated_delta(context):
    query = torch.randn((2, 2048, 8, 128), dtype=torch.bfloat16) * 0.1
    key, value = torch.randn_like(query) * 0.1, torch.randn_like(query) * 0.1
    gate = -torch.rand((2, 2048, 8), dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    arguments = (query, key, value, gate, beta, 128**-0.5)
    report_stage("generated_compilation")
    preparation, recurrence = (intent.compile(definition, target=context.target, compiler=context.compiler,
                                              tuning_config=context.tuning_config)
                               for definition in (chunk_gated_delta_prepare, chunk_gated_delta_recurrence))
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")
    generated, source = {}, {}

    def launch():
        generated["output"] = recurrence.run(*preparation.run(*arguments))

    def reference():
        source["output"] = runtime.chunk_gated_delta_fwd(*arguments)

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, lambda: generated["output"]),
        PreparedLaunch(reference, lambda: source["output"]),
        (Tolerance(atol=1e-1, rtol=5e-2), Tolerance(atol=1e-1, rtol=5e-2)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原B2/S2048/H8/K128/V128 BF16 chunk64 gated-delta完整prepare+recurrence，含中间buffer分配，比较output和f32 final state；CPU参考保留source guarded Neumann/serial triangular solve和chunk recurrence，BF16中间ABI，matmul使用f32而非原GPU TF32或Intent显式BF16输入转换，原容差。")


def linear_attention_backward(context):
    q = torch.randn((1, 2048, 16, 128), dtype=torch.float16)
    k, v, gradient = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    tolerance = Tolerance(atol=1e-1, rtol=5e-2)
    return prepare_host_comparison(context, fused_chunk_linear_attention_bwd, (q, k, v, gradient, 128**-0.5),
        "linear_attention_backward", (tolerance, tolerance, tolerance),
        note="原B1/S2048/H16/D128 f16输入，三个f32梯度；CPU参考使用source chunk64前缀/后缀矩阵公式，以f32计算；生成端保留作者局部score及state的f16转换，原容差。")


def retention(context):
    q = torch.randn((1, 2048, 16, 128), dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_host_run_only(context, chunk_retention_fwd, (q, k, v, 128**-0.5),
        note="原B1/S2048/H16/D128 f16 retention输入；作者decay为1-exp(-5-head)，原TileLang source为1-exp2(-5-head)，数值算法不同，保留当前Intent语义。无同合同CPU reference，仅运行，不作数值或相对性能结论。")


CASES = {
    "selective_state_scan": selective_scan,
    "streamed_online_softmax_f16": online_softmax,
    "streamed_online_softmax": online_softmax_f32,
    "recurrent_gated_delta_fwd": recurrent_gated_delta,
    "mamba_chunk_scan": mamba_chunk_scan,
    "mamba_chunk_state": mamba_chunk_state,
    "linear_attention_forward": linear_attention,
    "mamba_chunk_state_bf16": mamba_chunk_state_bf16,
    "mamba_state_passing": mamba_state_passing,
    "mamba_chunk_scan_bf16": mamba_chunk_scan_bf16,
    "mamba3_siso_step": mamba3_step,
    "mamba3_siso_forward": mamba3_forward,
    "chunk_gated_delta": chunk_gated_delta,
    "linear_attention_backward": linear_attention_backward,
    "chunk_retention": retention,
}
