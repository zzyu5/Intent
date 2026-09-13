import math

import intent
import torch

from kernels.ragged.moe import EXPERTS as MOE_EXPERTS
from kernels.ragged.moe import HIDDEN as MOE_HIDDEN
from kernels.ragged.moe import INTERMEDIATE as MOE_INTERMEDIATE
from kernels.ragged.moe import TOKENS as MOE_TOKENS
from kernels.ragged.moe import TOP_K as MOE_TOP_K
from kernels.ragged.moe import moe_expert_ffn
from kernels.ragged.grouped_gemm import GROUPS
from kernels.ragged.grouped_gemm import K as GROUPED_K
from kernels.ragged.grouped_gemm import N as GROUPED_N
from kernels.ragged.grouped_gemm import ROWS as GROUPED_ROWS
from kernels.ragged.grouped_gemm import ragged_grouped_gemm
from kernels.ragged.jagged_mean import jagged_mean
from kernels.ragged.nested_pool import DOCUMENTS
from kernels.ragged.nested_pool import FEATURES as NESTED_FEATURES
from kernels.ragged.nested_pool import SENTENCES
from kernels.ragged.nested_pool import TOKENS as NESTED_TOKENS
from kernels.ragged.nested_pool import nested_jagged_mean_pool
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def jagged_mean_case(context):
    lengths = torch.arange(512, dtype=torch.int32) % 128 + 1
    offsets = torch.empty(513, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn((33024, 128), dtype=torch.float32)
    return prepare_host_comparison(context, jagged_mean, (values, offsets), "jagged_mean",
                                   Tolerance(atol=2e-5, rtol=1e-5))


def grouped_gemm(context):
    x = torch.randn((GROUPED_ROWS, GROUPED_K), dtype=torch.float16)
    x /= math.sqrt(GROUPED_K)
    offsets = torch.tensor(
        [
            0,
            GROUPED_ROWS // 32,
            GROUPED_ROWS // 16,
            GROUPED_ROWS // 8,
            GROUPED_ROWS // 4,
            GROUPED_ROWS // 2,
            3 * GROUPED_ROWS // 4,
            7 * GROUPED_ROWS // 8,
            GROUPED_ROWS,
        ],
        dtype=torch.int32,
    )
    weight = torch.randn(
        (GROUPS, GROUPED_K, GROUPED_N), dtype=torch.float16
    )
    return prepare_host_comparison(
        context,
        ragged_grouped_gemm,
        (x, offsets, weight),
        "ragged_grouped_gemm",
        Tolerance(atol=5e-2),
    )


def nested_pool(context):
    document_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)],
        dtype=torch.int32,
    )
    sentence_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(SENTENCES)],
        dtype=torch.int32,
    )
    document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
    sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        nested_jagged_mean_pool,
        (document_offsets, sentence_offsets, values),
        "nested_jagged_mean_pool",
        (Tolerance(atol=2e-5), Tolerance(atol=2e-5)),
    )


def _make_moe_routes():
    route_count = MOE_TOKENS * MOE_TOP_K
    route_token = torch.arange(
        MOE_TOKENS, dtype=torch.int32
    ).repeat_interleave(MOE_TOP_K)
    route_weights = torch.full(
        (route_count,), 1.0 / MOE_TOP_K, dtype=torch.float32
    )
    topk_ids = (torch.arange(route_count) % MOE_EXPERTS).reshape(
        MOE_TOKENS, MOE_TOP_K
    )
    flat_experts = topk_ids.reshape(-1)
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    counts = torch.bincount(flat_experts, minlength=MOE_EXPERTS)
    route_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), counts.cumsum(0))
    ).to(torch.int32)
    return route_offsets, member_routes, route_token, route_weights


def moe_expert_ffn_case(context):
    configure_cpu_budget()
    route_offsets, member_routes, route_token, route_weights = _make_moe_routes()
    x = torch.randn((MOE_TOKENS, MOE_HIDDEN), dtype=torch.float16)
    x /= math.sqrt(MOE_HIDDEN)
    w1 = torch.randn(
        (MOE_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE), dtype=torch.float16
    )
    w1 /= math.sqrt(MOE_HIDDEN)
    w2 = torch.randn(
        (MOE_EXPERTS, MOE_INTERMEDIATE, MOE_HIDDEN), dtype=torch.float16
    )
    w2 /= math.sqrt(MOE_INTERMEDIATE)

    report_stage("generated_compilation")
    artifact = intent.compile(
        moe_expert_ffn,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_moe_expert_ffn",
    )
    generated_y = torch.zeros(
        (MOE_TOKENS, MOE_HIDDEN), dtype=torch.float32
    )
    source_y = torch.zeros_like(generated_y)

    arguments = (
        x,
        route_offsets,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
    )

    def generated_launch():
        artifact.run(*arguments, generated_y)

    def source_launch():
        runtime.moe_expert_ffn(*arguments, source_y)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: generated_y,
            prepare=lambda: generated_y.zero_(),
        ),
        PreparedLaunch(
            source_launch,
            lambda: source_y,
            prepare=lambda: source_y.zero_(),
        ),
        Tolerance(atol=2.0e-3),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 MoE expert FFN 完整调用；x/w1/w2 为 f16，y 为独立 f32 InOut；每次 launch 前分别恢复 y=0，计时含完整 host 调用与同步。",
    )


CASES = {
    "jagged_mean": jagged_mean_case,
    "ragged_grouped_gemm": grouped_gemm,
    "nested_jagged_mean_pool": nested_pool,
    "moe_expert_ffn": moe_expert_ffn_case,
}
