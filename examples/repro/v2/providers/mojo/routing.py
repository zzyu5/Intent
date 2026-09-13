from __future__ import annotations

import intent
import torch

from kernels.routing.mhc import mhc_apply_residual as mhc_apply_residual_definition
from kernels.routing.moe_align import BLOCK_SIZE
from kernels.routing.moe_align import EXPERTS
from kernels.routing.moe_align import PADDED_ROUTES
from kernels.routing.moe_align import ROUTES
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes
from kernels.routing.mqa_logits import fp8_mqa_logits

from ...loading import load_module
from ...measurement import report_stage
from ...model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def _canonical_alignment(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_value = int(total.reshape(-1)[0].item())
    blocks = total_value // BLOCK_SIZE
    canonical_ids = sorted_ids[:total_value].clone()
    active_experts = expert_ids[:blocks]
    for expert in range(EXPERTS):
        expert_blocks = torch.nonzero(active_experts == expert).flatten()
        if expert_blocks.numel() == 0:
            continue
        begin = int(expert_blocks[0].item()) * BLOCK_SIZE
        end = (int(expert_blocks[-1].item()) + 1) * BLOCK_SIZE
        canonical_ids[begin:end] = torch.sort(canonical_ids[begin:end]).values
    return canonical_ids, active_experts, total


def moe_alignment(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    tokens, topk = 4096, 2
    source_ids = (
        torch.arange(tokens * topk, dtype=torch.int32)
        .remainder(EXPERTS)
        .reshape(tokens, topk)
        .contiguous()
    )
    expert_counts = torch.zeros((EXPERTS,), dtype=torch.int32)
    expert_cursors = torch.zeros_like(expert_counts)
    sorted_routes = torch.full((PADDED_ROUTES,), ROUTES, dtype=torch.int32)

    report_stage("generated_compilation")
    count = intent.compile(
        moe_count_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    prefix = intent.compile(
        moe_prefix_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    scatter = intent.compile(
        moe_scatter_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    mark = intent.compile(
        moe_mark_expert_blocks,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def generated_prepare() -> None:
        expert_counts.zero_()
        expert_cursors.zero_()
        sorted_routes.fill_(ROUTES)

    def generated_launch() -> None:
        expert_counts_result = count.run(source_ids, expert_counts)
        expert_offsets, total_padded = prefix.run(expert_counts_result)
        _, sorted_routes_result = scatter.run(
            source_ids,
            expert_offsets,
            expert_cursors,
            sorted_routes,
        )
        expert_blocks = mark.run(expert_offsets)
        generated_state["output"] = (sorted_routes_result, expert_blocks, total_padded)

    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_moe_alignment",
    )
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch() -> None:
        source_state["output"] = runtime.moe_align_block_size(
            source_ids,
            BLOCK_SIZE,
            EXPERTS,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: _canonical_alignment(*generated_state["output"]),
            prepare=generated_prepare,
        ),
        PreparedLaunch(source_launch, lambda: _canonical_alignment(*source_state["output"])),
        tuple(Tolerance(atol=0.0) for _ in range(3)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有MoE alignment；T4096、top-k2、E64、block128；完整计数/前缀/散射/标记调用，辅助InOut每次恢复，输出为canonical三元组。",
    )


def mhc_post(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    tokens, hidden, streams = 4096, 2560, 4
    layer_output = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    residual = torch.randn((tokens, streams, hidden), dtype=torch.bfloat16)
    post_mix = torch.randn((tokens, streams), dtype=torch.float32)
    residual_mix = torch.randn((tokens, streams, streams), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        mhc_apply_residual_definition,
        (residual, layer_output, post_mix, residual_mix),
        "mhc_apply_residual",
        Tolerance(atol=5e-2, rtol=2e-2),
    )


def flaggems_fp8_mqa_logits(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    queries, keys, heads, dimension = 256, 4096, 32, 128
    q = (
        torch.randn((queries, heads, dimension), dtype=torch.float16) * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn((keys, dimension), dtype=torch.float16) * 0.25
    ).to(torch.float8_e4m3fn)
    kv_scale = 0.5 + torch.rand((keys,), dtype=torch.float32)
    head_weight = torch.rand((queries, heads), dtype=torch.float32) * 0.1
    indices = torch.arange(queries, dtype=torch.int32)
    key_start = indices % 17
    key_end = keys - indices % 19
    return prepare_host_comparison(
        context,
        fp8_mqa_logits,
        (q, kv, kv_scale, head_weight, key_start, key_end),
        "fp8_mqa_logits",
        Tolerance(atol=2.0e-2, rtol=2.0e-2),
    )


CASES = {
    "moe_alignment": moe_alignment,
    "mhc_post": mhc_post,
    "flaggems_fp8_mqa_logits": flaggems_fp8_mqa_logits,
}
