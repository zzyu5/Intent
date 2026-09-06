from __future__ import annotations

import torch

from kernels.ragged.grouped_gemm import routed_expert_projection_bf16
from kernels.routing.moe_align import BLOCK_SIZE
from kernels.routing.moe_align import EXPERT_BLOCKS
from kernels.routing.moe_align import EXPERTS
from kernels.routing.moe_align import PADDED_ROUTES
from kernels.routing.moe_align import ROUTES
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes

from ...measurement import compile_single
from ...measurement import initial_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module


def _runtime(context: Context):
    return runtime_module(
        context,
        "source/cutile/tilegym/moe/fused/moe_runtime.py",
        "intent_v2_cutile_moe_runtime",
    )


def _canonical_alignment(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_value = int(total[0].item())
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


def expert_projection(context: Context) -> PreparedComparison:
    tokens, hidden, intermediate, experts, topk = 4096, 4096, 14336, 8, 2
    hidden_states = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16
    )
    topk_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.long)
        .remainder(experts)
        .reshape(tokens, topk)
        .contiguous()
    )
    topk_weights = torch.full(
        (tokens, topk), 1.0 / topk, device="cuda", dtype=torch.bfloat16
    )
    runtime = _runtime(context)
    sorted_ids, expert_ids, padded_tokens, _, _ = initial_launch(
        lambda: runtime.align.moe_align_block_size(topk_ids, BLOCK_SIZE, experts),
        side="source",
    )
    flat_experts = topk_ids.flatten()
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    expert_offsets = torch.empty(
        (experts + 1,), device="cuda", dtype=torch.int32
    )
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(
        torch.bincount(flat_experts, minlength=experts), dim=0
    ).to(torch.int32)
    _, generated = compile_single(
        context,
        routed_expert_projection_bf16,
        (
            hidden_states,
            expert_offsets,
            member_routes,
            weights,
        ),
        constexprs={"TOP_K": topk},
    )
    source_output = torch.empty(
        (tokens, topk, intermediate), device="cuda", dtype=torch.bfloat16
    )
    config = {
        "TILE_SIZE_M": 128,
        "TILE_SIZE_N": 128,
        "TILE_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    }

    def source_launch():
        runtime.source.invoke_fused_moe_kernel(
            hidden_states,
            weights,
            source_output,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            padded_tokens,
            False,
            topk,
            config,
            torch.bfloat16,
            False,
        )

    initial_launch(source_launch, side="source")
    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
        # Source receives precomputed block ownership; Intent receives CSR routing.
        status="source_precomputed_block_ownership_contract_gap",
    )


def alignment(context: Context) -> PreparedComparison:
    tokens, topk = 4096, 2
    source_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        .remainder(EXPERTS)
        .reshape(tokens, topk)
        .contiguous()
    )
    expert_counts = torch.zeros((EXPERTS,), device="cuda", dtype=torch.int32)
    _, count = compile_single(
        context, moe_count_routes, (source_ids, expert_counts)
    )
    _, prefix = compile_single(context, moe_prefix_routes, (expert_counts,))
    expert_offsets, total_padded = prefix.outputs()
    expert_cursors = torch.zeros_like(expert_counts)
    sorted_routes = torch.full(
        (PADDED_ROUTES,), ROUTES, device="cuda", dtype=torch.int32
    )
    _, scatter = compile_single(
        context,
        moe_scatter_routes,
        (source_ids, expert_offsets, expert_cursors, sorted_routes),
    )
    _, mark = compile_single(context, moe_mark_expert_blocks, (expert_offsets,))
    expert_blocks = mark.outputs()

    def generated_prepare():
        expert_counts.zero_()
        expert_cursors.zero_()
        sorted_routes.fill_(ROUTES)

    def generated_launch():
        count.launch()
        prefix.launch()
        scatter.launch()
        mark.launch()

    def generated_outputs():
        return _canonical_alignment(sorted_routes, expert_blocks, total_padded)

    generated_prepare()
    initial_launch(generated_launch, side="generated")
    generated = PreparedLaunch(
        generated_launch,
        generated_outputs,
        prepare=generated_prepare,
    )
    runtime = _runtime(context)
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch():
        source_state["outputs"] = runtime.align.moe_align_block_size(
            source_ids, BLOCK_SIZE, EXPERTS
        )

    def source_outputs():
        sorted_ids, expert_ids, total, _, _ = source_state["outputs"]
        return _canonical_alignment(sorted_ids, expert_ids, total)

    initial_launch(source_launch, side="source")
    source = PreparedLaunch(source_launch, source_outputs)
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=0.0), Tolerance(atol=0.0), Tolerance(atol=0.0)),
        cuda_graph=False,
        status="source_algorithm_and_timing_contract_gap",
    )


CASES = {
    "moe_expert_projection": expert_projection,
    "moe_alignment": alignment,
}
