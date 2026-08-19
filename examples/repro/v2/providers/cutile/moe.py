from __future__ import annotations

import torch

from kernels.ragged.grouped_gemm import aligned_expert_projection_bf16
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
    sorted_ids, expert_ids, padded_tokens, _, _ = runtime.align.moe_align_block_size(
        topk_ids, BLOCK_SIZE, experts
    )
    generated_output = torch.empty(
        (tokens * topk, intermediate), device="cuda", dtype=torch.bfloat16
    )
    _, generated_base = compile_single(
        context,
        aligned_expert_projection_bf16,
        (
            hidden_states,
            sorted_ids,
            expert_ids,
            padded_tokens,
            weights,
            generated_output,
        ),
        constexprs={"TOP_K": topk, "BLOCK_SIZE": BLOCK_SIZE},
    )
    generated = PreparedLaunch(
        generated_base.launch,
        lambda: generated_output,
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

    source_launch()
    source = PreparedLaunch(source_launch, lambda: source_output.view(tokens * topk, intermediate))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def alignment(context: Context) -> PreparedComparison:
    tokens, topk = 4096, 2
    source_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.long)
        .remainder(EXPERTS)
        .reshape(tokens, topk)
        .contiguous()
    )
    generated_ids = source_ids.to(torch.int32)
    expert_counts = torch.zeros((EXPERTS,), device="cuda", dtype=torch.int32)
    _, count = compile_single(
        context, moe_count_routes, (generated_ids, expert_counts)
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
        (generated_ids, expert_offsets, expert_cursors, sorted_routes),
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
        total = int(total_padded[0].item())
        blocks = total // BLOCK_SIZE
        return sorted_routes[:total], expert_blocks[:blocks], total_padded

    generated_prepare()
    generated_launch()
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
        total_value = int(total[0].item())
        blocks = total_value // BLOCK_SIZE
        return sorted_ids[:total_value], expert_ids[:blocks], total

    source_launch()
    source = PreparedLaunch(source_launch, source_outputs)
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=0.0), Tolerance(atol=0.0), Tolerance(atol=0.0)),
        cuda_graph=False,
    )


CASES = {
    "moe_expert_projection": expert_projection,
    "moe_alignment": alignment,
}
