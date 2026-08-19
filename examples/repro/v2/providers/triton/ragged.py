from __future__ import annotations

import torch

from kernels.ragged.grouped_gemm import routed_expert_projection_bf16
from kernels.ragged.jagged_mean import jagged_mean

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def jagged_mean_case(context: Context) -> PreparedComparison:
    batch, features, max_length = 512, 128, 128
    lengths = (
        torch.arange(batch, device="cuda", dtype=torch.int32) % max_length
    ) + 1
    offsets = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn(
        int(offsets[-1].item()),
        features,
        device="cuda",
        dtype=torch.float32,
    )
    _, generated = compile_single(context, jagged_mean, (values, offsets))
    runtime = load_module(
        context.project_root
        / "source/triton/tritonbench/ragged/jagged_mean/jagged_mean_runtime.py",
        "intent_v2_triton_jagged_mean",
    )
    source = functional_launch(lambda: runtime.upstream((values, offsets)))
    return PreparedComparison(generated, source, Tolerance(atol=2e-5, rtol=1e-5), cuda_graph=True)


def moe_expert_projection(context: Context) -> PreparedComparison:
    tokens, hidden, output, experts, topk = 2048, 4096, 14336, 8, 2
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        (experts, output, hidden), device="cuda", dtype=torch.bfloat16
    )
    ids = torch.stack(
        (
            torch.arange(tokens, device="cuda") % experts,
            (torch.arange(tokens, device="cuda") + 1) % experts,
        ),
        dim=1,
    ).to(torch.int32)
    flat_ids = ids.reshape(-1)
    member_routes = torch.argsort(flat_ids, stable=True).to(torch.int32)
    counts = torch.bincount(flat_ids, minlength=experts)
    expert_offsets = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int64),
            counts.cumsum(0),
        )
    ).to(torch.int32)
    _, generated = compile_single(
        context,
        routed_expert_projection_bf16,
        (x, expert_offsets, member_routes, weight),
        constexprs={"TOP_K": topk},
    )

    projection_runtime = load_module(
        context.project_root
        / "source/triton/meta-applied-ai/moe/support/projection_runtime.py",
        "intent_v2_triton_moe_projection_runtime",
    )
    runtime = projection_runtime._runtime_module()
    source_module = runtime.load_source(
        context.project_root
        / "source/triton/meta-applied-ai/moe/grouped/v0_moe_fused.py",
        "intent_v2_triton_moe_grouped_source",
        stub_vllm=True,
    )
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    sorted_ids, expert_ids, padded = projection_runtime._balanced_metadata(
        ids,
        config["BLOCK_SIZE_M"],
        experts,
    )
    routed_weight = torch.full(
        (tokens, topk), 0.5, device="cuda", dtype=torch.float32
    )
    source_output = torch.zeros(
        (tokens, topk, output), device="cuda", dtype=torch.bfloat16
    )

    def launch():
        source_module.invoke_fused_moe_kernel(
            x,
            weight,
            source_output,
            routed_weight,
            ids,
            sorted_ids,
            expert_ids,
            padded,
            False,
            topk,
            config,
        )

    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


CASES = {
    "jagged_mean": jagged_mean_case,
    "moe_expert_projection": moe_expert_projection,
}

