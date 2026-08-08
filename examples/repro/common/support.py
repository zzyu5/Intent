from __future__ import annotations

from collections.abc import Callable

import torch

from intent.runtime import CompiledArtifact
from kernels.ragged.moe import EXPERTS
from kernels.ragged.moe import HIDDEN
from kernels.ragged.moe import TOKENS
from kernels.ragged.moe import TOP_K


def benchmark(
    function: Callable[[], object],
    *,
    warmup: int = 25,
    repetitions: int = 100,
) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)]
    )
    return samples.quantile(0.5).item(), samples.quantile(0.95).item()


def print_artifact(artifact: CompiledArtifact, target: str) -> None:
    print(f"=== Intent Kernel IR + {target} Plan MLIR ===")
    print(artifact.mlir, end="")
    print(f"=== Generated {target} source ===")
    print(artifact.source, end="")
    if artifact.backend_ir:
        print("backend IR levels: " + ", ".join(sorted(artifact.backend_ir)))


def make_moe_routes(device: torch.device):
    route_count = TOKENS * TOP_K
    route_token = torch.arange(
        TOKENS, device=device, dtype=torch.int32
    ).repeat_interleave(TOP_K)
    route_weights = torch.full(
        (route_count,), 1.0 / TOP_K, device=device, dtype=torch.float32
    )
    topk_ids = (torch.arange(route_count, device=device) % EXPERTS).reshape(
        TOKENS, TOP_K
    )
    flat_experts = topk_ids.reshape(-1)
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    counts = torch.bincount(flat_experts, minlength=EXPERTS)
    route_offsets = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0))
    ).to(torch.int32)
    return route_offsets, member_routes, route_token, route_weights, topk_ids


def moe_reference(
    x,
    route_offsets,
    member_routes,
    route_token,
    route_weights,
    w1,
    w2,
):
    result = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    for expert in range(EXPERTS):
        routes = member_routes[
            route_offsets[expert] : route_offsets[expert + 1]
        ].long()
        tokens = route_token[routes].long()
        hidden = torch.relu(x[tokens].float() @ w1[expert].float())
        values = hidden @ w2[expert].float()
        result.index_add_(0, tokens, route_weights[routes, None] * values)
    return result
