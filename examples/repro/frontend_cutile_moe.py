from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

import intent

from frontend_cutile_gemm import _bench
from frontend_softmax import moe_expert_ffn


TOKENS = 4096
HIDDEN = 4096
INTERMEDIATE = 14336
EXPERTS = 8
TOP_K = 2
TILE_M = 128
TILE_N = 64
TILE_K = 64


def _load_upstream(source_path: Path):
    spec = importlib.util.spec_from_file_location(
        "intent_upstream_cutile_moe", source_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _routes(device: torch.device):
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


def _reference(
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


def _upstream_baseline(
    upstream,
    x,
    route_weights,
    topk_ids,
    w1_upstream,
    w2_upstream,
    sorted_routes,
    sorted_experts,
):
    hidden = torch.empty(
        (TOKENS, TOP_K, INTERMEDIATE), device=x.device, dtype=torch.float16
    )
    upstream.invoke_fused_moe_kernel(
        x,
        w1_upstream,
        hidden,
        route_weights.reshape(TOKENS, TOP_K),
        sorted_routes,
        sorted_experts,
        mul_routed_weight=False,
        num_token_replicas=TOP_K,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
    )
    hidden = torch.relu(hidden)
    routed_output = torch.empty(
        (TOKENS, TOP_K, HIDDEN), device=x.device, dtype=torch.float16
    )
    upstream.invoke_fused_moe_kernel(
        hidden.reshape(TOKENS * TOP_K, INTERMEDIATE),
        w2_upstream,
        routed_output,
        route_weights.reshape(TOKENS, TOP_K),
        sorted_routes,
        sorted_experts,
        mul_routed_weight=True,
        num_token_replicas=1,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
    )
    return routed_output.float().sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-realize", required=True)
    parser.add_argument("--intent-translate", required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    artifact = intent.compile(
        moe_expert_ffn,
        target=intent.CuTileTarget(device=0),
        realizer=arguments.intent_realize,
        translator=arguments.intent_translate,
    )
    upstream = _load_upstream(arguments.baseline_source)
    route_offsets, member_routes, route_token, route_weights, topk_ids = _routes(
        device
    )
    sorted_routes, sorted_experts = upstream.moe_align_tile_size_torch(
        topk_ids, TILE_M, EXPERTS
    )
    x = torch.randn((TOKENS, HIDDEN), device=device, dtype=torch.float16)
    x /= math.sqrt(HIDDEN)
    w1 = torch.randn(
        (EXPERTS, HIDDEN, INTERMEDIATE), device=device, dtype=torch.float16
    )
    w1 /= math.sqrt(HIDDEN)
    w2 = torch.randn(
        (EXPERTS, INTERMEDIATE, HIDDEN), device=device, dtype=torch.float16
    )
    w2 /= math.sqrt(INTERMEDIATE)
    w1_upstream = w1.transpose(1, 2).contiguous()
    w2_upstream = w2.transpose(1, 2).contiguous()

    generated = torch.zeros(
        (TOKENS, HIDDEN), device=device, dtype=torch.float32
    )
    artifact(
        x,
        route_offsets,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
        generated,
    )
    original = _upstream_baseline(
        upstream,
        x,
        route_weights,
        topk_ids,
        w1_upstream,
        w2_upstream,
        sorted_routes,
        sorted_experts,
    )
    reference = _reference(
        x,
        route_offsets,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
    )
    stream.synchronize()
    generated_error = (generated - reference).abs().max().item()
    original_error = (original - reference).abs().max().item()
    if generated_error > 2.0e-3 or original_error > 2.0e-3:
        raise RuntimeError(
            "staged cuTile MoE numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={original_error}"
        )

    generated_p50, generated_p95 = _bench(
        lambda: artifact.run(
            x,
            route_offsets,
            member_routes,
            route_token,
            route_weights,
            w1,
            w2,
        ),
        warmup=5,
        repetitions=20,
    )
    original_p50, original_p95 = _bench(
        lambda: _upstream_baseline(
            upstream,
            x,
            route_weights,
            topk_ids,
            w1_upstream,
            w2_upstream,
            sorted_routes,
            sorted_experts,
        ),
        warmup=5,
        repetitions=20,
    )

    print("=== Intent Kernel IR + cuTile Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated cuTile source ===")
    print(artifact.source, end="")
    print(
        "staged cuTile MoE numerical comparison: PASS "
        f"(T={TOKENS}, D={HIDDEN}, F={INTERMEDIATE}, E={EXPERTS}, top_k={TOP_K}, "
        f"generated/reference={generated_error}, upstream/reference={original_error})"
    )
    print(
        "staged cuTile MoE end-to-end performance: "
        f"upstream_p50={original_p50:.4f} ms, upstream_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / original_p50:.4f}x"
    )


if __name__ == "__main__":
    main()
