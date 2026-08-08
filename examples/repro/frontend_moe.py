from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import torch
import triton

import intent
from frontend_softmax import moe_expert_ffn


TOKENS = 4096
HIDDEN = 4096
INTERMEDIATE = 14336
EXPERTS = 8
TOP_K = 2


def _load_upstream_grouped_gemm(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 213]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_upstream_triton_grouped_gemm",
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["group_gemm_fn"]


def _routes(device: torch.device):
    route_count = TOKENS * TOP_K
    route_token = torch.arange(
        TOKENS, device=device, dtype=torch.int32
    ).repeat_interleave(TOP_K)
    route_weights = torch.full(
        (route_count,), 1.0 / TOP_K, device=device, dtype=torch.float32
    )
    route_expert = torch.arange(route_count, device=device) % EXPERTS
    member_routes = torch.argsort(route_expert, stable=True).to(torch.int32)
    counts = torch.bincount(route_expert, minlength=EXPERTS)
    route_offsets = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0))
    ).to(torch.int32)
    return route_offsets, member_routes, route_token, route_weights


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
    grouped_gemm,
    x,
    route_offsets,
    member_routes,
    route_token,
    route_weights,
    w1,
    w2,
):
    routes_by_expert = [
        member_routes[route_offsets[e] : route_offsets[e + 1]].long()
        for e in range(EXPERTS)
    ]
    tokens_by_expert = [route_token[routes].long() for routes in routes_by_expert]
    grouped_input = [x[tokens] for tokens in tokens_by_expert]
    hidden = grouped_gemm(grouped_input, [w1[e] for e in range(EXPERTS)])
    hidden = [torch.relu(value) for value in hidden]
    output = grouped_gemm(hidden, [w2[e] for e in range(EXPERTS)])
    merged = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    for routes, tokens, values in zip(routes_by_expert, tokens_by_expert, output):
        merged.index_add_(
            0, tokens, route_weights[routes, None] * values.float()
        )
    return merged


def _run(intent_realize: str, intent_translate: str, baseline_source: Path) -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    artifact = intent.compile(
        moe_expert_ffn,
        target=intent.TritonTarget(device=0),
        realizer=intent_realize,
        translator=intent_translate,
    )
    grouped_gemm = _load_upstream_grouped_gemm(baseline_source)
    route_offsets, member_routes, route_token, route_weights = _routes(device)
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
    upstream = _upstream_baseline(
        grouped_gemm,
        x,
        route_offsets,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
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
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    if generated_error > 2.0e-3:
        raise RuntimeError(
            f"staged Triton MoE numerical comparison failed: {generated_error}"
        )

    generated_p50, generated_p95 = triton.testing.do_bench(
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
        rep=20,
        quantiles=[0.5, 0.95],
    )
    upstream_p50, upstream_p95 = triton.testing.do_bench(
        lambda: _upstream_baseline(
            grouped_gemm,
            x,
            route_offsets,
            member_routes,
            route_token,
            route_weights,
            w1,
            w2,
        ),
        warmup=5,
        rep=20,
        quantiles=[0.5, 0.95],
    )

    print("=== Intent Kernel IR + Triton Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated Triton source ===")
    print(artifact.source, end="")
    print(
        "staged Triton MoE numerical comparison: PASS "
        f"(T={TOKENS}, D={HIDDEN}, F={INTERMEDIATE}, E={EXPERTS}, top_k={TOP_K}, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error})"
    )
    print(
        "staged Triton MoE end-to-end performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent-realize",
        default="/tmp/intentdsl-build/tools/intent-realize/intent-realize",
    )
    parser.add_argument(
        "--intent-translate",
        default="/tmp/intentdsl-build/tools/intent-translate/intent-translate",
    )
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()
    _run(
        arguments.intent_realize,
        arguments.intent_translate,
        arguments.baseline_source,
    )


if __name__ == "__main__":
    main()
